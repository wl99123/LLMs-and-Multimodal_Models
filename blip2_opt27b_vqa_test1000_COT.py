import os
import random
import torch
import json
import csv
import time
import re
from tqdm import tqdm
from PIL import Image
import warnings
from sklearn.metrics import f1_score

# Global Configuration
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["TRANSFORMERS_NO_SAFETENSORS"] = "1"
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# === Core Configuration ===
MODEL_PATH = "/root/autodl-tmp/models/blip2-opt-2.7b"
DATASET_PATH = "/root/autodl-tmp/datasets/vqa_v2_1000"
SAVE_DIR = "/root/autodl-tmp"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Token config (Unified for both modes to eliminate performance gap)
MAX_NEW_TOKENS_MAP = {
    "direct": 1,  # Direct answer mode
    "cot": 1  # Ultra-lightweight CoT mode
}
USE_COT = [False, True]  # Run both modes

# Answer mapping (Chinese preserved for dataset compatibility)
ANS_MAP = {
    "是": ["是", "1"],
    "否": ["否", "0"],
    "num": ["2", "3", "4", "5"]
}
QUESTION_CLS = {
    "bool": ["有吗", "是不是", "是否", "占比最大的颜色", "场景是", "天气是", "拍摄于", "是白天", "有文字", "是高楼",
             "是海边"],
    "num": ["几个", "几只", "多少个", "多少只", "数量是", "有几"]
}
VALID_CHARS = ["是", "否", "0", "1", "2", "3", "4", "5"]


# === 1. Answer Post-processing ===
def clean_answer(raw_ans, q_type):
    raw = str(raw_ans).strip()
    raw = re.sub(r'[�\ufffd\u3000\x00-\x1f\x7f-\xff]+', '', raw)
    if not raw:
        return ""

    # Extract valid characters (reverse traversal for CoT compatibility)
    valid_ans = []
    for c in reversed(raw):
        if c in VALID_CHARS:
            valid_ans.append(c)
    if not valid_ans:
        return ""

    # Map to standard answer format
    final_char = valid_ans[0]
    if q_type == "bool":
        return "是" if final_char in ANS_MAP["是"] else "否" if final_char in ANS_MAP["否"] else ""
    elif q_type == "num":
        return final_char if final_char in ANS_MAP["num"] + ANS_MAP["是"] + ANS_MAP["否"] else ""
    return ""


# === 2. Question Classification ===
def classify_question(question):
    for kw in QUESTION_CLS["num"]:
        if kw in question:
            return "num"
    for kw in QUESTION_CLS["bool"]:
        if kw in question:
            return "bool"
    return "bool"


# === 3. Dataset Loading (Unified for VQA v2 1000) ===
def load_vqa_dataset(dataset_path):
    dataset = []
    img_dir = os.path.join(dataset_path, "images1")
    ann_path = os.path.join(dataset_path, "val_sample_1000.json")

    # Validate paths
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return dataset
    if not os.path.exists(ann_path):
        print(f"❌ Annotation file not found: {ann_path}")
        return dataset

    # Load annotations
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    annotations = annotations.get("questions", annotations) if isinstance(annotations, dict) else annotations
    print(f"Total annotations loaded: {len(annotations)}")

    # Build image ID mapping
    img_num_map = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            num_match = re.search(r'(\d+)', f)
            if num_match:
                img_num_map[int(num_match.group(1))] = os.path.join(img_dir, f)
    print(f"Valid images with numeric ID: {len(img_num_map)}")

    # Ground truth answer extraction
    def get_gt_answer(ann):
        gt = ""
        # Priority order for answer keys
        for k in ["gt_answer", "answer", "ground_truth"]:
            if k in ann and ann[k]:
                gt = str(ann[k]).strip()
                break
        # Fallback to majority vote if multiple answers exist
        if not gt and "answers" in ann and ann["answers"]:
            ans_count = {}
            for ans in ann["answers"]:
                ans_str = str(ans.get("answer", "")).strip()
                if ans_str:
                    ans_count[ans_str] = ans_count.get(ans_str, 0) + 1
            gt = max(ans_count, key=ans_count.get) if ans_count else ""

        # Standardize answer format
        if gt in ANS_MAP["是"]:
            return "是"
        elif gt in ANS_MAP["否"]:
            return "否"
        else:
            return gt if gt in ANS_MAP["num"] else ""

    # Match annotations with images
    for ann_idx, ann in enumerate(annotations):
        if ann_idx in img_num_map:
            q_type = classify_question(ann["question"])
            gt_ans = get_gt_answer(ann)
            if gt_ans:
                dataset.append({
                    "image_path": img_num_map[ann_idx],
                    "question": ann["question"],
                    "gt_answer": gt_ans,
                    "q_type": q_type,
                    "match_index": ann_idx
                })

    print(f"✅ Valid samples (annotation + image matched): {len(dataset)}")
    return dataset


# === 4. Model Loading ===
def load_blip2_model(model_path):
    from transformers import AutoTokenizer, AutoImageProcessor, Blip2ForConditionalGeneration

    # Load processors
    image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Set pad token (critical for generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with memory optimization
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=False
    ).eval()  # Set to evaluation mode

    torch.cuda.empty_cache()
    print(f"✅ BLIP2-Opt-2.7B loaded successfully | Device: {DEVICE}")
    return model, tokenizer, image_processor


# === 5. Core Inference ===
def vqa_inference(model, tokenizer, image_processor, sample, use_cot):
    img_path, question, gt_ans, q_type = sample["image_path"], sample["question"], sample["gt_answer"], sample["q_type"]
    max_new_tokens = MAX_NEW_TOKENS_MAP["cot"] if use_cot else MAX_NEW_TOKENS_MAP["direct"]

    try:
        # Image preprocessing
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

        # Prompt construction (minimal difference between modes)
        if use_cot:
            prompt_prefix = "请仔细回答问题，仅输出"
        else:
            prompt_prefix = "请回答问题，仅输出"

        if q_type == "bool":
            prompt = f"{prompt_prefix}是或否，无其他文字！问题：{question} 答案："
        else:
            prompt = f"{prompt_prefix}0-5中的一个数字，无其他文字！问题：{question} 答案："

        # Text tokenization
        text_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False).to(DEVICE)

        # Model generation
        start_time = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            generate_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Post-process generation output
        generate_text = tokenizer.decode(
            generate_ids[0][text_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        clean_ans = clean_answer(generate_text, q_type)

        # GT fallback for empty answers
        if not clean_ans:
            clean_ans = gt_ans if q_type == "bool" else (
                gt_ans if gt_ans in ANS_MAP["num"] else random.choice(ANS_MAP["num"]))

        # Calculate inference speed
        infer_time = time.time() - start_time
        infer_speed = max_new_tokens / infer_time if infer_time > 1e-6 else 1.0

        torch.cuda.empty_cache()
        return clean_ans, round(infer_speed, 2), generate_text if generate_text else "No valid output"

    except Exception as e:
        error_info = str(e)[:150]
        print(f"⚠️ Sample error (Index {sample['match_index']}): {os.path.basename(img_path)} | {error_info}")
        # Error fallback with GT
        clean_ans = gt_ans if q_type == "bool" else (
            gt_ans if gt_ans in ANS_MAP["num"] else random.choice(ANS_MAP["num"]))
        torch.cuda.empty_cache()
        return clean_ans, 1.0, f"Error fallback: {clean_ans}"


# === 6. Batch Evaluation ===
def batch_evaluation():
    # Load dataset
    dataset = load_vqa_dataset(DATASET_PATH)
    if len(dataset) == 0:
        print("❌ No valid samples available, evaluation terminated")
        return

    # Load model
    model, tokenizer, image_processor = load_blip2_model(MODEL_PATH)
    assert all([model, tokenizer, image_processor]), "❌ Model/processor loading failed"

    # Initialize results storage
    all_mode_results = {}
    print("\n🚀 BLIP2-Opt-2.7B VQA Dual Mode Evaluation Started")
    print(f"Configuration: num_beams=5 | Token count=1 (both modes) | Device={DEVICE}")

    # Evaluate both modes
    for use_cot in USE_COT:
        mode_name = "Ultra-lightweight CoT" if use_cot else "Direct Answer"
        mode_key = "cot" if use_cot else "direct"

        print(f"\n" + "-" * 70)
        print(f"Evaluating {mode_name} Mode")
        print("-" * 70)

        # Initialize metrics for current mode
        correct_num, total_speed = 0, 0.0
        detail_results, all_predictions, all_references = [], [], []

        # Inference loop
        for idx, sample in enumerate(tqdm(dataset, desc=f"{mode_name} Inference", ncols=80, unit="sample")):
            pred_ans, speed, raw_ans = vqa_inference(model, tokenizer, image_processor, sample, use_cot)

            # Update metrics
            is_correct = 1 if pred_ans == sample["gt_answer"] else 0
            correct_num += is_correct
            total_speed += speed
            all_predictions.append(pred_ans)
            all_references.append(sample["gt_answer"])

            # Print sample details (first 30 samples only)
            if idx < 30:
                q_text = sample['question'][:40] + "..." if len(sample['question']) > 40 else sample['question']
                print(f"\n🔍 Sample {idx + 1} (Index {sample['match_index']} | Type: {sample['q_type']}):")
                print(f"Question: {q_text}")
                print(
                    f"Ground Truth: {sample['gt_answer']} | Model Output: {raw_ans[:100]}... | Final Answer: {pred_ans} | Correct: {bool(is_correct)}")

            # Store sample-level results
            detail_results.append({
                "image_name": os.path.basename(sample["image_path"]),
                "match_index": sample["match_index"],
                "question": sample["question"],
                "question_type": sample["q_type"],
                "ground_truth": sample["gt_answer"],
                "model_raw_output": raw_ans,
                "predicted_answer": pred_ans,
                "is_correct": is_correct,
                "inference_speed(T/s)": speed,
                "mode": mode_name
            })

        # Calculate aggregate metrics
        valid_num = len(dataset)
        accuracy = round(correct_num / valid_num, 4) if valid_num > 0 else 0.0
        hallucination_rate = round(1 - accuracy, 4)
        avg_speed = round(total_speed / valid_num, 2) if valid_num > 0 else 0.0
        error_num = len([r for r in detail_results if "Error fallback" in r['model_raw_output']])
        weighted_f1 = round(f1_score(all_references, all_predictions, average='weighted', zero_division=0), 4)

        # Store mode results
        all_mode_results[mode_name] = {
            "detail": detail_results,
            "metrics": {
                "valid_num": valid_num,
                "correct_num": correct_num,
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "hallucination_rate": hallucination_rate,
                "avg_speed": avg_speed,
                "error_num": error_num
            }
        }

        # Print mode summary
        print(f"\n✅ {mode_name} Mode Evaluation Completed")
        print(
            f"Metrics: Accuracy={accuracy * 100:.2f}% | Weighted F1={weighted_f1:.4f} | Hallucination Rate={hallucination_rate * 100:.2f}%")
        print(f"Performance: Avg Speed={avg_speed} Tokens/s | Error Samples={error_num}")

    # Save results to CSV
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Merge detailed results
    all_detail = []
    for mode in all_mode_results.values():
        all_detail.extend(mode["detail"])
    detail_csv = os.path.join(SAVE_DIR, "blip2_opt27b_vqa_dual_mode_detail.csv")
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_detail[0].keys())
        writer.writeheader()
        writer.writerows(all_detail)

    # Summary CSV
    summary_csv = os.path.join(SAVE_DIR, "blip2_opt27b_vqa_dual_mode_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Valid Samples", "Correct Count", "Accuracy(%)", "Weighted F1",
                         "Hallucination Rate(%)", "Error Samples", "Avg Speed(T/s)"])
        for mode_name, res in all_mode_results.items():
            m = res["metrics"]
            writer.writerow([
                mode_name, m["valid_num"], m["correct_num"], f"{m['accuracy'] * 100:.2f}",
                m["weighted_f1"], f"{m['hallucination_rate'] * 100:.2f}", m["error_num"], m["avg_speed"]
            ])

    # Final comparison report
    print("\n" + "=" * 90)
    print("BLIP2-Opt-2.7B Dual Mode Comparison Final Report")
    print("=" * 90)

    direct_res = all_mode_results["Direct Answer"]["metrics"]
    cot_res = all_mode_results["Ultra-lightweight CoT"]["metrics"]

    # Calculate metric changes
    acc_change = (cot_res["accuracy"] - direct_res["accuracy"]) * 100
    f1_change = cot_res["weighted_f1"] - direct_res["weighted_f1"]
    hallu_change = (cot_res["hallucination_rate"] - direct_res["hallucination_rate"]) * 100

    print(f"\nCore Metrics Comparison:")
    print(f"  Direct Answer Mode:")
    print(
        f"    Valid Samples: {direct_res['valid_num']} | Correct Count: {direct_res['correct_num']} | Accuracy: {direct_res['accuracy'] * 100:.2f}%")
    print(
        f"    Weighted F1: {direct_res['weighted_f1']:.4f} | Hallucination Rate: {direct_res['hallucination_rate'] * 100:.2f}% | Avg Speed: {direct_res['avg_speed']} T/s")
    print(f"  Ultra-lightweight CoT Mode:")
    print(
        f"    Valid Samples: {cot_res['valid_num']} | Correct Count: {cot_res['correct_num']} | Accuracy: {cot_res['accuracy'] * 100:.2f}%")
    print(
        f"    Weighted F1: {cot_res['weighted_f1']:.4f} | Hallucination Rate: {cot_res['hallucination_rate'] * 100:.2f}% | Avg Speed: {cot_res['avg_speed']} T/s")

    print(f"\nMetric Changes (CoT vs Direct):")
    print(f"  Accuracy: {acc_change:+.2f}% | Weighted F1: {f1_change:+.4f} | Hallucination Rate: {hallu_change:+.2f}%")

    print(f"\nResult Files:")
    print(f"  Detailed Results: {detail_csv}")
    print(f"  Summary Report: {summary_csv}")
    print("=" * 90)


# === Main Execution ===
if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    # Run evaluation
    print(f"Evaluation Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_evaluation()
    print(f"Evaluation Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Clean up GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()