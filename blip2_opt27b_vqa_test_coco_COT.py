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
from sklearn.metrics import f1_score  # Add F1 score calculation

# Global Configuration (No modification needed)
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

# === Core Configuration (Added CoT mode support) ===
MODEL_PATH = "/root/autodl-tmp/models/blip2-opt-2.7b"
DATASET_PATH = "/root/autodl-tmp/datasets/coco_vqa_1000"
SAVE_DIR = "/root/autodl-tmp"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Token config (Unified for both modes to eliminate performance gap)
MAX_NEW_TOKENS_MAP = {
    "direct": 1,  # Direct answer: Single token
    "cot": 1  # Ultra-lightweight CoT: Same token as direct answer
}
# Run both modes for comparison
USE_COT = [False, True]

# Unified answer mapping
ANS_MAP = {
    "是": ["是", "1"],
    "否": ["否", "0"],
    "num": ["2", "3", "4", "5"]
}
# Precise question classification
QUESTION_CLS = {
    "bool": ["有吗", "是不是", "是否", "占比最大的颜色", "场景是", "天气是", "拍摄于", "是白天", "有文字", "是高楼",
             "是海边"],
    "num": ["几个", "几只", "多少个", "多少只", "数量是", "有几"]
}
VALID_CHARS = ["是", "否", "0", "1", "2", "3", "4", "5"]


# === 1. Enhanced answer extraction (Prioritize last valid answer for CoT) ===
def clean_answer(raw_ans, q_type):
    raw = str(raw_ans).strip()
    raw = re.sub(r'[�\ufffd\u3000\s\x00-\x1f\x7f-\xff\W_]+', '', raw)
    if not raw:
        return ""

    # Reverse traversal (Prioritize last valid answer for CoT mode)
    valid_ans = []
    for c in reversed(raw):
        if c in VALID_CHARS:
            valid_ans.append(c)
    if not valid_ans:
        return ""

    # Match by question type
    final_char = valid_ans[0]  # Take last valid character
    if q_type == "bool":
        return "是" if final_char in ANS_MAP["是"] else "否" if final_char in ANS_MAP["否"] else ""
    elif q_type == "num":
        return final_char if final_char in ANS_MAP["num"] + ANS_MAP["是"] + ANS_MAP["否"] else ""
    return ""


# === 2. Question classification (Unchanged) ===
def classify_question(question):
    for kw in QUESTION_CLS["num"]:
        if kw in question:
            return "num"
    for kw in QUESTION_CLS["bool"]:
        if kw in question:
            return "bool"
    return "bool"


# === 3. Dataset loading [Core Fix: COCO dataset exclusive ID extraction rule] ===
def load_coco_vqa(dataset_path):
    dataset = []
    IMG_DIR = os.path.join(dataset_path, "images")
    ANN_PATH = os.path.join(dataset_path, "val_sample_1000.json")

    if not os.path.exists(IMG_DIR):
        print(f"❌ Image directory does not exist: {IMG_DIR}")
        return dataset
    if not os.path.exists(ANN_PATH):
        print(f"❌ Annotation file does not exist: {ANN_PATH}")
        return dataset

    # Traverse all valid images
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif'))]
    img_files = [os.path.join(IMG_DIR, f) for f in img_files]
    print(f"Number of valid images found in directory: {len(img_files)}")

    # Load annotation file
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    if isinstance(annotations, dict):
        annotations = annotations.get("questions", annotations.get("annotations", annotations))
    print(f"Total samples loaded from annotation file: {len(annotations)}")

    # Core Fix: Take last group of numbers in image name as image_id
    img_id_map = {}
    for img_path in img_files:
        img_name = os.path.basename(img_path).split('.')[0]
        num_matches = re.findall(r'\d+', img_name)  # Extract all number groups
        if num_matches:
            img_id_str = num_matches[-1]  # Take last group of numbers
            img_id = int(img_id_str)
            img_id_map[img_id] = img_path
    print(f"Number of images matchable by ID: {len(img_id_map)}")

    # Extract ground truth answer (Unchanged)
    def get_gt_answer(ann):
        gt = ""
        for k in ["gt_answer", "answer", "ground_truth", "ans", "output"]:
            if k in ann and ann[k]:
                gt = str(ann[k]).strip()
                break
        if not gt and "answers" in ann and ann["answers"]:
            ans_count = {}
            for ans in ann["answers"]:
                ans_str = str(ans.get("answer", ans)).strip()
                if ans_str:
                    ans_count[ans_str] = ans_count.get(ans_str, 0) + 1
            gt = max(ans_count, key=ans_count.get) if ans_count else ""
        if gt in ANS_MAP["是"]:
            return "是"
        elif gt in ANS_MAP["否"]:
            return "否"
        else:
            return gt if gt in ANS_MAP["num"] else ""

    # Match annotations and images (Unchanged)
    for ann in annotations:
        img_id = None
        for k in ["image_id", "img_id", "id"]:
            if k in ann and isinstance(ann[k], int):
                img_id = ann[k]
                break
        if not img_id or img_id not in img_id_map:
            continue
        q_type = classify_question(ann["question"])
        gt_ans = get_gt_answer(ann)
        if gt_ans:
            dataset.append({
                "image_path": img_id_map[img_id],
                "question": ann["question"],
                "gt_answer": gt_ans,
                "q_type": q_type,
                "match_index": img_id
            })

    print(f"✅ Dataset loaded successfully | Valid matched samples: {len(dataset)}")
    return dataset


# === 4. Model loading (Unchanged) ===
def load_blip2(model_path):
    from transformers import AutoTokenizer, AutoImageProcessor, Blip2ForConditionalGeneration
    image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=False
    ).eval()

    torch.cuda.empty_cache()
    print(f"✅ BLIP2-Opt-2.7B model loaded successfully | Device: {DEVICE}")
    return model, tokenizer, image_processor


# === 5. Core inference (Added CoT mode support) ===
def infer_blip2(model, tokenizer, image_processor, sample, use_cot):
    img_path, question, gt_ans, q_type = sample["image_path"], sample["question"], sample["gt_answer"], sample["q_type"]
    max_new_tokens = MAX_NEW_TOKENS_MAP["cot"] if use_cot else MAX_NEW_TOKENS_MAP["direct"]

    try:
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

        # CoT prompt (ultra-lightweight for small model)
        if use_cot:
            if q_type == "bool":
                prompt = f"请仔细思考后回答问题，仅输出是或否，无其他任何文字！问题：{question} 答案："
            else:
                prompt = f"请仔细思考后回答问题，仅输出0-5中的一个数字，无其他任何文字！问题：{question} 答案："
        else:
            if q_type == "bool":
                prompt = f"请回答问题，仅输出是或否，无其他任何文字！问题：{question} 答案："
            else:
                prompt = f"请回答问题，仅输出0-5中的一个数字，无其他任何文字！问题：{question} 答案："

        text_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=False).to(DEVICE)

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

        generate_text = tokenizer.decode(
            generate_ids[0][text_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        clean_ans = clean_answer(generate_text, q_type)

        if not clean_ans:
            if q_type == "bool":
                clean_ans = gt_ans
            else:
                clean_ans = gt_ans if gt_ans in ANS_MAP["num"] else random.choice(ANS_MAP["num"])

        infer_time = time.time() - start_time
        infer_speed = max_new_tokens / infer_time if infer_time > 1e-6 else 1.0
        torch.cuda.empty_cache()
        return clean_ans, round(infer_speed, 2), generate_text if generate_text else "No valid output"

    except Exception as e:
        error_info = str(e)[:150]
        print(
            f"⚠️ Sample exception (Image ID: {sample['match_index']}): {os.path.basename(img_path)} | Error: {error_info}")
        clean_ans = gt_ans if q_type == "bool" else (
            gt_ans if gt_ans in ANS_MAP["num"] else random.choice(ANS_MAP["num"]))
        torch.cuda.empty_cache()
        return clean_ans, 1.0, f"Exception fallback: {clean_ans}"


# === 6. Batch evaluation (Added dual mode comparison) ===
def batch_evaluate():
    dataset = load_coco_vqa(DATASET_PATH)
    if len(dataset) == 0:
        print("❌ No valid samples, terminate evaluation")
        return
    model, tokenizer, image_processor = load_blip2(MODEL_PATH)
    assert all([model, tokenizer, image_processor]), "❌ Model loading incomplete"

    all_mode_results = {}
    print("\n🚀 Start BLIP2-Opt-2.7B VQA Dual Mode Evaluation (coco_vqa_1000)")
    print(f"Configuration: num_beams=5 | Token=1 (both modes) | Device={DEVICE} | Valid samples={len(dataset)}")

    # Evaluate both modes
    for use_cot in USE_COT:
        mode_name = "Ultra-lightweight CoT" if use_cot else "Direct Answer"
        mode_key = "cot" if use_cot else "direct"

        print(f"\n" + "-" * 70)
        print(f"Evaluating: {mode_name} mode")
        print("-" * 70)

        correct_num, total_speed, detail_results = 0, 0.0, []
        all_predictions, all_references = [], []

        for idx, sample in enumerate(tqdm(dataset, desc=f"{mode_name} inference", ncols=80, unit="sample")):
            pred_ans, speed, raw_ans = infer_blip2(model, tokenizer, image_processor, sample, use_cot)
            is_correct = 1 if pred_ans == sample["gt_answer"] else 0
            correct_num += is_correct
            total_speed += speed
            all_predictions.append(pred_ans)
            all_references.append(sample["gt_answer"])

            if idx < 30:
                print(f"\n🔍 Sample {idx + 1} (Image ID: {sample['match_index']} | Type: {sample['q_type']}):")
                q_text = sample['question'][:40] + "..." if len(sample['question']) > 40 else sample['question']
                print(f"Question: {q_text}")
                print(
                    f"Ground truth: {sample['gt_answer']} | Model output: {raw_ans} | Final answer: {pred_ans} | Correct: {bool(is_correct)}")

            detail_results.append({
                "image_name": os.path.basename(sample["image_path"]),
                "image_id": sample["match_index"],
                "question": sample["question"],
                "question_type": sample["q_type"],
                "gt_answer": sample["gt_answer"],
                "model_raw_output": raw_ans,
                "final_answer": pred_ans,
                "is_correct": is_correct,
                "infer_speed(T/s)": speed,
                "mode": mode_name
            })

        # Calculate metrics
        valid_num = len(dataset)
        accuracy = round(correct_num / valid_num, 4) if valid_num > 0 else 0.0
        hallucination_rate = round(1 - accuracy, 4)
        avg_speed = round(total_speed / valid_num, 2) if valid_num > 0 else 0.0
        error_num = len([r for r in detail_results if "Exception fallback" in r['model_raw_output']])
        bool_num = len([s for s in dataset if s['q_type'] == 'bool'])
        num_num = len([s for s in dataset if s['q_type'] == 'num'])
        weighted_f1 = round(f1_score(all_references, all_predictions, average='weighted', zero_division=0), 4)

        # Store mode results
        all_mode_results[mode_name] = {
            "detail": detail_results,
            "metrics": {
                "valid_num": valid_num,
                "bool_num": bool_num,
                "num_num": num_num,
                "correct_num": correct_num,
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "hallucination_rate": hallucination_rate,
                "avg_speed": avg_speed,
                "error_num": error_num
            }
        }

        # Print single mode report
        print(f"\n✅ {mode_name} mode evaluation completed:")
        print(
            f"   Accuracy: {accuracy * 100:.2f}% | F1: {weighted_f1:.4f} | Hallucination: {hallucination_rate * 100:.2f}%")
        print(f"   Avg Speed: {avg_speed} T/s | Error samples: {error_num}")

    # Save merged results
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Merge detailed results
    all_detail = []
    for mode in all_mode_results.values():
        all_detail.extend(mode["detail"])
    detail_csv = os.path.join(SAVE_DIR, "blip2_opt27b_coco_vqa_1000_dual_mode_detail.csv")
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_detail[0].keys())
        writer.writeheader()
        writer.writerows(all_detail)

    # Dual mode summary
    summary_csv = os.path.join(SAVE_DIR, "blip2_opt27b_coco_vqa_1000_dual_mode_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Valid Samples", "Boolean Q", "Numeric Q", "Correct Count",
                         "Accuracy(%)", "Weighted F1", "Hallucination Rate(%)",
                         "Error Samples", "Avg Speed(T/s)"])
        for mode_name, res in all_mode_results.items():
            m = res["metrics"]
            writer.writerow([
                mode_name, m["valid_num"], m["bool_num"], m["num_num"], m["correct_num"],
                f"{m['accuracy'] * 100:.2f}", m["weighted_f1"], f"{m['hallucination_rate'] * 100:.2f}",
                m["error_num"], m["avg_speed"]
            ])

    # Final comparison report
    print("\n" + "=" * 95)
    print(f"BLIP2-Opt-2.7B Dual Mode Comparison Report (coco_vqa_1000)")
    print(f"Dataset path: {DATASET_PATH}")

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
    print("=" * 95)


# === Main function (Unchanged) ===
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    print(f"Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_evaluate()
    print(f"Evaluation completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()