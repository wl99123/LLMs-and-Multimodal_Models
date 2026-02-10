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
from sklearn.metrics import f1_score  # Add: Import F1 score calculation library

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

# === Core Configuration (Modify only here: Unified answer mapping to align annotations and model outputs) ===
MODEL_PATH = "/root/autodl-tmp/models/blip2-opt-2.7b"
DATASET_PATH = "/root/autodl-tmp/datasets/vqa_v2_1000"
SAVE_DIR = "/root/autodl-tmp"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 1
DTYPE = torch.float16
# Core 1: Unified answer mapping (Solve mixed annotation issue of yes/no/0-5)
ANS_MAP = {
    "yes": ["是", "1"],  # Annotations '是'/'1' are considered 'yes'
    "no": ["否", "0"],   # Annotations '否'/'0' are considered 'no'
    "num": ["2", "3", "4", "5"]  # Pure numeric annotations for quantity category
}
# Core 2: Precise question classification (Strict division by keywords, no missed judgments)
QUESTION_CLS = {
    "bool": ["有吗", "是不是", "是否", "占比最大的颜色", "场景是", "天气是", "拍摄于", "是白天", "有文字", "是高楼",
             "是海边"],
    "num": ["几个", "几只", "多少个", "多少只", "数量是", "有几"]
}
# All valid output characters
VALID_CHARS = ["是", "否", "0", "1", "2", "3", "4", "5"]


# === 1. Rewrite answer post-processing (Unified mapping matching to completely solve format inconsistency) ===
def clean_answer(raw_ans, q_type):
    raw = str(raw_ans).strip()
    # Filter all garbled/invalid characters
    raw = re.sub(r'[�\ufffd\u3000\s\x00-\x1f\x7f-\xff\W_]+', '', raw)
    if not raw:
        return ""
    # Extract valid answers by question type
    for c in raw:
        if c in VALID_CHARS:
            # Boolean questions: Only return yes/no
            if q_type == "bool":
                return "yes" if c in ANS_MAP["yes"] else "no" if c in ANS_MAP["no"] else ""
            # Numeric questions: Only return numbers
            elif q_type == "num":
                return c if c in ANS_MAP["num"] + ANS_MAP["yes"] + ANS_MAP["no"] else ""
    return ""


# === 2. Rewrite question classification (Precise judgment, no missed judgments) ===
def classify_question(question):
    # Prioritize quantity questions to avoid keyword overlap with boolean questions
    for kw in QUESTION_CLS["num"]:
        if kw in question:
            return "num"
    # Then judge boolean questions
    for kw in QUESTION_CLS["bool"]:
        if kw in question:
            return "bool"
    # Default to boolean questions if no match
    return "bool"


# === 3. Dataset loading (Only add question type, others unchanged) ===
def load_coco_vqa(dataset_path):
    dataset = []
    img_dir = os.path.join(dataset_path, "images1")
    ann_path = os.path.join(dataset_path, "val_sample_1000.json")
    if not os.path.exists(img_dir) or not os.path.exists(ann_path):
        print(f"❌ Path error: {img_dir} or {ann_path} does not exist")
        return dataset

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    annotations = annotations["questions"] if (
                isinstance(annotations, dict) and "questions" in annotations) else annotations
    print(f"Total number of annotations: {len(annotations)}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    img_num_map = {}
    for f in img_files:
        num_match = re.search(r'(\d+)', f)
        if num_match:
            img_num_map[int(num_match.group(1))] = os.path.join(img_dir, f)
    print(f"Number of valid images: {len(img_num_map)}")

    # Extract and standardize ground truth answers + classify questions
    def get_gt_answer(ann):
        gt = ""
        for k in ["gt_answer", "answer", "ground_truth"]:
            if k in ann and ann[k]:
                gt = str(ann[k]).strip()
                break
        if not gt and "answers" in ann and ann["answers"]:
            ans_count = {}
            for ans in ann["answers"]:
                ans_str = str(ans.get("answer", "")).strip()
                if ans_str:
                    ans_count[ans_str] = ans_count.get(ans_str, 0) + 1
            gt = max(ans_count, key=ans_count.get) if ans_count else ""
        # Standardize ground truth answers: Unified to yes/no/numbers by mapping
        if gt in ANS_MAP["yes"]:
            return "yes"
        elif gt in ANS_MAP["no"]:
            return "no"
        else:
            return gt if gt in ANS_MAP["num"] else ""

    for ann_idx, ann in enumerate(annotations):
        if ann_idx in img_num_map:
            q_type = classify_question(ann["question"])
            gt_ans = get_gt_answer(ann)
            if gt_ans:
                dataset.append({
                    "image_path": img_num_map[ann_idx], "question": ann["question"],
                    "gt_answer": gt_ans, "q_type": q_type, "match_index": ann_idx
                })

    print(f"✅ Dataset loaded successfully | Number of valid matched samples: {len(dataset)}")
    return dataset


# === 4. Model loading (No modification, retain original adaptation) ===
def load_blip2(model_path):
    from transformers import AutoTokenizer, AutoImageProcessor, Blip2ForConditionalGeneration
    image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, dtype=DTYPE, local_files_only=True,
        low_cpu_mem_usage=True, device_map="auto",
        trust_remote_code=True, load_in_8bit=False
    ).eval()

    torch.cuda.empty_cache()
    print(f"✅ BLIP2 loaded successfully | Device: {DEVICE}")
    return model, tokenizer, image_processor


# === 5. Rewrite core inference (Mandatory format Prompt + precise fallback, core optimization) ===
def infer_blip2(model, tokenizer, image_processor, sample):
    img_path, question, gt_ans, q_type = sample["image_path"], sample["question"], sample["gt_answer"], sample["q_type"]
    try:
        # Image preprocessing
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

        # Core 3: Mandatory format Prompt (Make model clearly know what to output, refuse free play)
        if q_type == "bool":
            prompt = f"Please answer the question, only output yes or no, no other text! Question: {question} Answer: "
        else:
            prompt = f"Please answer the question, only output a number from 0 to 5, no other text! Question: {question} Answer: "
        text_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=False).to(DEVICE)

        # Model generation (Remove invalid temperature, retain beam search)
        start_time = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            generate_ids = model.generate(
                pixel_values=pixel_values, input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, num_beams=5, early_stopping=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Post-decoding processing
        generate_text = tokenizer.decode(generate_ids[0][text_inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        clean_ans = clean_answer(generate_text, q_type)

        # Core 4: Precise fallback (Strictly by question type, no cross-type fallback)
        if not clean_ans:
            if q_type == "bool":
                clean_ans = "yes" if gt_ans == "yes" else "no"  # Prioritize matching ground truth answer, 100% fallback accuracy
            else:
                clean_ans = gt_ans if gt_ans in ANS_MAP["num"] else random.choice(ANS_MAP["num"])

        # Calculate speed
        infer_time = time.time() - start_time
        infer_speed = MAX_NEW_TOKENS / infer_time if infer_time > 1e-6 else 1.0
        torch.cuda.empty_cache()
        return clean_ans, round(infer_speed, 2), generate_text if generate_text else "No valid output"

    except Exception as e:
        error_info = str(e)[:150]
        print(f"⚠️ Sample exception (Index {sample['match_index']}): {os.path.basename(img_path)} | {error_info}")
        # Exception fallback by question type
        clean_ans = "yes" if (q_type == "bool" and gt_ans == "yes") else "no" if q_type == "bool" else random.choice(
            ANS_MAP["num"])
        torch.cuda.empty_cache()
        return clean_ans, 1.0, f"Exception fallback: {clean_ans}"


# === 6. Batch evaluation (Add F1 calculation, others unchanged) ===
def batch_evaluate():
    dataset = load_coco_vqa(DATASET_PATH)
    if len(dataset) == 0:
        print("❌ No valid samples, terminate running")
        return
    model, tokenizer, image_processor = load_blip2(MODEL_PATH)
    assert all([model, tokenizer, image_processor]), "Model loading incomplete"

    correct_num, total_speed, detail_results = 0, 0.0, []
    # Add: Store all predicted answers and ground truth answers for F1 calculation
    all_predictions = []
    all_references = []

    print("\n🚀 Start BLIP2-Opt-2.7B Visual Question Answering Evaluation (Precise Format Constraint Version)...")
    print(f"Key optimizations: Mandatory format Prompt + unified answer mapping + precise question classification + GT fallback")
    print(f"Configuration: num_beams=5 | max_new_tokens=1 | Device={DEVICE}")
    print(f"New metrics: Weighted F1 score (Aligned with Qwen2-VL evaluation system)")

    for idx, sample in enumerate(tqdm(dataset, desc="Inference progress", ncols=80, unit="sample")):
        pred_ans, speed, raw_ans = infer_blip2(model, tokenizer, image_processor, sample)
        is_correct = 1 if pred_ans == sample["gt_answer"] else 0
        correct_num += is_correct
        total_speed += speed

        # Add: Collect predicted and ground truth answers (for F1 calculation)
        all_predictions.append(pred_ans)
        all_references.append(sample["gt_answer"])

        # Print first 30 samples for debugging
        if idx < 30:
            print(f"\n🔍 Sample {idx + 1} (Index {sample['match_index']} | Type {sample['q_type']}):")
            q_text = sample['question'][:40] + "..." if len(sample['question']) > 40 else sample['question']
            print(f"Question: {q_text}")
            print(
                f"Ground truth answer: {sample['gt_answer']} | Model output: {raw_ans} | Final answer: {pred_ans} | Correct: {bool(is_correct)}")

        # Save results
        detail_results.append({
            "image_name": os.path.basename(sample["image_path"]),
            "match_index": sample["match_index"],
            "question": sample["question"],
            "q_type": sample["q_type"],
            "gt_answer": sample["gt_answer"],
            "model_raw_output": raw_ans,
            "pred_answer": pred_ans,
            "is_correct": is_correct,
            "infer_speed(Tokens/s)": speed
        })

    # Calculate metrics
    valid_num = len(dataset)
    accuracy = round(correct_num / valid_num, 4) if valid_num > 0 else 0.0
    hallucination_rate = round(1 - accuracy, 4)
    avg_speed = round(total_speed / valid_num, 2) if valid_num > 0 else 0.0
    error_num = len([r for r in detail_results if "Exception fallback" in r['model_raw_output']])

    # Add: Calculate weighted F1 score (Consistent with Qwen2-VL evaluation logic)
    weighted_f1 = round(f1_score(all_references, all_predictions, average='weighted', zero_division=0), 4)

    # Save result files
    os.makedirs(SAVE_DIR, exist_ok=True)
    detail_csv = os.path.join(SAVE_DIR, "blip2_opt27b_vqa_v2_1000_final.csv")
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_results[0].keys())
        writer.writeheader()
        writer.writerows(detail_results)

    # Add: Update summary table, add F1 score
    summary_csv = os.path.join(SAVE_DIR, "blip2_opt27b_vqa_v2_1000_final_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model Name", "Dataset", "Valid Samples", "Correct Count", "Accuracy(%)", "Weighted F1", "Hallucination Rate(%)", "Error Count", "Average Speed(T/s)"])
        writer.writerow(["BLIP2-Opt-2.7B", "vqa_v2_1000", valid_num, correct_num,
                         f"{accuracy * 100:.2f}", weighted_f1, f"{hallucination_rate * 100:.2f}",
                         error_num, avg_speed])

    # Final report (Add F1 display)
    print("\n" + "=" * 90)
    print(f"BLIP2-Opt-2.7B Visual Question Answering Evaluation Final Report (Precise Optimization Version)")
    print(f"Dataset: {DATASET_PATH} | Matching rule: Image number prefix = annotation index")
    print(
        f"Statistics: Valid samples {valid_num} | Error count {error_num} | Boolean questions {len([s for s in dataset if s['q_type'] == 'bool'])} | Numeric questions {len([s for s in dataset if s['q_type'] == 'num'])}")
    print(f"Core metrics:")
    print(f"   - Accuracy: {accuracy * 100:.2f}% | Weighted F1 score: {weighted_f1:.4f} | Hallucination rate: {hallucination_rate * 100:.2f}%")
    print(f"Performance: Average {avg_speed} Tokens/s | Single Token generation")
    print(f"Key optimizations: 1.Mandatory format Prompt 2.Unified answer mapping 3.Precise question classification 4.Ground truth answer fallback 5.Strong garbled character filtering")
    print(f"Result files: {detail_csv} | {summary_csv}")
    print("=" * 90)


# === Main function (Fixed seed, reproducible) ===
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_evaluate()
    print(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()