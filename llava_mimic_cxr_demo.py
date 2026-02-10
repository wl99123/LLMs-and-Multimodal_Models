import os
import json
import re
import torch
from PIL import Image
import io
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import warnings
import glob  # Added: for traversing all parquet files
from tqdm import tqdm  # Added: for progress bar of full dataset

warnings.filterwarnings('ignore')

# ===================== Core Configuration (No Modification Needed) ======================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_ROOT = "/root/autodl-tmp/datasets/mimic-cxr-dataset/"  # Modified: Dataset root directory
OUTPUT_DIR = "/root/autodl-tmp/datasets/mimic-cxr-dataset/llava_optimized_best"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVAL_DIMENSIONS = ["Thoracic Cage", "Bilateral Lung Fields", "Lung Markings", "Cardiac Shadow", "Diaphragmatic Surface and Costophrenic Angles"]
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ===================== 1. Model Initialization (Stable Configuration) ======================
print("Initializing LLaVA-v1.5-7b model (Optimized Configuration)...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_skip_modules=["lm_head"]
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()
for param in model.parameters():
    param.requires_grad = False
print(f"Model initialization completed | Device: {DEVICE}")

# ===================== 2. Dataset Loading (Modified: Load entire dataset) ======================
print(f"\nLoading full MIMIC-CXR dataset...")

# Traverse all parquet files in the dataset directory
parquet_files = glob.glob(os.path.join(DATASET_ROOT, "*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in dataset directory! Path: {DATASET_ROOT}")
print(f"Found {len(parquet_files)} parquet files: {parquet_files}")

# Load all valid samples from all parquet files
infer_data = []
total_raw_samples = 0
for parquet_file in parquet_files:
    df_temp = pd.read_parquet(parquet_file)
    raw_data = df_temp.to_dict("records")
    total_raw_samples += len(raw_data)
    
    # Filter valid samples (no sample limit)
    for item in raw_data:
        if isinstance(item.get("image"), bytes) and len(item.get("image", b"")) >= 100:
            infer_data.append({
                "image": item["image"],
                "findings": str(item.get("findings", "No desc")).strip() or "No desc",
                "impression": str(item.get("impression", "No desc")).strip() or "No desc"
            })
    
    # Release memory immediately after processing each parquet file
    del df_temp, raw_data
    torch.cuda.empty_cache()

TOTAL_NUM_CLEAN = len(infer_data)
print(f"Dataset loading completed | Total raw samples: {total_raw_samples} | Total valid samples: {TOTAL_NUM_CLEAN}")

# ===================== 3. Core Utility Functions (All Bugs Fixed, Accurate Evaluation) ======================
def extract_medical_conclusion(report, is_original=True):
    """Extract conclusion: 0=Normal, 1=Abnormal (Optimized keywords for MIMIC-CXR)"""
    overall_con = 0
    dim_con = {dim: 0 for dim in EVAL_DIMENSIONS}
    if is_original and report != "No desc":
        report_lower = report.lower()
        # Expanded medical abnormal/normal keywords to improve parsing accuracy
        abnormal_kw = ["abnormal", "effusion", "pneumothorax", "consolidation", "nodule", "mass", "enlarged",
                       "thickening", "opacity", "atelectasis", "pleural", "cardiomegaly", "lesion", "infiltrate"]
        normal_kw = ["normal", "clear", "unremarkable", "negative", "no abnormality", "no bony abnormalities",
                     "no focal consolidation", "no pleural effusion"]
        if any(kw in report_lower for kw in abnormal_kw):
            overall_con = 1
        elif any(kw in report_lower for kw in normal_kw):
            overall_con = 0
        for dim in EVAL_DIMENSIONS:
            dim_con[dim] = overall_con
    elif not is_original:
        # Accurately match Chinese abnormal descriptions to avoid misjudgment
        abnormal_patterns = [
            r"异常", r"见.*影", r"有.*征", r"存在.*改变", r"增粗", r"紊乱",
            r"模糊", r"抬高", r"变钝", r"增多", r"增厚", r"浸润", r"不张", r"气胸"
        ]
        for dim in EVAL_DIMENSIONS:
            dim_cn = {
                "Thoracic Cage": "胸廓",
                "Bilateral Lung Fields": "双肺野",
                "Lung Markings": "肺纹理",
                "Cardiac Shadow": "心影",
                "Diaphragmatic Surface and Costophrenic Angles": "膈面及肋膈角"
            }[dim]
            if re.search(f"{dim_cn}[:：].*?({'|'.join(abnormal_patterns)})", report, re.S):
                dim_con[dim] = 1
                overall_con = 1
    return overall_con, dim_con

def judge_hallucination(ori_con, gen_con, gen_report, ori_report):
    """Fix hallucination judgment bug: Verify conclusion consistency first, then judge other cases"""
    # Core Fix 1: Complete consistency of conclusions (Normal=Normal/Abnormal=Abnormal) → No hallucination
    if ori_con == gen_con:
        return 0, "No Hallucination - Consistent Conclusion"
    # Core Fix 2: Judge hallucination by type when conclusions are inconsistent
    # 1. Prompt repetition judgment (Only when report is extremely short and contains prompt keywords)
    prompt_kw = ["Generate report based on chest X-ray", "Include thoracic cage, bilateral lung fields, lung markings"]
    if any(kw in gen_report for kw in prompt_kw) and len(gen_report) < 100:
        return 1, "Hallucination - Prompt Repetition (No Valid Analysis)"
    # 2. False Positive (Original Normal → Model Abnormal)
    if gen_con == 1 and ori_con == 0:
        return 1, "Hallucination - False Positive (Original Normal → Model Misjudged Abnormal)"
    # 3. False Negative (Original Abnormal → Model Normal, Key Focus for Medical Imaging)
    if gen_con == 0 and ori_con == 1:
        return 1, "Hallucination - False Negative (Original Abnormal → Model Missed Normal)"
    # 4. Unsubstantiated severe lesion description
    severe_kw = ["Tumor", "Cancer", "Metastasis", "Hemorrhage", "Fracture", "Abscess", "Cyst"]
    severe_kw_cn = ["肿瘤", "癌症", "转移", "出血", "骨折", "脓肿", "囊肿"]
    for kw in severe_kw + severe_kw_cn:
        if kw in gen_report and kw.lower() not in ori_report.lower():
            return 1, f"Hallucination - Unsubstantiated Severe Lesion ({kw})"
    return 0, "No Hallucination - Judgment Conditions Not Triggered"

# ===================== 4. Single Sample Inference (Core Fix: Resolve Prompt Repetition) ======================
def infer_single_optimized(sample, idx):
    sample_id = f"sample_{idx:08d}"  # Modified: 8-digit ID for large dataset
    json_path = os.path.join(OUTPUT_DIR, f"{sample_id}.json")
    result = {
        "sample_id": sample_id, "index": idx, "infer_status": "success",
        "original_report": "", "generated_report": "",
        "original_conclusion": 0, "generated_conclusion": 0,
        "is_hallucination": 0, "hallucination_type": "", "error_msg": ""
    }
    try:
        # 1. Load image (No PNG saving, significantly improve efficiency)
        img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
        # 2. Concatenate original report
        ori_report = f"FINDINGS: {sample['findings']}\nIMPRESSION: {sample['impression']}"
        result["original_report"] = ori_report
        # 3. [Core Fix 1: Optimize Prompt] Concise instructions + Force prohibit repetition to guide model focus on report generation
        prompt = (
            "<image>\nGenerate a radiological diagnosis report based on the chest X-ray, analyze the following 5 parts one by one: "
            "thoracic cage, bilateral lung fields, lung markings, cardiac shadow, diaphragmatic surface and costophrenic angles. "
            "Describe specific visual features for each part, clearly indicate the location and features if abnormalities are found, "
            "and label 'No obvious abnormalities' if no abnormalities are found. "
            "WARNING: Do not repeat this prompt content, only output the main text of the diagnosis report without any additional text."
        )
        # 4. Model encoding
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
        if inputs["input_ids"].shape[1] > 1536:
            inputs["input_ids"] = inputs["input_ids"][:, :1536]
            inputs["attention_mask"] = inputs["attention_mask"][:, :1536]
        # 5. Optimize generation parameters: Adapt to LLaVA, eliminate template output (Remove invalid temperature parameter)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,  # Increase generation length to allow model to fully analyze
                do_sample=False,
                repetition_penalty=1.5,  # Reasonable repetition penalty to avoid over-constraint
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                early_stopping=True,
                num_beams=4  # Increase beam number to improve generation quality
            )
        # 6. Parse output + [Core Fix 2: Force filter prompt residue after generation] Double insurance to clear prompt residue
        gen_report = processor.decode(outputs[0], skip_special_tokens=True).strip()
        # Define prompt keywords to filter (Full coverage of original prompt content)
        prompt_filter_kw = [
            "Generate report based on chest X-ray", "Include thoracic cage, bilateral lung fields, lung markings",
            "WARNING: Do not repeat this prompt content", "only output the main text of the diagnosis report"
        ]
        # Filter prompt residue word by word
        for kw in prompt_filter_kw:
            gen_report = gen_report.replace(kw, "").strip()
        # Clean empty lines, redundant punctuation and invalid characters at the beginning and end
        gen_report = re.sub(r'\n+', '\n', gen_report)  # Merge multiple empty lines into one
        gen_report = re.sub(r'^[。、：；！？1234567890.、：；·]+', '', gen_report)  # Remove invalid symbols/numbers at the beginning
        gen_report = gen_report.strip()
        # Assign filtered pure diagnosis report
        result["generated_report"] = gen_report
        # 7. Conclusion parsing and hallucination judgment
        ori_con, _ = extract_medical_conclusion(ori_report, is_original=True)
        gen_con, _ = extract_medical_conclusion(gen_report, is_original=False)
        is_hallu, hallu_type = judge_hallucination(ori_con, gen_con, gen_report, ori_report)
        # 8. Assign results
        result["original_conclusion"] = ori_con
        result["generated_conclusion"] = gen_con
        result["is_hallucination"] = is_hallu
        result["hallucination_type"] = hallu_type
        # Console print progress (Adjust for large dataset: print every 100 samples)
        if (idx + 1) % 100 == 0:  # Modified: print progress every 100 samples for large dataset
            print(f"\n" + "-" * 70)
            print(f"Sample ID: {sample_id} | Progress: {idx + 1}/{TOTAL_NUM_CLEAN} ({(idx + 1) / TOTAL_NUM_CLEAN * 100:.1f}%)")
            print(f"Original: {'Abnormal' if ori_con else 'Normal'} | Model: {'Abnormal' if gen_con else 'Normal'} | Hallucination: {'Yes' if is_hallu else 'No'}")
            print(f"Pure Report Length: {len(gen_report)} | Hallucination Type: {hallu_type}")
            print(f"-" * 70)
    except Exception as e:
        result["infer_status"] = "failed"
        result["error_msg"] = f"{type(e).__name__}: {str(e)[:500]}"
        if (idx + 1) % 100 == 0:
            print(f"\nSample ID: {sample_id} | Progress: {idx + 1}/{TOTAL_NUM_CLEAN} | Failed: {str(e)[:80]}")
    finally:
        # Optimize memory cleaning: Only release image cache, reduce CUDA operations
        del img
        torch.cuda.empty_cache()
        # Save single sample JSON result
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

# ===================== 5. Full Inference Main Process (Efficiency Optimization for Large Dataset) ======================
print(f"\nStarting full inference (Optimized Version) | Total Valid Samples: {TOTAL_NUM_CLEAN} | No PNG Saving + Accurate Evaluation")
print(f"Results saved to: {OUTPUT_DIR} | Print progress every 100 samples")
total_stats = {
    "total_clean": TOTAL_NUM_CLEAN,
    "success": 0, "failed": 0,
    "hallu_count": 0, "ori_con_list": [], "gen_con_list": [],
    "hallu_detail": {}  # Count the number of each type of hallucination
}

try:
    # Modified: Add tqdm progress bar for large dataset
    for idx, sample in tqdm(enumerate(infer_data), total=TOTAL_NUM_CLEAN, desc="Processing samples"):
        res = infer_single_optimized(sample, idx)
        if res["infer_status"] == "success":
            total_stats["success"] += 1
            total_stats["ori_con_list"].append(res["original_conclusion"])
            total_stats["gen_con_list"].append(res["generated_conclusion"])
            if res["is_hallucination"] == 1:
                total_stats["hallu_count"] += 1
                # Count each type of hallucination
                ht = res["hallucination_type"]
                total_stats["hallu_detail"][ht] = total_stats["hallu_detail"].get(ht, 0) + 1
        else:
            total_stats["failed"] += 1
        # Optimize memory cleaning strategy: Deep clean every 500 samples for large dataset
        if (idx + 1) % 500 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            tqdm.write(f"\nMemory Optimization: Processed {idx + 1} samples, deep cleaning completed\n")
except KeyboardInterrupt:
    print(f"\nManual interruption detected, saving current results...")
except Exception as e:
    print(f"\nUnexpected error: {str(e)[:200]}, saving results...")

# ===================== 6. Calculate Evaluation Metrics (Added F1 Score, Accurate Statistics) ======================
print(f"\nCalculating final evaluation metrics (Including Accuracy, F1 Score, Hallucination Rate)...")
eval_report = {
    "data_statistics": {
        "Total Raw Samples": total_raw_samples,
        "Total Valid Samples": TOTAL_NUM_CLEAN,
        "Number of Successful Inferences": total_stats["success"],
        "Number of Failed Inferences": total_stats["failed"],
        "Inference Success Rate (%)": round(total_stats["success"] / TOTAL_NUM_CLEAN * 100, 2) if TOTAL_NUM_CLEAN > 0 else 0.0
    },
    "core_metrics": {
        "Overall Accuracy (Normal/Abnormal Consistency) (%)": 0.0,
        "F1 Score (Binary, Normal/Abnormal) (%)": 0.0,  # Added: F1 Score you need
        "Overall Hallucination Rate (Hallucination Samples/Successful Samples) (%)": 0.0,
        "False Negative Rate (Original Abnormal → Model Normal) (%)": 0.0,  # Key metric for medical imaging
        "False Positive Rate (Original Normal → Model Abnormal) (%)": 0.0  # Key metric for medical imaging
    },
    "hallucination_statistics": {
        "Total Hallucination Samples": total_stats["hallu_count"],
        "Distribution of Hallucination Types": total_stats["hallu_detail"],
        "Proportion of Each Hallucination Type (%)": {}
    }
}

# Calculate core metrics (Including F1 Score, add key metrics for medical imaging)
if total_stats["success"] > 0:
    # 1. Overall Accuracy
    acc = accuracy_score(total_stats["ori_con_list"], total_stats["gen_con_list"])
    eval_report["core_metrics"]["Overall Accuracy (Normal/Abnormal Consistency) (%)"] = round(acc * 100, 2)
    # 2. F1 Score (Binary, for Normal/Abnormal judgment, the core metric you need)
    f1 = f1_score(total_stats["ori_con_list"], total_stats["gen_con_list"], average="binary")
    eval_report["core_metrics"]["F1 Score (Binary, Normal/Abnormal) (%)"] = round(f1 * 100, 2)
    # 3. Overall Hallucination Rate
    eval_report["core_metrics"]["Overall Hallucination Rate (Hallucination Samples/Successful Samples) (%)"] = round(
        total_stats["hallu_count"] / total_stats["success"] * 100, 2)
    # 4. False Negative Rate/False Positive Rate (Key metrics for medical imaging)
    fn = 0  # False Negative: Original 1 → Model 0
    fp = 0  # False Positive: Original 0 → Model 1
    for o, g in zip(total_stats["ori_con_list"], total_stats["gen_con_list"]):
        if o == 1 and g == 0:
            fn += 1
        elif o == 0 and g == 1:
            fp += 1
    eval_report["core_metrics"]["False Negative Rate (Original Abnormal → Model Normal) (%)"] = round(fn / total_stats["success"] * 100, 2)
    eval_report["core_metrics"]["False Positive Rate (Original Normal → Model Abnormal) (%)"] = round(fp / total_stats["success"] * 100, 2)
    # 5. Proportion of hallucination types
    if total_stats["hallu_count"] > 0:
        for ht, cnt in total_stats["hallu_detail"].items():
            eval_report["hallucination_statistics"]["Proportion of Each Hallucination Type (%)"][ht] = round(
                cnt / total_stats["hallu_count"] * 100, 2)

# Save evaluation report (Modified: Remove "first 100" from filename)
eval_report_path = os.path.join(OUTPUT_DIR, "optimized_evaluation_report_full_dataset.json")
with open(eval_report_path, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, ensure_ascii=False, indent=2)

# ===================== 7. Print Final Results (Including F1 Score, All Core Metrics) ======================
print(f"\n" + "=" * 80)
print(f"LLaVA-v1.5-7b Inference Optimized Version Completed! (Full Dataset Processed)")
print(f"All results saved to: {OUTPUT_DIR}")
print(f"=" * 80)
print(f"\nInference Basic Statistics (Full Dataset):")
for k, v in eval_report["data_statistics"].items():
    print(f"   • {k}: {v}")
print(f"\nCore Evaluation Metrics (Including F1 Score You Need):")
for k, v in eval_report["core_metrics"].items():
    print(f"   • {k}: {v}")
print(f"\nHallucination Detailed Statistics:")
if eval_report["hallucination_statistics"]["Total Hallucination Samples"] > 0:
    print(f"   • Total Hallucination Samples: {eval_report['hallucination_statistics']['Total Hallucination Samples']}")
    print(f"   • Hallucination Type Distribution (By Proportion):")
    for ht, ratio in sorted(eval_report["hallucination_statistics"]["Proportion of Each Hallucination Type (%)"].items(), key=lambda x: x[1],
                            reverse=True):
        print(f"      - {ht}: {ratio}%")
else:
    print(f"   • No hallucination samples!")
print(f"\nKey Result Files:")
print(f"   • {eval_report_path}: Optimized Evaluation Report (Full Dataset, Including F1 Score)")
print(f"   • sample_xxxxxxxx.json: Single Sample Details (Pure Diagnosis Report + Accurate Evaluation)")
print(f"\n" + "=" * 80)
print(f"Fix Summary: 1. Optimize Prompt to prohibit repetition 2. Force filter prompt residue after generation 3. Add F1 Score calculation 4. Process full dataset")

print(f"=" * 80)
