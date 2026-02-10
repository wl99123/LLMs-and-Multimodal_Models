import os
import json
import re
import torch
import torch.nn as nn
from PIL import Image
import io
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          CLIPImageProcessor, CLIPVisionModel,
                          BitsAndBytesConfig)
import warnings
import glob  # Added: for traversing all parquet files
from tqdm import tqdm  # Added: for progress bar

# Environment Configuration
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ===================== Core Configuration (Dimension Adaptation) ======================
MODEL_PATH = "/root/autodl-tmp/models/llava-med-v1.5"  # LLaVA-MED root directory
VISION_MODEL_PATH = "/root/autodl-tmp/models/clip-vit-large-patch14"  # Local CLIP path
DATASET_ROOT = "/root/autodl-tmp/datasets/mimic-cxr-dataset/"  # Dataset root directory
OUTPUT_DIR = "/root/autodl-tmp/datasets/mimic-cxr-dataset/llava_med_multimodal_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()

# LLaVA-MED fixed special tokens
IMAGE_TOKEN = "<image>"
IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_START_TOKEN = "<im_start>"
IMAGE_END_TOKEN = "<im_end>"

# Dimension configuration: CLIP-ViT-L/14 outputs 1024 dimensions, LLaVA-MED expects 4096 dimensions
VISION_FEATURE_DIM = 1024  # CLIP output dimension
LLAVA_EXPECTED_DIM = 4096  # LLaVA-MED input dimension

# ===================== 8-bit Quantization Configuration (Retained) ======================
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=TORCH_DTYPE
)


# ===================== Check Local CLIP Model Files ======================
def check_local_clip_model():
    required_files = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(VISION_MODEL_PATH, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    if missing_files:
        raise FileNotFoundError(f"Local CLIP model missing files: {missing_files} | Path: {VISION_MODEL_PATH}")
    print(f"✅ Local CLIP model files verified: {VISION_MODEL_PATH}")


# ===================== Define Dimension Mapping Layer (Core Fix) ======================
class VisionFeatureMapper(nn.Module):
    """Map CLIP's 1024-dimensional features to 4096-dimensional features required by LLaVA-MED"""

    def __init__(self, input_dim=1024, output_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # Initialize weights (avoid instability caused by random initialization)
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.projection(x)


# ===================== Load Multimodal Components (with Dimension Mapping) ======================
def load_llava_med_multimodal():
    print("Loading LLaVA-MED-v1.5 Multimodal Model (Dimension Fixed)...")
    # 1. Verify local CLIP model
    check_local_clip_model()

    # 2. Load local vision processor and vision encoder
    vision_processor = CLIPImageProcessor.from_pretrained(
        VISION_MODEL_PATH,
        local_files_only=True
    )
    vision_model = CLIPVisionModel.from_pretrained(
        VISION_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).eval()
    print(f"✅ Local vision model loaded: {VISION_MODEL_PATH}")

    # 3. Load dimension mapping layer (core fix)
    feature_mapper = VisionFeatureMapper(VISION_FEATURE_DIM, LLAVA_EXPECTED_DIM).to(DEVICE, TORCH_DTYPE).eval()
    print(f"✅ Dimension mapping layer loaded: {VISION_FEATURE_DIM} → {LLAVA_EXPECTED_DIM}")

    # 4. Load text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True
    )
    # Add image-related special tokens
    special_tokens = {
        "additional_special_tokens": [IMAGE_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]
    }
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✅ Text tokenizer loaded (vocab size: {len(tokenizer)})")

    # 5. Load LLaVA-MED language model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        local_files_only=True
    ).eval()
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ LLaVA-MED language model loaded (8-bit quantized)")

    return tokenizer, model, vision_processor, vision_model, feature_mapper


# Load multimodal model (with dimension mapping)
try:
    tokenizer, model, vision_processor, vision_model, feature_mapper = load_llava_med_multimodal()
except Exception as e:
    print(f"Multimodal model loading failed: {str(e)[:300]}")
    exit(1)

# ===================== Dataset Loading (Modified: Load entire dataset) ======================
print(f"\nLoading full MIMIC-CXR dataset...")

# Traverse all parquet files
parquet_files = glob.glob(os.path.join(DATASET_ROOT, "*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found! Path: {DATASET_ROOT}")
print(f"Found {len(parquet_files)} parquet files: {parquet_files}")

# Load all data and filter valid samples
infer_data = []
total_raw_samples = 0
for parquet_file in parquet_files:
    df_temp = pd.read_parquet(parquet_file)
    raw_data = df_temp.to_dict("records")
    total_raw_samples += len(raw_data)
    
    # Filter valid samples (with valid image data)
    for item in raw_data:
        if isinstance(item.get("image"), bytes) and len(item.get("image", b"")) >= 100:
            try:
                img = Image.open(io.BytesIO(item["image"])).convert("RGB")
                infer_data.append({
                    "image": img,
                    "findings": str(item.get("findings", "No description")).strip() or "No description",
                    "impression": str(item.get("impression", "No description")).strip() or "No description"
                })
            except Exception as e:
                continue
    del df_temp, raw_data  # Release memory

TOTAL_NUM_CLEAN = len(infer_data)
print(f"✅ Dataset loaded successfully | Total raw samples: {total_raw_samples} | Valid multimodal samples: {TOTAL_NUM_CLEAN}")


# ===================== Utility Functions (English Version) ======================
def extract_medical_conclusion(report, is_original=True):
    overall_con = 0
    if not is_original and report.strip() != "":
        conclusion_match = re.search(r'Overall Conclusion: (Normal|Abnormal)', report, re.IGNORECASE)
        if conclusion_match:
            return 1 if conclusion_match.group(1).lower() == "abnormal" else 0
        abnormal_en_kw = [
            "abnormal", "effusion", "pneumothorax", "consolidation", "nodule", "mass",
            "enlarged", "thickening", "opacity", "infiltrate", "atelectasis", "pleural",
            "edema", "fibrosis", "inflammation", "infection", "exudation", "cavity",
            "increased", "thickened", "infiltration", "collapse", "emphysema", "bullae",
            "mediastinal shift", "increased density", "decreased density", "blunted",
            "elevated", "enlargement", "distortion", "irregular", "calcification"
        ]
        if any(kw in report.lower() for kw in abnormal_en_kw):
            overall_con = 1
        return overall_con
    if is_original and report != "No description":
        report_lower = report.lower()
        abnormal_kw = [
            "abnormal", "effusion", "pneumothorax", "consolidation", "nodule", "mass",
            "enlarged", "thickening", "opacity", "infiltrate", "atelectasis", "pleural",
            "edema", "fibrosis", "inflammation", "infection", "exudation", "cavity"
        ]
        if any(kw in report_lower for kw in abnormal_kw):
            overall_con = 1
    return overall_con


def judge_hallucination(ori_con, gen_con):
    if ori_con == gen_con:
        return 0, "No Hallucination - Consistent Conclusion"
    if gen_con == 0 and ori_con == 1:
        return 1, "Hallucination - False Negative (Abnormal → Normal)"
    if gen_con == 1 and ori_con == 0:
        return 1, "Hallucination - False Positive (Normal → Abnormal)"
    return 1, "Hallucination - No Valid Analysis"


# ===================== Multimodal Inference Core Function (Dimension Adaptation Fixed) ======================
def infer_single_multimodal(sample, idx):
    sample_id = f"sample_{idx:06d}"
    result = {
        "sample_id": sample_id, "index": idx, "status": "success",
        "original_report": "", "generated_report": "",
        "original_con": 0, "gen_con": 0,
        "is_hallucination": 0, "hallu_type": "", "error": ""
    }
    try:
        # 1. Original report processing
        ori_report = f"FINDINGS: {sample['findings']}\nIMPRESSION: {sample['impression']}"
        result["original_report"] = ori_report
        result["original_con"] = extract_medical_conclusion(ori_report, is_original=True)

        # 2. Multimodal Prompt construction
        prompt = (
            f"{IMAGE_TOKEN}You are a senior radiologist, analyze the chest X-ray strictly:\n"
            f"Clinical findings: {sample['findings'][:200]} (truncated)\n"
            f"Analysis requirements: \n"
            f"1. Analyze thoracic cage, bilateral lung fields, lung markings, cardiac shadow, diaphragmatic surface, costophrenic angles one by one;\n"
            f"2. For each part, clearly state 'normal' or 'abnormal' (if abnormal, briefly describe the feature);\n"
            f"3. At the END, MUST add a line in fixed format: 'Overall Conclusion: [Normal/Abnormal]';\n"
            f"Output requirements: Only medical analysis, no redundant remarks."
        )

        # 3. Vision feature encoding + dimension mapping (core fix)
        image = sample["image"]
        vision_inputs = vision_processor(images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
        with torch.no_grad():
            # CLIP outputs 1024-dimensional features
            clip_features = vision_model(**vision_inputs).last_hidden_state  # (1, 257, 1024)
            # Map to 4096-dimensional features
            image_embeds = feature_mapper(clip_features)  # (1, 257, 4096)

        # 4. Text + image fusion encoding
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(DEVICE)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        image_token_idx = (input_ids == tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)).nonzero(as_tuple=True)[1][0]

        # Build fused input embeddings
        text_embeds = model.get_input_embeddings()(input_ids)  # Text embeddings dimension: (1, N, 4096)
        text_embeds = torch.cat([
            text_embeds[:, :image_token_idx + 1, :],
            image_embeds.to(TORCH_DTYPE),
            text_embeds[:, image_token_idx + 1:, :]
        ], dim=1)

        # Adjust attention mask
        new_attention_mask = torch.cat([
            attention_mask[:, :image_token_idx + 1],
            torch.ones((1, image_embeds.shape[1]), device=DEVICE),
            attention_mask[:, image_token_idx + 1:]
        ], dim=1)

        # 5. Multimodal inference
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=text_embeds,
                attention_mask=new_attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                num_beams=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True
            )

        # 6. Parse output
        gen_report = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        gen_report = gen_report.replace(prompt, "").strip()
        gen_report = re.sub(r'\n+', '\n', gen_report)
        gen_report = gen_report if len(gen_report) > 10 else "No valid analysis generated"

        # 7. Result statistics
        result["generated_report"] = gen_report
        result["gen_con"] = extract_medical_conclusion(gen_report, is_original=False)
        result["is_hallucination"], result["hallu_type"] = judge_hallucination(result["original_con"],
                                                                               result["gen_con"])

        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"\n" + "-" * 80)
            print(f"Sample {idx + 1}/{TOTAL_NUM_CLEAN} | ID: {sample_id}")
            print(f"Original: {'Abnormal' if result['original_con'] else 'Normal'}")
            print(f"Model: {'Abnormal' if result['gen_con'] else 'Normal'}")
            print(f"Hallucination: {'Yes' if result['is_hallucination'] else 'No'} | {result['hallu_type']}")
            print(f"Report Preview: {gen_report[:150]}...")
            print(f"-" * 80)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"Sample {sample_id} failed: {str(e)[:100]}")
    finally:
        # Clean up GPU memory
        if 'vision_inputs' in locals(): del vision_inputs
        if 'text_inputs' in locals(): del text_inputs
        if 'outputs' in locals(): del outputs
        torch.cuda.empty_cache()
        # Save results
        with open(os.path.join(OUTPUT_DIR, f"{sample_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ===================== Main Inference Process (Modified: Process entire dataset) ======================
print(f"\nStarting LLaVA-MED-v1.5 Multimodal Inference (Dimension Fixed)...")
total_stats = {
    "total": TOTAL_NUM_CLEAN, "success": 0, "failed": 0,
    "hallu_count": 0, "ori_cons": [], "gen_cons": []
}

# Use tqdm for progress bar
for idx, sample in tqdm(enumerate(infer_data), total=TOTAL_NUM_CLEAN, desc="Processing samples"):
    res = infer_single_multimodal(sample, idx)
    if res["status"] == "success":
        total_stats["success"] += 1
        total_stats["ori_cons"].append(res["original_con"])
        total_stats["gen_cons"].append(res["gen_con"])
        if res["is_hallucination"] == 1:
            total_stats["hallu_count"] += 1
    else:
        total_stats["failed"] += 1

# ===================== Metrics Calculation and Output (English Version) ======================
metrics = {
    "inference_success_rate(%)": round(total_stats["success"] / total_stats["total"] * 100, 2),
    "overall_accuracy(%)": round(accuracy_score(total_stats["ori_cons"], total_stats["gen_cons"]) * 100, 2) if
    total_stats["success"] > 0 else 0.0,
    "f1_score": round(
        f1_score(total_stats["ori_cons"], total_stats["gen_cons"], average='binary', zero_division=0) if total_stats[
                                                                                                             "success"] > 0 else 0.0,
        4),
    "hallucination_rate(%)": round(total_stats["hallu_count"] / total_stats["success"] * 100, 2) if total_stats[
                                                                                                        "success"] > 0 else 0.0,
    "false_negative_rate(%)": round(
        sum(1 for o, g in zip(total_stats["ori_cons"], total_stats["gen_cons"]) if o == 1 and g == 0) / total_stats[
            "success"] * 100, 2) if total_stats["success"] > 0 else 0.0,
    "false_positive_rate(%)": round(
        sum(1 for o, g in zip(total_stats["ori_cons"], total_stats["gen_cons"]) if o == 0 and g == 1) / total_stats[
            "success"] * 100, 2) if total_stats["success"] > 0 else 0.0
}

# Save final report
with open(os.path.join(OUTPUT_DIR, "llava_med_multimodal_fixed_evaluation_report.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_info": "LLaVA-MED-v1.5 Multimodal (Dimension Fixed: CLIP 1024 → LLaVA 4096)",
        "dataset_info": f"MIMIC-CXR | {TOTAL_NUM_CLEAN} samples | Chest X-ray + Clinical Findings",
        "dimension_fix_info": f"CLIP output dim: {VISION_FEATURE_DIM} → LLaVA expected dim: {LLAVA_EXPECTED_DIM}",
        "basic_statistics": total_stats,
        "core_metrics": metrics,
        "local_vision_model_path": VISION_MODEL_PATH
    }, f, ensure_ascii=False, indent=2)

# Print final results
print(f"\n" + "=" * 80)
print(f"LLaVA-MED-v1.5 Multimodal Inference Completed (Dimension Fixed)")
print(f"Result Directory: {OUTPUT_DIR}")
print(f"=" * 80)
print(f"\nBasic Statistics:")
print(f"   • Total samples: {total_stats['total']}")
print(f"   • Success count: {total_stats['success']} | Failed count: {total_stats['failed']}")
print(f"   • Success rate: {metrics['inference_success_rate(%)']}%")
print(f"\nCore Metrics (Dimension Fixed Mode):")
print(f"   • Overall Accuracy: {metrics['overall_accuracy(%)']}%")
print(f"   • F1 Score (Binary): {metrics['f1_score']}")
print(f"   • Hallucination Rate: {metrics['hallucination_rate(%)']}%")
print(f"   • False Negative Rate: {metrics['false_negative_rate(%)']}%")
print(f"   • False Positive Rate: {metrics['false_positive_rate(%)']}%")

print(f"\n" + "=" * 80)
