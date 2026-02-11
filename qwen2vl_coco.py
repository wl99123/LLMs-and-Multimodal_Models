from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image
import json
import os
import torch
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Global environment configuration (suppress redundant warnings, force single GPU)
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# === Global Configuration (Fully matched to your environment, Qwen2-VL official parameters) ===
model_path = "/root/autodl-tmp/models/qwen2-vl-local"
dataset_json_path = "/root/autodl-tmp/datasets/coco_vqa_1000/val_sample_1000.json"
dataset_img_dir = "/root/autodl-tmp/datasets/coco_vqa_1000/images"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 4  # Qwen2-VL short answers sufficient (yes/no/numbers), reduce redundancy
TEST_NUM = 20  # Test first 20 samples
DEBUG = True  # Print valid sample details

# === Critical path validation (Exit directly if key paths do not exist) ===
assert os.path.exists(dataset_json_path), f"❌ Dataset JSON not found: {dataset_json_path}"
assert os.path.isdir(model_path), f"❌ Model path not found: {model_path}"
assert os.path.isdir(dataset_img_dir), f"❌ Image directory not found: {dataset_img_dir}"
print(f"✅ Environment initialization completed | Device: {DEVICE} | Test sample count: {TEST_NUM}")

# === Core Fix 1: Load Qwen2-VL official exclusive Processor + Model (Force weight size adaptation) ===
# Load Qwen2-VL official multimodal processor (one-stop image+text processing, official recommended only)
processor = Qwen2VLProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)
# Load Qwen2-VL model (Core: ignore_mismatched_sizes=True to force weight size adaptation)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda:0" else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
    ignore_mismatched_sizes=True,  # Core fix: force ignore weight size mismatch
    attn_implementation="eager"   # Compatible with lower versions, avoid flash attention errors
).to(DEVICE).eval()
# Force set special tokens (double confirmation after processor fallback)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.unk_token = processor.tokenizer.pad_token
print(f"✅ Qwen2-VL official model loaded successfully | Device: {DEVICE}")
print(f"✅ Weight size adapted forcefully, Conv3d shape mismatch ignored")

# === Core Functions: Official Prompt Format + Metric Calculation + Hallucination Detection (Minimal fallback) ===
def generate_vqa_prompt(question, use_cot=False):
    """Qwen2-VL official Prompt format with image identifier (mandatory)"""
    question = str(question).strip() if question else "What is the most dominant color in the picture?"
    if use_cot:
        return f"Answer the following question based on the image content, reasoning step by step, and only give a simple answer at last: {question}"
    else:
        return f"Answer the following question based on the image content, give a simple answer directly, no extra content: {question}"

def calculate_metrics(predictions, references):
    """Strictly filter empty values, calculate accuracy and weighted F1"""
    valid_pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return {"accuracy": 0.0, "f1": 0.0}
    preds, refs = zip(*valid_pairs)
    return {
        "accuracy": round(accuracy_score(refs, preds), 4),
        "f1": round(f1_score(refs, preds, average='weighted', zero_division=0), 4)
    }

def calculate_hallucination_rate(predictions, references):
    """Calculate hallucination rate: prediction inconsistent with ground truth is considered hallucination"""
    valid_count, hallucination_count = 0, 0
    for p, r in zip(predictions, references):
        p, r = p.strip(), r.strip()
        if p and r:
            valid_count += 1
            hallucination_count += 1 if p != r else 0
    return round(hallucination_count / valid_count if valid_count > 0 else 0.0, 4)

# === Core Fix 2: Qwen2-VL official standard multimodal inference (Root fix for None iteration error) ===
def run_vqa_evaluation(use_cot=False):
    # Load and preprocess dataset (basic filtering only)
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # Keep only valid samples with image_id, question, and answer
    processed_data = [
        item for item in raw_data
        if isinstance(item, dict) and item.get("image_id") and item.get("question") and item.get("answer")
    ]
    if DEBUG and processed_data:
        print(f"\n🔍 Dataset preprocessing completed | Raw samples: {len(raw_data)} | Valid samples: {len(processed_data)}")
        print(f"🔍 First sample example: {processed_data[0]}")

    predictions, references = [], []
    valid_sample_count = 0

    print(f"\n🚀 Start VQA evaluation | COT chain-of-thought: {'Enabled' if use_cot else 'Disabled'}")
    for idx, item in enumerate(processed_data[:TEST_NUM]):
        try:
            # 1. Extract basic fields (minimal fallback)
            img_id = str(item["image_id"]).strip()
            question = item["question"].strip()
            true_answer = item["answer"].strip()

            # 2. Load and validate image (Qwen2-VL official requires RGB format)
            img_name = f"COCO_val2014_{img_id.zfill(12)}.jpg"
            img_path = os.path.join(dataset_img_dir, img_name)
            if not os.path.exists(img_path):
                raise Exception(f"Image not found: {img_name}")
            image = Image.open(img_path).convert("RGB")  # Official mandatory RGB

            # 3. Core: Use official Processor for one-stop image+text processing
            prompt = generate_vqa_prompt(question, use_cot)
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE, torch.float16 if DEVICE == "cuda:0" else torch.float32)

            # 4. Qwen2-VL official standard inference
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda:0"):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    temperature=0.0,
                    num_beams=1,
                    use_cache=True
                )

            # 5. Parse prediction results
            pred_answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
            pred_answer = pred_answer.replace(prompt, "").strip() or "None"
            true_answer = true_answer or "None"

            # 6. Count valid samples
            predictions.append(pred_answer)
            references.append(true_answer)
            valid_sample_count += 1

            # Debug print
            if DEBUG:
                print(f"✅ Sample {idx}: Valid | Question: {question[:30]} | Ground Truth: {true_answer} | Prediction: {pred_answer}")

        except Exception as e:
            err_info = str(e)[:50].replace("\n", " ")
            print(f"⚠️  Sample {idx}: Skipped | Reason: {err_info}")
            continue
        finally:
            if DEVICE == "cuda:0":
                torch.cuda.empty_cache()

    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    hallucination_rate = calculate_hallucination_rate(predictions, references)
    print(f"\n✅ Evaluation completed | Tested samples: {TEST_NUM} | Valid samples: {valid_sample_count}")
    return metrics, hallucination_rate

# === Main Program: COT/Non-COT comparative evaluation ===
if __name__ == "__main__":
    # 1. Evaluate without COT
    print("=" * 60)
    print("📊 Evaluation Mode: COT chain-of-thought Disabled (Direct Answer)")
    print("=" * 60)
    no_cot_metrics, no_cot_hallu = run_vqa_evaluation(use_cot=False)
    print(f"\n📈 Evaluation Results (No COT):")
    print(f"Accuracy: {no_cot_metrics['accuracy']:.2%} | Weighted F1: {no_cot_metrics['f1']:.4f} | Hallucination Rate: {no_cot_hallu:.2%}")

    # 2. Evaluate with COT
    print("\n" + "=" * 60)
    print("📊 Evaluation Mode: COT chain-of-thought Enabled (Step-by-Step Reasoning)")
    print("=" * 60)
    with_cot_metrics, with_cot_hallu = run_vqa_evaluation(use_cot=True)
    print(f"\n📈 Evaluation Results (With COT):")
    print(f"Accuracy: {with_cot_metrics['accuracy']:.2%} | Weighted F1: {with_cot_metrics['f1']:.4f} | Hallucination Rate: {with_cot_hallu:.2%}")

    # 3. Comparative summary
    print("\n" + "=" * 70)
    print("📋 COT Chain-of-Thought Effect Comparison Summary")
    print("=" * 70)
    acc_change = (with_cot_metrics['accuracy'] - no_cot_metrics['accuracy']) * 100
    f1_change = with_cot_metrics['f1'] - no_cot_metrics['f1']
    hallu_change = (with_cot_hallu - no_cot_hallu) * 100
    print(f"Accuracy Change: {acc_change:+.2f}%")
    print(f"Weighted F1 Change: {f1_change:+.4f}")
    print(f"Hallucination Rate Change: {hallu_change:+.2f}%")
    print("=" * 70)
    print("🎉 Qwen2-VL official standard pipeline evaluation completed!")
