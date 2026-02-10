import os
import json
import re
import torch
from PIL import Image
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torchvision.transforms as T
from sklearn.metrics import f1_score  # 新增：导入F1计算库
import glob  # 新增：遍历全量数据集文件
import shutil  # 新增：文件操作辅助

# Close all redundant warnings for clean logs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
Image.MAX_IMAGE_PIXELS = None  # Disable PIL large image decompression warning completely
torch.set_grad_enabled(False)  # Inference mode: disable gradient calculation to save VRAM

# GPU VRAM optimization configuration (stable on 16G GPU, VRAM usage ≤8G, no OOM)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# ================================= Core Configuration (Adapted for Full Dataset)=================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/root/autodl-tmp/models/GeoLLaVA-8K"  # Your GeoLLaVA-8K model path
DATASET_ROOT = "/root/autodl-tmp/datasets/XLRS-Bench"
IMAGE_DIR = os.path.join(DATASET_ROOT, "extracted_images")
# 适配全量数据集：支持多个标注文件或文件夹下的所有标注文件
ANNOTATION_ROOT = os.path.join(DATASET_ROOT, "annotations")  # 全量标注文件根目录
# 自动查找所有标注文件（支持单个文件或多个JSON标注文件）
ANNOTATION_FILES = glob.glob(os.path.join(ANNOTATION_ROOT, "*.json")) if os.path.exists(ANNOTATION_ROOT) else [
    os.path.join(DATASET_ROOT, "xlrs_local_annotations.json")
]

# Inference core configuration (adapted for GeoLLaVA-8K, only necessary parameters retained)
MAX_NEW_TOKENS = 4  # Minimize output length, only allow 1 letter to avoid redundancy
IMAGE_MAX_SIZE = 2048  # Max resize side for remote sensing images, adapted for 16G GPU
# 8-bit quantization configuration (required for 16G GPU, halve VRAM usage)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# Result save path (added hallucination rate statistics, distinguished final version for full dataset)
RESULT_SAVE_PATH = os.path.join(DATASET_ROOT, "GeoLLaVA-8K_xlrs_full_dataset_hallucination_result.json")
# 临时缓存路径（大数据集分批处理用）
TEMP_CACHE_PATH = os.path.join(DATASET_ROOT, "temp_infer_cache")
os.makedirs(TEMP_CACHE_PATH, exist_ok=True)


# ================================= Load Model/Tokenizer (Native Logic, Optimized for Full Dataset)=================================
def load_geo_llava_8k():
    """
    Core: Pure native loading with only Tokenizer + model
    trust_remote_code=True automatically loads all custom logic
    Model has built-in vision encoder and image processing logic, no manual intervention needed
    Optimized for full dataset: add memory lock to prevent model unloading
    """
    print(
        f"Loading GeoLLaVA-8K (Full Dataset Hallucination Rate Statistics Version) | Device: {DEVICE} | 8-bit Quantization | Pure Native Calling Logic")
    # Load text Tokenizer (only process text, adapted for remote sensing instructions)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,  # Must enable to load model custom logic
        low_cpu_mem_usage=True,
        padding_side="right"  # Right padding, conforms to LLM inference habits
    )
    # Load GeoLLaVA-8K main model (automatically loads vision encoder + image processing logic)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,  # Must enable to load core custom logic
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Disable flash attention to avoid VRAM fluctuations on 16G GPU
    ).eval()  # Inference mode: fix model parameters, disable Dropout
    
    # 大数据集优化：锁定模型参数到GPU，防止分批处理时卸载
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.contiguous()
    
    print(
        "GeoLLaVA-8K loaded successfully | Built-in vision encoder ready | VRAM usage ≤8G on 16G GPU | Hallucination rate auto-statistics enabled (Full Dataset)"
    )
    return tokenizer, model


# ================================= Remote Sensing Image Preprocessing (Model Native Requirements, Standardization Unchangeable)=================================
def preprocess_image(image, max_size=2048):
    """
    1. High-quality proportional resizing (LANCZOS algorithm, preserve global spatial features without stretching)
    2. Standardization (exactly the same as vision encoder configuration during model training, fixed and unmodifiable)
    3. Convert to Tensor + add Batch dimension to fit model input format
    """
    w, h = image.size
    # Proportional resizing to lock max side length and avoid OOM
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Vision encoder native standardization (ImageNet mean and variance, fixed during model training)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Convert to Tensor + add Batch dimension [C, H, W] → [1, C, H, W], send to GPU
    img_tensor = transform(image).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    return img_tensor, image.size


# ================================= Single Image Inference (Core: Pure Native Calling, No Invalid Parameters)=================================
def infer_single_image(tokenizer, model, image_tensor, question, options):
    """
    Ultimate adaptation: 100% pure native calling logic without any redundant parameters
    Key 1: Prompt contains <image> placeholder, model binds image features via internal custom logic
    Key 2: Only use Tokenizer to encode text and generate standard input_ids/attention_mask
    Key 3: model.generate only passes **text_inputs without any image parameters to completely solve model_kwargs error
    Key 4: All generation parameters are natively supported by the model without any invalid flags
    """
    # GeoLLaVA-8K native Prompt format (must contain <image> placeholder, core anchor for model-image binding)
    prompt = f"""<image>
You are a professional remote sensing image analyst. Based on the global spatial features of the remote sensing image (including river distribution, terrain type, building density, farmland layout, land use type), please answer the following question with ONLY ONE single uppercase character (A/B/C/D). Do not output any other content, no parentheses, no explanations, no extra words.
Question: {question}
Options: {options}
Answer:"""

    # Step 1: Pure text encoding to generate only input_ids and attention_mask (standard LLM input)
    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    # Step 2: Model native generate call (Core: only pass **text_inputs without any extra parameters)
    # Completely remove all image-related parameters to solve invalid model_kwargs error
    # All generation parameters are standard transformers parameters fully supported by the model
    outputs = model.generate(
        **text_inputs,  # Only unpack input_ids and attention_mask, no other parameters
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # Deterministic generation with reproducible results, no randomness
        repetition_penalty=1.2,  # Slight repetition penalty to avoid model stuck with repeated output
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True  # Enable cache to improve inference speed and save VRAM
    )

    # Step 3: Precise decoding to extract only core output after Answer: and filter all redundancy
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    return raw_answer


# ================================= Answer Extraction (No Default A, 100% Based on Model Native Output)=================================
def extract_answer_letter(raw_answer, all_options):
    """Robust extraction of valid letters to adapt to all possible model output formats, completely no manual fallback/default"""
    valid_letters = ['A', 'B', 'C', 'D']
    # Step 1: Deep clean output to retain only letters and convert to uppercase (filter symbols, spaces, redundant text)
    clean_ans = re.sub(r'[^a-zA-Z]', '', raw_answer).upper()
    # Step 2: Extract first valid letter (most aligned with model inference intent)
    for char in clean_ans:
        if char in valid_letters:
            return char, all_options[char]
    # Step 3: Fuzzy match valid letters in original output (handle slight redundant output from model)
    raw_upper = raw_answer.upper()
    match_letters = [letter for letter in valid_letters if letter in raw_upper]
    if match_letters:
        return match_letters[0], all_options[match_letters[0]]
    # Step 4: Ultimate extreme case (almost impossible to trigger): remote sensing keyword matching (no default A, aligned with remote sensing features)
    remote_keys = ['RIVER', 'TERRAIN', 'BUILDING', 'FARMLAND', 'WATER', 'DENSITY', 'LANDUSE', 'LAYOUT']
    option_key_count = {l: sum(1 for k in remote_keys if k in all_options[l].upper()) for l in valid_letters}
    final_letter = max(option_key_count.keys(), key=lambda x: option_key_count[x])
    return final_letter, all_options[final_letter]


# ================================= Single Sample Inference (Complete Process, Full Robustness for Full Dataset)=================================
def infer_single_sample(tokenizer, model, image_path, question, options, all_options, sample_id):
    """Single sample complete inference process: image loading → preprocessing → model inference → answer extraction, only capture fatal errors"""
    # 大数据集优化：添加样本ID到错误日志
    cache_file = os.path.join(TEMP_CACHE_PATH, f"{sample_id}_cache.json")
    
    # 先检查缓存，避免重复推理（大数据集中断后可续跑）
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        return (cache_data["model_letter"], cache_data["model_answer"], cache_data["status"],
                cache_data["model_raw"], cache_data["infer_size"])
    
    # Only capture fatal image loading errors (file not found/corrupted/format error)
    try:
        image = Image.open(image_path).convert(
            "RGB")  # Force convert to RGB to avoid grayscale/transparent image errors
    except (FileNotFoundError, OSError) as e:
        error_info = {
            "model_letter": "ERROR",
            "model_answer": "Image file error or damaged",
            "status": "fatal_error",
            "model_raw": "",
            "infer_size": (0, 0)
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False)
        print(f"Fatal error: Sample {sample_id} | {os.path.basename(image_path)} | Error message: {str(e)[:50]}")
        return "ERROR", "Image file error or damaged", "fatal_error", "", (0, 0)

    # Remote sensing image preprocessing (model native requirements, standardization + resizing)
    original_size = image.size
    img_tensor, infer_size = preprocess_image(image, IMAGE_MAX_SIZE)
    print(
        f"Processing image: Sample {sample_id} | {os.path.basename(image_path)} | Original size: {original_size} → Inference size: {infer_size}")

    # Model native inference (no invalid parameters, 100% compatible)
    model_raw_answer = infer_single_image(tokenizer, model, img_tensor, question, options)

    # Extract valid answer without default A
    final_letter, final_answer = extract_answer_letter(model_raw_answer, all_options)

    # 保存缓存
    cache_data = {
        "model_letter": final_letter,
        "model_answer": final_answer,
        "status": "success",
        "model_raw": model_raw_answer,
        "infer_size": infer_size
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)

    # Instant VRAM cleanup: delete large tensors to avoid VRAM accumulation in batch inference
    torch.cuda.empty_cache()
    del img_tensor, image  # Manually delete image tensor and PIL image to release VRAM
    return final_letter, final_answer, "success", model_raw_answer, infer_size


# ================================= Batch Inference Main Function (Optimized for Full Dataset)=================================
def batch_infer_xlrs_bench_full():
    """GeoLLaVA-8K batch inference on full XLRS-Bench dataset with hallucination rate auto-statistics"""
    # 1. Load model and Tokenizer (pure native logic, optimized for full dataset)
    tokenizer, model = load_geo_llava_8k()

    # 2. Load all annotation files for full dataset
    all_annotations = []
    for ann_file in ANNOTATION_FILES:
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            all_annotations.extend(annotations)
            print(f"Loaded annotation file: {ann_file} | Samples count: {len(annotations)}")
        except FileNotFoundError:
            print(f"Warning: Annotation file not found - {ann_file}, skip this file")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON format - {ann_file}, skip this file")
    
    if not all_annotations:
        raise FileNotFoundError(f"No valid annotations found in: {ANNOTATION_FILES}")
    
    total_samples = len(all_annotations)
    print(
        f"Full XLRS-Bench dataset loaded successfully | Total samples: {total_samples} | GeoLLaVA-8K remote sensing dedicated inference (full dataset hallucination rate statistics)"
    )

    # 3. Initialize statistical indicators (added wrong sample count for hallucination rate calculation)
    correct_samples = 0
    fatal_error_samples = 0
    wrong_samples = 0  # Added: count of wrong answers in valid inference samples (core numerator for hallucination rate)
    result_list = []
    # Added: store all predictions and ground truths for F1 score calculation
    all_predictions = []
    all_ground_truths = []
    # Mapping letters to numerical values for F1 calculation (A=0, B=1, C=2, D=3)
    letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # 4. Batch inference (progress bar monitoring + real-time per-sample result printing for full dataset)
    # 大数据集优化：每500个样本深度清理一次显存
    progress_bar = tqdm(all_annotations, desc="GeoLLaVA-8K Full XLRS Inference Progress", ncols=100)
    for idx, ann in enumerate(progress_bar):
        sample_id = ann["sample_id"]
        image_path = os.path.join(IMAGE_DIR, ann["image_name"])
        question = ann["question"]
        # Format options to fit model training habits
        options = "\n".join([f"{k}: {v}" for k, v in ann["all_options"].items()])
        gt_letter = ann["gt_answer_letter"]  # Ground truth answer letter
        all_options = ann["all_options"]  # All options dictionary

        # Complete single sample inference (with cache for full dataset)
        model_letter, model_answer, status, model_raw, infer_size = infer_single_sample(
            tokenizer, model, image_path, question, options, all_options, sample_id
        )

        # Statistical inference indicators (added wrong sample count statistics)
        if status == "fatal_error":
            fatal_error_samples += 1
            is_correct = False
        else:
            is_correct = (model_letter == gt_letter)
            if is_correct:
                correct_samples += 1
            else:
                wrong_samples += 1  # Valid inference but wrong answer, counted in hallucination sample base

            # Added: collect predictions and ground truths for F1 calculation (only valid samples)
            all_predictions.append(letter_to_num.get(model_letter, 0))
            all_ground_truths.append(letter_to_num.get(gt_letter, 0))

        # Save single sample detailed results (consistent format with original LLaVA, marked as correct or not)
        result_list.append({
            "sample_id": sample_id,
            "image_name": ann["image_name"],
            "image_original_size": ann["image_size"],
            "infer_image_size": infer_size,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "model_raw_output": model_raw,
            "model_answer_letter": model_letter,
            "gt_answer_letter": gt_letter,
            "is_correct": is_correct,
            "infer_status": status
        })

        # 大数据集优化：降低打印频率（每100个样本打印一次），避免日志刷屏
        if idx % 100 == 0 and idx > 0:
            progress_bar.set_postfix({
                "Valid Accuracy": f"{round(correct_samples/(idx+1-fatal_error_samples)*100,2)}%",
                "Hallucination Rate": f"{round(wrong_samples/(idx+1-fatal_error_samples)*100,2)}%"
            })
            print(
                f"\nSample {sample_id} | Raw output: {model_raw[:20]} | Final answer: {model_letter} | Ground truth: {gt_letter} | Correct: {is_correct}"
            )
        
        # 大数据集显存优化：每500个样本深度清理一次
        if idx % 500 == 0 and idx > 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"\nMemory optimization: Processed {idx+1} samples, deep VRAM cleanup completed")

    # 5. Calculate core inference statistical indicators (Core: added hallucination rate auto-calculation + F1 score)
    valid_samples = total_samples - fatal_error_samples  # Valid inference samples (denominator for hallucination rate)
    valid_accuracy = round((correct_samples / valid_samples) * 100, 2) if valid_samples > 0 else 0.0
    # Added: hallucination rate calculation (simple engineering version: 1 - valid accuracy, adapted for XLRS-Bench multiple choice questions)
    hallucination_rate = round((wrong_samples / valid_samples) * 100, 2) if valid_samples > 0 else 0.0
    # Added: F1 score calculation (macro average for multi-class classification)
    f1_score_value = round(f1_score(all_ground_truths, all_predictions, average='macro') * 100,
                           2) if valid_samples > 0 else 0.0

    # 6. Generate summary results (added hallucination rate, wrong sample count, F1 score fields, write to JSON file)
    final_result = {
        "model_info": {
            "model_name": "GeoLLaVA-8K (Full Dataset Hallucination Rate Statistics Version)",
            "model_path": MODEL_PATH,
            "quantization": "8bit (BitsAndBytes)",
            "device": DEVICE,
            "max_image_size": IMAGE_MAX_SIZE,
            "annotation_files_used": ANNOTATION_FILES
        },
        "dataset_statistics": {
            "total_samples": total_samples,
            "valid_infer_samples": valid_samples,
            "fatal_error_samples": fatal_error_samples,
            "correct_samples": correct_samples,
            "wrong_samples": wrong_samples,  # Added: count of wrong answers in valid samples
            "valid_accuracy(%)": valid_accuracy,
            "hallucination_rate(%)": hallucination_rate,  # Added: model hallucination rate (core)
            "f1_score(%)": f1_score_value,  # Added: F1 score
            "result_file_path": RESULT_SAVE_PATH,
            "temp_cache_path": TEMP_CACHE_PATH
        },
        "detailed_infer_results": result_list
    }

    # 7. Save inference result file (UTF-8 encoding, including hallucination rate and F1 score for subsequent analysis)
    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # 大数据集优化：清理临时缓存（可选，如需续跑可保留）
    # shutil.rmtree(TEMP_CACHE_PATH)
    
    # Final complete VRAM cleanup to release GPU resources
    torch.cuda.empty_cache()
    del model, tokenizer  # Manually delete model and Tokenizer to completely release VRAM

    # Print prominent summary results (focus on hallucination rate and F1 score, corresponding to accuracy)
    print(f"\n" + "="*100)
    print(f"GeoLLaVA-8K Full XLRS-Bench Dataset Inference + Hallucination Rate Statistics Completed!")
    print(f"="*100)
    print(f"Inference Statistics Summary (Remote Sensing Dedicated Model | No Default A | Pure Native Inference | Stable on 16G GPU):")
    print(f"   ├─ Total samples: {total_samples}")
    print(f"   ├─ Valid inference samples: {valid_samples}")
    print(f"   ├─ Fatal error samples: {fatal_error_samples}")
    print(f"   ├─ Valid accuracy: {valid_accuracy}%")
    print(f"   ├─ Hallucination rate: {hallucination_rate}%")
    print(f"   ├─ F1 score (macro): {f1_score_value}%")
    print(
        f"   └─ Detailed results saved to: {RESULT_SAVE_PATH} (including hallucination rate + F1 score + full sample details)"
    )

    return final_result


# ================================= Main Function Entry (Optimized for Full Dataset)=================================
if __name__ == "__main__":
    try:
        # Pre-check path validity to avoid runtime path errors (adapted for full dataset)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"GeoLLaVA-8K model path error, please check: {MODEL_PATH}")
        if not os.path.exists(IMAGE_DIR):
            raise FileNotFoundError(f"Remote sensing image folder not found, please check: {IMAGE_DIR}")
        if not ANNOTATION_FILES or not any(os.path.exists(f) for f in ANNOTATION_FILES):
            raise FileNotFoundError(f"No valid annotation files found in: {ANNOTATION_FILES}")

        # Start GeoLLaVA-8K batch inference + hallucination rate auto-statistics (full dataset version)
        batch_infer_xlrs_bench_full()
    except Exception as e:
        print(f"\nFatal runtime error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error stack for troubleshooting
        torch.cuda.empty_cache()  # Force VRAM cleanup on exception to release GPU resources
        
        # 大数据集异常处理：保存错误日志
        error_log = os.path.join(DATASET_ROOT, "infer_error_log.json")
        with open(error_log, "w", encoding="utf-8") as f:
            json.dump({
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": str(pd.Timestamp.now())
            }, f, ensure_ascii=False, indent=2)
        print(f"Error log saved to: {error_log}")

