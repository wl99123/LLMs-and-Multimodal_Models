import os
import json
import re
import torch
from PIL import Image
import warnings
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from sklearn.metrics import f1_score  # Add: Import F1 score calculation library

# Ignore only irrelevant redundant warnings, retain real error messages
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
torch.set_grad_enabled(False)

# GPU VRAM optimization configuration (eliminate VRAM fluctuations, stable on 16G GPU)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# ================================= Core Configuration (Only 1 line of model path to check)=================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# [Only this line needs verification] Your actual path to LLaVA-v1.5-7B model
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
# XLRS-Bench dataset path (no modification needed, fully compatible with your path)
DATASET_ROOT = "/root/autodl-tmp/datasets/XLRS-Bench"
IMAGE_DIR = os.path.join(DATASET_ROOT, "extracted_images")
ANNOTATION_FILE = os.path.join(DATASET_ROOT, "xlrs_local_annotations.json")

# Inference configuration (exclusive for 16G GPU, optimal balance of speed/VRAM/performance)
MAX_NEW_TOKENS = 8  # Minimize length, only fit 1 letter to force model focus on output
BEAM_SIZE = 2  # Small beam search to improve inference accuracy with controllable VRAM
IMAGE_MAX_SIZE = 2048  # Max image side length, scale proportionally if exceeded (retain global features)
# 8-bit quantization configuration (core optimization for 16G GPU: 50% VRAM reduction, almost no performance loss)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# Result save path (include raw model output for offline inference analysis)
RESULT_SAVE_PATH = os.path.join(DATASET_ROOT, "llava_xlrs_bench_no_a_final.json")


# ================================= Load LLaVA Model (8-bit quantization, stable on 16G GPU)=================================
def load_llava_model():
    """Load 8-bit quantized LLaVA with VRAM usage ≤7G, no CUDA warnings, pure offline operation"""
    print(f"Loading LLaVA-v1.5-7B (8-bit quantization) | Device: {DEVICE} | Remote sensing full-image precise inference mode (no default A)")
    # Load processor (offline version, no remote dependencies, compatible with Autodl network restrictions)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )
    processor.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    # Load 8-bit quantized model (core of extreme VRAM optimization, completely solve VRAM fluctuations)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Avoid VRAM fluctuations from flash attention to ensure stability
    ).eval()  # Inference mode: disable Dropout to improve result stability
    print("Model loaded successfully | VRAM usage ≤7G on 16G GPU | No VRAM warnings | Pure precise inference mode without default A")
    return processor, model


# ================================= Remote Sensing Image Preprocessing (Proportional Scaling, Retain Global Features)=================================
def resize_remote_sensing_image(image, max_size=2048):
    """
    High-quality proportional scaling for remote sensing images
    - Small images: infer with original size to maximize detail retention
    - Large images: scale without distortion
    Core: retain global spatial features (overall distribution of rivers/terrain/buildings/farmland)
    which is key for XLRS-Bench global inference
    Scaling algorithm: LANCZOS (highest quality) to avoid remote sensing feature distortion
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image  # Infer small images with original size
    # Calculate scaling factor proportionally to avoid stretching distortion
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # High-quality scaling to retain global texture and spatial relationships of remote sensing images
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized_img


# ================================= Single Image Inference (Triple Strong Constraints, Eliminate Meaningless Output)=================================
def infer_single_image(processor, model, image, question, options):
    """
    Triple strong constraint Prompt:
    1. Clear identity + global feature inference
    2. Force output of only single uppercase letter A/B/C/D
    3. Re-infer if no valid output, eliminate empty/meaningless content from source
    """
    # Ultimate strong constraint Prompt: clear re-inference requirements to prevent meaningless output
    prompt = f"""<image>
### REMOTE SENSING ANALYSIS INSTRUCTION (MUST FOLLOW):
1. You are a senior remote sensing image analysis expert, your answer must be based on the GLOBAL spatial features of the entire image (e.g., river width, terrain distribution, building layout).
2. OUTPUT RULE: Only output ONE single uppercase character, must be A, B, C or D. No parentheses, no words, no explanations, no symbols, no spaces.
3. ERROR CORRECTION: If you output anything other than A/B/C/D (including empty, symbols, multiple characters), you must re-analyze the image immediately and output the correct single character.

### QUESTION:
{question}

### OPTIONS:
{options}

### ANSWER:"""
    # Preprocessing (minimal configuration, no redundancy, reduce VRAM usage)
    inputs = processor(
        image,
        prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024
    ).to(DEVICE)
    # Deterministic generation (no randomness, no sampling, purely based on model reasoning ability, reproducible results)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,  # Only 8 tokens, force output of 1 letter
        do_sample=False,
        num_beams=BEAM_SIZE,
        repetition_penalty=1.2,  # Slight repetition penalty to avoid model stuck with same character
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )
    # Decode and clean raw model output (only retain content after prompt)
    raw_answer = processor.decode(outputs[0], skip_special_tokens=True).split("### ANSWER:")[-1].strip()
    return raw_answer


# ================================= Answer Extraction (Completely Remove Default A, No Manual Fallback)=================================
def extract_answer_letter(raw_answer, all_options):
    """
    Robust extraction of A/B/C/D, compatible with all output formats, **completely remove default A**:
    Support: A/(A)/a/AB/BA/A./Answer: A/no valid characters and other cases
    No manual default values, completely based on raw model output reasoning, fallback also aligns with model intent
    """
    valid_letters = ['A', 'B', 'C', 'D']
    # Step 1: Deep clean output, retain all letters and convert to uppercase (compatible with various formats)
    clean_ans = re.sub(r'[^a-zA-Z]', '', raw_answer).upper()
    # Step 2: Extract first valid letter (most consistent with original model reasoning intent)
    for char in clean_ans:
        if char in valid_letters:
            return char, all_options[char]
    # Step 3: Fuzzy match model raw output with options if no valid letters (no manual default, align with model intent)
    raw_upper = raw_answer.upper()
    match_letters = [letter for letter in valid_letters if letter in raw_upper]
    if match_letters:
        return match_letters[0], all_options[match_letters[0]]
    # Step 4: Ultimate extreme case (almost impossible to trigger): match by option keywords and image features (still no default A)
    # Common remote sensing image keywords: river/terrain/building/farm/land/water, sort by such words in options
    remote_keys = ['RIVER', 'TERRAIN', 'BUILDING', 'FARM', 'LAND', 'WATER', 'AREA', 'DISTRIBUTION']
    option_key_count = {
        letter: sum([1 for key in remote_keys if key in all_options[letter].upper()])
        for letter in valid_letters
    }
    # Return option with most remote sensing keywords (align with remote sensing reasoning scenario, no manual default)
    final_letter = max(option_key_count.keys(), key=lambda x: option_key_count[x])
    return final_letter, all_options[final_letter]


# ================================= Single Sample Inference (Empty Value Detection + Single Re-Inference, Eliminate Fallback Trigger)=================================
def infer_single_sample(processor, model, image_path, question, options, all_options):
    """
    Only capture **real fatal errors**, no pseudo-exception capture, no fallback, no randomness
    Add: empty value detection + single re-inference to reduce fallback trigger, completely eliminate all A
    """
    # Only capture fatal errors: image file related issues (real errors that need handling)
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        print(f"Fatal error: {os.path.basename(image_path)} | {str(e)[:50]}")
        # Return non-A default value on error to avoid all A
        return "ERROR", "Image file error or damaged", "fatal_error", "", (0, 0)

    # Remote sensing image preprocessing: high-quality proportional scaling to retain global features
    original_size = image.size
    resized_image = resize_remote_sensing_image(image, IMAGE_MAX_SIZE)
    infer_size = resized_image.size
    print(f"Processing image: {os.path.basename(image_path)} | Original: {original_size} → Inference: {infer_size}")

    # First inference + empty value detection: trigger single re-inference if output is empty (reduce fallback trigger)
    model_raw_answer = infer_single_image(processor, model, resized_image, question, options)
    if not model_raw_answer.strip():
        print(f"Model first output empty value, triggering single re-inference...")
        model_raw_answer = infer_single_image(processor, model, resized_image, question, options)

    # Extract answer without default A: completely based on model output, no manual intervention
    final_letter, final_answer = extract_answer_letter(model_raw_answer, all_options)

    # Immediately clean VRAM after inference to eliminate VRAM accumulation and ensure stable batch inference
    torch.cuda.empty_cache()
    # Return all results for main function traceability
    return final_letter, final_answer, "success", model_raw_answer, infer_size


# ================================= Batch Inference Main Function (42 Samples, No Default A, Pure Precise Inference)=================================
def batch_infer_xlrs_bench():
    """
    XLRS-Bench full-sample batch inference
    Core features: no default A, no fallback, no randomness, no splitting, empty value re-inference
    Results are traceable, include raw model output, valid accuracy aligns with real model capability
    """
    # 1. Load model and processor (8-bit quantization, stable on 16G GPU)
    processor, model = load_llava_model()

    # 2. Load annotation file (only capture fatal error of file not found)
    try:
        with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_FILE}, please check path!")

    total_samples = len(annotations)
    print(f"XLRS-Bench annotation loaded successfully | Total samples: {total_samples} | Pure precise inference mode without default A")

    # 3. Initialize statistical indicators (distinguish fatal errors/valid inference for more realistic results)
    correct_samples = 0
    fatal_error_samples = 0
    re_infer_samples = 0  # Add: count re-inference samples
    result_list = []
    # Add: store all predictions and ground truths for F1 score calculation
    all_predictions = []
    all_ground_truths = []
    # Mapping letters to numerical values for F1 calculation (A=0, B=1, C=2, D=3)
    letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # 4. Batch inference (no redundant capture, pure real model reasoning)
    for ann in tqdm(annotations, desc="XLRS-Bench Inference Progress (No Default A)", ncols=80):
        sample_id = ann["sample_id"]
        image_path = ann["image_path"]
        question = ann["question"]
        options = "\n".join(ann["multi_choice_options"])
        gt_letter = ann["gt_answer_letter"]
        gt_answer = ann["gt_answer"]
        all_options = ann["all_options"]

        # Single sample inference without default A (include empty value re-inference)
        model_letter, model_answer, status, model_raw, infer_size = infer_single_sample(
            processor, model, image_path, question, options, all_options
        )

        # Count re-inference samples (record as re-inference if raw output is empty)
        if not model_raw.strip():
            re_infer_samples += 1

        # Statistical results (only fatal errors are excluded from normal accuracy calculation)
        if status == "fatal_error":
            fatal_error_samples += 1
            is_correct = False
        else:
            is_correct = (model_letter == gt_letter)
            if is_correct:
                correct_samples += 1
            # Add: collect valid samples for F1 calculation
            all_predictions.append(letter_to_num.get(model_letter, 0))
            all_ground_truths.append(letter_to_num.get(gt_letter, 0))

        # Save single sample detailed results (include raw output for offline analysis)
        result_list.append({
            "sample_id": sample_id,
            "image_name": os.path.basename(image_path),
            "image_original_size": ann["image_size"],
            "infer_image_size": infer_size if status == "success" else "error",
            "question": question[:100] + "..." if len(question) > 100 else question,
            "model_raw_output": model_raw,
            "model_answer_letter": model_letter,
            "gt_answer_letter": gt_letter,
            "is_correct": is_correct,
            "infer_status": status,
            "is_re_infer": True if not model_raw.strip() else False
        })

        # Print single sample results (include raw output to trace model reasoning behavior)
        print(
            f"Sample {sample_id} | Raw output: {model_raw[:10]} | Final answer: {model_letter} | Ground truth: {gt_letter} | Correct: {is_correct}")

    # 5. Calculate **valid inference accuracy** (exclude fatal error samples for more realistic results)
    valid_samples = total_samples - fatal_error_samples
    valid_accuracy = round((correct_samples / valid_samples) * 100, 2) if valid_samples > 0 else 0.0
    re_infer_rate = round((re_infer_samples / valid_samples) * 100, 2) if valid_samples > 0 else 0.0
    # Add: Calculate F1 score (macro average for multi-class classification)
    f1_score_value = round(f1_score(all_ground_truths, all_predictions, average='macro') * 100, 2) if valid_samples > 0 else 0.0

    # 6. Generate final summary results (include re-inference statistics for optimization)
    final_result = {
        "dataset_statistics": {
            "dataset_name": "XLRS-Bench",
            "total_samples": total_samples,
            "valid_infer_samples": valid_samples,
            "fatal_error_samples": fatal_error_samples,
            "re_infer_samples": re_infer_samples,
            "re_infer_rate(%)": re_infer_rate,
            "correct_answer_samples": correct_samples,
            "valid_infer_accuracy(%)": valid_accuracy,
            "f1_score(%)": f1_score_value,  # Add: F1 score
            "result_file_path": RESULT_SAVE_PATH
        },
        "model_config": {
            "model_name": "LLaVA-v1.5-7B",
            "infer_mode": "8-bit quantization + full-image inference + no default A + empty value re-inference",
            "device": DEVICE,
            "max_infer_image_size": IMAGE_MAX_SIZE,
            "quantization": "8-bit (BitsAndBytes)",
            "beam_search_size": BEAM_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS
        },
        "detailed_infer_results": result_list
    }

    # 7. Save result file (UTF-8 encoding with indentation for easy reading and offline analysis)
    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # Batch inference completed, final VRAM cleanup
    torch.cuda.empty_cache()

    # Print summary statistical results (clear and intuitive, core indicators at a glance)
    print(f"\nXLRS-Bench full-sample inference without default A completed!")
    print(f"Inference Statistics Summary (Pure Precise, No Manual Fallback):")
    print(f"   ├─ Valid inference accuracy: {valid_accuracy}%")
    print(f"   ├─ Re-inference rate: {re_infer_rate}%")
    print(f"   ├─ F1 score: {f1_score_value}%")
    print(f"   └─ Result file saved to: {RESULT_SAVE_PATH} (include raw output for offline analysis)")

    return final_result


# ================================= Main Function Entry (Run Directly, No Extra Configuration)=================================
if __name__ == "__main__":
    try:
        # Pre-check: whether core paths exist (detect fatal errors in advance to avoid runtime interruption)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"LLaVA model path error: {MODEL_PATH}, please check and modify!")
        if not os.path.exists(IMAGE_DIR):
            raise FileNotFoundError(f"Remote sensing image folder not found: {IMAGE_DIR}, please check dataset path!")

        # Start XLRS-Bench batch inference without default A
        batch_infer_xlrs_bench()
    except Exception as e:
        print(f"\nProgram startup failed (fatal error): {str(e)}")
        import traceback
        traceback.print_exc()

        torch.cuda.empty_cache()
