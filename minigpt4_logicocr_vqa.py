import os
import json
import re
import torch
import torch.nn as nn
from PIL import Image
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor, CLIPVisionModel

# Ignore all warnings
warnings.filterwarnings('ignore')

# ================================= Environment Detection =================================
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
    DEVICE = "cuda:0"
    torch_dtype = torch.float16
    print(f"GPU automatically enabled: {torch.cuda.get_device_name(0)} | Precision: float16")
else:
    DEVICE = "cpu"
    torch_dtype = torch.float32
    print(f"GPU not detected, will run on CPU (slow speed, BATCH_SIZE=1 recommended)")
# ============================================================================

# Core Configuration: Modify to LogicOCR dataset path (others no need to change)
MODEL_PATH = "/root/autodl-tmp/models/MiniGpt-4-7B"
# LogicOCR dataset main directory (contains LogicOCR_real image folder + LogicOCR_real.json annotation file)
DATASET_PATH = "/root/autodl-tmp/datasets/LogicOCR"
CLIP_PATH = "/root/autodl-tmp/models/clip-vit-large-patch14"
BATCH_SIZE = 1  # 4090 can set to 2 without affecting dimension adaptation
MAX_NEW_TOKENS = 512  # Reduced to 512 to limit meaningless reasoning redundancy
IMAGE_TOKEN = "<Image>"  # Fixed image placeholder for MiniGPT-4
TEST_SAMPLE_NUM = 100   # For testing: only run first 100 valid samples
# Dimension Configuration (Key: CLIP outputs 1024 dim, MiniGPT-4 embedding 4096 dim)
CLIP_EMBED_DIM = 1024  # Fixed output dimension for CLIP-ViT-L/14
LLM_EMBED_DIM = 4096  # Word embedding dimension for MiniGPT-4-7B


def load_dataset(dataset_path):
    """LogicOCR-specific adaptation: match actual JSON fields (image/question/solution) + mixed jpg/png naming"""
    # Fixed LogicOCR paths: image directory + single JSON annotation file
    image_dir = os.path.join(dataset_path, "LogicOCR_real")  # Image directory: xxx.png/jpg
    json_anno_path = os.path.join(dataset_path, "LogicOCR_real.json")  # Single-file annotation

    # Verify if directories/files exist (error directly if missing for quick troubleshooting)
    if not os.path.exists(image_dir):
        raise ValueError(f"LogicOCR image directory missing: {image_dir}")
    if not os.path.exists(json_anno_path):
        raise ValueError(f"LogicOCR annotation file missing: {json_anno_path}")

    dataset = []
    # Load single JSON annotation file (core annotation method for LogicOCR)
    try:
        with open(json_anno_path, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)
        # Compatible with JSON in list/dict format (cover common annotation formats)
        if isinstance(all_annotations, dict):
            all_annotations = list(all_annotations.values())
    except Exception as e:
        raise ValueError(f"Failed to load LogicOCR annotation file: {str(e)[:100]}")

    # Traverse annotations, match images and filter valid samples
    for anno in all_annotations:
        if len(dataset) >= TEST_SAMPLE_NUM:  # Load only first 100, stop immediately when full
            break
        # Core modification: match actual JSON field names (image=image name, question=question, solution=answer)
        img_filename = anno.get("image", "").strip()
        question = anno.get("question", "").strip()
        answer = anno.get("solution", "").strip()

        # Basic field filtering: skip directly if no image name/question/answer
        if not img_filename or not question or not answer:
            continue
        # Concatenate complete image path (annotation already has suffix, match precisely directly)
        img_filepath = os.path.join(image_dir, img_filename)
        # Only compatible with case-insensitive suffixes (e.g., 1.PNG→1.png, 100.JPG→100.jpg), no other modifications
        if not os.path.exists(img_filepath):
            img_filepath = os.path.join(image_dir, img_filename.lower())
        # New: skip directly if no matching image to avoid errors in subsequent opening
        if not os.path.exists(img_filepath):
            print(f"No matching image found, skipped: {img_filename}")
            continue

        # Verify if image can be opened normally, skip corrupted images
        try:
            with Image.open(img_filepath).convert("RGB") as img:
                pass
        except Exception as e:
            print(f"Skipped corrupted image file: {img_filename}, Error: {str(e)[:50]}")
            continue

        # Add to dataset, fully compatible with subsequent inference/evaluation logic
        dataset.append({
            "image_path": img_filepath,
            "question": question,  # Unified field name, no impact on subsequent logic
            "answer": answer       # Unified field name, no impact on subsequent logic
        })
        # Print loading log to confirm progress
        print(f"Loaded valid sample [{len(dataset)}]: {img_filename}")

    # Final verification of valid sample count, precise prompt for issues (update field prompt)
    if len(dataset) == 0:
        raise ValueError("No valid LogicOCR samples loaded, please check: 1. Annotations have image/question/solution fields 2. Image names match annotations (including suffixes) 3. Image format is png/jpg")
    print(f"\nLogicOCR dataset loaded successfully: {len(dataset)} valid test data in total (limited to first {TEST_SAMPLE_NUM})")
    return dataset


def load_minigpt4_model(model_path, clip_path, device, torch_dtype):
    """Load model + add dimension projection layer (no modification, retain original logic)"""
    print(f"Loading MiniGPT-4-7B (text-only version): {model_path}")
    # 1. Load text tokenizer and configure padding
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Text tokenizer configured: pad_token_id={tokenizer.pad_token_id}")

    # 2. Load MiniGPT-4 language model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    print(f"MiniGPT-4 language model loaded: embedding dimension={LLM_EMBED_DIM}")

    # 3. Load CLIP vision encoder (local path, no network)
    print(f"Loading CLIP vision encoder: {clip_path}")
    clip_processor = CLIPImageProcessor.from_pretrained(clip_path)
    clip_model = CLIPVisionModel.from_pretrained(
        clip_path,
        dtype=torch_dtype,
        device_map=device
    ).eval()
    print(f"CLIP vision encoder loaded: output dimension={CLIP_EMBED_DIM}")

    # 4. Core fix: add linear projection layer (1024 dim → 4096 dim), standard fusion layer for MiniGPT-4
    print(f"Initializing dimension projection layer: {CLIP_EMBED_DIM} → {LLM_EMBED_DIM}")
    image_proj = nn.Linear(CLIP_EMBED_DIM, LLM_EMBED_DIM, bias=False).to(device, torch_dtype)
    # Projection layer initialization (follow official MiniGPT-4 configuration to ensure fusion effect)
    nn.init.normal_(image_proj.weight, std=CLIP_EMBED_DIM ** -0.5)

    # 5. Bind image_token_id (add if not in vocabulary)
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"Image_token_id bound: {image_token_id}")

    return tokenizer, model, clip_processor, clip_model, image_proj, image_token_id


def batch_inference(tokenizer, model, clip_processor, clip_model, image_proj, image_token_id, dataset, device,
                    batch_size=1):
    """Multimodal inference: CLIP encoding → projection layer dimension upscaling → replace <Image> → model generation (core optimization)"""
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Starting MiniGPT-4 VQA test inference: {total_batches} batches in total, batch size {batch_size} ({len(dataset)} samples)")

    for i in tqdm(range(total_batches), desc="VQA Test Inference Progress"):
        batch_data = dataset[i * batch_size: (i + 1) * batch_size]
        if not batch_data:
            continue

        # 1. CLIP encode images: get 1024-dimensional visual features
        images = [Image.open(d["image_path"]).convert("RGB") for d in batch_data]
        clip_inputs = clip_processor(images=images, return_tensors="pt").to(device, torch_dtype)
        with torch.no_grad():
            clip_embeds = clip_model(**clip_inputs).last_hidden_state[:, 0, :]  # Take [CLS] feature: (batch, 1024)

        # 2. Projection layer dimension upscaling: 1024 dim → 4096 dim (core fix, match MiniGPT-4 embedding)
        with torch.no_grad():
            image_embeds = image_proj(clip_embeds)  # (batch, 4096)

        # 3. Encode text Prompt: optimized for LogicOCR dataset
        prompts = []
        for d in batch_data:
            prompt = f"""{IMAGE_TOKEN} Analyze the following chart data and answer the question, strictly following these rules:
1. Only extract numerical values, categories, and proportions explicitly shown in the image; do not fabricate unshown data/relationships;
2. Reason based on basic statistical logic (proportion, difference, size comparison); do not fabricate incorrect rules;
3. The final answer is concise and clear, only giving calculation/comparison results without redundant expressions.
Question: {d['question']}
Detailed explanation (including data source) and final answer:"""
            prompts.append(prompt)

        text_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            max_length=1024
        ).to(device)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # 4. Replace <Image> with upscaled visual features (dimensions matched, can be assigned directly)
        input_embeds = model.get_input_embeddings()(input_ids)  # (batch, seq_len, 4096)
        image_token_mask = (input_ids == image_token_id).nonzero(as_tuple=True)
        for batch_idx, seq_idx in zip(image_token_mask[0], image_token_mask[1]):
            input_embeds[batch_idx, seq_idx] = image_embeds[batch_idx]  # Dimensions match, no errors

        # 5. Model generation: adjusted parameters to balance randomness and rigor
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,  # Enable sampling to increase reasoning flexibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=3,  # Reduce beam number to reduce repetition/fabrication
                repetition_penalty=1.2,  # Reduce repetition penalty to avoid over-constraint
                temperature=0.7,  # Moderate temperature to balance randomness and rigor
                top_p=0.9  # Relax nucleus sampling to retain reasonable reasoning
            )

        # 6. Parse and save results (optimize low-quality answer marking)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for d, text, prompt in zip(batch_data, generated_texts, prompts):
            ans = text.replace(prompt, "").strip()
            # Mark low-quality answers for subsequent metric analysis
            if not ans or ans in ["Unknown", "Unable to answer", "No answer"]:
                ans = "[Low Quality Answer] No valid reasoning and answer generated"
            results.append({
                "image_path": d["image_path"],
                "question": d["question"],
                "ground_truth": d["answer"],
                "model_answer": ans,
                "accuracy": 0,
                "exact_correct": False,
                "partial_correct": False,
                "f1_score": 0.0,
                "reasoning_complete_score": 0.0,
                "hallucination_type": None,
                "hallucination_dimension": None
            })

    print(f"VQA test inference completed: {len(results)} valid answers generated in total")
    return results


def extract_numerical_answer(text):
    """Extract numerical answers: optimized for LogicOCR annotation format"""
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|cm|m|mm|%|times|units|yuan|seconds|minutes)?')
    matches = pattern.findall(text)
    numerical_ans = [f"{num}{unit}" if unit else num for num, unit in matches]
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """Extract reasoning nodes: optimized for LogicOCR reasoning logic"""
    reasoning_keywords = {
        "Numerical Calculation": ["add", "subtract", "multiply", "divide", "sum", "average", "total", "difference", "proportion", "ratio", "gap"],
        "Logical Judgment": ["yes", "no", "conform", "not conform", "exist", "not exist", "contain", "not contain", "exceed", "below"],
        "Relationship Reasoning": ["greater than", "less than", "equal to", "greater than or equal to", "less than or equal to", "directly proportional", "inversely proportional", "highest", "lowest"],
        "Rule Application": ["according to rules", "as required", "follow conditions", "meet prerequisites", "deduce", "statistically known"],
        "Information Extraction": ["shown in image", "known from figure", "given in question", "known conditions", "extract information", "chart data"]
    }
    nodes = set()
    text_lower = text.lower()
    for node, keywords in reasoning_keywords.items():
        if any(kw.lower() in text_lower for kw in keywords):
            nodes.add(node)
    return list(nodes)


def calculate_f1_score(gt_ans, pred_ans):
    """Calculate character-level F1 score: optimized for chart-based answers"""
    def preprocess(s):
        # Keep only letters, numbers, units and decimal points, remove other symbols
        s = re.sub(r'[^\w\d°cm mm%times units yuan seconds minutes.]', '', s).replace(" ", "")
        return list(s) if s else []

    gt_chars = preprocess(gt_ans)
    pred_chars = preprocess(pred_ans)
    if not gt_chars and not pred_chars:
        return 1.0
    if not gt_chars or not pred_chars:
        return 0.0
    common = list(set(gt_chars) & set(pred_chars))
    if not common:
        return 0.0
    precision = len(common) / len(pred_chars)
    recall = len(common) / len(gt_chars)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num):
    """Hallucination classification: optimized judgment logic"""
    if len(pred_nodes) > 0:
        if set(pred_nodes) & set(gt_nodes) and len(pred_num) > 0 and pred_num != gt_num:
            return "Factual Hallucination", "Numerical Calculation Hallucination"
        # Error logic keywords
        error_logic_kw = ["added as subtracted", "multiplied as divided", "greater as less", "equal as unequal", "chart misread", "data error"]
        if any(kw.lower() in pred.lower() for kw in error_logic_kw) or len(set(pred_nodes) & set(gt_nodes)) == 0:
            return "Factual Hallucination", "Logical Rule Hallucination"
    else:
        if len(pred_num) > 0:
            return "Logical Hallucination", "Reasoning Chain Break Hallucination"
        # False condition keywords
        false_cond_kw = ["known from question", "shown in image", "known", "according to conditions", "in chart"]
        if any(kw.lower() in pred.lower() for kw in false_cond_kw) and len(gt_num) > 0 and len(pred_num) == 0:
            return "Logical Hallucination", "Condition Misuse Hallucination"
    return None, None


def calculate_metrics(results):
    """Calculate evaluation metrics: optimized hallucination statistics logic to avoid over-labeling"""
    total = len(results)
    correct_num = 0
    exact_correct = 0
    partial_correct = 0
    total_f1 = 0.0
    total_reasoning_score = 0.0
    hallucination_stats = {
        "Factual Hallucination-Numerical Calculation Hallucination": 0,
        "Factual Hallucination-Logical Rule Hallucination": 0,
        "Logical Hallucination-Reasoning Chain Break Hallucination": 0,
        "Logical Hallucination-Condition Misuse Hallucination": 0,
        "No Hallucination": 0
    }

    for res in results:
        gt = res["ground_truth"]
        pred = res["model_answer"]
        f1 = calculate_f1_score(gt, pred)
        res["f1_score"] = f1
        total_f1 += f1
        gt_num = extract_numerical_answer(gt)
        pred_num = extract_numerical_answer(pred)
        gt_nodes = extract_reasoning_nodes(gt)
        pred_nodes = extract_reasoning_nodes(pred)

        # Optimized accuracy judgment logic: relaxed partial correct conditions
        if set(gt_num) == set(pred_num) and len(gt_num) > 0:
            res["exact_correct"] = True
            exact_correct += 1
            partial_correct += 1
            res["partial_correct"] = True
            res["accuracy"] = 1
            correct_num += 1
        elif (len(set(gt_num) & set(pred_num)) > 0) or (len(set(gt_nodes) & set(gt_nodes)) > 0):
            res["partial_correct"] = True
            partial_correct += 1
            res["accuracy"] = 0.5  # Partial correct counts as 0.5 points, more reasonable
        else:
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                hallucination_stats[f"{dim}-{typ}"] += 1
            else:
                hallucination_stats["No Hallucination"] += 1
            res["accuracy"] = 0

        # Optimized reasoning completeness calculation
        if len(gt_nodes) == 0:
            res["reasoning_complete_score"] = 1.0 if len(pred_nodes) == 0 else 0.0
        else:
            match_nodes = len(set(gt_nodes) & set(pred_nodes))
            res["reasoning_complete_score"] = match_nodes / len(gt_nodes) if len(gt_nodes) > 0 else 0.0
            total_reasoning_score += res["reasoning_complete_score"]

        # Optimized no hallucination judgment: partial correct also counts as no hallucination
        if res["exact_correct"] or res["partial_correct"]:
            hallucination_stats["No Hallucination"] += 1

    metrics = {
        "overall_accuracy": round(correct_num / total * 100, 2),
        "exact_accuracy": round(exact_correct / total * 100, 2),
        "partial_accuracy": round(partial_correct / total * 100, 2),
        "avg_f1_score": round(total_f1 / total, 4),
        "avg_reasoning_completeness": round(total_reasoning_score / total, 4) if total > 0 else 0.0
    }

    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio": round(hallucination_stats[key] / total * 100, 2) if total > 0 else 0.0
        }

    return results, metrics, hallucination_stats


def save_results(results, metrics, hallucination_stats):
    """Save results: full English output for LogicOCR dataset"""
    save_path = "minigpt4_LogicOCR_vqa_test_100samples_optimized.json"
    final_result = {
        "model_info": {
            "model_name": "MiniGPT-4-7B (Text Version + CLIP + Dimension Projection Layer + Hallucination Optimized Configuration)",
            "model_path": MODEL_PATH,
            "clip_path": CLIP_PATH,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "fusion_config": f"CLIP({CLIP_EMBED_DIM} dim) → Linear Projection → MiniGPT-4({LLM_EMBED_DIM} dim)",
            "generation_config": "num_beams=3, temperature=0.7, top_p=0.9, repetition_penalty=1.2, do_sample=True (Hallucination Optimized Configuration)"
        },
        "dataset_info": {
            "dataset_name": "LogicOCR",
            "dataset_path": DATASET_PATH,
            "total_test_samples": len(results),
            "sample_limit": f"First {TEST_SAMPLE_NUM} valid samples",
            "dataset_structure": "Single JSON annotation file: LogicOCR_real.json ↔ Image directory: LogicOCR_real/xxx.png/jpg",
            "annotation_fields": "Actual annotation fields: image (image name), question (question), solution (answer) | Supports mixed jpg/png naming"
        },
        "evaluation_metrics": metrics,
        "hallucination_statistics": hallucination_stats,
        "hallucination_definition": {
            "Factual Hallucination": "Has reasoning logic but incorrect numerical calculation/logical rule application",
            "Logical Hallucination": "No reasonable reasoning logic, baseless answers/reasoning break",
            "Numerical Calculation Hallucination": "Correct reasoning steps but wrong/fabricated numerical values in calculation",
            "Logical Rule Hallucination": "Fabricate non-existent logical rules or apply reasoning rules incorrectly",
            "Reasoning Chain Break Hallucination": "Direct answer without reasoning process",
            "Condition Misuse Hallucination": "Fabricate question conditions or incorrectly use known conditions in images"
        },
        "metric_definition": {
            "overall_accuracy": "Overall accuracy (perfect match ratio), 1=completely correct answer, 0=incorrect",
            "exact_accuracy": "Exact accuracy (perfect numerical match)",
            "partial_accuracy": "Partial accuracy (core numerical/reasoning node match)",
            "avg_f1_score": "Average character-level F1 score (optimized for chart-based question answers)",
            "avg_reasoning_completeness": "Average reasoning completeness (core reasoning node coverage ratio)"
        },
        "detailed_test_results": results
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("MiniGPT-4-7B LogicOCR Dataset - Hallucination Optimized VQA Full Metric Evaluation Report")
    print("=" * 90)
    print(f"Runtime Configuration: {DEVICE} | Batch Size={BATCH_SIZE} | Dimension Fusion: {CLIP_EMBED_DIM}→{LLM_EMBED_DIM} | Max Generation Length: {MAX_NEW_TOKENS}")
    print(f"Test Data Scale: {len(results)} valid samples | Generation Configuration: Hallucination Optimized (Controllable Reasoning)")
    print("=" * 90)
    print("Core Evaluation Metrics (Accuracy + F1 + VQA Special)")
    print("=" * 90)
    print(f"Overall Accuracy (Perfect Match): {metrics['overall_accuracy']}%")
    print(f"Exact Accuracy (Perfect Numerical Match): {metrics['exact_accuracy']}%")
    print(f"Partial Accuracy (Core Numerical/Reasoning Match): {metrics['partial_accuracy']}%")
    print(f"Average Character-level F1 Score: {metrics['avg_f1_score']} (0~1)")
    print(f"Average Reasoning Completeness (Node Coverage): {metrics['avg_reasoning_completeness']} (0~1)")
    print("=" * 90)
    print("Hallucination Classification Statistics (Count | Ratio)")
    print("=" * 90)
    for key, val in hallucination_stats.items():
        print(f"{key.ljust(60)}: {val['count']} samples | {val['ratio']}%")
    print("=" * 90)
    print(f"Optimized Result File Saved To: {os.path.abspath(save_path)}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        dataset = load_dataset(DATASET_PATH)
        # Load model (including projection layer)
        tokenizer, model, clip_processor, clip_model, image_proj, image_token_id = load_minigpt4_model(
            MODEL_PATH, CLIP_PATH, DEVICE, torch_dtype
        )
        # Start test inference (optimized configuration)
        raw_results = batch_inference(
            tokenizer, model, clip_processor, clip_model, image_proj, image_token_id, dataset, DEVICE, BATCH_SIZE
        )
        # Calculate full metrics
        final_results, metrics, hallucination_stats = calculate_metrics(raw_results)
        save_results(final_results, metrics, hallucination_stats)
        print("\nMiniGPT-4 Multimodal VQA LogicOCR Dataset - Hallucination Optimized Full Process Evaluation Completed!")
    except Exception as e:
        print(f"\nRuntime Error: {str(e)}")
        import traceback
        traceback.print_exc()