import os
import json
import re
import torch
from PIL import Image
import warnings
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Ignore all warnings
warnings.filterwarnings('ignore')

# ================================= Environment Detection =================================
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
    DEVICE = "cuda:0"
    print(f"GPU automatically enabled: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("GPU not detected, will run on CPU (slow speed, please wait patiently)")
# ============================================================================

# Core Configuration (建议BATCH_SIZE=1 for CPU to reduce memory usage)
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_PATH = "/root/autodl-tmp/datasets/mathv_3040"
BATCH_SIZE = 1  # Set to 1 for CPU/GPU 8G, 2 for GPU 16G
MAX_NEW_TOKENS = 512


def load_dataset(dataset_path):
    """Load dataset and filter corrupted images"""
    anno_path = os.path.join(dataset_path, "annotations.json")
    image_dir = os.path.join(dataset_path, "images")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    dataset = []
    for anno in annotations:
        img_name = anno["image_name"]
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            try:
                Image.open(img_path).convert("RGB").close()
                dataset.append({
                    "image_path": img_path,
                    "question": anno["question"],
                    "answer": anno["answer"].strip()
                })
            except:
                continue
    print(f"Dataset loaded successfully: {len(dataset)} valid data entries in total")
    return dataset


def load_llava_model(model_path, device):
    """Load LLaVA model, adapt CPU/GPU precision, bind image_token_id"""
    print(f"Loading LLaVA-v1.5-7b model: {model_path} | Running device: {device}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    # Core: bind image_token_id to ensure <image> is correctly recognized
    processor.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    print(f"Image_token_id bound successfully: {processor.image_token_id}")

    # Match precision by device: float32 for CPU, float16 for GPU
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        trust_remote_code=True
    ).eval()
    model = model.to(device)
    print("LLaVA model loaded successfully, moved to target device and matched corresponding precision")
    return processor, model


def batch_inference(processor, model, dataset, device, batch_size=1):
    """Batch inference: adapt to CPU, disable truncation, retain <image> placeholder"""
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Starting batch inference: {total_batches} batches in total, batch size {batch_size}")

    for i in tqdm(range(total_batches), desc="Inference Progress"):
        batch_data = dataset[i * batch_size: (i + 1) * batch_size]
        if not batch_data:
            continue

        images = []
        valid_prompts = []
        valid_data = []
        for d in batch_data:
            try:
                img = Image.open(d["image_path"]).convert("RGB")
                images.append(img)
                # Fixed prompt format, retain <image> placeholder
                prompt = f"<image> {d['question']} Please provide detailed problem-solving process and final answer."
                valid_prompts.append(prompt)
                valid_data.append(d)
            except Exception as e:
                continue

        if not images or not valid_prompts:
            continue

        # Preprocessing: disable truncation, retain <image>, adapt to CPU precision
        inputs = processor(
            images=images,
            text=valid_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,  # Core: no truncation to avoid <image> loss
            max_length=1024  # Sufficient length to accommodate complete text
        ).to(device)
        # Force ensure all tensors are on the same device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Model generation: remove GPU mixed precision, only retain gradient-free calculation (adapt to CPU)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.1
            )

        # Parse results
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        model_answers = []
        for text, prompt in zip(generated_texts, valid_prompts):
            ans = text.replace(prompt, "").strip()
            model_answers.append(ans if ans else "No valid answer generated")

        for d, pred in zip(valid_data, model_answers):
            results.append({
                "image_path": d["image_path"],
                "question": d["question"],
                "ground_truth": d["answer"],
                "model_answer": pred,
                "exact_correct": False,
                "partial_correct": False,
                "f1_score": 0.0,
                "reasoning_complete_score": 0.0,
                "hallucination_type": None,
                "hallucination_dimension": None
            })

    return results


def extract_numerical_answer(text):
    """Extract numerical answers with units"""
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|cm|m|mm)?')
    matches = pattern.findall(text)
    numerical_ans = [f"{num}{unit}" if unit else num for num, unit in matches]
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """Extract reasoning nodes based on geometric theorems"""
    reasoning_keywords = {
        "Triangle Angle Sum": ["Triangle interior angle sum", "Interior angle sum theorem", "∠A+∠B+∠C=180"],
        "Parallel Line Properties": ["Two lines parallel", "Parallel", "Same-side interior angles supplementary",
                                     "Alternate interior angles equal", "Corresponding angles equal"],
        "Vertical Angles": ["Vertical angles", "Vertical angles equal"],
        "Pythagorean Theorem": ["Pythagorean theorem", "a²+b²=c²"],
        "Similar Triangles": ["Similar", "Similar triangles", "Corresponding sides proportional"],
        "Congruent Triangles": ["Congruent", "Congruent triangles", "SSS", "SAS", "ASA", "AAS"]
    }
    nodes = set()
    text_lower = text.lower()
    for node, keywords in reasoning_keywords.items():
        if any(kw.lower() in text_lower for kw in keywords):
            nodes.add(node)
    return list(nodes)


def calculate_f1_score(gt_ans, pred_ans):
    """Calculate character-level F1 score for mathematical answers"""

    def preprocess(s):
        # Keep only numbers, letters, units and decimal points
        s = re.sub(r'[^\w\d°cm m mm.]', '', s).replace(" ", "")
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
    """Classify hallucination types for mathematical reasoning"""
    if len(pred_nodes) > 0:
        if set(pred_nodes) & set(gt_nodes) and len(pred_num) > 0 and pred_num != gt_num:
            return "Factual Hallucination", "Numerical Calculation Hallucination"
        error_theorem_kw = ["Vertical angles supplementary", "Alternate interior angles supplementary",
                            "Corresponding angles supplementary", "Triangle interior angle sum 360"]
        if any(kw.lower() in pred.lower() for kw in error_theorem_kw) or len(set(pred_nodes) & set(gt_nodes)) == 0:
            return "Factual Hallucination", "Theorem/Concept Hallucination"
    else:
        if len(pred_num) > 0:
            return "Logical Hallucination", "Reasoning Chain Break Hallucination"
        false_cond_kw = ["Known from question", "Given in question", "Known", "According to conditions"]
        if any(kw.lower() in pred.lower() for kw in false_cond_kw) and len(gt_num) > 0 and len(pred_num) == 0:
            return "Logical Hallucination", "Condition Misuse Hallucination"
    return None, None


def calculate_metrics_and_hallucination(results):
    """Calculate evaluation metrics including F1 score and hallucination statistics"""
    total = len(results)
    exact_correct = 0
    partial_correct = 0
    total_f1 = 0.0
    total_reasoning_score = 0.0

    hallucination_stats = {
        "Factual Hallucination-Numerical Calculation Hallucination": 0,
        "Factual Hallucination-Theorem/Concept Hallucination": 0,
        "Logical Hallucination-Reasoning Chain Break Hallucination": 0,
        "Logical Hallucination-Condition Misuse Hallucination": 0,
        "No Hallucination": 0
    }

    for res in results:
        gt = res["ground_truth"]
        pred = res["model_answer"]

        # Calculate F1 score
        f1 = calculate_f1_score(gt, pred)
        res["f1_score"] = f1
        total_f1 += f1

        # Extract numerical answers and reasoning nodes
        gt_num = extract_numerical_answer(gt)
        pred_num = extract_numerical_answer(pred)
        gt_nodes = extract_reasoning_nodes(gt)
        pred_nodes = extract_reasoning_nodes(pred)

        # Judge accuracy
        if set(gt_num) == set(pred_num) and len(gt_num) > 0:
            res["exact_correct"] = True
            exact_correct += 1
            partial_correct += 1
            res["partial_correct"] = True
        elif (len(set(gt_num) & set(pred_num)) > 0) or (len(set(gt_nodes) & set(pred_nodes)) > 0):
            res["partial_correct"] = True
            partial_correct += 1
        else:
            # Classify hallucination
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                hallucination_stats[f"{dim}-{typ}"] += 1
            else:
                hallucination_stats["No Hallucination"] += 1

        # Calculate reasoning completeness
        if len(gt_nodes) == 0:
            res["reasoning_complete_score"] = 1.0 if len(pred_nodes) == 0 else 0.0
        else:
            match_nodes = len(set(gt_nodes) & set(pred_nodes))
            res["reasoning_complete_score"] = match_nodes / len(gt_nodes) if len(gt_nodes) > 0 else 0.0
            total_reasoning_score += res["reasoning_complete_score"]

        # Update no hallucination count
        if res["exact_correct"] or res["partial_correct"]:
            hallucination_stats["No Hallucination"] += 1

    # Calculate final metrics
    metrics = {
        "exact_accuracy": round(exact_correct / total * 100, 2) if total > 0 else 0.0,
        "partial_accuracy": round(partial_correct / total * 100, 2) if total > 0 else 0.0,
        "avg_f1_score": round(total_f1 / total, 4) if total > 0 else 0.0,
        "avg_reasoning_completeness": round(total_reasoning_score / total, 4) if total > 0 else 0.0
    }

    # Normalize hallucination statistics
    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio": round(hallucination_stats[key] / total * 100, 2) if total > 0 else 0.0
        }

    return results, metrics, hallucination_stats


def save_test_results(results, metrics, hallucination_stats):
    """Save test results with full English annotations"""
    save_path = "llava_mathv_test_cpu_fix_result.json"
    final_result = {
        "model_info": {"model_name": "llava-v1.5-7b", "model_path": MODEL_PATH, "device": DEVICE},
        "dataset_info": {"dataset_name": "MATH-Vision", "dataset_path": DATASET_PATH, "total_samples": len(results)},
        "evaluation_metrics": metrics,
        "hallucination_statistics": hallucination_stats,
        "hallucination_definition": {
            "Factual Hallucination": "Has reasoning logic but incorrect numerical calculation/mathematical theorems",
            "Logical Hallucination": "No reasonable reasoning logic, baseless answers/reasoning break",
            "Numerical Calculation Hallucination": "Correct reasoning steps but wrong/fabricated numerical values",
            "Theorem/Concept Hallucination": "Fabricate non-existent theorems or incorrectly use geometric concepts",
            "Reasoning Chain Break Hallucination": "Direct answer without reasoning process",
            "Condition Misuse Hallucination": "Fabricate question conditions or incorrectly use known conditions"
        },
        "metric_definition": {
            "exact_accuracy": "Exact accuracy (perfect answer match)",
            "partial_accuracy": "Partial accuracy (core numerical/reasoning match)",
            "avg_f1_score": "Average character-level F1 score (mathematical answer matching)",
            "avg_reasoning_completeness": "Average reasoning completeness (core reasoning node coverage)"
        },
        "detailed_results": results
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("LLaVA-v1.5-7b Test MATH-Vision - Final Evaluation Results (CPU Adapted Version)")
    print("=" * 80)
    print(f"Running Device: {DEVICE}")
    print(f"Exact Accuracy (Perfect Answer Match): {metrics['exact_accuracy']}%")
    print(f"Partial Accuracy (Core Numerical/Reasoning Match): {metrics['partial_accuracy']}%")
    print(f"Average F1 Score (Character-level): {metrics['avg_f1_score']} (0~1)")
    print(f"Average Reasoning Completeness (Core Node Coverage): {metrics['avg_reasoning_completeness']} (0~1)")
    print("\n" + "=" * 80)
    print("Hallucination Classification Statistics (Count | Ratio)")
    print("=" * 80)
    for key, val in hallucination_stats.items():
        print(f"{key}: {val['count']} entries | {val['ratio']}%")
    print("=" * 80)
    print(f"Result File Saved To: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    try:
        dataset = load_dataset(DATASET_PATH)
        if not dataset:
            raise ValueError("No valid data loaded, please check dataset path")
        processor, model = load_llava_model(MODEL_PATH, DEVICE)
        raw_results = batch_inference(processor, model, dataset, DEVICE, BATCH_SIZE)
        if not raw_results:
            raise ValueError("Inference failed, no valid results generated")
        final_results, metrics, hallucination_stats = calculate_metrics_and_hallucination(raw_results)
        save_test_results(final_results, metrics, hallucination_stats)
        print("\nTest process completed successfully! No matching errors, result file saved.")
    except Exception as e:
        print(f"\nRuntime Error: {str(e)}")
        print(
            "Troubleshooting Suggestions: 1. Confirm model/dataset path is correct 2. Confirm images are not corrupted 3. Keep BATCH_SIZE=1 for CPU runtime")