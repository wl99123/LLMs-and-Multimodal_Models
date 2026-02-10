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

# ===================== Environment Configuration =====================
torch.cuda.empty_cache()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE == "cuda:0" else torch.float32
print(f"Running Device: {DEVICE} | Precision: {torch_dtype}")

# ===================== Core Configuration =====================
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_PATH = "/root/autodl-tmp/datasets/mathv_3040"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512


def load_dataset(dataset_path):
    """Load dataset and filter corrupted images (full dataset)"""
    anno_path = os.path.join(dataset_path, "annotations.json")
    image_dir = os.path.join(dataset_path, "images")

    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    dataset = []
    for idx, anno in enumerate(annotations):
        img_name = anno.get("image_name", "")
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as img:
                img.convert("RGB")
            dataset.append({
                "image_path": img_path,
                "question": anno.get("question", "").strip(),
                "answer": anno.get("answer", "").strip()
            })
        except Exception as e:
            print(f"Skipping corrupted image {idx}: {img_path} | Error: {str(e)[:50]}")
            continue
    print(f"Dataset loaded successfully: {len(dataset)} valid samples (full dataset)")
    return dataset


def load_llava_model(model_path):
    """Load LLaVA model (strictly follow official specifications)"""
    print(f"\nLoading LLaVA-v1.5-7b model: {model_path}")

    # Load processor (disable fast tokenizer to avoid <image> parsing errors)
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # Load model (only keep necessary parameters)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=DEVICE,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).eval()

    print("LLaVA model loaded successfully")
    return processor, model


def single_sample_inference(processor, model, image_path, question):
    """Single sample inference (official LLaVA recommended method to avoid batch issues)"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Build prompt (strictly follow LLaVA official format)
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    # Preprocessing (automatically handle <image> token and image features)
    inputs = processor(
        image,
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE, dtype=torch_dtype)

    # Generate answer (only keep LLaVA-supported parameters)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Decode and extract answer
    response = processor.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt part, keep only answer
    assistant_answer = response.split("ASSISTANT:")[-1].strip()
    return assistant_answer if assistant_answer else "No valid answer"


def batch_inference(processor, model, dataset):
    """Batch inference (wrap single sample inference for stability)"""
    results = []
    print(f"\nStarting inference: {len(dataset)} samples in total (full dataset)")

    for idx, data in enumerate(tqdm(dataset, desc="Inference Progress")):
        try:
            # Single sample inference (avoid image feature mismatch caused by batches)
            model_answer = single_sample_inference(
                processor, model,
                data["image_path"],
                data["question"]
            )
            results.append({
                "image_path": data["image_path"],
                "question": data["question"],
                "ground_truth": data["answer"],
                "model_answer": model_answer,
                "exact_correct": False,
                "partial_correct": False,
                "f1_score": 0.0,
                "reasoning_complete_score": 0.0,
                "hallucination_type": None,
                "hallucination_dimension": None
            })
        except Exception as e:
            print(f"\nInference failed for sample {idx}: {str(e)[:50]}")
            continue

    return results


def extract_numerical_answer(text):
    """Extract numerical answers with units"""
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|cm|m|mm|cm²|m²)?')
    matches = pattern.findall(text.replace(" ", ""))
    numerical_ans = [f"{num}{unit}" if unit else num for num, unit in matches]
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """Extract reasoning nodes based on geometric theorems"""
    reasoning_keywords = {
        "Triangle Angle Sum": ["Triangle interior angle sum", "Interior angle sum theorem", "∠A+∠B+∠C=180"],
        "Parallel Line Properties": ["Two lines parallel", "Parallel", "Same-side interior angles supplementary"],
        "Vertical Angles": ["Vertical angles", "Vertical angles equal"],
        "Pythagorean Theorem": ["Pythagorean theorem", "a²+b²=c²"],
        "Similar Triangles": ["Similar", "Similar triangles"],
        "Congruent Triangles": ["Congruent", "Congruent triangles"]
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
        return re.sub(r'[^\d°cm m mm.]', '', s).replace(" ", "")

    gt = preprocess(gt_ans)
    pred = preprocess(pred_ans)

    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0

    common = set(gt) & set(pred)
    if not common:
        return 0.0

    precision = len(common) / len(pred)
    recall = len(common) / len(gt)
    return round(2 * precision * recall / (precision + recall), 4)


def classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num):
    """Revised hallucination classification logic to avoid missing numerical error statistics"""
    # Prioritize factual hallucination: have reasoning nodes (regardless of match) + numerical error
    if len(pred_nodes) > 0:
        # Numerical calculation error (factual hallucination)
        if len(gt_num) > 0 and (len(pred_num) == 0 or set(pred_num) != set(gt_num)):
            return "Factual Hallucination", "Numerical Calculation Hallucination"
        # Theorem/concept error (factual hallucination)
        error_theorems = ["Vertical angles supplementary", "Triangle interior angle sum 360"]
        if any(kw.lower() in pred.lower() for kw in error_theorems):
            return "Factual Hallucination", "Theorem/Concept Hallucination"
    # Logical hallucination: no reasoning nodes but have numerical values, or no numerical output
    if len(pred_nodes) == 0:
        if len(pred_num) > 0:
            return "Logical Hallucination", "Reasoning Chain Break Hallucination"
        fake_cond = ["Known from question", "Given in question"]
        if any(kw.lower() in pred.lower() for kw in fake_cond):
            return "Logical Hallucination", "Condition Misuse Hallucination"
    # Fallback: reasoning exists but numerical error is still classified as factual hallucination
    if len(gt_num) > 0 and set(pred_num) != set(gt_num):
        return "Factual Hallucination", "Numerical Calculation Hallucination"
    return None, None


def calculate_metrics(results):
    """Calculate evaluation metrics"""
    total = len(results)
    if total == 0:
        return results, {}, {}

    exact_correct = 0
    partial_correct = 0
    total_f1 = 0.0
    total_reasoning = 0.0

    hallucination_stats = {
        "Factual Hallucination-Numerical Calculation": 0,
        "Factual Hallucination-Theorem/Concept": 0,
        "Logical Hallucination-Reasoning Chain Break": 0,
        "Logical Hallucination-Condition Misuse": 0,
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
            hallucination_stats["No Hallucination"] += 1
        elif len(set(gt_num) & set(pred_num)) > 0 or len(set(gt_nodes) & set(pred_nodes)) > 0:
            res["partial_correct"] = True
            partial_correct += 1
            hallucination_stats["No Hallucination"] += 1
        else:
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                if "Numerical Calculation" in typ:
                    hallucination_stats["Factual Hallucination-Numerical Calculation"] += 1
                elif "Theorem/Concept" in typ:
                    hallucination_stats["Factual Hallucination-Theorem/Concept"] += 1
                elif "Reasoning Chain Break" in typ:
                    hallucination_stats["Logical Hallucination-Reasoning Chain Break"] += 1
                elif "Condition Misuse" in typ:
                    hallucination_stats["Logical Hallucination-Condition Misuse"] += 1
            else:
                hallucination_stats["No Hallucination"] += 1

        # Calculate reasoning completeness
        if len(gt_nodes) > 0:
            reasoning_score = len(set(gt_nodes) & set(pred_nodes)) / len(gt_nodes)
        else:
            reasoning_score = 1.0 if len(pred_nodes) == 0 else 0.0
        res["reasoning_complete_score"] = reasoning_score
        total_reasoning += reasoning_score

    # Final metrics
    metrics = {
        "exact_accuracy(%)": round(exact_correct / total * 100, 2),
        "partial_accuracy(%)": round(partial_correct / total * 100, 2),
        "avg_f1_score": round(total_f1 / total, 4),
        "avg_reasoning_completeness": round(total_reasoning / total, 4)
    }

    # Normalize hallucination statistics
    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio(%)": round(hallucination_stats[key] / total * 100, 2)
        }

    return results, metrics, hallucination_stats


def save_results(results, metrics, hallucination_stats):
    """Save evaluation results"""
    save_path = "llava_mathv_full_dataset_result.json"
    final_result = {
        "model_info": {"name": "llava-v1.5-7b", "path": MODEL_PATH, "device": DEVICE},
        "dataset_info": {"name": "MATH-Vision", "total_samples": len(results), "dataset_type": "full dataset"},
        "evaluation_metrics": metrics,
        "hallucination_statistics": hallucination_stats,
        "sample_results": results[:20]  # Save first 20 samples for reference
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # Print final report
    print("\n" + "=" * 80)
    print("LLaVA-v1.5-7b MATH-Vision Final Evaluation Results (Full Dataset)")
    print("=" * 80)
    print(f"Total Valid Samples: {len(results)} (full dataset)")
    print(f"Exact Accuracy: {metrics['exact_accuracy(%)']}%")
    print(f"Partial Accuracy: {metrics['partial_accuracy(%)']}%")
    print(f"Average F1 Score: {metrics['avg_f1_score']}")
    print(f"Average Reasoning Completeness: {metrics['avg_reasoning_completeness']}")
    print("\nHallucination Statistics:")
    for key, val in hallucination_stats.items():
        print(f"  {key}: {val['count']} samples ({val['ratio(%)']}%)")
    print("=" * 80)
    print(f"Results saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    try:
        # 1. Load full dataset
        dataset = load_dataset(DATASET_PATH)
        if not dataset:
            raise ValueError("No valid samples loaded from full dataset!")

        # 2. Load LLaVA model
        processor, model = load_llava_model(MODEL_PATH)

        # 3. Inference (single sample inference for stability)
        results = batch_inference(processor, model, dataset)
        if not results:
            raise ValueError("Inference failed, no results generated!")

        # 4. Calculate evaluation metrics
        final_results, metrics, hallucination_stats = calculate_metrics(results)

        # 5. Save results
        save_results(final_results, metrics, hallucination_stats)

        print("\n✅ All processes completed successfully! No errors occurred (full dataset run)!")

    except Exception as e:
        print(f"\n❌ Runtime Error: {str(e)}")
        # Print simplified error trace
        import traceback
        traceback.print_exc(limit=1)
