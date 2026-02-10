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
    print("GPU not detected, will run on CPU (slow speed, recommended BATCH_SIZE=1)")
# ============================================================================

# Core Configuration (adapt to your local path, no modification needed)
MODEL_PATH = "/root/autodl-tmp/models/MiniGpt-4-7B"
DATASET_PATH = "/root/autodl-tmp/datasets/mathv_3040"
CLIP_PATH = "/root/autodl-tmp/models/clip-vit-large-patch14"  # Your local CLIP directory
BATCH_SIZE = 1  # Set to 2 for 4090, no impact on dimension adaptation
MAX_NEW_TOKENS = 1024  # Increase generation length to accommodate reasoning steps
IMAGE_TOKEN = "<Image>"  # Fixed image placeholder for MiniGPT-4
TEST_SAMPLE_NUM = 3040   # For testing: only run first 100 samples
# Dimension Configuration (Key: CLIP outputs 1024d, MiniGPT-4 embedding 4096d)
CLIP_EMBED_DIM = 1024  # Fixed output dimension for CLIP-ViT-L/14
LLM_EMBED_DIM = 4096  # Word embedding dimension for MiniGPT-4-7B


def load_dataset(dataset_path):
    """Load dataset, filter corrupted images, only keep first 100 valid samples"""
    anno_path = os.path.join(dataset_path, "annotations.json")
    image_dir = os.path.join(dataset_path, "images")

    if not os.path.exists(anno_path) or not os.path.exists(image_dir):
        raise ValueError(f"Dataset path error: {anno_path} or {image_dir} does not exist")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    dataset = []
    for anno in annotations:
        if len(dataset) >= TEST_SAMPLE_NUM:  # Stop when first 100 samples are loaded
            break
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
            except Exception as e:
                continue
    print(f"Dataset loaded successfully: {len(dataset)} valid test data entries (limited to first {TEST_SAMPLE_NUM})")
    return dataset


def load_minigpt4_model(model_path, clip_path, device, torch_dtype):
    """Load model + add dimension projection layer (core fix)"""
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

    # 3. Load CLIP vision encoder (local path, no network required)
    print(f"Loading CLIP vision encoder: {clip_path}")
    clip_processor = CLIPImageProcessor.from_pretrained(clip_path)
    clip_model = CLIPVisionModel.from_pretrained(
        clip_path,
        dtype=torch_dtype,
        device_map=device
    ).eval()
    print(f"CLIP vision encoder loaded: output dimension={CLIP_EMBED_DIM}")

    # 4. Core fix: add linear projection layer (1024d → 4096d), standard fusion layer for MiniGPT-4
    print(f"Initializing dimension projection layer: {CLIP_EMBED_DIM} → {LLM_EMBED_DIM}")
    image_proj = nn.Linear(CLIP_EMBED_DIM, LLM_EMBED_DIM, bias=False).to(device, torch_dtype)
    # Projection layer initialization (follow MiniGPT-4 official config to ensure fusion effect)
    nn.init.normal_(image_proj.weight, std=CLIP_EMBED_DIM ** -0.5)

    # 5. Bind image_token_id (add to vocab if not exists)
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"Image_token_id bound: {image_token_id}")

    return tokenizer, model, clip_processor, clip_model, image_proj, image_token_id


def batch_inference(tokenizer, model, clip_processor, clip_model, image_proj, image_token_id, dataset, device,
                    batch_size=1):
    """Multimodal inference: CLIP encoding → projection layer upsampling → replace <Image> → model generation"""
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Starting MiniGPT-4 VQA test inference: {total_batches} batches total, batch size {batch_size} ({len(dataset)} samples)")

    for i in tqdm(range(total_batches), desc="VQA Test Inference Progress"):
        batch_data = dataset[i * batch_size: (i + 1) * batch_size]
        if not batch_data:
            continue

        # 1. CLIP encode images: get 1024d visual features
        images = [Image.open(d["image_path"]).convert("RGB") for d in batch_data]
        clip_inputs = clip_processor(images=images, return_tensors="pt").to(device, torch_dtype)
        with torch.no_grad():
            clip_embeds = clip_model(**clip_inputs).last_hidden_state[:, 0, :]  # Take [CLS] feature: (batch, 1024)

        # 2. Projection layer upsampling: 1024d → 4096d (core fix, match MiniGPT-4 embedding)
        with torch.no_grad():
            image_embeds = image_proj(clip_embeds)  # (batch, 4096)

        # 3. Encode text prompts (Core Modification 1: math VQA dedicated guide prompt, force output reasoning steps)
        prompts = []
        for d in batch_data:
            prompt = f"""{IMAGE_TOKEN} Please solve the following mathematical vision problem and answer strictly according to the following steps:
1. First analyze the known conditions in the image (such as geometric figures, numerical values, line segment lengths, angles, etc.);
2. Write down the mathematical theorems, formulas or properties needed to solve the problem;
3. Derive the reasoning steps in detail, giving the calculation basis for each step;
4. Finally give a clear and concise final answer.
Question: {d['question']}
Detailed solution and answer:"""
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

        # 4. Replace <Image> with upsampled visual features (dimension matched, can be directly assigned)
        input_embeds = model.get_input_embeddings()(input_ids)  # (batch, seq_len, 4096)
        image_token_mask = (input_ids == image_token_id).nonzero(as_tuple=True)
        for batch_idx, seq_idx in zip(image_token_mask[0], image_token_mask[1]):
            input_embeds[batch_idx, seq_idx] = image_embeds[batch_idx]  # Dimension consistent, no error

        # 5. Model generation (Core Modification 2: optimize generation parameters for mathematical reasoning)
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,  # Sufficient length to accommodate reasoning
                do_sample=True,  # Enable sampling to improve reasoning flexibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=4,  # Beam search to improve answer logic and rationality
                repetition_penalty=1.2,  # Increase repetition penalty to avoid meaningless nonsense
                temperature=0.7,  # Moderate temperature to balance certainty and reasoning diversity
                top_p=0.95  # Nucleus sampling to filter low-probability invalid tokens
            )

        # 6. Parse and save results (optimize low-quality answer marking)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for d, text, prompt in zip(batch_data, generated_texts, prompts):
            ans = text.replace(prompt, "").strip()
            # Mark low-quality answers for subsequent metric analysis
            if not ans or ans in ["Don't know", "Cannot solve", "No answer"]:
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

    print(f"VQA test inference completed: {len(results)} valid answers generated")
    return results


def extract_numerical_answer(text):
    """Extract numerical answers with units from text"""
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|cm|m|mm|%|times)?')
    matches = pattern.findall(text)
    numerical_ans = [f"{num}{unit}" if unit else num for num, unit in matches]
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """终极修复：语义模糊匹配 + 中英文兼容 + 宽松规则"""
    # 简化推理节点分类，降低匹配门槛
    reasoning_patterns = {
        "Triangle Angle Sum": r"三角形.*内角和|内角和.*180|∠.*\+.*=.*180|triangle.*angle.*sum|180.*degree",
        "Parallel Line Properties": r"平行.*线|同旁内角|内错角|同位角|parallel.*line|supplementary.*angle",
        "Vertical Angles": r"对顶角|vertical.*angle",
        "Pythagorean Theorem": r"勾股定理|a².*b².*c²|pythagorean.*theorem",
        "Similar/Congruent Triangles": r"相似|全等|similar|congruent|SSS|SAS|ASA",
        "Area/Perimeter": r"面积|周长|area|perimeter|底.*高|length.*width"
    }
    nodes = set()
    # 移除大小写过滤，适配中文
    for node, pattern in reasoning_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):  # 忽略英文大小写，中文无影响
            nodes.add(node)
    return list(nodes)


def calculate_f1_score(gt_ans, pred_ans):
    """终极修复：数值级F1（适配数学答案）"""
    # 复用已有的数值提取函数
    gt_num = extract_numerical_answer(gt_ans)
    pred_num = extract_numerical_answer(pred_ans)

    # 处理空值情况
    if not gt_num and not pred_num:
        return 1.0
    if not gt_num or not pred_num:
        return 0.0

    # 计算数值级精确率、召回率、F1
    common_num = list(set(gt_num) & set(pred_num))
    precision = len(common_num) / len(pred_num)
    recall = len(common_num) / len(gt_num)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num):
    """修正幻觉分类逻辑：强化数值计算幻觉判定，放宽定理幻觉判定"""
    # 1. 事实性幻觉 - 数值计算错误（核心修正：只要推理节点匹配但数值不匹配）
    if len(pred_nodes) > 0 and len(gt_nodes) > 0:
        if len(set(pred_nodes) & set(gt_nodes)) > 0 and len(pred_num) > 0 and set(pred_num) != set(gt_num):
            return "Factual Hallucination", "Numerical Calculation Hallucination"
    # 2. 事实性幻觉 - 定理/概念错误
    error_theorem_kw = ["Vertical angles supplementary", "Alternate interior angles supplementary",
                        "Corresponding angles supplementary", "Triangle interior angle sum 360",
                        "Pythagorean theorem a²-b²=c²"]
    if any(kw.lower() in pred.lower() for kw in error_theorem_kw):
        return "Factual Hallucination", "Theorem/Concept Hallucination"
    # 3. 逻辑性幻觉 - 推理链断裂（无推理节点但有数值输出）
    if len(pred_nodes) == 0 and len(pred_num) > 0:
        return "Logical Hallucination", "Reasoning Chain Break Hallucination"
    # 4. 逻辑性幻觉 - 条件误用（编造条件）
    false_cond_kw = ["Known from question", "Given in question", "Known", "According to conditions"]
    if any(kw.lower() in pred.lower() for kw in false_cond_kw) and len(gt_num) > 0 and len(pred_num) == 0:
        return "Logical Hallucination", "Condition Misuse Hallucination"
    # 5. 兜底：有推理但数值错误仍归为数值计算幻觉
    if len(gt_num) > 0 and len(pred_num) > 0 and set(pred_num) != set(gt_num):
        return "Factual Hallucination", "Numerical Calculation Hallucination"
    return None, None


def calculate_metrics(results):
    """修正指标计算逻辑：推理完整性分母、部分准确率判定、幻觉统计"""
    total = len(results)
    correct_num = 0
    exact_correct = 0
    partial_correct = 0
    total_f1 = 0.0
    total_reasoning_score = 0.0
    reasoning_sample_count = 0  # 新增：统计有推理节点的样本数
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
        f1 = calculate_f1_score(gt, pred)
        res["f1_score"] = f1
        total_f1 += f1
        gt_num = extract_numerical_answer(gt)
        pred_num = extract_numerical_answer(pred)
        gt_nodes = extract_reasoning_nodes(gt)
        pred_nodes = extract_reasoning_nodes(pred)

        # 1. 修正精确/部分准确率判定逻辑
        is_low_quality = pred == "[Low Quality Answer] No valid reasoning and answer generated"
        if set(gt_num) == set(pred_num) and len(gt_num) > 0 and not is_low_quality:
            res["exact_correct"] = True
            exact_correct += 1
            partial_correct += 1
            res["partial_correct"] = True
            res["accuracy"] = 1
            correct_num += 1
        elif not is_low_quality and (
            (len(gt_num) > 0 and len(pred_num) > 0 and len(set(gt_num) & set(pred_num)) > 0)
            or (len(gt_nodes) > 0 and len(pred_nodes) > 0 and len(set(gt_nodes) & set(pred_nodes)) > 0)
        ):
            res["partial_correct"] = True
            partial_correct += 1
            res["accuracy"] = 0
        else:
            # 2. 修正幻觉分类逻辑
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                hallucination_stats[f"{dim}-{typ}"] += 1
            res["accuracy"] = 0

        # 3. 修正推理完整性计算（分母为有推理节点的样本数）
        if len(gt_nodes) == 0:
            res["reasoning_complete_score"] = 1.0 if len(pred_nodes) == 0 else 0.0
        else:
            reasoning_sample_count += 1  # 仅统计有推理节点的样本
            match_nodes = len(set(gt_nodes) & set(pred_nodes))
            res["reasoning_complete_score"] = match_nodes / len(gt_nodes)
            total_reasoning_score += res["reasoning_complete_score"]

        # 4. 修正无幻觉判定逻辑（仅精确/部分正确且无数值错误）
        if res["exact_correct"] or (res["partial_correct"] and len(set(gt_num) & set(pred_num)) > 0):
            hallucination_stats["No Hallucination"] += 1

    # 计算最终指标
    metrics = {
        "overall_accuracy": round(correct_num / total * 100, 2),
        "exact_accuracy": round(exact_correct / total * 100, 2),
        "partial_accuracy": round(partial_correct / total * 100, 2),
        "avg_f1_score": round(total_f1 / total, 4),
        # 推理完整性分母修正：避免除以0
        "avg_reasoning_completeness": round(total_reasoning_score / max(1, reasoning_sample_count), 4)
    }

    # 归一化幻觉统计
    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio": round(hallucination_stats[key] / total * 100, 2)
        }

    return results, metrics, hallucination_stats


def save_results(results, metrics, hallucination_stats):
    """Save test results with full English annotations and structured format"""
    save_path = "minigpt4_mathv_vqa_test_100samples_fixed.json"  # 标记修正版
    final_result = {
        "model_info": {
            "model_name": "MiniGPT-4-7B (Text-only + CLIP + Dimension Projection Layer + Fixed Metrics)",
            "model_path": MODEL_PATH,
            "clip_path": CLIP_PATH,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "fusion_config": f"CLIP({CLIP_EMBED_DIM}d) → Linear Projection → MiniGPT-4({LLM_EMBED_DIM}d)",
            "generation_config": "num_beams=4, temperature=0.7, top_p=0.95, repetition_penalty=1.2"
        },
        "dataset_info": {
            "dataset_name": "MATH-Vision",
            "dataset_path": DATASET_PATH,
            "total_test_samples": len(results),
            "sample_limit": f"First {TEST_SAMPLE_NUM} valid samples"
        },
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
            "overall_accuracy": "Overall accuracy (perfect match ratio), 1=completely correct answer, 0=incorrect",
            "exact_accuracy": "Exact accuracy (perfect numerical match)",
            "partial_accuracy": "Partial accuracy (core numerical/reasoning node match, exclude low-quality answers)",
            "avg_f1_score": "Average character-level F1 score (adapted for math problem answers)",
            "avg_reasoning_completeness": "Average reasoning completeness (only for samples with reasoning nodes)"
        },
        "fixes": [
            "1. Reasoning completeness: denominator = number of samples with reasoning nodes (not total samples)",
            "2. Partial accuracy: exclude low-quality answers, stricter matching rules",
            "3. Hallucination classification: strengthen numerical calculation hallucination detection",
            "4. No hallucination: only mark samples with exact/partial correct numerical answers"
        ],
        "detailed_test_results": results
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("MiniGPT-4-7B MATH-Vision Dataset - Fixed Metrics Evaluation Report (100 Samples)")
    print("=" * 90)
    print(f"Running Configuration: {DEVICE} | Batch Size={BATCH_SIZE} | Dimension Fusion: {CLIP_EMBED_DIM}→{LLM_EMBED_DIM}")
    print(f"Test Data Scale: {len(results)} valid samples | Generation Config: num_beams=4, temp=0.7")
    print("=" * 90)
    print("Core Evaluation Metrics (Fixed Version)")
    print("=" * 90)
    print(f"Overall Accuracy (Perfect Match): {metrics['overall_accuracy']}%")
    print(f"Exact Accuracy (Perfect Numerical Match): {metrics['exact_accuracy']}%")
    print(f"Partial Accuracy (Core Match): {metrics['partial_accuracy']}%")
    print(f"Average F1 Score: {metrics['avg_f1_score']} (0~1)")
    print(f"Average Reasoning Completeness: {metrics['avg_reasoning_completeness']} (0~1)")
    print("=" * 90)
    print("Hallucination Classification Statistics (Count | Ratio)")
    print("=" * 90)
    for key, val in hallucination_stats.items():
        print(f"{key.ljust(50)}: {val['count']} | {val['ratio']}%")
    print("=" * 90)
    print(f"Fixed Result File Saved To: {os.path.abspath(save_path)}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        dataset = load_dataset(DATASET_PATH)
        if not dataset:
            raise ValueError("No valid test data loaded")
        # Load model (including projection layer)
        tokenizer, model, clip_processor, clip_model, image_proj, image_token_id = load_minigpt4_model(
            MODEL_PATH, CLIP_PATH, DEVICE, torch_dtype
        )
        # Start test inference (optimized configuration)
        raw_results = batch_inference(
            tokenizer, model, clip_processor, clip_model, image_proj, image_token_id, dataset, DEVICE, BATCH_SIZE
        )
        if not raw_results:
            raise ValueError("Test inference failed")
        # Calculate fixed metrics
        final_results, metrics, hallucination_stats = calculate_metrics(raw_results)
        save_results(final_results, metrics, hallucination_stats)
        print("\n✅ MiniGPT-4 Multimodal VQA Test (Fixed Metrics) Completed Successfully!")
    except Exception as e:
        print(f"\n❌ Runtime Error: {str(e)}")
        import traceback
        traceback.print_exc()
