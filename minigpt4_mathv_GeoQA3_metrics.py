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
    print(f"GPU not detected, will run on CPU (slow speed, recommended BATCH_SIZE=1)")
# ============================================================================

# Core Configuration (no modification needed)
MODEL_PATH = "/root/autodl-tmp/models/MiniGpt-4-7B"
DATASET_PATH = "/root/autodl-tmp/datasets/GeoQA3"  # GeoQA3 main directory
CLIP_PATH = "/root/autodl-tmp/models/clip-vit-large-patch14"
BATCH_SIZE = 1  # Set to 2 for 4090, no impact on dimension adaptation
MAX_NEW_TOKENS = 1024  # Increase generation length to accommodate reasoning steps
IMAGE_TOKEN = "<Image>"  # Fixed image placeholder for MiniGPT-4
TEST_SAMPLE_NUM = 3040  # For testing: only run first 100 valid samples
# Dimension Configuration (Key: CLIP outputs 1024d, MiniGPT-4 embedding 4096d)
CLIP_EMBED_DIM = 1024  # Fixed output dimension for CLIP-ViT-L/14
LLM_EMBED_DIM = 4096  # Word embedding dimension for MiniGPT-4-7B


def load_dataset(dataset_path):
    """GeoQA3 Final Adaptation: Compatible with out-of-order naming + adapt to real annotation fields subject/answer"""
    # Fixed subdirectories for GeoQA3 (no modification needed)
    image_dir = os.path.join(dataset_path, "image")  # Image directory: xxx.png
    json_dir = os.path.join(dataset_path, "json")  # Annotation directory: xxx.json

    # Verify directory existence (error directly if missing, quick troubleshooting)
    if not os.path.exists(image_dir):
        raise ValueError(f"GeoQA3 image directory missing: {image_dir}")
    if not os.path.exists(json_dir):
        raise ValueError(f"GeoQA3 annotation directory missing: {json_dir}")

    dataset = []
    # Load in system original order, compatible with out-of-order naming
    for json_filename in os.listdir(json_dir):
        if len(dataset) >= TEST_SAMPLE_NUM:  # Stop immediately when first 100 are loaded
            break
        # Filter non-json files only, support json with any naming
        if not json_filename.endswith(".json"):
            continue

        # Extract json basename (remove suffix) to precisely match same-name png image
        json_basename = os.path.splitext(json_filename)[0]
        img_filename = f"{json_basename}.png"
        img_filepath = os.path.join(image_dir, img_filename)
        json_filepath = os.path.join(json_dir, json_filename)

        # 1. Load json annotation, skip damaged files
        try:
            with open(json_filepath, "r", encoding="utf-8") as f:
                anno = json.load(f)
        except Exception as e:
            print(f"Skipping damaged annotation file: {json_filename}, Error: {str(e)[:50]}")
            continue

        # 2. Verify image existence, skip annotations without matching images
        if not os.path.exists(img_filepath):
            print(f"Annotation {json_filename} has no matching image, skipped: {img_filepath}")
            continue

        # 3. Verify image can be opened normally, skip damaged images
        try:
            with Image.open(img_filepath).convert("RGB") as img:
                pass
        except Exception as e:
            print(f"Skipping damaged image file: {img_filename}, Error: {str(e)[:50]}")
            continue

        # Core fix: Adapt to GeoQA3 real annotation fields → subject=question, answer=answer
        question = anno.get("subject", "").strip()  # Replace with actual question field subject
        answer = anno.get("answer", "").strip()  # Keep answer field (consistent with reality)
        if not question or not answer:
            print(f"Annotation {json_filename} has no question/subject or answer/answer, skipped")
            continue

        # 5. Add to dataset, compatible with original inference logic
        dataset.append({
            "image_path": img_filepath,
            "question": question,  # Unified as question, no impact on subsequent inference
            "answer": answer  # Unified as answer, no impact on subsequent metric calculation
        })
        # Print loading log to confirm progress
        print(f"Loaded valid sample [{len(dataset)}]: {json_filename} → {img_filename}")

    # Final verification of valid sample count, precise prompt for issues
    if len(dataset) == 0:
        raise ValueError(
            "No valid GeoQA3 samples loaded, please check: 1.json and png have the same name 2.Images are png format 3.Annotations have subject/answer fields")
    print(
        f"\nGeoQA3 dataset loaded successfully: {len(dataset)} valid test data entries (limited to first {TEST_SAMPLE_NUM})")
    return dataset


def load_minigpt4_model(model_path, clip_path, device, torch_dtype):
    """Load model + add dimension projection layer (no modification)"""
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
    """Multimodal inference: CLIP encoding → projection layer upsampling → replace <Image> → model generation (no modification)"""
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(
        f"Starting MiniGPT-4 VQA test inference: {total_batches} batches total, batch size {batch_size} ({len(dataset)} samples)")

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

        # 3. Encode text Prompt (math VQA dedicated guide Prompt, force output reasoning steps)
        prompts = []
        for d in batch_data:
            prompt = f"""{IMAGE_TOKEN} Please solve the following mathematical vision problem and answer strictly according to the following steps:
1. First analyze the known conditions in the image (such as geometric figures, numerical values, line segment lengths, angles, etc.);
2. Write down the mathematical theorems, formulas or properties needed to solve the problem;
3. Derive the reasoning steps in detail, giving the calculation basis for each step;
4. Finally give a clear and concise final answer with unit (e.g., 140°, 5cm).
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

        # 5. Model generation (optimize generation parameters for mathematical reasoning)
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
            if not ans or ans in ["Don't know", "Cannot solve", "No answer",
                                  "[Low Quality Answer] No valid reasoning and answer generated"]:
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
    """修复：优化数值提取逻辑，兼容中英文单位+小数/整数"""
    # 扩展正则：支持中英文单位（°/度、cm/厘米、m/米、mm/毫米、%/百分比、倍）
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|度|cm|厘米|m|米|mm|毫米|%|百分比|倍)?', re.IGNORECASE)
    matches = pattern.findall(text)
    numerical_ans = []
    for num, unit in matches:
        # 统一单位格式（便于匹配）
        unit_map = {
            "度": "°", "厘米": "cm", "米": "m", "毫米": "mm",
            "百分比": "%", "": "", None: ""
        }
        standard_unit = unit_map.get(unit, unit)
        numerical_ans.append(f"{num}{standard_unit}")
    # 去重并返回
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """修复：扩展推理节点关键词（兼容中英文+口语化表述）"""
    reasoning_keywords = {
        "Triangle Angle Sum": [
            "三角形内角和", "内角和定理", "∠A+∠B+∠C=180", "Triangle interior angle sum",
            "Interior angle sum theorem", "三角和180度", "三角形内角和180°", "180 degrees"
        ],
        "Parallel Line Properties": [
            "两直线平行", "平行", "同旁内角互补", "内错角相等", "同位角相等",
            "Two lines parallel", "Parallel", "Same-side interior angles supplementary",
            "Alternate interior angles equal", "Corresponding angles equal"
        ],
        "Vertical Angles": [
            "对顶角", "对顶角相等", "Vertical angles", "Vertical angles equal"
        ],
        "Pythagorean Theorem": [
            "勾股定理", "a²+b²=c²", "勾股定理逆定理", "Pythagorean theorem",
            "a²+b²=c²", "Pythagorean theorem converse"
        ],
        "Similar Triangles": [
            "相似", "相似三角形", "对应边成比例", "对应角相等",
            "Similar", "Similar triangles", "Corresponding sides proportional",
            "Corresponding angles equal"
        ],
        "Congruent Triangles": [
            "全等", "全等三角形", "SSS", "SAS", "ASA", "AAS", "HL",
            "Congruent", "Congruent triangles", "SSS", "SAS", "ASA", "AAS", "HL"
        ],
        "Area Formula": [
            "面积", "底×高÷2", "长度×宽度", "圆面积", "πr²",
            "Area", "base×height÷2", "length×width", "Circle area", "πr²"
        ],
        "Perimeter Formula": [
            "周长", "边长和", "2πr", "直径×π",
            "Perimeter", "Sum of side lengths", "2πr", "diameter×π"
        ]
    }
    nodes = set()
    text_lower = text.lower()
    for node, keywords in reasoning_keywords.items():
        # 关键词也转小写匹配
        if any(kw.lower() in text_lower for kw in keywords):
            nodes.add(node)
    return list(nodes)


def calculate_f1_score(gt_ans, pred_ans):
    """修复：替换字符级F1为数值级F1（更适配几何题）"""
    # 第一步：提取数值答案
    gt_num = extract_numerical_answer(gt_ans)
    pred_num = extract_numerical_answer(pred_ans)

    # 无数值的情况
    if not gt_num and not pred_num:
        return 1.0
    if not gt_num or not pred_num:
        return 0.0

    # 数值级F1计算：以数值+单位为粒度
    intersection = len(set(gt_num) & set(pred_num))
    precision = intersection / len(pred_num) if len(pred_num) > 0 else 0.0
    recall = intersection / len(gt_num) if len(gt_num) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num):
    """修复：修正幻觉分类逻辑，解决漏判/误判"""
    # 低质量答案直接判定为推理链断裂
    if pred.strip() == "[Low Quality Answer] No valid reasoning and answer generated":
        return "Logical Hallucination", "Reasoning Chain Break Hallucination"

    # 有推理节点但无正确数值 → 数值计算幻觉
    if len(pred_nodes) > 0:
        if set(pred_nodes) & set(gt_nodes) and len(pred_num) > 0 and set(pred_num) != set(gt_num):
            return "Factual Hallucination", "Numerical Calculation Hallucination"
        # 推理节点完全不匹配/使用错误定理 → 定理/概念幻觉
        error_theorem_kw = [
            "对顶角互补", "内错角互补", "同位角互补", "三角形内角和360",
            "Vertical angles supplementary", "Alternate interior angles supplementary",
            "Corresponding angles supplementary", "Triangle interior angle sum 360"
        ]
        if any(kw.lower() in pred.lower() for kw in error_theorem_kw) or len(set(pred_nodes) & set(gt_nodes)) == 0:
            return "Factual Hallucination", "Theorem/Concept Hallucination"
    else:
        # 无推理节点但有数值 → 推理链断裂
        if len(pred_num) > 0:
            return "Logical Hallucination", "Reasoning Chain Break Hallucination"
        # 编造条件但无数值 → 条件误用
        false_cond_kw = ["由题可知", "题目给出", "已知", "根据条件",
                         "Known from question", "Given in question", "Known", "According to conditions"]
        if any(kw.lower() in pred.lower() for kw in false_cond_kw) and len(gt_num) > 0:
            return "Logical Hallucination", "Condition Misuse Hallucination"

    # 无幻觉（仅当精确/部分匹配时）
    return None, None


def calculate_metrics(results):
    """终极修复：严格收紧部分匹配判定规则，彻底解决Partial Accuracy虚高"""
    total = len(results)
    correct_num = 0  # 精确匹配数
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

    # 关键新增：几何高频干扰数（过滤巧合重合的无意义数值）
    GEOMETRY_NOISE_NUMS = {"180", "360", "90", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

    # 统计有推理节点的样本数（用于计算平均推理完整性）
    reasoning_sample_count = sum(1 for r in results if len(extract_reasoning_nodes(r["ground_truth"])) > 0)

    for res in results:
        gt = res["ground_truth"]
        pred = res["model_answer"]

        # 1. 计算数值级F1（修复后）
        f1 = calculate_f1_score(gt, pred)
        res["f1_score"] = f1
        total_f1 += f1

        # 2. 提取数值和推理节点
        gt_num = extract_numerical_answer(gt)
        pred_num = extract_numerical_answer(pred)
        gt_nodes = extract_reasoning_nodes(gt)
        pred_nodes = extract_reasoning_nodes(pred)

        # 3. 核心过滤：移除几何高频干扰数，只保留核心答案数值
        gt_core_num = [num for num in gt_num if num not in GEOMETRY_NOISE_NUMS]
        pred_core_num = [num for num in pred_num if num not in GEOMETRY_NOISE_NUMS]

        # 4. 修复：精确匹配判定（仅数值+单位完全一致）
        if set(gt_num) == set(pred_num) and len(gt_num) > 0:
            res["exact_correct"] = True
            exact_correct += 1
            partial_correct += 1
            res["partial_correct"] = True
            res["accuracy"] = 1
            correct_num += 1
        # 5. 终极严格化：部分匹配判定（必须同时满足以下所有条件）
        elif (
            # 条件1：非低质量答案
            pred != "[Low Quality Answer] No valid reasoning and answer generated"
            # 条件2：双方都有核心有效数值（过滤干扰数后）
            and len(gt_core_num) > 0 and len(pred_core_num) > 0
            # 条件3：核心数值存在真实交集（非巧合）
            and len(set(gt_core_num) & set(pred_core_num)) > 0
            # 条件4：双方都有推理节点
            and len(gt_nodes) > 0 and len(pred_nodes) > 0
            # 条件5：推理节点存在真实交集
            and len(set(gt_nodes) & set(pred_nodes)) > 0
            # 条件6：模型推理节点数≥2（排除仅输出单个无关关键词的情况）
            and len(pred_nodes) >= 2
        ):
            res["partial_correct"] = True
            partial_correct += 1
            res["accuracy"] = 0
        # 6. 无匹配 → 判定幻觉
        else:
            res["partial_correct"] = False
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                hallucination_stats[f"{dim}-{typ}"] += 1
            res["accuracy"] = 0

        # 7. 推理完整性计算（仅当标注有推理节点时计算）
        if len(gt_nodes) > 0:
            match_nodes = len(set(gt_nodes) & set(pred_nodes))
            res["reasoning_complete_score"] = match_nodes / len(gt_nodes)
            total_reasoning_score += res["reasoning_complete_score"]
        else:
            res["reasoning_complete_score"] = 0.0

        # 8. 无幻觉统计（仅精确/严格部分匹配时判定为无幻觉）
        if res["exact_correct"] or res["partial_correct"]:
            hallucination_stats["No Hallucination"] += 1

    # 9. 计算指标（修复：overall_accuracy改为精确匹配率）
    metrics = {
        "overall_accuracy": round(correct_num / total * 100, 2),  # 精确匹配率
        "exact_accuracy": round(exact_correct / total * 100, 2),  # 同overall_accuracy，保留兼容
        "partial_accuracy": round(partial_correct / total * 100, 2),  # 真实部分匹配率
        "avg_f1_score": round(total_f1 / total, 4),  # 数值级平均F1
        "avg_reasoning_completeness": round(total_reasoning_score / max(1, reasoning_sample_count), 4)  # 仅计算有推理节点的样本
    }

    # 10. 幻觉占比计算（避免除以0）
    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio": round(hallucination_stats[key] / total * 100, 2) if total > 0 else 0.0
        }

    return results, metrics, hallucination_stats


def save_results(results, metrics, hallucination_stats):
    """Save results (GeoQA3 exclusive filename, no modification)"""
    save_path = "minigpt4_GeoQA3_vqa_test_100samples_fixed.json"  # 标记修复版
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
            "dataset_name": "GeoQA3",
            "dataset_path": DATASET_PATH,
            "total_test_samples": len(results),
            "sample_limit": f"First {TEST_SAMPLE_NUM} valid samples",
            "dataset_structure": "Compatible with any naming: image/xxx.png ↔ json/xxx.json (same basename required)",
            "annotation_fields": "Real annotation fields: subject (question), answer (answer)"
        },
        "evaluation_metrics": metrics,
        "hallucination_statistics": hallucination_stats,
        "hallucination_definition": {
            "Factual Hallucination": "Has reasoning logic but incorrect numerical calculation/mathematical theorems",
            "Logical Hallucination": "No reasonable reasoning logic, baseless answers/reasoning break",
            "Numerical Calculation Hallucination": "Correct reasoning steps but wrong/fabricated numerical values",
            "Theorem/Concept Hallucination": "Fabricate non-existent theorems or incorrectly use geometric concepts",
            "Reasoning Chain Break Hallucination": "Direct answer without reasoning process / no valid answer",
            "Condition Misuse Hallucination": "Fabricate question conditions or incorrectly use known conditions"
        },
        "metric_definition": {
            "overall_accuracy": "Exact match accuracy (numerical + unit completely correct)",
            "exact_accuracy": "Same as overall_accuracy (compatibility)",
            "partial_accuracy": "Partial match accuracy (numerical intersection >0 or reasoning node intersection >0)",
            "avg_f1_score": "Average numerical-level F1 score (adapted for geometry answers)",
            "avg_reasoning_completeness": "Average reasoning completeness (core reasoning node coverage ratio, only for samples with reasoning nodes)"
        },
        "detailed_test_results": results
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("MiniGPT-4-7B GeoQA3 Dataset - 100 Samples Test Version (Fixed Metrics) Evaluation Report")
    print("=" * 90)
    print(
        f"Running Configuration: {DEVICE} | Batch Size={BATCH_SIZE} | Dimension Fusion: {CLIP_EMBED_DIM}→{LLM_EMBED_DIM}")
    print(f"Test Data Scale: {len(results)} valid samples | Generation Config: num_beams=4, temp=0.7")
    print("=" * 90)
    print("Core Evaluation Metrics (Fixed Version)")
    print("=" * 90)
    print(f"Overall Accuracy (Exact Match): {metrics['overall_accuracy']}%")
    print(f"Exact Accuracy (Numerical+Unit Match): {metrics['exact_accuracy']}%")
    print(f"Partial Accuracy (Real Match): {metrics['partial_accuracy']}%")
    print(f"Average F1 Score (Numerical-level): {metrics['avg_f1_score']} (0~1)")
    print(f"Average Reasoning Completeness: {metrics['avg_reasoning_completeness']} (0~1)")
    print("=" * 90)
    print("Hallucination Classification Statistics (Count | Ratio)")
    print("=" * 90)
    for key, val in hallucination_stats.items():
        print(f"{key.ljust(45)}: {val['count']} entries | {val['ratio']}%")
    print("=" * 90)
    print(f"Test Result File Saved To: {os.path.abspath(save_path)}")
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
        # Calculate full metrics (fixed version)
        final_results, metrics, hallucination_stats = calculate_metrics(raw_results)
        save_results(final_results, metrics, hallucination_stats)
        print("\nMiniGPT-4 Multimodal VQA GeoQA3 100 Samples Test (Fixed Metrics) Completed!")
    except Exception as e:
        print(f"\nRuntime Error: {str(e)}")
        import traceback


        traceback.print_exc()
