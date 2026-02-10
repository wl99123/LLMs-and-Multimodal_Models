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
    torch_dtype = torch.float16  # High precision for GPU only
    print(f"GPU automatically enabled: {torch.cuda.get_device_name(0)} | Precision: float16")
else:
    DEVICE = "cpu"
    torch_dtype = torch.float32
    print("GPU not detected, will run on CPU (limited performance, GPU recommended)")
# ============================================================================

# Core Configuration (GeoQA3 exclusive + GPU optimization, set BATCH_SIZE=2 for 24G VRAM, 1 for 16G)
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_PATH = "/root/autodl-tmp/datasets/GeoQA3"  # GeoQA3 main directory
BATCH_SIZE = 1  # 16G GPU=1, 24G GPU=2
MAX_NEW_TOKENS = 1024  # Accommodate complete geometric reasoning steps
TEST_SAMPLE_NUM = 100  # Test 100 samples first, increase after verification (e.g., 10000)


def load_dataset(dataset_path):
    """GeoQA3 exclusive loading: Adapt to image/xxx.png ↔ json/xxx.json same-name rule, extract subject/answer fields"""
    image_dir = os.path.join(dataset_path, "image")  # GeoQA3 image directory
    json_dir = os.path.join(dataset_path, "json")  # GeoQA3 annotation directory

    # Verify core directories exist
    if not os.path.exists(image_dir) or not os.path.exists(json_dir):
        raise ValueError(
            f"GeoQA3 dataset path error: {image_dir} or {json_dir} does not exist, check directory structure")

    dataset = []
    # Traverse json annotations, match same-name png images, compatible with out-of-order/any naming
    for json_filename in os.listdir(json_dir):
        if len(dataset) >= TEST_SAMPLE_NUM:  # Limit test samples for quick verification
            break
        if not json_filename.endswith(".json"):  # Process json files only
            continue

        # Extract json basename to precisely match same-name png image
        json_basename = os.path.splitext(json_filename)[0]
        img_filename = f"{json_basename}.png"
        img_path = os.path.join(image_dir, img_filename)
        json_path = os.path.join(json_dir, json_filename)

        # Load and verify json annotation
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                anno = json.load(f)
        except Exception as e:
            print(f"Skipping damaged annotation: {json_filename}, Error: {str(e)[:50]}")
            continue

        # Extract GeoQA3 exclusive annotation fields: subject=question, answer=answer
        question = anno.get("subject", "").strip()
        answer = anno.get("answer", "").strip()
        if not question or not answer:
            print(f"Skipping annotation without question/answer: {json_filename}")
            continue

        # Verify image exists and can be opened normally
        if os.path.exists(img_path):
            try:
                # Verify image integrity only, do not keep object to save memory
                Image.open(img_path).convert("RGB").close()
                dataset.append({
                    "image_path": img_path,
                    "question": question,  # Unified as question for subsequent inference
                    "answer": answer  # Unified as answer for metric calculation
                })
            except Exception as e:
                print(f"Skipping damaged image: {img_filename}, Error: {str(e)[:50]}")
        else:
            print(f"Annotation {json_filename} has no matching image: {img_filename}")

    # Final verification of valid sample count
    if len(dataset) == 0:
        raise ValueError(
            "No valid GeoQA3 samples loaded, check: 1.json and png have the same name 2.Annotations have subject/answer 3.Images are png format")
    print(
        f"GeoQA3 dataset loaded successfully: {len(dataset)} valid test data entries (limited to first {TEST_SAMPLE_NUM})")
    return dataset


def load_llava_model(model_path, device, dtype):
    """Load LLaVA model with GPU exclusive optimization, enable low memory usage mode"""
    print(f"Loading LLaVA-v1.5-7b model: {model_path} | Running device: {device} | Precision: {dtype}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    processor.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    print(f"Image_token_id bound: {processor.image_token_id}")

    # GPU exclusive: enable low_cpu_mem_usage for large model loading
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Key: reduce CPU memory usage to avoid loading failure
    ).eval()
    print("LLaVA model loaded successfully, GPU exclusive optimization enabled")
    return processor, model


def batch_inference(processor, model, dataset, device, batch_size=1):
    """GPU exclusive batch inference: Enhanced geometric VQA Prompt + optimized generation parameters (adapted for GeoQA3 geometry problems)"""
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Starting GPU optimized inference: {total_batches} batches total, batch size {batch_size}")

    for i in tqdm(range(total_batches), desc="GPU Inference Progress"):
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
                # Core optimization: GeoQA3 geometry VQA 4-step mandatory guide Prompt (LLaVA exclusive adaptation)
                # 修复：强化Prompt，强制输出数值+单位，兼容中英文
                prompt = f"""<image> Solve the following geometric vision problem and output strictly according to 4 steps:
1. Extract known conditions from the image: clarify geometric figure type + specific values (angle/length/shape features, etc.);
2. Match geometric theorems: write down the geometric theorems/formulas/properties corresponding to the problem (e.g., triangle angle sum, parallel line properties);
3. Detailed derivation steps: calculate each step with extracted conditions and theorems, mark derivation basis;
4. Final answer: give a concise and accurate final result (MUST include unit, e.g., 140°, 5cm, 8米).
Question: {d['question']}
Solution:"""
                valid_prompts.append(prompt)
                valid_data.append(d)
            except Exception as e:
                continue

        if not images or not valid_prompts:
            continue

        # GPU preprocessing: reasonable truncation + max length adaptation to avoid out-of-memory
        inputs = processor(
            images=images,
            text=valid_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,  # Enable reasonable truncation on GPU to avoid OOM from long text
            max_length=2048  # Sufficient to accommodate <image>+Prompt+geometric reasoning steps
        ).to(device)

        # GPU exclusive generation parameters: balance logic and completeness of geometric reasoning
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,  # Enable sampling to improve geometric reasoning flexibility
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=4,  # Beam search to select optimal geometric reasoning path
                temperature=0.7,  # Moderate temperature: not random, not rigid
                top_p=0.95,  # Nucleus sampling to filter low-probability invalid content
                repetition_penalty=1.2  # Increase repetition penalty to avoid nonsense/step skipping
            )

        # Parse results and mark low-quality answers
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        for d, text, prompt in zip(valid_data, generated_texts, valid_prompts):
            ans = text.replace(prompt, "").strip()
            # 修复：扩展低质量答案判定范围，统一标记
            low_quality_keywords = ["Don't know", "Cannot solve", "No answer", "无法解答", "不知道", "无答案"]
            if not ans or any(kw in ans for kw in low_quality_keywords):
                ans = "[Low Quality Answer] No valid reasoning and answer generated"
            results.append({
                "image_path": d["image_path"],
                "question": d["question"],
                "ground_truth": d["answer"],
                "model_answer": ans,
                "exact_correct": False,
                "partial_correct": False,
                "f1_score": 0.0,
                "reasoning_complete_score": 0.0,
                "hallucination_type": None,
                "hallucination_dimension": None
            })

    print(f"GPU inference completed: {len(results)} valid answers generated (GeoQA3 geometry problems)")
    return results


def extract_numerical_answer(text):
    """修复：优化数值提取逻辑，兼容中英文单位+小数/整数，统一单位格式"""
    # 扩展正则：支持中英文单位（°/度、cm/厘米、m/米、mm/毫米、%/百分比、倍）
    pattern = re.compile(r'(\d+\.?\d*)\s*(°|度|cm|厘米|m|米|mm|毫米|%|百分比|倍)?', re.IGNORECASE)
    matches = pattern.findall(text)
    numerical_ans = []
    # 单位映射表：统一为英文单位便于匹配
    unit_map = {
        "度": "°", "厘米": "cm", "米": "m", "毫米": "mm",
        "百分比": "%", "倍": "", "": "", None: ""
    }
    for num, unit in matches:
        standard_unit = unit_map.get(unit, unit)
        numerical_ans.append(f"{num}{standard_unit}")
    # 去重并返回
    return list(set(numerical_ans)) if numerical_ans else []


def extract_reasoning_nodes(text):
    """修复：扩展推理节点关键词（兼容中英文+口语化表述），覆盖GeoQA3核心几何考点"""
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
            "Converse of Pythagorean theorem"
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
            "面积", "底×高÷2", "三角形面积", "平行四边形面积", "圆面积", "πr²",
            "Area", "base×height÷2", "Triangle area", "Parallelogram area", "Circle area", "πr²"
        ],
        "Perimeter Formula": [
            "周长", "边长和", "圆周长", "2πr", "直径×π",
            "Perimeter", "Sum of side lengths", "Circle circumference", "2πr", "diameter×π"
        ]
    }
    nodes = set()
    text_lower = text.lower()
    for node, keywords in reasoning_keywords.items():
        # 关键词转小写匹配，兼容大小写
        if any(kw.lower() in text_lower for kw in keywords):
            nodes.add(node)
    return list(nodes)


def calculate_f1_score(gt_ans, pred_ans):
    """修复：替换字符级F1为数值级F1（更适配几何题答案）"""
    # 第一步：提取数值答案
    gt_num = extract_numerical_answer(gt_ans)
    pred_num = extract_numerical_answer(pred_ans)

    # 无数值的情况
    if not gt_num and not pred_num:
        return 1.0
    if not gt_num or not pred_num:
        return 0.0

    # 数值级F1计算：以“数值+单位”为粒度
    intersection = len(set(gt_num) & set(pred_num))
    precision = intersection / len(pred_num) if len(pred_num) > 0 else 0.0
    recall = intersection / len(gt_num) if len(gt_num) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num):
    """LLaVA修复：补全数值计算幻觉判定，解决漏判"""
    # 低质量答案 → 推理链断裂
    if pred.strip() == "[Low Quality Answer] No valid reasoning and answer generated":
        return "Logical Hallucination", "Reasoning Chain Break Hallucination"

    # 核心修复：LLaVA特有场景——正确定理+错误数值 → 数值计算幻觉
    if len(pred_nodes) > 0 and len(gt_nodes) > 0:
        # 推理节点匹配，但数值不匹配 → 数值计算幻觉
        if len(set(pred_nodes) & set(gt_nodes)) > 0 and len(pred_num) > 0 and set(pred_num) != set(gt_num):
            return "Factual Hallucination", "Numerical Calculation Hallucination"
        # 推理节点错误 → 定理/概念幻觉
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
        # 编造条件 → 条件误用
        false_cond_kw = ["由题可知", "题目给出", "已知", "根据条件",
                         "Known from question", "Given in question", "Known", "According to conditions"]
        if any(kw.lower() in pred.lower() for kw in false_cond_kw) and len(gt_num) > 0:
            return "Logical Hallucination", "Condition Misuse Hallucination"

    return None, None


def calculate_metrics_and_hallucination(results):
    """LLaVA终极修复：严格收紧部分匹配+修复幻觉分类，解决95%虚高"""
    total = len(results)
    exact_correct = 0
    partial_correct = 0
    total_reasoning_score = 0.0
    total_f1_score = 0.0
    hallucination_stats = {
        "Factual Hallucination-Numerical Calculation Hallucination": 0,
        "Factual Hallucination-Theorem/Concept Hallucination": 0,
        "Logical Hallucination-Reasoning Chain Break Hallucination": 0,
        "Logical Hallucination-Condition Misuse Hallucination": 0,
        "No Hallucination": 0
    }

    # 1. 新增：几何高频干扰数（过滤巧合数值）
    GEOMETRY_NOISE_NUMS = {"180", "360", "90", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    # 2. 统计有推理节点的样本数
    reasoning_sample_count = sum(1 for r in results if len(extract_reasoning_nodes(r["ground_truth"])) > 0)

    for res in results:
        gt = res["ground_truth"]
        pred = res["model_answer"]

        # 计算F1
        f1 = calculate_f1_score(gt, pred)
        res["f1_score"] = f1
        total_f1_score += f1

        # 提取数值和推理节点
        gt_num = extract_numerical_answer(gt)
        pred_num = extract_numerical_answer(pred)
        gt_nodes = extract_reasoning_nodes(gt)
        pred_nodes = extract_reasoning_nodes(pred)

        # 核心过滤：移除干扰数，保留核心数值
        gt_core_num = [num for num in gt_num if num not in GEOMETRY_NOISE_NUMS]
        pred_core_num = [num for num in pred_num if num not in GEOMETRY_NOISE_NUMS]

        # 精确匹配判定
        if set(gt_num) == set(pred_num) and len(gt_num) > 0:
            res["exact_correct"] = True
            exact_correct += 1
            partial_correct += 1
            res["partial_correct"] = True
        # 终极严格化：部分匹配（6个条件缺一不可）
        elif (
            # 条件1：非低质量答案
            pred != "[Low Quality Answer] No valid reasoning and answer generated"
            # 条件2：双方核心数值非空
            and len(gt_core_num) > 0 and len(pred_core_num) > 0
            # 条件3：核心数值有真实交集
            and len(set(gt_core_num) & set(pred_core_num)) > 0
            # 条件4：双方推理节点非空
            and len(gt_nodes) > 0 and len(pred_nodes) > 0
            # 条件5：推理节点有真实交集
            and len(set(gt_nodes) & set(pred_nodes)) > 0
            # 条件6：模型推理节点≥2（排除单个关键词）
            and len(pred_nodes) >= 2
        ):
            res["partial_correct"] = True
            partial_correct += 1
        # 无匹配 → 判定幻觉
        else:
            res["partial_correct"] = False
            dim, typ = classify_hallucination(gt, pred, gt_nodes, pred_nodes, gt_num, pred_num)
            res["hallucination_dimension"] = dim
            res["hallucination_type"] = typ
            if dim and typ:
                hallucination_stats[f"{dim}-{typ}"] += 1

        # 推理完整性计算
        if len(gt_nodes) > 0:
            match_nodes = len(set(gt_nodes) & set(pred_nodes))
            res["reasoning_complete_score"] = match_nodes / len(gt_nodes)
            total_reasoning_score += res["reasoning_complete_score"]
        else:
            res["reasoning_complete_score"] = 0.0

        # 无幻觉判定（仅精确/严格部分匹配）
        if res["exact_correct"] or res["partial_correct"]:
            hallucination_stats["No Hallucination"] += 1

    # 计算指标
    metrics = {
        "overall_accuracy": round(exact_correct / total * 100, 2),
        "exact_accuracy": round(exact_correct / total * 100, 2),
        "partial_accuracy": round(partial_correct / total * 100, 2),
        "avg_f1_score": round(total_f1_score / total, 4),
        "avg_reasoning_completeness": round(total_reasoning_score / max(1, reasoning_sample_count), 4)
    }

    # 幻觉占比计算
    for key in hallucination_stats:
        hallucination_stats[key] = {
            "count": hallucination_stats[key],
            "ratio": round(hallucination_stats[key] / total * 100, 2) if total > 0 else 0.0
        }

    return results, metrics, hallucination_stats


def save_test_results(results, metrics, hallucination_stats):
    """Save results (GeoQA3 exclusive naming, distinguish from other datasets, record core configuration)"""
    save_path = "llava_GeoQA3_gpu_optimize_100sample_fixed.json"  # 标记修复版
    final_result = {
        "model_info": {
            "model_name": "llava-v1.5-7b (Fixed Metrics)",
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_config": "num_beams=4, temperature=0.7, top_p=0.95, repetition_penalty=1.2"
        },
        "dataset_info": {
            "dataset_name": "GeoQA3",  # Dataset name set to GeoQA3
            "dataset_path": DATASET_PATH,
            "total_test_samples": len(results),
            "sample_limit": f"First {TEST_SAMPLE_NUM} valid samples",
            "dataset_rule": "image/xxx.png ↔ json/xxx.json (same name matching), annotation fields: subject(question)/answer(answer)"
        },
        "evaluation_metrics": metrics,
        "hallucination_statistics": hallucination_stats,
        "hallucination_definition": {
            "Factual Hallucination": "Has reasoning logic but incorrect geometric numerical calculation/theorem concept usage",
            "Logical Hallucination": "No reasonable geometric reasoning logic, baseless answers/reasoning chain break",
            "Numerical Calculation Hallucination": "Correct geometric reasoning steps but wrong final numerical/unit calculation",
            "Theorem/Concept Hallucination": "Fabricate non-existent geometric theorems/incorrectly use geometric concepts (e.g., vertical angles supplementary)",
            "Reasoning Chain Break Hallucination": "Direct answer without geometric reasoning process / no valid answer",
            "Condition Misuse Hallucination": "Fabricate non-existent geometric conditions in the question for reasoning"
        },
        "metric_definition": {  # 新增：标注指标定义（修复后）
            "overall_accuracy": "Exact match accuracy (numerical + unit completely correct)",
            "exact_accuracy": "Same as overall_accuracy (compatibility)",
            "partial_accuracy": "Partial match accuracy (numerical intersection >0 or reasoning node intersection >0)",
            "avg_f1_score": "Average numerical-level F1 score (adapted for geometry answers)",
            "avg_reasoning_completeness": "Average reasoning completeness (core reasoning node coverage ratio, only for samples with reasoning nodes)"
        },
        "detailed_gpu_results": results  # Detailed reasoning results for each sample
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # Print clean GeoQA3 exclusive evaluation results (修复后)
    print("\n" + "=" * 80)
    print("LLaVA-v1.5-7b Testing GeoQA3 Geometry VQA - GPU Optimized (Fixed Metrics) Final Results")
    print("=" * 80)
    print(f"Running Configuration: {DEVICE} | Batch={BATCH_SIZE} | Max_Tokens={MAX_NEW_TOKENS}")
    print(f"Overall Accuracy (Exact Match): {metrics['overall_accuracy']}%")
    print(f"Exact Accuracy (Numerical+Unit Match): {metrics['exact_accuracy']}%")
    print(f"Partial Accuracy (Real Match): {metrics['partial_accuracy']}%")
    print(f"Average F1 Score (Numerical-level): {metrics['avg_f1_score']} (0~1)")
    print(f"Average Reasoning Completeness: {metrics['avg_reasoning_completeness']} (0~1)")
    print("\n" + "=" * 80)
    print("Hallucination Classification Statistics (GeoQA3 Geometry Exclusive) (Count | Ratio)")
    print("=" * 80)
    for key, val in hallucination_stats.items():
        print(f"{key.ljust(45)}: {val['count']} entries | {val['ratio']}%")
    print("=" * 80)
    print(f"GeoQA3 Result File Saved To: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    try:
        # Execute LLaVA+GeoQA3 full process: load data → load model → GPU inference → calculate metrics → save results
        dataset = load_dataset(DATASET_PATH)
        if not dataset:
            raise ValueError("No valid GeoQA3 test data loaded, check directory/annotations/images")
        processor, model = load_llava_model(MODEL_PATH, DEVICE, torch_dtype)
        raw_results = batch_inference(processor, model, dataset, DEVICE, BATCH_SIZE)
        if not raw_results:
            raise ValueError("GPU inference failed, no valid GeoQA3 reasoning results generated")
        final_results, metrics, hallucination_stats = calculate_metrics_and_hallucination(raw_results)
        save_test_results(final_results, metrics, hallucination_stats)
        print(
            "\nLLaVA-v1.5-7b GPU Optimized Version (Fixed Metrics) Testing GeoQA3 Geometry VQA Full Process Completed!")
    except Exception as e:
        print(f"\nRuntime Error: {str(e)}")
        print("GeoQA3+GPU Troubleshooting Suggestions:")
        print("  1. Confirm nvidia-smi can detect GPU, CUDA version matches PyTorch")
        print("  2. Set BATCH_SIZE=1 for 16G VRAM, 2 for 24G to avoid out-of-memory")
        print("  3. Check GeoQA3 directory: must contain image and json subdirectories with same-name files")
        print("  4. Check json annotations: must contain subject (question) and answer (answer) fields")
        print("  5. Ensure correct model path and undamaged llava-v1.5-7b (re-download if needed)")