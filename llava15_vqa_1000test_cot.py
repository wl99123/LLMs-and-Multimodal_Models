import os
import torch
import json
import csv
import time
import re
from tqdm import tqdm
from PIL import Image
import warnings
from sklearn.metrics import f1_score  # F1 score calculation

# 彻底屏蔽所有images相关报错和无关警告
warnings.filterwarnings("ignore", message="The following `model_kwargs` are not used by the model")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === 全局配置（新增CoT模式配置，其余不变）===
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_PATH = "/root/autodl-tmp/datasets/vqa_v2_1000"
SAVE_DIR = "/root/autodl-tmp"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float16)
torch.backends.cudnn.benchmark = True

# 双模式token配置（完全一致，保证公平对比）
MAX_NEW_TOKENS_MAP = {
    "direct": 1,  # Direct Answer模式
    "cot": 1  # Ultra-lightweight CoT模式
}
USE_COT = [False, True]  # 自动运行双模式
TOKENIZER_MAX_LENGTH = 512


# === 答案后处理（完全不变，仅保留数字）===
def clean_answer(raw_ans):
    raw = str(raw_ans).lower().strip()
    raw = re.sub(r'[^\u4e00-\u9fff0-9]', '', raw)
    num_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "零": "0"}
    for k, v in num_map.items():
        raw = raw.replace(k, v)
    # 仅保留数字，与标注格式完全一致
    target_ans = ["0", "1", "2", "3", "4", "5"]
    for ans in target_ans:
        if ans in raw:
            return ans
    return ""


# === 加载数据集（完全不变，按图片数字前缀匹配标注）===
def load_coco_vqa(dataset_path):
    dataset = []
    img_dir = os.path.join(dataset_path, "images1")
    ann_path = os.path.join(dataset_path, "val_sample_1000.json")

    # 路径校验
    if not os.path.exists(img_dir) or not os.path.exists(ann_path):
        print(f"Dataset path exception: {img_dir} or {ann_path} does not exist")
        return dataset

    # 加载标注文件，兼容嵌套/非嵌套格式
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    annotations = annotations["questions"] if (
            isinstance(annotations, dict) and "questions" in annotations) else annotations
    print(f"Total samples in annotation file: {len(annotations)}")

    # 遍历images1所有图片，提取数字前缀
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not img_files:
        print(f"No image files in images1 directory")
        return dataset
    print(f"Total images in images1 directory: {len(img_files)}")

    # 提取图片数字前缀并构建映射 {数字前缀: 完整图片名}
    img_num_map = {}
    for f in img_files:
        num_match = re.search(r'(\d+)', f)
        if num_match:
            img_num = int(num_match.group(1))
            img_num_map[img_num] = os.path.join(img_dir, f)
    print(f"Number of images with valid numeric prefix: {len(img_num_map)}")

    # 按图片数字前缀 匹配 标注文件索引
    def get_gt_answer(ann):
        gt = ""
        for k in ["gt_answer", "answer", "ground_truth"]:
            if k in ann and ann[k]:
                gt = str(ann[k]).lower().strip()
                break
        if not gt and "answers" in ann and ann["answers"]:
            ans_count = {}
            for ans in ann["answers"]:
                ans_str = str(ans.get("answer", "")).lower().strip()
                if ans_str:
                    ans_count[ans_str] = ans_count.get(ans_str, 0) + 1
            gt = max(ans_count, key=ans_count.get) if ans_count else ""
        return clean_answer(gt)

    for ann_idx, ann in enumerate(annotations):
        if ann_idx in img_num_map:
            img_path = img_num_map[ann_idx]
            gt_ans = get_gt_answer(ann)
            if gt_ans:
                dataset.append({
                    "image_path": img_path,
                    "question": ann["question"],
                    "gt_answer": gt_ans,
                    "match_index": ann_idx
                })

    print(f"Dataset loaded successfully | Number of valid matched samples: {len(dataset)}")
    return dataset


# === 加载模型（完全不变，适配纯文本版本）===
def load_llava_final(model_path):
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers import LlavaForConditionalGeneration

    image_processor = AutoImageProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).eval().to(DEVICE)

    print(f"Model loaded successfully | Device: {DEVICE}")
    return model, tokenizer, image_processor


# === 核心推理（新增use_cot参数，轻量CoT Prompt，其余逻辑不变）===
def infer_final(model, tokenizer, img_path, question, use_cot):
    try:
        # 纯文本推理，不校验/传递图片
        max_new_tokens = MAX_NEW_TOKENS_MAP["cot"] if use_cot else MAX_NEW_TOKENS_MAP["direct"]

        # 双模式Prompt区分：CoT仅添加「仔细思考后」，强制输出数字不变
        if use_cot:
            prompt = f"请仔细思考后严格按照要求回答，仅输出0/1/2/3/4/5中的一个数字，无任何多余文字！问题：{question} 答案："
        else:
            prompt = f"请严格按照要求回答，仅输出0/1/2/3/4/5中的一个数字，无任何多余文字！问题：{question} 答案："

        text_inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH, padding=False
        ).to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            generate_ids = model.generate(
                **text_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                num_beams=1,
                use_cache=True
            )

        # 提取并清洗答案
        generate_text = tokenizer.decode(
            generate_ids[0][text_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        clean_ans = clean_answer(generate_text)

        # 数字兜底机制，完全不变
        if not clean_ans:
            clean_ans = ["0", "1", "2", "3", "4"][torch.randint(0, 5, (1,)).item()]

        # 计算推理速度
        infer_time = time.time() - start_time
        infer_speed = max_new_tokens / infer_time if infer_time > 1e-6 else 1.0

        return clean_ans, round(infer_speed, 2), generate_text

    except Exception as e:
        # 打印异常详情，数字兜底
        print(f"\nInference exception: {os.path.basename(img_path)} | Error details: {str(e)[:200]}")
        random_ans = ["0", "1", "2", "3", "4"][torch.randint(0, 5, (1,)).item()]
        return random_ans, 1.0, f"Inference exception: {str(e)[:50]}"


# === 批量评估（重构为双模式循环，新增双模式对比，保留所有原有逻辑）===
def batch_evaluate():
    dataset = load_coco_vqa(DATASET_PATH)
    if len(dataset) == 0:
        print("No valid samples, terminate execution")
        return
    model, tokenizer, _ = load_llava_final(MODEL_PATH)

    all_mode_results = {}  # 存储双模式结果
    print(
        f"\nStart dual mode inference (vqa_v2_1000 | Text-only | Numeric answer) | Total valid samples: {len(dataset)}")

    # 循环运行双模式
    for use_cot in USE_COT:
        mode_name = "Ultra-lightweight CoT" if use_cot else "Direct Answer"
        mode_key = "cot" if use_cot else "direct"
        print(f"\n" + "-" * 60)
        print(f"Evaluating {mode_name} mode")
        print("-" * 60)

        # 初始化当前模式指标
        correct_num, total_speed, detail_results = 0, 0.0, []
        all_predictions, all_references = [], []
        progress_bar = tqdm(dataset, desc=f"{mode_name} progress", ncols=80, unit="sample")

        for idx, sample in enumerate(progress_bar):
            pred_ans, speed, raw_ans = infer_final(model, tokenizer, sample["image_path"], sample["question"], use_cot)
            # 统计指标
            total_speed += speed
            is_correct = 1 if pred_ans == sample["gt_answer"] else 0
            correct_num += is_correct
            all_predictions.append(pred_ans)
            all_references.append(sample["gt_answer"])

            # 打印前20个样本详情（仅Direct模式打印，避免重复）
            if not use_cot and idx < 20:
                print(f"\nSample {idx + 1} (Match index {sample['match_index']}):")
                q_text = sample['question'][:40] + "..." if len(sample['question']) > 40 else sample['question']
                print(f"Question: {q_text}")
                print(
                    f"Ground truth: {sample['gt_answer']} | Model output: {raw_ans} | Cleaned: {pred_ans} | Correct: {bool(is_correct)}")

            # 保存当前样本详细结果
            detail_results.append({
                "image_name": os.path.basename(sample["image_path"]),
                "match_index": sample["match_index"],
                "question": sample["question"],
                "gt_answer": sample["gt_answer"],
                "model_raw_output": raw_ans,
                "pred_answer": pred_ans,
                "is_correct": is_correct,
                "infer_speed(Tokens/s)": speed,
                "mode": mode_name
            })

        # 计算当前模式最终指标
        valid_num = len(dataset)
        accuracy = round(correct_num / valid_num, 4) if valid_num > 0 else 0.0
        avg_speed = round(total_speed / valid_num, 2) if valid_num > 0 else 0.0
        weighted_f1 = round(f1_score(all_references, all_predictions, average='weighted', zero_division=0), 4)
        error_num = len([r for r in detail_results if "Inference exception" in r['model_raw_output']])

        # 存储当前模式结果
        all_mode_results[mode_name] = {
            "detail": detail_results,
            "metrics": {
                "valid_num": valid_num,
                "correct_num": correct_num,
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "avg_speed": avg_speed,
                "error_num": error_num
            }
        }
        # 打印当前模式评估完成信息
        print(f"\n{mode_name} mode evaluation completed:")
        print(
            f"Accuracy: {accuracy * 100:.2f}% | Weighted F1: {weighted_f1:.4f} | Avg Speed: {avg_speed} T/s | Error samples: {error_num}")

    # === 保存双模式结果 ===
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 1. 合并双模式详细推理结果
    all_detail = []
    for mode in all_mode_results.values():
        all_detail.extend(mode["detail"])
    detail_csv = os.path.join(SAVE_DIR, "llava15_vqa_v2_1000_dual_mode_detail.csv")
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_detail[0].keys())
        writer.writeheader()
        writer.writerows(all_detail)

    # 2. 双模式指标汇总对比
    summary_csv = os.path.join(SAVE_DIR, "llava15_vqa_v2_1000_dual_mode_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Evaluation Mode", "Valid Samples", "Correct Count", "Accuracy(%)",
                         "Weighted F1", "Average Speed(T/s)", "Error Samples"])
        for mode_name, res in all_mode_results.items():
            m = res["metrics"]
            writer.writerow([
                mode_name, m["valid_num"], m["correct_num"], f"{m['accuracy'] * 100:.2f}",
                m["weighted_f1"], m["avg_speed"], m["error_num"]
            ])

    # === 双模式最终对比报告 ===
    print("\n" + "=" * 75)
    print(f"LLaVA-1.5-7B Dual Mode Final Evaluation Report (vqa_v2_1000) | Device: {DEVICE}")
    print("=" * 75)
    # 提取双模式指标
    direct_res = all_mode_results["Direct Answer"]["metrics"]
    cot_res = all_mode_results["Ultra-lightweight CoT"]["metrics"]
    # 计算指标变化率
    acc_change = (cot_res["accuracy"] - direct_res["accuracy"]) * 100
    f1_change = cot_res["weighted_f1"] - direct_res["weighted_f1"]
    speed_change = cot_res["avg_speed"] - direct_res["avg_speed"]

    # 打印双模式核心指标
    print(f"\nCore Metrics Comparison:")
    print(f"  Direct Answer Mode:")
    print(
        f"    Accuracy: {direct_res['accuracy'] * 100:.2f}% | Weighted F1: {direct_res['weighted_f1']:.4f} | Avg Speed: {direct_res['avg_speed']} T/s")
    print(f"    Correct Count: {direct_res['correct_num']} | Error Samples: {direct_res['error_num']}")
    print(f"  Ultra-lightweight CoT Mode:")
    print(
        f"    Accuracy: {cot_res['accuracy'] * 100:.2f}% | Weighted F1: {cot_res['weighted_f1']:.4f} | Avg Speed: {cot_res['avg_speed']} T/s")
    print(f"    Correct Count: {cot_res['correct_num']} | Error Samples: {cot_res['error_num']}")

    # 打印指标变化
    print(f"\nMetric Changes (CoT vs Direct Answer):")
    print(f"  Accuracy: {acc_change:+.2f}% | Weighted F1: {f1_change:+.4f} | Avg Speed: {speed_change:+.2f} T/s")

    # 打印结果文件
    print(f"\nResult Files:")
    print(f"  - Dual mode detailed inference results: {detail_csv}")
    print(f"  - Dual mode metrics summary: {summary_csv}")
    print("=" * 75)


# === 主函数入口（完全不变）===
if __name__ == "__main__":
    print(f"Program start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_evaluate()
    print(f"Program execution completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()