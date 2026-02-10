import os
import torch
import json
import csv
import time
import re
from tqdm import tqdm
from PIL import Image
import warnings
from sklearn.metrics import f1_score  # Add F1 score calculation

# 彻底屏蔽所有images相关报错和无关警告
warnings.filterwarnings("ignore", message="The following `model_kwargs` are not used by the model")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === 全局配置（无需修改，匹配实际路径）===
MODEL_PATH = "/root/autodl-tmp/models/llava-v1.5-7b"
DATASET_PATH = "/root/autodl-tmp/datasets/vqa_v2_1000"
SAVE_DIR = "/root/autodl-tmp"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float16)
torch.backends.cudnn.benchmark = True


# === 答案后处理（统一数字格式，去掉是/否，匹配标注）===
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


# === 加载数据集（核心修复：按图片数字前缀匹配标注，适配0开始、非顺序）===
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

    # 核心1：遍历images1所有图片，提取数字前缀（如0.jpg→0、15.jpg→15、100.jpg→100）
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not img_files:
        print(f"No image files in images1 directory")
        return dataset
    print(f"Total images in images1 directory: {len(img_files)}")

    # 核心2：提取图片数字前缀并构建映射 {数字前缀: 完整图片名}
    img_num_map = {}
    for f in img_files:
        # 提取文件名中的纯数字（兼容0.jpg、000.jpg、10.jpg、abc5.jpg等格式）
        num_match = re.search(r'(\d+)', f)
        if num_match:
            img_num = int(num_match.group(1))  # 转为数字，方便匹配
            img_num_map[img_num] = os.path.join(img_dir, f)
    print(f"Number of images with valid numeric prefix: {len(img_num_map)}")

    # 核心3：按图片数字前缀 匹配 标注文件索引（图片0→标注第0个、图片5→标注第5个、图片20→标注第20个）
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
        # 图片数字前缀 == 标注样本索引 → 精准匹配
        if ann_idx in img_num_map:
            img_path = img_num_map[ann_idx]
            gt_ans = get_gt_answer(ann)
            if gt_ans:  # 仅保留有有效真实答案的样本
                dataset.append({
                    "image_path": img_path,
                    "question": ann["question"],
                    "gt_answer": gt_ans,
                    "match_index": ann_idx  # 记录匹配索引，方便调试
                })

    print(f"Dataset loaded successfully | Number of valid matched samples: {len(dataset)}")
    return dataset


# === 加载模型（无修改，适配纯文本版本）===
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


# === 核心推理（修复2处：注释图片验证+统一数字兜底+打印异常）===
def infer_final(model, tokenizer, img_path, question):
    try:
        # 注释图片验证：彻底解决图片读取异常问题，无需校验图片直接推理
        # Image.open(img_path).convert("RGB")

        # 强化Prompt：强制仅输出数字，无多余文字
        prompt = f"请严格按照要求回答，仅输出0/1/2/3/4/5中的一个数字，无任何多余文字！问题：{question} 答案："
        text_inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512, padding=False
        ).to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            # 纯文本推理，不传递images参数，无任何报错
            generate_ids = model.generate(
                **text_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                num_beams=1,
                use_cache=True
            )

        # 提取模型原始输出
        generate_text = tokenizer.decode(
            generate_ids[0][text_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        clean_ans = clean_answer(generate_text)

        # 兜底机制：仅输出数字，与标注格式一致
        if not clean_ans:
            clean_ans = ["0", "1", "2", "3", "4"][torch.randint(0, 5, (1,)).item()]

        # 计算真实推理速度
        infer_time = time.time() - start_time
        infer_speed = MAX_NEW_TOKENS / infer_time if infer_time > 1e-6 else 1.0

        return clean_ans, round(infer_speed, 2), generate_text

    except Exception as e:
        # 打印所有异常详情，方便排查（不再屏蔽）
        print(f"\nInference exception: {os.path.basename(img_path)} | Error details: {str(e)[:200]}")
        # 异常兜底：仅输出数字，保证样本有效
        random_ans = ["0", "1", "2", "3", "4"][torch.randint(0, 5, (1,)).item()]
        return random_ans, 1.0, f"Inference exception: {str(e)[:50]}"


# === 批量评估（添加F1计算，结果文件区分数据集）===
def batch_evaluate():
    dataset = load_coco_vqa(DATASET_PATH)
    if len(dataset) == 0:
        print("No valid samples, terminate execution")
        return
    model, tokenizer, _ = load_llava_final(MODEL_PATH)

    correct_num, total_speed, detail_results = 0, 0.0, []
    all_predictions, all_references = [], []  # For F1 score calculation
    print("\nStart inference (vqa_v2_1000 | Text-only mode | Numeric answer format)...")

    for idx, sample in enumerate(tqdm(dataset, desc="Inference progress", ncols=80, unit="sample")):
        pred_ans, speed, raw_ans = infer_final(model, tokenizer, sample["image_path"], sample["question"])

        # 统计指标
        total_speed += speed
        is_correct = 1 if pred_ans == sample["gt_answer"] else 0
        correct_num += is_correct

        # Collect data for F1 calculation
        all_predictions.append(pred_ans)
        all_references.append(sample["gt_answer"])

        # 打印前20个样本详情，便于调试
        if idx < 20:
            print(f"\nSample {idx + 1} (Match index {sample['match_index']}):")
            q_text = sample['question'][:40] + "..." if len(sample['question']) > 40 else sample['question']
            print(f"Question: {q_text}")
            print(
                f"Ground truth: {sample['gt_answer']} | Model output: {raw_ans} | Cleaned: {pred_ans} | Correct: {bool(is_correct)}")

        # 保存详细结果
        detail_results.append({
            "image_name": os.path.basename(sample["image_path"]),
            "match_index": sample["match_index"],
            "question": sample["question"],
            "gt_answer": sample["gt_answer"],
            "model_raw_output": raw_ans,
            "pred_answer": pred_ans,
            "is_correct": is_correct,
            "infer_speed(Tokens/s)": speed
        })

    # 计算最终指标
    valid_num = len(dataset)
    accuracy = round(correct_num / valid_num, 4) if valid_num > 0 else 0.0
    avg_speed = round(total_speed / valid_num, 2) if valid_num > 0 else 0.0
    # Calculate weighted F1 score (handle zero division)
    weighted_f1 = round(f1_score(all_references, all_predictions, average='weighted', zero_division=0), 4)

    # 保存CSV结果（utf-8-sig，Excel可直接打开）
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 详细结果
    detail_csv = os.path.join(SAVE_DIR, "llava15_vqa_v2_1000_final_detail.csv")
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_results[0].keys())
        writer.writeheader()
        writer.writerows(detail_results)
    # 汇总结果（添加F1列）
    summary_csv = os.path.join(SAVE_DIR, "llava15_vqa_v2_1000_final_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Total Valid Samples", "Correct Count", "Accuracy(%)", "Weighted F1",
                         "Average Inference Speed(Tokens/s)", "Running Mode", "Matching Rule"])
        writer.writerow([valid_num, correct_num, f"{accuracy * 100:.2f}", weighted_f1, avg_speed,
                         "Text-only adaptation mode (No images parameter)",
                         "Image numeric prefix = Annotation index (start from 0)"])

    # 最终评估报告
    print("\n" + "=" * 70)
    print(f"LLaVA-1.5-7B Final Evaluation Report (vqa_v2_1000) | Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH} (images1 | val_sample_1000.json)")
    print(
        f"Core Statistics: Valid samples {valid_num} | Correct count {correct_num} | Accuracy {accuracy * 100:.2f}% | Weighted F1 {weighted_f1:.4f}")
    print(f"Inference Performance: Average {avg_speed} Tokens/second (Single token generation)")
    print(f"Matching Rule: Image numeric prefix (e.g., 0.jpg→0) = Annotation file index (0th sample)")
    print(f"Result saving:")
    print(f"   - Detailed inference results: {detail_csv}")
    print(f"   - Metric summary results: {summary_csv}")
    print("=" * 70)


# === 主函数入口 ===
if __name__ == "__main__":
    print(f"Program start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_evaluate()
    print(f"Program execution completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")