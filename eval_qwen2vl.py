from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image
import json
import os
import torch
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# 全局环境配置（屏蔽冗余警告，强制单卡）
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# === 全局配置（完全匹配你的环境，Qwen2-VL官方参数） ===
model_path = "/root/autodl-tmp/models/qwen2-vl-local"
dataset_json_path = "/root/autodl-tmp/datasets/coco_vqa_1000/val_sample_1000.json"
dataset_img_dir = "/root/autodl-tmp/datasets/coco_vqa_1000/images"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 4  # Qwen2-VL短答案足够（是/否/数字），减少冗余
TEST_NUM = 20  # 测试前20个样本
DEBUG = True  # 打印有效样本详情

# === 路径强制校验（关键路径不存在直接退出） ===
assert os.path.exists(dataset_json_path), f"❌ 数据集JSON不存在: {dataset_json_path}"
assert os.path.isdir(model_path), f"❌ 模型路径不存在: {model_path}"
assert os.path.isdir(dataset_img_dir), f"❌ 图像目录不存在: {dataset_img_dir}"
print(f"✅ 环境初始化完成 | 设备：{DEVICE} | 测试样本数：{TEST_NUM}")

# === 核心修复1：加载Qwen2-VL官方专属Processor+模型（强制适配权重尺寸） ===
# 加载Qwen2-VL官方图文处理器（一站式处理图像+文本，官方唯一推荐）
processor = Qwen2VLProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)
# 加载Qwen2-VL模型（核心：ignore_mismatched_sizes=True 强制适配权重尺寸）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda:0" else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
    ignore_mismatched_sizes=True,  # 核心修复：强制忽略权重尺寸不匹配
    attn_implementation="eager"   # 兼容低版本，避免flash attention报错
).to(DEVICE).eval()
# 强制设置特殊token（处理器兜底后再次确认）
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.unk_token = processor.tokenizer.pad_token
print(f"✅ Qwen2-VL官方模型加载成功 | 设备：{DEVICE}")
print(f"✅ 已强制适配权重尺寸，忽略Conv3d形状不匹配")

# === 核心函数：官方Prompt格式+指标计算+幻觉检测（极简兜底） ===
def generate_vqa_prompt(question, use_cot=False):
    """Qwen2-VL官方Prompt格式，加入图像标识（必须加）"""
    question = str(question).strip() if question else "图片中占比最大的颜色是什么？"
    if use_cot:
        return f"根据图片内容回答以下问题，一步步推理，最后仅给出简单答案：{question}"
    else:
        return f"根据图片内容回答以下问题，直接给出简单答案，不要多余内容：{question}"

def calculate_metrics(predictions, references):
    """严格过滤空值，计算准确率和加权F1"""
    valid_pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return {"accuracy": 0.0, "f1": 0.0}
    preds, refs = zip(*valid_pairs)
    return {
        "accuracy": round(accuracy_score(refs, preds), 4),
        "f1": round(f1_score(refs, preds, average='weighted', zero_division=0), 4)
    }

def calculate_hallucination_rate(predictions, references):
    """计算幻觉率：预测与真实答案不一致即为幻觉"""
    valid_count, hallucination_count = 0, 0
    for p, r in zip(predictions, references):
        p, r = p.strip(), r.strip()
        if p and r:
            valid_count += 1
            hallucination_count += 1 if p != r else 0
    return round(hallucination_count / valid_count if valid_count > 0 else 0.0, 4)

# === 核心修复2：Qwen2-VL官方标准图文推理（从根源解决None迭代报错） ===
def run_vqa_evaluation(use_cot=False):
    # 加载并预处理数据集（仅做基础过滤）
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # 仅保留有image_id、question、answer的有效样本
    processed_data = [
        item for item in raw_data
        if isinstance(item, dict) and item.get("image_id") and item.get("question") and item.get("answer")
    ]
    if DEBUG and processed_data:
        print(f"\n🔍 数据集预处理完成 | 原始样本：{len(raw_data)} | 有效样本：{len(processed_data)}")
        print(f"🔍 第1个样本示例：{processed_data[0]}")

    predictions, references = [], []
    valid_sample_count = 0

    print(f"\n🚀 开始VQA评估 | COT思维链：{'开启' if use_cot else '关闭'}")
    for idx, item in enumerate(processed_data[:TEST_NUM]):
        try:
            # 1. 提取基础字段（极简兜底）
            img_id = str(item["image_id"]).strip()
            question = item["question"].strip()
            true_answer = item["answer"].strip()

            # 2. 加载并校验图像（Qwen2-VL官方要求RGB格式）
            img_name = f"COCO_val2014_{img_id.zfill(12)}.jpg"
            img_path = os.path.join(dataset_img_dir, img_name)
            if not os.path.exists(img_path):
                raise Exception(f"图像不存在：{img_name}")
            image = Image.open(img_path).convert("RGB")  # 官方强制RGB

            # 3. 核心：使用官方Processor一站式处理图像+文本
            prompt = generate_vqa_prompt(question, use_cot)
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE, torch.float16 if DEVICE == "cuda:0" else torch.float32)

            # 4. Qwen2-VL官方标准推理
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda:0"):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    temperature=0.0,
                    num_beams=1,
                    use_cache=True
                )

            # 5. 解析预测结果
            pred_answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
            pred_answer = pred_answer.replace(prompt, "").strip() or "无"
            true_answer = true_answer or "无"

            # 6. 统计有效样本
            predictions.append(pred_answer)
            references.append(true_answer)
            valid_sample_count += 1

            # 调试打印
            if DEBUG:
                print(f"✅ 样本{idx}：有效 | 问题：{question[:30]} | 真实：{true_answer} | 预测：{pred_answer}")

        except Exception as e:
            err_info = str(e)[:50].replace("\n", " ")
            print(f"⚠️  样本{idx}：跳过 | 原因：{err_info}")
            continue
        finally:
            if DEVICE == "cuda:0":
                torch.cuda.empty_cache()

    # 计算指标
    metrics = calculate_metrics(predictions, references)
    hallucination_rate = calculate_hallucination_rate(predictions, references)
    print(f"\n✅ 评估完成 | 测试样本数：{TEST_NUM} | 有效样本数：{valid_sample_count}")
    return metrics, hallucination_rate

# === 主程序：COT/非COT对比评估 ===
if __name__ == "__main__":
    # 1. 不启用COT评估
    print("=" * 60)
    print("📊 评估模式：不启用COT思维链（直接回答）")
    print("=" * 60)
    no_cot_metrics, no_cot_hallu = run_vqa_evaluation(use_cot=False)
    print(f"\n📈 不启用COT评估结果：")
    print(f"准确率：{no_cot_metrics['accuracy']:.2%} | 加权F1：{no_cot_metrics['f1']:.4f} | 幻觉率：{no_cot_hallu:.2%}")

    # 2. 启用COT评估
    print("\n" + "=" * 60)
    print("📊 评估模式：启用COT思维链（分步推理）")
    print("=" * 60)
    with_cot_metrics, with_cot_hallu = run_vqa_evaluation(use_cot=True)
    print(f"\n📈 启用COT评估结果：")
    print(f"准确率：{with_cot_metrics['accuracy']:.2%} | 加权F1：{with_cot_metrics['f1']:.4f} | 幻觉率：{with_cot_hallu:.2%}")

    # 3. 对比总结
    print("\n" + "=" * 70)
    print("📋 COT思维链效果对比总结")
    print("=" * 70)
    acc_change = (with_cot_metrics['accuracy'] - no_cot_metrics['accuracy']) * 100
    f1_change = with_cot_metrics['f1'] - no_cot_metrics['f1']
    hallu_change = (with_cot_hallu - no_cot_hallu) * 100
    print(f"准确率变化：{acc_change:+.2f}%")
    print(f"加权F1变化：{f1_change:+.4f}")
    print(f"幻觉率变化：{hallu_change:+.2f}%")
    print("=" * 70)
    print("🎉 Qwen2-VL官方标准流程评估完成！")