import os
import torch
import json
import re
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# 离线环境配置（屏蔽所有日志，专注推理）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LOCAL_FILES_ONLY"] = "1"
os.environ["TORCH_NO_WARNINGS"] = "1"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

# 固定路径（确保正确，无需修改）
MODEL_PATH = "/root/autodl-tmp/models/qwen2-vl-local"
IMG_DIR = "/root/autodl-tmp/datasets/mathv_3040/images"
ANN_PATH = "/root/autodl-tmp/datasets/mathv_3040/annotations.json"

# 核心配置（极简，完全适配Qwen2-VL）
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
GEN_MAX_LEN = 2  # 适配数字/字母GT，足够且不冗余
PROMPT = "直接回答，仅输出数字或大写字母，不要其他内容。"

# 答案后处理（极简高效，无兜底，保留模型真实输出）
def clean_answer(s):
    if not s or s.strip() == "":
        return ""
    s = str(s).strip().upper()
    res = re.findall(r'[0-9A-Z]+', s)  # 提取数字/字母（单/多字符均适配）
    return res[0] if res else ""

# 加载数据集（前50条测试，一键切全量，强化校验）
def load_dataset():
    dataset = []
    try:
        with open(ANN_PATH, "r", encoding="utf-8") as f:
            anns = json.load(f)[:50]  # 测试前50条，快出结果
            # anns = json.load(f)  # 全量3040条，测试成功后打开这行
    except Exception as e:
        print(f"❌ 标注文件错误：{str(e)[:30]}")
        return []
    for idx, a in enumerate(anns, 1):
        img_path = os.path.join(IMG_DIR, a["image_name"])
        if os.path.exists(img_path) and img_path.lower().endswith(('.jpg','.jpeg','.png')):
            dataset.append({
                "img_path": img_path,
                "question": a["question"],
                "gt": a["gt"] if "gt" in a else a["answer"],
                "idx": idx
            })
    print(f"✅ 加载 {len(dataset)} 条有效样本 | 设备：{DEVICE}")
    return dataset

# 加载模型+处理器（4.57.6专属，无任何冗余配置）
def load_model():
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    # 加载专属处理器（原生配置，无冲突）
    processor = Qwen2VLProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"✅ Qwen2-VL处理器加载完成")
    # 加载专属模型（极简配置，适配RTX4090）
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).eval()
    print(f"✅ Qwen2-VL模型加载完成 | 4.57.6 无冲突")
    return model, processor

# 核心推理（彻底移除所有冗余参数，原生纯推理，无任何错误！）
def infer(model, processor, sample):
    try:
        # 1. 安全加载图像
        with Image.open(sample["img_path"]) as f:
            image = f.convert("RGB")
        # 2. 构造原生输入（处理器自动生成所有必需参数，无手动干预）
        prompt = f"{sample['question']} {PROMPT}"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
        # 3. 模型纯推理（仅保留核心有效参数，彻底移除所有冗余！）
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_LEN,  # 仅生成指定长度
                do_sample=False,  # 贪心搜索，无随机
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        # 4. 解码真实输出（仅提取模型生成部分，无兜底）
        gen_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        pred = clean_answer(raw)
        return pred, raw
    except Exception as e:
        err = str(e)[:30].replace("\n","").replace(" ","")
        return "", f"err:{err}"

# 主函数（彩色打印，统计真实准确率，无任何人工干预）
if __name__ == "__main__":
    print("="*90)
    print("🔴 Qwen2-VL 最终纯推理版 | 4.57.6 | 零冗余 | 无兜底/不抄GT")
    print("="*90)
    torch.cuda.empty_cache()
    # 加载数据和模型
    data = load_dataset()
    if not data: exit()
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)[:50]}")
        exit()
    # 批量推理
    print("\n🚀 开始纯推理...（无兜底，模型自主输出，有对有错）")
    total, correct = len(data), 0
    show_num = 20  # 打印前20条结果
    for s in data:
        pred, raw = infer(model, processor, s)
        if pred and pred == s["gt"]:
            correct += 1
        # 彩色打印
        if s["idx"] <= show_num:
            if pred and pred == s["gt"]:
                print(f"\033[32m样本{s['idx']:2d} | GT:{s['gt']:3s} | PRED:{pred:3s} | RAW:{raw[:8]} ✅\033[0m")
            else:
                print(f"\033[31m样本{s['idx']:2d} | GT:{s['gt']:3s} | PRED:{pred:3s} | RAW:{raw[:15]} ❌\033[0m")
    # 最终统计
    acc = (correct/total)*100 if total>0 else 0.0
    torch.cuda.empty_cache()
    print("="*90)
    print(f"\033[34m🔴 推理完成 | 总{total}条 | 正确{correct}条 | 真实准确率：{acc:.1f}%\033[0m")
    print("="*90)
    print("💡 切全量：将load_dataset中 anns = json.load(f)[:50] 改为 anns = json.load(f)")
    print("💡 结果说明：准确率非100%为模型真实能力，无任何人工兜底/抄GT！")