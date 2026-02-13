import os
import json
import re
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import io
from sklearn.metrics import accuracy_score, f1_score
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPImageProcessor, CLIPVisionModel,
    BitsAndBytesConfig
)

# ===================== 环境配置 ======================
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ===================== 核心配置 ======================
MODEL_DIR = "/root/autodl-tmp/models/llava-med-v1.5"
CLIP_DIR = "/root/autodl-tmp/models/clip-vit-large-patch14-336"
PARQUET_PATHS = [
    "/root/autodl-tmp/datasets/mimic-cxr-dataset/train-00000-of-00002.parquet",
    "/root/autodl-tmp/datasets/mimic-cxr-dataset/train-00001-of-00002.parquet"
]
OUTPUT_DIR = "/root/autodl-tmp/datasets/mimic-cxr-dataset/llava_med_final_infer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 关键配置：固定参数，避免维度冲突
TARGET_SAMPLES = 1000  # 先跑1000个验证，稳定后改成整个数据集
IMAGE_EMBED_LEN = 256  # 固定图像embedding长度
VISION_FEATURE_DIM = 1024

# 8bit量化（最低资源消耗）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=TORCH_DTYPE
)


# ===================== 极简视觉投影层 ======================
class SimpleVisionProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 强制固定到256长度，彻底解决维度问题
        if x.shape[1] != IMAGE_EMBED_LEN:
            if x.shape[1] > IMAGE_EMBED_LEN:
                x = x[:, :IMAGE_EMBED_LEN, :]
            else:
                pad = torch.zeros((x.shape[0], IMAGE_EMBED_LEN - x.shape[1], x.shape[2]),
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        return self.norm(self.proj(x))


# ===================== 加载模型（极简版） ======================
def load_models():
    print("Loading LLaVA-Med (Simplified Version)...")

    # 1. CLIP（只负责提取特征，不参与token替换）
    vis_processor = CLIPImageProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    vis_model = CLIPVisionModel.from_pretrained(
        CLIP_DIR, torch_dtype=TORCH_DTYPE, device_map=DEVICE, local_files_only=True
    ).eval()

    # 2. Tokenizer（最简单配置）
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, use_fast=False, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. LLM（直接加载，不修改tokenizer）
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb_config, torch_dtype=TORCH_DTYPE,
        device_map="auto", trust_remote_code=True, local_files_only=True
    ).eval()

    # 4. 投影层（固定维度）
    hidden_dim = llm.config.hidden_size
    projector = SimpleVisionProjector(VISION_FEATURE_DIM, hidden_dim).to(DEVICE, TORCH_DTYPE)

    # 加载投影层权重（兼容任意格式）
    try:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(MODEL_DIR, "model-00001-of-00004.safetensors"), device=DEVICE)
    except:
        state_dict = torch.load(os.path.join(MODEL_DIR, "model-00001-of-00004.safetensors"),
                                map_location=DEVICE, weights_only=False)

    proj_dict = {k.replace("vision_projector.", ""): v for k, v in state_dict.items() if "vision_projector" in k}
    projector.load_state_dict(proj_dict, strict=False)

    print(f"✅ Models loaded! LLM hidden dim: {hidden_dim}")
    return tokenizer, llm, vis_processor, vis_model, projector


# ===================== 加载数据集 ======================
def load_data():
    print(f"\nLoading MIMIC-CXR (target: {TARGET_SAMPLES} samples)...")
    infer_data = []
    for parquet_path in PARQUET_PATHS:
        if not os.path.exists(parquet_path):
            print(f"⚠️  Skip missing file: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        for item in df.to_dict("records"):
            if len(infer_data) >= TARGET_SAMPLES:
                break
            if isinstance(item.get("image"), bytes) and len(item["image"]) > 100 and item.get("findings"):
                try:
                    img = Image.open(io.BytesIO(item["image"])).convert("RGB")
                    infer_data.append({
                        "image": img,
                        "findings": str(item["findings"]).strip()[:200]
                    })
                except:
                    continue
        del df
    print(f"✅ Loaded {len(infer_data)} valid samples")
    return infer_data


# ===================== 工具函数（精准版） ======================
def get_conclusion(report):
    """精准提取正常/异常结论"""
    if not report:
        return 0  # 默认正常

    report = report.lower()
    # 异常关键词（优先级更高）
    abnormal = ["effusion", "pneumothorax", "consolidation", "nodule", "edema",
                "pneumonia", "infiltrate", "atelectasis", "enlarged", "opacity"]
    # 正常关键词
    normal = ["normal", "clear", "unremarkable", "no acute", "no evidence"]

    if any(word in report for word in abnormal):
        return 1
    elif any(word in report for word in normal):
        return 0
    else:
        return 0  # 兜底


def judge_hallucination(ori, gen):
    if ori == gen:
        return 0, "Consistent"
    elif gen == 0 and ori == 1:
        return 1, "False Negative (Abnormal → Normal)"
    else:
        return 1, "False Positive (Normal → Abnormal)"


# ===================== 核心推理（LLaVA官方稳定方案） ======================
def infer_sample(sample, idx, tokenizer, llm, vis_processor, vis_model, projector):
    sample_id = f"sample_{idx:06d}"
    result = {"sample_id": sample_id, "status": "success", "error": "",
              "original": "", "generated": "", "ori_con": 0, "gen_con": 0,
              "is_hallu": 0, "hallu_type": ""}

    try:
        # 1. 原始报告处理
        ori_report = sample["findings"]
        result["original"] = ori_report
        result["ori_con"] = get_conclusion(ori_report)

        # 2. 图像特征提取（固定流程）
        img = sample["image"]
        img_input = vis_processor(images=img, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
        with torch.no_grad():
            img_feat = vis_model(**img_input).last_hidden_state
            img_embed = projector(img_feat)  # (1, 256, hidden_dim)

        # 3. 文本Prompt（极简，避免token冲突）
        prompt = f"Analyze this chest X-ray: {ori_report}\nConclusion: Normal or Abnormal?"
        text_input = tokenizer(prompt, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(DEVICE)

        # 4. 核心：图像embedding前置拼接（LLaVA官方最稳定方案）
        with torch.no_grad():
            text_embed = llm.get_input_embeddings()(text_input["input_ids"])
            # 拼接：图像embedding + 文本embedding
            full_embed = torch.cat([img_embed, text_embed], dim=1)
            # 注意力掩码：图像部分全1，文本部分用原掩码
            img_mask = torch.ones((1, IMAGE_EMBED_LEN), device=DEVICE)
            full_mask = torch.cat([img_mask, text_input["attention_mask"]], dim=1)

        # 5. 生成（极简参数，避免冲突）
        with torch.no_grad():
            output = llm.generate(
                inputs_embeds=full_embed,
                attention_mask=full_mask,
                max_new_tokens=100,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                num_beams=1  # 关闭束搜索，减少维度问题
            )

        # 6. 解析输出
        gen_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        gen_text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?()-]', '', gen_text)
        result["generated"] = gen_text
        result["gen_con"] = get_conclusion(gen_text)
        result["is_hallu"], result["hallu_type"] = judge_hallucination(result["ori_con"], result["gen_con"])

        if (idx + 1) % 10 == 0:
            print(f"🔹 Progress: {idx + 1}/{TARGET_SAMPLES} | {sample_id} | "
                  f"Ori: {'Abn' if result['ori_con'] else 'Nor'} | Gen: {'Abn' if result['gen_con'] else 'Nor'}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:100]
        print(f"❌ {sample_id} failed: {result['error']}")
    finally:
        with open(os.path.join(OUTPUT_DIR, f"{sample_id}.json"), "w") as f:
            json.dump(result, f, indent=2)
        torch.cuda.empty_cache()

    return result


# ===================== 案例分析 ======================
def analyze_results(all_results):
    print(f"\n" + "=" * 80)
    print(f"📊 Case Analysis (Total: {len(all_results)})")
    print(f"=" * 80)

    # 过滤有效结果
    valid = [r for r in all_results if r["status"] == "success"]
    correct = [r for r in valid if r["is_hallu"] == 0]
    error = [r for r in valid if r["is_hallu"] == 1]

    # 打印10个正确案例
    print(f"\n✅ Top 10 Correct Cases:")
    for i, case in enumerate(correct[:10], 1):
        print(f"\n{i}. ID: {case['sample_id']}")
        print(f"   Original: {case['original'][:80]}...")
        print(f"   Generated: {case['generated'][:80]}...")
        print(f"   Conclusion: {'Normal' if case['ori_con'] else 'Abnormal'} (Match)")

    # 打印10个错误案例
    print(f"\n❌ Top 10 Error Cases:")
    for i, case in enumerate(error[:10], 1):
        print(f"\n{i}. ID: {case['sample_id']} | Type: {case['hallu_type']}")
        print(f"   Original: {case['original'][:80]}...")
        print(f"   Generated: {case['generated'][:80]}...")
        print(
            f"   Ori: {'Normal' if case['ori_con'] else 'Abnormal'} | Gen: {'Normal' if case['gen_con'] else 'Abnormal'}")

    # 统计
    if valid:
        acc = accuracy_score([r["ori_con"] for r in valid], [r["gen_con"] for r in valid])
        hallu_rate = len(error) / len(valid)
        print(f"\n📈 Metrics:")
        print(f"   Success Rate: {len(valid) / len(all_results) * 100:.1f}%")
        print(f"   Accuracy: {acc * 100:.1f}%")
        print(f"   Hallucination Rate: {hallu_rate * 100:.1f}%")


# ===================== 主函数 ======================
def main():
    # 加载模型和数据
    tokenizer, llm, vis_processor, vis_model, projector = load_models()
    infer_data = load_data()
    if not infer_data:
        print("❌ No data loaded!")
        return

    # 推理
    print(f"\nStarting inference...")
    all_results = []
    for idx, sample in enumerate(infer_data):
        res = infer_sample(sample, idx, tokenizer, llm, vis_processor, vis_model, projector)
        all_results.append(res)

    # 统计和分析
    success = len([r for r in all_results if r["status"] == "success"])
    print(f"\n" + "=" * 80)
    print(f"Inference Done! Success: {success}/{len(all_results)}")
    print(f"=" * 80)

    # 案例分析
    analyze_results(all_results)


if __name__ == "__main__":
    main()
