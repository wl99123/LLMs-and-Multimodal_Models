# LLMs-and-Multimodal_Models
<p align="center">
  <h2 align="center"><strong>LLMs-and-Multimodal_Models: A Comprehensive Evaluation Framework for Hallucination Detection in Multimodal Large Language Models</strong></h2>
</p>

<div align="center">
<h5>
<em>Comprehensive Evaluation of LLMs and Multimodal Models on Visual QA, Reasoning, and Medical Imaging Tasks</em>
</h5>
<h5 align="center">
<a href="https://github.com/your-username/LLMs-and-Multimodal_Models"> <img src="https://img.shields.io/badge/GitHub-Repository-green?logo=GitHub"></a>
<a href="https://huggingface.co/datasets/MiliLab"> <img src="https://img.shields.io/badge/HuggingFace-Datasets-blue?logo=HuggingFace"></a>
<a href="https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=PyTorch"></a>
</h5>
</div>

<figure>
<div align="center">
<img src="figs/overview.png" width="80%">
</div>
<div align="center">
<figcaption align = "center"><b>Figure 1: Overview of the LLMs-and-Multimodal_Models evaluation framework, covering VQA, visual-text reasoning, cross-lingual reasoning, and medical imaging tasks.
 </b></figcaption>
</div>
</figure>

# 🐯 Contents
- [🔥 Update](#-update)
- [🌞 Intro](#-intro)
- [🔍 Overview](#-overview)
- [📖 Datasets](#-datasets)
- [🔧 Environment Preparation](#-environment-preparation)
- [🚀 Quick Start](#-quick-start)
- [🙏 Acknowledgements](#-acknowledgements)

# 🔥 Update
**2026.02.10**
- Released the full evaluation framework for multimodal models, covering VQA, mathematical/geometric reasoning, and medical imaging tasks.
- Added support for BLIP2-OPT-2.7b, LLaVA-v1.5-7b, Qwen2.5-VL-7B, and medical-adapted LLaVA-MED-v1.5.

**2026.02.05**
- Initial release of the hallucination detection metrics and core evaluation scripts for VQA tasks.

**2026.01.20**
- Completed dataset integration (COCO, VQAv2, GeoQA, MathVision, MIMIC-CXR, etc.).

# 🌞 Intro
LLMs-and-Multimodal_Models is a comprehensive evaluation framework designed to assess the performance and hallucination detection capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse task scenarios. 

This framework supports systematic evaluation on:
- General visual question answering (VQA) tasks
- Mathematical/geometric visual-text reasoning tasks
- OCR-based logical reasoning tasks
- Cross-lingual & multimodal reasoning tasks
- Medical imaging interpretation tasks (chest X-ray diagnosis)

It provides standardized metrics (accuracy, F1 score, hallucination rate, false negative/positive rates) and reproducible evaluation scripts to facilitate research on multimodal model reliability and reasoning capabilities.

# 🔍 Overview
The framework is structured to evaluate multimodal models from four core dimensions:
1. **General VQA**: Basic visual understanding and answer accuracy on COCO/VQAv2 benchmarks
2. **Mathematical/Geometric Reasoning**: Complex visual-text fusion reasoning on GeoQA/MathVision
3. **Advanced Reasoning**: OCR-based logical reasoning (LogicOCR) and cross-lingual reasoning (XLRS-Bench)
4. **Domain-Specific Evaluation**: Medical imaging interpretation on MIMIC-CXR dataset with clinical metrics

Key features:
- Support for mainstream open-source multimodal models (LLaVA, BLIP2, MiniGPT-4, Qwen2.5-VL, GeoLLaVA)
- Customizable hallucination detection metrics and classification dimensions
- Domain-specific evaluation for medical imaging (focus on clinical reliability metrics)
- Reproducible evaluation pipelines with detailed logging and result analysis

# 📖 Datasets
The framework integrates multiple benchmark datasets covering general and domain-specific multimodal tasks:

| Dataset          | Task Type                  | Language | Use Case                                  |
|------------------|----------------------------|----------|-------------------------------------------|
| COCO             | General VQA                | English  | Basic visual understanding evaluation     |
| VQAv2            | General VQA                | English  | Standard VQA benchmark                    |
| GeoQA            | Geometric Reasoning        | English  | Visual geometric reasoning evaluation     |
| MathVision       | Mathematical Reasoning     | English  | Visual mathematical reasoning evaluation  |
| LogicOCR         | OCR-based Logical Reasoning| Multilingual | Complex text-image reasoning        |
| XLRS-Bench       | Cross-lingual Reasoning    | Multilingual | Cross-lingual multimodal evaluation  |
| MIMIC-CXR        | Medical Imaging            | English  | Chest X-ray diagnosis & hallucination detection |

### Dataset Download & Structure
All datasets are organized under the `datasets/` directory with standardized structure:
```bash

datasets/
├── coco
├── GeoQA3
├── LogicOCR
├── MathVision
├── mimic-cxr-dataset
├── VQAv2
└── XLRS-Bench
   
```
Download links for raw datasets:
- COCO: [https://huggingface.co/datasets/jxie/coco_captions](https://huggingface.co/datasets/jxie/coco_captions)
- VQAv2: [https://huggingface.co/datasets/lmms-lab/VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2)
- GeoQA: [https://huggingface.co/datasets/leonardPKU/GEOQA_8K_R1V/viewer](https://huggingface.co/datasets/leonardPKU/GEOQA_8K_R1V/viewer)
- MathVision: [https://huggingface.co/datasets/MathLLMs/MathVision/tree/main](https://huggingface.co/datasets/MathLLMs/MathVision/tree/main)
- LogicOCR: [https://huggingface.co/datasets/MiliLab/LogicOCR](https://huggingface.co/datasets/MiliLab/LogicOCR)
- XLRS-Bench: [https://huggingface.co/datasets/initiacms/XLRS-Bench-lite/tree/main](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite/tree/main)
- MIMIC-CXR: [https://huggingface.co/datasets/itsanmolgupta/mimic-cxr-dataset](https://huggingface.co/datasets/itsanmolgupta/mimic-cxr-dataset)

# 🔧 Environment Preparation
## 📋 Hardware Requirements
- GPU: A100-48G (recommended for efficient evaluation of 7B-scale models)
- CUDA: 12.1 (required for GPU acceleration)

## 📁 Clone the Repository

git clone https://github.com/your-username/LLMs-and-Multimodal_Models.git
cd LLMs-and-Multimodal_Models

```bash
models/
├── blip2-opt-2.7b
├── clip-vit-large-patch14
├── GeoLLaVA-8K
├── llava-med-v1.5
├── llava-v1.5-7b
├── MiniGpt-4-7B
└── qwen2-vl-local
   
```
Download links for pre-trained models:
- BLIP2-OPT-2.7b: [https://huggingface.co/Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- LLaVA-v1.5-7b: [https://huggingface.co/liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- LLaVA-MED-v1.5: [https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)
- MiniGpt-4-7B: [https://www.modelscope.cn/models/alv001/MiniGpt-4-7B](https://www.modelscope.cn/models/alv001/MiniGpt-4-7B)
- CLIP-ViT-Large-Patch14: [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- Qwen2.5-VL-7B: [https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- GeoLLaVA-8K: [https://huggingface.co/initiacms/GeoLLaVA-8K](https://huggingface.co/initiacms/GeoLLaVA-8K)

🧱 Setup Conda Environment
Create & Activate Environment
```bash
# Create conda environment with Python 3.9.19
conda create -n mllm python=3.9.19
conda activate mllm

# Install dependencies
pip install -r requirements.txt
```
🚀 Quick Start
1. Hallucination Detection Metric Design
Design custom hallucination detection metrics and classification dimensions:
```bash
   python hallucination_evaluation_cot.py
```
2. General-Scenario VQA Task Evaluation
Evaluate basic visual understanding and answer accuracy on COCO/VQAv2:
```bash
# BLIP2-OPT-2.7b on VQAv2 (1000 samples)
python blip2_opt27b_vqa_test1000.py
python blip2_opt27b_vqa_test1000_COT.py

# BLIP2-OPT-2.7b on COCO
python blip2_opt27b_vqa_test_coco.py
python blip2_opt27b_vqa_test_coco_COT.py

# LLaVA-v1.5-7b on COCO and VQAv2
python llava15_coco_vqa_test.py
python llava15_vqa_1000test.py
python llava15_vqa_1000test_cot.py
python llava15_coco_vqa_test_cot.py

# Qwen2.5-VL-7B evaluation
python eval_qwen2vl.py
```
3. Visual-Text Reasoning (Mathematical & Geometric)
Evaluate complex reasoning capabilities on mathematical/geometric tasks:
```bash
# LLaVA-v1.5-7b on MathVision and GeoQA3
python llava_mathv_test_final.py
python llava_mathv_test_GeoAQ3.py

# MiniGpt-4-7B on MathVision and GeoQA3
python minigpt4_mathv_all_metrics.py
python minigpt4_mathv_GeoQA3_metrics.py
```
4. Advanced Visual-Text Reasoning
Scenario 1: OCR-based Logical Reasoning (LogicOCR)
```bash
# LLaVA-v1.5-7b on LogicOCR
python llava_logicocr_vqa.py

# MiniGpt-4-7B on LogicOCR
python minigpt4_logicocr_vqa.py
```
Scenario 2: Cross-Lingual & Multimodal Reasoning (XLRS-Bench)
```bash
# GeoLLaVA-8K on XLRS-bench
python GeoLLaVA-8K_xlrs_bench_infer.py

# LLaVA-v1.5-7b on XLRS-bench
python llava_xlrs_bench_infer.py
```
5. Medical Domain Evaluation (MIMIC-CXR)
Evaluate medical image interpretation on chest X-ray dataset:
```bash
# LLaVA-MED-v1.5 on MIMIC-CXR
python llava_med_mimic_cxr_demo.py

# LLaVA-v1.5-7b on MIMIC-CXR
python llava_mimic_cxr_demo.py
```
🙏 Acknowledgements
This project benefits from the following open-source models and datasets:
Open-Source Models
BLIP2-OPT-2.7b 
MiniGpt-4-7B 
LLaVA-MED-v1.5
LLaVA-v1.5-7b
CLIP-ViT-Large-Patch14
Qwen2.5-VL-7B
GeoLLaVA-8K
Open-Source Datasets
COCO
VQAv2
GeoQA
MathVision
LogicOCR
XLRS-Bench
MIMIC-CXR
