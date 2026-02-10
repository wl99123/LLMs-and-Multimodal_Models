import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Tuple


# ===================== 1. Hallucination Classification and Judgment =====================
def clean_answer(ans: str) -> str:
    ans = str(ans).lower().strip()
    valid_chars = '0123456789abcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿是与否有无多少'
    return ''.join([c for c in ans if c in valid_chars])


def judge_hallucination(pred_ans: str, true_ans: str, task_type: str = "vqa") -> Tuple[str, str]:
    pred_clean = clean_answer(pred_ans)
    true_clean = clean_answer(true_ans)

    if task_type == "math":
        reasoning_keywords = ["因为", "所以", "步骤", "推理", "首先", "其次", "最终"]
        has_reasoning = any(kw in pred_ans for kw in reasoning_keywords)
        if has_reasoning and pred_clean != true_clean:
            halluc_type1 = "Logical Hallucination"
        else:
            halluc_type1 = "Factual Hallucination" if pred_clean != true_clean else "No Hallucination"
    else:
        if pred_clean != true_clean and len(pred_clean) > 0:
            contradiction_patterns = [
                r"是.*否", r"有.*无", r"正常.*异常", r"存在.*不存在"
            ]
            has_contradiction = any(re.search(pattern, pred_ans) for pattern in contradiction_patterns)
            halluc_type1 = "Logical Hallucination" if has_contradiction else "Factual Hallucination"
        else:
            halluc_type1 = "No Hallucination"

    if halluc_type1 == "No Hallucination":
        halluc_type2 = "No Hallucination"
    else:
        abnormal_keywords = ["异常", "存在", "有", "阳性", "病变", "误判", "增多"]
        true_abnormal = any(kw in true_ans for kw in abnormal_keywords)
        pred_abnormal = any(kw in pred_ans for kw in abnormal_keywords)

        if true_abnormal and not pred_abnormal:
            halluc_type2 = "False Negative Hallucination"
        elif not true_abnormal and pred_abnormal:
            halluc_type2 = "False Positive Hallucination"
        else:
            halluc_type2 = "Other Hallucination"

    return halluc_type1, halluc_type2


# ===================== 2. Extended Evaluation Metrics =====================
def calculate_hallucination_metrics(pred_answers: List[str], true_answers: List[str], task_type: str = "vqa") -> Dict:
    base_metrics = calculate_vqa_metrics(pred_answers, true_answers)

    hallucination_stats = {
        "Factual Hallucination Count": 0,
        "Logical Hallucination Count": 0,
        "False Positive Hallucination Count": 0,
        "False Negative Hallucination Count": 0,
        "Total Hallucination Count": 0
    }

    for p, t in zip(pred_answers, true_answers):
        hallu1, hallu2 = judge_hallucination(p, t, task_type)
        if hallu1 != "No Hallucination":
            hallucination_stats["Total Hallucination Count"] += 1
            if hallu1 == "Factual Hallucination":
                hallucination_stats["Factual Hallucination Count"] += 1
            elif hallu1 == "Logical Hallucination":
                hallucination_stats["Logical Hallucination Count"] += 1

        if hallu2 == "False Positive Hallucination":
            hallucination_stats["False Positive Hallucination Count"] += 1
        elif hallu2 == "False Negative Hallucination":
            hallucination_stats["False Negative Hallucination Count"] += 1

    sample_num = base_metrics["sample_num"]
    hallucination_rate = hallucination_stats["Total Hallucination Count"] / sample_num if sample_num > 0 else 0.0
    false_positive_rate = hallucination_stats[
                              "False Positive Hallucination Count"] / sample_num if sample_num > 0 else 0.0
    false_negative_rate = hallucination_stats[
                              "False Negative Hallucination Count"] / sample_num if sample_num > 0 else 0.0

    full_metrics = {
        "Accuracy": base_metrics["accuracy"],
        "Average F1 Score": base_metrics["avg_f1"],
        "Hallucination Rate": round(hallucination_rate, 4),
        "False Positive Rate": round(false_positive_rate, 4),
        "False Negative Rate": round(false_negative_rate, 4),
        "Sample Number": sample_num,
        "Hallucination Classification Statistics": hallucination_stats
    }

    return full_metrics


# ===================== 3. CoT Prompt Engineering =====================
def build_prompt(question: str, task_type: str = "vqa", use_cot: bool = False) -> str:
    base_prompt = f"Please answer the following question: {question}\nAnswer Requirement: Concise, accurate, only give the core answer."

    if use_cot:
        if task_type == "math":
            cot_prompt = f"""Please answer the question step by step:
1. Analyze the core requirements and known conditions of the question;
2. Derive the solution process step by step, write down the logic and calculation of each step;
3. Finally give a clear answer.
Question: {question}
Solution Process:
"""
        elif task_type == "medical":
            cot_prompt = f"""Please analyze the medical imaging report step by step:
1. Extract key medical features from the report;
2. Judge whether each feature is normal based on medical knowledge;
3. Give a final conclusion by synthesizing all features.
Report Content: {question}
Analysis Process:
"""
        else:
            cot_prompt = f"""Please answer the question step by step:
1. Understand the core intention of the question;
2. Analyze the key information involved in the question;
3. Reason step by step to get the answer;
4. Finally give a clear answer.
Question: {question}
Reasoning Process:
"""
        return cot_prompt
    else:
        return base_prompt


# ===================== 4. Basic Metrics Calculation =====================
def calculate_vqa_metrics(pred_answers: List[str], true_answers: List[str]) -> Dict:
    if len(pred_answers) != len(true_answers):
        raise ValueError("The number of predicted answers and ground truth answers does not match!")

    pred_clean = [clean_answer(x) for x in pred_answers]
    true_clean = [clean_answer(x) for x in true_answers]
    sample_num = len(pred_clean)
    correct_num = 0
    f1_scores = []

    for p, t in zip(pred_clean, true_clean):
        if p == t:
            correct_num += 1

        p_chars = set(p) if p else set()
        t_chars = set(t) if t else set()
        if not p_chars and not t_chars:
            f1 = 1.0
        elif not p_chars or not t_chars:
            f1 = 0.0
        else:
            intersection = len(p_chars & t_chars)
            precision = intersection / len(p_chars)
            recall = intersection / len(t_chars)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    accuracy = correct_num / sample_num if sample_num > 0 else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "avg_f1": round(avg_f1, 4),
        "sample_num": sample_num
    }


# ===================== 5. Full Evaluation Report =====================
def generate_full_eval_report(results_dict: Dict, task_type: str = "vqa") -> None:
    print("\n" + "=" * 80)
    print(f"{task_type.upper()} Multi-Model Hallucination Evaluation Report (Comparison with/without CoT)")
    print("=" * 80)

    df_data = {}
    for model_name, metrics in results_dict.items():
        df_data[model_name] = {
            "Accuracy": metrics["Accuracy"],
            "Average F1 Score": metrics["Average F1 Score"],
            "Hallucination Rate": metrics["Hallucination Rate"],
            "False Positive Rate": metrics["False Positive Rate"],
            "False Negative Rate": metrics["False Negative Rate"],
            "Sample Number": metrics["Sample Number"]
        }

    df = pd.DataFrame(df_data).T
    df = df.round(4)
    df.index.name = "Model Name (CoT Status)"
    print(df)

    cot_models = [name for name in results_dict.keys() if "CoT" in name]
    non_cot_models = [name for name in results_dict.keys() if "No CoT" in name]
    if cot_models and non_cot_models:
        cot_avg_hallu = np.mean([results_dict[name]["Hallucination Rate"] for name in cot_models])
        non_cot_avg_hallu = np.mean([results_dict[name]["Hallucination Rate"] for name in non_cot_models])
        cot_effect = non_cot_avg_hallu - cot_avg_hallu
        print(f"\nCoT Effect Comparison:")
        print(f"   Average Hallucination Rate (No CoT): {non_cot_avg_hallu:.4f}")
        print(f"   Average Hallucination Rate (With CoT): {cot_avg_hallu:.4f}")
        print(f"   Hallucination Rate Reduction by CoT: {cot_effect:.4f} ({cot_effect * 100:.2f}%)")

    report_path = f"/root/autodl-tmp/{task_type}_hallucination_eval_report.csv"
    df.to_csv(report_path, encoding="utf-8-sig")

    hallu_stats_path = f"/root/autodl-tmp/{task_type}_hallucination_stats.json"
    hallu_stats = {name: metrics["Hallucination Classification Statistics"] for name, metrics in results_dict.items()}
    with open(hallu_stats_path, "w", encoding="utf-8") as f:
        json.dump(hallu_stats, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation report saved to: {report_path}")
    print(f"Hallucination classification statistics saved to: {hallu_stats_path}")


# ===================== 6. Automated Evaluation Main Process =====================
def automated_evaluation(
        model_predictions: Dict[str, Dict[str, List[str]]],
        true_answers: List[str],
        task_type: str = "vqa"
) -> None:
    results_dict = {}
    for model_name, pred_data in model_predictions.items():
        preds = pred_data["preds"]
        metrics = calculate_hallucination_metrics(preds, true_answers, task_type)
        results_dict[model_name] = metrics

    generate_full_eval_report(results_dict, task_type)


# ===================== Test =====================
if __name__ == "__main__":
    # VQA Task Test
    test_true_answers = ["是", "3只", "苹果", "否", "800x600"]
    test_model_predictions = {
        "Qwen2-VL-7B (With CoT)": {"preds": ["否", "3", "苹果", "否", "800x600"]},
        "Qwen2-VL-7B (No CoT)": {"preds": ["是", "4只", "香蕉", "否", "无"]},
        "LLaVA-7B-v1.5 (With CoT)": {"preds": ["否", "3", "苹果", "是", "800x600"]},
        "LLaVA-7B-v1.5 (No CoT)": {"preds": ["是", "5只", "橙子", "是", "未知"]}
    }
    automated_evaluation(test_model_predictions, test_true_answers, task_type="vqa")

    # Math Task Test
    math_true = ["180度", "5cm", "直角三角形"]
    math_preds = {
        "MathGLM (With CoT)": {"preds": ["因为三角形内角和为180度，所以答案是180度", "5cm（计算过程：边长=√(3²+4²)=5）",
                                         "锐角三角形（推理错误）"]},
        "MathGLM (No CoT)": {"preds": ["180", "6cm", "直角三角形"]}
    }
    automated_evaluation(math_preds, math_true, task_type="math")