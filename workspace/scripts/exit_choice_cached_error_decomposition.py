"""Descriptive error decomposition from saved development predictions only."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_two_hour_session_20261012"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")


def describe(method, rows):
    p = np.asarray([r["probabilities"] for r in rows])
    y = np.asarray([r["true_exit"] - 1 for r in rows])
    pred = p.argmax(1)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-6))
    cm = [[int(np.sum((y == i) & (pred == j))) for j in range(3)] for i in range(3)]
    zero = p[np.arange(len(y)), y] <= 1e-6
    return {"method": method, "n": len(rows), "accuracy": float((pred == y).mean()),
            "log_loss": float(nll.mean()), "confusion_true_rows_predicted_columns": cm,
            "recall_E1": float((pred[y == 0] == 0).mean()),
            "recall_E2": float((pred[y == 1] == 1).mean()), "recall_E3": None,
            "predicted_mean_exit_probabilities": p.mean(0).tolist(),
            "true_choice_probability_at_or_below_floor": int(zero.sum()),
            "zero_probability_nll_fraction": float(nll[zero].sum() / nll.sum()),
            "true_choice_probability_under_0_1": int(np.sum(p[np.arange(len(y)), y] < 0.1)),
            "worst_five_case_ids": [rows[i]["case_id"] for i in np.argsort(-nll)[:5]],
            "maximum_case_nll": float(nll.max())}


def main():
    calibration = read(ROOT / "outputs/exit_choice_calibration_campaign_20261012/predictions.json")
    context = read(ROOT / "outputs/exit_choice_supervised_context_20261012/predictions.json")
    groups = {}
    for method in ("raw_qwen", "bias_calibration", "observables_logit", "observables_nonlinear", "hybrid_qwen_logit"):
        groups[method] = sorted([r for r in calibration if r["channel"] == "primary" and r["method"] == method], key=lambda r: r["case_id"])
    for mode in ("zero_same_settings", "training_prior", "all_training_records"):
        groups[mode] = sorted([r for r in context if r["mode"] == mode], key=lambda r: r["case_id"])
    assert all(len(rows) == 103 for rows in groups.values())
    keys = {r["case_id"] for r in groups["raw_qwen"]}
    assert all({r["case_id"] for r in rows} == keys for rows in groups.values())
    result = [describe(m, rows) for m, rows in groups.items()]
    zero = {r["case_id"]: r for r in groups["zero_same_settings"]}
    full = {r["case_id"]: r for r in groups["all_training_records"]}
    paired = []
    counts = defaultdict(int)
    for key in sorted(zero):
        a, b = zero[key], full[key]
        assert a["true_exit"] == b["true_exit"]
        y = a["true_exit"] - 1
        correct_a = int(np.argmax(a["probabilities"])) == y
        correct_b = int(np.argmax(b["probabilities"])) == y
        category = ("corrected" if correct_b else "still_wrong") if not correct_a else ("still_correct" if correct_b else "spoiled")
        counts[category] += 1
        delta = -np.log(max(b["probabilities"][y], 1e-6)) + np.log(max(a["probabilities"][y], 1e-6))
        paired.append({"case_id": key, "date": a["date"], "true_exit": a["true_exit"],
                       "category": category, "zero_probabilities": a["probabilities"],
                       "record_probabilities": b["probabilities"], "nll_change": float(delta)})
    save("cached_error_decomposition.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "descriptive post-result decomposition; no new confirmation", "summaries": result,
        "context_paired_correctness_counts": dict(counts), "new_llm_calls": 0, "new_fits": 0})
    save("context_casewise_changes.json", paired)
    labels = {"raw_qwen": "原设置零样本Qwen", "bias_calibration": "偏差校准Qwen",
              "observables_logit": "观测Logit", "observables_nonlinear": "Logit加非线性",
              "hybrid_qwen_logit": "Logit加Qwen分数", "zero_same_settings": "新设置零样本Qwen",
              "training_prior": "新设置加训练频率", "all_training_records": "新设置加全部训练记录"}
    lines = ["# 已保存预测的错误分解", "", "2026年10月12日。结果之后的描述性缓存分析，无新调用、拟合或真人样本。主实验和模型选择不改变。", "",
             "出口1有81例，出口2有22例，出口3没有真实选择例；出口3召回率保持未定义。混淆矩阵以真实出口为行、预测出口为列。", "",
             "| 方法 | E1召回 | E2召回 | 平均E3概率 | 真实选择概率≤1e-6的例数 | 这些例子占总NLL |", "|---|---:|---:|---:|---:|---:|"]
    for r in result:
        lines.append(f"| {labels[r['method']]} | {r['recall_E1']:.2%} | {r['recall_E2']:.2%} | {r['predicted_mean_exit_probabilities'][2]:.2%} | {r['true_choice_probability_at_or_below_floor']} | {r['zero_probability_nll_fraction']:.2%} |")
    lines.extend(["", f"完整记录对同设置零样本，纠正{counts['corrected']}例、破坏{counts['spoiled']}例，保留正确{counts['still_correct']}例、仍错误{counts['still_wrong']}例；净多正确{counts['corrected']-counts['spoiled']}例。", "",
                  "准确率只看最大概率出口，NLL同时惩罚对真实选择过度自信地给极低概率。因此准确率上升可与概率损失退化并存。零概率使用预先定义的1e-6下限，没有删掉高损失案例。不同原设置和新设置不作提示效果的因果比较。", "",
                  "表中E3概率可描述本样本预测份额偏差；没有E3实际选择者，不能据此断言该出口不合法，也不能把概率归零后称修复通过。E1/E2召回仍来自同一建筑已看过的开发日期，不能当作新的独立证据。"])
    (ROOT / "reports/Exit_choice缓存错误分解_2026-10-12.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"methods": len(result), "cases": 103, "context_paired_counts": dict(counts),
                      "report": "reports/Exit_choice缓存错误分解_2026-10-12.md"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
