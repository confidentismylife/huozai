"""Complete-cohort descriptive effects; no adaptive fitting or model calls."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_cross_floor_transfer_20261012"


def read(p): return json.loads(p.read_text(encoding="utf-8"))


def compare(predictions, candidate, reference):
    a = {r["case_id"]: r for r in predictions if r["method"] == candidate}
    b = {r["case_id"]: r for r in predictions if r["method"] == reference}
    assert set(a) == set(b) and len(a) == 71
    paired = []
    for key in sorted(a):
        ra, rb = a[key], b[key]; assert ra["true_exit"] == rb["true_exit"]
        y = ra["true_exit"]-1
        nll_a = -np.log(max(ra["probabilities"][y], 1e-6))
        nll_b = -np.log(max(rb["probabilities"][y], 1e-6))
        correct_a = int(np.argmax(ra["probabilities"])) == y
        correct_b = int(np.argmax(rb["probabilities"])) == y
        paired.append({"case_id": key, "date": ra["date"], "true_exit": ra["true_exit"],
                       "nll_change": float(nll_a-nll_b), "accuracy_change": int(correct_a)-int(correct_b),
                       "candidate_correct": correct_a, "reference_correct": correct_b})
    dates = []
    for date in sorted({r["date"] for r in paired}):
        rr = [r for r in paired if r["date"] == date]
        dates.append({"date": date, "n": len(rr), "nll_change": float(np.mean([r["nll_change"] for r in rr])),
                      "accuracy_change_pp": float(100*np.mean([r["accuracy_change"] for r in rr]))})
    return {"candidate": candidate, "reference": reference, "n": len(paired),
            "date_macro_nll_change": float(np.mean([r["nll_change"] for r in dates])),
            "pooled_nll_change": float(np.mean([r["nll_change"] for r in paired])),
            "pooled_accuracy_change_pp": float(100*np.mean([r["accuracy_change"] for r in paired])),
            "dates_improved_nll": sum(r["nll_change"] < 0 for r in dates), "dates": dates,
            "corrected": sum(r["candidate_correct"] and not r["reference_correct"] for r in paired),
            "spoiled": sum(not r["candidate_correct"] and r["reference_correct"] for r in paired)}, paired


def main():
    summary = read(OUT / "summary.json"); assert summary["complete"], "Do not compare incomplete cohorts"
    predictions = read(OUT / "predictions.json"); target = read(OUT / "target_cases.json")
    pairs = [("qwen_source_training_records", "portable_nonlinear"),
             ("qwen_source_training_records", "qwen_zero_same_settings"),
             ("portable_nonlinear", "source_index_logit_diagnostic")]
    effects = []; changes = []
    for candidate, reference in pairs:
        effect, rows = compare(predictions, candidate, reference)
        effects.append(effect); changes.extend({"candidate": candidate, "reference": reference, **r} for r in rows)
    coverage = []
    for date in sorted({r["date"] for r in target}):
        rr = [r for r in target if r["date"] == date]
        count = Counter(r["initial_label"] for r in rr)
        coverage.append({"date": date, "n": len(rr), "exit_counts": [count[j] for j in (1, 2, 3)]})
    diagnostics = []
    for method in ("qwen_zero_same_settings", "qwen_source_training_records"):
        rr = [r for r in predictions if r["method"] == method]
        losses = [-np.log(max(r["probabilities"][r["true_exit"]-1], 1e-6)) for r in rr]
        at_floor = [r["probabilities"][r["true_exit"]-1] <= 1e-6 for r in rr]
        diagnostics.append({"method": method, "true_exit_probability_at_or_below_floor": sum(at_floor),
                            "floor_case_fraction_of_nll": float(sum(v for v, flag in zip(losses, at_floor) if flag)/sum(losses)),
                            "fallbacks": sum(r["fallback"] for r in rr)})
    result = {"at_utc": datetime.now(timezone.utc).isoformat(), "status": "exploratory complete-cohort paired description",
              "effects": effects, "coverage": coverage, "overconfidence_diagnostics": diagnostics,
              "confidence_intervals_or_significance_tests": "not claimed; three same-building dates and shared source data",
              "new_llm_calls_from_this_analysis": 0, "new_model_fits": 0}
    (OUT / "effects.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "paired_changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "reports/Exit_choice跨楼层迁移实验_2026-10-12.md"
    text = report.read_text(encoding="utf-8")
    lines = ["", "## 完整队列的预先指定比较", "",
             "差为候选减参照，对数损失负值更好。下面不把三日期当作大量独立事件进行显著性推断。", "",
             "| 比较 | 日期等权NLL差 | 合并NLL差 | 合并准确率差，百分点 | NLL改善日期 |", "|---|---:|---:|---:|---:|"]
    names = ["源域记录Qwen减非线性属性效用", "源域记录Qwen减同设置零样本Qwen", "非线性属性效用减编号截距诊断"]
    for name, e in zip(names, effects):
        lines.append(f"| {name} | {e['date_macro_nll_change']:+.6f} | {e['pooled_nll_change']:+.6f} | {e['pooled_accuracy_change_pp']:+.2f} | {e['dates_improved_nll']}/3 |")
    lines.extend(["", "| 日期 | E1选择 | E2选择 | E3选择 |", "|---|---:|---:|---:|"])
    for c in coverage: lines.append(f"| {c['date']} | {' | '.join(str(n) for n in c['exit_counts'])} |")
    lines.extend(["", "若出口3记录集中在一个日期，该类召回主要检验该事件和这条路线，不能说已经验证多个独立事件中的出口3识别。源域与目标域的常用路线、几何及事件背景变化同时存在，不能把差异唯一归因于某个心理因素。", "",
                  f"逐日期配对差、纠正与破坏的案例数量及概率诊断见[effects.json]({(OUT/'effects.json').as_posix()})。没有根据这些结果修改提示、增加训练量或挑选更有利的目标记录。"])
    marker = "\n## 完整队列的预先指定比较\n"
    if marker in text: text = text.split(marker)[0]
    report.write_text(text+"\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"effects": effects, "coverage": coverage}, ensure_ascii=False))


if __name__ == "__main__": main()
