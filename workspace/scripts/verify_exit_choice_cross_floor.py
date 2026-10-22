"""Independent saved-parameter and request-boundary check for floor transfer."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_cross_floor_transfer_20261012"


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def independent_matrix(cases, method):
    rows = []
    for case in cases:
        xx = []
        for j, exit_row in enumerate(case["exits"]):
            if method == "source_index_logit_diagnostic":
                values = [float(j == 0), float(j == 1), exit_row["dist"]/10, exit_row["people"]/10]
            elif method == "familiarity_only": values = [exit_row["fam"]]
            else:
                values = [exit_row["dist"]/10, exit_row["people"]/10, exit_row["fam"]]
                if method == "portable_nonlinear": values.append(float(np.log1p(exit_row["people"])))
            xx.append(values)
        rows.append(xx)
    return np.array(rows)


def main():
    source = read(ROOT / "outputs/exit_choice_initial_benchmark_20261012/strict_initial_cases.json")
    target = read(OUT / "target_cases.json")
    source_map = {r["record_id"]: r for r in source}
    target_map = {r["record_id"]: r for r in target}
    baseline = read(OUT / "baseline_predictions.json")
    source_fits = read(OUT / "source_fits.json")
    max_delta = 0.0; max_permutation_delta = 0.0
    for fit in source_fits:
        cases = [r for r in target if r["date"] == fit["date"]]
        train = [source_map[k] for k in fit["training_ids"]]
        assert all(r["date"] != fit["date"] for r in train)
        method = fit["method"]
        if method == "nearest":
            pp = softmax(-np.array([[e["dist"] for e in c["exits"]] for c in cases]) / fit["parameter"])
        elif method == "source_exit_frequency_diagnostic":
            counts = np.bincount([r["initial_label"]-1 for r in train], minlength=3)
            pp = np.repeat(((counts+1)/(len(train)+3))[None, :], len(cases), axis=0)
        else:
            matrix = independent_matrix(cases, method)
            coefficients = np.array(fit["coefficients"])
            pp = softmax(np.einsum("njd,d->nj", matrix, coefficients))
            if method != "source_index_logit_diagnostic":
                reversed_pp = softmax(np.einsum("njd,d->nj", matrix[:, ::-1, :], coefficients))[:, ::-1]
                max_permutation_delta = max(max_permutation_delta, float(abs(reversed_pp-pp).max()))
        stored = {r["case_id"]: r["probabilities"] for r in baseline if r["method"] == method and r["date"] == fit["date"]}
        for case, p in zip(cases, pp):
            max_delta = max(max_delta, float(abs(p-np.array(stored[case["record_id"]])).max()))
    assert max_delta < 1e-12 and max_permutation_delta < 1e-12
    tasks = read(OUT / "requests.json")
    for task in tasks:
        train_ids = task["training_ids"]
        assert set(train_ids) == {k for k, c in source_map.items() if c["date"] != task["date"]}
        messages = task["payload"]["messages"]
        target_payload = json.loads(messages[2]["content"].split("：", 1)[1])
        expected = {"exits": [{"exit": f"E{e['exit']}", "distance": e["dist"],
                              "previous_choosers": e["people"], "commonly_used": e["fam"]}
                             for e in target_map[task["case_id"]]["exits"]]}
        assert target_payload == expected
        context = json.loads(messages[1]["content"].split("：", 1)[1])
        if task["mode"] == "zero_same_settings":
            assert context == {"training_records": None, "training_exit_counts": None}
        else:
            expected_table = [[*[e["dist"] for e in source_map[k]["exits"]],
                               *[e["people"] for e in source_map[k]["exits"]],
                               source_map[k]["initial_label"]] for k in train_ids]
            expected_counts = [sum(source_map[k]["initial_label"] == j for k in train_ids) for j in (1, 2, 3)]
            assert context["training_records"] == expected_table and context["training_exit_counts"] == expected_counts
    result = {"at_utc": datetime.now(timezone.utc).isoformat(), "statistical_predictions_recomputed": len(baseline),
              "maximum_probability_difference": max_delta, "portable_utility_permutation_difference": max_permutation_delta,
              "requests_checked": len(tasks), "current_attributes_exact": True,
              "training_labels_from_other_dates_source_floor_only": True,
              "observer_info_is_not_proven_actor_perception": True, "passed": True}
    if (OUT / "predictions.json").exists():
        predictions = read(OUT / "predictions.json"); metrics = read(OUT / "metrics.json")
        maximum_metric_difference = 0.0
        for row in metrics:
            selected = [p for p in predictions if p["method"] == row["method"]]
            def values(rr):
                y = np.asarray([r["true_exit"]-1 for r in rr]); p = np.asarray([r["probabilities"] for r in rr])
                truth = np.eye(3)[y]
                return {"accuracy": float((p.argmax(1) == y).mean()),
                        "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-6)).mean()),
                        "brier": float(((p-truth)**2).sum(1).mean()),
                        "exit_share_tv": float(abs(p.mean(0)-truth.mean(0)).sum()/2)}
            if row["date"] == "date_macro":
                date_metrics = [values([r for r in selected if r["date"] == d]) for d in sorted({r["date"] for r in selected})]
                calculated = {k: float(np.mean([d[k] for d in date_metrics])) for k in date_metrics[0]}
            else: calculated = values([r for r in selected if row["date"] == "pooled" or r["date"] == row["date"]])
            maximum_metric_difference = max(maximum_metric_difference, max(abs(row[k]-v) for k, v in calculated.items()))
        assert maximum_metric_difference < 1e-12
        result.update(metric_rows_recomputed=len(metrics), maximum_metric_difference=maximum_metric_difference)
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
