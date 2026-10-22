"""Independent read-only reconciliation of the completed 2026-10-12 session.

No fitting, inference, simulator, or reserved-scenario access. All new results
are descriptive transformations of previously saved development predictions.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_two_hour_session_20261012"
SOURCE = ROOT / "outputs/exit_choice_initial_benchmark_20261012"
CAL = ROOT / "outputs/exit_choice_calibration_campaign_20261012"
CTX = ROOT / "outputs/exit_choice_supervised_context_20261012"
ORDER = ROOT / "outputs/exit_choice_order_stability_20261012"
PROBE = ROOT / "outputs/exit_choice_training_table_probe_20261012"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def metric(rows):
    p = np.asarray([r["probabilities"] for r in rows], dtype=float)
    y = np.asarray([r["true_exit"] - 1 for r in rows])
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.max(abs(p.sum(axis=1) - 1)) < 1e-10
    truth = np.eye(3)[y]
    return {"n": len(rows), "accuracy": float((p.argmax(1) == y).mean()),
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-6)).mean()),
            "brier": float(((p - truth) ** 2).sum(1).mean()),
            "exit_share_tv": float(abs(p.mean(0) - truth.mean(0)).sum() / 2),
            "predicted_shares": p.mean(0).tolist()}


def check_metrics(predictions, stored, selectors, date_key):
    maximum = 0.0
    checked = 0
    for row in stored:
        matches = [p for p in predictions if all(p[k] == row[k] for k in selectors)]
        if row["date"] == "date_macro":
            dates = sorted({p[date_key] for p in matches})
            by_date = [metric([p for p in matches if p[date_key] == d]) for d in dates]
            calc = {k: float(np.mean([m[k] for m in by_date]))
                    for k in ("accuracy", "log_loss", "brier", "exit_share_tv")}
        else:
            matches = [p for p in matches if row["date"] == "pooled" or p[date_key] == row["date"]]
            calc = metric(matches)
            assert calc["n"] == row["n"]
        for key in ("accuracy", "log_loss", "brier", "exit_share_tv"):
            maximum = max(maximum, abs(calc[key] - row[key]))
        checked += 1
    assert maximum < 1e-11
    return {"metric_rows_recomputed": checked, "maximum_numeric_difference": maximum}


def main():
    cases = read(SOURCE / "strict_initial_cases.json")
    lookup = {r["record_id"]: r for r in cases}
    assert len(cases) == len(lookup) == 103
    dates = sorted({c["date"] for c in cases})
    calibration = check_metrics(read(CAL / "predictions.json"), read(CAL / "metrics.json"),
                                ["channel", "method"], "test_date")
    context_predictions = read(CTX / "predictions.json")
    context = check_metrics(context_predictions, read(CTX / "metrics.json"), ["mode"], "date")
    tasks = read(CTX / "requests.json")
    plan = read(CTX / "frozen_plan.json")
    assert sha(CTX / "requests.json") == plan["requests_sha256"]
    records = [read(p) for p in sorted((CTX / "responses").glob("*.json"))]
    assert len(tasks) == len(records) == 309
    assert len({(r["case_id"], r["mode"]) for r in records}) == 309
    responses = {(r["case_id"], r["mode"]): r for r in records}
    for task in tasks:
        test_date = task["test_date"]
        train_ids = task["training_ids"]
        assert set(train_ids) == {k for k, c in lookup.items() if c["date"] != test_date}
        assert len(train_ids) == len(set(train_ids))
        assert all(lookup[k]["date"] != test_date for k in train_ids)
        payload = task["payload"]
        training = json.loads(payload["messages"][1]["content"].split("：", 1)[1])
        counts = [sum(lookup[k]["initial_label"] == j for k in train_ids) for j in (1, 2, 3)]
        expected = [[*[e["dist"] for e in lookup[k]["exits"]],
                     *[e["people"] for e in lookup[k]["exits"]],
                     lookup[k]["initial_label"]] for k in train_ids]
        assert training["training_exit_counts"] == (None if task["mode"] == "zero_same_settings" else counts)
        assert training["training_records"] == (expected if task["mode"] == "all_training_records" else None)
        target = json.loads(payload["messages"][2]["content"].split("：", 1)[1])
        assert set(target) == {"exits"}
        assert all(set(e) == {"exit", "distance", "previous_choosers", "commonly_used"} for e in target["exits"])
        rec = responses[(task["case_id"], task["mode"])]
        assert rec["payload"] == payload and rec["valid"]
        resp = rec["response"]
        assert resp["done"] and resp.get("done_reason") == "stop"
        assert resp["prompt_eval_count"] + resp["eval_count"] < plan["maximum_context"]
        assert payload["options"] == plan["options"]
    paired = []
    for date in dates:
        rr = [r for r in context_predictions if r["date"] == date]
        z = metric([r for r in rr if r["mode"] == "zero_same_settings"])
        f = metric([r for r in rr if r["mode"] == "all_training_records"])
        paired.append({"date": date, "n": z["n"], "zero_accuracy": z["accuracy"],
                       "records_accuracy": f["accuracy"], "accuracy_change_pp": 100 * (f["accuracy"] - z["accuracy"]),
                       "zero_log_loss": z["log_loss"], "records_log_loss": f["log_loss"],
                       "log_loss_change": f["log_loss"] - z["log_loss"],
                       "zero_predicted_shares": z["predicted_shares"],
                       "records_predicted_shares": f["predicted_shares"]})
    probes = [read(p) for p in sorted(PROBE.glob("2019-*.json"))]
    assert len(probes) == 8 and all(p["valid"] for p in probes)
    assert sha(PROBE / "requests.json") == read(PROBE / "frozen_plan.json")["requests_sha256"]
    probe_rows = [{"test_date": p["date"], "mode": p["mode"], "expected": p["expected"],
                   "answer": p["answer"], "exact_match": p["exact_match"]} for p in probes]
    order_records = [read(p) for p in (ORDER / "responses").glob("*.json")]
    assert len(order_records) == 24 and all(r["valid"] for r in order_records)
    preserved = {}
    for folder in (CAL, CTX):
        for p in read(folder / "preservation_before.json"):
            if p["path"] in preserved:
                assert preserved[p["path"]] == p["sha256"]
            preserved[p["path"]] = p["sha256"]
    for p, expected in preserved.items():
        assert sha(Path(p)) == expected
    source_manifest = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                       for folder in (SOURCE, CAL, CTX, ORDER, PROBE)
                       for p in sorted(folder.glob("*.json"))
                       if not p.name.startswith("preservation_")]
    summary = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "human_records": 103, "dates": dates, "date_counts": dict(Counter(c["date"] for c in cases)),
               "recorded_exit_counts": dict(Counter(str(c["initial_label"]) for c in cases)),
               "new_context_calls": 309, "new_order_calls": 24, "new_counting_probe_calls": 8,
               "total_new_local_llm_calls": 341, "all_new_calls_valid": True,
               "calibration_reconciliation": calibration, "context_reconciliation": context,
               "context_rows_and_label_sources_match": True, "target_date_labels_absent": True,
               "context_capacity_check_passed": True,
               "protected_source_files_unchanged": len(preserved),
               "new_ppo_training": 0, "new_evacuation_episodes": 0,
               "reserved_scenarios_accessed_by_this_script": 0,
               "scope": "Development-data prediction and presentation diagnostics; no independent final validation",
               "human_source_verification": "pending; agent checks are not human manuscript approval"}
    save("evidence_audit.json", summary)
    save("context_per_date_comparison.json", paired)
    save("counting_probe_details.json", probe_rows)
    save("source_manifest.json", source_manifest)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
