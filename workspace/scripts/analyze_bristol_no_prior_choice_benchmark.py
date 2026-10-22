"""Audited scoring-only repair for baseline round-off, no new inference.

The original frozen runner's cell-equality assertion failed on four historical
Logit cells with probability spreads <=1.11e-16. Keep that runner unchanged and
archive this scoring revision. Actual per-record scores, weights, clipping and
the predeclared comparisons are unchanged; only the equality check is tolerant.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import run_bristol_no_prior_choice_benchmark as benchmark
from run_bristol_prior_mechanism_probe import sha, write_json


def evaluate(rows, method):
    cells = defaultdict(list)
    losses, briers, correct = [], [], []
    for row in rows:
        p = min(1 - 1e-9, max(1e-9, float(row[method])))
        y = int(row["observed_exit"]) == 1
        loss = -math.log(p if y else 1 - p)
        brier = (p - y) ** 2
        losses.append(loss); briers.append(brier); correct.append((p >= .5) == y)
        cells[benchmark.condition(row)].append((p, y, loss, brier))
    cell_loss = [statistics.mean(x[2] for x in values) for values in cells.values()]
    cell_brier = [statistics.mean(x[3] for x in values) for values in cells.values()]
    cell_errors = []
    for values in cells.values():
        ps = [x[0] for x in values]
        if max(ps) - min(ps) > 1e-12:
            raise ValueError("Substantively different within-condition probabilities")
        cell_errors.append(abs(values[0][0] - statistics.mean(x[1] for x in values)))
    return {"records": len(rows), "conditions": len(cells),
        "condition_macro_NLL": statistics.mean(cell_loss), "condition_macro_Brier": statistics.mean(cell_brier),
        "participant_NLL": statistics.mean(losses), "participant_accuracy": statistics.mean(correct),
        "mean_absolute_condition_probability_error": statistics.mean(cell_errors)}


def main():
    cfg, plan = benchmark.prepare()
    rows = list(csv.DictReader((benchmark.ROOT / cfg["evaluation_source"]).open(encoding="utf-8-sig")))
    spreads = []
    for method in cfg["methods"][:-1]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[benchmark.condition(row)].append(float(row[method]))
        for cond, values in grouped.items():
            spread = max(values) - min(values)
            if spread:
                spreads.append({"method": method, "condition": list(cond), "spread": spread})
    assert max(r["spread"] for r in spreads) <= 1e-12
    revision = {"reason": "Strict within-condition float equality failed on unchanged historical Logit probabilities",
        "change": "Equality assertion only: allow probability spread <=1e-12; score all original record probabilities unchanged",
        "baseline_probability_spreads": spreads, "manifest_sha256": sha(benchmark.OUTPUT / "frozen_manifest.json"),
        "responses_sha256": sha(benchmark.OUTPUT / "responses.jsonl"),
        "analysis_repair_sha256": sha(__file__), "inference_runner_sha256": plan["runner_sha256"],
        "prompt_changes": 0, "new_inference_requests": 0, "metric_definition_changes": 0}
    path = benchmark.OUTPUT / "analysis_repair_v1.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        assert {k: v for k, v in previous.items() if k != "recorded_at_utc"} == revision
    else:
        revision["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, revision)
    benchmark.evaluate = evaluate
    benchmark.analyze(cfg, plan)


if __name__ == "__main__":
    main()
