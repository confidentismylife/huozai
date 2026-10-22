"""Separate fixed-plan prediction error from rolling-execution effects.

Fixed scheduled sampling, no policy changes. This is a mechanistic cache audit,
not another independent confirmation experiment or a learned gate calibration.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from fire_evac.safestep_public_benchmark import Graph, future_min_fed_policy
from fire_evac.safestep_causal_forecast import make_forecast, predicted_interval_costs, fixed_policy_values
from fire_evac.safestep_robust_gate import delivered_history

RUN = ROOT / "outputs/safestep_robust_gate_20261018"
OUT = ROOT / "outputs/safestep_gate_plan_semantics_20261018"


def save(p, data):
    assert not p.exists(), p
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(p, rows):
    if not rows:
        return
    assert not p.exists()
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def label(difference):
    return "helpful" if difference < -1e-10 else "harmful" if difference > 1e-10 else "neutral"


def plan_cost(graph, policy, costs, state):
    dose = 0.
    for k in range(len(costs)):
        if state in graph.exit_ids:
            return dose
        action = int(policy[k, state])
        if action < 0:
            raise ValueError("Sampled proposed fixed plan does not reach by deadline")
        dose += float(costs[k, state])
        state = graph.destination(state, action)
    assert state in graph.exit_ids
    return dose


def inspect_world(spec, cfg, graph):
    directory = RUN / "worlds" / spec["id"]
    signatures = defaultdict(int)
    with gzip.open(directory / "episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["method"] not in ("consensus_gate", "switch_margin_gate"):
                continue
            for event in r["gate_opportunities"]:
                signatures[r["departure_fire_s"], event["step"], event["state"]] += 1
    with np.load(directory / "native_inputs.npz") as raw:
        times = raw["acquisition_times"]
        measured_regions = raw["measured_region_FER"]
        mapping = raw["cell_to_region"]
    measured = np.zeros((len(times), len(graph.cells)))
    for state, region in enumerate(mapping):
        if region >= 0:
            measured[:, state] = measured_regions[:, region]
    rows, table_cache, continuations = [], {}, {}
    for departure in sorted({key[0] for key in signatures}):
        with np.load(directory / f"tables_t{departure}.npz") as source:
            table_cache[departure] = {k: source[k] for k in source.files}
        table = table_cache[departure]
        continuations[departure] = (
            fixed_policy_values(graph, table["first_snapshot_hold"], table["true_costs"]),
            fixed_policy_values(graph, table["first_ensemble_mean"], table["true_costs"]))
    for (departure, k, state), occurrences in sorted(signatures.items()):
        table = table_cache[departure]
        costs = table["true_costs"][k:]
        history = delivered_history(times, measured, departure + k * cfg["dt_s"], spec["sensor"]["latency_s"])
        predicted = []
        for model in cfg["forecast_models"]:
            forecast = make_forecast(history, model["window_s"], model["projection_cap_s"], model["slope_gain"])
            predicted.append(predicted_interval_costs(forecast, history.decision_time,
                                                     cfg["dt_s"], cfg["maximum_actions"] - k))
        held = np.tile(history.rates[-1] * cfg["dt_s"], (cfg["maximum_actions"] - k, 1))
        baseline, _, _ = future_min_fed_policy(graph, held)
        nominee, _, _ = future_min_fed_policy(graph, np.mean(predicted, axis=0))
        assert int(baseline[0, state]) == table["first_snapshot_hold"][k, state]
        assert int(nominee[0, state]) == table["first_ensemble_mean"][k, state]
        gains = []
        for forecast_costs in predicted:
            b = plan_cost(graph, baseline, forecast_costs, state)
            p = plan_cost(graph, nominee, forecast_costs, state)
            gains.append((b - p) / max(b, 1e-12))
        score = min(gains)
        assert abs(score - table["min_gain"][k, state]) < 1e-8
        actual_fixed_baseline = plan_cost(graph, baseline, costs, state)
        actual_fixed_nominee = plan_cost(graph, nominee, costs, state)
        fixed_difference = actual_fixed_nominee - actual_fixed_baseline
        snap_values, ensemble_values = continuations[departure]
        dest = graph.destination(state, int(nominee[0, state]))
        one_action_difference = float(table["true_costs"][k, state] + snap_values[k + 1, dest]
                                      - snap_values[k, state])
        persistent_difference = float(ensemble_values[k, state] - snap_values[k, state])
        rows.append(dict(world_id=spec["id"], family=spec["family"], sensor_id=spec["sensor"]["id"],
            seed=spec["seed"], departure_fire_s=departure, step=k, state=state,
            visited_occurrences_across_two_gates=occurrences,
            minimum_predicted_relative_gain=score,
            fixed_nominee_actual_FED=actual_fixed_nominee,
            fixed_baseline_actual_FED=actual_fixed_baseline,
            fixed_plan_difference_FED=fixed_difference, fixed_plan_label=label(fixed_difference),
            one_action_then_snapshot_difference_FED=one_action_difference,
            one_action_label=label(one_action_difference),
            persistent_ensemble_difference_FED=persistent_difference,
            persistent_ensemble_label=label(persistent_difference)))
    return rows


def main():
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    cfg = frozen["configuration"]
    confirm = [s for s in frozen["world_schedule"] if s["stage"] == "synthetic_confirmation"]
    selected = confirm[::10]
    if not OUT.exists():
        OUT.mkdir()
        save(OUT / "analysis_plan.json", dict(created_UTC=datetime.now(timezone.utc).isoformat(),
            status="MECHANISTIC_AUDIT_FROZEN_DURING_BATCH_BEFORE_CONFIRMATION_AGGREGATE_INSPECTION",
            selection="Every tenth confirmation world in the already frozen order; all distinct (departure,step,state) opportunities visited by either primary gate",
            selected_worlds=[s["id"] for s in selected], sample_worlds=len(selected),
            policy_or_threshold_changes=False, new_closed_loop_episodes=0,
            output_labels=["same nominee and comparator fixed plans", "one nominee action then snapshot", "persistent ensemble versus snapshot"],
            interpretation="Three different execution semantics; do not mislabel agreement scores as calibrated probabilities",
            script_sha256=sha(Path(__file__)), frozen_run_plan_sha256=sha(RUN / "frozen_plan.json")))
    graph_data = json.loads((RUN / "graph.json").read_text("utf-8"))
    graph = Graph(graph_data["cells"], graph_data["exits"])
    rows = []
    start = time.perf_counter()
    for ordinal, spec in enumerate(selected):
        result_path = OUT / f"{spec['id']}.json"
        if result_path.exists():
            local = json.loads(result_path.read_text("utf-8"))
        else:
            while not (RUN / "worlds" / spec["id"] / "completion_manifest.json").exists():
                if list(RUN.glob("failure_*.txt")) or time.perf_counter() - start > 7200:
                    raise RuntimeError("Batch stopped or watch budget exceeded; prior results retained")
                time.sleep(10)
            local = inspect_world(spec, cfg, graph)
            save(result_path, local)
        rows += local
        if (ordinal + 1) % 10 == 0:
            print(json.dumps(dict(inspected_worlds=ordinal + 1, sample_worlds=len(selected),
                distinct_opportunity_states=len(rows))), flush=True)
    write_csv(OUT / "all_distinct_plan_states.csv", rows)
    bins = [-float("inf"), 0., .01, .02, .05, .10, .20, .50, float("inf")]
    bin_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        group = [r for r in rows if lo <= r["minimum_predicted_relative_gain"] < hi]
        record = dict(score_interval=f"[{lo},{hi})", unique_states=len(group))
        for semantics in ("fixed_plan", "one_action", "persistent_ensemble"):
            for outcome in ("helpful", "harmful", "neutral"):
                record[f"{semantics}_{outcome}"] = sum(r[f"{semantics}_label"] == outcome for r in group)
        bin_rows.append(record)
    write_csv(OUT / "score_bins_and_execution_semantics.csv", bin_rows)
    disagreement = sum(r["fixed_plan_label"] != r["one_action_label"] for r in rows)
    strong = [r for r in rows if r["minimum_predicted_relative_gain"] >= .05]
    save(OUT / "analysis_results.json", dict(status="COMPLETE_MECHANISTIC_CACHE_AUDIT",
        sampled_worlds=len(selected), distinct_proposal_states=len(rows),
        fixed_vs_one_action_label_disagreements=disagreement,
        agreement_score_at_least_5pct_states=len(strong),
        high_score_but_actual_fixed_plan_harmful=sum(r["fixed_plan_label"] == "harmful" for r in strong),
        high_score_but_one_action_harmful=sum(r["one_action_label"] == "harmful" for r in strong),
        new_closed_loop_episodes=0, extra_fixed_plan_ground_truth_scores=2*len(rows),
        new_training=0, new_LLM_calls=0, sealed_final48_access=False))
    files = [p for p in OUT.iterdir() if p.is_file()]
    save(OUT / "completion_manifest.json", dict(status="COMPLETE",
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", sampled_worlds=len(selected),
        distinct_states=len(rows), differing_execution_labels=disagreement)), flush=True)


if __name__ == "__main__":
    main()
