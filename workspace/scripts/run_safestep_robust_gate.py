"""Frozen synthetic stress experiment; stream one world at a time.

No independent physical-fire claim. --prepare freezes methods/generator/seeds;
--run executes context, development and unchanged fresh-seed confirmation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_public_benchmark import (
    Graph, RateField, future_min_fed_policy, forward_min_fed_reference, rollout,
)
from fire_evac.safestep_causal_forecast import fixed_policy_values
from fire_evac.safestep_robust_gate import (
    delivered_history, build_bundle, gated_action, switch_threshold,
    region_distances, make_synthetic_truth, noisy_measurements, PlanBundle,
)

CONFIG = ROOT / "schemas/safestep_robust_gate_20261018.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(p, data):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                            allow_nan=False, default=convert), encoding="utf-8")


def csv_save(p, rows):
    assert not p.exists() and rows, p
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check(hashes):
    for rel, digest in hashes.items():
        assert sha(ROOT / rel) == digest, rel


def schedule(cfg):
    context = dict(stage="context", seed=0, family="source", sensor=cfg["sensors"][0])
    context["id"] = "context_source_clean"
    answer = [context]
    rng = np.random.default_rng(20261018)
    for stage, seeds in (("development", cfg["development_seeds"]),
                         ("synthetic_confirmation", cfg["synthetic_confirmation_seeds"])):
        batch = []
        for seed in seeds:
            arms = [(family, sensor) for family in cfg["truth_families"] for sensor in cfg["sensors"]]
            arms += [("source", sensor) for sensor in cfg["sensors"]
                     if sensor["id"] in cfg["source_only_sensor_arms"]]
            for family, sensor in arms:
                batch.append(dict(id=f"{stage}_s{seed}_{family}_{sensor['id']}",
                                  stage=stage, seed=seed, family=family, sensor=sensor))
        answer += [batch[int(i)] for i in rng.permutation(len(batch))]
    assert len(answer) == cfg["execution"]["expected_worlds"]
    return answer


def prepare(cfg, out):
    assert not out.exists(), out
    review = ROOT / "outputs/safestep_causal_forecast_review_20261018"
    old = json.loads((review / "analysis_plan.json").read_text("utf-8"))["protected_sha256"]
    protected = dict(old)
    latest = json.loads((review / "completion_manifest.json").read_text("utf-8"))
    protected.update(latest["artifact_sha256"])
    protected[(review / "completion_manifest.json").relative_to(ROOT).as_posix()] = sha(review / "completion_manifest.json")
    check(protected)
    inputs = [CONFIG, Path(__file__), ROOT / "src/fire_evac/safestep_robust_gate.py",
              ROOT / "tests/test_safestep_robust_gate.py",
              ROOT / "src/fire_evac/safestep_causal_forecast.py",
              ROOT / "src/fire_evac/safestep_public_benchmark.py",
              ROOT / "src/fire_evac/safestep_author_numerics.py"]
    out.mkdir()
    plan = dict(frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="FROZEN_BEFORE_GENERATING_REAL_GRAPH_STRESS_WORLDS", configuration=cfg,
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        protected_previous_sha256=protected, world_schedule=schedule(cfg),
        prior_knowledge="Original clean fire, departure sweeps and earlier predictor failures are known development data",
        seed_confirmation_scope="Fresh mathematical perturbation/noise draws only, sharing original topology, fire and fixed generator")
    save(out / "frozen_plan.json", plan)
    print(json.dumps(dict(status="PREPARED", worlds=len(plan["world_schedule"]),
        expected_episodes=cfg["execution"]["expected_episodes"], protected=len(protected))), flush=True)


def build_tables(graph, times, measured, latency, departure, cfg):
    horizon, dt = cfg["maximum_actions"], cfg["dt_s"]
    n = len(graph.cells)
    tables = {f"first_{name}": np.full((horizon, n), -1, dtype=int)
              for name in ("snapshot_hold", "trend30", "half30", "ensemble_mean")}
    tables.update(base_exit=np.full((horizon, n), -1, dtype=int),
                  nominee_exit=np.full((horizon, n), -1, dtype=int),
                  min_gain=np.full((horizon, n), -np.inf),
                  model_gains=np.full((horizon, 3, n), -np.inf))
    audits, timings = [], []
    invariant = None
    for k in range(horizon):
        decision = departure + k * dt
        history = delivered_history(times, measured, decision, latency)
        assert history.times[-1] + latency <= decision + 1e-9
        assert decision - history.times[-1] < latency + 5. + 1e-9
        start = time.perf_counter()
        bundle = build_bundle(graph, history, dt, horizon - k, cfg["forecast_models"])
        timings.append(time.perf_counter() - start)
        for method in ("snapshot_hold", "trend30", "half30", "ensemble_mean"):
            tables[f"first_{method}"][k] = bundle.first_actions[method]
        tables["base_exit"][k] = bundle.baseline_exit
        tables["nominee_exit"][k] = bundle.nominee_exit
        tables["min_gain"][k] = bundle.minimum_relative_gain
        tables["model_gains"][k] = bundle.relative_gain_by_model
        audits.append(dict(step=k, decision_time=decision,
            latest_acquisition_s=float(history.times[-1]),
            latest_delivery_s=float(history.times[-1] + latency),
            measurement_age_s=float(decision - history.times[-1])))
        if departure == 300 and k == 1:
            altered = measured.copy()
            altered[times + latency > decision + 1e-9] = 987654.
            other = delivered_history(times, altered, decision, latency)
            assert np.array_equal(history.times, other.times)
            assert np.array_equal(history.rates, other.rates)
            changed = build_bundle(graph, other, dt, horizon - k, cfg["forecast_models"])
            assert all(np.array_equal(bundle.first_actions[m], changed.first_actions[m])
                       for m in bundle.first_actions)
            assert np.array_equal(bundle.relative_gain_by_model, changed.relative_gain_by_model)
            invariant = dict(decision_s=decision, future_delivered_measurements_changed=True,
                             actions_and_gate_scores_unchanged=True)
    return tables, audits, timings, invariant


def reconstruct_bundle(tables, k):
    return PlanBundle({name: tables[f"first_{name}"][k] for name in
                       ("snapshot_hold", "trend30", "half30", "ensemble_mean")},
        tables["base_exit"][k], tables["nominee_exit"][k], tables["min_gain"][k],
        tables["model_gains"][k])


def execute(graph, tables, costs, start, method, cfg, snapshot_values, oracle_policy=None):
    state, previous_exit = start, -1
    path, actions, increments, selected_exits = [state], [], [], []
    gate_rows = []
    immediate_reversals, exit_changes, total = 0, 0, 0.
    for k in range(len(costs)):
        if state in graph.exit_ids:
            break
        bundle = reconstruct_bundle(tables, k)
        if method == "full_future_reference":
            action = int(oracle_policy[k, state])
            selected_exit = -1
        elif method in ("snapshot_hold", "trend30", "half30", "ensemble_mean"):
            action = int(tables[f"first_{method}"][k, state])
            # Route-switch count is defined only where an explicit selected plan
            # exit is available; baseline/ensemble provide it; raw trend does not.
            selected_exit = int(tables["base_exit" if method == "snapshot_hold" else "nominee_exit"][k, state]) if method in ("snapshot_hold", "ensemble_mean") else -1
        else:
            threshold = cfg["gate"]["consensus_threshold"]
            if method == "switch_margin_gate":
                margin = cfg["gate"]["switch_margin"]
                threshold = switch_threshold(previous_exit, int(bundle.nominee_exit[state]),
                    margin["initial"], margin["same_nominated_exit_as_previous_selected_plan"],
                    margin["different_nominated_exit"])
            for sensitivity in cfg["development_only_sensitivity"]:
                if method == sensitivity["id"]:
                    threshold = sensitivity["threshold"]
            action, selected_exit, opportunity, accepted = gated_action(bundle, state, threshold)
            if opportunity:
                candidate = int(bundle.first_actions["ensemble_mean"][state])
                dest = graph.destination(state, candidate)
                baseline_dose = float(snapshot_values[k, state])
                nominee_dose = float(costs[k, state] + snapshot_values[k + 1, dest])
                finite = np.isfinite(baseline_dose) and np.isfinite(nominee_dose)
                difference = nominee_dose - baseline_dose if finite else None
                tol = cfg["evaluation"]["paired_tolerance_FED"]
                classification = ("failure" if not finite else "helpful" if difference < -tol
                                  else "harmful" if difference > tol else "neutral")
                gate_rows.append(dict(step=k, state=state, proposed_action=candidate,
                    snapshot_action=int(bundle.first_actions["snapshot_hold"][state]),
                    accepted=accepted, threshold=threshold,
                    minimum_predicted_relative_gain=float(bundle.minimum_relative_gain[state]),
                    offline_one_action_then_snapshot_difference=difference,
                    offline_classification=classification))
        if action < 0:
            break
        destination = graph.destination(state, action)
        if len(path) >= 2 and destination == path[-2]:
            immediate_reversals += 1
        if selected_exit >= 0 and previous_exit >= 0 and selected_exit != previous_exit:
            exit_changes += 1
        if selected_exit >= 0:
            previous_exit = selected_exit
        selected_exits.append(selected_exit)
        dose = float(costs[k, state])
        total += dose
        increments.append(dose)
        actions.append(action)
        state = destination
        path.append(state)
    return dict(reached=state in graph.exit_ids, steps=len(actions), FED=total,
        travel_s=len(actions) * cfg["dt_s"], path_ids=path, actions=actions,
        FED_increments=increments, immediate_reversals=immediate_reversals,
        selected_plan_exit_changes=exit_changes if method not in ("trend30", "half30", "full_future_reference") else None,
        selected_exit_ids=selected_exits, gate_opportunities=gate_rows)


def summarize(rows, baseline, ensemble, half, cfg):
    tol = cfg["evaluation"]["paired_tolerance_FED"]
    successful = [r for r in rows if r["reached"]]
    result = dict(n=len(rows), reached=len(successful),
        mean_FED=float(np.mean([r["FED"] for r in successful])) if successful else None,
        maximum_FED=max((r["FED"] for r in successful), default=None),
        mean_travel_s=float(np.mean([r["travel_s"] for r in successful])) if successful else None,
        immediate_reversals=sum(r["immediate_reversals"] for r in rows),
        selected_plan_exit_changes=sum(r["selected_plan_exit_changes"] or 0 for r in rows)
            if rows[0]["selected_plan_exit_changes"] is not None else None)
    for label, comparator in (("snapshot", baseline), ("ensemble", ensemble), ("half", half)):
        both = [(r, comparator[r["start_id"]]) for r in rows
                if r["reached"] and comparator[r["start_id"]]["reached"]]
        diff = [r["FED"] - b["FED"] for r, b in both]
        result[f"paired_{label}"] = len(both)
        result[f"lower_vs_{label}"] = sum(x < -tol for x in diff)
        result[f"equal_vs_{label}"] = sum(abs(x) <= tol for x in diff)
        result[f"higher_vs_{label}"] = sum(x > tol for x in diff)
        result[f"mean_difference_vs_{label}"] = float(np.mean(diff)) if both else None
        result[f"sum_comparator_FED_{label}"] = sum(b["FED"] for r, b in both)
        result[f"sum_paired_difference_{label}"] = sum(diff)
    return result


def confusion(rows):
    opportunities = [x for r in rows for x in r["gate_opportunities"]]
    return dict(opportunities=len(opportunities), accepted=sum(x["accepted"] for x in opportunities),
        **{f"{decision}_{outcome}": sum(x["accepted"] == (decision == "accepted")
              and x["offline_classification"] == outcome for x in opportunities)
           for decision in ("accepted", "rejected") for outcome in ("helpful", "harmful", "neutral", "failure")})


def run_world(spec, directory, graph, times, source, distances, region_ids, cell_to_region, cfg):
    wall_start = time.perf_counter()
    directory.mkdir()
    truth_regions, generation = make_synthetic_truth(times, source, distances, spec["seed"], spec["family"])
    sensor = spec["sensor"]
    measured_regions = noisy_measurements(truth_regions, spec["seed"], sensor["noise_cv"])
    n = len(graph.cells)
    true_cells, measured_cells = np.zeros((len(times), n)), np.zeros((len(times), n))
    for i, region in enumerate(cell_to_region):
        if region >= 0:
            true_cells[:, i] = truth_regions[:, region]
            measured_cells[:, i] = measured_regions[:, region]
    np.savez_compressed(directory / "native_inputs.npz", acquisition_times=times,
        truth_region_FER=truth_regions, measured_region_FER=measured_regions,
        cell_to_region=cell_to_region, region_names=np.asarray(region_ids))
    save(directory / "world_definition.json", dict(**spec, generator=generation))
    field = RateField(times, true_cells)
    methods = list(cfg["primary_methods"])
    if spec["stage"] == "development":
        methods += [m["id"] for m in cfg["development_only_sensitivity"]]
    summary, timing, checks = [], [], []
    count, moves, reached, opportunities = 0, 0, 0, 0
    with gzip.open(directory / "episodes.jsonl.gz", "wt", encoding="utf-8", compresslevel=4) as trace:
        for departure in cfg["departure_fire_times_s"]:
            tables, audit, durations, invariant = build_tables(graph, times, measured_cells,
                sensor["latency_s"], departure, cfg)
            costs = field.interval_costs(departure, cfg["dt_s"], cfg["maximum_actions"])
            oracle, values, _ = future_min_fed_policy(graph, costs)
            snapshot_values = fixed_policy_values(graph, tables["first_snapshot_hold"], costs)
            np.savez_compressed(directory / f"tables_t{departure}.npz", **tables,
                                true_costs=costs, oracle_actions=oracle, oracle_values=values)
            save(directory / f"timestamps_t{departure}.json", audit)
            timing.append(dict(departure_fire_s=departure, shared_bundle_solves=len(durations),
                total_shared_bundle_s=sum(durations), median_shared_bundle_s=float(np.median(durations)),
                p95_shared_bundle_s=float(np.quantile(durations, .95)),
                latest_measurement_max_age_s=max(r["measurement_age_s"] for r in audit)))
            if invariant is not None:
                checks.append(dict(kind="future_delivery_suffix_invariance", **invariant))
            for xy in ((5, 1), (10, 9)):
                state = graph.index[xy]
                independent, _ = forward_min_fed_reference(graph, costs, state)
                assert abs(independent - values[0, state]) < 1e-10
                checks.append(dict(kind="independent_forward_reference", departure_fire_s=departure,
                    start=xy, FED=independent))
            local = []
            for method in methods:
                for start in range(len(graph.cells)):
                    if start in graph.exit_ids:
                        continue
                    result = execute(graph, tables, costs, start, method, cfg, snapshot_values, oracle)
                    if result["reached"]:
                        assert result["FED"] >= values[0, start] - 1e-10
                    record = dict(world_id=spec["id"], stage=spec["stage"], seed=spec["seed"],
                        family=spec["family"], sensor_id=sensor["id"], departure_fire_s=departure,
                        start_id=start, method=method, oracle_FED=float(values[0, start]), **result)
                    trace.write(json.dumps(record, ensure_ascii=True, allow_nan=False, default=convert) + "\n")
                    local.append(record)
                    count += 1
                    moves += result["steps"]
                    reached += result["reached"]
                    opportunities += len(result["gate_opportunities"])
            lookup = {m: {r["start_id"]: r for r in local if r["method"] == m}
                      for m in ("snapshot_hold", "ensemble_mean", "half30")}
            for method in methods:
                rows = [r for r in local if r["method"] == method]
                summary.append(dict(world_id=spec["id"], stage=spec["stage"], seed=spec["seed"],
                    family=spec["family"], sensor_id=sensor["id"], departure_fire_s=departure, method=method,
                    **summarize(rows, lookup["snapshot_hold"], lookup["ensemble_mean"], lookup["half30"], cfg),
                    **confusion(rows)))
    csv_save(directory / "summary.csv", summary)
    save(directory / "timing.json", timing)
    save(directory / "verification.json", dict(episodes=count, moves=moves, reached=reached,
        gate_opportunities=opportunities, checks=checks, online_future_rows_delivered=0))
    files = [p for p in directory.iterdir() if p.is_file()]
    save(directory / "completion_manifest.json", dict(status="COMPLETE_WORLD", spec=spec,
        episodes=count, moves=moves, reached=reached, elapsed_wall_s=time.perf_counter() - wall_start,
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    return summary, count, moves, reached


def load_completed(directory):
    manifest = json.loads((directory / "completion_manifest.json").read_text("utf-8"))
    check(manifest["artifact_sha256"])
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # Restore numeric types for aggregated CSV/report generation.
    text = {"world_id", "stage", "family", "sensor_id", "method"}
    for row in rows:
        for key, value in row.items():
            if key in text:
                continue
            row[key] = None if value == "" else float(value) if any(c in value for c in ".eE") else int(value)
    return rows, manifest["episodes"], manifest["moves"], manifest["reached"]


def run(cfg, out, frozen):
    begin = time.perf_counter()
    check(frozen["input_sha256"])
    check(frozen["protected_previous_sha256"])
    if not (out / "started.json").exists():
        save(out / "started.json", dict(started_UTC=datetime.now(timezone.utc).isoformat()))
    tests = subprocess.run([sys.executable, "-c",
        "import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'src'); import pytest; "
        "raise SystemExit(pytest.main(['-q','tests/test_safestep_robust_gate.py',"
        "'tests/test_safestep_causal_forecast.py','tests/test_safestep_public_benchmark.py','-p','no:cacheprovider']))"],
        cwd=ROOT, capture_output=True, text=True)
    test_path = out / "unit_checks.log"
    if not test_path.exists():
        test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    assert tests.returncode == 0, "Unit checks failed"
    data = load_author_inputs(ROOT / cfg["source_directory"])
    starts = [tuple(c) for c in data["empty"]]
    graph = Graph(starts + list(data["exits"]), data["exits"])
    region_by_cell = [data["mapping"][c] for c in graph.cells]
    regions = sorted({r for c, r in zip(graph.cells, region_by_cell) if c not in graph.exits},
                     key=lambda r: int(r[1:]))
    assert len(regions) == 58 and len(starts) == 61
    times = data["rates"]["Time"]
    source = np.column_stack([data["rates"][r] for r in regions])
    distances = region_distances(graph, region_by_cell, regions)
    cell_to_region = np.array([regions.index(r) if r in regions else -1 for r in region_by_cell])
    graph_path = out / "graph.json"
    if not graph_path.exists():
        save(graph_path, dict(cells=graph.cells, exits=graph.exits,
            region_by_cell=region_by_cell, region_names=regions, cell_to_region=cell_to_region))
    all_summary = []
    completed, episodes, moves, reached = 0, 0, 0, 0
    world_root = out / "worlds"
    world_root.mkdir(exist_ok=True)
    stage_last = None
    for spec in frozen["world_schedule"]:
        if time.perf_counter() - begin > cfg["execution"]["max_compute_wall_s"] - 120:
            raise TimeoutError("Frozen maximum computation time reached; completed worlds retained")
        if spec["stage"] != stage_last:
            check(frozen["input_sha256"])
            stage_last = spec["stage"]
            marker = out / f"stage_{stage_last}_started.json"
            if not marker.exists():
                save(marker, dict(stage=stage_last, UTC=datetime.now(timezone.utc).isoformat(),
                    unchanged_method_hashes=frozen["input_sha256"],
                    no_model_or_threshold_selection=True))
        directory = world_root / spec["id"]
        if (directory / "completion_manifest.json").exists():
            rows, n, m, r = load_completed(directory)
        else:
            # Preserve any partial attempt and use a fresh attempt directory.
            attempt = 2
            while directory.exists():
                directory = world_root / f"{spec['id']}_attempt{attempt}"
                attempt += 1
                if (directory / "completion_manifest.json").exists():
                    break
            if (directory / "completion_manifest.json").exists():
                rows, n, m, r = load_completed(directory)
            else:
                rows, n, m, r = run_world(spec, directory, graph, times, source, distances,
                    regions, cell_to_region, cfg)
        all_summary.extend(rows)
        completed += 1
        episodes += n
        moves += m
        reached += r
        print(json.dumps(dict(completed_worlds=completed, total_worlds=len(frozen["world_schedule"]),
            stage=spec["stage"], world=spec["id"], episodes=episodes, moves=moves,
            elapsed_s=time.perf_counter() - begin)), flush=True)
    assert completed == cfg["execution"]["expected_worlds"]
    assert episodes == cfg["execution"]["expected_episodes"]
    csv_save(out / "all_world_departure_summary.csv", all_summary)
    check(frozen["input_sha256"])
    check(frozen["protected_previous_sha256"])
    save(out / "run_results.json", dict(status="COMPLETE_PLANNED_SYNTHETIC_MATRIX",
        worlds=completed, episodes=episodes, executed_moves=moves, reached=reached,
        elapsed_wall_s=time.perf_counter() - begin, original_physical_fires=1,
        independent_physical_fire_claim=False, LLM_requests=0, training_updates=0,
        FDS_runs=0, sealed_final48_accessed=False, unit_tests=19))
    files = [p for p in out.iterdir() if p.is_file()]
    world_manifests = [p for p in world_root.rglob("completion_manifest.json")]
    files += world_manifests
    save(out / "completion_manifest.json", dict(status="COMPLETE_COMPUTATION_PENDING_INDEPENDENT_REVIEW",
        completed_UTC=datetime.now(timezone.utc).isoformat(),
        protected_previous_files_verified=len(frozen["protected_previous_sha256"]),
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", worlds=completed, episodes=episodes,
        executed_moves=moves, reached=reached), ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text("utf-8"))
    out = ROOT / cfg["output_directory"]
    if args.prepare:
        prepare(cfg, out)
    elif args.run:
        frozen = json.loads((out / "frozen_plan.json").read_text("utf-8"))
        try:
            run(cfg, out, frozen)
        except Exception:
            # Unique failure log, never replace completed results.
            (out / f"failure_{time.time_ns()}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    else:
        parser.error("Choose --prepare or --run")


if __name__ == "__main__":
    main()
