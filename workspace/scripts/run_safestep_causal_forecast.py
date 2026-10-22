"""Bounded, frozen-plan past-only forecasting experiment (one known fire).

Usage: --prepare, then --run. Both refuse to overwrite a previous run.
All online choices are made from detached native observation prefixes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
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
from fire_evac.safestep_causal_forecast import (
    available_prefix, causal_first_actions, predicted_interval_costs, fixed_policy_values,
)

CONFIG = ROOT / "schemas/safestep_causal_forecast_20261018.json"
REPORT = ROOT / "reports/SafeStep历史观测预测与路径规划开发实验_2026-10-18.md"
LABELS = {
    "snapshot_hold": "最新观测保持不变",
    "trend_w30_c60": "30秒历史趋势／60秒外推",
    "trend_w30_c60_half": "30秒历史趋势／半斜率",
    "trend_w15_c60": "15秒历史／60秒外推",
    "trend_w60_c60": "60秒历史／60秒外推",
    "trend_w30_c30": "30秒历史／30秒外推",
    "trend_w30_c120": "30秒历史／120秒外推",
    "full_future_reference": "完整未来参照（额外信息）",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False, default=scalar), encoding="utf-8")


def write_csv(path, rows):
    if path.exists() or not rows:
        raise ValueError(f"New file and nonempty rows required: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def checked_hashes(hashes):
    for rel, expected in hashes.items():
        if sha(ROOT / rel) != expected:
            raise AssertionError(f"Frozen file changed: {rel}")


def protected_previous():
    # Curated manifests only. No traversal or reading of sealed crowd scenarios.
    manifests = [
        ROOT / "outputs/safestep_public_benchmark_20261017/frozen_plan.json",
        ROOT / "outputs/safestep_public_benchmark_20261017/completion_manifest.json",
        ROOT / "outputs/safestep_five_residual_cases_20261017/completion_manifest.json",
        ROOT / "outputs/safestep_stage_closure_20261017/completion_manifest.json",
    ]
    protected = {}
    for manifest in manifests:
        data = json.loads(manifest.read_text("utf-8"))
        for key in ("protected_prior_sha256", "artifact_sha256"):
            for rel, expected in data.get(key, {}).items():
                if rel in protected and protected[rel] != expected:
                    raise AssertionError(f"Conflicting archived hashes: {rel}")
                protected[rel] = expected
        protected[manifest.relative_to(ROOT).as_posix()] = sha(manifest)
    checked_hashes(protected)
    return protected


def prepare(cfg, out):
    if out.exists() or REPORT.exists():
        raise FileExistsError("Run directory/report already exists; preserve it")
    source = ROOT / cfg["source_directory"]
    inputs = [CONFIG, Path(__file__),
              ROOT / "src/fire_evac/safestep_causal_forecast.py",
              ROOT / "src/fire_evac/safestep_public_benchmark.py",
              ROOT / "src/fire_evac/safestep_author_numerics.py",
              ROOT / "tests/test_safestep_causal_forecast.py",
              ROOT / "tests/test_safestep_public_benchmark.py",
              source / "main.py", source / "coordinates2areanumber.py",
              source / "FED rate - room24.xlsx", source / "fire hazard zones FED - room24.xlsx",
              ROOT / cfg["previous_benchmark"] / "episode_metrics.csv"]
    protected = protected_previous()
    out.mkdir()
    save(out / "frozen_plan.json", {
        "frozen_at_UTC": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_NEW_REAL_DATA_POLICY_SCORING",
        "configuration": cfg,
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "protected_previous_sha256": protected,
        "prior_knowledge": "All departures and five residual cases were inspected in previous work; no confirmation status claimed",
    })
    print(json.dumps({"status": "PREPARED", "protected_files": len(protected),
                      "output": str(out)}, ensure_ascii=True), flush=True)


def unit_checks(out):
    proc = subprocess.run([
        sys.executable, "-c",
        "import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'src'); "
        "import pytest; raise SystemExit(pytest.main(['-q',"
        "'tests/test_safestep_causal_forecast.py','tests/test_safestep_public_benchmark.py',"
        "'-p','no:cacheprovider']))",
    ], cwd=ROOT, capture_output=True, text=True)
    (out / "unit_checks.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise AssertionError("Unit checks failed; see unit_checks.log")


def summarize(rows, baseline, tol):
    both = [(r, baseline[r["case_id"]]) for r in rows
            if r["reached"] and baseline[r["case_id"]]["reached"]]
    success = [r for r in rows if r["reached"]]
    differences = [r["FED"] - b["FED"] for r, b in both]
    mean = float(np.mean([r["FED"] for r, b in both])) if both else None
    base_mean = float(np.mean([b["FED"] for r, b in both])) if both else None
    return {
        "n": len(rows), "reached": len(success), "paired_reached": len(both),
        "mean_FED": float(np.mean([r["FED"] for r in success])) if success else None,
        "maximum_FED": max((r["FED"] for r in success), default=None),
        "mean_travel_s": float(np.mean([r["travel_s"] for r in success])) if success else None,
        "mean_revisits": float(np.mean([r["revisits"] for r in success])) if success else None,
        "lower_FED_than_snapshot": sum(d < -tol for d in differences),
        "equal_FED_to_snapshot": sum(abs(d) <= tol for d in differences),
        "higher_FED_than_snapshot": sum(d > tol for d in differences),
        "mean_paired_FED_minus_snapshot": float(np.mean(differences)) if both else None,
        "relative_mean_reduction_vs_snapshot": (base_mean - mean) / base_mean if base_mean else None,
        "largest_FED_gain": max((-d for d in differences), default=None),
        "largest_FED_harm": max(differences, default=None),
    }


def evaluate_actions(graph, policies, snap_policy, costs, runs, method, departure, tol):
    """One changed action, then resume snapshot; scorer only, never a gate."""
    continuation = fixed_policy_values(graph, snap_policy, costs)
    rows = []
    for run in runs:
        for step in run["executed_steps"]:
            k = step["step"]
            i = graph.index[tuple(step["source"])]
            a = int(policies[k, i])
            baseline_a = int(snap_policy[k, i])
            if a == baseline_a:
                continue
            next_i = graph.destination(i, a)
            base = float(continuation[k, i])
            changed = float(costs[k, i] + continuation[k + 1, next_i])
            if not np.isfinite(base) or not np.isfinite(changed):
                label = "deadline_failure"
                difference = None
            else:
                difference = changed - base
                label = "harmful" if difference > tol else "helpful" if difference < -tol else "neutral"
            rows.append(dict(case_id=run["case_id"], method=method,
                departure_fire_s=departure, step=k, cell_x=graph.cells[i][0], cell_y=graph.cells[i][1],
                action=a, snapshot_action=baseline_a,
                one_action_then_snapshot_FED=changed if np.isfinite(changed) else None,
                snapshot_continuation_FED=base if np.isfinite(base) else None,
                difference_FED=difference, classification=label))
    return rows


def run_experiment(cfg, out, frozen):
    wall_start = time.perf_counter()
    if (out / "started.json").exists():
        raise FileExistsError("This run was already started; preserve cached artifacts")
    checked_hashes(frozen["input_sha256"])
    checked_hashes(frozen["protected_previous_sha256"])
    save(out / "started.json", {"UTC": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__})
    unit_checks(out)
    dt = cfg["movement"]["dt_s"]
    horizon = cfg["movement"]["maximum_actions"]
    tol = cfg["analysis"]["paired_tolerance_FED"]
    loaded = load_author_inputs(ROOT / cfg["source_directory"], cfg["primary_departure_fire_s"])
    starts = [tuple(x) for x in loaded["empty"]]
    assert len(starts) == 61
    graph = Graph(starts + list(loaded["exits"]), loaded["exits"])
    native_times = loaded["rates"]["Time"]
    measured = np.column_stack([loaded["rates"][loaded["mapping"][c]]
        if c not in graph.exits else np.zeros(len(native_times)) for c in graph.cells])
    truth = RateField(native_times, measured)
    assert max(cfg["departure_fire_times_s"]) + horizon * dt <= native_times[-1] + 1e-9
    unique_regions = {}
    for i, cell in enumerate(starts):
        unique_regions.setdefault(loaded["mapping"][cell], i)
    region_ids = list(unique_regions.values())
    assert len(region_ids) == 58
    prior = read_csv(ROOT / cfg["previous_benchmark"] / "episode_metrics.csv")
    prior_oracle = {r["case_id"]: float(r["FED"]) for r in prior
                    if r["method"] == "full_future_minFED_reference"}
    rng = np.random.default_rng(cfg["analysis"]["random_order_seed"])
    online = cfg["primary_methods"]
    episodes, all_runs, timings, audits, errors, changed_actions, summaries = [], [], [], [], [], [], []
    truth_cache, snapshot_cache = {}, {}
    oracle_matches, forward_checks, no_future_checks = 0, [], []
    plan_paths = out / "policy_tables"
    plan_paths.mkdir()

    def time_limit():
        if time.perf_counter() - wall_start > cfg["stages"]["maximum_wall_s"]:
            raise TimeoutError("Frozen experiment wall-time limit reached")

    def build_policies(departure, methods, stage):
        policies = {m["id"]: np.full((horizon, len(graph.cells)), -1, dtype=int) for m in methods}
        for k in range(horizon):
            time_limit()
            decision = departure + k * dt
            history = available_prefix(native_times, measured, decision)
            assert history.times[-1] <= decision + 1e-9
            age = decision - history.times[-1]
            assert -1e-9 <= age < 5. + 1e-9
            if stage == "primary":
                audits.append(dict(departure_fire_s=departure, step=k, decision_fire_s=decision,
                    latest_observed_fire_s=float(history.times[-1]), observation_age_s=age,
                    last_native_index=int(np.searchsorted(native_times, history.times[-1])),
                    observed_rows=len(history.times), future_rows_delivered=0))
            for idx in rng.permutation(len(methods)):
                method = methods[int(idx)]
                begin = time.perf_counter()
                actions, _, _, forecast = causal_first_actions(graph, history, dt, horizon - k,
                    method["window_s"], method["projection_cap_s"], method["slope_gain"])
                elapsed = time.perf_counter() - begin
                policies[method["id"]][k] = actions
                timings.append(dict(stage=stage, method=method["id"], departure_fire_s=departure,
                    step=k, remaining_steps=horizon - k, all_state_solve_s=elapsed))
                if stage == "primary":
                    for ahead in (10, 30, 60):
                        target = decision + ahead
                        if target > native_times[-1] + 1e-9:
                            continue
                        predicted = forecast.sample(target)[region_ids]
                        actual = truth.sample(min(target, native_times[-1]))[region_ids]
                        difference = predicted - actual
                        errors.append(dict(method=method["id"], departure_fire_s=departure,
                            step=k, decision_fire_s=decision, ahead_s=ahead,
                            unique_regions=58, MAE_FER=float(np.mean(np.abs(difference))),
                            MSE_FER=float(np.mean(difference**2)),
                            bias_FER=float(np.mean(difference))))
            if stage == "primary" and k in (0, 1, 7):
                # Explicit real-data prefix-invariance check. Future data is made
                # wildly different but never delivered to either online planner.
                altered = measured.copy()
                altered[native_times > decision + 1e-9] = 12345.
                h2 = available_prefix(native_times, altered, decision)
                assert np.array_equal(history.times, h2.times)
                assert np.array_equal(history.rates, h2.rates)
                primary = methods[1]
                a2, _, _, f2 = causal_first_actions(graph, h2, dt, horizon - k,
                    primary["window_s"], primary["projection_cap_s"], primary["slope_gain"])
                assert np.array_equal(a2, policies[primary["id"]][k])
                f1 = causal_first_actions(graph, history, dt, horizon - k,
                    primary["window_s"], primary["projection_cap_s"], primary["slope_gain"])[3]
                assert np.array_equal(predicted_interval_costs(f1, decision, dt, horizon - k),
                                      predicted_interval_costs(f2, decision, dt, horizon - k))
                no_future_checks.append(dict(departure_fire_s=departure, step=k,
                    decision_fire_s=decision, future_replaced_with=12345., identical=True))
        np.savez_compressed(plan_paths / f"{stage}_t{departure}.npz", **policies)
        return policies

    def score_stage(departure, policies, stage, costs, future_values):
        records = []
        for method, policy in policies.items():
            method_runs = []
            for start in starts:
                si = graph.index[start]
                case_id = f"t{departure:04d}_x{start[0]:02d}_y{start[1]:02d}"
                run = rollout(graph, policy, costs, si)
                if run["reached"]:
                    assert run["FED"] >= future_values[0, si] - tol
                row = dict(stage=stage, case_id=case_id, departure_fire_s=departure,
                    start_x=start[0], start_y=start[1], start_region=loaded["mapping"][start],
                    method=method, reached=bool(run["reached"]), steps=run["steps"],
                    travel_s=run["steps"] * dt, FED=run["FED"], revisits=run["revisits"],
                    oracle_FED=float(future_values[0, si]),
                    excess_over_oracle=run["FED"] - float(future_values[0, si]) if run["reached"] else None,
                    exit_region=loaded["mapping"][tuple(run["exit"])] if run["exit"] else "")
                record = dict(**row, path=run["path"], executed_steps=run["trace"])
                records.append(row)
                all_runs.append(record)
                method_runs.append(record)
            if method not in ("snapshot_hold", "full_future_reference"):
                changed_actions.extend(evaluate_actions(graph, policy, snapshot_cache[departure],
                    costs, method_runs, method, departure, tol))
        episodes.extend(records)
        base = {r["case_id"]: r for r in episodes
                if r["method"] == "snapshot_hold" and r["departure_fire_s"] == departure}
        for method in policies:
            summaries.append(dict(stage=stage, departure_fire_s=departure, method=method,
                **summarize([r for r in records if r["method"] == method], base, tol)))
        # Incremental outputs preserve an interrupted run without changing raw inputs.
        save(out / f"{stage}_t{departure}_episodes.json", records)
        print(json.dumps(dict(stage=stage, completed_departure=departure,
            rollouts_so_far=len(episodes), elapsed_s=time.perf_counter() - wall_start)), flush=True)

    # First untimed warmup uses only an already delivered prefix after freezing.
    warm = available_prefix(native_times, measured, cfg["departure_fire_times_s"][0])
    for method in online:
        causal_first_actions(graph, warm, dt, horizon, method["window_s"],
                             method["projection_cap_s"], method["slope_gain"])
    for departure in cfg["departure_fire_times_s"]:
        costs = truth.interval_costs(departure, dt, horizon)
        oracle, values, _ = future_min_fed_policy(graph, costs)
        truth_cache[departure] = (costs, values)
        for start in starts:
            case_id = f"t{departure:04d}_x{start[0]:02d}_y{start[1]:02d}"
            assert abs(float(values[0, graph.index[start]]) - prior_oracle[case_id]) < tol
            oracle_matches += 1
        for xy in ((5, 1), (10, 9)):
            i = graph.index[xy]
            independent, _ = forward_min_fed_reference(graph, costs, i)
            assert abs(independent - values[0, i]) < tol
            forward_checks.append(dict(departure_fire_s=departure, start=xy,
                forward_FED=independent, backward_FED=float(values[0, i])))
        policies = build_policies(departure, online, "primary")
        snapshot_cache[departure] = policies["snapshot_hold"]
        policies["full_future_reference"] = oracle
        score_stage(departure, policies, "primary", costs, values)
    primary_summary = list(summaries)
    triggered = [r for r in primary_summary if r["method"].startswith("trend_")
                 and r["reached"] == 61 and r["paired_reached"] == 61
                 and r["relative_mean_reduction_vs_snapshot"] >= .01]
    sensitivity = bool(triggered)
    save(out / "sensitivity_decision.json", {"triggered": sensitivity,
        "frozen_rule": cfg["stages"]["sensitivity"], "trigger_rows": triggered,
        "new_primary_selection": False})
    if sensitivity:
        for departure in cfg["departure_fire_times_s"]:
            policies = build_policies(departure, cfg["sensitivity_methods"], "sensitivity")
            score_stage(departure, policies, "sensitivity", *truth_cache[departure])
    write_csv(out / "episode_metrics.csv", episodes)
    write_csv(out / "departure_summary.csv", summaries)
    write_csv(out / "planning_times.csv", timings)
    write_csv(out / "observation_audit.csv", audits)
    write_csv(out / "forecast_errors_by_origin.csv", errors)
    if changed_actions:
        write_csv(out / "changed_action_counterfactuals.csv", changed_actions)
    save(out / "rollouts.json", all_runs)
    save(out / "no_future_checks.json", no_future_checks)
    save(out / "independent_reference_checks.json", forward_checks)
    error_summary = []
    for method in [m["id"] for m in online]:
        for ahead in (10, 30, 60):
            rows = [r for r in errors if r["method"] == method and r["ahead_s"] == ahead]
            error_summary.append(dict(method=method, ahead_s=ahead, origins=len(rows),
                unique_regions_per_origin=58, MAE_FER=float(np.mean([r["MAE_FER"] for r in rows])),
                RMSE_FER=float(np.sqrt(np.mean([r["MSE_FER"] for r in rows]))),
                bias_FER=float(np.mean([r["bias_FER"] for r in rows]))))
    write_csv(out / "forecast_error_summary.csv", error_summary)
    timing_summary = []
    for method in dict.fromkeys(r["method"] for r in timings):
        seconds = [r["all_state_solve_s"] for r in timings if r["method"] == method]
        timing_summary.append(dict(method=method, global_policy_solves=len(seconds),
            mean_s=float(np.mean(seconds)), median_s=float(np.median(seconds)),
            p95_s=float(np.quantile(seconds, .95)), maximum_s=max(seconds)))
    write_csv(out / "timing_summary.csv", timing_summary)
    action_summary = []
    for method in dict.fromkeys(r["method"] for r in changed_actions):
        rows = [r for r in changed_actions if r["method"] == method]
        action_summary.append(dict(method=method, changed_visited_actions=len(rows),
            **{label: sum(r["classification"] == label for r in rows)
               for label in ("helpful", "harmful", "neutral", "deadline_failure")}))
    if action_summary:
        write_csv(out / "changed_action_summary.csv", action_summary)
    paired_rows, context_rows = [], []
    baseline = {r["case_id"]: r for r in episodes if r["method"] == "snapshot_hold"}
    for r in episodes:
        if r["method"] in ("snapshot_hold", "full_future_reference"):
            continue
        b = baseline[r["case_id"]]
        diff = r["FED"] - b["FED"] if r["reached"] and b["reached"] else None
        paired_rows.append(dict(case_id=r["case_id"], method=r["method"],
            departure_fire_s=r["departure_fire_s"], start_region=r["start_region"],
            start_x=r["start_x"], start_y=r["start_y"],
            jointly_reached=bool(r["reached"] and b["reached"]),
            forecast_FED=r["FED"], snapshot_FED=b["FED"], difference_FED=diff,
            forecast_steps=r["steps"], snapshot_steps=b["steps"],
            classification="failure" if diff is None else "harmful" if diff > tol else
                           "helpful" if diff < -tol else "neutral"))
    write_csv(out / "paired_case_differences.csv", paired_rows)
    old_snapshot = {r["case_id"]: float(r["FED"]) for r in prior
                    if r["method"] == "current_snapshot_replan"}
    for departure in cfg["departure_fire_times_s"]:
        rows = [r for r in episodes if r["method"] == "snapshot_hold"
                and r["departure_fire_s"] == departure]
        context_rows.append(dict(departure_fire_s=departure,
            new_native_hold_mean_FED=float(np.mean([r["FED"] for r in rows])),
            old_interpolated_snapshot_mean_FED=float(np.mean([old_snapshot[r["case_id"]] for r in rows])),
            changed_case_FED=sum(abs(r["FED"] - old_snapshot[r["case_id"]]) > tol for r in rows),
            interpretation="Both native observation timing and common shrinking-horizon implementation differ; not a new forecasting gain"))
    write_csv(out / "previous_snapshot_context.csv", context_rows)
    metadata = dict(status="COMPLETED_DEVELOPMENT_EXPERIMENT", rollouts=len(episodes),
        executed_moves=sum(r["steps"] for r in episodes), reached=sum(r["reached"] for r in episodes),
        start_departure_combinations=61 * len(cfg["departure_fire_times_s"]), independent_fires=1,
        unique_hazard_regions=58, native_observation_audits=len(audits),
        maximum_observation_age_s=max(r["observation_age_s"] for r in audits),
        future_rows_delivered=0, future_suffix_invariance_checks=len(no_future_checks),
        archived_oracle_values_matched=oracle_matches,
        independent_forward_reference_checks=len(forward_checks),
        sensitivity_triggered=sensitivity, changed_action_diagnostics=len(changed_actions),
        policy_solves=len(timings), unit_tests=13,
        elapsed_wall_s_before_report=time.perf_counter() - wall_start,
        LLM_requests=0, training_updates=0, FDS_runs=0, sealed_final48_accessed=False)
    save(out / "verification.json", metadata)
    # Save all aggregates once, making report regeneration possible without rerunning policies.
    save(out / "analysis_results.json", dict(metadata=metadata, summary=summaries,
        forecast_errors=error_summary, timings=timing_summary, action_diagnostics=action_summary,
        old_snapshot_context=context_rows,
        worst_harm=sorted([r for r in paired_rows if r["difference_FED"] is not None],
                          key=lambda x: x["difference_FED"], reverse=True)[:15],
        best_gain=sorted([r for r in paired_rows if r["difference_FED"] is not None],
                         key=lambda x: x["difference_FED"])[:15]))
    make_report_and_figure(cfg, out, summaries, error_summary, timing_summary,
                           action_summary, metadata, paired_rows)
    checked_hashes(frozen["input_sha256"])
    checked_hashes(frozen["protected_previous_sha256"])
    artifacts = [p for p in out.rglob("*") if p.is_file()] + [REPORT]
    save(out / "completion_manifest.json", dict(status="COMPLETE", completed_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_wall_s=time.perf_counter() - wall_start,
        protected_previous_files_verified=len(frozen["protected_previous_sha256"]),
        input_files_verified=len(frozen["input_sha256"]),
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}))
    print(json.dumps(dict(status="COMPLETE", metadata=metadata,
        primary_300=[r for r in summaries if r["departure_fire_s"] == 300 and r["stage"] == "primary"]),
        ensure_ascii=True, default=scalar), flush=True)


def make_report_and_figure(cfg, out, summaries, errors, timing, actions, meta, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m["id"] for m in cfg["primary_methods"]]
    styles = [("#0072B2", "o", "-"), ("#D55E00", "s", "-"),
              ("#009E73", "^", "-"), ("#666666", "x", "--")]
    with plt.rc_context({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for method, (color, marker, line) in zip(methods + ["full_future_reference"], styles):
            rows = [r for r in summaries if r["method"] == method]
            for ax, field in zip(axes, ("mean_FED", "mean_travel_s")):
                ax.plot([r["departure_fire_s"] for r in rows], [r[field] for r in rows],
                    color=color, marker=marker, linestyle=line, label=LABELS[method], markersize=4)
        for ax in axes:
            ax.set_xlabel("出发时的火灾时刻／秒")
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel("61个起点的平均FED")
        axes[1].set_ylabel("平均行走时间／秒")
        axes[0].set_title("在线方法仅用当时已有的5秒采样行")
        axes[1].set_title("同时检查暴露和到达时间")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
        fig.suptitle("一个已知火灾上的开发实验；完整未来仅作参照", fontsize=11)
        fig.savefig(out / "causal_forecast_comparison.png", dpi=180)
        fig.savefig(out / "causal_forecast_comparison.svg")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for method, (color, marker, _) in zip(methods[1:], styles[1:]):
            rows = [r for r in summaries if r["method"] == method]
            t = [r["departure_fire_s"] for r in rows]
            axes[0].plot(t, [100 * r["relative_mean_reduction_vs_snapshot"] for r in rows],
                         color=color, marker=marker, label=LABELS[method])
            axes[1].plot(t, [r["lower_FED_than_snapshot"] for r in rows], color=color,
                         marker=marker, linestyle="-", label=LABELS[method] + "：更低")
            axes[1].plot(t, [r["higher_FED_than_snapshot"] for r in rows], color=color,
                         marker=marker, linestyle="--", label=LABELS[method] + "：更高")
        axes[0].axhline(0, color="black", linewidth=.8)
        axes[0].set_ylabel("相对快照的平均FED降低／%（负值为恶化）")
        axes[1].set_ylabel("配对起点数／61")
        axes[1].set_ylim(0, 61)
        for ax in axes:
            ax.set_xlabel("出发时的火灾时刻／秒")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("全部起点配对呈现：改善与伤害同时保留", fontsize=11)
        fig.savefig(out / "paired_benefit_and_harm.png", dpi=180)
        fig.savefig(out / "paired_benefit_and_harm.svg")
        plt.close(fig)
    primary = [r for r in summaries if r["departure_fire_s"] == 300 and r["stage"] == "primary"]
    ptable = "\n".join(f"| {LABELS[r['method']]} | {r['reached']}/61 | {r['mean_FED']:.6f} | "
                      f"{r['mean_travel_s']:.2f} | {r['lower_FED_than_snapshot']}/{r['equal_FED_to_snapshot']}/{r['higher_FED_than_snapshot']} |"
                      for r in primary)
    detailed = []
    for departure in cfg["departure_fire_times_s"]:
        by = {r["method"]: r for r in summaries if r["departure_fire_s"] == departure}
        main, damp = by[methods[1]], by[methods[2]]
        detailed.append(f"| {departure} | {by[methods[0]]['mean_FED']:.6f} | {main['mean_FED']:.6f} | "
            f"{100*main['relative_mean_reduction_vs_snapshot']:.2f}% | {main['lower_FED_than_snapshot']}/{main['equal_FED_to_snapshot']}/{main['higher_FED_than_snapshot']} | "
            f"{damp['mean_FED']:.6f} | {100*damp['relative_mean_reduction_vs_snapshot']:.2f}% |")
    etable = "\n".join(f"| {LABELS[r['method']]} | {r['ahead_s']} | {r['origins']} | {r['MAE_FER']:.8f} | {r['RMSE_FER']:.8f} |" for r in errors)
    ttable = "\n".join(f"| {LABELS[r['method']]} | {r['global_policy_solves']} | {r['median_s']*1000:.2f} | {r['p95_s']*1000:.2f} |" for r in timing)
    atable = "\n".join(f"| {LABELS[r['method']]} | {r['changed_visited_actions']} | {r['helpful']} | {r['harmful']} | {r['neutral']} | {r['deadline_failure']} |" for r in actions)
    candidates = [r for r in pairs if r["method"] == "trend_w30_c60" and r["jointly_reached"]]
    worst = max(candidates, key=lambda r: r["difference_FED"])
    best = min(candidates, key=lambda r: r["difference_FED"])
    main300 = next(r for r in primary if r["method"] == "trend_w30_c60")
    main_all = [r for r in summaries if r["method"] == "trend_w30_c60"]
    pos = sum(r["relative_mean_reduction_vs_snapshot"] > 1e-12 for r in main_all)
    neg = sum(r["relative_mean_reduction_vs_snapshot"] < -1e-12 for r in main_all)
    rel = out.relative_to(ROOT).as_posix()
    def link(name):
        return (ROOT / rel / name).as_posix()
    sensitivity = [r for r in summaries if r["stage"] == "sensitivity"]
    stable = "未触发固定敏感性分支。"
    if sensitivity:
        stable = "\n".join(["固定敏感性分支已按冻结条件触发；所有参数都在本轮计分前指定，没有改选主方法。", "",
            "| 方法 | 300秒平均FED | 相对同轮快照降低 | 更低／相同／更高 |", "|---|---:|---:|---:|"] +
            [f"| {LABELS[r['method']]} | {r['mean_FED']:.6f} | {100*r['relative_mean_reduction_vs_snapshot']:.2f}% | "
             f"{r['lower_FED_than_snapshot']}/{r['equal_FED_to_snapshot']}/{r['higher_FED_than_snapshot']} |"
             for r in sensitivity if r["departure_fire_s"] == 300])
    content = f'''# SafeStep历史观测预测与路径规划开发实验

日期：2026-10-18。状态：已完成。研究对象为同一份已查看过的公开火灾危险率；本轮属于开发实验。

## 结果先说清楚

本轮实际完成{meta['rollouts']}条完整方法轨迹、{meta['executed_moves']}个执行动作，{meta['reached']}条到达网格出口。主趋势方法在9个出发时刻中有{pos}个平均FED较快照降低、{neg}个升高。

300秒出发时，主趋势相对同轮快照的平均FED降低{100*main300['relative_mean_reduction_vs_snapshot']:.2f}%，61个起点中{main300['lower_FED_than_snapshot']}个降低、{main300['equal_FED_to_snapshot']}个近似相同、{main300['higher_FED_than_snapshot']}个升高。负的“降低百分比”表示恶化；不能只看改善案例。

| 方法 | 到达 | 平均FED | 平均行走秒数 | 比快照更低／相同／更高 |
|---|---:|---:|---:|---:|
{ptable}

## 为什么做这一轮

此前R24两条路线的排序会因未来危险率变化而反转。本轮检验一个更窄的问题：仅靠当时已有的危险率历史做简单预测，是否能降低实际路线暴露？参数没有针对R24或已知最优路线拟合。全部61个起点、9个出发时刻都呈现。

## 实验设计和信息边界

在线方法都获得相同的全图5秒间隔原生FER历史及地图、当前位置、决策时间。最新数据行必须满足时间戳不晚于决策时刻。例如206.67秒决策时可用到205秒，不用210秒数据插出“当前测量”。允许使用出发前历史，但不计出发前暴露。

快照方法保持最新观测不变。主趋势方法用最新30秒内的7个采样估计OLS斜率，锚定最新测量，最多外推60秒后保持；预测率低于零时置零。半斜率方法预先指定为次要诊断。这里60秒是趋势延伸上限，并非只规划60秒。

三者使用同一时间展开动态规划求解器、同一剩余截止步数45−k、同一动作约束和计分目标。每次仅执行第一步，再依据新观测重算。2米一步、速度0.3米/秒、最多45步，不包含等待或人群拥堵。

真实计分器对原始FER分段线性积分，按出发格驻留计入FED。在线模块只接收已复制、只读的历史前缀。完整未来参照仅用于计分和衡量空间，未参与在线建议或门控。

## 全部出发时刻

| 出发秒数 | 快照平均FED | 主趋势平均FED | 主趋势降低 | 主趋势更低／同／更高 | 半斜率平均FED | 半斜率降低 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detailed)}

![平均暴露和行走时间]({link('causal_forecast_comparison.png')})

![配对改善和伤害]({link('paired_benefit_and_harm.png')})

图中连线只连接已计算的9个出发时刻。没有平滑、置信区间或独立样本显著性检验。每个时刻共用同一个火灾和61个起点。

## 预测误差与控制收益分别评价

以下预测误差按58个独立区域标签等权，避免R56～R58共享网格造成重复加权。每个决策时刻用同样历史产生10、30、60秒后的预测，再由离线评分器对照实际FER。超出900秒覆盖的目标时间不评分。多个出发时刻可能含重复绝对观测时刻，均属开发条件的重复测量。

| 方法 | 预测提前秒数 | 观测起点数 | FER平均绝对误差 | FER均方根误差 |
|---|---:|---:|---:|---:|
{etable}

预测误差更小不自动等于路线更好；只有改变路线排序、并在执行后降低FED，才形成控制收益。反过来，局部预测不准也未必会改变动作。

## 有害改道的两个口径

逐回合比较见配对表：从同一起点、同一出发时刻分别执行完整趋势策略和快照策略，比较最终FED。主方法最大改善来自{best['case_id']}（{best['start_region']}），FED差额{best['difference_FED']:.8f}；最大增加来自{worst['case_id']}（{worst['start_region']}），差额{worst['difference_FED']:.8f}。这里差额为趋势减快照。

另对趋势轨迹上每次与“同位置、同时间的快照动作”不同的动作做诊断：仅执行这一步趋势动作，然后全部恢复快照策略，与直接执行快照的完整后续比较。这回答一次改动作是否有帮助，不等于持续使用预测策略的总效应。重叠机会不能相加为总收益，也不是独立样本。

| 方法 | 不同动作机会 | 有益 | 有害 | 中性 | 截止时间失败 |
|---|---:|---:|---:|---:|---:|---:|
{atable}

## 固定参数敏感性

{stable}

其他出发时刻的全部参数结果保存在departure_summary.csv；不以300秒单点或最佳参数代替整体评价。

## 计算时间

| 方法 | 全图策略求解次数 | 中位毫秒 | 95分位毫秒 |
|---|---:|---:|---:|
{ttable}

方法顺序按固定种子随机排列，三个主方法预热后计时。每次计算当前时刻所有网格的首动作表，再在61个起点间共享。上表是全图策略求解耗时，不能当作61次独立推理，也不包括真实传感器或通信延迟。本轮执行至报告前耗时约{meta['elapsed_wall_s_before_report']:.1f}秒；不为凑满1—2小时增加重复运行。

## 验证与局限

- 13项单元检查通过，包含历史边界、未来后缀扰动、解析积分、趋势改变动作、共同截止时域和失败处理。
- {meta['native_observation_audits']}个原生观测时间检查通过，最大观测年龄{meta['maximum_observation_age_s']:.6f}秒；交给在线规划器的未来采样行数为0。
- {meta['future_suffix_invariance_checks']}项真实数据检查中，把未观察未来改成极大值，历史、预测积分和动作完全一致。
- {meta['archived_oracle_values_matched']}项完整未来最小FED与既有缓存一致；{meta['independent_forward_reference_checks']}项另用正向最短路核对。所有成功轨迹的FED均不低于对应参照。
- 新快照采用原生采样保持与统一剩余时域，和旧报告的理想插值快照口径不同；差异另列previous_snapshot_context.csv，不能混作新增预测收益。
- 没有LLM/API调用、PPO/DQN训练或FDS运行，48个封存最终人群场景未读取。
- 全部结果来自一个已知火灾；本轮固定方案提高可追溯性，但不把已知数据变成独立确认集。
- FED是模型中的累积剂量指标。网格出口到达、剂量变化和预测误差不能直接解释为真人伤亡变化。

## 本轮如何使用

本轮回答“简单的历史趋势预测是否有开发价值”，不证明跨火灾有效，也未检验LLM。应先检查收益是否伴随有害改道，以及参数变化是否使结论反转。只有在另行提供独立火灾条件和冻结验证方案后，才进入新方法效果确认。若独立材料仍缺失，将本轮作为复现章节的探索性扩展即可。

## 可复查文件

- [冻结方案]({link('frozen_plan.json')})
- [分时刻和参数汇总]({link('departure_summary.csv')})
- [全部配对起点]({link('paired_case_differences.csv')})
- [逐动作反事实诊断]({link('changed_action_counterfactuals.csv')})
- [观测时间审计]({link('observation_audit.csv')})
- [完整轨迹]({link('rollouts.json')})
- [验证记录]({link('verification.json')})

来源为项目内已固定版本的SafeStep公开地图和危险率工作表，来源与版本见既有复现报告及本轮输入哈希。
'''
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_text(content, encoding="utf-8")


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
            run_experiment(cfg, out, frozen)
        except Exception:
            failure = out / "failure.txt"
            if not failure.exists():
                failure.write_text(traceback.format_exc(), encoding="utf-8")
            raise
    else:
        parser.error("Choose --prepare or --run")


if __name__ == "__main__":
    main()
