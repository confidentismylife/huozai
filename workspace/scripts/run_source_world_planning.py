"""Fixed development comparison of ambiguity-aware reference plans."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.counterfactual_shield import reactive_followup_action
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.terminal_shield import ObservableHistoryWrapper
from fire_evac.source_world_planning import (
    PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds,
    evaluate_plan, select_reference_plan,
)

ARMS = ("live_graded", "live_full", "wait3", "split50_full", "fixed_forecast_full",
        "point_source_planner", "equal_world_planner", "minimax_world_planner", "oracle_source_planner")
BASELINE_INDICES = (0, 1, 3, 9, 5)


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector(outcome):
    return np.asarray([getattr(outcome, field) for field in OUTCOME_FIELDS], dtype=np.float64)


def make_env(scenarios):
    return ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(
        CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), scenarios), "none"))


def replay_world(env, scenario, population_seed, commands):
    observation, _ = env.reset(seed=population_seed, options={"scenario": scenario})
    done = False
    for action in commands:
        observation, _, term, trunc, _ = env.step(action)
        done = term or trunc
        if done:
            break
    return observation, done


def summarize(episodes, output):
    summaries = []
    effects = []
    rng = np.random.default_rng(20261402)
    for subset in ("initial", "all_states", "ambiguous", "identified", "forecast_wrong_initial", "forecast_correct_initial"):
        def include(row):
            if subset == "initial": return row["step"] == 0
            if subset == "all_states": return True
            if subset == "ambiguous": return row["supported_world_count"] == 2
            if subset == "identified": return row["supported_world_count"] == 1
            if subset == "forecast_wrong_initial": return row["step"] == 0 and row["forecast_wrong"]
            return row["step"] == 0 and not row["forecast_wrong"]
        selected = [r for r in episodes if include(r)]
        if not selected:
            continue
        baseline = {(r["state_id"], r["future_seed"]): r for r in selected if r["arm"] == ARMS[0]}
        for arm in ARMS:
            rows = [r for r in selected if r["arm"] == arm]
            families = sorted({r["source_id"] for r in rows})
            summary = {"subset": subset, "arm": arm, "episodes": len(rows), "scenario_families": len(families)}
            for metric in ("final_incapacitation", "final_fed", "final_evacuation", "final_remaining", "steps", "mean_release", "wait_requests"):
                summary[metric] = float(np.mean([r[metric] for r in rows]))
            summaries.append(summary)
            family_effects = []
            for family in families:
                frows = [r for r in rows if r["source_id"] == family]
                family_effects.append(float(np.mean([r["final_incapacitation"] - baseline[(r["state_id"], r["future_seed"])]["final_incapacitation"] for r in frows])))
            values = np.asarray(family_effects)
            bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1) * 100
            effects.append({"subset": subset, "arm": arm, "scenario_families": len(families),
                            "scenario_equal_weight_delta_pp": float(values.mean() * 100),
                            "scenario_bootstrap_ci_low_pp": float(np.quantile(bootstrap, .025)),
                            "scenario_bootstrap_ci_high_pp": float(np.quantile(bootstrap, .975)),
                            "improved_families": int((values < 0).sum()), "worsened_families": int((values > 0).sum())})
    write_csv(summaries, output / "subset_summary.csv")
    write_csv(effects, output / "scenario_paired_effects.csv")
    return summaries, effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/terminal_shield_development_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/source_world_planning_development_v1")
    parser.add_argument("--planning-futures", type=int, default=3)
    parser.add_argument("--evaluation-futures", type=int, default=3)
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.planning_futures, args.evaluation_futures) < 2:
        raise ValueError("at least two distinct futures in each phase required")
    if args.output.exists() and not args.resume:
        raise FileExistsError("preserve existing output; use a new directory or exact --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.source / "development_manifest.json").read_text(encoding="utf-8"))
    scenarios = tuple(DynamicConflictScenario(**s) for s in source_manifest["scenario_records"])
    if not args.include_mirrors:
        scenarios = scenarios[:source_manifest["original_scenarios"]]
    by_id = {s.scenario_id: s for s in scenarios}
    with (args.source / "state_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = [r for r in csv.DictReader(handle) if r["scenario_id"] in by_id]
    source_files = ("src/fire_evac/source_world_planning.py", "scripts/run_source_world_planning.py", "src/fire_evac/terminal_shield.py",
                    "src/fire_evac/confirmatory.py", "src/fire_evac/dynamic_conflicts.py", "src/fire_evac/env.py", "src/fire_evac/behavior.py", "src/fire_evac/hazard.py")
    manifest = {"evaluation_role": "reused_development_reference_planning_not_independent_validation", "freeze_allowed": False,
                "planning_futures": args.planning_futures, "evaluation_futures": args.evaluation_futures, "include_mirrors": args.include_mirrors,
                "source_dataset_sha256": hashlib.sha256((args.source / "dataset.npz").read_bytes()).hexdigest(),
                "plans": [asdict(p) for p in PLANS], "arms": ARMS,
                "compatibility_tolerance": 1e-6, "compatible_world_model": "correct growth/onset/background/map; source only uncertain; noiseless sensors",
                "primary_subset": "initial: two original populations and three held-out future streams per scenario",
                "primary_metric": "final incapacitation; paired scenario-equal effects vs live_graded",
                "selection": "max or equal/point mean over supported source worlds; mean over planning futures; lexicographic incap,FED,remaining,congestion",
                "evaluation_seeds_disjoint_from_planning": True,
                "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
                "behavior_artifact_sha256": hashlib.sha256((ROOT / "outputs/behavior_pipeline/behavior_model.json").read_bytes()).hexdigest()}
    manifest = json.loads(json.dumps(manifest))
    path = args.output / "development_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("resume configuration or code differs")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    state_dir = args.output / "states"
    state_dir.mkdir(exist_ok=True)
    actual_env = make_env(scenarios)
    world_envs = [make_env(scenarios), make_env(scenarios)]
    episodes = []
    state_rows = []
    planning_rollouts = evaluation_rollouts = 0
    start = time.monotonic()
    for index, scenario in enumerate(scenarios):
        scenario_metadata = [r for r in metadata if r["scenario_id"] == scenario.scenario_id]
        population_seeds = sorted({int(r["population_seed"]) for r in scenario_metadata})
        for population_seed in population_seeds:
            relevant = {int(r["step"]): r for r in scenario_metadata if int(r["population_seed"]) == population_seed}
            actual_observation, _ = actual_env.reset(seed=population_seed, options={"scenario": scenario})
            commands = []
            step = 0
            done = False
            while not done and step <= max(relevant):
                if step in relevant:
                    row = relevant[step]
                    state_id = row["state_id"]
                    original = np.load(args.source / "states" / f"{state_id}.npz", allow_pickle=False)
                    np.testing.assert_array_equal(actual_observation, original["observation"])
                    destination = state_dir / f"{state_id}.npz"
                    future_seeds = tuple(4_000_000 + index * 10_000 + (population_seed % 1000) * 100 + step * 10 + f for f in range(args.planning_futures))
                    eval_seeds = tuple(8_000_000 + index * 10_000 + (population_seed % 1000) * 100 + step * 10 + f for f in range(args.evaluation_futures))
                    if set(future_seeds) & set(eval_seeds):
                        raise RuntimeError("planning and evaluation RNG streams overlap")
                    if not destination.exists():
                        hypotheses = source_world_scenarios(scenario)
                        supported, errors, mismatch = compatible_source_worlds(hypotheses, actual_observation, step)
                        if not supported[scenario.source_branch]:
                            raise RuntimeError("observed actual source excluded by the declared noiseless model")
                        risks = np.full((2, len(PLANS), args.planning_futures, len(OUTCOME_FIELDS)), np.nan)
                        source_dones = []
                        for source, hypothesis in enumerate(hypotheses):
                            world_observation, early_done = replay_world(world_envs[source], hypothesis, population_seed, commands)
                            source_dones.append(early_done)
                            if not supported[source]:
                                continue
                            if early_done:
                                raise RuntimeError("a supported reconstructed world ended before this state")
                            for plan_index, plan in enumerate(PLANS):
                                for f, seed in enumerate(future_seeds):
                                    risks[source, plan_index, f] = vector(evaluate_plan(world_envs[source], world_observation, plan, seed))
                        chosen = list(BASELINE_INDICES)
                        for mode in ("point", "mean", "robust", "oracle"):
                            chosen.append(select_reference_plan(risks, supported, scenario.forecast_source_branch, mode, true_source=scenario.source_branch if mode == "oracle" else None))
                        outcomes = np.empty((len(ARMS), args.evaluation_futures, len(OUTCOME_FIELDS)))
                        evaluated = {}
                        for arm_index, plan_index in enumerate(chosen):
                            for f, seed in enumerate(eval_seeds):
                                key = (plan_index, seed)
                                if key not in evaluated:
                                    evaluated[key] = vector(evaluate_plan(actual_env, actual_observation, PLANS[plan_index], seed))
                                outcomes[arm_index, f] = evaluated[key]
                        np.savez_compressed(destination, observation=actual_observation, supported=supported, world_errors=errors,
                                            model_mismatch=np.asarray(mismatch), risks=risks, chosen_indices=np.asarray(chosen),
                                            evaluation_outcomes=outcomes, planning_seeds=np.asarray(future_seeds), evaluation_seeds=np.asarray(eval_seeds),
                                            planning_rollouts=np.asarray(int(supported.sum()) * len(PLANS) * args.planning_futures),
                                            evaluation_rollouts=np.asarray(len(evaluated)), world_early_done=np.asarray(source_dones))
                    cached = np.load(destination, allow_pickle=False)
                    np.testing.assert_array_equal(cached["observation"], actual_observation)
                    planning_rollouts += int(cached["planning_rollouts"])
                    evaluation_rollouts += int(cached["evaluation_rollouts"])
                    supported = cached["supported"]
                    state_rows.append({"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": row["source_id"],
                                       "conflict_type": scenario.conflict_type, "population_seed": population_seed, "step": step,
                                       "supported_world_count": int(supported.sum()), "model_mismatch": bool(cached["model_mismatch"]),
                                       "forecast_wrong": scenario.forecast_source_branch != scenario.source_branch,
                                       **{arm: PLANS[int(cached['chosen_indices'][a])].name for a, arm in enumerate(ARMS)}})
                    for a, arm in enumerate(ARMS):
                        for f, seed in enumerate(cached["evaluation_seeds"]):
                            outcome = cached["evaluation_outcomes"][a, f]
                            episodes.append({"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": row["source_id"],
                                             "conflict_type": scenario.conflict_type, "population_seed": population_seed,
                                             "step": step, "supported_world_count": int(supported.sum()),
                                             "forecast_wrong": scenario.forecast_source_branch != scenario.source_branch,
                                             "arm": arm, "selected_plan": PLANS[int(cached['chosen_indices'][a])].name,
                                             "future_seed": int(seed), **dict(zip(OUTCOME_FIELDS, outcome))})
                if step == max(relevant):
                    break
                # Match the exact original collection policy for this repetition.
                policy = relevant[0]["trajectory_policy"]
                repetition = int(relevant[0]["state_id"].split("_")[1])
                action = reactive_followup_action(actual_observation) if policy == "reactive" else int(np.random.default_rng(3_000_000 + index * 10_000 + repetition * 1000 + step).integers(9))
                commands.append(action)
                actual_observation, _, term, trunc, _ = actual_env.step(action)
                done = term or trunc
                step += 1
        print(f"Evaluated {index + 1}/{len(scenarios)} {scenario.scenario_id}: {len(state_rows)} states, {time.monotonic() - start:.1f}s", flush=True)
        (args.output / "progress.json").write_text(json.dumps({"phase": "planning_and_evaluation", "completed_scenarios": index + 1, "states": len(state_rows)}), encoding="utf-8")
    actual_env.close()
    for env in world_envs:
        env.close()
    write_csv(episodes, args.output / "evaluation_episode_results.csv")
    write_csv(state_rows, args.output / "state_selection.csv")
    summaries, effects = summarize(episodes, args.output)
    selections = {arm: {plan.name: sum(r[arm] == plan.name for r in state_rows) for plan in PLANS} for arm in ARMS[5:]}
    summary = {"evaluation_role": manifest["evaluation_role"], "freeze_allowed": False, "scenarios": len(scenarios),
               "states": len(state_rows), "ambiguous_states": sum(r["supported_world_count"] == 2 for r in state_rows),
               "identified_states": sum(r["supported_world_count"] == 1 for r in state_rows),
               "model_mismatch_states": sum(r["model_mismatch"] for r in state_rows),
               "planning_rollouts": planning_rollouts, "unique_evaluation_rollouts": evaluation_rollouts,
               "evaluation_arm_records": len(episodes), "initial_summary": [r for r in summaries if r["subset"] == "initial"],
               "initial_scenario_effects": [r for r in effects if r["subset"] == "initial"], "plan_selection_counts": selections,
               "elapsed_seconds": time.monotonic() - start,
               "limitations": ["reused development scenarios, all intervals exploratory", "source uncertainty only; other forecast parameters assumed correct",
                               "exact smoke compatibility valid only in noiseless synthetic model", "privileged offline crowd replay and known simulator",
                               "three planning/evaluation streams are finite, not risk bounds", "plan selected once at each counterfactual start; plans react online but no receding plan search",
                               "time-division signs do not impose exact pedestrian split ratios", "no learned risk model, LLM, or PPO gain established"]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "progress.json").write_text(json.dumps({"phase": "complete", "states": len(state_rows), "freeze_allowed": False}), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "plan_selection_counts"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
