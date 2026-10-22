"""Predeclared development check of a 12-manual + 5-frozen-PPO teacher."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import gymnasium
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.counterfactual_shield import reactive_followup_action
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.source_world_planning import (
    PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds, select_reference_plan,
)
from fire_evac.ppo_reference_planning import (
    FROZEN_PPO_SEEDS, TRACE_FIELDS, FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256,
)
from run_source_world_planning import make_env, replay_world, vector, write_csv

MODES = ("point", "mean", "robust", "oracle")
EXTENDED_ARMS = ("extended_point", "extended_mean", "extended_minimax", "extended_oracle")
MANUAL_ARMS = ("manual_point", "manual_mean", "manual_minimax", "manual_oracle")
FIXED_ARMS = tuple(f"pure_ppo_{seed}" for seed in FROZEN_PPO_SEEDS)
ARMS = EXTENDED_ARMS + MANUAL_ARMS + FIXED_ARMS + ("live_graded", "uniform_candidate")
VIRTUAL_MEAN = "fixed_ppo_mean_expectation"
METRICS = OUTCOME_FIELDS[:9]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_schedule(state_index, planning_futures=3, evaluation_futures=3):
    if state_index < 0 or not 2 <= planning_futures < 100 or not 2 <= evaluation_futures < 100:
        raise ValueError("nonnegative state index and 2..99 futures required")
    offset = state_index * 100
    return (tuple(20_000_000 + offset + f for f in range(planning_futures)),
            tuple(40_000_000 + offset + f for f in range(evaluation_futures)))


def select_indices(risks, supported, forecast_source, true_source, state_index):
    # Only the explicitly privileged oracle sees true_source during selection.
    extended = [select_reference_plan(risks, supported, forecast_source, mode,
                true_source=true_source if mode == "oracle" else None) for mode in MODES]
    manual = [select_reference_plan(risks[:, :len(PLANS)], supported, forecast_source, mode,
              true_source=true_source if mode == "oracle" else None) for mode in MODES]
    random_index = int(np.random.default_rng(60_000_000 + state_index).integers(risks.shape[1]))
    return extended + manual + list(range(len(PLANS), risks.shape[1])) + [0, random_index]


def append_virtual_mean(episodes):
    """Expected outcomes of uniform episode-level selection among fixed actors."""
    groups = {}
    for row in episodes:
        if row["arm"] in FIXED_ARMS:
            groups.setdefault((row["state_id"], row["future_seed"]), []).append(row)
    virtual = []
    for rows in groups.values():
        if {r["arm"] for r in rows} != set(FIXED_ARMS) or len(rows) != 5:
            raise ValueError("five unique fixed model outcomes required")
        row = {k: v for k, v in rows[0].items() if k not in OUTCOME_FIELDS}
        row.update(arm=VIRTUAL_MEAN, selected_plan="expected_uniform_fixed_model_outcome")
        row.update({metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS})
        virtual.append(row)
    return episodes + virtual


def summarize(episodes, selections, output):
    rows_with_mean = append_virtual_mean(episodes)
    rng = np.random.default_rng(20261502)
    summaries, effects = [], []
    subsets = ("initial", "all_states", "ambiguous", "identified", "forecast_wrong_initial",
               "forecast_correct_initial") + tuple(f"conflict_initial:{c}" for c in sorted({r["conflict_type"] for r in episodes}))
    comparisons = []
    for i, arm in enumerate(EXTENDED_ARMS):
        comparisons.extend((arm, control) for control in (VIRTUAL_MEAN,) + FIXED_ARMS + (MANUAL_ARMS[i],))
    comparisons.extend((("extended_minimax", "extended_point"), ("extended_minimax", "extended_mean"),
                        ("extended_minimax", "extended_oracle"), ("extended_minimax", "live_graded"),
                        ("extended_minimax", "uniform_candidate")))
    for subset in subsets:
        def include(r):
            if subset == "initial": return r["step"] == 0
            if subset == "all_states": return True
            if subset == "ambiguous": return r["supported_world_count"] == 2
            if subset == "identified": return r["supported_world_count"] == 1
            if subset == "forecast_wrong_initial": return r["step"] == 0 and r["forecast_wrong"]
            if subset == "forecast_correct_initial": return r["step"] == 0 and not r["forecast_wrong"]
            return r["step"] == 0 and r["conflict_type"] == subset.split(":", 1)[1]
        selected = [r for r in rows_with_mean if include(r)]
        if not selected:
            continue
        by_arm = {arm: [r for r in selected if r["arm"] == arm] for arm in ARMS + (VIRTUAL_MEAN,)}
        for arm, rows in by_arm.items():
            families = sorted({r["source_id"] for r in rows})
            summary = {"subset": subset, "arm": arm, "records": len(rows), "scenario_families": len(families),
                       "virtual_expectation": arm == VIRTUAL_MEAN}
            for metric in METRICS:
                summary[metric] = float(np.mean([r[metric] for r in rows]))
                summary[f"family_equal_{metric}"] = float(np.mean([
                    np.mean([r[metric] for r in rows if r["source_id"] == family]) for family in families]))
            summaries.append(summary)
        for arm, control in comparisons:
            lookup = {(r["state_id"], r["future_seed"]): r for r in by_arm[control]}
            families = sorted({r["source_id"] for r in by_arm[arm]})
            values = np.asarray([np.mean([
                r["final_incapacitation"] - lookup[(r["state_id"], r["future_seed"])]["final_incapacitation"]
                for r in by_arm[arm] if r["source_id"] == family]) for family in families])
            draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1) * 100
            effects.append({"subset": subset, "arm": arm, "control": control, "scenario_families": len(families),
                            "scenario_equal_delta_pp": float(values.mean() * 100),
                            "exploratory_ci_low_pp": float(np.quantile(draws, .025)),
                            "exploratory_ci_high_pp": float(np.quantile(draws, .975)),
                            "improved_families": int((values < -1e-12).sum()),
                            "worsened_families": int((values > 1e-12).sum()),
                            "tied_families": int((np.abs(values) <= 1e-12).sum())})
    write_csv(summaries, output / "subset_summary.csv")
    write_csv(effects, output / "scenario_paired_effects.csv")
    selection_counts = {}
    for subset in ("initial", "all_states", "ambiguous", "identified"):
        selected = [r for r in selections if subset == "all_states" or
                    (subset == "initial" and r["step"] == 0) or
                    (subset == "ambiguous" and r["supported_world_count"] == 2) or
                    (subset == "identified" and r["supported_world_count"] == 1)]
        selection_counts[subset] = {arm: {name: sum(r[arm] == name for r in selected)
                                        for name in sorted({r[arm] for r in selected})}
                                    for arm in EXTENDED_ARMS + MANUAL_ARMS + ("uniform_candidate",)}
    return summaries, effects, selection_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/terminal_shield_development_v1")
    parser.add_argument("--models", type=Path, default=ROOT / "outputs/guarded_confirmatory_v2/models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ppo_extended_world_planning_development_v1")
    parser.add_argument("--planning-futures", type=int, default=3)
    parser.add_argument("--evaluation-futures", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_schedule(0, args.planning_futures, args.evaluation_futures)
    if args.output.exists() and not args.resume:
        raise FileExistsError("preserve existing outputs; choose a new directory or exact --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source_manifest = json.loads((args.source / "development_manifest.json").read_text(encoding="utf-8"))
    scenarios = tuple(DynamicConflictScenario(**s) for s in source_manifest["scenario_records"][:source_manifest["original_scenarios"]])
    by_id = {s.scenario_id: s for s in scenarios}
    with (args.source / "state_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = [r for r in csv.DictReader(handle) if r["scenario_id"] in by_id]
    state_indices = {r["state_id"]: i for i, r in enumerate(metadata)}
    if len(state_indices) != len(metadata):
        raise ValueError("state IDs must be unique")
    schedule = {r["state_id"]: seed_schedule(i, args.planning_futures, args.evaluation_futures)
                for i, r in enumerate(metadata)}
    planning_seeds = {s for pair in schedule.values() for s in pair[0]}
    evaluation_seeds = {s for pair in schedule.values() for s in pair[1]}
    stream_sets = (planning_seeds, evaluation_seeds,
                   {s + 9_000_001 for s in planning_seeds}, {s + 9_000_001 for s in evaluation_seeds})
    for i, seeds in enumerate(stream_sets):
        for other in stream_sets[i + 1:]:
            if seeds & other:
                raise ValueError("domain and sensor seed namespaces overlap")
    controllers = []
    model_records = []
    for seed in FROZEN_PPO_SEEDS:
        path = args.models / f"seed_{seed}_pure_ppo.zip"
        model = PPO.load(path, device="cpu")
        controllers.append(FrozenPPOContinuation(seed, model))
        model_records.append({"name": controllers[-1].name, "training_seed": seed, "path": str(path.resolve()),
                              "checkpoint_sha256": sha256(path), "parameter_sha256": policy_parameter_sha256(model),
                              "original_training_steps": int(model.num_timesteps), "inputs": 34, "actions": 9})
    candidates = PLANS + tuple(controllers)
    source_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/fire_evac").glob("*.py"))
    source_files += ["scripts/run_source_world_planning.py", "scripts/run_ppo_extended_world_planning.py",
                     "tests/test_ppo_reference_planning.py", "reports/PPO扩展参考教师开发方案.md", "pyproject.toml"]
    manifest = {"evaluation_role": "reused_development_frozen_ppo_reference_library_not_fair_new_training",
                "freeze_allowed": False, "source_directory": str(args.source.resolve()),
                "scenarios": len(scenarios), "states": len(metadata), "scenario_records": [asdict(s) for s in scenarios],
                "manual_plans": [asdict(p) for p in PLANS], "frozen_ppo_models": model_records,
                "candidate_names": [p.name for p in candidates], "arms": ARMS,
                "virtual_comparator": "arithmetic expectation of uniform episode-level fixed-PPO selection, not action ensemble",
                "planning_futures": args.planning_futures, "evaluation_futures": args.evaluation_futures,
                "seed_registry": {key: {"planning": value[0], "evaluation": value[1],
                                  "uniform_selector": 60_000_000 + state_indices[key]} for key, value in schedule.items()},
                "all_domain_sensor_phase_streams_disjoint": True,
                "compatibility_tolerance": 1e-6, "world_model": "noiseless; only source uncertain; exact historical crowd replay",
                "primary_subset": "initial 36 families x 2 populations x 3 new evaluation futures per arm",
                "primary_metric": "final incapacitation; family-equal paired effects",
                "predeclared_focus_arm": "extended_minimax", "previously_known_strong_fixed_comparator": "pure_ppo_601",
                "selection": "unchanged lexicographic finite-library mean/worst supported world, three future means",
                "bootstrap": {"unit": "scenario_family", "draws": 2000, "seed": 20261502, "role": "exploratory_no_multiplicity_correction"},
                "trace_fields": TRACE_FIELDS, "outcome_fields": OUTCOME_FIELDS,
                "source_dataset_sha256": sha256(args.source / "dataset.npz"),
                "source_manifest_sha256": sha256(args.source / "development_manifest.json"),
                "source_metadata_sha256": sha256(args.source / "state_metadata.csv"),
                "source_state_sha256": {r["state_id"]: sha256(args.source / "states" / f"{r['state_id']}.npz") for r in metadata},
                "source_hashes": {p: sha256(ROOT / p) for p in source_files},
                "behavior_artifact_sha256": sha256(ROOT / "outputs/behavior_pipeline/behavior_model.json"),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                            "stable_baselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__, "torch_threads": 1}}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "development_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("resume configuration, inputs, models, or source hashes differ")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in source_files:
        target = args.output / "source_snapshot" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != manifest["source_hashes"][p]:
            raise ValueError("source snapshot differs")
        shutil.copy2(ROOT / p, target)
    for record in model_records:
        target = args.output / "model_snapshot" / Path(record["path"]).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(record["path"], target)
    state_dir = args.output / "states"
    state_dir.mkdir(exist_ok=True)
    actual_env = make_env(scenarios)
    world_envs = [make_env(scenarios), make_env(scenarios)]
    episodes, selections, optimism = [], [], []
    planning_rollouts = evaluation_rollouts = trace_steps = neutral_steps = blocked_steps = 0
    start = time.monotonic()
    for index, scenario in enumerate(scenarios):
        scenario_metadata = [r for r in metadata if r["scenario_id"] == scenario.scenario_id]
        for population_seed in sorted({int(r["population_seed"]) for r in scenario_metadata}):
            relevant = {int(r["step"]): r for r in scenario_metadata if int(r["population_seed"]) == population_seed}
            actual_observation, _ = actual_env.reset(seed=population_seed, options={"scenario": scenario})
            commands, step, done = [], 0, False
            while not done and step <= max(relevant):
                if step in relevant:
                    row = relevant[step]
                    state_id = row["state_id"]
                    with np.load(args.source / "states" / f"{state_id}.npz", allow_pickle=False) as original:
                        np.testing.assert_array_equal(actual_observation, original["observation"])
                    destination = state_dir / f"{state_id}.npz"
                    future_seeds, eval_seeds = schedule[state_id]
                    if not destination.exists():
                        hypotheses = source_world_scenarios(scenario)
                        supported, errors, mismatch = compatible_source_worlds(hypotheses, actual_observation, step)
                        if not supported[scenario.source_branch]:
                            raise RuntimeError("noiseless model excludes actual source")
                        risks = np.full((2, len(candidates), args.planning_futures, len(OUTCOME_FIELDS)), np.nan)
                        source_dones = []
                        for source, hypothesis in enumerate(hypotheses):
                            world_observation, early_done = replay_world(world_envs[source], hypothesis, population_seed, commands)
                            source_dones.append(early_done)
                            if not supported[source]:
                                continue
                            if early_done:
                                raise RuntimeError("supported reconstructed world already ended")
                            for plan_index, candidate in enumerate(candidates):
                                for f, seed in enumerate(future_seeds):
                                    risks[source, plan_index, f] = vector(evaluate_continuation(
                                        world_envs[source], world_observation, candidate, seed))
                        chosen = select_indices(risks, supported, scenario.forecast_source_branch,
                                                scenario.source_branch, state_indices[state_id])
                        outcomes = np.empty((len(ARMS), args.evaluation_futures, len(OUTCOME_FIELDS)))
                        evaluated, traces = {}, {}
                        for a, plan_index in enumerate(chosen):
                            for f, seed in enumerate(eval_seeds):
                                key = (plan_index, f)
                                if key not in evaluated:
                                    trace = []
                                    evaluated[key] = vector(evaluate_continuation(
                                        actual_env, actual_observation, candidates[plan_index], seed, trace))
                                    traces[key] = np.asarray(trace, dtype=np.float64)
                                outcomes[a, f] = evaluated[key]
                        keys = sorted(traces)
                        offsets = np.cumsum([0] + [len(traces[key]) for key in keys])
                        temporary = destination.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, observation=actual_observation, supported=supported, world_errors=errors,
                                            model_mismatch=np.asarray(mismatch), risks=risks, chosen_indices=np.asarray(chosen),
                                            evaluation_outcomes=outcomes, planning_seeds=np.asarray(future_seeds),
                                            evaluation_seeds=np.asarray(eval_seeds), world_early_done=np.asarray(source_dones),
                                            planning_rollouts=np.asarray(int(supported.sum()) * len(candidates) * args.planning_futures),
                                            evaluation_rollouts=np.asarray(len(evaluated)), trace_keys=np.asarray(keys),
                                            trace_offsets=offsets, execution_trace=np.concatenate([traces[k] for k in keys]))
                        temporary.replace(destination)
                    with np.load(destination, allow_pickle=False) as cached:
                        np.testing.assert_array_equal(cached["observation"], actual_observation)
                        np.testing.assert_array_equal(cached["planning_seeds"], future_seeds)
                        np.testing.assert_array_equal(cached["evaluation_seeds"], eval_seeds)
                        planning_rollouts += int(cached["planning_rollouts"])
                        evaluation_rollouts += int(cached["evaluation_rollouts"])
                        trace_steps += len(cached["execution_trace"])
                        neutral_steps += int((cached["execution_trace"][:, 2] == -1).sum())
                        blocked_steps += int(cached["execution_trace"][:, 5].sum())
                        supported = cached["supported"]
                        common = {"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": row["source_id"],
                                  "conflict_type": scenario.conflict_type, "population_seed": population_seed, "step": step,
                                  "supported_world_count": int(supported.sum()),
                                  "forecast_wrong": scenario.forecast_source_branch != scenario.source_branch}
                        selections.append({**common, "model_mismatch": bool(cached["model_mismatch"]),
                                           **{arm: candidates[int(cached["chosen_indices"][a])].name for a, arm in enumerate(ARMS)}})
                        for a, arm in enumerate(ARMS):
                            plan_index = int(cached["chosen_indices"][a])
                            for f, seed in enumerate(eval_seeds):
                                episodes.append({**common, "arm": arm, "selected_plan": candidates[plan_index].name,
                                                 "future_seed": int(seed),
                                                 **dict(zip(OUTCOME_FIELDS, cached["evaluation_outcomes"][a, f]))})
                            if arm in EXTENDED_ARMS + MANUAL_ARMS:
                                planning_actual = float(cached["risks"][scenario.source_branch, plan_index, :, 0].mean())
                                evaluation_actual = float(cached["evaluation_outcomes"][a, :, 0].mean())
                                optimism.append({**common, "arm": arm, "selected_plan": candidates[plan_index].name,
                                                 "planning_actual_world_mean_incapacitation": planning_actual,
                                                 "evaluation_mean_incapacitation": evaluation_actual,
                                                 "evaluation_minus_planning_pp": (evaluation_actual - planning_actual) * 100})
                if step == max(relevant):
                    break
                policy = relevant[0]["trajectory_policy"]
                repetition = int(relevant[0]["state_id"].split("_")[1])
                action = reactive_followup_action(actual_observation) if policy == "reactive" else int(np.random.default_rng(
                    3_000_000 + index * 10_000 + repetition * 1000 + step).integers(9))
                commands.append(action)
                actual_observation, _, term, trunc, _ = actual_env.step(action)
                done, step = term or trunc, step + 1
        print(f"Extended library {index + 1}/{len(scenarios)} {scenario.scenario_id}: "
              f"{len(selections)} states, {planning_rollouts} planning, {evaluation_rollouts} unique evaluation, "
              f"{time.monotonic() - start:.1f}s", flush=True)
        (args.output / "progress.json").write_text(json.dumps({"phase": "planning_and_evaluation",
            "completed_scenarios": index + 1, "scenarios": len(scenarios), "states": len(selections),
            "planning_rollouts": planning_rollouts, "unique_evaluation_rollouts": evaluation_rollouts}), encoding="utf-8")
    actual_env.close()
    for env in world_envs:
        env.close()
    if len(selections) != len(metadata):
        raise RuntimeError("some development snapshots were not replayed")
    for controller, record in zip(controllers, model_records):
        if policy_parameter_sha256(controller.model) != record["parameter_sha256"]:
            raise RuntimeError("frozen PPO parameters changed")
        if controller.model.num_timesteps != record["original_training_steps"] or sha256(Path(record["path"])) != record["checkpoint_sha256"]:
            raise RuntimeError("frozen PPO training count or checkpoint changed")
    write_csv(episodes, args.output / "evaluation_episode_results.csv")
    write_csv(selections, args.output / "state_selection.csv")
    write_csv(optimism, args.output / "selection_generalization.csv")
    summaries, effects, counts = summarize(episodes, selections, args.output)
    initial_optimism = []
    for arm in EXTENDED_ARMS + MANUAL_ARMS:
        rows = [r for r in optimism if r["arm"] == arm and r["step"] == 0]
        initial_optimism.append({"arm": arm, "states": len(rows),
                                "planning_actual_world_mean_incapacitation": float(np.mean([r["planning_actual_world_mean_incapacitation"] for r in rows])),
                                "evaluation_mean_incapacitation": float(np.mean([r["evaluation_mean_incapacitation"] for r in rows])),
                                "evaluation_minus_planning_pp": float(np.mean([r["evaluation_minus_planning_pp"] for r in rows]))})
    summary = {"evaluation_role": manifest["evaluation_role"], "freeze_allowed": False,
               "scenarios": len(scenarios), "states": len(selections), "candidate_count": len(candidates),
               "ambiguous_states": sum(r["supported_world_count"] == 2 for r in selections),
               "identified_states": sum(r["supported_world_count"] == 1 for r in selections),
               "model_mismatch_states": sum(r["model_mismatch"] for r in selections),
               "planning_rollouts": planning_rollouts, "unique_evaluation_rollouts": evaluation_rollouts,
               "total_unique_full_rollouts": planning_rollouts + evaluation_rollouts,
               "evaluation_arm_records": len(episodes), "virtual_mean_records_are_not_simulations": True,
               "actual_execution_trace_steps": trace_steps, "neutral_release_trace_steps": neutral_steps,
               "sign_hold_blocked_trace_steps": blocked_steps, "all_frozen_models_unchanged": True,
               "initial_summary": [r for r in summaries if r["subset"] == "initial"],
               "initial_scenario_effects": [r for r in effects if r["subset"] == "initial"],
               "initial_selection_generalization": initial_optimism, "selection_counts": counts,
               "this_invocation_elapsed_seconds": time.monotonic() - start,
               "limitations": ["reused development scenarios; new RNG futures do not make an independent scenario test",
                               "all intervals exploratory with no multiplicity correction", "privileged exact crowd replay and known simulator",
                               "source alone uncertain; exact smoke compatibility not validated with noise",
                               "five previously trained actors plus offline search cost exceed single actor resources; not a fair new training comparison",
                               "same future starting RNG, but action-dependent random draw consumption",
                               "finite candidate library and sampled futures are not global optimum or safety bounds",
                               "one plan selection per counterfactual start; no receding plan search",
                               "no risk distillation, new LLM calls, new PPO training, or hard-shield qualification in this run"]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "progress.json").write_text(json.dumps({"phase": "complete", "states": len(selections),
                                                        "freeze_allowed": False}), encoding="utf-8")
    print(json.dumps({"phase": "complete", "states": len(selections), "total_unique_full_rollouts": planning_rollouts + evaluation_rollouts,
                      "elapsed_seconds": time.monotonic() - start,
                      "initial_incapacitation": {r["arm"]: r["final_incapacitation"] for r in summary["initial_summary"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
