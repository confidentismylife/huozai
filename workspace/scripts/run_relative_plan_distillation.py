"""Predeclared 2x2 observable plan distillation development experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from stable_baselines3 import PPO
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.counterfactual_shield import reactive_followup_action
from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank
from fire_evac.source_world_planning import (
    PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds, select_reference_plan,
)
from fire_evac.ppo_reference_planning import (
    FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, TRACE_FIELDS,
)
from fire_evac.plan_distillation import (
    STUDENT_SEEDS, PRIORITY_FIELDS, ObservablePlanStudent, relative_priority_targets,
    stratified_family_split, choose_priority, family_weights,
)
from fire_evac.relative_plan_ranking import ObservableScoreStudent, fit_score_student, baseline_relative_targets
from run_source_world_planning import make_env, replay_world, vector, write_csv
from run_observable_plan_distillation import sha256, json_write, read_csv, scenario_signature

GROUPS = ("small_mse", "small_rank", "large_mse", "large_rank")
FRESH_ARMS = tuple(f"{group}_{seed}" for group in GROUPS for seed in STUDENT_SEEDS)
HISTORICAL_ARMS = tuple(f"historical_mse_{seed}" for seed in STUDENT_SEEDS)
STUDENT_ARMS = FRESH_ARMS + HISTORICAL_ARMS
FIXED_ARMS = tuple(f"pure_ppo_{seed}" for seed in range(601, 606))
ARMS = STUDENT_ARMS + FIXED_ARMS + (
    "teacher_minimax", "teacher_point", "teacher_oracle", "small_global_plan", "large_global_plan",
    "uniform_ppo", "uniform_candidate", "live_graded",
)
COLLECTION_STEPS = (0, 8, 16, 24, 40)


def training_seeds(scenario_index, repetition, sample_index):
    index = 10 * scenario_index + 5 * repetition + sample_index
    return index, tuple(170_000_000 + 100 * index + f for f in range(3))


def audit_seeds(scenario_index, repetition, sample_index):
    index = 10 * scenario_index + 5 * repetition + sample_index
    return index, tuple(210_000_000 + 100 * index + f for f in range(3)), tuple(
        230_000_000 + 100 * index + f for f in range(3))


def initial_global_plan(targets, families, train_mask, steps):
    mask = np.asarray(train_mask, bool) & (np.asarray(steps) == 0)
    if not mask.any():
        raise ValueError("training initial states are required")
    priorities = (targets[mask] * family_weights(families[mask])[:, None, None]).sum(axis=0)
    return choose_priority(priorities)


def load_student(output, record):
    cls = ObservablePlanStudent if record["model_class"] == "ObservablePlanStudent" else ObservableScoreStudent
    return cls.load(output / record["checkpoint"])


def candidate_risks(world_envs, hypotheses, population_seed, commands, observation, step, candidates, future_seeds):
    supported, errors, mismatch = compatible_source_worlds(hypotheses, observation, step)
    risks = np.full((2, 17, 3, len(OUTCOME_FIELDS)), np.nan)
    for source, hypothesis in enumerate(hypotheses):
        if not supported[source]:
            continue
        world_observation, early_done = replay_world(world_envs[source], hypothesis, population_seed, commands)
        if early_done:
            raise RuntimeError("supported world ended before continuation start")
        for p, candidate in enumerate(candidates):
            for f, seed in enumerate(future_seeds):
                risks[source, p, f] = vector(evaluate_continuation(world_envs[source], world_observation, candidate, seed))
    return supported, errors, mismatch, risks


def collect(args, scenarios, candidates, phase, students=None, frozen=None):
    """Training collects labels only; auditing starts after checkpoint freeze."""
    audit = phase == "audit"
    directory = args.output / ("audit_states" if audit else "training_states")
    directory.mkdir(exist_ok=True)
    actual_env = make_env(scenarios)
    world_envs = [make_env(scenarios), make_env(scenarios)]
    selections, episodes, diagnostics = [], [], []
    planning_count = evaluation_count = trace_steps = neutral_steps = blocked_steps = 0
    started = time.monotonic()
    try:
        for j, scenario in enumerate(scenarios):
            hypotheses = source_world_scenarios(scenario)
            for rep in range(2):
                population_seed = (190_000_000 if audit else 150_000_000) + 2 * j + rep
                observation, _ = actual_env.reset(seed=population_seed, options={"scenario": scenario})
                commands, step, done = [], 0, False
                while not done and step <= max(COLLECTION_STEPS):
                    if step in COLLECTION_STEPS:
                        sample = COLLECTION_STEPS.index(step)
                        if audit:
                            index, futures, evaluation_seeds = audit_seeds(j, rep, sample)
                        else:
                            index, futures = training_seeds(j, rep, sample)
                        state_id = f"{scenario.scenario_id}_{rep:02d}_{step:03d}"
                        path = directory / f"{state_id}.npz"
                        if not path.exists():
                            # Observable students select before any privileged planning query.
                            scores = np.asarray([s.priorities(observation) for s in students]) if audit else None
                            supported, errors, mismatch, risks = candidate_risks(
                                world_envs, hypotheses, population_seed, commands, observation, step, candidates, futures)
                            if not supported[scenario.source_branch]:
                                raise RuntimeError("noiseless compatibility excluded true source")
                            cache = {"observation": observation, "supported": supported, "world_errors": errors,
                                     "model_mismatch": np.asarray(mismatch), "risks": risks,
                                     "planning_seeds": np.asarray(futures), "population_seed": np.asarray(population_seed),
                                     "command_history": np.asarray(commands, dtype=np.int64),
                                     "planning_rollouts": np.asarray(int(supported.sum()) * 17 * 3)}
                            if audit:
                                chosen = [choose_priority(s) for s in scores] + list(range(12, 17))
                                chosen += [select_reference_plan(risks, supported, scenario.forecast_source_branch, mode,
                                    true_source=scenario.source_branch if mode == "oracle" else None)
                                    for mode in ("robust", "point", "oracle")]
                                chosen += [frozen["small_global_plan_index"], frozen["large_global_plan_index"],
                                    int(np.random.default_rng(251_000_000 + index).integers(12, 17)),
                                    int(np.random.default_rng(250_000_000 + index).integers(17)), 0]
                                assert len(chosen) == len(ARMS) == 38
                                outcomes = np.empty((len(ARMS), 3, len(OUTCOME_FIELDS)))
                                evaluated, traces = {}, {}
                                for a, p in enumerate(chosen):
                                    for f, seed in enumerate(evaluation_seeds):
                                        if (p, f) not in evaluated:
                                            trace = []
                                            evaluated[(p, f)] = vector(evaluate_continuation(actual_env, observation, candidates[p], seed, trace))
                                            traces[(p, f)] = np.asarray(trace, np.float64)
                                        outcomes[a, f] = evaluated[(p, f)]
                                keys = sorted(evaluated)
                                cache.update(student_priorities=scores, chosen_indices=np.asarray(chosen),
                                    evaluation_outcomes=outcomes, evaluation_seeds=np.asarray(evaluation_seeds),
                                    evaluation_rollouts=np.asarray(len(evaluated)), trace_keys=np.asarray(keys),
                                    trace_offsets=np.cumsum([0] + [len(traces[k]) for k in keys]),
                                    execution_trace=np.concatenate([traces[k] for k in keys]))
                            temporary = path.with_suffix(".tmp.npz")
                            np.savez_compressed(temporary, **cache)
                            temporary.replace(path)
                        with np.load(path, allow_pickle=False) as state:
                            np.testing.assert_array_equal(state["observation"], observation)
                            np.testing.assert_array_equal(state["planning_seeds"], futures)
                            np.testing.assert_array_equal(state["command_history"], commands)
                            assert int(state["population_seed"]) == population_seed
                            assert state["supported"][scenario.source_branch]
                            planning_count += int(state["planning_rollouts"])
                            common = {"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": scenario.scenario_id,
                                "conflict_type": scenario.conflict_type, "population_seed": population_seed, "repetition": rep,
                                "trajectory_policy": "random" if rep == 0 else "reactive", "step": step,
                                "supported_world_count": int(state["supported"].sum()),
                                "forecast_wrong": scenario.forecast_source_branch != scenario.source_branch,
                                "model_mismatch": bool(state["model_mismatch"])}
                            row = dict(common)
                            if audit:
                                np.testing.assert_array_equal(state["evaluation_seeds"], evaluation_seeds)
                                evaluation_count += int(state["evaluation_rollouts"])
                                trace_steps += len(state["execution_trace"])
                                neutral_steps += int((state["execution_trace"][:, 2] == -1).sum())
                                blocked_steps += int(state["execution_trace"][:, 5].sum())
                                targets = relative_priority_targets(state["risks"], state["supported"])
                                teacher_choice = choose_priority(targets)
                                for a, student in enumerate(students):
                                    np.testing.assert_array_equal(state["student_priorities"][a], student.priorities(observation))
                                    assert int(state["chosen_indices"][a]) == student.choose(observation)
                                    prediction_targets = baseline_relative_targets(targets) if getattr(student, "objective", "mse") == "relative_rank" else targets
                                    p = int(state["chosen_indices"][a])
                                    diagnostics.append({**common, "arm": STUDENT_ARMS[a], "selected_plan": candidates[p].name,
                                        "exact_teacher_agreement": p == teacher_choice,
                                        "teacher_incap_regret_pp": float(targets[p, 0] * 100),
                                        "standardized_score_mse": float(np.mean(((state["student_priorities"][a] - prediction_targets) / student.target_scale) ** 2))})
                                for a, arm in enumerate(ARMS):
                                    p = int(state["chosen_indices"][a])
                                    row[arm] = candidates[p].name
                                    for f, seed in enumerate(evaluation_seeds):
                                        episodes.append({**common, "arm": arm, "selected_plan": candidates[p].name, "future_seed": int(seed),
                                            **dict(zip(OUTCOME_FIELDS, state["evaluation_outcomes"][a, f]))})
                            selections.append(row)
                    if step == max(COLLECTION_STEPS):
                        break
                    random_base = 200_000_000 if audit else 160_000_000
                    action = reactive_followup_action(observation) if rep == 1 else int(np.random.default_rng(
                        random_base + 10_000 * j + 1_000 * rep + step).integers(9))
                    commands.append(action)
                    observation, _, term, trunc, _ = actual_env.step(action)
                    done, step = term or trunc, step + 1
            print(f"{phase} {j + 1}/{len(scenarios)} {scenario.scenario_id}: {len(selections)} states, "
                  f"{planning_count} planning, {evaluation_count} unique evaluation, {time.monotonic()-started:.1f}s", flush=True)
            json_write(args.output / "progress.json", {"phase": phase, "completed_scenarios": j + 1, "total_scenarios": len(scenarios),
                "states": len(selections), "planning_rollouts": planning_count, "unique_evaluation_rollouts": evaluation_count})
    finally:
        actual_env.close()
        for env in world_envs:
            env.close()
    write_csv(selections, args.output / ("state_selection.csv" if audit else "training_state_selection.csv"))
    if audit:
        write_csv(episodes, args.output / "evaluation_episode_results.csv")
        write_csv(diagnostics, args.output / "new_family_distillation_diagnostics.csv")
    return {"families": len(scenarios), "states": len(selections), "initial_states": sum(r["step"] == 0 for r in selections),
        "ambiguous_states": sum(r["supported_world_count"] == 2 for r in selections),
        "model_mismatch_states": sum(r["model_mismatch"] for r in selections), "planning_rollouts": planning_count,
        "unique_evaluation_rollouts": evaluation_count, "execution_arm_records": len(episodes),
        "actual_execution_trace_steps": trace_steps, "neutral_release_trace_steps": neutral_steps, "sign_hold_blocked_trace_steps": blocked_steps}


def training_dataset(args, split):
    rows = [(row, args.source / "states", False) for row in read_csv(args.source / "state_selection.csv")]
    rows += [(row, args.output / "training_states", True) for row in read_csv(args.output / "training_state_selection.csv")]
    x, targets, families, steps, state_ids, new_rows = [], [], [], [], [], []
    for row, directory, is_new in rows:
        with np.load(directory / f"{row['state_id']}.npz", allow_pickle=False) as state:
            x.append(state["observation"])
            targets.append(relative_priority_targets(state["risks"], state["supported"]))
        families.append(row["source_id"])
        steps.append(int(row["step"]))
        state_ids.append(row["state_id"])
        new_rows.append(is_new)
    families, steps, new_rows = np.asarray(families), np.asarray(steps), np.asarray(new_rows, bool)
    small_train = np.asarray([split.get(f) == "train" for f in families])
    calibration = np.asarray([split.get(f) == "calibration" for f in families])
    data = {"observations": np.asarray(x), "targets": np.asarray(targets), "families": families,
        "steps": steps, "state_ids": np.asarray(state_ids), "new_training_mask": new_rows,
        "small_train_mask": small_train, "large_train_mask": small_train | new_rows,
        "calibration_mask": calibration, "selection_mask": calibration & (steps == 0)}
    path = args.output / "training_dataset.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for key, value in data.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(path, **data)
    return data


def train_and_freeze(args, manifest, historical, dataset):
    path = args.output / "frozen_students.json"
    small_global = initial_global_plan(dataset["targets"], dataset["families"], dataset["small_train_mask"], dataset["steps"])
    large_global = initial_global_plan(dataset["targets"], dataset["families"], dataset["large_train_mask"], dataset["steps"])
    provenance = {"protocol_sha256": sha256(args.output / "development_manifest.json"),
        "dataset_sha256": sha256(args.output / "training_dataset.npz"),
        "new_training_state_sha256": {p.stem: sha256(p) for p in sorted((args.output / "training_states").glob("*.npz"))},
        "small_global_plan_index": small_global, "large_global_plan_index": large_global,
        "small_global_plan_name": manifest["candidate_names"][small_global],
        "large_global_plan_name": manifest["candidate_names"][large_global]}
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if [r["arm"] for r in frozen["models"]] != list(STUDENT_ARMS):
            raise ValueError("frozen arm set/order differs")
        for key, value in provenance.items():
            if frozen[key] != value:
                raise ValueError(f"frozen training provenance differs: {key}")
        for record in frozen["models"]:
            assert sha256(args.output / record["checkpoint"]) == record["sha256"]
        return [load_student(args.output, record) for record in frozen["models"]], frozen
    if (args.output / "audit_started_after_freeze.json").exists() or any((args.output / "audit_states").glob("*.npz")):
        raise RuntimeError("audit already exists without complete frozen models; refuse retraining")
    directory = args.output / "models"
    directory.mkdir(exist_ok=True)
    students, records, curves, diagnostics = [], [], [], []
    for group in GROUPS:
        size, method = group.split("_")
        train = dataset[f"{size}_train_mask"]
        included = train | dataset["calibration_mask"]
        x, y, families = (dataset[key][included] for key in ("observations", "targets", "families"))
        train, calibration, selection = (dataset[key][included] for key in (f"{size}_train_mask", "calibration_mask", "selection_mask"))
        objective = "mse" if method == "mse" else "relative_rank"
        for seed in STUDENT_SEEDS:
            arm = f"{group}_{seed}"
            student, curve, info = fit_score_student(x, y, families, train, calibration, seed,
                objective=objective, selection_mask=selection)
            checkpoint = directory / f"{arm}.pt"
            student.save(checkpoint)
            students.append(student)
            records.append({"arm": arm, "group": group, "seed": seed, "model_class": "ObservableScoreStudent",
                "historical": False, "checkpoint": str(checkpoint.relative_to(args.output)).replace("\\", "/"),
                "sha256": sha256(checkpoint), **info})
            curves.extend({"arm": arm, **point} for point in curve)
            prediction = np.asarray([student.priorities(o) for o in x])
            chosen = np.asarray([choose_priority(p) for p in prediction])
            yt = baseline_relative_targets(y) if objective == "relative_rank" else y
            for name, mask in (("train", train), ("calibration", calibration), ("calibration_initial", selection)):
                weights = family_weights(families[mask])
                error = (((prediction[mask] - yt[mask]) / student.target_scale) ** 2).mean(axis=(1, 2))
                regret = y[np.arange(len(y)), chosen, 0]
                exact = chosen == np.asarray([choose_priority(t) for t in y])
                diagnostics.append({"arm": arm, "subset": name, "states": int(mask.sum()), "families": len(set(families[mask])),
                    "family_equal_standardized_mse": float(np.dot(error, weights)),
                    "family_equal_teacher_incap_regret_pp": float(np.dot(regret[mask], weights) * 100),
                    "family_equal_exact_teacher_agreement": float(np.dot(exact[mask], weights))})
            print(f"Frozen {arm}: epoch {info['selected_epoch']}/{info['actual_epochs']}, "
                  f"initial calibration regret {info['best_calibration_primary_regret']*100:.4f}pp", flush=True)
    for seed in STUDENT_SEEDS:
        old = next(r for r in historical["models"] if r["arm"] == f"full_student_{seed}")
        original = args.historical / old["checkpoint"]
        assert sha256(original) == old["sha256"]
        arm = f"historical_mse_{seed}"
        checkpoint = directory / f"{arm}.pt"
        shutil.copy2(original, checkpoint)
        assert sha256(checkpoint) == old["sha256"]
        students.append(ObservablePlanStudent.load(checkpoint))
        records.append({**old, "arm": arm, "group": "historical_mse", "model_class": "ObservablePlanStudent",
            "historical": True, "objective": "mse", "historical_original_arm": old["arm"],
            "original_checkpoint": str(original.resolve()),
            "checkpoint": str(checkpoint.relative_to(args.output)).replace("\\", "/")})
    # Loss term columns differ across objectives. Write a stable union schema.
    columns = list(dict.fromkeys(k for row in curves for k in row))
    write_csv([{key: row.get(key) for key in columns} for row in curves], args.output / "training_curves.csv")
    write_csv(diagnostics, args.output / "training_distillation_diagnostics.csv")
    frozen = {**provenance, "models": records,
        "small_train_families": sorted(set(dataset["families"][dataset["small_train_mask"]])),
        "large_train_families": sorted(set(dataset["families"][dataset["large_train_mask"]])),
        "calibration_families": sorted(set(dataset["families"][dataset["calibration_mask"]])),
        "selection_states": dataset["state_ids"][dataset["selection_mask"]].tolist(),
        "global_plan_selection": "own_training_initial_states_family_equal_priorities_only",
        "new_audit_used_for_training_or_selection": False, "frozen_before_any_new_audit_simulation": True,
        "research_freeze_allowed": False}
    json_write(path, frozen)
    return students, frozen


def seed_registry(training, audit):
    records = []
    for phase, scenarios in (("training", training), ("audit", audit)):
        for j, scenario in enumerate(scenarios):
            for rep in range(2):
                population = (150_000_000 if phase == "training" else 190_000_000) + 2 * j + rep
                for sample, step in enumerate(COLLECTION_STEPS):
                    record = {"phase": phase, "scenario_id": scenario.scenario_id, "repetition": rep, "step": step,
                        "population_seed": population, "population_sensor_seed": population + 9_000_001}
                    if phase == "training":
                        index, planning = training_seeds(j, rep, sample)
                    else:
                        index, planning, evaluation = audit_seeds(j, rep, sample)
                        record.update(evaluation_seeds=evaluation, evaluation_sensor_seeds=[s + 9_000_001 for s in evaluation],
                            uniform_candidate_seed=250_000_000 + index, uniform_ppo_seed=251_000_000 + index)
                    record.update(state_index=index, planning_seeds=planning, planning_sensor_seeds=[s + 9_000_001 for s in planning])
                    records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/ppo_extended_world_planning_development_v1")
    parser.add_argument("--historical", type=Path, default=ROOT / "outputs/observable_plan_distillation_development_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline_relative_plan_distillation_development_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError("preserve results; choose new output or exact --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.monotonic()
    old = json.loads((args.source / "development_manifest.json").read_text(encoding="utf-8"))
    historical = json.loads((args.historical / "frozen_students.json").read_text(encoding="utf-8"))
    previous = json.loads((args.historical / "development_manifest.json").read_text(encoding="utf-8"))
    split = stratified_family_split(old["scenario_records"])
    training = make_domain_randomized_trust_bank(60, 20261701, "PREX")
    audit = make_domain_randomized_trust_bank(36, 20261702, "PRAU")
    banks = [old["scenario_records"], previous["audit_scenario_records"], [asdict(s) for s in training], [asdict(s) for s in audit]]
    signatures = []
    for bank in banks:
        values = {scenario_signature(s) for s in bank}
        assert len(values) == len(bank)
        assert all(not values & prior for prior in signatures)
        signatures.append(values)
    source_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/fire_evac").glob("*.py"))
    source_files += ["scripts/run_relative_plan_distillation.py", "scripts/run_observable_plan_distillation.py",
        "scripts/run_source_world_planning.py", "tests/test_relative_plan_ranking.py", "tests/test_relative_plan_runner.py",
        "reports/扩展训练与相对PPO排序开发方案.md", "pyproject.toml", "pytest.ini"]
    registry = seed_registry(training, audit)
    manifest = {"evaluation_role": "factorial_training_coverage_and_baseline_relative_ranking_development_audit_not_final",
        "freeze_allowed": False, "source_directory": str(args.source.resolve()), "historical_directory": str(args.historical.resolve()),
        "source_manifest_sha256": sha256(args.source / "development_manifest.json"),
        "source_state_selection_sha256": sha256(args.source / "state_selection.csv"),
        "source_state_sha256": {p.stem: sha256(p) for p in sorted((args.source / "states").glob("*.npz"))},
        "historical_frozen_manifest_sha256": sha256(args.historical / "frozen_students.json"),
        "excluded_previous_audit_records": previous["audit_scenario_records"], "old_scenario_records": old["scenario_records"],
        "old_family_split": split, "split_seed": 20261600, "training_seed": 20261701, "audit_seed": 20261702,
        "training_scenario_records": banks[2], "audit_scenario_records": banks[3], "all_scenario_parameters_disjoint": True,
        "student_seeds": STUDENT_SEEDS, "groups": GROUPS, "arms": ARMS, "candidate_names": old["candidate_names"],
        "input_dim": 49, "plan_count": 17, "priority_fields": PRIORITY_FIELDS, "trace_fields": TRACE_FIELDS,
        "outcome_fields": OUTCOME_FIELDS, "collection_steps": COLLECTION_STEPS, "planning_futures": 3, "evaluation_futures": 3,
        "training": {"architecture": [49, 64, 64, 102], "maximum_epochs": 600, "patience": 100, "evaluate_every": 10,
            "lr": .003, "weight_decay": .001, "batch": "full", "normalization": "own_train_families_only",
            "checkpoint_selection": "common_old_calibration_initial_primary_teacher_regret_ties_keep_earliest_1e-12",
            "mse": "min_relative_standardized_six_priority_MSE",
            "rank": "baseline_601_center_after_aggregation_primary_MSE_plus_0.5_gap_logistic_plus_0.1_other_heads_MSE"},
        "global_plan_selection": "each_training_bank_initial_states_family_equal_only",
        "primary_subset": "initial_36_families_2_populations_3_independent_evaluation_futures",
        "primary_arm": "five_large_rank_expectation", "strong_fixed_comparator": "pure_ppo_601",
        "bootstrap": {"unit": "scenario_family", "draws": 2000, "seed": 20261703, "role": "exploratory"},
        "seed_formula": {"index": "10*j+5*r+sample", "training_population": "150000000+2*j+r",
            "training_trajectory": "160000000+10000*j+1000*r+t", "training_planning": "170000000+100*i+f",
            "audit_population": "190000000+2*j+r", "audit_trajectory": "200000000+10000*j+1000*r+t",
            "audit_planning": "210000000+100*i+f", "audit_evaluation": "230000000+100*i+f",
            "sensors": "domain_seed+9000001", "uniform_candidate": "250000000+i", "uniform_ppo": "251000000+i"},
        "seed_registry": registry, "source_hashes": {p: sha256(ROOT / p) for p in source_files},
        "behavior_artifact_sha256": sha256(ROOT / "outputs/behavior_pipeline/behavior_model.json"),
        "frozen_ppo_models": old["frozen_ppo_models"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "torch_threads": 1}}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "development_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("resume source/design/data provenance differs")
    else:
        json_write(manifest_path, manifest)
    for relative in source_files:
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha256(target) == manifest["source_hashes"][relative]
        else:
            shutil.copy2(ROOT / relative, target)
    controllers = []
    for record in manifest["frozen_ppo_models"]:
        path = Path(record["path"])
        assert sha256(path) == record["checkpoint_sha256"]
        controller = FrozenPPOContinuation(record["training_seed"], PPO.load(path, device="cpu"))
        assert policy_parameter_sha256(controller.model) == record["parameter_sha256"]
        controllers.append(controller)
    candidates = PLANS + tuple(controllers)
    assert [c.name for c in candidates] == manifest["candidate_names"]
    training_summary = collect(args, training, candidates, "training")
    json_write(args.output / "training_collection_summary.json", training_summary)
    dataset = training_dataset(args, split)
    json_write(args.output / "progress.json", {"phase": "training_twenty_students", "new_audit_started": False})
    students, frozen = train_and_freeze(args, manifest, historical, dataset)
    assert [r["arm"] for r in frozen["models"]] == list(STUDENT_ARMS)
    frozen_hash = sha256(args.output / "frozen_students.json")
    before = {"frozen_students_sha256": frozen_hash, "supervised_checkpoints": {r["arm"]: r["sha256"] for r in frozen["models"]},
        "audit_started": True, "all_models_frozen_before_any_audit_simulation": True}
    before_path = args.output / "audit_started_after_freeze.json"
    if before_path.exists():
        assert json.loads(before_path.read_text(encoding="utf-8")) == before
    else:
        json_write(before_path, before)
    audit_summary = collect(args, audit, candidates, "audit", students, frozen)
    for record in frozen["models"]:
        assert sha256(args.output / record["checkpoint"]) == record["sha256"]
    for controller, record in zip(controllers, manifest["frozen_ppo_models"]):
        assert policy_parameter_sha256(controller.model) == record["parameter_sha256"]
        assert sha256(Path(record["path"])) == record["checkpoint_sha256"]
    summary = {"evaluation_role": manifest["evaluation_role"], "freeze_allowed": False,
        "new_training_families": 60, "old_training_families": 24, "large_training_families": 84,
        "old_calibration_families": 12, "new_scenario_families": 36, "fresh_students": 20, "historical_students": 5,
        "execution_arms": len(ARMS), "training_collection": training_summary, **audit_summary,
        "total_unique_full_rollouts": training_summary["planning_rollouts"] + audit_summary["planning_rollouts"] + audit_summary["unique_evaluation_rollouts"],
        "all_candidate_models_and_students_unchanged": True, "student_freeze_sha256_before_audit": frozen_hash,
        "small_training_global_plan": frozen["small_global_plan_name"], "large_training_global_plan": frozen["large_global_plan_name"],
        "elapsed_seconds_this_invocation": time.monotonic() - start,
        "limitations": ["development families only; no formal final evaluation", "expanded data requires more simulation and full-batch computation",
            "ranking method includes baseline centering and ranking loss; separate effects not identified", "previous calibration is development information",
            "privileged offline labels lack individual observable crowd attributes", "same generator only; no map/behavior/noise generalization test",
            "source uncertainty only with exact noiseless compatibility", "three futures are not risk bounds", "supervised seeds are not PPO seeds",
            "one selected continuation per reachable start, not continuously reselected student", "scores are uncalibrated; no safety probability or hard mask",
            "no new PPO training, LLM calls, safety qualification, or equal-budget LLM/RL ablation"]}
    json_write(args.output / "run_summary.json", summary)
    json_write(args.output / "progress.json", {"phase": "complete", "states": audit_summary["states"], "freeze_allowed": False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
