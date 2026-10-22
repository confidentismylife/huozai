"""Freeze observable score students on old families, then audit new families."""
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

import numpy as np
from stable_baselines3 import PPO
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.counterfactual_shield import reactive_followup_action
from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank
from fire_evac.source_world_planning import PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds, select_reference_plan
from fire_evac.ppo_reference_planning import FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, TRACE_FIELDS
from fire_evac.plan_distillation import (
    STUDENT_SEEDS, PRIORITY_FIELDS, ObservablePlanStudent, relative_priority_targets,
    fit_student, stratified_family_split, choose_priority, family_weights,
)
from run_source_world_planning import make_env, replay_world, vector, write_csv

FULL_ARMS = tuple(f"full_student_{s}" for s in STUDENT_SEEDS)
FORECAST_ARMS = tuple(f"forecast_student_{s}" for s in STUDENT_SEEDS)
FIXED_ARMS = tuple(f"pure_ppo_{s}" for s in range(601, 606))
ARMS = FULL_ARMS + FORECAST_ARMS + FIXED_ARMS + (
    "teacher_minimax", "teacher_point", "teacher_oracle", "train_global_plan", "uniform_candidate", "uniform_ppo", "live_graded")
COLLECTION_STEPS = (0, 8, 16, 24, 40)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_seeds(scenario_index, repetition, sample_index):
    state_index = scenario_index * 10 + repetition * 5 + sample_index
    return state_index, tuple(90_000_000 + state_index * 100 + f for f in range(3)), tuple(110_000_000 + state_index * 100 + f for f in range(3))


def scenario_signature(record):
    return json.dumps({k: v for k, v in record.items() if k != "scenario_id"}, sort_keys=True)


def train_and_freeze(args, old_manifest, split, manifest_path):
    selections = read_csv(args.source / "state_selection.csv")
    x, targets, families, state_ids, steps = [], [], [], [], []
    for row in selections:
        with np.load(args.source / "states" / f"{row['state_id']}.npz", allow_pickle=False) as state:
            x.append(state["observation"])
            targets.append(relative_priority_targets(state["risks"], state["supported"]))
        families.append(row["source_id"])
        state_ids.append(row["state_id"])
        steps.append(int(row["step"]))
    x, targets, families, steps = np.asarray(x), np.asarray(targets), np.asarray(families), np.asarray(steps)
    train_mask = np.asarray([split[f] == "train" for f in families])
    calibration_mask = ~train_mask
    dataset_path = args.output / "old_development_targets.npz"
    if dataset_path.exists():
        with np.load(dataset_path, allow_pickle=False) as saved:
            for key, values in (("observations", x), ("targets", targets), ("families", families),
                                ("train_mask", train_mask), ("calibration_mask", calibration_mask)):
                np.testing.assert_array_equal(saved[key], values)
    else:
        np.savez_compressed(dataset_path, observations=x, targets=targets, families=families,
                            state_ids=np.asarray(state_ids), steps=steps, train_mask=train_mask, calibration_mask=calibration_mask)
    global_plan = choose_priority((targets[train_mask] * family_weights(families[train_mask])[:, None, None]).sum(axis=0))
    freeze_path = args.output / "frozen_students.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen["protocol_sha256"] != sha256(manifest_path) or frozen["dataset_sha256"] != sha256(dataset_path):
            raise ValueError("frozen training provenance differs")
        if frozen["global_plan_index"] != global_plan:
            raise ValueError("training-only global plan differs")
        for record in frozen["models"]:
            if sha256(args.output / record["checkpoint"]) != record["sha256"]:
                raise ValueError("frozen student checkpoint changed")
        students = [ObservablePlanStudent.load(args.output / r["checkpoint"]) for r in frozen["models"]]
        return students, frozen
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    students, records, curves, diagnostics = [], [], [], []
    for forecast_only, arms in ((False, FULL_ARMS), (True, FORECAST_ARMS)):
        for seed, arm in zip(STUDENT_SEEDS, arms):
            student, curve, info = fit_student(x, targets, families, train_mask, calibration_mask, seed, forecast_only)
            checkpoint = model_dir / f"{arm}.pt"
            student.save(checkpoint)
            students.append(student)
            records.append({"arm": arm, "seed": seed, "forecast_only": forecast_only,
                            "checkpoint": str(checkpoint.relative_to(args.output)).replace("\\", "/"),
                            "sha256": sha256(checkpoint), **info})
            curves.extend({"arm": arm, **point} for point in curve)
            predictions = np.asarray([student.priorities(observation) for observation in x])
            chosen = np.asarray([choose_priority(p) for p in predictions])
            teacher_chosen = np.asarray([choose_priority(t) for t in targets])
            for subset, mask in (("train", train_mask), ("calibration", calibration_mask),
                                  ("calibration_initial", calibration_mask & (steps == 0))):
                weights = family_weights(families[mask])
                normalized_error = (((predictions[mask] - targets[mask]) / student.target_scale) ** 2).mean(axis=(1, 2))
                regrets = targets[np.arange(len(x)), chosen, 0]
                diagnostics.append({"arm": arm, "subset": subset, "states": int(mask.sum()), "families": len(set(families[mask])),
                                    "family_equal_standardized_mse": float(np.dot(normalized_error, weights)),
                                    "family_equal_teacher_incap_regret_pp": float(np.dot(regrets[mask], weights) * 100),
                                    "family_equal_exact_teacher_agreement": float(np.dot((chosen[mask] == teacher_chosen[mask]), weights))})
            print(f"Frozen {arm}: epoch {info['selected_epoch']}/{info['actual_epochs']}, "
                  f"calibration loss {info['best_calibration_loss']:.5f}", flush=True)
    write_csv(curves, args.output / "training_curves.csv")
    write_csv(diagnostics, args.output / "old_family_distillation_diagnostics.csv")
    frozen = {"protocol_sha256": sha256(manifest_path), "dataset_sha256": sha256(dataset_path), "models": records,
              "global_plan_index": global_plan, "global_plan_name": old_manifest["candidate_names"][global_plan],
              "train_families": sorted(set(families[train_mask])), "calibration_families": sorted(set(families[calibration_mask])),
              "new_audit_used_for_training_or_selection": False,
              "frozen_before_any_new_audit_simulation": True, "research_freeze_allowed": False}
    json_write(freeze_path, frozen)
    return students, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/ppo_extended_world_planning_development_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/observable_plan_distillation_development_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError("preserve results; choose new output or exact --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.monotonic()
    old_manifest = json.loads((args.source / "development_manifest.json").read_text(encoding="utf-8"))
    split = stratified_family_split(old_manifest["scenario_records"])
    audit_scenarios = make_domain_randomized_trust_bank(24, 20261601, "PDDEV")
    old_signatures = {scenario_signature(s) for s in old_manifest["scenario_records"]}
    audit_signatures = {scenario_signature(asdict(s)) for s in audit_scenarios}
    if old_signatures & audit_signatures or len(audit_signatures) != len(audit_scenarios):
        raise ValueError("audit scenario parameters duplicated")
    source_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/fire_evac").glob("*.py"))
    source_files += ["scripts/run_observable_plan_distillation.py", "scripts/run_source_world_planning.py",
                     "tests/test_plan_distillation.py", "reports/可观测计划软评分蒸馏开发方案.md", "pyproject.toml", "pytest.ini"]
    manifest = {"evaluation_role": "new_development_family_audit_of_observable_plan_distillation_not_final_confirmation",
                "freeze_allowed": False, "source_directory": str(args.source.resolve()),
                "source_manifest_sha256": sha256(args.source / "development_manifest.json"),
                "source_state_sha256": {p.stem: sha256(p) for p in sorted((args.source / "states").glob("*.npz"))},
                "old_family_split": split, "split_seed": 20261600, "candidate_names": old_manifest["candidate_names"],
                "audit_seed": 20261601, "audit_scenario_records": [asdict(s) for s in audit_scenarios],
                "audit_parameters_disjoint_from_training_calibration": True,
                "student_seeds": STUDENT_SEEDS, "arms": ARMS, "input_dim": 49, "plan_count": 17,
                "priority_fields": PRIORITY_FIELDS, "trace_fields": TRACE_FIELDS, "outcome_fields": OUTCOME_FIELDS,
                "collection_steps": COLLECTION_STEPS, "planning_futures": 3, "evaluation_futures": 3,
                "training": {"hidden": [64, 64], "epochs": 600, "patience_epochs": 100, "evaluate_every": 10,
                             "lr": .003, "weight_decay": .001, "loss": "family-equal standardized six-score MSE",
                             "normalization": "train families only", "target": "relative six lexicographic robust priorities",
                             "checkpoint_selection": "old calibration loss only"},
                "primary_subset": "initial 24 families x 2 populations x 3 new evaluation futures per execution arm",
                "primary_arm": "five_full_students_expectation", "strong_fixed_comparator": "pure_ppo_601",
                "bootstrap": {"unit": "scenario_family", "draws": 2000, "seed": 20261602, "role": "exploratory"},
                "seed_formula": {"population": "70000000+2*scenario_index+rep", "trajectory": "80000000+10000*scenario_index+1000*rep+step",
                                 "state_index": "10*scenario_index+5*rep+sample_index", "planning": "90000000+100*state_index+f",
                                 "evaluation": "110000000+100*state_index+f", "sensors": "domain_seed+9000001",
                                 "uniform_candidate": "130000000+state_index", "uniform_ppo": "131000000+state_index"},
                "source_hashes": {p: sha256(ROOT / p) for p in source_files},
                "behavior_artifact_sha256": sha256(ROOT / "outputs/behavior_pipeline/behavior_model.json"),
                "frozen_ppo_models": old_manifest["frozen_ppo_models"],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "torch_threads": 1}}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "development_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("resume data, design, source code, or model provenance differs")
    json_write(manifest_path, manifest)
    for relative in source_files:
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != manifest["source_hashes"][relative]:
            raise ValueError("source snapshot differs")
        shutil.copy2(ROOT / relative, target)
    json_write(args.output / "progress.json", {"phase": "training_old_families", "new_audit_started": False})
    students, frozen = train_and_freeze(args, old_manifest, split, manifest_path)
    frozen_hash = sha256(args.output / "frozen_students.json")
    assert len(students) == 10
    controllers = []
    for record in manifest["frozen_ppo_models"]:
        path = Path(record["path"])
        if sha256(path) != record["checkpoint_sha256"]:
            raise ValueError("frozen PPO file differs from prior protocol")
        controller = FrozenPPOContinuation(record["training_seed"], PPO.load(path, device="cpu"))
        if policy_parameter_sha256(controller.model) != record["parameter_sha256"]:
            raise ValueError("frozen PPO parameter hash differs")
        controllers.append(controller)
    candidates = PLANS + tuple(controllers)
    if [p.name for p in candidates] != manifest["candidate_names"]:
        raise ValueError("candidate IDs differ from distillation target IDs")
    json_write(args.output / "audit_started_after_freeze.json", {"frozen_students_sha256": frozen_hash,
        "supervised_checkpoints": {r["arm"]: r["sha256"] for r in frozen["models"]}, "audit_started": True})
    state_dir = args.output / "audit_states"
    state_dir.mkdir(exist_ok=True)
    actual_env = make_env(audit_scenarios)
    world_envs = [make_env(audit_scenarios), make_env(audit_scenarios)]
    episodes, selections, student_diagnostics = [], [], []
    planning_rollouts = eval_rollouts = execution_steps = neutral_steps = blocked_steps = 0
    for scenario_index, scenario in enumerate(audit_scenarios):
        hypotheses = source_world_scenarios(scenario)
        for rep in range(2):
            population_seed = 70_000_000 + scenario_index * 2 + rep
            observation, _ = actual_env.reset(seed=population_seed, options={"scenario": scenario})
            commands, step, done = [], 0, False
            while not done and step <= max(COLLECTION_STEPS):
                if step in COLLECTION_STEPS:
                    sample_index = COLLECTION_STEPS.index(step)
                    state_index, future_seeds, eval_seeds = audit_seeds(scenario_index, rep, sample_index)
                    state_id = f"{scenario.scenario_id}_{rep:02d}_{step:03d}"
                    destination = state_dir / f"{state_id}.npz"
                    if not destination.exists():
                        # Student recommendations are computed before any world-score query.
                        student_scores = np.asarray([student.priorities(observation) for student in students])
                        chosen = [choose_priority(scores) for scores in student_scores]
                        supported, errors, mismatch = compatible_source_worlds(hypotheses, observation, step)
                        if not supported[scenario.source_branch]:
                            raise RuntimeError("noiseless model excluded true source")
                        risks = np.full((2, 17, 3, len(OUTCOME_FIELDS)), np.nan)
                        for source, hypothesis in enumerate(hypotheses):
                            world_observation, early_done = replay_world(world_envs[source], hypothesis, population_seed, commands)
                            if not supported[source]:
                                continue
                            if early_done:
                                raise RuntimeError("supported world ended before counterfactual start")
                            for p, candidate in enumerate(candidates):
                                for f, seed in enumerate(future_seeds):
                                    risks[source, p, f] = vector(evaluate_continuation(world_envs[source], world_observation, candidate, seed))
                        chosen += list(range(12, 17))
                        chosen += [select_reference_plan(risks, supported, scenario.forecast_source_branch, mode,
                                   true_source=scenario.source_branch if mode == "oracle" else None) for mode in ("robust", "point", "oracle")]
                        chosen += [frozen["global_plan_index"], int(np.random.default_rng(130_000_000 + state_index).integers(17)),
                                   int(np.random.default_rng(131_000_000 + state_index).integers(12, 17)), 0]
                        if len(chosen) != len(ARMS):
                            raise RuntimeError("plan IDs and arm IDs differ")
                        outcomes = np.empty((len(ARMS), 3, len(OUTCOME_FIELDS)))
                        evaluated, traces = {}, {}
                        for a, p in enumerate(chosen):
                            for f, seed in enumerate(eval_seeds):
                                if (p, f) not in evaluated:
                                    trace = []
                                    evaluated[(p, f)] = vector(evaluate_continuation(actual_env, observation, candidates[p], seed, trace))
                                    traces[(p, f)] = np.asarray(trace, np.float64)
                                outcomes[a, f] = evaluated[(p, f)]
                        keys = sorted(evaluated)
                        offsets = np.cumsum([0] + [len(traces[k]) for k in keys])
                        temporary = destination.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, observation=observation, supported=supported, world_errors=errors,
                            model_mismatch=np.asarray(mismatch), risks=risks, student_priorities=student_scores,
                            chosen_indices=np.asarray(chosen), evaluation_outcomes=outcomes, planning_seeds=np.asarray(future_seeds),
                            evaluation_seeds=np.asarray(eval_seeds), planning_rollouts=np.asarray(int(supported.sum()) * 17 * 3),
                            evaluation_rollouts=np.asarray(len(evaluated)), execution_trace=np.concatenate([traces[k] for k in keys]),
                            trace_keys=np.asarray(keys), trace_offsets=offsets)
                        temporary.replace(destination)
                    with np.load(destination, allow_pickle=False) as cached:
                        np.testing.assert_array_equal(cached["observation"], observation)
                        np.testing.assert_array_equal(cached["planning_seeds"], future_seeds)
                        np.testing.assert_array_equal(cached["evaluation_seeds"], eval_seeds)
                        for a, student in enumerate(students):
                            np.testing.assert_array_equal(cached["student_priorities"][a], student.priorities(observation))
                            assert int(cached["chosen_indices"][a]) == student.choose(observation)
                        supported = cached["supported"]
                        planning_rollouts += int(cached["planning_rollouts"])
                        eval_rollouts += int(cached["evaluation_rollouts"])
                        execution_steps += len(cached["execution_trace"])
                        neutral_steps += int((cached["execution_trace"][:, 2] == -1).sum())
                        blocked_steps += int(cached["execution_trace"][:, 5].sum())
                        common = {"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": scenario.scenario_id,
                                  "conflict_type": scenario.conflict_type, "population_seed": population_seed, "repetition": rep,
                                  "trajectory_policy": "random" if rep == 0 else "reactive", "step": step,
                                  "supported_world_count": int(supported.sum()),
                                  "forecast_wrong": scenario.forecast_source_branch != scenario.source_branch}
                        selections.append({**common, "model_mismatch": bool(cached["model_mismatch"]),
                            **{arm: candidates[int(cached["chosen_indices"][a])].name for a, arm in enumerate(ARMS)}})
                        targets = relative_priority_targets(cached["risks"], supported)
                        teacher_choice = choose_priority(targets)
                        for a, arm in enumerate(ARMS):
                            p = int(cached["chosen_indices"][a])
                            for f, seed in enumerate(eval_seeds):
                                episodes.append({**common, "arm": arm, "selected_plan": candidates[p].name, "future_seed": int(seed),
                                                 **dict(zip(OUTCOME_FIELDS, cached["evaluation_outcomes"][a, f]))})
                            if arm in FULL_ARMS + FORECAST_ARMS:
                                student_diagnostics.append({**common, "arm": arm, "selected_plan": candidates[p].name,
                                    "exact_teacher_agreement": p == teacher_choice, "teacher_incap_regret_pp": float(targets[p, 0] * 100),
                                    "standardized_score_mse": float(np.mean(((cached["student_priorities"][a] - targets) / students[a].target_scale) ** 2))})
                if step == max(COLLECTION_STEPS):
                    break
                action = reactive_followup_action(observation) if rep == 1 else int(np.random.default_rng(
                    80_000_000 + scenario_index * 10_000 + rep * 1000 + step).integers(9))
                commands.append(action)
                observation, _, term, trunc, _ = actual_env.step(action)
                done, step = term or trunc, step + 1
        print(f"New development audit {scenario_index + 1}/24 {scenario.scenario_id}: "
              f"{len(selections)} states, {planning_rollouts} planning, {eval_rollouts} unique evaluation, "
              f"{time.monotonic() - start:.1f}s", flush=True)
        json_write(args.output / "progress.json", {"phase": "new_family_audit", "completed_scenarios": scenario_index + 1,
                   "states": len(selections), "planning_rollouts": planning_rollouts, "unique_evaluation_rollouts": eval_rollouts})
    actual_env.close()
    for env in world_envs:
        env.close()
    for student_record in frozen["models"]:
        if sha256(args.output / student_record["checkpoint"]) != student_record["sha256"]:
            raise RuntimeError("student checkpoint changed during audit")
    for controller, record in zip(controllers, manifest["frozen_ppo_models"]):
        if policy_parameter_sha256(controller.model) != record["parameter_sha256"] or sha256(Path(record["path"])) != record["checkpoint_sha256"]:
            raise RuntimeError("frozen candidate PPO changed")
    write_csv(episodes, args.output / "evaluation_episode_results.csv")
    write_csv(selections, args.output / "state_selection.csv")
    write_csv(student_diagnostics, args.output / "new_family_distillation_diagnostics.csv")
    summary = {"evaluation_role": manifest["evaluation_role"], "freeze_allowed": False, "new_scenario_families": 24,
               "states": len(selections), "initial_states": sum(r["step"] == 0 for r in selections),
               "ambiguous_states": sum(r["supported_world_count"] == 2 for r in selections),
               "model_mismatch_states": sum(r["model_mismatch"] for r in selections),
               "planning_rollouts": planning_rollouts, "unique_evaluation_rollouts": eval_rollouts,
               "total_unique_full_rollouts": planning_rollouts + eval_rollouts, "execution_arm_records": len(episodes),
               "actual_execution_trace_steps": execution_steps, "neutral_release_trace_steps": neutral_steps,
               "sign_hold_blocked_trace_steps": blocked_steps, "all_candidate_models_and_students_unchanged": True,
               "student_freeze_sha256_before_audit": frozen_hash, "training_global_plan": frozen["global_plan_name"],
               "elapsed_seconds_this_invocation": time.monotonic() - start,
               "limitations": ["development families only; not formal final evaluation", "observable students lack privileged individual crowd attributes",
                   "source uncertainty only, exact noiseless compatibility", "three teacher and evaluation futures are not risk bounds",
                   "supervised initialization seeds are not PPO training seeds", "one plan selection per counterfactual start",
                   "negative predicted relative scores are uncalibrated rankings, not safety probabilities",
                   "no new PPO, LLM, hard-filter qualification, or equal-budget LLM/RL ablation"]}
    json_write(args.output / "run_summary.json", summary)
    json_write(args.output / "progress.json", {"phase": "complete", "states": len(selections), "freeze_allowed": False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
