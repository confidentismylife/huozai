"""Small predeclared set-supervision x empirical override development study."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.dynamic_conflicts import DynamicConflictScenario, make_domain_randomized_trust_bank
from fire_evac.source_world_planning import PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds, select_reference_plan
from fire_evac.ppo_reference_planning import FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, TRACE_FIELDS
from fire_evac.relative_plan_ranking import ObservableScoreStudent
from fire_evac.plan_distillation import choose_priority
from run_source_world_planning import make_env, vector
from run_crossed_teacher_rng_diagnostic import digest, read_json, read_csv, write_json, write_csv
from set_aware_plan_student import OBJECTIVES, SEEDS, fit, training_targets, SetGainStudent, nominate, ensemble_decision, calibrate_margin

OUT = ROOT / "outputs/set_aware_distillation_development_v1"
SOURCE = ROOT / "outputs/baseline_relative_plan_distillation_development_v1"
PROTOCOL = ROOT / "reports/集合监督与覆盖不确定性小规模开发方案_2026-10-03.md"
ENSEMBLE_ARMS = ("single_ungated", "single_guarded", "set_ungated", "set_guarded")
STUDENT_ARMS = tuple(f"{objective}_{seed}" for objective in OBJECTIVES for seed in SEEDS)
FIXED_ARMS = tuple(f"pure_ppo_{seed}" for seed in range(601, 606))
ARMS = ENSEMBLE_ARMS + STUDENT_ARMS + FIXED_ARMS + ("teacher_robust3", "legacy_large_rank_ensemble", "live_graded")


def seeds(j, r):
    index = 2 * j + r
    return 340_000_000 + index, tuple(350_000_000 + 1000 * index + f for f in range(3)), tuple(370_000_000 + 1000 * index + f for f in range(3))


def calibration_split(manifest, dataset):
    records = {r["scenario_id"]: r for r in manifest["old_scenario_records"]}
    cal = set(dataset["families"][dataset["calibration_mask"]].tolist())
    checkpoint, margin = [], []
    for kind in sorted({records[f]["conflict_type"] for f in cal}):
        ids = sorted(f for f in cal if records[f]["conflict_type"] == kind)
        assert len(ids) == 2
        checkpoint.append(ids[0])
        margin.append(ids[1])
    assert len(checkpoint) == len(margin) == 6 and set(checkpoint).isdisjoint(margin)
    return checkpoint, margin


def scenario_signature(record):
    import json
    return json.dumps({k: v for k, v in record.items() if k != "scenario_id"}, sort_keys=True)


def prepare(output):
    path = output / "development_manifest.json"
    if path.exists():
        manifest = read_json(path)
        for file, expected in manifest["source_input_sha256"].items():
            assert digest(file) == expected, file
        return manifest
    prior = read_json(ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1/completion_manifest.json")
    for file, expected in prior["files"].items():
        assert digest(ROOT / file) == expected, file
    old = read_json(SOURCE / "development_manifest.json")
    legacy = read_json(SOURCE / "frozen_students.json")
    with np.load(SOURCE / "training_dataset.npz", allow_pickle=False) as z:
        dataset = {k: z[k].copy() for k in z.files}
    cp, margin = calibration_split(old, dataset)
    scenarios = [asdict(s) for s in make_domain_randomized_trust_bank(24, 20262001, "SGDEV")]
    previous_scenarios = old["old_scenario_records"] + old["training_scenario_records"] + old["audit_scenario_records"] + old["excluded_previous_audit_records"]
    previous_scenarios += read_json(ROOT / "outputs/teacher_factorial_pilot_development_v1/development_manifest.json")["scenario_records"]
    files = {Path(__file__), PROTOCOL, SOURCE / "training_dataset.npz", SOURCE / "development_manifest.json", SOURCE / "frozen_students.json",
             ROOT / "scripts/set_aware_plan_student.py", ROOT / "scripts/analyze_set_aware_distillation.py",
             ROOT / "scripts/run_crossed_teacher_rng_diagnostic.py", ROOT / "scripts/run_source_world_planning.py",
             ROOT / "tests/test_set_aware_plan_student.py", ROOT / "outputs/behavior_pipeline/behavior_model.json",
             ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1/completion_manifest.json",
             ROOT / "outputs/teacher_factorial_pilot_development_v1/development_manifest.json"}
    files.update((ROOT / "src/fire_evac").glob("*.py"))
    for folder in ("confirmatory_joint_control", "guarded_confirmatory_v2"):
        for name in ("train_scenarios.json", "final_scenarios.json"):
            file = ROOT / "outputs" / folder / name
            previous_scenarios += read_json(file)
            files.add(file)
    prior_signatures = {scenario_signature(s) for s in previous_scenarios}
    assert len({scenario_signature(s) for s in scenarios}) == 24
    assert all(scenario_signature(s) not in prior_signatures for s in scenarios)
    assert set(Counter(s["conflict_type"] for s in scenarios).values()) == {4}
    # Check population/planning/evaluation and their sensor namespaces against all
    # prior 1,137 cached states; hashed preservation is distinct from fitting.
    prior_audit = ROOT / "outputs/observable_teacher_label_audit_development_v1"
    preservation = read_json(prior_audit / "diagnostic_manifest.json")["source_input_sha256"]
    for file, expected in preservation.items():
        assert digest(file) == expected, file
    previous_seeds = {int(r["population_seed"]) for r in read_csv(prior_audit / "state_primary_diagnostics.csv")}
    for file in preservation:
        if Path(file).suffix == ".npz" and Path(file).parent.name in ("states", "training_states", "audit_states"):
            with np.load(file, allow_pickle=False) as z:
                for key in ("planning_seeds", "evaluation_seeds"):
                    if key in z:
                        previous_seeds.update(map(int, z[key]))
    previous_seeds |= {seed + 9_000_001 for seed in list(previous_seeds)}
    registry = [{"family": scenarios[j]["scenario_id"], "rep": r, "population": seeds(j, r)[0],
                 "planning": list(seeds(j, r)[1]), "evaluation": list(seeds(j, r)[2])} for j in range(24) for r in range(2)]
    new_sets = [{entry["population"] for entry in registry}, {s for entry in registry for s in entry["planning"]},
                {s for entry in registry for s in entry["evaluation"]}]
    new_sets += [{s + 9_000_001 for s in group} for group in new_sets.copy()]
    for i, group in enumerate(new_sets):
        assert not group.intersection(previous_seeds)
        assert all(not group.intersection(other) for other in new_sets[i + 1:])
    legacy_records = [r for r in legacy["models"] if r["group"] == "large_rank"]
    assert len(legacy_records) == 5
    for record in legacy_records:
        file = SOURCE / record["checkpoint"]
        assert digest(file) == record["sha256"]
        files.add(file)
    for record in old["frozen_ppo_models"]:
        assert digest(record["path"]) == record["checkpoint_sha256"]
        files.add(Path(record["path"]))
    source_hashes = {str(Path(p).resolve()): expected for p, expected in preservation.items()}
    source_hashes.update({str(file.resolve()): digest(file) for file in files})
    body = {"role": "new_same_generator_development_set_supervision_empirical_override_pilot_not_final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_allowed": False,
        "audit_scenario_records": scenarios, "audit_seed": 20262001, "analysis_seed": 20262002,
        "candidate_names": old["candidate_names"], "arms": list(ARMS), "objectives": list(OBJECTIVES), "student_seeds": list(SEEDS),
        "checkpoint_family_ids": cp, "margin_family_ids": margin, "training_family_ids": sorted(set(dataset["families"][dataset["large_train_mask"]].tolist())),
        "set_tolerance_pp": .10, "ensemble_scale_floor_pp": .10, "minimum_override_lower_score_pp": .10,
        "margin_quantile": .80, "margin_quantile_method": "higher", "margin_k_minimum": 1,
        "training": {"architecture": [49, 64, 64, 119], "maximum_epochs": 300, "patience": 80, "evaluate_every": 10,
            "learning_rate": .003, "weight_decay": .001, "batch": "full", "primary_mse": 1, "auxiliary_mse": .1, "classification": 1,
            "checkpoint_selection": "checkpoint_families_initial_teacher_primary_regret_ties_keep_earliest"},
        "seed_registry": registry, "new_seeds_disjoint_prior_1137_cache_namespaces": True,
        "planning_rollout_budget": 4896, "evaluation_rollout_upper_bound": 2448, "full_rollout_upper_bound": 7344,
        "new_training_label_simulations": 0, "new_supervised_training_runs": 10, "new_ppo_training_runs": 0, "new_llm_calls": 0,
        "frozen_ppo_models": old["frozen_ppo_models"], "legacy_large_rank_records": legacy_records,
        "outcome_fields": list(OUTCOME_FIELDS), "trace_fields": list(TRACE_FIELDS), "source_input_sha256": source_hashes}
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, body)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for file in (Path(__file__), ROOT / "scripts/set_aware_plan_student.py", ROOT / "scripts/analyze_set_aware_distillation.py", PROTOCOL):
        shutil.copy2(file, snapshot / file.name)
    return body


def train(output, manifest):
    frozen_path = output / "frozen_students.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        assert frozen["manifest_sha256"] == digest(output / "development_manifest.json")
        for record in frozen["models"]:
            assert digest(output / record["checkpoint"]) == record["sha256"]
        return [SetGainStudent.load(output / r["checkpoint"]) for r in frozen["models"]], frozen
    assert not any((output / "audit_states").glob("*.npz")), "refuse training after audit"
    with np.load(SOURCE / "training_dataset.npz", allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    train_mask = d["large_train_mask"]
    cp = np.isin(d["families"], manifest["checkpoint_family_ids"])
    margin = np.isin(d["families"], manifest["margin_family_ids"])
    assert not (train_mask & (cp | margin)).any() and not (cp & margin).any()
    assert (train_mask | cp | margin).all() and int(train_mask.sum()) == 592
    included = train_mask | cp
    data_file = output / "derived_training_targets.npz"
    np.savez_compressed(data_file, **d, checkpoint_mask=cp, margin_mask=margin,
        signed_gains=training_targets(d["targets"], "set")[0], acceptable_sets=training_targets(d["targets"], "set")[1])
    (output / "models").mkdir(exist_ok=True)
    models, records, curves = [], [], []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            model, curve, info = fit(d["observations"][included], d["targets"][included], d["families"][included], d["steps"][included],
                train_mask[included], cp[included], seed, objective)
            assert set(info["fit_families"]).isdisjoint(manifest["margin_family_ids"])
            arm = f"{objective}_{seed}"
            path = output / "models" / f"{arm}.pt"
            model.save(path)
            records.append({"arm": arm, "objective": objective, "seed": seed, "checkpoint": f"models/{arm}.pt", "sha256": digest(path), **info})
            curves.extend({"arm": arm, **row} for row in curve)
            models.append(model)
            print(f"Frozen {arm}: epoch {info['selected_epoch']}/{info['actual_epochs']}, checkpoint regret {info['best_checkpoint_primary_regret_pp']:.4f} pp", flush=True)
    for seed in SEEDS:
        pair = [r for r in records if r["seed"] == seed]
        assert pair[0]["initial_network_sha256"] == pair[1]["initial_network_sha256"]
    margin_initial = margin & (d["steps"] == 0)
    actual = training_targets(d["targets"][margin_initial], "set")[0]
    margins = {}
    for objective in OBJECTIVES:
        chosen_models = [m for m in models if m.objective == objective]
        predictions = np.asarray([[m.predictions(x)[0] for x in d["observations"][margin_initial]] for m in chosen_models])
        k, info = calibrate_margin(predictions, actual, d["families"][margin_initial])
        assert len(info["family_max_optimistic_residuals"]) == 6
        margins[objective] = info
    frozen = {"manifest_sha256": digest(output / "development_manifest.json"), "derived_dataset_sha256": digest(data_file),
              "models": records, "margins": margins, "frozen_before_new_audit": True,
              "new_audit_used_for_training_or_margin": False, "research_freeze_allowed": False}
    write_csv(output / "training_curves.csv", curves)
    write_json(frozen_path, frozen)
    return models, frozen


def student_choices(observation, models, frozen, legacy):
    predictions = [m.predictions(observation) for m in models]
    gains, probabilities = np.asarray([x[0] for x in predictions]), np.asarray([x[1] for x in predictions])
    choices, diagnostics = [], []
    for index, objective in enumerate(OBJECTIVES):
        lo, hi = 5 * index, 5 * index + 5
        nominee, guarded, info = ensemble_decision(gains[lo:hi], probabilities[lo:hi], frozen["margins"][objective]["k"])
        choices += [nominee, guarded]
        diagnostics.append(info)
    choices += [nominate(g, p) for g, p in predictions]
    legacy_priorities = np.asarray([m.priorities(observation) for m in legacy])
    legacy_choice = choose_priority(legacy_priorities.mean(axis=0))
    return choices, gains, probabilities, diagnostics, legacy_choice, legacy_priorities


def evaluate(output, manifest, models, frozen):
    candidates = list(PLANS)
    for record in manifest["frozen_ppo_models"]:
        model = PPO.load(record["path"], device="cpu")
        assert policy_parameter_sha256(model) == record["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(record["training_seed"], model))
    assert [p.name for p in candidates] == manifest["candidate_names"]
    legacy = [ObservableScoreStudent.load(SOURCE / record["checkpoint"]) for record in manifest["legacy_large_rank_records"]]
    scenarios = [DynamicConflictScenario(**s) for s in manifest["audit_scenario_records"]]
    actual, worlds = make_env(scenarios), [make_env(scenarios), make_env(scenarios)]
    state_dir = output / "audit_states"
    state_dir.mkdir(exist_ok=True)
    marker = {"frozen_students_sha256": digest(output / "frozen_students.json"), "audit_started_after_freeze": True}
    if (output / "audit_started_after_freeze.json").exists():
        assert read_json(output / "audit_started_after_freeze.json") == marker
    else:
        write_json(output / "audit_started_after_freeze.json", marker)
    started, new_planning, new_evaluation = time.monotonic(), 0, 0
    try:
        for j, scenario in enumerate(scenarios):
            for r in range(2):
                population, planning, evaluation = seeds(j, r)
                state_id = f"{scenario.scenario_id}_{r:02d}_000"
                path = state_dir / f"{state_id}.npz"
                observation, _ = actual.reset(seed=population, options={"scenario": scenario})
                # Observable recommendations are frozen before any teacher query.
                choices, gains, probabilities, diagnostics, legacy_choice, legacy_priorities = student_choices(observation, models, frozen, legacy)
                if not path.exists():
                    supported, errors, mismatch = compatible_source_worlds(source_world_scenarios(scenario), observation, 0)
                    assert supported.all() and not mismatch
                    risks = np.empty((2, 17, 3, len(OUTCOME_FIELDS)))
                    for w, hypothesis in enumerate(source_world_scenarios(scenario)):
                        world_observation, _ = worlds[w].reset(seed=population, options={"scenario": hypothesis})
                        for p, candidate in enumerate(candidates):
                            for f, seed in enumerate(planning):
                                risks[w, p, f] = vector(evaluate_continuation(worlds[w], world_observation, candidate, seed))
                                new_planning += 1
                    teacher = select_reference_plan(risks, supported, scenario.forecast_source_branch, "robust")
                    selected = choices + list(range(12, 17)) + [teacher, legacy_choice, 0]
                    assert len(selected) == len(ARMS) == 22
                    outcomes, evaluated, traces = np.empty((22, 3, len(OUTCOME_FIELDS))), {}, {}
                    for arm, p in enumerate(selected):
                        for f, seed in enumerate(evaluation):
                            if (p, f) not in evaluated:
                                trace = []
                                evaluated[(p, f)] = vector(evaluate_continuation(actual, observation, candidates[p], seed, trace))
                                traces[(p, f)] = np.asarray(trace, float)
                                new_evaluation += 1
                            outcomes[arm, f] = evaluated[(p, f)]
                    keys = sorted(evaluated)
                    tmp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(tmp, observation=observation, supported=supported, world_errors=errors,
                        population_seed=np.asarray(population), planning_seeds=np.asarray(planning), evaluation_seeds=np.asarray(evaluation),
                        risks=risks, gains=gains, probabilities=probabilities, legacy_priorities=legacy_priorities,
                        chosen_indices=np.asarray(selected), evaluation_outcomes=outcomes,
                        planning_rollouts=np.asarray(102), evaluation_rollouts=np.asarray(len(evaluated)),
                        trace_keys=np.asarray(keys), trace_offsets=np.cumsum([0] + [len(traces[k]) for k in keys]),
                        execution_trace=np.concatenate([traces[k] for k in keys]))
                    tmp.replace(path)
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["observation"], observation)
                    np.testing.assert_array_equal(z["gains"], gains)
                    np.testing.assert_array_equal(z["probabilities"], probabilities)
                    np.testing.assert_array_equal(z["legacy_priorities"], legacy_priorities)
                    np.testing.assert_array_equal(z["chosen_indices"][:14], choices)
                    np.testing.assert_array_equal(z["planning_seeds"], planning)
                    np.testing.assert_array_equal(z["evaluation_seeds"], evaluation)
                    assert int(z["population_seed"]) == population
            elapsed = time.monotonic() - started
            write_json(output / "progress.json", {"phase": "new_development_audit", "families_completed": j + 1, "families_total": 24,
                "new_planning_rollouts_this_invocation": new_planning, "new_evaluation_rollouts_this_invocation": new_evaluation,
                "elapsed_seconds_this_invocation": elapsed, "new_supervised_training_runs_total": 10})
            print(f"Audit {j+1}/24 {scenario.scenario_id}; new planning {new_planning}, evaluation {new_evaluation}; {elapsed:.1f}s", flush=True)
    finally:
        actual.close()
        for world in worlds:
            world.close()
    for record, candidate in zip(manifest["frozen_ppo_models"], candidates[12:]):
        assert policy_parameter_sha256(candidate.model) == record["parameter_sha256"]
    return new_planning, new_evaluation, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output == SOURCE or SOURCE in output.parents:
        raise ValueError("use a new output directory within the workspace")
    if output.exists() and not (args.resume or args.analyze_only):
        raise FileExistsError("preserve existing output; choose new output or exact --resume")
    manifest = prepare(output)
    if args.prepare_only:
        print("Prepared: 10 supervised fits; at most 7,344 new full rollouts; no simulation yet.")
        return
    torch.set_num_threads(1)
    if args.analyze_only:
        planning, evaluation, elapsed = 0, 0, 0
    else:
        models, frozen = train(output, manifest)
        planning, evaluation, elapsed = evaluate(output, manifest, models, frozen)
    from analyze_set_aware_distillation import analyze
    analyze(output, planning, evaluation, elapsed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
