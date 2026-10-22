"""Family-level exploratory reporting for the predeclared 2x2 developer audit.

This script never fits a model, changes a candidate, or reads an audit outcome
to select a checkpoint. Teacher planning scores and independent execution
outcomes are kept separate throughout the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.plan_distillation import ObservablePlanStudent, choose_priority, family_weights, relative_priority_targets, observable_features
from fire_evac.relative_plan_ranking import ObservableScoreStudent, baseline_relative_targets
from fire_evac.source_world_planning import OUTCOME_FIELDS, select_reference_plan

GROUPS = ("small_mse", "small_rank", "large_mse", "large_rank", "historical_mse")
SEEDS = tuple(range(801, 806))
STUDENT_ARMS = tuple(f"{group}_{seed}" for group in GROUPS for seed in SEEDS)
FIXED_ARMS = tuple(f"pure_ppo_{seed}" for seed in range(601, 606))
CONTROL_ARMS = ("teacher_minimax", "teacher_point", "teacher_oracle", "small_global_plan",
                "large_global_plan", "uniform_ppo", "uniform_candidate", "live_graded")
ARMS = STUDENT_ARMS + FIXED_ARMS + CONTROL_ARMS
VIRTUAL_GROUPS = {f"five_{group}_expectation": tuple(f"{group}_{seed}" for seed in SEEDS) for group in GROUPS}
VIRTUAL_GROUPS["five_fixed_ppo_expectation"] = FIXED_ARMS
FOCUS = "five_large_rank_expectation"
METRICS = OUTCOME_FIELDS[:9]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(rows, path):
    if not rows:
        raise ValueError(f"refusing an empty analysis table: {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def typed_rows(path):
    rows = read_csv(path)
    for row in rows:
        for field in METRICS:
            row[field] = float(row[field])
        for field in ("step", "repetition", "supported_world_count", "future_seed", "population_seed"):
            row[field] = int(row[field])
        row["forecast_wrong"] = str(row["forecast_wrong"]).lower() == "true"
    return rows


def append_expectations(episodes):
    """Average outcome of five alternatives, never vote on model actions."""
    result = list(episodes)
    for virtual, arms in VIRTUAL_GROUPS.items():
        grouped = {}
        for row in episodes:
            if row["arm"] in arms:
                grouped.setdefault((row["state_id"], row["future_seed"]), []).append(row)
        if not grouped:
            raise ValueError(f"no episodes for {virtual}")
        for values in grouped.values():
            if len(values) != 5 or {r["arm"] for r in values} != set(arms):
                raise ValueError(f"five aligned retained models required for {virtual}")
            for field in ("source_id", "step", "forecast_wrong", "supported_world_count"):
                if len({r[field] for r in values}) != 1:
                    raise ValueError(f"expectation rows disagree on {field}")
            row = {k: v for k, v in values[0].items() if k not in OUTCOME_FIELDS}
            row.update(arm=virtual, selected_plan="expected_uniform_member_outcome")
            row.update({metric: float(np.mean([r[metric] for r in values])) for metric in METRICS})
            result.append(row)
    return result


def family_means(rows, metric="final_incapacitation"):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_id"], []).append(float(row[metric]))
    return {family: float(np.mean(values)) for family, values in grouped.items()}


def paired_family_deltas(treatment, control, metric="final_incapacitation"):
    lookup = {(r["state_id"], r["future_seed"]): r for r in control}
    if len(lookup) != len(control):
        raise ValueError("duplicate control start/future keys")
    keys = {(r["state_id"], r["future_seed"]) for r in treatment}
    if len(keys) != len(treatment) or keys != set(lookup):
        raise ValueError("paired comparisons require identical unique starts and futures")
    differences = []
    for row in treatment:
        other = lookup[(row["state_id"], row["future_seed"])]
        if row["source_id"] != other["source_id"]:
            raise ValueError("paired results belong to different families")
        differences.append({"source_id": row["source_id"], metric: float(row[metric]) - float(other[metric])})
    return family_means(differences, metric)


def bootstrap_effect(deltas, rng):
    values = np.asarray([deltas[family] for family in sorted(deltas)], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty scene-family deltas required")
    boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1) * 100
    return {"families": len(values), "delta_pp": float(values.mean() * 100),
            "exploratory_ci_low_pp": float(np.quantile(boot, .025)),
            "exploratory_ci_high_pp": float(np.quantile(boot, .975)),
            "improved_families": int((values < -1e-12).sum()),
            "worsened_families": int((values > 1e-12).sum()),
            "tied_families": int((np.abs(values) <= 1e-12).sum())}


def subset_includes(subset, row):
    if subset == "initial":
        return row["step"] == 0
    if subset == "all_states":
        return True
    if subset == "ambiguous":
        return row["supported_world_count"] == 2
    if subset == "identified":
        return row["supported_world_count"] == 1
    if subset == "forecast_wrong_initial":
        return row["step"] == 0 and row["forecast_wrong"]
    if subset == "forecast_correct_initial":
        return row["step"] == 0 and not row["forecast_wrong"]
    return row["step"] == 0 and row["conflict_type"] == subset.split(":", 1)[1]


def summarize(episodes, output):
    rows = append_expectations(episodes)
    rng = np.random.default_rng(20261703)
    subsets = ("initial", "all_states", "ambiguous", "identified", "forecast_wrong_initial", "forecast_correct_initial")
    subsets += tuple("conflict_initial:" + c for c in sorted({r["conflict_type"] for r in rows}))
    comparisons = [(FOCUS, control) for control in (
        "pure_ppo_601", "large_global_plan", "small_global_plan", "five_fixed_ppo_expectation",
        "five_historical_mse_expectation", "teacher_minimax", "teacher_point", "teacher_oracle",
        "uniform_ppo", "uniform_candidate", "live_graded")]
    comparisons += [("five_large_rank_expectation", "five_large_mse_expectation"),
                    ("five_large_rank_expectation", "five_small_rank_expectation"),
                    ("five_large_mse_expectation", "five_small_mse_expectation"),
                    ("five_small_rank_expectation", "five_small_mse_expectation")]
    comparisons += [(arm, "pure_ppo_601") for arm in STUDENT_ARMS]
    comparisons += [("teacher_minimax", "pure_ppo_601"), ("large_global_plan", "pure_ppo_601"),
                    ("small_global_plan", "pure_ppo_601")]
    summaries, effects, family_rows, interactions = [], [], [], []
    for subset in subsets:
        selected = [r for r in rows if subset_includes(subset, r)]
        if not selected:
            continue
        by_arm = {arm: [r for r in selected if r["arm"] == arm] for arm in ARMS + tuple(VIRTUAL_GROUPS)}
        for arm, values in by_arm.items():
            if not values:
                raise ValueError(f"missing {arm} in {subset}")
            means = {metric: family_means(values, metric) for metric in METRICS}
            summaries.append({"subset": subset, "arm": arm, "records": len(values),
                "families": len(means[METRICS[0]]), "virtual_expectation": arm in VIRTUAL_GROUPS,
                **{metric: float(np.mean(list(family_values.values()))) for metric, family_values in means.items()}})
        for arm, control in comparisons:
            deltas = paired_family_deltas(by_arm[arm], by_arm[control])
            effects.append({"subset": subset, "arm": arm, "control": control, **bootstrap_effect(deltas, rng)})
            family_rows += [{"subset": subset, "arm": arm, "control": control,
                            "source_id": family, "delta_pp": delta * 100} for family, delta in sorted(deltas.items())]
        ranking_large = paired_family_deltas(by_arm["five_large_rank_expectation"], by_arm["five_large_mse_expectation"])
        ranking_small = paired_family_deltas(by_arm["five_small_rank_expectation"], by_arm["five_small_mse_expectation"])
        if set(ranking_large) != set(ranking_small):
            raise ValueError("factorial interaction must use the same families")
        difference = {family: ranking_large[family] - ranking_small[family] for family in ranking_large}
        interactions.append({"subset": subset, "contrast": "(large_rank-large_mse)-(small_rank-small_mse)",
            **bootstrap_effect(difference, rng)})
    write_csv(summaries, output / "subset_summary.csv")
    write_csv(effects, output / "scenario_paired_effects.csv")
    write_csv(family_rows, output / "family_paired_deltas.csv")
    write_csv(interactions, output / "factorial_interactions.csv")
    return summaries, effects, interactions


def dataset_checks(output, manifest, frozen, students, audit_families, new_training_families):
    """Check immutable dataset partitions and training-only fixed choices."""
    with np.load(output / "training_dataset.npz", allow_pickle=False) as data:
        families, steps = data["families"], data["steps"]
        small = np.asarray(data["small_train_mask"], bool)
        large = np.asarray(data["large_train_mask"], bool)
        cal = np.asarray(data["calibration_mask"], bool)
        assert small.shape == large.shape == cal.shape == steps.shape == families.shape
        assert not (small & cal).any() and not (large & cal).any()
        assert (large[small]).all() and (large | cal).all()
        small_f, large_f, cal_f = set(families[small]), set(families[large]), set(families[cal])
        assert len(small_f) == 24 and len(large_f) == 84 and len(cal_f) == 12
        assert not small_f & cal_f and not large_f & cal_f and not large_f & audit_families and not cal_f & audit_families
        assert large_f - small_f == new_training_families
        assert data["observations"].shape == (len(families), 49)
        assert data["targets"].shape == (len(families), 17, 6)
        assert np.isfinite(data["observations"]).all() and np.isfinite(data["targets"]).all()
        np.testing.assert_array_equal(data["new_training_mask"], large & ~small)
        np.testing.assert_array_equal(data["selection_mask"], cal & (steps == 0))
        assert set(frozen["small_train_families"]) == small_f
        assert set(frozen["large_train_families"]) == large_f
        assert set(frozen["calibration_families"]) == cal_f
        assert frozen["selection_states"] == data["state_ids"][data["selection_mask"]].tolist()
        for row, state_id in enumerate(data["state_ids"]):
            directory = output / "training_states" if data["new_training_mask"][row] else Path(manifest["source_directory"]) / "states"
            with np.load(directory / f"{state_id}.npz", allow_pickle=False) as cached:
                np.testing.assert_array_equal(data["observations"][row], cached["observation"])
                np.testing.assert_array_equal(data["targets"][row], relative_priority_targets(cached["risks"], cached["supported"]))
        x = observable_features(data["observations"])
        for record in frozen["models"]:
            if record["historical"]:
                continue
            arm, student = record["arm"], students[record["arm"]]
            train_mask = large if arm.startswith("large_") else small
            train_families = set(families[train_mask])
            assert set(record["training_family_ids"]) == train_families
            assert set(record["calibration_family_ids"]) == cal_f
            assert set(record["selection_family_ids"]) == cal_f
            assert record["training_rows"] == int(train_mask.sum())
            assert record["selection_rows"] == int((cal & (steps == 0)).sum())
            weights = family_weights(families[train_mask])
            mean = np.sum(x[train_mask] * weights[:, None], axis=0)
            std = np.sqrt(np.sum((x[train_mask] - mean) ** 2 * weights[:, None], axis=0))
            std = np.where(std < 1e-3, 1, std).astype(np.float32)
            y = np.asarray(data["targets"], np.float32)
            if student.objective == "relative_rank":
                y = baseline_relative_targets(y)
            target_mean = np.sum(y[train_mask].mean(axis=1) * weights[:, None], axis=0)
            variance = np.sum(((y[train_mask] - target_mean) ** 2).mean(axis=1) * weights[:, None], axis=0)
            scale = np.maximum(np.sqrt(variance), 1e-3).astype(np.float32)
            np.testing.assert_array_equal(student.input_mean, mean.astype(np.float32))
            np.testing.assert_array_equal(student.input_std, std)
            np.testing.assert_array_equal(student.target_scale, scale)
        for key, mask in (("small_global_plan_index", small), ("large_global_plan_index", large)):
            initial = mask & (steps == 0)
            assert int(initial.sum()) == len(set(families[mask])) * 2
            priorities = (data["targets"][initial] * family_weights(families[initial])[:, None, None]).sum(axis=0)
            assert int(frozen[key]) == choose_priority(priorities)
        return {"small_training_families": len(small_f), "large_training_families": len(large_f),
                "calibration_families": len(cal_f), "small_training_rows": int(small.sum()),
                "large_training_rows": int(large.sum()), "calibration_rows": int(cal.sum()),
                "calibration_initial_selection_rows": int((cal & (steps == 0)).sum())}


def integrity_audit(output, manifest, run_summary, frozen):
    """Recompute recommendations and physical execution accounting from caches."""
    for relative, digest in manifest["source_hashes"].items():
        assert sha256(ROOT / relative) == sha256(output / "source_snapshot" / relative) == digest, relative
    source = Path(manifest["source_directory"])
    if "source_manifest_sha256" in manifest:
        assert sha256(source / "development_manifest.json") == manifest["source_manifest_sha256"]
    assert sha256(source / "state_selection.csv") == manifest["source_state_selection_sha256"]
    historical_path = Path(manifest["historical_directory"])
    assert sha256(historical_path / "frozen_students.json") == manifest["historical_frozen_manifest_sha256"]
    for state_id, digest in manifest.get("source_state_sha256", {}).items():
        assert sha256(source / "states" / f"{state_id}.npz") == digest
    freeze_hash = sha256(output / "frozen_students.json")
    before = json.loads((output / "audit_started_after_freeze.json").read_text(encoding="utf-8"))
    assert before["frozen_students_sha256"] == run_summary["student_freeze_sha256_before_audit"] == freeze_hash
    assert frozen["protocol_sha256"] == sha256(output / "development_manifest.json")
    assert frozen["dataset_sha256"] == sha256(output / "training_dataset.npz")
    models = frozen["models"]
    assert len(models) == 25 and {r["arm"] for r in models} == set(STUDENT_ARMS)
    students = {}
    for record in models:
        arm = record["arm"]
        assert sha256(output / record["checkpoint"]) == record["sha256"] == before["supervised_checkpoints"][arm]
        cls = ObservablePlanStudent if record["historical"] else ObservableScoreStudent
        students[arm] = cls.load(output / record["checkpoint"])
        assert int(students[arm].training_seed) == int(arm.rsplit("_", 1)[1])
        if not record["historical"]:
            assert not students[arm].forecast_only
            assert students[arm].objective == ("relative_rank" if "_rank_" in arm else "mse")
        else:
            assert sha256(Path(record["original_checkpoint"])) == record["sha256"]
    for record in manifest["frozen_ppo_models"]:
        assert sha256(Path(record["path"])) == record["checkpoint_sha256"]
    audit_scenarios = {s["scenario_id"]: (i, s) for i, s in enumerate(manifest["audit_scenario_records"])}
    training_scenarios = {s["scenario_id"]: (i, s) for i, s in enumerate(manifest["training_scenario_records"])}
    assert len(audit_scenarios) == 36 and len(training_scenarios) == 60
    assert not set(audit_scenarios) & set(training_scenarios)
    registered = {}
    for entry in manifest["seed_registry"]:
        phase = entry["phase"]
        for key in ("population_seed", "population_sensor_seed", "planning_seeds", "planning_sensor_seeds",
                    "evaluation_seeds", "evaluation_sensor_seeds", "uniform_candidate_seed", "uniform_ppo_seed"):
            if key not in entry:
                continue
            values = entry[key] if isinstance(entry[key], list) else [entry[key]]
            registered.setdefault((phase, key), set()).update(values)
    for i, seeds in enumerate(registered.values()):
        assert all(not seeds & other for other in list(registered.values())[i + 1:])
    assert len(manifest["seed_registry"]) == (60 + 36) * 2 * 5
    signatures = []
    for bank in (manifest["old_scenario_records"], manifest["excluded_previous_audit_records"],
                 manifest["training_scenario_records"], manifest["audit_scenario_records"]):
        current = {json.dumps({k: v for k, v in scenario.items() if k != "scenario_id"}, sort_keys=True) for scenario in bank}
        assert len(current) == len(bank) and all(not current & old for old in signatures)
        signatures.append(current)
    dataset = dataset_checks(output, manifest, frozen, students, set(audit_scenarios), set(training_scenarios))
    curves = read_csv(output / "training_curves.csv")
    for record in models:
        if record["historical"]:
            continue
        curve = [point for point in curves if point["arm"] == record["arm"]]
        assert [int(point["epoch"]) for point in curve] == list(range(1, record["actual_epochs"] + 1))
        best, epoch = np.inf, None
        for point in curve:
            if point["calibration_primary_regret"] not in ("", "None"):
                regret = float(point["calibration_primary_regret"])
                if regret < best - 1e-12:
                    best, epoch = regret, int(point["epoch"])
        assert epoch == record["selected_epoch"] == students[record["arm"]].selected_epoch
        assert best == record["best_calibration_primary_regret"]
    arms = tuple(manifest.get("arms", [r["arm"] for r in models] + list(FIXED_ARMS + CONTROL_ARMS)))
    assert len(arms) == 38 and set(arms) == set(ARMS)
    arm_index = {arm: i for i, arm in enumerate(arms)}
    candidates = manifest["candidate_names"]
    assert len(candidates) == 17
    records = read_csv(output / "state_selection.csv")
    episodes = read_csv(output / "evaluation_episode_results.csv")
    episode_lookup = {(row["state_id"], row["arm"], int(row["future_seed"])): row for row in episodes}
    assert len(episodes) == len(episode_lookup) == len(records) * 38 * 3
    assert {r["source_id"] for r in records} == set(audit_scenarios)
    training_records = read_csv(output / "training_state_selection.csv")
    assert {r["source_id"] for r in training_records} == set(training_scenarios)
    planning_seeds_all, eval_seeds_all, training_seed_all = set(), set(), set()
    audit_plans = eval_count = trace_steps = neutral_steps = blocked_steps = duplicates = training_plans = 0
    for row in training_records:
        scenario_index, scenario = training_scenarios[row["scenario_id"]]
        rep, step = int(row["repetition"]), int(row["step"])
        state_index = 10 * scenario_index + 5 * rep + tuple(manifest["collection_steps"]).index(step)
        expected_seeds = tuple(170_000_000 + 100 * state_index + f for f in range(3))
        expected_population = 150_000_000 + 2 * scenario_index + rep
        with np.load(output / "training_states" / f"{row['state_id']}.npz", allow_pickle=False) as cached:
            assert sha256(output / "training_states" / f"{row['state_id']}.npz") == frozen["new_training_state_sha256"][row["state_id"]]
            assert "evaluation_outcomes" not in cached.files
            assert cached["observation"].shape == (49,) and "command_history" in cached.files
            assert len(cached["command_history"]) == step and int(cached["population_seed"]) == expected_population
            np.testing.assert_array_equal(cached["planning_seeds"], expected_seeds)
            assert cached["risks"].shape == (2, 17, 3, len(OUTCOME_FIELDS))
            support = cached["supported"]
            assert support[scenario["source_branch"]]
            assert np.isfinite(cached["risks"][support]).all() and np.isnan(cached["risks"][~support]).all()
            assert len(cached["planning_seeds"]) == 3
            training_seed_all.update(int(s) for s in cached["planning_seeds"])
            assert int(cached["planning_rollouts"]) == int(support.sum()) * 17 * 3
            training_plans += int(cached["planning_rollouts"])
    for row in records:
        scenario_index, scenario = audit_scenarios[row["scenario_id"]]
        rep, step = int(row["repetition"]), int(row["step"])
        state_index = 10 * scenario_index + 5 * rep + tuple(manifest["collection_steps"]).index(step)
        expected_planning = tuple(210_000_000 + 100 * state_index + f for f in range(3))
        expected_evaluation = tuple(230_000_000 + 100 * state_index + f for f in range(3))
        expected_population = 190_000_000 + 2 * scenario_index + rep
        with np.load(output / "audit_states" / f"{row['state_id']}.npz", allow_pickle=False) as cached:
            observation, support, chosen = cached["observation"], cached["supported"], cached["chosen_indices"]
            assert observation.shape == (49,) and np.isfinite(observation).all()
            assert len(cached["command_history"]) == step and int(cached["population_seed"]) == expected_population
            np.testing.assert_array_equal(cached["planning_seeds"], expected_planning)
            np.testing.assert_array_equal(cached["evaluation_seeds"], expected_evaluation)
            assert cached["risks"].shape == (2, 17, 3, len(OUTCOME_FIELDS))
            assert np.isfinite(cached["risks"][support]).all() and np.isnan(cached["risks"][~support]).all()
            assert support.any() and int(support.sum()) == int(row["supported_world_count"])
            assert support[scenario["source_branch"]]
            assert chosen.shape == (38,) and ((chosen >= 0) & (chosen < 17)).all()
            assert len(cached["planning_seeds"]) == len(cached["evaluation_seeds"]) == 3
            planning_seeds_all.update(int(s) for s in cached["planning_seeds"])
            eval_seeds_all.update(int(s) for s in cached["evaluation_seeds"])
            for score_index, record in enumerate(models):
                arm, student = record["arm"], students[record["arm"]]
                priorities = student.priorities(observation)
                np.testing.assert_array_equal(cached["student_priorities"][score_index], priorities)
                assert int(chosen[arm_index[arm]]) == student.choose(observation)
                if "_rank_" in arm:
                    np.testing.assert_array_equal(priorities[12], np.zeros(6))
            for index, arm in enumerate(FIXED_ARMS, 12):
                assert int(chosen[arm_index[arm]]) == index
            for arm, mode in (("teacher_minimax", "robust"), ("teacher_point", "point"), ("teacher_oracle", "oracle")):
                assert int(chosen[arm_index[arm]]) == select_reference_plan(cached["risks"], support,
                    scenario["forecast_source_branch"], mode, true_source=scenario["source_branch"] if mode == "oracle" else None)
            assert int(chosen[arm_index["small_global_plan"]]) == int(frozen["small_global_plan_index"])
            assert int(chosen[arm_index["large_global_plan"]]) == int(frozen["large_global_plan_index"])
            assert int(chosen[arm_index["live_graded"]]) == 0
            assert int(chosen[arm_index["uniform_ppo"]]) == int(np.random.default_rng(251_000_000 + state_index).integers(12, 17))
            assert int(chosen[arm_index["uniform_candidate"]]) == int(np.random.default_rng(250_000_000 + state_index).integers(17))
            for arm, index in arm_index.items():
                assert row[arm] == candidates[int(chosen[index])]
            traces, offsets, keys = cached["execution_trace"], cached["trace_offsets"], cached["trace_keys"]
            assert traces.ndim == 2 and traces.shape[1] == 6
            assert offsets[0] == 0 and offsets[-1] == len(traces) and np.all(np.diff(offsets) > 0)
            lookup = {tuple(k): i for i, k in enumerate(keys)}
            assert len(lookup) == len(keys) == len(set(chosen)) * 3 == int(cached["evaluation_rollouts"])
            outcomes = cached["evaluation_outcomes"]
            assert outcomes.shape == (38, 3, len(OUTCOME_FIELDS)) and np.isfinite(outcomes).all()
            np.testing.assert_allclose(outcomes[..., 0] + outcomes[..., 2] + outcomes[..., 3], 1, atol=1e-12)
            for a, plan in enumerate(chosen):
                for f in range(3):
                    k = lookup[(plan, f)]
                    trace, result = traces[offsets[k]:offsets[k + 1]], outcomes[a, f]
                    episode = episode_lookup[(row["state_id"], arms[a], int(cached["evaluation_seeds"][f]))]
                    assert episode["source_id"] == row["source_id"] and episode["selected_plan"] == candidates[int(plan)]
                    assert int(episode["step"]) == step and int(episode["population_seed"]) == expected_population
                    np.testing.assert_array_equal([float(episode[field]) for field in OUTCOME_FIELDS], result)
                    assert len(trace) == result[5] and trace[0, 1] == result[9] and trace[0, 2] == result[10]
                    np.testing.assert_allclose([trace[:, 4].mean(), (trace[:, 1] == 8).sum(), trace[:, 5].mean()], result[6:9], atol=1e-12)
                    neutral = trace[:, 2] == -1
                    assert (trace[neutral, 3] == 2).all() and (trace[neutral, 4] > 0).all()
                    for b in range(a):
                        if chosen[b] == plan:
                            np.testing.assert_array_equal(result, outcomes[b, f])
                            duplicates += 1
            assert int(cached["planning_rollouts"]) == int(support.sum()) * 17 * 3
            audit_plans += int(cached["planning_rollouts"])
            eval_count += int(cached["evaluation_rollouts"])
            trace_steps += len(traces)
            neutral_steps += int((traces[:, 2] == -1).sum())
            blocked_steps += int(traces[:, 5].sum())
    streams = (training_seed_all, planning_seeds_all, eval_seeds_all)
    streams += tuple({s + 9_000_001 for s in seeds} for seeds in streams)
    for i, seeds in enumerate(streams):
        assert all(not seeds & other for other in streams[i + 1:])
    all_new_future_streams = set.union(*streams)
    for directory, subdirectory in ((source, "states"), (historical_path, "audit_states")):
        old_seeds = set()
        for path in (directory / subdirectory).glob("*.npz"):
            with np.load(path, allow_pickle=False) as cached:
                for key in ("planning_seeds", "evaluation_seeds"):
                    if key in cached.files:
                        old_seeds.update(int(seed) for seed in cached[key])
        old_seeds |= {seed + 9_000_001 for seed in old_seeds}
        assert not all_new_future_streams & old_seeds
    assert training_plans == run_summary["training_planning_rollouts"]
    assert audit_plans == run_summary["audit_planning_rollouts"]
    assert eval_count == run_summary["unique_audit_evaluation_rollouts"]
    assert training_plans + audit_plans + eval_count == run_summary["total_unique_full_rollouts"]
    assert len(records) == run_summary["states"] and len(training_records) == run_summary["new_training_states"]
    assert len(records) * 38 * 3 == run_summary["execution_arm_records"]
    assert trace_steps == run_summary["actual_execution_trace_steps"] and neutral_steps == run_summary["neutral_release_trace_steps"]
    if "sign_hold_blocked_trace_steps" in run_summary:
        assert blocked_steps == run_summary["sign_hold_blocked_trace_steps"]
    result = {"all_checks_passed": True, "all_25_students_frozen_before_new_audit": True,
              "train_calibration_audit_families_disjoint": True, "student_recommendations_recomputed_from_observation_only": True,
              "global_plans_use_training_initial_states_only": True, "supervised_factorial_groups_use_all_five_paired_seeds": True,
              "dataset_targets_reconstructed_from_planning_caches": True, "normalization_uses_own_training_only": True,
              "checkpoint_selection_uses_common_primary_regret_and_earliest_ties": True,
              "episode_csv_matches_execution_caches": True, "new_future_streams_disjoint_from_previous_rounds": True,
              "registered_population_future_sensor_and_selector_streams_disjoint": True,
              "states_verified": len(records), "new_training_states_verified": len(training_records), **dataset,
              "training_planning_rollouts": training_plans, "audit_planning_rollouts": audit_plans,
              "unique_audit_evaluation_rollouts": eval_count, "actual_execution_steps_verified": trace_steps,
              "duplicate_plan_equalities_verified": duplicates, "neutral_release_steps_verified": neutral_steps,
              "domain_sensor_streams_disjoint": True}
    json_write(output / "integrity_audit.json", result)
    return result


def diagnostics_from_caches(output, manifest, frozen):
    """Plan regret is a label diagnostic; never substitute it for execution."""
    arms = tuple(manifest.get("arms", ARMS))
    metadata = read_csv(output / "state_selection.csv")
    rows = []
    for row in metadata:
        with np.load(output / "audit_states" / f"{row['state_id']}.npz", allow_pickle=False) as cached:
            targets = relative_priority_targets(cached["risks"], cached["supported"])
            teacher = choose_priority(targets)
            for model in frozen["models"]:
                arm = model["arm"]
                p = int(cached["chosen_indices"][arms.index(arm)])
                rows.append({"state_id": row["state_id"], "source_id": row["source_id"], "step": int(row["step"]),
                    "arm": arm, "exact_teacher_agreement": int(p == teacher), "teacher_incap_regret_pp": float(targets[p, 0] * 100)})
    summaries = []
    for subset in ("initial", "all_states"):
        for arm in STUDENT_ARMS:
            selected = [r for r in rows if r["arm"] == arm and (subset == "all_states" or r["step"] == 0)]
            summaries.append({"subset": subset, "arm": arm, "states": len(selected),
                "family_equal_teacher_incap_regret_pp": float(np.mean(list(family_means(selected, "teacher_incap_regret_pp").values()))),
                "family_equal_teacher_agreement": float(np.mean(list(family_means(selected, "exact_teacher_agreement").values())))})
    write_csv(summaries, output / "distillation_diagnostic_summary.csv")
    return summaries


def chinese_names():
    groups = {"small_mse": "24家庭MSE", "small_rank": "24家庭相对排序", "large_mse": "84家庭MSE",
              "large_rank": "84家庭相对排序", "historical_mse": "上一版历史MSE"}
    names = {f"five_{group}_expectation": label + "（五初始化期望）" for group, label in groups.items()}
    names.update({f"{group}_{seed}": f"{label} {seed}" for group, label in groups.items() for seed in SEEDS})
    names.update({arm: "固定PPO " + arm.rsplit("_", 1)[1] for arm in FIXED_ARMS})
    names.update({"five_fixed_ppo_expectation": "五固定PPO期望", "teacher_minimax": "离线最坏世界教师",
                  "teacher_point": "离线单火源教师", "teacher_oracle": "真实火源有限库oracle",
                  "small_global_plan": "24家庭全局固定计划", "large_global_plan": "84家庭全局固定计划",
                  "uniform_ppo": "随机五PPO", "uniform_candidate": "随机17候选", "live_graded": "实时分级规则"})
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline_relative_plan_distillation_development_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/扩展训练与相对PPO排序开发实验报告.md")
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--report-date", default="2026-10-02")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.output / "development_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((args.output / "run_summary.json").read_text(encoding="utf-8"))
    run = {**run, "training_planning_rollouts": run["training_collection"]["planning_rollouts"],
           "audit_planning_rollouts": run["planning_rollouts"], "unique_audit_evaluation_rollouts": run["unique_evaluation_rollouts"],
           "new_training_states": run["training_collection"]["states"]}
    frozen = json.loads((args.output / "frozen_students.json").read_text(encoding="utf-8"))
    audit = integrity_audit(args.output, manifest, run, frozen)
    summaries, effects, interactions = summarize(typed_rows(args.output / "evaluation_episode_results.csv"), args.output)
    diagnostics = diagnostics_from_caches(args.output, manifest, frozen)
    initial = {r["arm"]: r for r in summaries if r["subset"] == "initial"}
    def effect(control, subset="initial", arm=FOCUS):
        return next(r for r in effects if r["subset"] == subset and r["arm"] == arm and r["control"] == control)
    primary = effect("pure_ppo_601")
    global_effect = effect("large_global_plan")
    teacher_effect = effect("pure_ppo_601", arm="teacher_minimax")
    if primary["exploratory_ci_high_pp"] < 0 and global_effect["exploratory_ci_high_pp"] < 0:
        decision = "大训练集相对排序组在本轮开发初始回合平均优于强固定PPO601和训练全局计划，值得继续软建议开发；这是探索性结果，不能作为最终资格证据。"
        next_step = "冻结本版本，预先固定与同信息同动作同训练预算的PPO软建议对照及另一个新开发审计；不要将学生评分当作安全概率。"
    else:
        decision = "大训练集相对排序组尚未同时稳定超过强固定PPO601和训练全局计划；这轮不能确认它保留了离线教师的主要收益，不推进为硬护栏或PPO强制风险监督。"
        next_step = "先审计离线教师相对强固定601的逐家庭改善空间和预测错误/长期封闭误选，再检查学生观测标签歧义；保留本四组和失败版本，在另一个预先固定开发方案中验证诊断，暂不追加正式PPO训练。"
    selections = read_csv(args.output / "state_selection.csv")
    counts = {arm: dict(Counter(r[arm] for r in selections if int(r["step"]) == 0)) for arm in STUDENT_ARMS + ("teacher_minimax",)}
    summary = {**run, "decision": decision, "tests_passed": args.tests_passed,
               "initial_summary": list(initial.values()), "initial_paired_effects": [r for r in effects if r["subset"] == "initial"],
               "initial_factorial_interaction": next(r for r in interactions if r["subset"] == "initial"),
               "teacher_vs_strong_fixed_initial": teacher_effect,
               "initial_selection_counts": counts, "integrity_audit": audit, "next_step": next_step,
               "all_intervals_exploratory": True, "bootstrap_unit": "scenario_family", "bootstrap_resamples": 2000,
               "bootstrap_seed": 20261703, "multiplicity_adjusted": False, "freeze_allowed": False}
    json_write(args.output / "summary.json", summary)
    names = chinese_names()
    plot_path = args.output / "initial_factorial_comparison.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for group, style in zip(GROUPS, ("o-", "s-", "^-", "D-", "x:")):
            axes[0].plot(SEEDS, [initial[f"{group}_{seed}"]["final_incapacitation"] * 100 for seed in SEEDS], style, label=group)
        axes[0].axhline(initial["pure_ppo_601"]["final_incapacitation"] * 100, color="#555555", linestyle="--", label="Fixed PPO 601")
        axes[0].axhline(initial["teacher_minimax"]["final_incapacitation"] * 100, color="#26734D", linestyle=":", label="Offline teacher")
        axes[0].set_xticks(SEEDS)
        axes[0].set_xlabel("Paired supervised initialization (all five retained)")
        axes[0].set_ylabel("Initial final incapacitation (%)")
        axes[0].legend(fontsize=7)
        plotted = [effect(c) for c in ("pure_ppo_601", "large_global_plan", "five_large_mse_expectation", "five_small_rank_expectation")]
        values = np.asarray([r["delta_pp"] for r in plotted])
        low = np.asarray([r["exploratory_ci_low_pp"] for r in plotted])
        high = np.asarray([r["exploratory_ci_high_pp"] for r in plotted])
        axes[1].errorbar(values, np.arange(len(values)), xerr=np.maximum(0, [values - low, high - values]), fmt="o", capsize=4)
        axes[1].axvline(0, color="#555555", linestyle="--")
        axes[1].set_yticks(np.arange(len(values)), ["vs PPO 601", "vs large fixed plan", "vs large MSE", "vs small ranking"])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Large ranking minus control (percentage points)")
        axes[1].set_title("36-family exploratory paired bootstrap 95% CI", fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except ImportError:
        plot_path = None
    lines = ["# 扩展训练与相对PPO排序开发实验报告", "", f"日期：{args.report_date}。按事先固定的2×2开发方案执行，所有区间均为探索性。", "",
             f"**结论：{decision}**", "", "## 1. 实验做了什么", "",
             "保留17候选：12个手写闭环计划和五个既有PPO。比较小数据/大数据与MSE/相对排序两个因素。四组都重新训练801–805五个配对初始化，共20模型；上一轮五个完整观测MSE检查点保持原样，作为历史参照。没有按新审计成绩删除模型或挑出最好初始化替代组平均。", "",
             f"小训练为旧24家庭、{audit['small_training_rows']}状态；大训练为同24家庭加60个新PREX家庭，共84家庭、{audit['large_training_rows']}状态。共同校准为旧12家庭、{audit['calibration_rows']}状态，检查点只看其中{audit['calibration_initial_selection_rows']}个初始人口状态的教师失能遗憾，同值保留最早轮次。校准家庭以前查看过，仍属于开发信息。", "",
             "四组同为49→64→64→102网络、可观测输入、建议通道清零；归一化仅来自各自训练家庭。MSE预测六项最低计划中心化代价；排序方案先聚合可能火源世界，再减固定PPO601的六分数，结合主要失能MSE、差距加权排序损失与其他五分数MSE。这里检验的是‘基准中心化＋排序损失’整体方案，不能单独归因于排序损失。", "",
             "相同AdamW设置、最多600轮、100轮无主要校准遗憾改善即停，并不等于相同计算总成本。大训练集需要更多教师模拟和更大全批训练。五监督初始化也不算新增PPO训练种子。", "",
             f"新审计36个PRAU家庭，每冲突6家庭、两人口，按0/8/16/24/40步可达起点采集{run['states']}状态，其中初始{run['initial_states']}、火源未识别{run['ambiguous_states']}、模型不匹配{run['model_mismatch_states']}。每候选3个规划未来，执行使用另外3个检验未来。所有25学生在新审计模拟前冻结。", "",
             f"实际完整回合模拟**{run['total_unique_full_rollouts']:,}次**：新增训练规划{run['training_planning_rollouts']:,}、新审计规划{run['audit_planning_rollouts']:,}、新审计唯一检验{run['unique_audit_evaluation_rollouts']:,}。38执行组逻辑记录{run['execution_arm_records']:,}条；同候选共享的成绩不是独立模拟。旧249状态规划标签复用。", "",
             "## 2. 初始回合结果", "",
             "主要子集为36家庭×2人口×3未来，每执行组216条记录；每家庭等权。五模型平均是随机选择一个初始化模型所获成绩的期望，未实现动作投票或在线集成。失能率越低越好。", "",
             "| 策略 | 平均失能率 | 平均FED |", "|---|---:|---:|"]
    for arm in tuple(VIRTUAL_GROUPS) + FIXED_ARMS + ("small_global_plan", "large_global_plan", "teacher_minimax", "teacher_point", "teacher_oracle", "uniform_ppo", "uniform_candidate", "live_graded"):
        row = initial[arm]
        lines.append(f"| {names[arm]} | {row['final_incapacitation']:.4%} | {row['final_fed']:.4f} |")
    candidates = manifest["candidate_names"]
    lines += ["", f"小、大训练全局固定计划分别为`{candidates[int(frozen['small_global_plan_index'])]}`、`{candidates[int(frozen['large_global_plan_index'])]}`，仅用各自训练家庭的初始规划分数选择，未看校准和新审计成绩。", "",
              "| 大数据相对排序平均减对照 | 差值（百分点） | 探索性95%区间 | 改善/变差/持平家庭 |", "|---|---:|---:|---:|"]
    for control in ("pure_ppo_601", "large_global_plan", "small_global_plan", "five_fixed_ppo_expectation", "five_historical_mse_expectation", "teacher_minimax", "uniform_ppo"):
        row = effect(control)
        lines.append(f"| {names[control]} | {row['delta_pp']:+.3f} | [{row['exploratory_ci_low_pp']:+.3f}, {row['exploratory_ci_high_pp']:+.3f}] | {row['improved_families']}/{row['worsened_families']}/{row['tied_families']} |")
    lines += ["", f"本轮离线最坏世界教师相对固定601仅{teacher_effect['delta_pp']:+.3f}个百分点，探索性区间[{teacher_effect['exploratory_ci_low_pp']:+.3f}, {teacher_effect['exploratory_ci_high_pp']:+.3f}]，{teacher_effect['improved_families']}家庭改善、{teacher_effect['worsened_families']}家庭变差、{teacher_effect['tied_families']}持平。教师在此前开发银行的明显优势没有在本36家庭上稳定复现，因此不能继续把结果只归因于学生没有学会。应先核查教师在哪些场景有可重复的改善空间。", "",
              "相同固定计划的检验结果完全相同；表中等效对照的区间微小差异来自各对比的有限次Bootstrap抽样，不能解释为策略差异。"]
    lines += ["", "## 3. 两个因素是否有帮助", "",
              "| 事先固定的比较 | 差值（百分点） | 探索性95%区间 | 改善/变差/持平家庭 |", "|---|---:|---:|---:|"]
    factor_pairs = ((FOCUS, "five_large_mse_expectation"), (FOCUS, "five_small_rank_expectation"),
                    ("five_large_mse_expectation", "five_small_mse_expectation"),
                    ("five_small_rank_expectation", "five_small_mse_expectation"))
    for arm, control in factor_pairs:
        row = effect(control, arm=arm)
        lines.append(f"| {names[arm]} − {names[control]} | {row['delta_pp']:+.3f} | [{row['exploratory_ci_low_pp']:+.3f}, {row['exploratory_ci_high_pp']:+.3f}] | {row['improved_families']}/{row['worsened_families']}/{row['tied_families']} |")
    inter = summary["initial_factorial_interaction"]
    lines += ["", "四项因素对照的探索性区间都包含零，当前没有建立扩充训练或相对排序方案的稳定收益。这不是它们无效的证据。大排序相对历史学生的改善还同时包含检查点规则变化，不能单独归因于排序损失。", "",
              f"探索性因素交互（大排序−大MSE）−（小排序−小MSE）为{inter['delta_pp']:+.3f}个百分点，区间[{inter['exploratory_ci_low_pp']:+.3f}, {inter['exploratory_ci_high_pp']:+.3f}]。它表示目标方案的效果是否随训练覆盖变化，不是另一个独立确认性检验。", "",
              "所有区间按36个独立家庭Bootstrap2,000次（种子20261703），人口、未来、初始化及中途状态先在家庭内平均；未作多重比较调整。不能把216条结果、五初始化或上百状态当作216/五/上百个独立场景。", "",
              "## 4. 全部初始化与拟合诊断", "",
              "| 学生 | 检查点/实际轮数 | 校准主要遗憾（百分点） | 新审计初始失能 | 教师规划遗憾（百分点） | 精确计划一致率 |", "|---|---:|---:|---:|---:|---:|"]
    for model in frozen["models"]:
        arm = model["arm"]
        diagnostic = next(r for r in diagnostics if r["subset"] == "initial" and r["arm"] == arm)
        regret = model.get("best_calibration_primary_regret")
        regret_text = "历史模型按MSE选择" if regret is None else f"{float(regret) * 100:.3f}"
        lines.append(f"| {names[arm]} | {model.get('selected_epoch', '?')}/{model.get('actual_epochs', '?')} | {regret_text} | {initial[arm]['final_incapacitation']:.4%} | {diagnostic['family_equal_teacher_incap_regret_pp']:.3f} | {diagnostic['family_equal_teacher_agreement']:.2%} |")
    lines += ["", "教师规划遗憾是学生所选计划在三个规划未来中的最坏世界失能分数相对教师最低值的差；它不是独立检验未来的实际失能改善。精确计划一致率也受FED、剩余人群、拥堵和平局次序影响。低训练损失或低校准遗憾不能替代闭环成绩。历史模型采用上一版MSE检查点规则，不能混入本轮四组因子分析。", "",
              "| 大数据相对排序初始化 | 初始人口的计划选择次数 |", "|---|---|"]
    for seed in SEEDS:
        arm = f"large_rank_{seed}"
        lines.append(f"| {seed} | " + "；".join(f"{plan}: {count}" for plan, count in sorted(counts[arm].items())) + " |")
    lines += ["", "## 5. 新开发场景分组与误选", "",
              "| 初始冲突子集 | 家庭数 | 大MSE平均 | 大排序平均 | 固定601 | 离线最坏世界教师 |", "|---|---:|---:|---:|---:|---:|"]
    for subset in sorted({r["subset"] for r in summaries if r["subset"].startswith("conflict_initial:")}):
        by_arm = {r["arm"]: r for r in summaries if r["subset"] == subset}
        lines.append(f"| {subset.split(':', 1)[1]} | {by_arm[FOCUS]['families']} | {by_arm['five_large_mse_expectation']['final_incapacitation']:.4%} | {by_arm[FOCUS]['final_incapacitation']:.4%} | {by_arm['pure_ppo_601']['final_incapacitation']:.4%} | {by_arm['teacher_minimax']['final_incapacitation']:.4%} |")
    lines += ["", "| 子集 | 家庭数 | 大排序平均失能 | 相对601差及探索性区间（百分点） |", "|---|---:|---:|---:|"]
    for subset in ("forecast_wrong_initial", "forecast_correct_initial", "ambiguous", "identified", "all_states"):
        row = next((r for r in summaries if r["subset"] == subset and r["arm"] == FOCUS), None)
        if row is None:
            continue
        paired = effect("pure_ppo_601", subset)
        lines.append(f"| {subset} | {row['families']} | {row['final_incapacitation']:.4%} | {paired['delta_pp']:+.3f} [{paired['exploratory_ci_low_pp']:+.3f}, {paired['exploratory_ci_high_pp']:+.3f}] |")
    worsening = sorted([r for r in read_csv(args.output / "family_paired_deltas.csv") if r["subset"] == "initial" and r["arm"] == FOCUS and r["control"] == "pure_ppo_601" and float(r["delta_pp"]) > 1e-10], key=lambda r: float(r["delta_pp"]), reverse=True)
    lines += ["", "相对固定601变差的家庭完整保存在family_paired_deltas.csv，以下列出最大差值；不能只报告改善家庭。", "",
              "| 家庭 | 初始差值（百分点） |", "|---|---:|"]
    for row in worsening[:12]:
        lines.append(f"| {row['source_id']} | {float(row['delta_pp']):+.3f} |")
    if not worsening:
        lines.append("| 本轮无正差值家庭 | 0 |")
    lines += ["", "中途起点含不同既往暴露，属于同家庭重复测量；‘全部状态’不能与初始主分析混为独立确认。每冲突仅6家庭，不据此宣称稳定跨场景泛化。", "",
              "## 6. 完整性、适用范围与下一步", "",
              f"自动审核通过：{audit['states_verified']}新审计状态、{audit['new_training_states_verified']}新增训练状态；重新从观测计算25学生推荐，核对训练/校准/新审计家庭隔离、训练初始状态全局选择、冻结前后哈希和{audit['actual_execution_steps_verified']:,}步实际执行。{audit['neutral_release_steps_verified']:,}步中性指示伴随正放行保留-1编码。共享计划的{audit['duplicate_plan_equalities_verified']:,}项结果逐项相同。", ""]
    if args.tests_passed is not None:
        lines.append(f"本次运行前/后的代码验证报告为{args.tests_passed}项测试通过；测试计数由实际pytest结果传入，不从旧报告沿用。")
    lines += ["", "这轮只测试相同生成分布的全新开发家庭，无传感噪声，火源相容判断仍假定其他参数完全准确。教师读取学生不可见的个体属性和暴露历史；三未来软分数不是安全概率或校准风险界。真实火源oracle只是有限17候选库内的比较，不是全局最优。", "",
              "五固定PPO仍34维输入、此前训练库和原权重；学生49维并持有候选库。此比较检验观测推荐是否值得继续，不能称为同信息同训练成本的RL提升证据。本轮没有LLM调用、新PPO训练或硬护栏资格测试，不能归因于LLM或宣称其贡献。", "",
              "PPO601是前轮开发阶段确定的强固定基准，没有依据本轮新审计重新挑选。学生在每个采样起点只推荐一次完整闭环候选，尚未检验在线每步重新选择的效果；候选内的手写计划/PPO仍会根据后续观测动作。", "",
              "护栏原门槛危险召回≥95%、安全误拒≤10%、所选危险≤5%、镜像一致性≥95%保持不变，尚未通过。正式至少十PPO训练种子和全新最终银行仍未启动。PRAU36家庭在本轮后成为已查看开发数据，不能改名为最终集。", "",
              next_step, "",
              "```powershell", "python scripts/run_relative_plan_distillation.py --output outputs/relative_plan_reproduction",
              "python scripts/analyze_relative_plan_distillation.py --output outputs/relative_plan_reproduction --report reports/相对排序复现报告.md",
              "python -m pytest -q", "```", "",
              "复现保存development_manifest.json、training_dataset.npz、逐状态训练/审计缓存、训练曲线、frozen_students.json、audit_started_after_freeze.json、run_summary.json、summary.json、subset_summary.csv、scenario_paired_effects.csv、family_paired_deltas.csv、factorial_interactions.csv、distillation_diagnostic_summary.csv、integrity_audit.json及源码快照。断点续跑必须匹配原清单、数据和源码哈希。"]
    if plot_path:
        lines += ["", f"![全部监督初始化与主要配对差异]({plot_path.resolve().as_posix()})"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot = args.output / "analysis_source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot)
    json_write(args.output / "analysis_manifest.json", {"role": "predeclared_factorial_family_reporting_and_postcollection_integrity_audit",
        "analysis_code_sha256": sha256(Path(__file__)), "summary_sha256": sha256(args.output / "summary.json"),
        "report_sha256": sha256(args.report), "report": str(args.report.resolve()), "decision": decision,
        "tests_passed": args.tests_passed, "freeze_allowed": False, "primary_paired_effect": primary})
    print(json.dumps({"decision": decision, "initial_incapacitation": {arm: initial[arm]["final_incapacitation"] for arm in
        (FOCUS, "five_large_mse_expectation", "five_small_rank_expectation", "five_small_mse_expectation", "pure_ppo_601", "teacher_minimax")},
        "primary_effect": primary, "integrity": audit, "report": str(args.report.resolve())}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
