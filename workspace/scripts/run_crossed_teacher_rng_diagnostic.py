"""Complete two missing population x future-stream cells per selected family.

Exactly 3,060 new planning continuations; diagonal cells reuse original caches.
No policy training, new evaluation streams, or LLM requests are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.config import EnvConfig
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.plan_distillation import observable_features
from fire_evac.source_world_planning import PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds, select_reference_plan
from fire_evac.ppo_reference_planning import FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, TRACE_FIELDS
from run_source_world_planning import make_env, vector

OUT = ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1"
AUDIT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
PROTOCOL = ROOT / "reports/人口与规划随机流交叉诊断方案_2026-10-03.md"
EPS = 1e-12
BASE = 12
TOLERANCES = (0, .05, .10, .20)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar) + "\n", encoding="utf-8")


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell_scores(risks, supported):
    means = np.asarray(risks, np.float64)[np.asarray(supported, bool)].mean(axis=2)
    assert np.isfinite(means).all()
    worst, average = means.max(axis=0), means.mean(axis=0)
    return np.stack((worst[:, 0], average[:, 0], worst[:, 1], average[:, 1], worst[:, 3], worst[:, 4]), axis=1)


def overlap(costs, tolerance_pp=0):
    costs = np.asarray(costs, np.float64)
    if costs.ndim != 2 or not np.isfinite(costs).all() or tolerance_pp < 0:
        raise ValueError("finite states x plans costs and nonnegative tolerance required")
    regret = (costs - costs.min(axis=1, keepdims=True)) * 100
    return int(np.all(regret <= tolerance_pp + EPS, axis=0).sum()), float(regret.max(axis=0).min())


def contrasts(gains):
    gains = np.asarray(gains, np.float64)
    if gains.ndim != 3 or gains.shape[:2] != (2, 2) or not np.isfinite(gains).all():
        raise ValueError("finite 2-population x 2-stream x plans gains required")
    return ((gains[1, 0] + gains[1, 1] - gains[0, 0] - gains[0, 1]) / 2,
            (gains[0, 1] + gains[1, 1] - gains[0, 0] - gains[1, 0]) / 2,
            gains[1, 1] - gains[1, 0] - gains[0, 1] + gains[0, 0])


def pooled_scores(cube, supported):
    # cube: population,stream,world,plan,future,metric. Pool within world FIRST.
    return np.asarray([cell_scores(np.concatenate((cube[p, 0], cube[p, 1]), axis=2), supported) for p in range(2)])


def validate_cell(z, group, population_index, stream_index):
    population = group["populations"][population_index]
    stream = group["populations"][stream_index]
    assert int(z["population_index"]) == population_index
    assert int(z["stream_index"]) == stream_index
    assert int(z["population_seed"]) == population["population_seed"]
    np.testing.assert_array_equal(z["planning_seeds"], stream["planning_seeds"])
    np.testing.assert_array_equal(z["supported"], group["supported"])
    assert z["risks"].shape == (2, 17, 3, len(OUTCOME_FIELDS))
    assert np.isfinite(z["risks"]).all()
    sums = z["risks"][..., [0, 2, 3]].sum(axis=-1)
    np.testing.assert_allclose(sums, 1, atol=EPS, rtol=0)
    assert int(z["planning_rollouts"]) == 102


def people_fingerprint(env):
    domain = env.unwrapped.domain_env
    content = {"people": [asdict(p) for p in domain.people], "zones": domain.zones}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_groups():
    files = {PROTOCOL, Path(__file__), ROOT / "scripts/run_source_world_planning.py",
             ROOT / "tests/test_crossed_teacher_rng_diagnostic.py", ROOT / "outputs/behavior_pipeline/behavior_model.json"}
    files.update((ROOT / "src/fire_evac").glob("*.py"))
    for name in ("completion_manifest.json", "diagnostic_manifest.json", "exact_observation_groups.csv", "state_primary_diagnostics.csv", "summary.json"):
        files.add(AUDIT / name)
    completion = read_json(AUDIT / "completion_manifest.json")
    for path, expected in completion["files"].items():
        assert digest(ROOT / path) == expected, path
    specs = {r["bank"]: r for r in read_json(AUDIT / "summary.json")["source_banks"]}
    states = {r["state_id"]: r for r in read_csv(AUDIT / "state_primary_diagnostics.csv")}
    selected = [r for r in read_csv(AUDIT / "exact_observation_groups.csv")
                if r["scope"] == "initial" and r["view"] == "full49" and r["shared_plans_at_0.00pp"] == "0"]
    assert len(selected) == 15
    groups, frozen_models, candidate_names = [], None, None
    for row in selected:
        bank = row["bank"]
        directory = Path(specs[bank]["directory"])
        manifest_path = directory / "development_manifest.json"
        old = read_json(manifest_path)
        files.add(manifest_path)
        records = old["training_scenario_records"] if bank == "PREX" else old["scenario_records" if bank in ("TSDEV", "PTDEV") else "audit_scenario_records"]
        ids = sorted(json.loads(row["state_ids"]))
        assert len(ids) == 2 and ids[0].split("_")[0] == ids[1].split("_")[0]
        family = states[ids[0]]["family"]
        scenario = next(r for r in records if r["scenario_id"] == family)
        if frozen_models is None:
            frozen_models, candidate_names = old["frozen_ppo_models"], old["candidate_names"]
        else:
            assert old["frozen_ppo_models"] == frozen_models and old["candidate_names"] == candidate_names
        subdir = "training_states" if bank == "PREX" else ("states" if bank in ("TSDEV", "PTDEV") else "audit_states")
        populations, features, primary = [], [], []
        for state_id in ids:
            path = directory / subdir / f"{state_id}.npz"
            files.add(path)
            with np.load(path, allow_pickle=False) as z:
                assert z["supported"].all()
                assert int(states[state_id]["step"]) == 0
                if "command_history" in z:
                    assert z["command_history"].size == 0
                seeds = z["planning_seeds"][:3].astype(np.int64).tolist()
                assert len(set(seeds)) == 3 and min(seeds) >= 0
                score = cell_scores(z["risks"][:, :, :3], z["supported"])
                primary.append(score[:, 0])
                features.append(observable_features(z["observation"]))
                populations.append({"state_id": state_id, "population_seed": int(states[state_id]["population_seed"]),
                                    "planning_seeds": seeds, "cache": str(path.resolve()), "cache_sha256": digest(path)})
        np.testing.assert_array_equal(features[0], features[1])
        assert overlap(primary)[0] == 0
        assert populations[0]["population_seed"] != populations[1]["population_seed"]
        assert set(populations[0]["planning_seeds"]).isdisjoint(populations[1]["planning_seeds"])
        groups.append({"bank": bank, "family": family, "role": states[ids[0]]["role"],
                       "conflict_type": scenario["conflict_type"], "scenario": scenario,
                       "supported": [True, True], "populations": populations,
                       "original_diagonal_minimum_worst_primary_regret_pp": overlap(primary)[1]})
    for model in frozen_models:
        path = Path(model["path"])
        assert digest(path) == model["checkpoint_sha256"]
        files.add(path)
    assert dict(Counter(g["bank"] for g in groups)) == {"TSDEV": 4, "PREX": 6, "PRAU": 1, "PDDEV": 2, "PTDEV": 2}
    return groups, frozen_models, candidate_names, files


def prepare(output):
    if (output / "development_manifest.json").exists():
        manifest = read_json(output / "development_manifest.json")
        for path, expected in manifest["source_input_sha256"].items():
            assert digest(path) == expected, path
        return manifest
    groups, models, names, files = selected_groups()
    schedule = [{"family_index": i, "population_index": p, "stream_index": 1 - p}
                for i in range(15) for p in range(2)]
    order = np.random.default_rng(20261901).permutation(len(schedule))
    schedule = [schedule[int(i)] for i in order]
    output.mkdir(parents=True, exist_ok=True)
    (output / "cells").mkdir(exist_ok=True)
    manifest = {"role": "selected_development_conflict_population_future_crossed_mechanism_diagnostic_not_efficacy",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_allowed": False,
                "groups": groups, "candidate_names": names, "frozen_ppo_models": models,
                "population_levels": 2, "planning_stream_levels": 2, "planning_futures_per_stream": 3,
                "new_planning_rollout_budget": 3060, "reused_diagonal_planning_records": 3060,
                "new_evaluation_rollout_budget": 0, "new_training_runs": 0, "new_llm_calls": 0,
                "run_order_seed": 20261901, "schedule": schedule, "env_config": asdict(EnvConfig()),
                "noise": "none", "advice_mode": "none", "sensor_seed_offset": 9000001,
                "tolerance_sensitivity_pp": list(TOLERANCES), "outcome_fields": list(OUTCOME_FIELDS), "trace_fields": list(TRACE_FIELDS),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "threads": 1},
                "source_input_sha256": {str(p.resolve()): digest(p) for p in sorted(files)}}
    write_json(output / "development_manifest.json", manifest)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(PROTOCOL, snapshot / PROTOCOL.name)
    return manifest


def collect(output, manifest):
    torch.set_num_threads(1)
    candidates = list(PLANS)
    for record in manifest["frozen_ppo_models"]:
        assert digest(record["path"]) == record["checkpoint_sha256"]
        model = PPO.load(record["path"], device="cpu")
        assert policy_parameter_sha256(model) == record["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(record["training_seed"], model))
    assert [candidate.name for candidate in candidates] == manifest["candidate_names"]
    scenarios = [DynamicConflictScenario(**g["scenario"]) for g in manifest["groups"]]
    worlds = [make_env(scenarios), make_env(scenarios)]
    started = time.monotonic()
    new_rollouts = 0
    try:
        for index, task in enumerate(manifest["schedule"]):
            group = manifest["groups"][task["family_index"]]
            p, s = task["population_index"], task["stream_index"]
            assert p != s
            destination = output / "cells" / f"{group['family']}_p{p}_s{s}.npz"
            if destination.exists():
                with np.load(destination, allow_pickle=False) as z:
                    validate_cell(z, group, p, s)
            else:
                population = group["populations"][p]
                stream = group["populations"][s]
                scenario = scenarios[task["family_index"]]
                with np.load(population["cache"], allow_pickle=False) as z:
                    original_observation = z["observation"].copy()
                risks = np.empty((2, 17, 3, len(OUTCOME_FIELDS)), np.float64)
                observations, fingerprints, traces = [], [], []
                for w, hypothesis in enumerate(source_world_scenarios(scenario)):
                    observation, _ = worlds[w].reset(seed=population["population_seed"], options={"scenario": hypothesis})
                    np.testing.assert_array_equal(observation, original_observation)
                    supported, _, mismatch = compatible_source_worlds(source_world_scenarios(scenario), observation, 0)
                    assert supported.all() and not mismatch
                    observations.append(observation)
                    fingerprints.append(people_fingerprint(worlds[w]))
                    for plan, candidate in enumerate(candidates):
                        for future, seed in enumerate(stream["planning_seeds"]):
                            trace = []
                            risks[w, plan, future] = vector(evaluate_continuation(worlds[w], observation, candidate, seed, trace))
                            traces.append(np.asarray(trace, np.float64))
                            new_rollouts += 1
                assert fingerprints[0] == fingerprints[1]
                assert new_rollouts <= manifest["new_planning_rollout_budget"]
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, risks=risks, observation=original_observation,
                    world_observations=np.asarray(observations), population_index=np.asarray(p), stream_index=np.asarray(s),
                    population_seed=np.asarray(population["population_seed"]), planning_seeds=np.asarray(stream["planning_seeds"]),
                    supported=np.ones(2, bool), planning_rollouts=np.asarray(102), initial_people_fingerprint=np.asarray(fingerprints[0]),
                    trace_offsets=np.cumsum([0] + [len(t) for t in traces]), execution_trace=np.concatenate(traces))
                temporary.replace(destination)
                with np.load(destination, allow_pickle=False) as z:
                    validate_cell(z, group, p, s)
            elapsed = time.monotonic() - started
            progress = {"phase": "crossed_planning", "cells_completed": index + 1, "cells_total": 30,
                        "new_planning_rollouts_this_invocation": new_rollouts, "elapsed_seconds_this_invocation": elapsed,
                        "new_evaluation_rollouts": 0, "new_training_runs": 0, "new_llm_calls": 0}
            write_json(output / "progress.json", progress)
            print(f"{index+1}/30 {group['family']} P{p} S{s}; new planning {new_rollouts}; {elapsed:.1f}s", flush=True)
    finally:
        for world in worlds:
            world.close()
    for record, candidate in zip(manifest["frozen_ppo_models"], candidates[12:]):
        assert policy_parameter_sha256(candidate.model) == record["parameter_sha256"]
        assert digest(record["path"]) == record["checkpoint_sha256"]
    return new_rollouts, time.monotonic() - started


def analyze(output, manifest, new_rollouts, elapsed):
    cell_rows, pair_rows, family_rows, contrast_rows = [], [], [], []
    fingerprints, trace_steps, neutral_steps, hold_steps = {}, 0, 0, 0
    for group in manifest["groups"]:
        cube = np.empty((2, 2, 2, 17, 3, len(OUTCOME_FIELDS)))
        scores = np.empty((2, 2, 17, 6))
        choices = np.empty((2, 2), int)
        for p in range(2):
            for s in range(2):
                original = p == s
                path = Path(group["populations"][p]["cache"]) if original else output / "cells" / f"{group['family']}_p{p}_s{s}.npz"
                with np.load(path, allow_pickle=False) as z:
                    cube[p, s] = z["risks"][:, :, :3]
                    if not original:
                        validate_cell(z, group, p, s)
                        fingerprints[(group["family"], p)] = str(z["initial_people_fingerprint"])
                        trace = z["execution_trace"]
                        offsets = z["trace_offsets"]
                        assert len(offsets) == 103 and offsets[-1] == len(trace)
                        assert np.isin(trace[:, 1], range(9)).all()
                        assert np.isin(trace[:, 2], [-1] + list(range(9))).all()
                        for key, (w, plan, f) in enumerate(np.ndindex(2, 17, 3)):
                            t = trace[offsets[key]:offsets[key + 1]]
                            o = cube[p, s, w, plan, f]
                            np.testing.assert_array_equal(t[:, 0], np.arange(len(t)))
                            assert len(t) == o[5]
                            assert abs(t[:, 4].mean() - o[6]) < EPS
                            assert int((t[:, 1] == 8).sum()) == o[7]
                            assert abs(t[:, 5].mean() - o[8]) < EPS
                            assert t[0, 1] == o[9] and t[0, 2] == o[10]
                        trace_steps += len(trace)
                        neutral_steps += int(((trace[:, 3] == 2) & (trace[:, 4] > 0)).sum())
                        hold_steps += int(trace[:, 5].sum())
                scores[p, s] = cell_scores(cube[p, s], group["supported"])
                chosen = select_reference_plan(cube[p, s], group["supported"], group["scenario"]["forecast_source_branch"], "robust")
                choices[p, s] = chosen
                cell_rows.append({"bank": group["bank"], "family": group["family"], "role": group["role"],
                    "population_index": p, "stream_index": s, "reused_original_cell": original,
                    "population_seed": group["populations"][p]["population_seed"],
                    "planning_seeds": json.dumps(group["populations"][s]["planning_seeds"]),
                    "selected_plan": manifest["candidate_names"][chosen], "primary_best_cost_percent": float(scores[p, s, :, 0].min() * 100),
                    "601_primary_cost_percent": float(scores[p, s, BASE, 0] * 100),
                    "teacher_primary_gain_vs601_pp": float((scores[p, s, BASE, 0] - scores[p, s, :, 0].min()) * 100)})
        assert fingerprints[(group["family"], 0)] != fingerprints[(group["family"], 1)]
        diagonal = scores[[0, 1], [0, 1], :, 0]
        np.testing.assert_allclose(overlap(diagonal)[1], group["original_diagonal_minimum_worst_primary_regret_pp"], atol=EPS, rtol=0)
        axis_counts = {}
        for axis in ("population", "planning_stream"):
            axis_records = []
            for fixed in range(2):
                pair = scores[:, fixed, :, 0] if axis == "population" else scores[fixed, :, :, 0]
                labels = choices[:, fixed] if axis == "population" else choices[fixed]
                row = {"bank": group["bank"], "family": group["family"], "role": group["role"],
                       "axis": axis, "fixed_other_level": fixed, "different_lexical_teacher": bool(labels[0] != labels[1]),
                       "minimum_worst_primary_regret_pp": overlap(pair)[1]}
                for tolerance in TOLERANCES:
                    row[f"shared_plans_at_{tolerance:.2f}pp"] = overlap(pair, tolerance)[0]
                axis_records.append(row)
                pair_rows.append(row)
            axis_counts[axis] = sum(r["shared_plans_at_0.00pp"] == 0 for r in axis_records)
        primary = scores[..., 0]
        gains = (primary[..., BASE, None] - primary) * 100
        population, stream, interaction = contrasts(gains)
        for plan, name in enumerate(manifest["candidate_names"]):
            contrast_rows.append({"bank": group["bank"], "family": group["family"], "candidate": name,
                "population_mean_difference_gain_pp": float(population[plan]),
                "planning_stream_mean_difference_gain_pp": float(stream[plan]),
                "interaction_difference_in_differences_gain_pp": float(interaction[plan])})
        pooled = pooled_scores(cube, group["supported"])
        pooled_primary = pooled[..., 0]
        population_conflict, stream_conflict = bool(axis_counts["population"]), bool(axis_counts["planning_stream"])
        category = ("both_axes_pairwise_conflict" if population_conflict and stream_conflict else
                    "population_axis_only_pairwise_conflict" if population_conflict else
                    "planning_stream_axis_only_pairwise_conflict" if stream_conflict else
                    "neither_axis_pairwise_conflict_diagonal_still_conflicts")
        row = {"bank": group["bank"], "family": group["family"], "role": group["role"], "conflict_type": group["conflict_type"],
               "population_comparisons_with_primary_conflict": axis_counts["population"],
               "planning_stream_comparisons_with_primary_conflict": axis_counts["planning_stream"],
               "pairwise_conflict_category": category, "original_diagonal_minimum_worst_regret_pp": overlap(diagonal)[1],
               "pooled_six_future_population_minimum_worst_regret_pp": overlap(pooled_primary)[1]}
        for tolerance in TOLERANCES:
            row[f"pooled_population_shared_plans_at_{tolerance:.2f}pp"] = overlap(pooled_primary, tolerance)[0]
        for p in range(2):
            chosen = min(range(17), key=lambda plan: tuple(pooled[p, plan]) + (plan,))
            row[f"pooled_population{p}_selected_plan"] = manifest["candidate_names"][chosen]
        for name, values in (("population", population), ("planning_stream", stream), ("interaction", interaction)):
            row[f"{name}_gain_contrast_rms_all17_pp"] = float(np.sqrt(np.mean(values ** 2)))
            row[f"{name}_gain_contrast_rms_five_ppo_pp"] = float(np.sqrt(np.mean(values[12:] ** 2)))
        family_rows.append(row)
    for name, rows in (("cell_diagnostics.csv", cell_rows), ("axis_pair_diagnostics.csv", pair_rows),
                       ("family_diagnostics.csv", family_rows), ("candidate_factorial_contrasts.csv", contrast_rows)):
        write_csv(output / name, rows)
    changed = [path for path, expected in manifest["source_input_sha256"].items() if digest(path) != expected]
    assert not changed
    caches = {p.name: digest(p) for p in sorted((output / "cells").glob("*.npz"))}
    assert len(caches) == 30
    counts = {}
    for axis in ("population", "planning_stream"):
        subset = [r for r in pair_rows if r["axis"] == axis]
        counts[axis] = {"comparisons": len(subset), "different_lexical_labels": sum(r["different_lexical_teacher"] for r in subset),
                        "primary_conflicting_comparisons": sum(r["shared_plans_at_0.00pp"] == 0 for r in subset),
                        "families_with_any_primary_conflict": sum(r[f"{axis}_comparisons_with_primary_conflict"] > 0 for r in family_rows),
                        "conflicts_beyond_0.20pp": sum(r["shared_plans_at_0.20pp"] == 0 for r in subset)}
    result = {"role": manifest["role"], "freeze_allowed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "families": 15, "cells": 60, "new_cells": 30, "reused_diagonal_cells": 30,
              "new_planning_rollouts_total": 3060, "reused_diagonal_planning_records": 3060,
              "new_planning_rollouts_this_invocation": new_rollouts, "elapsed_seconds_this_invocation": elapsed,
              "new_evaluation_rollouts": 0, "new_training_runs": 0, "new_llm_calls": 0,
              "axis_comparisons": counts, "pairwise_conflict_family_categories": dict(Counter(r["pairwise_conflict_category"] for r in family_rows)),
              "pooled_six_future_population_primary_conflicting_families": sum(r["pooled_population_shared_plans_at_0.00pp"] == 0 for r in family_rows),
              "pooled_six_future_population_conflicts_beyond_0.20pp": sum(r["pooled_population_shared_plans_at_0.20pp"] == 0 for r in family_rows),
              "new_planning_trace_steps": trace_steps, "neutral_positive_release_trace_steps": neutral_steps,
              "hold_blocked_trace_steps": hold_steps, "new_cell_sha256": caches,
              "limitations": ["Selected previously observed conflicting development families; no population conflict-rate estimate.",
                  "Two population initializations and two finite three-future streams, not universal causal attribution.",
                  "Population factor bundles person attributes and initial queue assignments.",
                  "Future-stream factor bundles stochastic behavior and sensor RNG; sensor noise amplitude is zero.",
                  "Candidate-dependent RNG consumption means shared seeds are not identical person-level latent outcomes.",
                  "Primary set overlaps need not be transitive; pairwise categories do not fully explain diagonal conflict.",
                  "Worst-world aggregation is nonlinear and may contribute to measured interactions.",
                  "Pooled six-future evidence is still finite and shares all original streams.",
                  "No independently seeded control efficacy evaluation, student learning, or LLM ablation."]}
    write_json(output / "summary.json", result)
    write_json(output / "integrity_audit.json", {"all_checks_passed": True, "source_files_hashed_unchanged": len(manifest["source_input_sha256"]),
        "new_cells": 30, "new_planning_rollouts": 3060, "population_conservation_checked": True,
        "initial_observations_matched_original_cache": True, "same_population_across_worlds_fingerprint_equal": True,
        "different_population_fingerprints_distinct": True, "actual_action_traces_checked": trace_steps,
        "frozen_ppo_files_parameters_unchanged": True, "original_diagonal_regret_reconstructed": True, "changed_original_inputs": []})
    write_json(output / "analysis_manifest.json", {"collector_sha256": digest(__file__),
        "development_manifest_sha256": digest(output / "development_manifest.json"), "new_cell_sha256": caches,
        "output_csv_sha256": {p.name: digest(p) for p in sorted(output.glob("*.csv"))},
        "summary_sha256": digest(output / "summary.json"), "integrity_sha256": digest(output / "integrity_audit.json")})
    write_json(output / "progress.json", {"phase": "complete", "new_planning_rollouts_total": 3060,
        "new_planning_rollouts_this_invocation": new_rollouts, "cells_completed": 30})
    print(json.dumps({"completed": True, "new_planning_rollouts": 3060, "axis_comparisons": counts,
                      "family_categories": result["pairwise_conflict_family_categories"],
                      "pooled_six_future_conflicting_families": result["pooled_six_future_population_primary_conflicting_families"]},
                     default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == AUDIT or AUDIT in output.parents or ROOT not in output.parents:
        raise ValueError("use a separate output directory within this workspace")
    if output.exists() and not (args.resume or args.analyze_only):
        raise FileExistsError("preserve output; use exact --resume or a new output directory")
    manifest = prepare(output)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "families": 15, "planning_rollout_budget": 3060, "new_rollouts": 0}))
        return
    if args.analyze_only:
        new, elapsed = 0, 0
    else:
        new, elapsed = collect(output, manifest)
    analyze(output, manifest, new, elapsed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
