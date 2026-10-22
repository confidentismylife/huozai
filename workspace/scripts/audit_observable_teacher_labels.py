"""Read-only diagnostic of observable plan labels and baseline overrides.

All inputs are saved development caches. No environment is created, no model is
trained, and no LLM is called. Exact label conflict is distinguished from primary
outcome regret, and from changes caused by finite planning-future samples.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.plan_distillation import choose_priority, family_weights, observable_features
from fire_evac.source_world_planning import select_reference_plan

OUT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
PROTOCOL = ROOT / "reports/可观测教师标签与覆盖收益审计方案.md"
BASE = 12
EPS = 1e-12
TOLERANCES_PP = (0, .05, .10, .20)
FOLDERS = {
    "TSDEV": "ppo_extended_world_planning_development_v1",
    "PREX": "baseline_relative_plan_distillation_development_v1",
    "PRAU": "baseline_relative_plan_distillation_development_v1",
    "PDDEV": "observable_plan_distillation_development_v1",
    "PTDEV": "teacher_factorial_pilot_development_v1",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert) + "\n", encoding="utf-8")


def write_csv(path, rows):
    assert rows, path
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def priorities(risks, supported, futures=None):
    # Match the frozen training target generator, which promotes cached float32
    # outcomes before averaging and subtracting the per-field minima.
    risks = np.asarray(risks, dtype=np.float64)
    selected = risks if futures is None else risks[:, :, list(futures), :]
    means = selected[np.asarray(supported, bool)].mean(axis=2)
    if not np.isfinite(means).all():
        raise ValueError("supported risks must be finite")
    worst, average = means.max(axis=0), means.mean(axis=0)
    return np.stack([worst[:, 0], average[:, 0], worst[:, 1], average[:, 1],
                     worst[:, 3], worst[:, 4]], axis=1)


def exact_key(observation, forecast_only=False):
    features = observable_features(observation, forecast_only)
    features[features == 0] = 0  # Canonicalize negative zero without rounding.
    return features.astype("<f4").tobytes()


def primary_overlap(costs, tolerance_pp=0):
    """Shared empirical near-optimum and unavoidable *sampled* regret.

    Empty intersection describes finite cached labels, not population Bayes risk.
    """
    costs = np.asarray(costs, float)
    if costs.ndim != 2 or not np.isfinite(costs).all() or tolerance_pp < 0:
        raise ValueError("finite [states,plans] costs and nonnegative tolerance required")
    regrets = (costs - costs.min(axis=1, keepdims=True)) * 100
    intersection = np.all(regrets <= tolerance_pp + EPS, axis=0)
    return int(intersection.sum()), float(regrets.max(axis=0).min())


def nearest_index(query, references, std, active, eligible):
    eligible = np.asarray(eligible, bool)
    active = np.asarray(active, bool)
    if not eligible.any() or not active.any():
        raise ValueError("eligible references and active dimensions required")
    difference = (references[:, active] - query[active]) / std[active]
    distance = np.sqrt(np.mean(difference ** 2, axis=1))
    distance[~eligible] = np.inf
    index = int(np.argmin(distance))
    return index, float(distance[index])


def group_mean(rows, field):
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(float(row[field]))
    return float(np.mean([np.mean(v) for v in by_family.values()]))


def load_sources():
    files = {Path(__file__), PROTOCOL}
    files.update((ROOT / "src/fire_evac").glob("*.py"))
    specs, state_records = [], []
    training_dir = ROOT / "outputs/baseline_relative_plan_distillation_development_v1"
    training_manifest = read_json(training_dir / "development_manifest.json")
    training_frozen = read_json(training_dir / "frozen_students.json")
    dataset_path = training_dir / "training_dataset.npz"
    files.update((training_dir / "development_manifest.json", training_dir / "frozen_students.json", dataset_path))
    for record in training_frozen["models"]:
        path = training_dir / record["checkpoint"]
        assert digest(path) == record["sha256"]
        files.add(path)
    normalization_record = next(r for r in training_frozen["models"] if r["arm"] == "large_mse_801")
    checkpoint = torch.load(training_dir / normalization_record["checkpoint"], map_location="cpu", weights_only=True)
    with np.load(dataset_path, allow_pickle=False) as z:
        dataset = {key: z[key].copy() for key in z.files}
    ids = {str(key): i for i, key in enumerate(dataset["state_ids"])}
    train_x = observable_features(dataset["observations"][dataset["large_train_mask"]])
    weights = family_weights(dataset["families"][dataset["large_train_mask"]])
    expected_mean = (train_x * weights[:, None]).sum(axis=0)
    raw_std = np.sqrt(((train_x - expected_mean) ** 2 * weights[:, None]).sum(axis=0))
    expected_std = np.where(raw_std < 1e-3, 1, raw_std).astype(np.float32)
    np.testing.assert_array_equal(checkpoint["input_mean"].numpy(), expected_mean)
    np.testing.assert_array_equal(checkpoint["input_std"].numpy(), expected_std)
    normalization = {"mean": expected_mean, "std": expected_std, "active": raw_std >= 1e-3,
                     "source_checkpoint": str(training_dir / normalization_record["checkpoint"])}

    for bank, folder in FOLDERS.items():
        directory = ROOT / "outputs" / folder
        manifest_path = directory / "development_manifest.json"
        manifest = read_json(manifest_path)
        files.update((manifest_path, directory / "summary.json"))
        if bank == "PREX":
            metadata = {r["state_id"]: r for r in read_csv(directory / "training_state_selection.csv")}
            files.add(directory / "training_state_selection.csv")
            cache_dir, scenarios = directory / "training_states", manifest["training_scenario_records"]
            teacher_arm, students = None, []
        else:
            metadata = {r["state_id"]: r for r in read_csv(directory / "state_selection.csv")}
            files.add(directory / "state_selection.csv")
            cache_dir = directory / ("states" if bank in ("TSDEV", "PTDEV") else "audit_states")
            scenarios = manifest["scenario_records" if bank in ("TSDEV", "PTDEV") else "audit_scenario_records"]
            teacher_arm = {"TSDEV": "extended_minimax", "PRAU": "teacher_minimax",
                           "PDDEV": "teacher_minimax", "PTDEV": "worst3"}[bank]
            students = []
            if bank in ("PDDEV", "PRAU"):
                frozen_path = directory / "frozen_students.json"
                frozen = read_json(frozen_path)
                files.add(frozen_path)
                students = frozen["models"]
                for model in students:
                    path = directory / model["checkpoint"]
                    assert digest(path) == model["sha256"]
                    files.add(path)
        scenario_lookup = {r["scenario_id"]: r for r in scenarios}
        for path in sorted(cache_dir.glob("*.npz")):
            files.add(path)
            meta = metadata[path.stem]
            family = meta["source_id"]
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files if k not in ("execution_trace", "trace_offsets", "trace_keys")}
            score = priorities(data["risks"], data["supported"], range(3))
            teacher = choose_priority(score)
            scenario = scenario_lookup[family]
            assert teacher == select_reference_plan(data["risks"][:, :, :3], data["supported"],
                                                      scenario["forecast_source_branch"], "robust")
            role = "new_training" if bank == "PREX" else "audit"
            if bank == "TSDEV":
                role = "old_training" if training_manifest["old_family_split"][family] == "train" else "calibration"
            if bank in ("TSDEV", "PREX"):
                index = ids[path.stem]
                np.testing.assert_array_equal(data["observation"], dataset["observations"][index])
                np.testing.assert_array_equal(score - score.min(axis=0), dataset["targets"][index])
                assert bool(dataset["large_train_mask"][index]) == (role != "calibration")
            if teacher_arm is not None:
                arm_index = manifest["arms"].index(teacher_arm)
                assert int(data["chosen_indices"][arm_index]) == teacher
                assert meta[teacher_arm] == manifest["candidate_names"][teacher]
                assert len(data["chosen_indices"]) == len(manifest["arms"])
                if "evaluation_outcomes" in data:
                    sums = data["evaluation_outcomes"][:, :, [0, 2, 3]].sum(axis=2)
                    np.testing.assert_allclose(sums, 1, rtol=0, atol=EPS)
            record = {"bank": bank, "role": role, "state_id": path.stem, "family": family,
                      "conflict_type": meta["conflict_type"], "step": int(meta["step"]),
                      "population_seed": int(meta["population_seed"]),
                      "true_source": scenario["source_branch"], "forecast_source": scenario["forecast_source_branch"],
                      "teacher": teacher, "score": score, "data": data, "arms": manifest["arms"],
                      "students": students, "plans": manifest["candidate_names"], "path": str(path)}
            state_records.append(record)
        specs.append({"bank": bank, "directory": str(directory), "states": len(metadata),
                      "families": len(scenario_lookup), "teacher": teacher_arm or "reconstructed_robust3"})
    assert len(state_records) == 1137
    return state_records, normalization, sorted(files), specs


def state_diagnostics(states):
    rows, sampling, world = [], [], []
    for state in states:
        d, score, teacher = state["data"], state["score"], state["teacher"]
        primary = score[:, 0]
        best = float(primary.min())
        common = {k: state[k] for k in ("bank", "role", "state_id", "family", "conflict_type", "step")}
        row = {**common, "population_seed": state["population_seed"],
               "supported_worlds": int(d["supported"].sum()), "teacher_plan": state["plans"][teacher],
               "primary_gain_vs_601_pp": (primary[BASE] - best) * 100,
               "primary_best_cost_percent": best * 100, "primary_601_cost_percent": primary[BASE] * 100,
               "teacher_differs_from_601": teacher != BASE,
               "teacher_differs_on_primary_tie": teacher != BASE and abs(primary[BASE] - best) <= EPS}
        for tolerance in TOLERANCES_PP:
            row[f"acceptable_plans_at_{tolerance:.2f}pp"] = int(((primary - best) * 100 <= tolerance + EPS).sum())
        rows.append(row)
        choices = []
        for omitted in range(3):
            folded = priorities(d["risks"], d["supported"], [f for f in range(3) if f != omitted])
            selected = choose_priority(folded)
            regret = (primary[selected] - best) * 100
            choices.append({**common, "method": "leave_one_of_first_three_out", "sample_index": omitted,
                            "selected_plan": state["plans"][selected], "changed_plan": selected != teacher,
                            "reference_primary_regret_pp": regret,
                            "reference": "full_first_three_planning_futures"})
        if state["bank"] == "PTDEV":
            full = priorities(d["risks"], d["supported"])
            full_best, full_choice = full[:, 0].min(), choose_priority(full)
            for block in range(4):
                selected = choose_priority(priorities(d["risks"], d["supported"], range(3 * block, 3 * block + 3)))
                choices.append({**common, "method": "four_disjoint_three_future_blocks", "sample_index": block,
                                "selected_plan": state["plans"][selected], "changed_plan": selected != full_choice,
                                "reference_primary_regret_pp": (full[selected, 0] - full_best) * 100,
                                "reference": "full_twelve_planning_futures"})
        sampling.extend(choices)
        if d["supported"].sum() == 2:
            means = d["risks"][:, :, :3, 0].mean(axis=2)
            shared, minimax_regret = primary_overlap(means)
            world.append({**common, "shared_primary_optimal_plans": shared,
                          "sampled_world_minimax_regret_pp": minimax_regret,
                          "teacher_true_world_regret_pp": (means[state["true_source"], teacher]
                                                            - means[state["true_source"]].min()) * 100,
                          "hidden_worlds_are_model_hypotheses": True})
    return rows, sampling, world


def collisions(states):
    rows, summaries = [], []
    for bank in FOLDERS:
        for scope in ("initial", "all_states_descriptive"):
            subset = [s for s in states if s["bank"] == bank and (scope != "initial" or s["step"] == 0)]
            for forecast_only in (False, True):
                groups = defaultdict(list)
                for state in subset:
                    groups[exact_key(state["data"]["observation"], forecast_only)].append(state)
                repeated = [group for group in groups.values() if len(group) > 1]
                disagreements = 0
                conflicts = {t: 0 for t in TOLERANCES_PP}
                for key, group in groups.items():
                    if len(group) < 2:
                        continue
                    labels = {s["teacher"] for s in group}
                    disagreements += len(labels) > 1
                    costs = np.asarray([s["score"][:, 0] for s in group])
                    row = {"bank": bank, "scope": scope, "view": "forecast_only" if forecast_only else "full49",
                           "group_hash": hashlib.sha256(key).hexdigest(), "states": len(group),
                           "families": len({s["family"] for s in group}),
                           "state_ids": json.dumps([s["state_id"] for s in group]),
                           "different_teacher_labels": len(labels),
                           "teacher_plans": json.dumps(sorted({s["plans"][s["teacher"]] for s in group})),
                           "distinct_population_seeds": len({s["population_seed"] for s in group}),
                           "distinct_planning_seed_sets": len({tuple(s["data"]["planning_seeds"]) for s in group}),
                           "population_and_future_sampling_not_separated": True}
                    for tolerance in TOLERANCES_PP:
                        shared, regret = primary_overlap(costs, tolerance)
                        row[f"shared_plans_at_{tolerance:.2f}pp"] = shared
                        conflicts[tolerance] += shared == 0
                    row["minimum_worst_empirical_primary_regret_pp"] = regret
                    rows.append(row)
                summaries.append({"bank": bank, "scope": scope, "view": "forecast_only" if forecast_only else "full49",
                                  "states": len(subset), "unique_observation_groups": len(groups),
                                  "repeated_observation_groups": len(repeated),
                                  "states_in_repeated_groups": sum(map(len, repeated)),
                                  "groups_with_different_teacher_labels": disagreements,
                                  **{f"groups_without_shared_plan_at_{t:.2f}pp": conflicts[t] for t in TOLERANCES_PP}})
    return rows, summaries


def nearest_diagnostics(states, normalization):
    train = [s for s in states if s["role"] in ("old_training", "new_training")]
    assert len(train) == 592 and len({s["family"] for s in train}) == 84
    query = [s for s in states if s["role"] == "audit"]
    assert set(s["family"] for s in query).isdisjoint(s["family"] for s in train)
    rows = []
    for forecast_only in (False, True):
        features = observable_features(np.asarray([s["data"]["observation"] for s in train]), forecast_only)
        active = normalization["active"].copy()
        if forecast_only:
            active[:19] = False
            active[28:] = False
        for state in query:
            x = observable_features(state["data"]["observation"], forecast_only)
            eligible = [(t["step"] == 0) == (state["step"] == 0) and t["family"] != state["family"] for t in train]
            idx, distance = nearest_index(x, features, normalization["std"], active, eligible)
            neighbor = train[idx]
            regret = (state["score"][neighbor["teacher"], 0] - state["score"][:, 0].min()) * 100
            rows.append({"bank": state["bank"], "state_id": state["state_id"], "family": state["family"],
                         "step": state["step"], "view": "forecast_only" if forecast_only else "full49",
                         "distance_rms_train_standardized": distance, "active_dimensions": int(active.sum()),
                         "neighbor_state_id": neighbor["state_id"], "neighbor_family": neighbor["family"],
                         "different_teacher_plan": state["teacher"] != neighbor["teacher"],
                         "query_primary_regret_of_neighbor_plan_pp": regret,
                         "not_a_learnability_or_independent_efficacy_test": True})
    summaries = []
    for bank in ("PRAU", "PDDEV", "PTDEV"):
        for scope in ("initial", "all_states_descriptive"):
            for view in ("full49", "forecast_only"):
                selected = [r for r in rows if r["bank"] == bank and r["view"] == view
                            and (scope != "initial" or r["step"] == 0)]
                summaries.append({"bank": bank, "scope": scope, "view": view, "states": len(selected),
                                  "median_distance": float(np.median([r["distance_rms_train_standardized"] for r in selected])),
                                  "different_plan_states": sum(r["different_teacher_plan"] for r in selected),
                                  "primary_regret_over_0.05pp_states": sum(r["query_primary_regret_of_neighbor_plan_pp"] > .05 + EPS for r in selected),
                                  "family_equal_primary_regret_pp": group_mean(selected, "query_primary_regret_of_neighbor_plan_pp")})
    return rows, summaries


def student_diagnostics(states):
    rows = []
    for state in states:
        if not state["students"]:
            continue
        d, arms, score = state["data"], state["arms"], state["score"]
        baseline = d["evaluation_outcomes"][arms.index("pure_ppo_601"), :, 0]
        teacher_gain = (score[BASE, 0] - score[:, 0].min()) * 100
        for j, record in enumerate(state["students"]):
            arm_index = arms.index(record["arm"])
            selected = int(d["chosen_indices"][arm_index])
            assert choose_priority(d["student_priorities"][j]) == selected
            if selected == BASE:
                np.testing.assert_array_equal(d["evaluation_outcomes"][arm_index],
                                               d["evaluation_outcomes"][arms.index("pure_ppo_601")])
            regret = (score[selected, 0] - score[:, 0].min()) * 100
            if teacher_gain > EPS:
                category = ("missed_primary_override" if selected == BASE else
                            "suboptimal_primary_override" if regret > EPS else "primary_optimal_override")
            else:
                category = ("kept_601_on_primary_tie" if selected == BASE else
                            "positive_regret_override_on_primary_tie" if regret > EPS else "zero_regret_override_on_primary_tie")
            rows.append({"bank": state["bank"], "state_id": state["state_id"], "family": state["family"],
                         "conflict_type": state["conflict_type"], "step": state["step"], "arm": record["arm"],
                         "group": record["group"] if "group" in record else ("forecast" if record["forecast_only"] else "full"),
                         "teacher_primary_gain_vs601_pp": teacher_gain, "selected_plan": state["plans"][selected],
                         "exact_teacher_agreement": selected == state["teacher"], "selected_601": selected == BASE,
                         "primary_regret_pp": regret, "category": category,
                         "evaluation_delta_vs601_pp": float(np.mean(d["evaluation_outcomes"][arm_index, :, 0] - baseline)) * 100})
    summaries, contributions = [], []
    groups = sorted({(r["bank"], r["group"]) for r in rows})
    for bank, group in groups:
        for scope in ("initial", "all_states_descriptive"):
            selected = [r for r in rows if r["bank"] == bank and r["group"] == group
                        and (scope != "initial" or r["step"] == 0)]
            count = len(selected)
            summary = {"bank": bank, "group": group, "scope": scope, "student_state_records": count,
                       "unique_states": len({r["state_id"] for r in selected}),
                       "families": len({r["family"] for r in selected}),
                       "model_count": len({r["arm"] for r in selected}),
                       "exact_teacher_agreement_rate": sum(r["exact_teacher_agreement"] for r in selected) / count,
                       "primary_optimal_rate": sum(r["primary_regret_pp"] <= EPS for r in selected) / count,
                       "primary_regret_within_0.05pp_rate": sum(r["primary_regret_pp"] <= .05 + EPS for r in selected) / count,
                       "family_equal_primary_regret_pp": group_mean(selected, "primary_regret_pp"),
                       "family_equal_eval_delta_vs601_pp": group_mean(selected, "evaluation_delta_vs601_pp")}
            summaries.append(summary)
            for category in sorted({r["category"] for r in selected}):
                category_rows = [r for r in selected if r["category"] == category]
                weighted = [{**r, "contribution": r["evaluation_delta_vs601_pp"] if r["category"] == category else 0} for r in selected]
                contributions.append({"bank": bank, "group": group, "scope": scope, "category": category,
                                      "records": len(category_rows), "record_fraction": len(category_rows) / count,
                                      "contribution_to_overall_eval_delta_pp": group_mean(weighted, "contribution")})
            total = sum(r["contribution_to_overall_eval_delta_pp"] for r in contributions
                        if (r["bank"], r["group"], r["scope"]) == (bank, group, scope))
            assert abs(total - summary["family_equal_eval_delta_vs601_pp"]) < EPS
    return rows, summaries, contributions


def main():
    states, normalization, inputs, specs = load_sources()
    hashes = {str(path): digest(path) for path in inputs}
    contract = {"role": "post_hoc_saved_development_observable_label_audit_not_final",
                "freeze_allowed": False, "protocol_sha256": digest(PROTOCOL), "code_sha256": digest(__file__),
                "source_input_sha256": hashes, "source_banks": specs,
                "primary_teacher_futures": 3, "tolerance_sensitivity_pp": list(TOLERANCES_PP),
                "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "diagnostic_manifest.json").exists():
        assert read_json(OUT / "diagnostic_manifest.json") == contract, "diagnostic contract changed; use a new version"
    else:
        write_json(OUT / "diagnostic_manifest.json", contract)
    state_rows, sampling_rows, world_rows = state_diagnostics(states)
    collision_rows, collision_summary = collisions(states)
    nearest_rows, nearest_summary = nearest_diagnostics(states, normalization)
    student_rows, student_summary, contributions = student_diagnostics(states)
    for name, rows in (("state_primary_diagnostics.csv", state_rows), ("sampling_sensitivity.csv", sampling_rows),
                       ("compatible_world_tradeoffs.csv", world_rows), ("exact_observation_groups.csv", collision_rows),
                       ("exact_group_summary.csv", collision_summary), ("training_nearest_neighbors.csv", nearest_rows),
                       ("nearest_neighbor_summary.csv", nearest_summary), ("student_state_diagnostics.csv", student_rows),
                       ("student_summary.csv", student_summary), ("student_effect_contributions.csv", contributions)):
        write_csv(OUT / name, rows)
    state_summary, sampling_summary, world_summary = [], [], []
    for bank in FOLDERS:
        for scope in ("initial", "all_states_descriptive"):
            selected = [r for r in state_rows if r["bank"] == bank and (scope != "initial" or r["step"] == 0)]
            state_summary.append({"bank": bank, "scope": scope, "states": len(selected),
                                  "families": len({r["family"] for r in selected}),
                                  "601_primary_optimal_states": sum(r["primary_gain_vs_601_pp"] <= EPS for r in selected),
                                  "teacher_has_primary_gain_states": sum(r["primary_gain_vs_601_pp"] > EPS for r in selected),
                                  "teacher_differs_on_primary_tie_states": sum(r["teacher_differs_on_primary_tie"] for r in selected),
                                  "median_primary_optimal_plan_count": float(np.median([r["acceptable_plans_at_0.00pp"] for r in selected]))})
            folded = [r for r in sampling_rows if r["bank"] == bank and (scope != "initial" or r["step"] == 0)]
            for method in sorted({r["method"] for r in folded}):
                by_state = defaultdict(list)
                for row in folded:
                    if row["method"] == method:
                        by_state[row["state_id"]].append(row)
                sampling_summary.append({"bank": bank, "scope": scope, "method": method, "states": len(by_state),
                                         "states_with_any_plan_change": sum(any(r["changed_plan"] for r in v) for v in by_state.values()),
                                         "states_with_any_positive_reference_regret": sum(any(r["reference_primary_regret_pp"] > EPS for r in v) for v in by_state.values()),
                                         "states_with_any_reference_regret_over_0.05pp": sum(any(r["reference_primary_regret_pp"] > .05 + EPS for r in v) for v in by_state.values())})
            wr = [r for r in world_rows if r["bank"] == bank and (scope != "initial" or r["step"] == 0)]
            if wr:
                world_summary.append({"bank": bank, "scope": scope, "ambiguous_states": len(wr),
                                      "disjoint_world_primary_optima_states": sum(r["shared_primary_optimal_plans"] == 0 for r in wr),
                                      "world_minimax_regret_over_0.05pp_states": sum(r["sampled_world_minimax_regret_pp"] > .05 + EPS for r in wr)})
    changed = [path for path, expected in hashes.items() if digest(path) != expected]
    assert not changed
    result = {"role": contract["role"], "freeze_allowed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "states": len(states), "families": len({s["family"] for s in states}),
              "initial_states": sum(s["step"] == 0 for s in states),
              "roles": dict(Counter(s["role"] for s in states)), "source_banks": specs,
              "normalization": {"training_states": 592, "training_families": 84,
                                "active_dimensions": int(normalization["active"].sum()),
                                "checkpoint": normalization["source_checkpoint"], "reconstructed_exactly": True},
              "state_primary_summary": state_summary, "exact_group_summary": collision_summary,
              "planning_sample_sensitivity": sampling_summary, "compatible_world_summary": world_summary,
              "training_neighbor_summary": nearest_summary, "student_summary": student_summary,
              "student_effect_contributions": contributions,
              "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0,
              "limitations": ["Viewed development data; no independent efficacy confirmation.",
                              "Different latent populations and planning streams are confounded in exact-feature groups.",
                              "Exact empirical label conflict is not a certified irreducible real-world risk bound.",
                              "Near-neighbor differences do not establish nonlearnability; no classifier is fit.",
                              "Primary ties can still differ in FED, mean-world risk and efficiency.",
                              "Compatible-world conflicts concern model hypotheses and finite future samples.",
                              "Resampled planning choices are correlated, not new evaluation replications.",
                              "Student execution comparisons use only existing independently-seeded evaluation caches.",
                              "Conditional category contributions are descriptive, not causal mediation effects."]}
    write_json(OUT / "summary.json", result)
    integrity = {"all_checks_passed": True, "input_files_hashed_and_unchanged": len(hashes),
                 "states_checked": len(states), "all_robust_teacher_choices_reconstructed": True,
                 "training_observations_targets_and_normalization_matched": True,
                 "student_choices_matched_cached_predictions": True,
                 "selected_601_has_identical_cached_outcome": True,
                 "independent_evaluation_cache_population_conservation": True,
                 "category_effect_contributions_sum_to_original_student_effect": True,
                 "changed_original_inputs": [], "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0}
    write_json(OUT / "integrity_audit.json", integrity)
    write_json(OUT / "analysis_manifest.json", {"code_sha256": digest(__file__),
               "diagnostic_manifest_sha256": digest(OUT / "diagnostic_manifest.json"),
               "summary_sha256": digest(OUT / "summary.json"), "integrity_sha256": digest(OUT / "integrity_audit.json"),
               "output_csv_sha256": {p.name: digest(p) for p in sorted(OUT.glob("*.csv"))}})
    print(json.dumps({"completed": True, "states": len(states), "families": result["families"],
                      "source_files_unchanged": len(hashes), "initial_state_summary": [r for r in state_summary if r["scope"] == "initial"],
                      "initial_exact_full49": [r for r in collision_summary if r["scope"] == "initial" and r["view"] == "full49"],
                      "initial_students": [r for r in student_summary if r["scope"] == "initial"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
