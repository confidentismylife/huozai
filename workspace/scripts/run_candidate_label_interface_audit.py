"""Cache-only five-PPO coverage audit and planning-only all-candidate export."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
from candidate_label_interface import (PPO_NAMES, DIAGNOSTIC_ROLE, candidate_indices,
    planning_labels, choose_candidate, family_weights, require_training_partition)
from candidate_library_scores import scores, choose
from finalize_set_aware_distillation import check_completion

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/candidate_library_coverage_audit_development_v1"
OUT = ROOT / "outputs/ppo_candidate_label_interface_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
PROTOCOL = ROOT / "reports/五PPO候选集合与全候选标签接口审计方案_2026-10-03.md"


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def cls(g): return "beneficial" if g > .001 else "harmful" if g < -1e-12 else "neutral_or_small"
def check(manifest):
    for path, expected in manifest["source_input_sha256"].items(): assert sha(path) == expected, path


def prepare():
    assert not OUT.exists(), "preserve completed and partial stages"
    assert check_completion(PRIOR.name, STATUS) == 133
    previous = read(PRIOR / "development_manifest.json")
    for path, expected in previous["source_input_sha256"].items(): assert sha(path) == expected
    indices = candidate_indices(previous["candidate_names"])
    assert tuple(indices) == tuple(range(12, 17))
    OUT.mkdir(); (OUT / "source_snapshot").mkdir()
    (OUT / "previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes())
    sources = dict(previous["source_input_sha256"])
    for path, expected in read(PRIOR / "completion_manifest.json")["files"].items():
        file = ROOT / path
        file = OUT / "previous_status_exact_bytes.json" if file.resolve() == STATUS else file
        sources[str(file.resolve())] = expected
    frozen = [PROTOCOL, Path(__file__), ROOT / "scripts/candidate_label_interface.py",
              ROOT / "tests/test_candidate_label_interface.py", PRIOR / "completion_manifest.json"]
    frozen += sorted(p for p in (ROOT / "artifacts/candidate_label_interface_test_repair_20261003").iterdir() if p.is_file())
    for file in frozen:
        sources[str(file.resolve())] = sha(file)
        if file.name != "completion_manifest.json": shutil.copyfile(file, OUT / "source_snapshot" / file.name)
    manifest = {"role": "viewed_cache_only_five_PPO_and_all_candidate_label_interface_audit", "created_at_utc": stamp(),
        "source_input_sha256": sources, "previous_status_sha256": sha(STATUS),
        "prior_completion_sha256": sha(PRIOR / "completion_manifest.json"),
        "prior_manifest_sha256": sha(PRIOR / "development_manifest.json"),
        "prior_cache": str(PRIOR), "candidate_names": previous["candidate_names"],
        "five_candidate_names": list(PPO_NAMES), "five_candidate_indices": indices.tolist(),
        "selected_scenario_records": previous["selected_scenario_records"], "seed_registry": previous["seed_registry"],
        "prior_margin": previous["prior_margin"], "outcome_fields": previous["outcome_fields"],
        "label_dataset_role": DIAGNOSTIC_ROLE, "expected_label_rows": 120, "feature_dimension": 54,
        "model_feature_fields": ["observable49_with_existing_28_to32_mask", "five_candidate_onehot"],
        "model_feature_exclusions": ["true_source", "actual_outcomes", "latent_population", "old_predictions", "old_proposal"],
        "label_sources": ["planning_risks", "supported"], "actual_results_are_diagnostic_only": True,
        "new_simulations": 0, "new_fits": 0, "new_ppo_training": 0, "new_llm_calls": 0,
        "qualification_thresholds_changed": False, "selection_threshold_fraction": .001,
        "no_new_train_calibration_final_split": True,
        "cached_followup_not_independent_reconfirmation": True}
    write(OUT / "development_manifest.json", manifest)
    print(json.dumps({"prepared": True, "source_inputs": len(sources), "new_simulations": 0}))


def run(manifest):
    assert not (OUT / "analysis_manifest.json").exists(), "preserve prior analysis"
    check(manifest)
    names = manifest["candidate_names"]; indices = np.array(manifest["five_candidate_indices"])
    plans, data, label_rows, blocks = {}, {}, [], defaultdict(list)
    records = {r["scenario_id"]: r for r in manifest["selected_scenario_records"]}
    # This phase reads planning files only; no actual files are opened until the bundle is frozen.
    for entry in manifest["seed_registry"]:
        state = entry["state_id"]; p = load(PRIOR / "states" / (state + "_planning.npz"))
        labels = planning_labels(p, names); R, M = scores(p["planning_risks"], p["supported"])
        np.testing.assert_allclose(labels["target_R"], R[indices], atol=1e-12)
        np.testing.assert_allclose(labels["target_M"], M[indices], atol=1e-12)
        data[state] = p; plans[state] = {"R": R, "M": M, "labels": labels}
        blocks["features"].append(labels["features"]); blocks["target_R"].append(labels["target_R"])
        blocks["target_M"].append(labels["target_M"])
        blocks["paired_future_gains"].append(labels["paired_future_gains"].transpose(1, 0, 2))
        blocks["world_mean_losses"].append(labels["world_mean_losses"].T)
        blocks["supported"].append(np.tile(labels["supported"], (5, 1)))
        blocks["planning_seeds"].append(np.tile(p["planning_seeds"], (5, 1)))
        for local, index in enumerate(indices):
            old = int(p["old_proposal"])
            label_rows.append({"state_id": state, "family": entry["family"], "plan_index": int(index), "plan": names[index],
                "old_plan_index": old, "old_nominee_only_row_eligible": old != 12 and index == old,
                "old_proposal_is601": old == 12, "target_R_pp": labels["target_R"][local] * 100,
                "target_M_pp": labels["target_M"][local] * 100, "R_class": cls(labels["target_R"][local]),
                "M_class": cls(labels["target_M"][local]), "dataset_role": DIAGNOSTIC_ROLE})
    arrays = {key: np.concatenate(parts) for key, parts in blocks.items()}
    arrays.update({"state_ids": np.array([r["state_id"] for r in label_rows]),
                   "families": np.array([r["family"] for r in label_rows]),
                   "plan_indices": np.array([r["plan_index"] for r in label_rows]),
                   "plan_names": np.array([r["plan"] for r in label_rows])})
    arrays["sample_weights"] = family_weights(arrays["families"])
    assert arrays["features"].shape == (120, 54)
    np.savez_compressed(OUT / "all_candidate_planning_labels.npz", **arrays)
    csv_write(OUT / "planning_label_rows.csv", label_rows)
    try:
        require_training_partition(DIAGNOSTIC_ROLE, arrays["families"], arrays["families"], [])
    except ValueError:
        fit_guard_rejected = True
    else:
        raise AssertionError("viewed diagnostic data must reject training")
    bundle = {"role": DIAGNOSTIC_ROLE, "frozen_at_utc": stamp(), "new_fit_allowed": False,
        "fit_guard_rejected_this_dataset": fit_guard_rejected, "features": 54, "rows": 120,
        "family_group_key": "families", "family_total_weight": 1 / 12,
        "label_units": "loss fractions, not percentage points", "paired_future_gain_shape": [120, 2, 6],
        "target_sources": ["planning_risks", "supported"], "no_actual_outcome_or_true_source_in_bundle": True,
        "planning_inputs_sha256": {e["state_id"]: sha(PRIOR / "states" / (e["state_id"] + "_planning.npz")) for e in manifest["seed_registry"]},
        "npz_sha256": sha(OUT / "all_candidate_planning_labels.npz"), "rows_csv_sha256": sha(OUT / "planning_label_rows.csv")}
    write(OUT / "label_bundle_manifest.json", bundle)
    write(OUT / "actual_diagnostic_read_started.json", {"at_utc": stamp(), "label_bundle_manifest_sha256": sha(OUT / "label_bundle_manifest.json"),
        "planning_label_npz_sha256": sha(OUT / "all_candidate_planning_labels.npz"), "cached_data_already_viewed_in_prior_stage": True})
    arms, coverage, actual_rows, changed = [], [], [], []
    for entry in manifest["seed_registry"]:
        state = entry["state_id"]; p = data[state]; a = load(PRIOR / "states" / (state + "_actual.npz"))
        outcomes = a["evaluation_outcomes"]; means = outcomes.mean(axis=1); actual = means[12, 0] - means[:, 0]
        common = {"state_id": state, "family": entry["family"], "conflict_type": records[entry["family"]]["conflict_type"]}
        for index in indices:
            actual_rows.append({**common, "plan_index": int(index), "plan": names[index], "actual_gain_pp": float(actual[index] * 100),
                "actual_class": cls(actual[index]), **{metric: float(means[index, j]) for j, metric in enumerate(manifest["outcome_fields"][:6])}})
        old = int(p["old_proposal"])
        coverage.append({**common, "old_plan_index": old, "old_proposal_is601": old == 12,
            "full17_actual_beneficial_plans": int((actual > .001).sum()),
            "five_actual_beneficial_plans": int((actual[indices] > .001).sum()),
            "full17_empirical_max_gain_pp": float(actual.max() * 100),
            "five_empirical_max_gain_pp": float(actual[indices].max() * 100),
            "R_planning_best_lost_pp": float((plans[state]["R"].max() - plans[state]["R"][indices].max()) * 100),
            "M_planning_best_lost_pp": float((plans[state]["M"].max() - plans[state]["M"][indices].max()) * 100)})
        for objective in ("R", "M"):
            for gate, margin in (("point", 0.), ("margin", manifest["prior_margin"])):
                gain = plans[state][objective]; full = choose(gain, margin); five = int(indices[choose_candidate(gain[indices], margin)])
                for library, index in (("full17", full), ("five", five)):
                    arms.append({**common, "arm": library + "_" + objective + "_" + gate,
                        "objective_gate": objective + "_" + gate, "library": library, "plan_index": index,
                        "plan": names[index], "override": index != 12, "actual_gain_pp": float(actual[index] * 100),
                        "actual_class": cls(actual[index]),
                        **{metric: float(means[index, j]) for j, metric in enumerate(manifest["outcome_fields"][:6])},
                        **{"delta_" + metric + "_vs601": float(means[index, j] - means[12, j]) for j, metric in enumerate(manifest["outcome_fields"][:6])}})
                if full != five:
                    changed.append({**common, "objective_gate": objective + "_" + gate, "full17_plan": names[full], "five_plan": names[five],
                        "full17_planning_gain_pp": float(gain[full] * 100), "five_planning_gain_pp": float(gain[five] * 100),
                        **{"five_minus_full17_" + metric: float(means[five, j] - means[full, j]) for j, metric in enumerate(manifest["outcome_fields"][:6])}})
    csv_write(OUT / "candidate_actual_diagnostic.csv", actual_rows)
    csv_write(OUT / "state_library_coverage.csv", coverage); csv_write(OUT / "arm_state_results.csv", arms)
    if changed: csv_write(OUT / "changed_library_choices.csv", changed)
    arm_summary, comparisons = [], []
    for name in sorted({r["arm"] for r in arms}):
        rows = [r for r in arms if r["arm"] == name]
        arm_summary.append({"arm": name, "states": len(rows), "overrides": sum(r["override"] for r in rows),
            "actual_classes": dict(Counter(r["actual_class"] for r in rows)),
            **{"mean_delta_" + metric + "_vs601": float(np.mean([r["delta_" + metric + "_vs601"] for r in rows])) for metric in manifest["outcome_fields"][:6]}})
    lookup = {(r["state_id"], r["arm"]): r for r in arms}
    for objective_gate in ("R_point", "M_point", "R_margin", "M_margin"):
        comparisons.append({"objective_gate": objective_gate, "changed_states": sum(r["objective_gate"] == objective_gate for r in changed),
            **{"mean_five_minus_full17_" + metric: float(np.mean([lookup[e["state_id"], "five_" + objective_gate][metric] - lookup[e["state_id"], "full17_" + objective_gate][metric] for e in manifest["seed_registry"]])) for metric in manifest["outcome_fields"][:6]}})
    # Identical feature collisions: arithmetic lower bounds for these finite labels, not fits.
    groups = defaultdict(list)
    for i, x in enumerate(arrays["features"]): groups[x.tobytes()].append(i)
    collision_rows = []; lower_bounds = {}
    for target in ("R", "M"):
        mae, mse, conflicting = 0., 0., 0
        y, weights = arrays["target_" + target], arrays["sample_weights"]
        for key, rowids in groups.items():
            ids = np.array(rowids); v, w = y[ids], weights[ids]
            order = np.argsort(v); median = v[order[np.searchsorted(np.cumsum(w[order]), w.sum() / 2)]]
            mean = np.average(v, weights=w); mae += float(np.sum(w * np.abs(v - median))); mse += float(np.sum(w * (v - mean) ** 2))
            if np.ptp(v) > 1e-12:
                conflicting += 1
                collision_rows.append({"target": target, "feature_sha256": hashlib.sha256(key).hexdigest(),
                    "rows": len(ids), "plan": str(arrays["plan_names"][ids[0]]),
                    "state_ids": "|".join(arrays["state_ids"][ids]), "families": "|".join(sorted(set(arrays["families"][ids]))),
                    "min_label_pp": float(v.min() * 100), "max_label_pp": float(v.max() * 100),
                    "range_pp": float(np.ptp(v) * 100)})
        lower_bounds[target] = {"conflicting_feature_groups": conflicting, "finite_label_minimum_MAE_pp": mae * 100,
                               "finite_label_minimum_RMSE_pp": np.sqrt(mse) * 100}
    if collision_rows: csv_write(OUT / "identical_input_label_conflicts.csv", collision_rows)
    kept = [r for r in label_rows if r["old_nominee_only_row_eligible"]]
    actual_lookup = {(r["state_id"], r["plan_index"]): r for r in actual_rows}
    counts = {"all_candidate_rows": len(label_rows), "old_nominee_only_rows_in_this_subset": len(kept),
        "baseline_rows_retained": sum(r["plan_index"] == 12 for r in label_rows),
        "old601_states_retained": sum(r["old_proposal_is601"] for r in coverage),
        "old601_rows_retained": sum(r["old_proposal_is601"] for r in label_rows),
        "all_R_beneficial_rows": sum(r["R_class"] == "beneficial" for r in label_rows),
        "all_M_beneficial_rows": sum(r["M_class"] == "beneficial" for r in label_rows),
        "old_only_R_beneficial_rows": sum(r["R_class"] == "beneficial" for r in kept),
        "old_only_M_beneficial_rows": sum(r["M_class"] == "beneficial" for r in kept),
        "actual_beneficial_candidate_rows": sum(r["actual_class"] == "beneficial" for r in actual_rows),
        "actual_beneficial_rows_kept_by_old_nominee_only": sum(actual_lookup[r["state_id"], r["plan_index"]]["actual_class"] == "beneficial" for r in kept)}
    summary = {"role": manifest["role"], "new_simulations": 0, "new_fits": 0, "new_ppo_training": 0, "new_llm_calls": 0,
        "families": 12, "population_states": 24, "candidate_count": 5,
        "label_interface_counts": counts, "arm_summary": arm_summary, "five_vs_full17": comparisons,
        "opportunity_preservation": {"full17_beneficial_states": sum(r["full17_actual_beneficial_plans"] > 0 for r in coverage),
            "five_beneficial_states": sum(r["five_actual_beneficial_plans"] > 0 for r in coverage),
            "states_with_lost_actual_beneficial_plan": sum(r["five_actual_beneficial_plans"] < r["full17_actual_beneficial_plans"] for r in coverage),
            "empirical_best_gain_equal_all_states": all(abs(r["full17_empirical_max_gain_pp"] - r["five_empirical_max_gain_pp"]) <= 1e-10 for r in coverage)},
        "input_label_audit": {"unique_feature_vectors": len(groups), "duplicate_feature_groups": sum(len(v) > 1 for v in groups.values()),
            "finite_empirical_fit_lower_bounds": lower_bounds, "not_deployment_error_lower_bound": True},
        "counterfactual_budget_arithmetic": {"full17_planning_rollouts_per_state": 204, "five_planning_rollouts_per_state": 60,
            "full17_actual_rollouts_per_state": 102, "five_actual_rollouts_per_state": 30,
            "designed_candidate_rollout_reduction_percent": 100 * (1 - 5 / 17), "not_measured_runtime_speedup": True},
        "viewed_dataset_fit_rejected": True, "new_actual_data_used_for_training": False,
        "freeze_allowed": False, "hard_shield_qualification_passed": False,
        "limits": ["Viewed prior caches and result-informed five-PPO follow-up; no independent replication or final test.",
            "Observed incap opportunity preservation does not imply all secondary outcomes preserved or global optimality.",
            "R/M labels retain simulator and latent-population privileges and six-future sampling error.",
            "Identical-input fit bounds concern these finite deterministic labels, not population or deployment error.",
            "Exports are diagnostic-only; this stage does not train, calibrate, deploy or relax gates.",
            "LLM contribution, new-family generalization and current-margin efficacy remain unestablished."]}
    write(OUT / "summary.json", summary); check(manifest)
    write(OUT / "analysis_manifest.json", {"runner_sha256": sha(__file__), "interface_sha256": sha(ROOT / "scripts/candidate_label_interface.py"),
        "development_manifest_sha256": sha(OUT / "development_manifest.json"),
        "files_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "analysis_manifest.json"}})
    print(json.dumps({"completed": True, "new_simulations": 0, "rows": counts, "opportunity": summary["opportunity_preservation"],
                      "input_label_audit": summary["input_label_audit"], "five_vs_full17": comparisons}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only: prepare()
    elif args.run: run(read(OUT / "development_manifest.json"))
    else: parser.error("choose --prepare-only or --run")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
