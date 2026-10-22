"""Independent cache arithmetic and label-lineage verifier; no fitting."""
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ppo_candidate_label_interface_development_v1"
PRIOR = ROOT / "outputs/candidate_library_coverage_audit_development_v1"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def close(a, b): np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
def cls(g): return "beneficial" if g > .001 else "harmful" if g < -1e-12 else "neutral_or_small"
def selection(gains, candidates, margin):
    values = {k: 0. if k == 12 else float(gains[k]) - margin for k in candidates}
    best = max(values.values()); tied = [k for k in candidates if best - values[k] <= 1e-12]
    k = min(tied, key=lambda j: (j != 12, j))
    return k if values[k] > .001 else 12


def main():
    assert not (OUT / "verification.json").exists()
    m, s, analysis, bundle = (read(OUT / p) for p in ("development_manifest.json", "summary.json", "analysis_manifest.json", "label_bundle_manifest.json"))
    for path, expected in m["source_input_sha256"].items(): assert sha(path) == expected, path
    assert sha(ROOT / "outputs/third_round_development_status.json") == m["previous_status_sha256"]
    assert sha(PRIOR / "completion_manifest.json") == m["prior_completion_sha256"]
    assert analysis["runner_sha256"] == sha(ROOT / "scripts/run_candidate_label_interface_audit.py")
    assert analysis["interface_sha256"] == sha(ROOT / "scripts/candidate_label_interface.py")
    assert analysis["development_manifest_sha256"] == sha(OUT / "development_manifest.json")
    for name, expected in analysis["files_sha256"].items(): assert sha(OUT / name) == expected
    assert all(s[k] == m[k] == 0 for k in ("new_simulations", "new_fits", "new_ppo_training", "new_llm_calls"))
    assert not m["qualification_thresholds_changed"] and not bundle["new_fit_allowed"] and bundle["fit_guard_rejected_this_dataset"]
    assert bundle["role"] == m["label_dataset_role"] == "viewed_development_diagnostic_not_training"
    assert bundle["npz_sha256"] == sha(OUT / "all_candidate_planning_labels.npz")
    assert bundle["rows_csv_sha256"] == sha(OUT / "planning_label_rows.csv")
    started = read(OUT / "actual_diagnostic_read_started.json")
    assert datetime.fromisoformat(bundle["frozen_at_utc"]) <= datetime.fromisoformat(started["at_utc"])
    assert started["label_bundle_manifest_sha256"] == sha(OUT / "label_bundle_manifest.json")
    assert started["planning_label_npz_sha256"] == bundle["npz_sha256"]
    arrays = load(OUT / "all_candidate_planning_labels.npz")
    assert set(arrays) == {"features", "target_R", "target_M", "paired_future_gains", "world_mean_losses", "supported", "planning_seeds", "state_ids", "families", "plan_indices", "plan_names", "sample_weights"}
    assert arrays["features"].shape == (120, 54) and arrays["paired_future_gains"].shape == (120, 2, 6)
    assert np.isfinite(arrays["features"]).all() and not arrays["features"][:, 28:33].any()
    names = m["candidate_names"]; indices = m["five_candidate_indices"]
    assert indices == list(range(12, 17)) and [names[k] for k in indices] == [f"pure_ppo_{i}" for i in range(601, 606)]
    labels = {(r["state_id"], int(r["plan_index"])): r for r in table(OUT / "planning_label_rows.csv")}
    actualcsv = {(r["state_id"], int(r["plan_index"])): r for r in table(OUT / "candidate_actual_diagnostic.csv")}
    arms = {(r["state_id"], r["arm"]): r for r in table(OUT / "arm_state_results.csv")}
    coverage = {r["state_id"]: r for r in table(OUT / "state_library_coverage.csv")}
    changed = {(r["state_id"], r["objective_gate"]): r for r in table(OUT / "changed_library_choices.csv")}
    counts = Counter(); expected_arm_records = []; expected_changed = []
    expected_opportunity = Counter(); expected_R = []; expected_M = []
    for i, entry in enumerate(m["seed_registry"]):
        state = entry["state_id"]; p = load(PRIOR / "states" / (state + "_planning.npz")); a = load(PRIOR / "states" / (state + "_actual.npz"))
        assert bundle["planning_inputs_sha256"][state] == sha(PRIOR / "states" / (state + "_planning.npz"))
        losses = p["planning_risks"][:, :, :, 0]; support = p["supported"]; means = losses[support].mean(axis=2)
        R = np.array([means[:, 12].max() - means[:, k].max() for k in range(17)])
        M = np.array([min(means[w, 12] - means[w, k] for w in range(len(means))) for k in range(17)])
        R[12] = M[12] = 0; expected_R.extend(R[indices]); expected_M.extend(M[indices])
        offset = slice(5 * i, 5 * i + 5)
        x = p["observation"].astype(float).copy(); x[28:33] = 0
        close(arrays["features"][offset, :49], np.tile(x, (5, 1))); close(arrays["features"][offset, 49:], np.eye(5))
        np.testing.assert_array_equal(arrays["state_ids"][offset], [state] * 5)
        np.testing.assert_array_equal(arrays["families"][offset], [entry["family"]] * 5)
        np.testing.assert_array_equal(arrays["plan_indices"][offset], indices)
        np.testing.assert_array_equal(arrays["plan_names"][offset], [names[k] for k in indices])
        np.testing.assert_array_equal(arrays["planning_seeds"][offset], np.tile(p["planning_seeds"], (5, 1)))
        np.testing.assert_array_equal(arrays["supported"][offset], np.tile(support, (5, 1)))
        for j, k in enumerate(indices):
            close(arrays["paired_future_gains"][5 * i + j], losses[:, 12, :] - losses[:, k, :])
            close(arrays["world_mean_losses"][5 * i + j], losses[:, k, :].mean(axis=1))
            row = labels[state, k]; old = int(p["old_proposal"]); eligible = old != 12 and k == old
            assert row["old_proposal_is601"] == str(old == 12) and row["old_nominee_only_row_eligible"] == str(eligible)
            close(float(row["target_R_pp"]), R[k] * 100); close(float(row["target_M_pp"]), M[k] * 100)
            assert row["R_class"] == cls(R[k]) and row["M_class"] == cls(M[k]) and row["dataset_role"] == bundle["role"]
            counts["all_candidate_rows"] += 1; counts["old_nominee_only_rows_in_this_subset"] += eligible
            counts["baseline_rows_retained"] += k == 12; counts["old601_rows_retained"] += old == 12
            counts["all_R_beneficial_rows"] += R[k] > .001; counts["all_M_beneficial_rows"] += M[k] > .001
            counts["old_only_R_beneficial_rows"] += eligible and R[k] > .001
            counts["old_only_M_beneficial_rows"] += eligible and M[k] > .001
        actual_means = a["evaluation_outcomes"].mean(axis=1); gain = actual_means[12, 0] - actual_means[:, 0]
        for k in indices:
            row = actualcsv[state, k]; close(float(row["actual_gain_pp"]), gain[k] * 100); assert row["actual_class"] == cls(gain[k])
            for j, metric in enumerate(m["outcome_fields"][:6]): close(float(row[metric]), actual_means[k, j])
            counts["actual_beneficial_candidate_rows"] += gain[k] > .001
            counts["actual_beneficial_rows_kept_by_old_nominee_only"] += int(p["old_proposal"]) != 12 and k == int(p["old_proposal"]) and gain[k] > .001
        counts["old601_states_retained"] += int(p["old_proposal"]) == 12
        row = coverage[state]
        assert int(row["full17_actual_beneficial_plans"]) == int((gain > .001).sum())
        assert int(row["five_actual_beneficial_plans"]) == int((gain[indices] > .001).sum())
        close(float(row["full17_empirical_max_gain_pp"]), gain.max() * 100); close(float(row["five_empirical_max_gain_pp"]), gain[indices].max() * 100)
        for objective, values in (("R", R), ("M", M)):
            close(float(row[objective + "_planning_best_lost_pp"]), (values.max() - values[indices].max()) * 100)
            for gate, margin in (("point", 0.), ("margin", m["prior_margin"])):
                key = objective + "_" + gate
                chosen = {"full17": selection(values, range(17), margin), "five": selection(values, indices, margin)}
                for library, k in chosen.items():
                    arm = library + "_" + key; ar = arms[state, arm]
                    assert int(ar["plan_index"]) == k and ar["override"] == str(k != 12) and ar["actual_class"] == cls(gain[k])
                    for j, metric in enumerate(m["outcome_fields"][:6]):
                        close(float(ar[metric]), actual_means[k, j]); close(float(ar["delta_" + metric + "_vs601"]), actual_means[k, j] - actual_means[12, j])
                    expected_arm_records.append(ar)
                if chosen["full17"] != chosen["five"]:
                    cr = changed[state, key]; expected_changed.append((state, key))
                    assert cr["full17_plan"] == names[chosen["full17"]] and cr["five_plan"] == names[chosen["five"]]
                    close(float(cr["full17_planning_gain_pp"]), values[chosen["full17"]] * 100)
                    close(float(cr["five_planning_gain_pp"]), values[chosen["five"]] * 100)
                    for j, metric in enumerate(m["outcome_fields"][:6]): close(float(cr["five_minus_full17_" + metric]), actual_means[chosen["five"], j] - actual_means[chosen["full17"], j])
        expected_opportunity["full17_beneficial_states"] += (gain > .001).any()
        expected_opportunity["five_beneficial_states"] += (gain[indices] > .001).any()
        expected_opportunity["states_with_lost_actual_beneficial_plan"] += (gain[indices] > .001).sum() < (gain > .001).sum()
    close(arrays["target_R"], expected_R); close(arrays["target_M"], expected_M)
    assert dict(counts) == s["label_interface_counts"] and set(expected_changed) == set(changed)
    assert len(labels) == len(actualcsv) == 120 and len(arms) == 192 and len(coverage) == 24
    for key, expected in expected_opportunity.items(): assert s["opportunity_preservation"][key] == expected
    assert s["opportunity_preservation"]["empirical_best_gain_equal_all_states"] == all(abs(float(r["five_empirical_max_gain_pp"]) - float(r["full17_empirical_max_gain_pp"])) <= 1e-10 for r in coverage.values())
    for family in set(arrays["families"]):
        idx = arrays["families"] == family
        assert idx.sum() == 10; close(arrays["sample_weights"][idx], np.full(10, 1 / 120))
    close(arrays["sample_weights"].sum(), 1)
    for row in s["arm_summary"]:
        part = [r for r in expected_arm_records if r["arm"] == row["arm"]]
        assert row["overrides"] == sum(r["override"] == "True" for r in part)
        assert row["actual_classes"] == dict(Counter(r["actual_class"] for r in part))
        for metric in m["outcome_fields"][:6]: close(row["mean_delta_" + metric + "_vs601"], np.mean([float(r["delta_" + metric + "_vs601"]) for r in part]))
    for row in s["five_vs_full17"]:
        key = row["objective_gate"]; assert row["changed_states"] == sum(k == key for _, k in expected_changed)
        for metric in m["outcome_fields"][:6]: close(row["mean_five_minus_full17_" + metric], np.mean([float(arms[e["state_id"], "five_" + key][metric]) - float(arms[e["state_id"], "full17_" + key][metric]) for e in m["seed_registry"]]))
    # Uniform row weights in this fixed balanced bank: median and mean bounds recomputed.
    groups = defaultdict(list)
    for i, x in enumerate(arrays["features"]): groups[x.tobytes()].append(i)
    assert len(groups) == s["input_label_audit"]["unique_feature_vectors"] == 60
    assert all(len(ids) == 2 for ids in groups.values())
    conflicts = {(r["target"], r["feature_sha256"]): r for r in table(OUT / "identical_input_label_conflicts.csv")}; nconf = 0
    for target in ("R", "M"):
        total_abs, total_square, n = 0., 0., 0
        for key, ids in groups.items():
            y = arrays["target_" + target][ids]; total_abs += np.abs(y - np.median(y)).sum() / 120
            total_square += ((y - y.mean()) ** 2).sum() / 120
            if y.max() - y.min() > 1e-12:
                n += 1; nconf += 1; row = conflicts[target, hashlib.sha256(key).hexdigest()]
                close(float(row["range_pp"]), np.ptp(y) * 100); close(float(row["min_label_pp"]), y.min() * 100); close(float(row["max_label_pp"]), y.max() * 100)
        expected = s["input_label_audit"]["finite_empirical_fit_lower_bounds"][target]
        assert expected["conflicting_feature_groups"] == n
        close(expected["finite_label_minimum_MAE_pp"], total_abs * 100); close(expected["finite_label_minimum_RMSE_pp"], np.sqrt(total_square) * 100)
    assert nconf == len(conflicts)
    snapshots = read(OUT / "snapshot_name_collision_note.json")
    for row in snapshots["records"]: assert sha(row["runtime_source"]) == sha(row["correct_runtime_snapshot"]) == row["runtime_sha256"]
    result = {"all_checks_passed": True, "source_inputs_unchanged": len(m["source_input_sha256"]),
        "planning_label_rows_verified": 120, "actual_diagnostic_rows_verified": 120, "arm_state_rows_verified": 192,
        "all_old601_states_and_candidates_retained": True, "planning_only_label_bundle_independently_recomputed": True,
        "features_no_actual_source_old_prediction_fields": True, "viewed_diagnostic_fit_guard_rejection_verified": True,
        "equal_family_weights_and_whole_candidate_groups_verified": True, "changed_choices_and_all_six_outcomes_verified": True,
        "finite_identical_input_bounds_independently_recomputed": True, "failed_test_revisions_and_correct_snapshots_preserved": True,
        "new_simulations": 0, "new_fits": 0, "new_llm_calls": 0,
        "verifier_sha256": sha(__file__), "analysis_manifest_sha256": sha(OUT / "analysis_manifest.json")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
