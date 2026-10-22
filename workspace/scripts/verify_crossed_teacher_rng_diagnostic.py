"""Verify crossed-factor outcomes and actuator traces without new rollouts."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1"
EPS = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def primary_and_priority(raw):
    means = raw.mean(axis=2)
    worst, average = means.max(axis=0), means.mean(axis=0)
    priority = np.column_stack((worst[:, 0], average[:, 0], worst[:, 1], average[:, 1], worst[:, 3], worst[:, 4]))
    return priority[:, 0], min(range(17), key=lambda p: tuple(priority[p]) + (p,))


def check_shared(costs, tolerance):
    regrets = (costs - costs.min(axis=1, keepdims=True)) * 100
    return int(np.all(regrets <= tolerance + EPS, axis=0).sum()), float(np.min(np.max(regrets, axis=0)))


def main():
    manifest = read_json(OUT / "development_manifest.json")
    summary = read_json(OUT / "summary.json")
    analysis = read_json(OUT / "analysis_manifest.json")
    for path, expected in manifest["source_input_sha256"].items():
        assert digest(path) == expected, path
    for name, expected in analysis["new_cell_sha256"].items():
        assert digest(OUT / "cells" / name) == expected
    for name, expected in analysis["output_csv_sha256"].items():
        assert digest(OUT / name) == expected
    for key, file in (("summary_sha256", "summary.json"), ("integrity_sha256", "integrity_audit.json"),
                      ("development_manifest_sha256", "development_manifest.json")):
        assert digest(OUT / file) == analysis[key]
    assert digest(ROOT / "scripts/run_crossed_teacher_rng_diagnostic.py") == analysis["collector_sha256"]
    cell_rows = {(r["family"], int(r["population_index"]), int(r["stream_index"])): r for r in read_csv(OUT / "cell_diagnostics.csv")}
    pairs = {(r["family"], r["axis"], int(r["fixed_other_level"])): r for r in read_csv(OUT / "axis_pair_diagnostics.csv")}
    families = {r["family"]: r for r in read_csv(OUT / "family_diagnostics.csv")}
    contrast_rows = {(r["family"], r["candidate"]): r for r in read_csv(OUT / "candidate_factorial_contrasts.csv")}
    categories = []
    axis_totals = {axis: {"comparisons": 0, "different_lexical_labels": 0, "primary_conflicting_comparisons": 0,
                        "families_with_any_primary_conflict": 0, "conflicts_beyond_0.20pp": 0}
                   for axis in ("population", "planning_stream")}
    pooled_conflicts = pooled_beyond = steps = neutral = holds = new_rollouts = 0
    for group in manifest["groups"]:
        family = group["family"]
        raw = np.empty((2, 2, 2, 17, 3, 11))
        cost, chosen = np.empty((2, 2, 17)), np.empty((2, 2), int)
        fingerprints = []
        for p in range(2):
            for s in range(2):
                diagonal = p == s
                path = Path(group["populations"][p]["cache"]) if diagonal else OUT / "cells" / f"{family}_p{p}_s{s}.npz"
                with np.load(path, allow_pickle=False) as z:
                    raw[p, s] = z["risks"][:, :, :3]
                    with np.load(group["populations"][p]["cache"], allow_pickle=False) as original:
                        np.testing.assert_array_equal(z["observation"], original["observation"])
                    if not diagonal:
                        assert int(z["population_index"]) == p and int(z["stream_index"]) == s
                        assert int(z["population_seed"]) == group["populations"][p]["population_seed"]
                        np.testing.assert_array_equal(z["planning_seeds"], group["populations"][s]["planning_seeds"])
                        assert z["supported"].all()
                        np.testing.assert_array_equal(z["world_observations"], np.tile(z["observation"], (2, 1)))
                        fingerprints.append(str(z["initial_people_fingerprint"]))
                        trace, offsets = z["execution_trace"], z["trace_offsets"]
                        assert len(offsets) == 103 and offsets[0] == 0 and offsets[-1] == len(trace)
                        assert np.isfinite(trace).all() and np.isin(trace[:, 1], range(9)).all()
                        assert np.isin(trace[:, 2], [-1] + list(range(9))).all()
                        assert np.isin(trace[:, 3], [0, 1, 2]).all()
                        assert np.isin(trace[:, 5], [0, 1]).all()
                        assert np.all((trace[:, 4] >= 0) & (trace[:, 4] <= 1))
                        np.testing.assert_array_equal(trace[:, 2] == -1, (trace[:, 1] != 8) & (trace[:, 3] == 2))
                        for i, key in enumerate(np.ndindex(2, 17, 3)):
                            t = trace[offsets[i]:offsets[i + 1]]
                            o = z["risks"][key]
                            assert 0 < len(t) <= manifest["env_config"]["max_steps"] and len(t) == o[5]
                            np.testing.assert_array_equal(t[:, 0], np.arange(len(t)))
                            np.testing.assert_allclose([t[:, 4].mean(), (t[:, 1] == 8).sum(), t[:, 5].mean(), t[0, 1], t[0, 2]], o[6:11], atol=EPS, rtol=0)
                        new_rollouts += int(z["planning_rollouts"])
                        steps += len(trace)
                        neutral += int(((trace[:, 3] == 2) & (trace[:, 4] > 0)).sum())
                        holds += int(trace[:, 5].sum())
                np.testing.assert_allclose(raw[p, s, ..., [0, 2, 3]].sum(axis=0), 1, atol=EPS, rtol=0)
                cost[p, s], chosen[p, s] = primary_and_priority(raw[p, s])
                row = cell_rows[(family, p, s)]
                assert row["selected_plan"] == manifest["candidate_names"][chosen[p, s]]
                assert abs(float(row["teacher_primary_gain_vs601_pp"]) - (cost[p, s, 12] - cost[p, s].min()) * 100) < EPS
        assert len(set(fingerprints)) == 2
        flags = []
        for axis in ("population", "planning_stream"):
            has_conflict = False
            for fixed in range(2):
                pair_cost = cost[:, fixed] if axis == "population" else cost[fixed]
                labels = chosen[:, fixed] if axis == "population" else chosen[fixed]
                row = pairs[(family, axis, fixed)]
                for tolerance in (0, .05, .10, .20):
                    assert int(row[f"shared_plans_at_{tolerance:.2f}pp"]) == check_shared(pair_cost, tolerance)[0]
                assert abs(float(row["minimum_worst_primary_regret_pp"]) - check_shared(pair_cost, 0)[1]) < EPS
                assert (row["different_lexical_teacher"] == "True") == (labels[0] != labels[1])
                totals = axis_totals[axis]
                totals["comparisons"] += 1
                totals["different_lexical_labels"] += int(labels[0] != labels[1])
                conflict = check_shared(pair_cost, 0)[0] == 0
                totals["primary_conflicting_comparisons"] += int(conflict)
                totals["conflicts_beyond_0.20pp"] += int(check_shared(pair_cost, .20)[0] == 0)
                has_conflict |= conflict
            axis_totals[axis]["families_with_any_primary_conflict"] += int(has_conflict)
            flags.append(has_conflict)
        category = ("both_axes_pairwise_conflict" if all(flags) else "population_axis_only_pairwise_conflict" if flags[0]
                    else "planning_stream_axis_only_pairwise_conflict" if flags[1]
                    else "neither_axis_pairwise_conflict_diagonal_still_conflicts")
        assert families[family]["pairwise_conflict_category"] == category
        categories.append(category)
        # Reshape stream and future into six futures separately for each world.
        pooled = raw.transpose(0, 2, 3, 1, 4, 5).reshape(2, 2, 17, 6, 11)
        pooled_cost = np.asarray([primary_and_priority(pooled[p])[0] for p in range(2)])
        for tolerance in (0, .05, .10, .20):
            assert int(families[family][f"pooled_population_shared_plans_at_{tolerance:.2f}pp"]) == check_shared(pooled_cost, tolerance)[0]
        pooled_conflicts += int(check_shared(pooled_cost, 0)[0] == 0)
        pooled_beyond += int(check_shared(pooled_cost, .20)[0] == 0)
        gains = (cost[..., 12, None] - cost) * 100
        population_delta = gains[1].mean(axis=0) - gains[0].mean(axis=0)
        stream_delta = gains[:, 1].mean(axis=0) - gains[:, 0].mean(axis=0)
        interaction = (gains[1, 1] - gains[1, 0]) - (gains[0, 1] - gains[0, 0])
        for i, name in enumerate(manifest["candidate_names"]):
            row = contrast_rows[(family, name)]
            np.testing.assert_allclose([float(row["population_mean_difference_gain_pp"]), float(row["planning_stream_mean_difference_gain_pp"]),
                                       float(row["interaction_difference_in_differences_gain_pp"])],
                                      [population_delta[i], stream_delta[i], interaction[i]], atol=EPS, rtol=0)
    assert new_rollouts == 3060
    assert axis_totals == summary["axis_comparisons"]
    assert dict(Counter(categories)) == summary["pairwise_conflict_family_categories"]
    assert pooled_conflicts == summary["pooled_six_future_population_primary_conflicting_families"]
    assert pooled_beyond == summary["pooled_six_future_population_conflicts_beyond_0.20pp"]
    assert steps == summary["new_planning_trace_steps"] and neutral == summary["neutral_positive_release_trace_steps"] and holds == summary["hold_blocked_trace_steps"]
    result = {"role": "independent_read_only_crossed_diagnostic_verification", "all_checks_passed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "groups_checked": 15, "cells_checked": 60,
        "new_planning_rollouts_verified": new_rollouts, "actual_trace_steps_verified": steps,
        "new_rollouts_in_verification": 0, "new_training_in_verification": 0, "new_llm_calls_in_verification": 0,
        "source_inputs_unchanged": len(manifest["source_input_sha256"]), "factor_axis_and_pooled_costs_recomputed": True,
        "candidate_factorial_contrasts_recomputed": True, "actuator_feedback_and_neutral_release_checked": True,
        "verifier_sha256": digest(__file__), "analysis_manifest_sha256": digest(OUT / "analysis_manifest.json")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
