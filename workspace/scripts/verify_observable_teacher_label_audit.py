"""Independently verify saved observable-label diagnostics and cached effects.

This never imports an environment or loads a policy for prediction. Conditional
teacher effects are descriptive comparisons on saved evaluation streams, not a
deployable rule and not a new independent evaluation.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
POST = OUT / "postanalysis"
BASE = 12
EPS = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def family_mean(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(float(row[key]))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def main():
    POST.mkdir(exist_ok=True)
    contract = read_json(OUT / "diagnostic_manifest.json")
    summary = read_json(OUT / "summary.json")
    analysis = read_json(OUT / "analysis_manifest.json")
    for path, expected in contract["source_input_sha256"].items():
        assert digest(path) == expected, path
    for key, name in (("code_sha256", "scripts/audit_observable_teacher_labels.py"),
                      ("diagnostic_manifest_sha256", str((OUT / "diagnostic_manifest.json").relative_to(ROOT))),
                      ("summary_sha256", str((OUT / "summary.json").relative_to(ROOT))),
                      ("integrity_sha256", str((OUT / "integrity_audit.json").relative_to(ROOT)))):
        assert digest(ROOT / name) == analysis[key]
    for name, expected in analysis["output_csv_sha256"].items():
        assert digest(OUT / name) == expected
    preserved_console_summary = read_json(OUT / "analysis_revisions/completed_summary_before_console_repair.json")
    before = {k: v for k, v in preserved_console_summary.items() if k != "completed_at_utc"}
    after = {k: v for k, v in summary.items() if k != "completed_at_utc"}
    assert before == after, "console repair changed numeric analysis"

    specs = {row["bank"]: row for row in summary["source_banks"]}
    bank_manifests = {bank: read_json(Path(spec["directory"]) / "development_manifest.json")
                      for bank, spec in specs.items()}
    primary_rows = read_csv(OUT / "state_primary_diagnostics.csv")
    scores, actual_features, cache_data = {}, {}, {}
    for row in primary_rows:
        bank, state_id = row["bank"], row["state_id"]
        subdir = "training_states" if bank == "PREX" else ("states" if bank in ("TSDEV", "PTDEV") else "audit_states")
        with np.load(Path(specs[bank]["directory"]) / subdir / f"{state_id}.npz", allow_pickle=False) as z:
            means = np.asarray(z["risks"], dtype=np.float64)[np.asarray(z["supported"], bool), :, :3].mean(axis=2)
            worst, average = means.max(axis=0), means.mean(axis=0)
            priority = np.stack((worst[:, 0], average[:, 0], worst[:, 1], average[:, 1], worst[:, 3], worst[:, 4]), axis=1)
            chosen = min(range(17), key=lambda p: tuple(priority[p]) + (p,))
            assert bank_manifests[bank]["candidate_names"][chosen] == row["teacher_plan"]
            assert abs((worst[BASE, 0] - worst[:, 0].min()) * 100 - float(row["primary_gain_vs_601_pp"])) < EPS
            scores[state_id] = priority[:, 0]
            feature = np.asarray(z["observation"], np.float32).copy()
            feature[28:33] = 0
            feature[feature == 0] = 0
            actual_features[state_id] = feature
            if bank in ("PRAU", "PDDEV"):
                cache_data[state_id] = (z["chosen_indices"].copy(), z["evaluation_outcomes"].copy())
    assert len(primary_rows) == 1137
    collision_rows = read_csv(OUT / "exact_observation_groups.csv")
    for row in collision_rows:
        ids = json.loads(row["state_ids"])
        features = np.asarray([actual_features[s] for s in ids])
        if row["view"] == "forecast_only":
            features[:, :19] = 0
            features[:, 28:] = 0
        assert np.all(features == features[0])
        costs = np.asarray([scores[s] for s in ids])
        regrets = (costs - costs.min(axis=1, keepdims=True)) * 100
        for tolerance in (0, .05, .10, .20):
            shared = int(np.all(regrets <= tolerance + EPS, axis=0).sum())
            assert shared == int(row[f"shared_plans_at_{tolerance:.2f}pp"])
        assert abs(regrets.max(axis=0).min() - float(row["minimum_worst_empirical_primary_regret_pp"])) < EPS

    students = read_csv(OUT / "student_state_diagnostics.csv")
    for row in students:
        manifest = bank_manifests[row["bank"]]
        choices, outcomes = cache_data[row["state_id"]]
        index = manifest["arms"].index(row["arm"])
        selected = int(choices[index])
        assert manifest["candidate_names"][selected] == row["selected_plan"]
        primary = scores[row["state_id"]]
        regret = (primary[selected] - primary.min()) * 100
        assert abs(regret - float(row["primary_regret_pp"])) < EPS
        gain = (primary[BASE] - primary.min()) * 100
        if gain > EPS:
            category = "missed_primary_override" if selected == BASE else ("suboptimal_primary_override" if regret > EPS else "primary_optimal_override")
        else:
            category = "kept_601_on_primary_tie" if selected == BASE else ("positive_regret_override_on_primary_tie" if regret > EPS else "zero_regret_override_on_primary_tie")
        assert category == row["category"]
        baseline = outcomes[manifest["arms"].index("pure_ppo_601"), :, 0]
        delta = float((outcomes[index, :, 0] - baseline).mean() * 100)
        teacher_delta = float((outcomes[manifest["arms"].index("teacher_minimax"), :, 0] - baseline).mean() * 100)
        assert abs(delta - float(row["evaluation_delta_vs601_pp"])) < EPS
        row["teacher_eval_delta_vs601_pp"] = teacher_delta
        row["student_minus_teacher_pp"] = delta - teacher_delta

    decomposition = []
    for saved in summary["student_summary"]:
        bank, group, scope = saved["bank"], saved["group"], saved["scope"]
        rows = [r for r in students if r["bank"] == bank and r["group"] == group
                and (scope != "initial" or r["step"] == "0")]
        assert len(rows) == saved["student_state_records"]
        np.testing.assert_allclose(family_mean(rows, "evaluation_delta_vs601_pp"), saved["family_equal_eval_delta_vs601_pp"], atol=EPS, rtol=0)
        primary_optimal = sum(float(r["primary_regret_pp"]) <= EPS for r in rows) / len(rows)
        assert abs(primary_optimal - saved["primary_optimal_rate"]) < EPS
        total_student = total_teacher = total_gap = 0.0
        for category in sorted({r["category"] for r in rows}):
            weighted = [{**r,
                         "student_contribution": float(r["evaluation_delta_vs601_pp"]) if r["category"] == category else 0,
                         "teacher_contribution": r["teacher_eval_delta_vs601_pp"] if r["category"] == category else 0,
                         "gap_contribution": r["student_minus_teacher_pp"] if r["category"] == category else 0}
                        for r in rows]
            student_delta = family_mean(weighted, "student_contribution")
            teacher_delta = family_mean(weighted, "teacher_contribution")
            gap = family_mean(weighted, "gap_contribution")
            assert abs(gap - student_delta + teacher_delta) < EPS
            count = sum(r["category"] == category for r in rows)
            decomposition.append({"bank": bank, "group": group, "scope": scope, "category": category,
                                  "records": count, "record_fraction": count / len(rows),
                                  "student_contribution_delta_vs601_pp": student_delta,
                                  "teacher_contribution_delta_vs601_pp": teacher_delta,
                                  "contribution_to_student_minus_teacher_gap_pp": gap,
                                  "descriptive_not_deployable_or_causal": True})
            total_student += student_delta
            total_teacher += teacher_delta
            total_gap += gap
        assert abs(total_gap - total_student + total_teacher) < EPS

    with (POST / "cached_student_teacher_effect_decomposition.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decomposition[0]))
        writer.writeheader()
        writer.writerows(decomposition)
    initial = [r for r in summary["exact_group_summary"] if r["scope"] == "initial" and r["view"] == "full49"]
    totals = {"initial_observation_groups": sum(r["unique_observation_groups"] for r in initial),
              "groups_with_different_teacher_labels": sum(r["groups_with_different_teacher_labels"] for r in initial),
              "groups_without_shared_primary_optimum": sum(r["groups_without_shared_plan_at_0.00pp"] for r in initial),
              "groups_without_shared_plan_at_0.20pp": sum(r["groups_without_shared_plan_at_0.20pp"] for r in initial)}
    totals["different_label_groups_with_shared_primary_optimum"] = totals["groups_with_different_teacher_labels"] - totals["groups_without_shared_primary_optimum"]
    assert totals == {"initial_observation_groups": 180, "groups_with_different_teacher_labels": 47,
                      "groups_without_shared_primary_optimum": 15, "groups_without_shared_plan_at_0.20pp": 7,
                      "different_label_groups_with_shared_primary_optimum": 32}
    result = {"role": "independent_cached_development_audit_verification_not_final", "all_checks_passed": True,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "states_recomputed": 1137,
              "student_state_records_recomputed": len(students), "source_files_unchanged": len(contract["source_input_sha256"]),
              "exact_feature_group_and_regret_checks_passed": True,
              "student_effects_and_categories_recomputed": True,
              "console_repair_numeric_summary_unchanged": True,
              "descriptive_initial_collision_totals": totals,
              "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0,
              "source_script_sha256": digest(__file__),
              "console_repair_wrapper_sha256": digest(ROOT / "scripts/run_observable_teacher_label_audit_cached.py"),
              "effect_decomposition_sha256": digest(POST / "cached_student_teacher_effect_decomposition.csv")}
    write_json(POST / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
