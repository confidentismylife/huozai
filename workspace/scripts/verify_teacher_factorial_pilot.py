"""Check completed pilot artifacts and describe choices without new simulation.

This supplementary check does not alter the frozen collector, caches, analysis,
or original experiment outputs. It writes only post-analysis verification files.
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
OUTPUT = ROOT / "outputs/teacher_factorial_pilot_development_v1"
DIAGNOSIS = ROOT / "outputs/plan_teacher_diagnosis_development_v1"
EPS = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=0, atol=EPS), (actual, expected)


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261803)
    samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def main():
    summary = read_json(OUTPUT / "summary.json")
    manifest = read_json(OUTPUT / "development_manifest.json")
    analysis = read_json(OUTPUT / "analysis_manifest.json")
    for path, expected in manifest["source_input_sha256"].items():
        assert digest(path) == expected, path
    for filename, key in (("summary.json", "summary_sha256"),
                          ("development_manifest.json", "development_manifest_sha256")):
        assert digest(OUTPUT / filename) == analysis[key]
    assert digest(ROOT / "scripts/run_teacher_factorial_pilot.py") == analysis["code_sha256"]
    assert digest(ROOT / "scripts/analyze_teacher_factorial_pilot_cached.py") == analysis["wrapper_code_sha256"]
    assert digest(ROOT / "reports/教师世界聚合与规划预算全新开发验证报告.md") == analysis["report_sha256"]

    states = read_csv(OUTPUT / "state_selection.csv")
    episodes = read_csv(OUTPUT / "evaluation_episode_results.csv")
    lookup = {(r["state_id"], r["arm"], int(r["future_seed"])): r for r in episodes}
    assert len(states) == 48 and len(lookup) == len(episodes) == 3168
    planning = evaluation = steps = 0
    for state in states:
        name = state["state_id"] + ".npz"
        path = OUTPUT / "states" / name
        assert digest(path) == analysis["state_cache_sha256"][name]
        with np.load(path, allow_pickle=False) as cache:
            planning += int(cache["planning_rollouts"])
            evaluation += int(cache["evaluation_rollouts"])
            steps += len(cache["execution_trace"])
            for a, arm in enumerate(manifest["arms"]):
                assert state[arm] == manifest["candidate_names"][int(cache["chosen_indices"][a])]
                for f, seed in enumerate(cache["evaluation_seeds"]):
                    row = lookup[state["state_id"], arm, int(seed)]
                    for m, field in enumerate(manifest["outcome_fields"]):
                        close(float(row[field]), float(cache["evaluation_outcomes"][a, f, m]))
    assert planning == summary["planning_rollouts"] == 19584
    assert evaluation == summary["unique_evaluation_rollouts"] == 1566
    assert steps == summary["actual_execution_trace_steps"] == 41255

    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["arm"], row["source_id"]].append(row)
    families = sorted({r["source_id"] for r in states})
    means = {}
    for (arm, family), rows in grouped.items():
        assert len(rows) == 12
        for metric in manifest["outcome_fields"][:9]:
            means[arm, family, metric] = float(np.mean([float(r[metric]) for r in rows]))
    for row in summary["arm_summary"]:
        for metric in manifest["outcome_fields"][:9]:
            close(float(np.mean([means[row["arm"], family, metric] for family in families])), row[metric])
    for row in summary["paired_effects"]:
        scale = 100 if row["metric"] == "final_incapacitation" else 1
        values = [scale * (means[row["treatment"], f, row["metric"]]
                           - means[row["control"], f, row["metric"]]) for f in families]
        close(float(np.mean(values)), row["delta"])
        lo, hi = bootstrap(values)
        close(lo, row["exploratory_ci_low"])
        close(hi, row["exploratory_ci_high"])
        assert sum(v < -EPS for v in values) == row["improved_families"]
        assert sum(v > EPS for v in values) == row["worsened_families"]
    interaction = [100 * (means["mean12", f, "final_incapacitation"]
                          - means["worst12", f, "final_incapacitation"]
                          - means["mean3", f, "final_incapacitation"]
                          + means["worst3", f, "final_incapacitation"]) for f in families]
    close(float(np.mean(interaction)), summary["factorial_interaction"]["delta_pp"])
    lo, hi = bootstrap(interaction)
    close(lo, summary["factorial_interaction"]["exploratory_ci_low_pp"])
    close(hi, summary["factorial_interaction"]["exploratory_ci_high_pp"])

    comparisons, changes = [], []
    for treatment, control in (("worst12", "worst3"), ("mean12", "mean3"),
                               ("mean12", "worst12"), ("mean3", "worst3")):
        different = [s for s in states if s[treatment] != s[control]]
        item = {"treatment": treatment, "control": control,
                "changed_plan_starts": len(different),
                "changed_plan_families": len({s["source_id"] for s in different}),
                "paired_logical_evaluation_records": 288}
        for metric in ("final_incapacitation", "final_fed"):
            diffs = [float(r[metric]) - float(lookup[r["state_id"], control, int(r["future_seed"])][metric])
                     for r in episodes if r["arm"] == treatment]
            item[metric + "_changed_record_count"] = sum(abs(v) > EPS for v in diffs)
        comparisons.append(item)
        changes.extend({"treatment": treatment, "control": control, "state_id": s["state_id"],
                        "source_id": s["source_id"], "conflict_type": s["conflict_type"],
                        "treatment_plan": s[treatment], "control_plan": s[control]} for s in different)

    kind_lookup = {s["source_id"]: s["conflict_type"] for s in states}
    improvements = []
    for family in families:
        delta = 100 * (means["worst3", family, "final_incapacitation"]
                       - means["pure_ppo_601", family, "final_incapacitation"])
        if abs(delta) > EPS:
            improvements.append({"source_id": family, "conflict_type": kind_lookup[family], "delta_pp": delta})

    failed = OUTPUT / "analysis_revisions/initial_serialization_failure"
    csv_unchanged = {p.name: digest(p) == digest(OUTPUT / p.name) for p in sorted(failed.glob("*.csv"))}
    assert len(csv_unchanged) == 6 and all(csv_unchanged.values())
    diagnosis_manifest = read_json(DIAGNOSIS / "diagnostic_manifest.json")
    for path, expected in diagnosis_manifest["source_input_sha256"].items():
        assert digest(path) == expected, path

    historical = read_json(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique_historical = {r["path"]: r["current_sha256"] for r in historical["records"]}
    status = ROOT / "outputs/third_round_development_status.json"
    saved = ROOT / "outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json"
    assert digest(saved) == unique_historical[str(status)]
    unchanged = 0
    for path, expected in unique_historical.items():
        if Path(path) != status:
            assert digest(path) == expected, path
            unchanged += 1
    qa = read_json(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")
    for artifact in qa["artifacts"]:
        assert digest(artifact["path"]) == artifact["sha256"]

    destination = OUTPUT / "postanalysis"
    destination.mkdir(exist_ok=True)
    with (destination / "changed_plan_selections.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    result = {"role": "post_hoc_cached_verification_and_description_not_new_evaluation",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
              "verification_code_sha256": digest(__file__), "new_rollouts": 0,
              "new_training_runs": 0, "new_llm_calls": 0,
              "pilot_frozen_inputs_unchanged": len(manifest["source_input_sha256"]),
              "diagnosis_original_inputs_unchanged": len(diagnosis_manifest["source_input_sha256"]),
              "pilot_state_cache_hashes_verified": 48, "logical_records_matched_to_caches": 3168,
              "arm_means_paired_effects_and_bootstrap_interaction_rebuilt": True,
              "pre_repair_csv_files_byte_identical": csv_unchanged,
              "historical_unique_snapshot_files": len(unique_historical),
              "historical_files_unchanged_excluding_status": unchanged,
              "historical_status_preserved_exactly": str(saved.relative_to(ROOT)).replace("\\", "/"),
              "historical_report_md_docx_pdf_hashes_unchanged": True,
              "selection_comparisons": comparisons, "worst3_nonzero_family_effects": improvements,
              "limitations": ["Descriptions use the now-viewed development bank.",
                              "No claim of policy equivalence follows from identical incapacitation values.",
                              "Bootstrap intervals remain exploratory and unadjusted for multiplicity."]}
    (destination / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "selection_comparisons": comparisons,
                      "historical_files_unchanged_excluding_status": unchanged}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
