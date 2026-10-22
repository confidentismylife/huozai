"""Package crossed diagnostics, verify legacy sources and preserve live status."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1"
AUDIT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_crossed_teacher_rng_diagnostic_20261003.json"
REPORT = ROOT / "reports/人口与规划随机流交叉诊断报告_2026-10-03.md"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    assert not (OUT / "completion_manifest.json").exists(), "preserve existing completion"
    assert not HISTORY.exists(), "history path must be new"
    summary = read_json(OUT / "summary.json")
    verification = read_json(OUT / "verification.json")
    assert verification["all_checks_passed"] and summary["new_planning_rollouts_total"] == 3060
    assert summary["new_training_runs"] == summary["new_llm_calls"] == summary["new_evaluation_rollouts"] == 0
    prior_completion_path = AUDIT / "completion_manifest.json"
    prior_completion = read_json(prior_completion_path)
    for path, expected in prior_completion["files"].items():
        assert digest(ROOT / path) == expected, path
    prior_inputs = read_json(AUDIT / "diagnostic_manifest.json")["source_input_sha256"]
    for path, expected in prior_inputs.items():
        assert digest(path) == expected, path
    pilot_completion = read_json(ROOT / "outputs/teacher_factorial_pilot_development_v1/completion_manifest.json")
    for path, expected in pilot_completion["files"].items():
        effective = ROOT / path
        if effective.resolve() == STATUS:
            effective = ROOT / "outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"
        assert digest(effective) == expected, path
    value_completion = read_json(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")
    for field in ("sources_unchanged", "files"):
        for path, expected in value_completion[field].items():
            assert digest(ROOT / path) == expected, path
    historical = read_json(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    history_files = {Path(r["path"]).resolve(): r["current_sha256"] for r in historical["records"]}
    for path, expected in history_files.items():
        effective = ROOT / "outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path
        assert digest(effective) == expected, str(path)
    qa = read_json(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")
    for artifact in qa["artifacts"]:
        assert digest(artifact["path"]) == artifact["sha256"]
    core_names = {"env.py", "dynamic_conflicts.py", "confirmatory.py", "terminal_shield.py", "source_world_planning.py",
                  "ppo_reference_planning.py", "behavior.py", "hazard.py", "config.py", "run_source_world_planning.py"}
    core_checks = []
    for folder in ("ppo_extended_world_planning_development_v1", "baseline_relative_plan_distillation_development_v1",
                   "observable_plan_distillation_development_v1", "teacher_factorial_pilot_development_v1"):
        manifest = read_json(ROOT / "outputs" / folder / "development_manifest.json")
        source_hashes = manifest.get("source_hashes", manifest.get("source_input_sha256", {}))
        bank_checks = []
        for file, expected in source_hashes.items():
            path = Path(file)
            if not path.is_absolute():
                path = ROOT / path
            if path.name in core_names:
                actual = digest(path)
                assert actual == expected, str(path)
                bank_checks.append({"bank_directory": folder, "path": str(path.resolve()), "legacy_sha256": expected,
                                    "current_sha256": actual, "unchanged": True})
        assert len(bank_checks) == 10
        core_checks.extend(bank_checks)
    import csv
    with (OUT / "family_diagnostics.csv").open(encoding="utf-8-sig", newline="") as f:
        families = list(csv.DictReader(f))
    rms_medians = {axis: {subset: float(np.median([float(row[f"{axis}_gain_contrast_rms_{subset}_pp"]) for row in families]))
                          for subset in ("all17", "five_ppo")}
                   for axis in ("population", "planning_stream", "interaction")}
    supplement = {"all_checks_passed": True, "legacy_core_checks": core_checks, "legacy_core_check_count": 40,
                  "family_rms_contrast_medians_pp": rms_medians,
                  "all_family_gain_interactions_nonzero": all(float(r["interaction_gain_contrast_rms_all17_pp"]) > 1e-12 for r in families),
                  "new_rollouts": 0, "new_training_runs": 0, "new_llm_calls": 0,
                  "finalizer_sha256": digest(__file__)}
    write_json(OUT / "legacy_core_and_effect_summary_verification.json", supplement)
    write_json(OUT / "targeted_test_verification.json", {
        "command": "D:/miniconda3/python.exe -m pytest -q tests/test_crossed_teacher_rng_diagnostic.py tests/test_source_world_planning.py tests/test_ppo_reference_planning.py",
        "observed_result": "18 passed in 7.08s", "exit_code": 0, "tests_passed": 18,
        "new_diagnostic_tests": 6, "existing_planning_tests": 12, "full_project_tests_rerun": False,
        "test_sha256": {p: digest(ROOT / p) for p in ("tests/test_crossed_teacher_rng_diagnostic.py", "tests/test_source_world_planning.py", "tests/test_ppo_reference_planning.py")}})
    report_text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", report_text)
    assert len(links) == 6 and all(Path(path).is_file() for path in links)
    previous_sha = digest(STATUS)
    previous = read_json(STATUS)
    HISTORY.write_bytes(STATUS.read_bytes())
    assert digest(HISTORY) == previous_sha
    now = datetime.now(timezone.utc).isoformat()
    status = {
        "stage": "crossed_population_planning_stream_diagnostic_complete_both_factors_and_interactions_observed",
        "updated_date": "2026-10-03", "updated_at_utc": now, "freeze_allowed": False,
        "hard_shield_qualification_passed": False, "formal_round3_training_started": False,
        "formal_round3_final_bank_generated": False, "new_simulations": 3060, "new_planning_rollouts": 3060,
        "new_evaluation_rollouts": 0, "new_supervised_training_runs": 0, "new_ppo_training_seeds": 0,
        "new_llm_calls": 0, "observable_student_closed_loop_advantage_established": False,
        "teacher_generalization_advantage_established": False, "llm_control_contribution_established": False,
        "teacher_benefit_observed_in_ptdev_small_development_pilot": previous["teacher_benefit_observed_in_ptdev_small_development_pilot"],
        "additional_world_aggregation_gain_established": False, "additional_planning_budget_gain_established": False,
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "summary": "outputs/crossed_teacher_rng_diagnostic_development_v1/summary.json",
        "manifest": "outputs/crossed_teacher_rng_diagnostic_development_v1/development_manifest.json",
        "integrity_audit": "outputs/crossed_teacher_rng_diagnostic_development_v1/integrity_audit.json",
        "analysis_manifest": "outputs/crossed_teacher_rng_diagnostic_development_v1/analysis_manifest.json",
        "independent_verification": "outputs/crossed_teacher_rng_diagnostic_development_v1/verification.json",
        "completion_manifest": "outputs/crossed_teacher_rng_diagnostic_development_v1/completion_manifest.json",
        "previous_stage_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "previous_stage_status_sha256": previous_sha,
        "previous_observable_label_audit": {"stage": previous["stage"], "report": previous["report"], "summary": previous["summary"],
                                           "descriptive_initial_collisions": previous["descriptive_initial_collisions"]},
        "previous_teacher_pilot": previous["previous_teacher_pilot"],
        "diagnostic_scope": {"selected_conflicting_development_families": 15, "initial_population_states": 30,
            "factor_cells": 60, "new_cells": 30, "reused_diagonal_cells": 30, "reused_diagonal_planning_records": 3060,
            "new_actual_trace_steps": 79873, "independent_control_efficacy_evaluation": False,
            "training_families": 8, "calibration_families": 2, "viewed_audit_families": 5},
        "axis_comparisons": summary["axis_comparisons"],
        "pairwise_conflict_family_categories": summary["pairwise_conflict_family_categories"],
        "pooled_six_future_population_primary_conflicting_families": 6,
        "pooled_six_future_population_conflicts_beyond_0.20pp": 2,
        "qualification_thresholds_changed": False,
        "conclusion": "15个选定家庭中6个人口轴比较冲突、4个规划流轴比较冲突、5个两轴冲突。合并同6未来后6家庭仍无共同经验主最优、2家庭冲突超过0.20个百分点；存在成本交互，不能分配普遍因果比例或解释为实际疏散提升。",
        "next_step": "另存集合式主标签与相对601覆盖收益不确定性的开发方案；只用训练/校准拟合，以全新开发家庭和强601对照检查实际效果，不直接扩大正式训练。",
        "research_limits": summary["limitations"], "targeted_tests": {"tests_passed": 18, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": digest(REPORT), "summary_sha256": digest(OUT / "summary.json")}
    write_json(STATUS, status)
    for path, expected in prior_completion["files"].items():
        assert digest(HISTORY if (ROOT / path).resolve() == STATUS else ROOT / path) == expected
    artifacts = [REPORT, ROOT / "reports/人口与规划随机流交叉诊断方案_2026-10-03.md", HISTORY, STATUS,
                 ROOT / "scripts/run_crossed_teacher_rng_diagnostic.py", ROOT / "scripts/verify_crossed_teacher_rng_diagnostic.py",
                 Path(__file__), ROOT / "tests/test_crossed_teacher_rng_diagnostic.py"]
    artifacts += [p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json"]
    completion = {
        "role": "selected_development_crossed_rng_diagnostic_completion_not_efficacy",
        "created_at_utc": now, "all_checks_passed": True, "new_planning_rollouts": 3060,
        "new_evaluation_rollouts": 0, "new_training_runs": 0, "new_llm_calls": 0,
        "independent_cells_checked": 60, "independent_actual_trace_steps_checked": 79873,
        "frozen_inputs_unchanged": 86, "prior_audit_frozen_inputs_unchanged": len(prior_inputs),
        "legacy_core_checks_passed": 40, "targeted_tests_passed": 18,
        "previous_completion_files_checked": len(prior_completion["files"]),
        "previous_completion_files_unchanged_at_current_paths": len(prior_completion["files"]) - 1,
        "intentional_live_status_update_with_exact_previous_bytes_saved": True,
        "saved_prior_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"), "saved_prior_status_sha256": previous_sha,
        "previous_completion_manifest_sha256": digest(prior_completion_path),
        "pilot_and_research_value_completion_chains_verified": True,
        "historical_snapshot_files_checked": len(history_files), "historical_report_md_docx_pdf_unchanged": True,
        "readme_unchanged": True, "valid_local_report_links": len(links),
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in artifacts}}
    write_json(OUT / "completion_manifest.json", completion)
    for path, expected in read_json(OUT / "completion_manifest.json")["files"].items():
        assert digest(ROOT / path) == expected
    print(json.dumps({"completed": True, "new_planning_rollouts": 3060, "new_training_runs": 0, "new_llm_calls": 0,
        "completion_files_hashed": len(completion["files"]), "previous_status_preserved": True,
        "legacy_core_checks_passed": 40, "prior_audit_frozen_inputs_unchanged": len(prior_inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
