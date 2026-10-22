"""Preserve prior research status and package the completed read-only audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"
REPORT = ROOT / "reports/可观测教师标签与覆盖收益审计报告_2026-10-03.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    assert not (OUT / "completion_manifest.json").exists(), "already finalized; preserve published completion"
    assert not HISTORY.exists(), "history path must be new"
    verification = read_json(OUT / "postanalysis/verification.json")
    assert verification["all_checks_passed"]
    diagnostic = read_json(OUT / "diagnostic_manifest.json")
    for path, expected in diagnostic["source_input_sha256"].items():
        assert digest(path) == expected, path
    prior_completion_path = ROOT / "outputs/teacher_factorial_pilot_development_v1/completion_manifest.json"
    prior_completion = read_json(prior_completion_path)
    for path, expected in prior_completion["files"].items():
        assert digest(ROOT / path) == expected, path
    value_completion_path = ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json"
    value_completion = read_json(value_completion_path)
    for category in ("sources_unchanged", "files"):
        for path, expected in value_completion[category].items():
            assert digest(ROOT / path) == expected, path
    historical = read_json(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    historical_files = {Path(row["path"]).resolve(): row["current_sha256"] for row in historical["records"]}
    old_status = ROOT / "outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json"
    for path, expected in historical_files.items():
        assert digest(old_status if path == STATUS else path) == expected, str(path)
    qa = read_json(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")
    for artifact in qa["artifacts"]:
        assert digest(artifact["path"]) == artifact["sha256"]
    report_text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", report_text)
    assert len(links) == 6 and all(Path(path).is_file() for path in links)
    previous_bytes = STATUS.read_bytes()
    previous = read_json(STATUS)
    previous_sha = digest(STATUS)
    HISTORY.write_bytes(previous_bytes)
    assert digest(HISTORY) == previous_sha
    now = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "targeted_test_verification.json", {
        "command": "D:/miniconda3/python.exe -m pytest -q tests/test_observable_teacher_label_audit.py",
        "observed_result": "9 passed in 2.82s", "tests_passed": 9, "exit_code": 0,
        "test_sha256": digest(ROOT / "tests/test_observable_teacher_label_audit.py"),
        "audit_code_sha256": diagnostic["code_sha256"], "full_project_tests_rerun": False,
        "historical_117_test_run_preserved_in_previous_history": True})
    status = {
        "stage": "observable_teacher_label_audit_complete_label_disagreement_often_primary_ties_override_value_unstable",
        "updated_date": "2026-10-03", "updated_at_utc": now,
        "freeze_allowed": False, "hard_shield_qualification_passed": False,
        "formal_round3_training_started": False, "formal_round3_final_bank_generated": False,
        "new_simulations": 0, "new_supervised_training_runs": 0,
        "new_ppo_training_seeds": 0, "new_llm_calls": 0,
        "observable_student_closed_loop_advantage_established": False,
        "teacher_generalization_advantage_established": False,
        "llm_control_contribution_established": False,
        "teacher_benefit_observed_in_ptdev_small_development_pilot": True,
        "additional_world_aggregation_gain_established": False,
        "additional_planning_budget_gain_established": False,
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "summary": "outputs/observable_teacher_label_audit_development_v1/summary.json",
        "manifest": "outputs/observable_teacher_label_audit_development_v1/diagnostic_manifest.json",
        "integrity_audit": "outputs/observable_teacher_label_audit_development_v1/integrity_audit.json",
        "analysis_manifest": "outputs/observable_teacher_label_audit_development_v1/analysis_manifest.json",
        "independent_verification": "outputs/observable_teacher_label_audit_development_v1/postanalysis/verification.json",
        "completion_manifest": "outputs/observable_teacher_label_audit_development_v1/completion_manifest.json",
        "previous_stage_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "previous_stage_status_sha256": previous_sha,
        "previous_teacher_pilot": {"stage": previous["stage"], "families": previous["new_development_families"],
            "initial_states": previous["new_development_initial_states"], "total_unique_full_rollouts": previous["total_unique_full_rollouts"],
            "primary_paired_effect": previous["primary_paired_effect"], "report": previous["report"],
            "summary": previous["summary"], "additional_factor_gain_established": False},
        "audit_scope": {"banks": 5, "families": 180, "states": 1137, "initial_states": 360,
            "training_families": 84, "training_states": 592, "calibration_states": 81, "audit_states": 464,
            "student_state_records_recomputed": 7880, "inputs_hashed_unchanged": 1227,
            "independent_final_evaluation": False},
        "descriptive_initial_collisions": verification["descriptive_initial_collision_totals"],
        "qualification_thresholds_changed": False,
        "conclusion": "初始47组教师标签不同，其中32组仍有共同经验主最优计划；15组有实质经验冲突，但人口流与规划采样未分离。学生存在漏覆盖及错误覆盖；PRAU部分有规划收益的状态教师检验也变差，不可只归因学生模仿不足。",
        "next_step": "另存小预算人口随机流×规划未来随机流交叉诊断方案，先解释现有15组冲突；不直接追加大规模学生或正式PPO训练。",
        "proposed_crossed_diagnostic": {"selected_existing_conflict_groups": 15, "maximum_new_planning_rollouts": 3060,
            "started": False, "post_hoc_mechanism_diagnostic_not_general_conflict_rate_estimation": True},
        "research_limits": read_json(OUT / "summary.json")["limitations"],
        "targeted_tests": {"tests_passed": 9, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": digest(REPORT), "summary_sha256": digest(OUT / "summary.json")}
    write_json(STATUS, status)
    assert digest(HISTORY) == previous_sha
    for path, expected in prior_completion["files"].items():
        assert digest(HISTORY if (ROOT / path).resolve() == STATUS else ROOT / path) == expected
    artifacts = [REPORT, ROOT / "reports/可观测教师标签与覆盖收益审计方案.md", HISTORY, STATUS,
                 ROOT / "scripts/audit_observable_teacher_labels.py",
                 ROOT / "scripts/run_observable_teacher_label_audit_cached.py",
                 ROOT / "scripts/verify_observable_teacher_label_audit.py", Path(__file__),
                 ROOT / "tests/test_observable_teacher_label_audit.py"]
    artifacts += [p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json"]
    completion = {
        "role": "observable_teacher_label_read_only_development_audit_completion_not_confirmatory",
        "created_at_utc": now, "all_checks_passed": True,
        "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0,
        "source_inputs_unchanged": 1227, "independent_states_recomputed": 1137,
        "independent_student_records_recomputed": 7880, "targeted_tests_passed": 9,
        "prior_pilot_completion_files_checked": len(prior_completion["files"]),
        "prior_pilot_files_unchanged_at_current_paths": len(prior_completion["files"]) - 1,
        "intentional_live_status_update_with_exact_previous_bytes_saved": True,
        "saved_prior_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"), "saved_prior_status_sha256": previous_sha,
        "prior_pilot_completion_manifest_sha256": digest(prior_completion_path),
        "research_value_sources_unchanged": len(value_completion["sources_unchanged"]),
        "research_value_artifacts_unchanged": len(value_completion["files"]),
        "research_value_completion_manifest_sha256": digest(value_completion_path),
        "historical_unique_snapshot_files_checked": len(historical_files),
        "historical_files_unchanged_excluding_already_preserved_live_status": len(historical_files) - 1,
        "historical_report_md_docx_pdf_unchanged": True, "readme_unchanged": True,
        "valid_local_report_links": len(links), "console_repair_numeric_summary_unchanged": True,
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in artifacts}}
    write_json(OUT / "completion_manifest.json", completion)
    for path, expected in read_json(OUT / "completion_manifest.json")["files"].items():
        assert digest(ROOT / path) == expected
    print(json.dumps({"completed": True, "completion_manifest": str((OUT / "completion_manifest.json").relative_to(ROOT)),
                      "previous_status_preserved": True, "report_links_checked": len(links),
                      "prior_completion_files_checked": len(prior_completion["files"]),
                      "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
