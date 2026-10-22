"""Preserve previous status, verify chains and package cached development work."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sys

from finalize_set_aware_distillation import sha, read, write, check_completion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/consistent_gain_crossfit_development_v1"
PRIOR = ROOT / "outputs/set_aware_distillation_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_consistent_gain_crossfit_20261003.json"
REPORT = ROOT / "reports/统一提名与覆盖收益交叉拟合诊断报告_2026-10-03.md"


def main():
    assert not (OUT / "completion_manifest.json").exists() and not HISTORY.exists()
    m, s, v = (read(OUT / path) for path in ("development_manifest.json", "summary.json", "verification.json"))
    assert v["all_checks_passed"] and s["new_lightweight_linear_fits"] == 5
    assert s["new_simulations"] == s["new_neural_training"] == s["new_ppo_training"] == s["new_llm_calls"] == 0
    assert v["verifier_sha256"] == sha(ROOT / "scripts/verify_consistent_gain_crossfit.py")
    for path, expected in m["source_input_sha256"].items():
        assert sha(path) == expected
    assert sha(STATUS) == m["previous_live_status_sha256"] == sha(OUT / "previous_status_exact_bytes.json")
    assert sha(PRIOR / "completion_manifest.json") == m["prior_completion_sha256"]
    prior_counts = {
        "set_aware": check_completion("set_aware_distillation_development_v1", STATUS),
        "crossed": check_completion("crossed_teacher_rng_diagnostic_development_v1", ROOT / "outputs/third_round_status_history/before_set_aware_distillation_20261003.json"),
        "observable": check_completion("observable_teacher_label_audit_development_v1", ROOT / "outputs/third_round_status_history/before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot": check_completion("teacher_factorial_pilot_development_v1", ROOT / "outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"),
        "research_value": check_completion("research_value_quantification_20261003", STATUS)}
    for path, expected in read(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items():
        assert sha(ROOT / path) == expected
    old_audit = read(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    old_files = {Path(r["path"]).resolve(): r["current_sha256"] for r in old_audit["records"]}
    for path, expected in old_files.items():
        assert sha(ROOT / "outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path) == expected
    for artifact in read(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    previous_package = read(PRIOR / "packaging_verification.json")
    for entry in previous_package["legacy_core_checks"]:
        assert sha(entry["file"]) == entry["sha256"]
    text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", text)
    assert len(links) == 9 and all(Path(path).is_file() for path in links)
    assert "1392" in text and "576" in text and "没有新增火灾模拟" in text
    for block, arms in (("train_oof_initial", ("pure_ppo_601", "ridge_gain_ungated", "ridge_gain_consistent", "cached_teacher_primary")),
                        ("viewed_sgdev_initial", tuple(r["arm"] for r in s["summary_rows"] if r["block"] == "viewed_sgdev_initial"))):
        for row in s["summary_rows"]:
            if row["block"] == block and row["arm"] in arms:
                assert row["arm"] in text and f"{row['cached_primary_gain_pp_family_equal']:+.4f}" in text
    write(OUT / "targeted_test_verification.json", {"tests_passed": 10, "exit_code": 0,
        "command": "D:/miniconda3/python.exe -m pytest -q tests/test_consistent_gain_policy.py",
        "observed_result": "10 passed in 0.35s", "observed_before_prepare_and_fitting": True,
        "test_sha256": sha(ROOT / "tests/test_consistent_gain_policy.py"), "full_project_tests_rerun": False})
    write(OUT / "packaging_verification.json", {"all_checks_passed": True, "source_input_count": len(m["source_input_sha256"]),
        "prior_completion_files_checked": prior_counts, "legacy_core_checks_passed": len(previous_package["legacy_core_checks"]),
        "historical_unique_files_checked": len(old_files), "historical_md_docx_pdf_unchanged": True, "readme_unchanged": True,
        "valid_local_report_links": len(links), "new_simulations": 0, "new_fits_in_finalization": 0,
        "finalizer_sha256": sha(__file__)})
    previous = read(STATUS); before_sha = sha(STATUS)
    HISTORY.write_bytes(STATUS.read_bytes()); assert sha(HISTORY) == before_sha
    status = dict(previous)
    for field in ("primary_paired_effect", "gate_summary", "set_aware_pilot_scope"):
        status.pop(field, None)
    now = datetime.now(timezone.utc).isoformat()
    status.update({"stage": "nomination_override_gain_consistency_fixed_cached_crossfit_prediction_quality_unresolved",
        "updated_date": "2026-10-03", "updated_at_utc": now, "freeze_allowed": False,
        "hard_shield_qualification_passed": False, "formal_round3_training_started": False, "formal_round3_final_bank_generated": False,
        "new_simulations": 0, "new_planning_rollouts": 0, "new_evaluation_rollouts": 0,
        "new_supervised_training_runs": 5, "new_lightweight_linear_fits": 5, "new_supervised_neural_training_runs": 0,
        "new_ppo_training_seeds": 0, "new_llm_calls": 0,
        "nomination_override_same_score_verified": True, "actual_control_efficacy_evaluated_in_current_stage": False,
        "observable_student_closed_loop_advantage_established": False, "teacher_generalization_advantage_established": False,
        "llm_control_contribution_established": False,
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "summary": "outputs/consistent_gain_crossfit_development_v1/summary.json",
        "manifest": "outputs/consistent_gain_crossfit_development_v1/development_manifest.json",
        "integrity_audit": "outputs/consistent_gain_crossfit_development_v1/verification.json",
        "analysis_manifest": "outputs/consistent_gain_crossfit_development_v1/analysis_manifest.json",
        "independent_verification": "outputs/consistent_gain_crossfit_development_v1/verification.json",
        "completion_manifest": "outputs/consistent_gain_crossfit_development_v1/completion_manifest.json",
        "previous_stage_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"), "previous_stage_status_sha256": before_sha,
        "previous_set_aware_pilot": {"stage": previous["stage"], "report": previous["report"], "summary": previous["summary"],
            "primary_paired_effect": previous["primary_paired_effect"], "gate_summary": previous["gate_summary"]},
        "cached_diagnostic_scope": {"training_families": 84, "oof_states": 592, "oof_initial_states": 168,
            "checkpoint_families": 6, "viewed_sgdev_families": 24, "new_linear_fits": 5,
            "unified_decision_rows_verified": 576, "all_diagnostic_rows_verified": 1392,
            "oof_guard_uses_calibration_reuse_not_complete_cross_validation": True},
        "additive_margin_pp": s["additive_margin_pp"],
        "oof_consistent_cached_diagnostic": next(r for r in s["summary_rows"] if r["block"] == "train_oof_initial" and r["arm"] == "ridge_gain_consistent"),
        "qualification_thresholds_changed": False,
        "conclusion": "提名与覆盖统一为同一扣余量收益分数，576条一致决策验证通过。OOF原型允许30起点覆盖，缓存规划8有益4有害18持平；SGDEV允许6次覆盖均主指标持平。未建立实际疏散收益。",
        "next_step": "先补值得覆盖预测和校准样本证据，另存小预算方案；不扩大PPO训练，不放松门槛，不以缓存规划收益替代实际执行收益。",
        "research_limits": s["limitations"], "targeted_tests": {"tests_passed": 10, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": sha(REPORT), "summary_sha256": sha(OUT / "summary.json")})
    write(STATUS, status)
    check_completion("set_aware_distillation_development_v1", HISTORY)
    artifacts = [REPORT, ROOT / "reports/统一提名与覆盖收益交叉拟合开发方案_2026-10-03.md", STATUS, HISTORY,
        ROOT / "tests/test_consistent_gain_policy.py"]
    artifacts += [ROOT / "scripts" / name for name in ("consistent_gain_policy.py", "run_consistent_gain_crossfit.py", "verify_consistent_gain_crossfit.py",
        "write_consistent_gain_report.py", "finalize_consistent_gain_crossfit.py")]
    artifacts += [p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json"]
    completion = {"role": "cached_gain_consistency_crossfit_completion_not_actual_efficacy", "created_at_utc": now, "all_checks_passed": True,
        "new_linear_fits": 5, "new_neural_training": 0, "new_ppo_training": 0, "new_simulations": 0, "new_llm_calls": 0,
        "oof_states_verified": 592, "diagnostic_records_verified": 1392, "coherent_decisions_verified": 576,
        "source_inputs_unchanged": len(m["source_input_sha256"]), "targeted_tests_passed": 10,
        "historical_unique_files_checked": len(old_files), "legacy_core_checks_passed": len(previous_package["legacy_core_checks"]),
        "prior_completion_files_checked": prior_counts, "prior_completion_manifest_sha256": sha(PRIOR / "completion_manifest.json"),
        "saved_prior_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"), "saved_prior_status_sha256": before_sha,
        "intentional_live_status_update_with_exact_previous_bytes_saved": True,
        "actual_efficacy_claimed": False, "qualification_thresholds_changed": False,
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in artifacts}}
    write(OUT / "completion_manifest.json", completion)
    for file, expected in completion["files"].items():
        assert sha(ROOT / file) == expected
    print(json.dumps({"completed": True, "files_hashed": len(completion["files"]), "source_inputs_unchanged": len(m["source_input_sha256"]),
        "old_unique_files_checked": len(old_files), "status_history_preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
