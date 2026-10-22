"""Preserve the full audit chain and package the cache-only interface follow-up."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sys
import numpy as np
from finalize_set_aware_distillation import read, sha, write, check_completion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ppo_candidate_label_interface_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_ppo_candidate_label_interface_20261003.json"
REPORT = ROOT / "reports/五PPO候选集合与全候选标签接口审计报告_2026-10-03.md"
PROTOCOL = ROOT / "reports/五PPO候选集合与全候选标签接口审计方案_2026-10-03.md"


def main():
    assert not (OUT / "completion_manifest.json").exists() and not HISTORY.exists()
    m, s, v, am, assets = (read(OUT / p) for p in
        ("development_manifest.json", "summary.json", "verification.json", "analysis_manifest.json", "report_assets_manifest.json"))
    assert v["all_checks_passed"] and v["source_inputs_unchanged"] == len(m["source_input_sha256"]) == 2088
    assert all(s[key] == 0 for key in ("new_simulations", "new_fits", "new_ppo_training", "new_llm_calls"))
    assert v["verifier_sha256"] == sha(ROOT / "scripts/verify_candidate_label_interface_audit.py")
    assert v["analysis_manifest_sha256"] == sha(OUT / "analysis_manifest.json")
    for name, expected in am["files_sha256"].items(): assert sha(OUT / name) == expected
    for path, expected in m["source_input_sha256"].items(): assert sha(path) == expected, path
    assert sha(STATUS) == m["previous_status_sha256"] == sha(OUT / "previous_status_exact_bytes.json")
    prior = ROOT / "outputs/candidate_library_coverage_audit_development_v1/completion_manifest.json"
    assert sha(prior) == m["prior_completion_sha256"]
    hd = ROOT / "outputs/third_round_status_history"
    counts = {
        "candidate_library_coverage": check_completion("candidate_library_coverage_audit_development_v1", STATUS),
        "proposal_gain_factorial": check_completion("proposal_gain_factorial_development_v1", hd / "before_candidate_library_coverage_20261003.json"),
        "gain_target_alignment": check_completion("gain_target_alignment_diagnostic_development_v1", hd / "before_proposal_gain_factorial_20261003.json"),
        "override_worth": check_completion("override_worth_evidence_development_v1", hd / "before_gain_target_alignment_diagnostic_20261003.json"),
        "consistent_gain": check_completion("consistent_gain_crossfit_development_v1", hd / "before_override_worth_evidence_20261003.json"),
        "set_aware": check_completion("set_aware_distillation_development_v1", hd / "before_consistent_gain_crossfit_20261003.json"),
        "crossed": check_completion("crossed_teacher_rng_diagnostic_development_v1", hd / "before_set_aware_distillation_20261003.json"),
        "observable": check_completion("observable_teacher_label_audit_development_v1", hd / "before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot": check_completion("teacher_factorial_pilot_development_v1", hd / "before_observable_teacher_label_audit_20261003.json"),
        "research_value": check_completion("research_value_quantification_20261003", STATUS)}
    assert counts["candidate_library_coverage"] == 133 and counts["proposal_gain_factorial"] == 350
    for path, expected in read(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items():
        assert sha(ROOT / path) == expected
    historical = read(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique = {Path(r["path"]).resolve(): r["current_sha256"] for r in historical["records"]}
    for path, expected in unique.items():
        effective = hd / "before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path
        assert sha(effective) == expected
    for artifact in read(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    core = read(ROOT / "outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for row in core: assert sha(row["file"]) == row["sha256"]
    assert assets["report_sha256"] == sha(REPORT) and assets["summary_sha256"] == sha(OUT / "summary.json")
    assert assets["writer_sha256"] == sha(ROOT / "scripts/write_candidate_label_interface_report.py")
    text = REPORT.read_text(encoding="utf-8"); links = re.findall(r"\]\((C:/[^)]+)\)", text)
    assert len(links) == assets["local_links"] == 18 and all(Path(path).is_file() for path in links)
    for row in s["arm_summary"]:
        assert row["arm"] in text and f"{100*row['mean_delta_final_incapacitation_vs601']:+.6f}" in text
    for row in s["five_vs_full17"]:
        for metric in ("final_fed", "additional_congestion", "steps"):
            assert f"{row['mean_five_minus_full17_' + metric]:+.6f}" in text
    for target in ("R", "M"):
        q = s["input_label_audit"]["finite_empirical_fit_lower_bounds"][target]
        assert f"{q['finite_label_minimum_MAE_pp']:.6f}" in text and f"{q['finite_label_minimum_RMSE_pp']:.6f}" in text
    assert "120行不等于120个独立场景" in text and "不能重新称为独立复现" in text
    assert "新增模拟0" in text and "5失败5通过" in text and "源快照" in text
    failure_dir = ROOT / "artifacts/candidate_label_interface_test_repair_20261003"
    assert "5 failed, 5 passed" in (failure_dir / "initial_test_failure.txt").read_text(encoding="utf-8")
    repair = read(OUT / "snapshot_name_collision_note.json")
    for row in repair["records"]:
        assert sha(row["runtime_source"]) == sha(row["correct_runtime_snapshot"]) == row["runtime_sha256"]
        assert sha(row["original_root_snapshot"]) == sha(failure_dir / Path(row["original_root_snapshot"]).name)
    tests = {"command": "D:/miniconda3/python.exe -m pytest -q tests/test_candidate_label_interface.py",
        "tests_passed": 10, "exit_code": 0, "observed_result": "10 passed in 0.31s",
        "observed_before_prepare_and_cache_analysis": True, "full_project_tests_rerun": False,
        "initial_failed_revision_and_output_preserved": True, "initial_result": "5 failed, 5 passed",
        "test_sha256": sha(ROOT / "tests/test_candidate_label_interface.py")}
    write(OUT / "targeted_test_verification.json", tests)
    write(OUT / "packaging_verification.json", {"all_checks_passed": True,
        "prior_completion_files_checked": counts, "historical_snapshot_records_checked": len(historical["records"]),
        "historical_unique_files_checked": len(unique), "legacy_core_checks_passed": len(core),
        "historical_reports_and_readme_unchanged": True, "valid_local_report_links": len(links),
        "displayed_primary_and_secondary_metrics_checked": True, "same_input_finite_fit_bounds_checked": True,
        "failure_revisions_and_snapshot_resolution_preserved": True,
        "new_simulations": 0, "new_fits": 0, "finalizer_sha256": sha(__file__)})
    previous, before_sha = read(STATUS), sha(STATUS); HISTORY.write_bytes(STATUS.read_bytes()); assert sha(HISTORY) == before_sha
    now = datetime.now(timezone.utc).isoformat()
    previous_results = {key: previous[key] for key in ("stage", "report", "summary", "manifest", "diagnostic_scope",
        "primary_paired_effect", "opportunity_summary", "arm_summary", "retained_margin_pp", "conclusion", "next_step")}
    status = {"stage": "five_PPO_cached_coverage_and_all_candidate_label_interface_complete_no_new_training",
        "updated_date": "2026-10-03", "updated_at_utc": now, "freeze_allowed": False,
        "hard_shield_qualification_passed": False, "formal_round3_training_started": False, "formal_round3_final_bank_generated": False,
        "new_simulations": 0, "new_planning_rollouts": 0, "new_evaluation_rollouts": 0,
        "new_fits": 0, "new_supervised_training_runs": 0, "new_lightweight_linear_fits": 0,
        "new_lightweight_kernel_fits": 0, "new_lightweight_classifier_fits": 0, "new_supervised_neural_training_runs": 0,
        "new_ppo_training_seeds": 0, "new_llm_calls": 0,
        "existing_cached_control_results_reanalyzed": True, "actual_control_efficacy_evaluated_in_current_stage": False,
        "observable_student_closed_loop_advantage_established": False, "teacher_generalization_advantage_established": False,
        "llm_control_contribution_established": False, "qualification_thresholds_changed": False,
        "report": REPORT.relative_to(ROOT).as_posix(), "summary": (OUT / "summary.json").relative_to(ROOT).as_posix(),
        "manifest": (OUT / "development_manifest.json").relative_to(ROOT).as_posix(),
        "integrity_audit": (OUT / "verification.json").relative_to(ROOT).as_posix(),
        "analysis_manifest": (OUT / "analysis_manifest.json").relative_to(ROOT).as_posix(),
        "independent_verification": (OUT / "verification.json").relative_to(ROOT).as_posix(),
        "completion_manifest": (OUT / "completion_manifest.json").relative_to(ROOT).as_posix(),
        "previous_stage_status": HISTORY.relative_to(ROOT).as_posix(), "previous_stage_status_sha256": before_sha,
        "previous_candidate_library_audit": previous_results,
        "diagnostic_scope": {"viewed_development_families": 12, "reused_population_states": 24,
            "fixed_frozen_PPO_candidates": 5, "planning_label_rows": 120,
            "actual_diagnostic_rows": 120, "cached_arm_state_records": 192, "frozen_inputs_unchanged": 2088,
            "viewed_label_dataset_role": m["label_dataset_role"], "new_fit_of_this_dataset_allowed": False,
            "reused_prior_total_rollouts": 7344, "no_new_independent_replication": True},
        "label_interface_counts": s["label_interface_counts"], "opportunity_preservation": s["opportunity_preservation"],
        "input_label_audit": s["input_label_audit"], "five_vs_full17": s["five_vs_full17"],
        "counterfactual_budget_arithmetic": s["counterfactual_budget_arithmetic"], "retained_margin_pp": 100 * m["prior_margin"],
        "conclusion": "缓存检查显示固定601—605保留完整17库的4个失能收益起点；新接口导出120条规划标签，保留旧601的12起点60行。仍无现行余量覆盖收益，存在相同观测标签冲突；没有新增模型拟合、模拟或LLM证据。",
        "next_step": "另存全新家庭训练/校准/留出方案，接入固定五候选全标签接口，先估计可观测条件收益与不确定性，再做小规模轻量排名对照。现有GTEST只作开发诊断，不以当前4正例调门槛，不扩大PPO训练量。",
        "research_limits": s["limits"], "targeted_tests": {"tests_passed": 10, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": sha(REPORT), "summary_sha256": sha(OUT / "summary.json")}
    write(STATUS, status); check_completion("candidate_library_coverage_audit_development_v1", HISTORY)
    artifacts = [REPORT, PROTOCOL, STATUS, HISTORY, ROOT / "tests/test_candidate_label_interface.py"]
    artifacts += [ROOT / "scripts" / name for name in ("candidate_label_interface.py", "run_candidate_label_interface_audit.py",
        "verify_candidate_label_interface_audit.py", "write_candidate_label_interface_report.py", "finalize_candidate_label_interface_audit.py")]
    artifacts += sorted(p for p in failure_dir.iterdir() if p.is_file())
    artifacts += sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json")
    completion = {"role": "cache_only_five_PPO_full_candidate_label_interface_development_completion",
        "all_checks_passed": True, "created_at_utc": now, "new_simulations": 0, "new_fits": 0,
        "new_ppo_training": 0, "new_llm_calls": 0, "source_inputs_unchanged": 2088,
        "targeted_tests_passed": 10, "planning_label_rows_verified": 120, "actual_diagnostic_rows_verified": 120,
        "cached_arm_state_records_verified": 192, "prior_completion_files_checked": counts,
        "historical_unique_files_checked": len(unique), "legacy_core_checks_passed": len(core),
        "prior_completion_manifest_sha256": m["prior_completion_sha256"],
        "saved_prior_status": HISTORY.relative_to(ROOT).as_posix(), "saved_prior_status_sha256": before_sha,
        "intentional_status_update_previous_exact_bytes_saved": True,
        "all_old601_states_retained_in_planning_labels": True, "diagnostic_dataset_training_rejected": True,
        "cache_reuse_not_new_confirmation_reported": True, "secondary_metric_tradeoffs_reported": True,
        "initial_test_failures_and_snapshot_repair_preserved": True, "old_sources_models_reports_preserved": True,
        "qualification_thresholds_changed": False, "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}}
    write(OUT / "completion_manifest.json", completion)
    for path, expected in read(OUT / "completion_manifest.json")["files"].items(): assert sha(ROOT / path) == expected
    print(json.dumps({"completed": True, "files_hashed": len(completion["files"]), "source_inputs_unchanged": 2088,
        "prior_completion_files_checked": counts, "new_simulations": 0, "new_fits": 0,
        "status_history_preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
