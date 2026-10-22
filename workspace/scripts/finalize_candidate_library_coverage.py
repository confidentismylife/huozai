"""Package the fixed full-library audit without changing frozen experiments."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
import matplotlib

from finalize_set_aware_distillation import sha, read, write, check_completion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/candidate_library_coverage_audit_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_candidate_library_coverage_20261003.json"
REPORT = ROOT / "reports/完整候选计划覆盖审计报告_2026-10-03.md"
PROTOCOL = ROOT / "reports/完整候选计划覆盖审计固定方案_2026-10-03.md"


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    assert not (OUT / "completion_manifest.json").exists() and not HISTORY.exists()
    manifest, summary, verification, analysis = (read(OUT / name) for name in
        ("development_manifest.json", "summary.json", "verification.json", "analysis_manifest.json"))
    assert verification["all_checks_passed"]
    assert summary["total_unique_full_rollouts"] == manifest["maximum_full_rollouts"] == 7344
    assert summary["new_fits"] == summary["new_ppo_training"] == summary["new_llm_calls"] == 0
    assert summary["budget"]["planning_rollouts"] == verification["planning_rollouts_verified"] == 4896
    assert summary["budget"]["actual_rollouts"] == verification["actual_rollouts_verified"] == 2448
    assert verification["verifier_sha256"] == sha(ROOT / "scripts/verify_candidate_library_coverage.py")
    assert verification["analysis_manifest_sha256"] == sha(OUT / "analysis_manifest.json")
    assert analysis["runner_sha256"] == sha(ROOT / "scripts/run_candidate_library_coverage.py")
    assert analysis["development_manifest_sha256"] == sha(OUT / "development_manifest.json")
    assert analysis["summary_sha256"] == sha(OUT / "summary.json")
    for filename, expected in analysis["csv_sha256"].items():
        assert sha(OUT / filename) == expected, filename
    for filename, expected in analysis["state_file_sha256"].items():
        assert sha(OUT / "states" / filename) == expected, filename
    for path, expected in manifest["source_input_sha256"].items():
        assert sha(path) == expected, path
    assert sha(STATUS) == manifest["previous_status_sha256"] == sha(OUT / "previous_status_exact_bytes.json")
    assert manifest["prior_completion_sha256"] == sha(ROOT / "outputs/proposal_gain_factorial_development_v1/completion_manifest.json")
    assert not manifest["qualification_thresholds_changed"]
    history_dir = ROOT / "outputs/third_round_status_history"
    counts = {
        "proposal_gain_factorial": check_completion("proposal_gain_factorial_development_v1", STATUS),
        "gain_target_alignment": check_completion("gain_target_alignment_diagnostic_development_v1", history_dir / "before_proposal_gain_factorial_20261003.json"),
        "override_worth": check_completion("override_worth_evidence_development_v1", history_dir / "before_gain_target_alignment_diagnostic_20261003.json"),
        "consistent_gain": check_completion("consistent_gain_crossfit_development_v1", history_dir / "before_override_worth_evidence_20261003.json"),
        "set_aware": check_completion("set_aware_distillation_development_v1", history_dir / "before_consistent_gain_crossfit_20261003.json"),
        "crossed": check_completion("crossed_teacher_rng_diagnostic_development_v1", history_dir / "before_set_aware_distillation_20261003.json"),
        "observable": check_completion("observable_teacher_label_audit_development_v1", history_dir / "before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot": check_completion("teacher_factorial_pilot_development_v1", history_dir / "before_observable_teacher_label_audit_20261003.json"),
        "research_value": check_completion("research_value_quantification_20261003", STATUS)}
    assert counts == {"proposal_gain_factorial": 350, "gain_target_alignment": 46, "override_worth": 140,
                      "consistent_gain": 32, "set_aware": 97, "crossed": 52, "observable": 30, "pilot": 13, "research_value": 9}
    for path, expected in read(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items():
        assert sha(ROOT / path) == expected, path
    historical = read(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique = {Path(r["path"]).resolve(): r["current_sha256"] for r in historical["records"]}
    for path, expected in unique.items():
        effective = history_dir / "before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path
        assert sha(effective) == expected, path
    for artifact in read(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    core = read(ROOT / "outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for entry in core:
        assert sha(entry["file"]) == entry["sha256"]
    assets = read(OUT / "report_assets_manifest.json")
    assert assets["report_sha256"] == sha(REPORT)
    assert assets["writer_sha256"] == sha(ROOT / "scripts/write_candidate_library_coverage_report.py")
    assert assets["summary_sha256"] == sha(OUT / "summary.json")
    assert assets["layout_repair_script_sha256"] == sha(ROOT / "scripts/repair_candidate_library_figure.py")
    for filename, expected in assets["plots_sha256"].items():
        assert sha(OUT / filename) == expected
    for filename, expected in assets["initial_figure_and_writer_preserved"].items():
        assert sha(OUT / "figure_layout_initial" / filename) == expected
    assert assets["legend_moved_outside_data_axes"] and assets["legend_point_overlap_geometry_check_passed"]
    text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", text)
    assert len(links) == 14 and all(Path(path).is_file() for path in links)
    assert all(str(n) in text for n in (7344, 4896, 2448, 1951, 60385, 2539, 237))
    for effect in summary["paired_effects"]:
        if effect["control"] == "pure_ppo_601" and effect["treatment"] != "pure_ppo_601":
            assert effect["treatment"] in text
            assert f"{effect['delta_incap_pp']:+.6f}" in text
            assert f"[{effect['exploratory_ci_low_pp']:.6f}, {effect['exploratory_ci_high_pp']:.6f}]" in text
    primary = summary["primary_paired_effect"]
    assert f"{primary['delta_incap_pp']:+.6f}" in text
    assert primary["treatment"] == "M_point" and primary["control"] == "old_proposal"
    opportunity = summary["opportunity_summary"]
    assert opportunity["states_with_empirical_library_benefit"] == opportunity["missed_by_old_states"] == 4
    assert opportunity["families_with_empirical_library_benefit"] == 2
    assert opportunity["states_with_manual_library_benefit"] == 0
    assert opportunity["R_point_captured_missed_states"] == 3 and opportunity["M_point_captured_missed_states"] == 2
    assert opportunity["states_with_any_zero_baseline_world"] == 18
    assert opportunity["zero_world_states_with_empirical_actual_opportunity"] == 0
    missed = [r for r in table(OUT / "state_opportunity_audit.csv") if r["missed_by_old_proposal"] == "True"]
    lookup = {(r["state_id"], int(r["plan_index"])): r for r in table(OUT / "full_library_plan_evidence.csv")}
    assert len(missed) == 4
    for row in missed:
        evidence = lookup[row["state_id"], int(row["empirical_best_index"])]
        assert row["old_plan"] == "pure_ppo_601" and row["state_id"] in text and evidence["plan"] in text
        for field in ("old_prediction_gain_pp", "actual_gain_pp", "actual_first3_gain_pp", "actual_last3_gain_pp"):
            assert f"{float(evidence[field]):.4f}" in text
    assert "具有模拟特权的离线教师" in text and "区间上端为0" in text
    assert "没有直接解释本次4个遗漏" in text and "R_margin选1个非601计划但主指标持平" in text
    tests = {"command": "D:/miniconda3/python.exe -m pytest -q tests/test_candidate_library_scores.py",
             "tests_passed": 8, "exit_code": 0, "observed_result": "8 passed in 0.16s",
             "observed_before_prepare_and_simulation": True, "full_project_tests_rerun": False,
             "test_sha256": sha(ROOT / "tests/test_candidate_library_scores.py")}
    write(OUT / "targeted_test_verification.json", tests)
    write(OUT / "packaging_verification.json", {"all_checks_passed": True, "prior_completion_files_checked": counts,
        "historical_snapshot_records_checked": len(historical["records"]), "historical_unique_files_checked": len(unique),
        "historical_report_md_docx_pdf_unchanged": True, "readme_unchanged_against_pilot_hash": True,
        "legacy_core_checks_passed": len(core), "valid_local_report_links": len(links),
        "displayed_plan_gains_and_arm_effects_checked": True, "figure_visually_inspected": True,
        "initial_legend_overlap_preserved_and_fixed": True, "essential_figure_data_clipped": False,
        "all_analysis_inputs_unchanged_since_independent_verification": True,
        "new_full_rollouts_in_packaging": 0, "new_fits": 0, "new_llm_calls": 0,
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "finalizer_sha256": sha(__file__)})
    old, prior_sha = read(STATUS), sha(STATUS)
    HISTORY.write_bytes(STATUS.read_bytes())
    assert sha(HISTORY) == prior_sha
    now = datetime.now(timezone.utc).isoformat()
    old_results = {key: old[key] for key in ("stage", "report", "summary", "manifest", "diagnostic_scope",
                  "primary_paired_effect", "prediction_quality", "gate_summary", "margins_pp", "conclusion", "next_step")}
    status = {"stage": "candidate_library_coverage_audit_complete_four_missed_starts_two_families_no_margin_gain",
        "updated_date": "2026-10-03", "updated_at_utc": now, "freeze_allowed": False,
        "hard_shield_qualification_passed": False, "formal_round3_training_started": False,
        "formal_round3_final_bank_generated": False, "new_simulations": 7344, "new_planning_rollouts": 4896,
        "new_evaluation_rollouts": 2448, "new_fits": 0, "new_supervised_training_runs": 0,
        "new_lightweight_linear_fits": 0, "new_lightweight_kernel_fits": 0, "new_lightweight_classifier_fits": 0,
        "new_supervised_neural_training_runs": 0, "new_ppo_training_seeds": 0, "new_llm_calls": 0,
        "actual_control_efficacy_evaluated_in_current_stage": True,
        "observable_student_closed_loop_advantage_established": False, "teacher_generalization_advantage_established": False,
        "llm_control_contribution_established": False, "qualification_thresholds_changed": False,
        "report": REPORT.relative_to(ROOT).as_posix(), "summary": (OUT / "summary.json").relative_to(ROOT).as_posix(),
        "manifest": (OUT / "development_manifest.json").relative_to(ROOT).as_posix(),
        "integrity_audit": (OUT / "verification.json").relative_to(ROOT).as_posix(),
        "analysis_manifest": (OUT / "analysis_manifest.json").relative_to(ROOT).as_posix(),
        "independent_verification": (OUT / "verification.json").relative_to(ROOT).as_posix(),
        "completion_manifest": (OUT / "completion_manifest.json").relative_to(ROOT).as_posix(),
        "previous_stage_status": HISTORY.relative_to(ROOT).as_posix(), "previous_stage_status_sha256": prior_sha,
        "previous_proposal_gain_factorial": old_results,
        "diagnostic_scope": {"viewed_development_families": 12, "reused_population_states": 24,
            "complete_existing_plans": 17, "manual_plans": 12, "frozen_PPO_plans": 5,
            "source_worlds": 2, "new_planning_futures": 6, "independent_new_actual_futures": 6,
            "source_inputs_unchanged": 1951, "actual_trace_steps": 60385,
            "not_final_confirmation_or_deployment_prevalence": True},
        "primary_paired_effect": primary, "opportunity_summary": opportunity,
        "arm_summary": summary["arm_summary"], "retained_margin_pp": 100 * manifest["prior_margin"],
        "retained_margin_vs601_incapacitation_improvement_observed": False,
        "conclusion": "固定17候选审计发现2家庭4起点有益计划，旧提案全部漏掉；均来自其他冻结PPO种子。先选后验R捕捉3起点、M捕捉2起点。M相对旧提案改善0.095486个百分点，探索区间触及0；保留现行余量后相对601没有失能改善。尚无学生选择、LLM独立贡献或稳定泛化优势。",
        "next_step": "先固定601—605五个冻结PPO的常驻候选集合，用缓存审计机会保留与全候选标签接口，纳入旧提案为601的场景；另存新开发协议后再讨论排名训练和独立新家庭验证。暂缓增加PPO训练量，保留当前门槛。",
        "research_limits": summary["limits"], "targeted_tests": {"tests_passed": 8, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": sha(REPORT), "summary_sha256": sha(OUT / "summary.json")}
    write(STATUS, status)
    check_completion("proposal_gain_factorial_development_v1", HISTORY)
    artifacts = [REPORT, PROTOCOL, STATUS, HISTORY, ROOT / "tests/test_candidate_library_scores.py"]
    artifacts += [ROOT / "scripts" / name for name in ("candidate_library_scores.py", "run_candidate_library_coverage.py",
        "verify_candidate_library_coverage.py", "write_candidate_library_coverage_report.py",
        "repair_candidate_library_figure.py", "finalize_candidate_library_coverage.py")]
    artifacts += sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest.json")
    completion = {"role": "fixed_viewed_development_full17_candidate_coverage_audit_not_final_qualification",
        "all_checks_passed": True, "created_at_utc": now, "new_full_rollouts": 7344,
        "new_planning_rollouts": 4896, "new_actual_rollouts": 2448, "new_fits": 0,
        "new_ppo_training": 0, "new_llm_calls": 0, "source_inputs_unchanged": 1951,
        "targeted_tests_passed": 8, "actual_trace_steps_verified": 60385,
        "prior_completion_files_checked": counts, "historical_unique_files_checked": len(unique),
        "legacy_core_checks_passed": len(core), "prior_completion_manifest_sha256": manifest["prior_completion_sha256"],
        "saved_prior_status": HISTORY.relative_to(ROOT).as_posix(), "saved_prior_status_sha256": prior_sha,
        "intentional_status_update_previous_exact_bytes_saved": True,
        "four_missed_starts_and_two_families_reported": True, "benefits_from_other_PPO_seeds_not_LLM_reported": True,
        "R_M_simulator_privilege_and_hindsight_optimism_reported": True,
        "retained_margin_no_incapacitation_gain_vs601_reported": True,
        "initial_figure_layout_issue_and_originals_preserved": True,
        "old_sources_models_reports_preserved": True, "qualification_thresholds_changed": False,
        "files": {path.relative_to(ROOT).as_posix(): sha(path) for path in artifacts}}
    write(OUT / "completion_manifest.json", completion)
    for path, expected in read(OUT / "completion_manifest.json")["files"].items():
        assert sha(ROOT / path) == expected, path
    print(json.dumps({"completed": True, "files_hashed": len(completion["files"]),
        "source_inputs_unchanged": 1951, "prior_completion_files_checked": counts,
        "historical_unique_files_checked": len(unique), "legacy_core_checks_passed": len(core),
        "status_history_preserved": True, "new_fits": 0}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
