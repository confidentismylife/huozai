"""Package the fresh family experiment and preserve every prior stage."""
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import csv,json,re,sys
import numpy as np
from finalize_set_aware_distillation import read,sha,write,check_completion

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT / "outputs/five_ppo_fresh_family_student_development_v1"
STATUS=ROOT / "outputs/third_round_development_status.json"
HISTORY=ROOT / "outputs/third_round_status_history/before_five_ppo_fresh_student_20261004.json"
REPORT=ROOT / "reports/五PPO全新家庭轻量选择实验报告_2026-10-04.md"
PROTOCOL=ROOT / "reports/五PPO全新家庭轻量选择实验固定方案_2026-10-04.md"
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    assert not (OUT / "completion_manifest.json").exists() and not HISTORY.exists()
    m,s,v,am,assets,d=(read(OUT / p) for p in ("development_manifest.json","summary.json","verification.json","analysis_manifest.json","report_assets_manifest.json","miss_diagnostic.json"))
    assert v["all_checks_passed"] and s["total_unique_full_rollouts"]==m["maximum_full_rollouts"]==9360
    assert s["budget"]["planning_rollouts"]==7200 and s["budget"]["actual_rollouts"]==2160
    assert s["new_matrix_model_fits"]==4 and s["new_training_mean_references"]==2
    assert s["new_ppo_training"]==s["new_llm_calls"]==0 and not m["qualification_thresholds_changed"]
    assert v["verifier_sha256"]==sha(ROOT / "scripts/verify_five_ppo_fresh_student.py") and v["analysis_manifest_sha256"]==sha(OUT / "analysis_manifest.json")
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    assert len(m["source_input_sha256"])==2124 and sha(STATUS)==m["previous_status_sha256"]==sha(OUT / "previous_status_exact_bytes.json")
    collection=read(OUT / "collection_complete.json")
    for p,h in collection["family_file_sha256"].items():assert sha(OUT / p)==h
    assert am["collection_complete_sha256"]==sha(OUT / "collection_complete.json") and am["summary_sha256"]==sha(OUT / "summary.json")
    for p,h in am["csv_sha256"].items():assert sha(OUT / p)==h
    hd=ROOT / "outputs/third_round_status_history"
    counts={
        "five_candidate_interface":check_completion("ppo_candidate_label_interface_development_v1",STATUS),
        "candidate_library_coverage":check_completion("candidate_library_coverage_audit_development_v1",hd / "before_ppo_candidate_label_interface_20261003.json"),
        "proposal_gain_factorial":check_completion("proposal_gain_factorial_development_v1",hd / "before_candidate_library_coverage_20261003.json"),
        "gain_target_alignment":check_completion("gain_target_alignment_diagnostic_development_v1",hd / "before_proposal_gain_factorial_20261003.json"),
        "override_worth":check_completion("override_worth_evidence_development_v1",hd / "before_gain_target_alignment_diagnostic_20261003.json"),
        "consistent_gain":check_completion("consistent_gain_crossfit_development_v1",hd / "before_override_worth_evidence_20261003.json"),
        "set_aware":check_completion("set_aware_distillation_development_v1",hd / "before_consistent_gain_crossfit_20261003.json"),
        "crossed":check_completion("crossed_teacher_rng_diagnostic_development_v1",hd / "before_set_aware_distillation_20261003.json"),
        "observable":check_completion("observable_teacher_label_audit_development_v1",hd / "before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot":check_completion("teacher_factorial_pilot_development_v1",hd / "before_observable_teacher_label_audit_20261003.json"),
        "research_value":check_completion("research_value_quantification_20261003",STATUS)}
    assert counts["five_candidate_interface"]==40 and counts["candidate_library_coverage"]==133 and counts["proposal_gain_factorial"]==350
    assert m["prior_completion_sha256"]==sha(ROOT / "outputs/ppo_candidate_label_interface_development_v1/completion_manifest.json")
    for p,h in read(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items():assert sha(ROOT / p)==h
    historic=read(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique={Path(r["path"]).resolve():r["current_sha256"] for r in historic["records"]}
    for p,h in unique.items():assert sha(hd / "before_teacher_diagnosis_and_factorial_pilot.json" if p==STATUS else p)==h
    for artifact in read(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:assert sha(artifact["path"])==artifact["sha256"]
    core=read(ROOT / "outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for row in core:assert sha(row["file"])==row["sha256"]
    assert assets["report_sha256"]==sha(REPORT) and assets["summary_sha256"]==sha(OUT / "summary.json")
    assert assets["writer_sha256"]==sha(ROOT / "scripts/write_five_ppo_fresh_student_report.py") and assets["miss_diagnostic_sha256"]==sha(OUT / "miss_diagnostic.json")
    for p,h in d["plots_sha256"].items():assert sha(OUT / p)==h
    assert d["script_sha256"]==sha(ROOT / "scripts/diagnose_five_ppo_fresh_student.py")
    text=REPORT.read_text(encoding="utf-8");links=re.findall(r"\]\((C:/[^)]+)\)",text)
    assert len(links)==assets["links_including_image"]==23 and all(Path(p).is_file() for p in links)
    for row in s["all_paired_effects"]:
        if row["control"]=="pure_ppo_601":assert row["treatment"] in text and f"{row['delta_incap_pp']:+.6f}" in text
    for row in s["prediction_quality"]:
        if row["block"]=="test":
            for k in ("R_MAE_pp","M_MAE_pp","actual_MAE_pp"):assert f"{row[k]:.6f}" in text
    for value in s["margins_pp"].values():assert f"{value:.6f}" in text
    assert "新的“汇总人口后”经验估计对象" in text and "不是运行中逐步切换策略" in text
    assert "−0.300926" in text and "1变差" in text and "0/9召回是可估计的0" in text
    assert all(str(n) in text for n in (9360,7200,2160,55940,3529,308,2124,756,1512))
    # Check summary arm means and class counts again against independently verified CSVs.
    familyrows=table(OUT / "arm_family_results.csv");poprows=table(OUT / "arm_population_results.csv")
    for row in s["arm_summary"]:
        records=[r for r in familyrows if r["block"]==row["block"] and r["arm"]==row["arm"]]
        pops=[r for r in poprows if r["block"]==row["block"] and r["arm"]==row["arm"]]
        assert row["override_families"]==sum(r["override"]=="True" for r in records)
        assert row["actual_family_classes"]==dict(Counter(r["actual_class"] for r in records))
        assert row["actual_population_classes"]==dict(Counter(r["actual_class"] for r in pops))
        for metric in m["outcome_fields"][:6]:np.testing.assert_allclose(row[metric],np.mean([float(r[metric]) for r in records]),atol=1e-10,rtol=0)
    startup=ROOT / "artifacts/five_ppo_fresh_startup_record_20261004.json";failure=read(startup)
    assert failure["run_exit_code"]==1 and failure["new_simulations_in_failed_invocation"]==failure["new_model_fits_in_failed_invocation"]==0
    tests={"command":"D:/miniconda3/python.exe -m pytest -q tests/test_five_ppo_student.py","tests_passed":8,"exit_code":0,
        "observed_result":"8 passed in 0.92s","observed_before_prepare_and_training":True,"full_project_tests_rerun":False,
        "test_sha256":sha(ROOT / "tests/test_five_ppo_student.py")}
    write(OUT / "targeted_test_verification.json",tests)
    write(OUT / "packaging_verification.json",{"all_checks_passed":True,"prior_completion_files_checked":counts,
        "historical_unique_files_checked":len(unique),"historical_snapshot_records_checked":len(historic["records"]),
        "legacy_core_checks_passed":len(core),"historical_reports_and_readme_unchanged":True,"valid_report_links":len(links),
        "all21_arms_all7_MAE_references_and_margins_checked":True,"mixed_ungated_signal_and_qualification_failure_reported":True,
        "figure_visually_inspected":True,"essential_figure_data_clipped":False,"startup_failure_preserved":True,
        "new_simulations_in_packaging":0,"new_fits_in_packaging":0,"finalizer_sha256":sha(__file__)})
    old,before_sha=read(STATUS),sha(STATUS);HISTORY.write_bytes(STATUS.read_bytes());assert sha(HISTORY)==before_sha
    now=datetime.now(timezone.utc).isoformat()
    previous={key:old[key] for key in ("stage","report","summary","manifest","diagnostic_scope","label_interface_counts","opportunity_preservation","input_label_audit","five_vs_full17","conclusion","next_step")}
    status={"stage":"fresh_five_PPO_student_complete_no_margin_advantage_mixed_ungated_signal",
        "updated_date":"2026-10-04","updated_at_utc":now,"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "formal_round3_training_started":False,"formal_round3_final_bank_generated":False,
        "new_simulations":9360,"new_planning_rollouts":7200,"new_evaluation_rollouts":2160,"new_fits":4,
        "new_supervised_training_runs":4,"new_lightweight_linear_fits":2,"new_lightweight_kernel_fits":2,
        "new_training_mean_references":2,"new_lightweight_classifier_fits":0,"new_supervised_neural_training_runs":0,
        "new_ppo_training_seeds":0,"new_llm_calls":0,"observable_student_closed_loop_advantage_established":False,
        "teacher_generalization_advantage_established":False,"llm_control_contribution_established":False,
        "qualification_thresholds_changed":False,"actual_control_efficacy_evaluated_in_current_stage":True,
        "report":REPORT.relative_to(ROOT).as_posix(),"summary":(OUT / "summary.json").relative_to(ROOT).as_posix(),
        "manifest":(OUT / "development_manifest.json").relative_to(ROOT).as_posix(),
        "integrity_audit":(OUT / "verification.json").relative_to(ROOT).as_posix(),
        "analysis_manifest":(OUT / "analysis_manifest.json").relative_to(ROOT).as_posix(),
        "independent_verification":(OUT / "verification.json").relative_to(ROOT).as_posix(),
        "completion_manifest":(OUT / "completion_manifest.json").relative_to(ROOT).as_posix(),
        "previous_stage_status":HISTORY.relative_to(ROOT).as_posix(),"previous_stage_status_sha256":before_sha,
        "previous_five_candidate_interface":previous,
        "diagnostic_scope":{"new_train_families":24,"new_calibration_families":12,"new_holdout_families":24,
            "population_starts":120,"frozen_PPO_candidates":5,"models":4,"constant_score_references":2,"arms":21,
            "candidate_R_M_label_records":600,"actual_trace_steps":55940,"frozen_inputs_unchanged":2124,
            "banks_fresh_at_freeze_now_viewed_development":True,"population_pool_before_world_scoring":True,
            "selector_only_at_initial_state":True,"not_new_LLМ_dynamic_correction":True},
        "primary_paired_effect":s["primary_paired_effect"],"opportunity_summary":s["opportunity_summary"],
        "prediction_quality":s["prediction_quality"],"arm_summary":s["arm_summary"],"margins_pp":s["margins_pp"],
        "miss_diagnostic":{"actual_beneficial_candidates":9,"M_target_nonpositive_actual_positive":6,
            "M_target_positive_kernel_missed":3,"positive_target_misses_all_in_family":"F5TE0018",
            "nearest_training_family":"F5TR0006","post_outcome_diagnostic_only":True},
        "conclusion":"五候选新数据仍有4留出家庭9个有益候选，但4学生及2均值余量组均24/24回退601，主差值0。R线性无余量点估计改善0.300926个百分点，区间包含0且1家庭变差；R核无余量点估计变差0.041956。全部学生实际收益MAE不如零预测。尚无稳定学生或LLM贡献。",
        "next_step":"先用已冻结缓存审计F5TE0018与F5TR0006的可观测信息和源世界收益反转，区分目标差异、隐藏时序/人口/未来噪声与学生平滑误差。暂缓追加学生/PPO训练；本轮留出已查看，修改输入/目标需另存协议和新的独立银行，不用现有正例调余量。",
        "research_limits":s["limits"],"targeted_tests":{"tests_passed":8,"exit_code":0,"full_project_tests_rerun":False},
        "report_sha256":sha(REPORT),"summary_sha256":sha(OUT / "summary.json")}
    write(STATUS,status);check_completion("ppo_candidate_label_interface_development_v1",HISTORY)
    artifacts=[REPORT,PROTOCOL,STATUS,HISTORY,startup,ROOT / "tests/test_five_ppo_student.py"]
    artifacts += [ROOT / "scripts" / name for name in ("five_ppo_student.py","run_five_ppo_fresh_student.py","analyze_five_ppo_fresh_student.py",
        "diagnose_five_ppo_fresh_student.py","verify_five_ppo_fresh_student.py","write_five_ppo_fresh_student_report.py","finalize_five_ppo_fresh_student.py")]
    artifacts += sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="completion_manifest.json")
    completion={"role":"fresh_family_pooled_five_PPO_student_development_completion_not_formal_final_qualification",
        "all_checks_passed":True,"created_at_utc":now,"new_full_rollouts":9360,"new_planning_rollouts":7200,"new_actual_rollouts":2160,
        "new_matrix_model_fits":4,"new_training_mean_references":2,"new_ppo_training":0,"new_llm_calls":0,
        "source_inputs_unchanged":2124,"targeted_tests_passed":8,"actual_trace_steps_verified":55940,
        "prior_completion_files_checked":counts,"historical_unique_files_checked":len(unique),"legacy_core_checks_passed":len(core),
        "prior_completion_manifest_sha256":m["prior_completion_sha256"],"saved_prior_status":HISTORY.relative_to(ROOT).as_posix(),"saved_prior_status_sha256":before_sha,
        "intentional_status_update_previous_exact_bytes_saved":True,"new_pooling_estimand_and_initial_only_scope_reported":True,
        "all_guarded_student_fallback_and_mixed_unguarded_signal_reported":True,
        "actual_MAE_zero_reference_superiority_reported":True,"source_world_conflict_and_student_miss_separated":True,
        "startup_sequence_failure_preserved_no_lost_simulations":True,"old_sources_models_reports_preserved":True,
        "qualification_thresholds_changed":False,"files":{p.relative_to(ROOT).as_posix():sha(p) for p in artifacts}}
    write(OUT / "completion_manifest.json",completion)
    for p,h in read(OUT / "completion_manifest.json")["files"].items():assert sha(ROOT / p)==h
    print(json.dumps({"completed":True,"files_hashed":len(completion["files"]),"source_inputs_unchanged":2124,"prior_completion_files_checked":counts,
        "new_rollouts":9360,"new_matrix_fits":4,"new_ppo_training":0,"status_history_preserved":True},ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
