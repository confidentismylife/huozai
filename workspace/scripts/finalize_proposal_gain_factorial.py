"""Package the four-cell experiment, preserving all earlier frozen evidence."""
from datetime import datetime,timezone
from pathlib import Path
import json,re,sys
import numpy as np
import matplotlib
from finalize_set_aware_distillation import sha,read,write,check_completion

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/proposal_gain_factorial_development_v1"
STATUS=ROOT/"outputs/third_round_development_status.json"
HISTORY=ROOT/"outputs/third_round_status_history/before_proposal_gain_factorial_20261003.json"
REPORT=ROOT/"reports/收益标签与轻量模型四组对照实验报告_2026-10-03.md"


def main():
    assert not (OUT/"completion_manifest.json").exists() and not HISTORY.exists()
    m,s,v=(read(OUT/p) for p in ("development_manifest.json","summary.json","verification.json"))
    assert v["all_checks_passed"] and s["new_fits"]==4 and s["total_unique_full_rollouts"]==6180
    assert s["new_ppo_training"]==s["new_llm_calls"]==s["new_deep_network_training"]==0
    assert v["verifier_sha256"]==sha(ROOT/"scripts/verify_proposal_gain_factorial.py")
    assert v["analysis_manifest_sha256"]==sha(OUT/"analysis_manifest.json")
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    assert sha(STATUS)==m["previous_status_sha256"]==sha(OUT/"previous_status_exact_bytes.json")
    counts={
        "gain_target_alignment":check_completion("gain_target_alignment_diagnostic_development_v1",STATUS),
        "override_worth":check_completion("override_worth_evidence_development_v1",ROOT/"outputs/third_round_status_history/before_gain_target_alignment_diagnostic_20261003.json"),
        "consistent_gain":check_completion("consistent_gain_crossfit_development_v1",ROOT/"outputs/third_round_status_history/before_override_worth_evidence_20261003.json"),
        "set_aware":check_completion("set_aware_distillation_development_v1",ROOT/"outputs/third_round_status_history/before_consistent_gain_crossfit_20261003.json"),
        "crossed":check_completion("crossed_teacher_rng_diagnostic_development_v1",ROOT/"outputs/third_round_status_history/before_set_aware_distillation_20261003.json"),
        "observable":check_completion("observable_teacher_label_audit_development_v1",ROOT/"outputs/third_round_status_history/before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot":check_completion("teacher_factorial_pilot_development_v1",ROOT/"outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"),
        "research_value":check_completion("research_value_quantification_20261003",STATUS)}
    for p,h in read(ROOT/"outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items():assert sha(ROOT/p)==h
    historical=read(ROOT/"artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique={Path(r["path"]).resolve():r["current_sha256"] for r in historical["records"]}
    for p,h in unique.items():
        effective=ROOT/"outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if p==STATUS else p
        assert sha(effective)==h,p
    for a in read(ROOT/"artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:assert sha(a["path"])==a["sha256"]
    core=read(ROOT/"outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for e in core:assert sha(e["file"])==e["sha256"]
    assets=read(OUT/"report_assets_manifest.json")
    assert assets["report_sha256"]==sha(REPORT) and assets["writer_sha256"]==sha(ROOT/"scripts/write_proposal_gain_factorial_report.py")
    assert assets["summary_sha256"]==sha(OUT/"summary.json")
    for p,h in assets["plots_sha256"].items():assert sha(OUT/p)==h
    text=REPORT.read_text(encoding="utf-8");links=re.findall(r"\]\((C:/[^)]+)\)",text)
    assert len(links)==14 and all(Path(p).is_file() for p in links)
    assert all(str(n) in text for n in (6180,5160,1020,312,1872,1440,1601,25296,2606))
    assert "没有有益提案" in text and "指标记为不可估计" in text and "零预测参考" in text
    assert s["test_proposal_actual_classes"]=={"neutral_or_small":26,"harmful":2}
    assert s["primary_paired_effect"]["delta_incap_pp"]==0
    # Verify every displayed MAE in the six-model table against structured results.
    for q in s["prediction_quality"]:
        if q["scope"]=="all_proposals" and q["block"] in ("calibration","test"):
            for metric in ("R_mae_pp","M_mae_pp","actual_mae_pp"):assert f"{q[metric]:.4f}" in text
    write(OUT/"targeted_test_verification.json",{"tests_passed":11,"exit_code":0,
        "command":"D:/miniconda3/python.exe -m pytest -q tests/test_proposal_gain_models.py",
        "observed_result":"11 passed in 0.28s","observed_before_prepare_and_fitting":True,
        "test_sha256":sha(ROOT/"tests/test_proposal_gain_models.py"),"full_project_tests_rerun":False})
    write(OUT/"packaging_verification.json",{"all_checks_passed":True,"prior_completion_files_checked":counts,
        "historical_unique_files_checked":len(unique),"historical_report_md_docx_pdf_unchanged":True,"legacy_core_checks_passed":len(core),
        "valid_local_report_links":len(links),"report_mae_values_checked":True,"figure_visually_inspected":True,"essential_figure_data_clipped":False,
        "new_full_rollouts":0,"new_fits":0,"runtime_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "finalizer_sha256":sha(__file__)})
    old,prior_sha=read(STATUS),sha(STATUS);HISTORY.write_bytes(STATUS.read_bytes());assert sha(HISTORY)==prior_sha
    status=dict(old)
    for key in ("new_world_pair_replication_rollouts","new_true_scene_audit_rollouts","replication_family_results","fixed_gate_diagnostics","targeted_actual_counterfactual_diagnosis_in_current_stage"):
        status.pop(key,None)
    now=datetime.now(timezone.utc).isoformat()
    status.update({"stage":"proposal_gain_factorial_complete_new_predictions_not_better_than_zero_no_beneficial_test_nominees",
        "updated_date":"2026-10-03","updated_at_utc":now,"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "formal_round3_training_started":False,"formal_round3_final_bank_generated":False,"new_simulations":6180,
        "new_planning_rollouts":5160,"new_evaluation_rollouts":1020,"new_supervised_training_runs":4,
        "new_lightweight_linear_fits":2,"new_lightweight_kernel_fits":2,"new_lightweight_classifier_fits":0,
        "new_supervised_neural_training_runs":0,"new_ppo_training_seeds":0,"new_llm_calls":0,
        "actual_control_efficacy_evaluated_in_current_stage":True,"observable_student_closed_loop_advantage_established":False,
        "teacher_generalization_advantage_established":False,"llm_control_contribution_established":False,
        "report":str(REPORT.relative_to(ROOT)).replace("\\","/"),"summary":"outputs/proposal_gain_factorial_development_v1/summary.json",
        "manifest":"outputs/proposal_gain_factorial_development_v1/development_manifest.json",
        "integrity_audit":"outputs/proposal_gain_factorial_development_v1/verification.json",
        "analysis_manifest":"outputs/proposal_gain_factorial_development_v1/analysis_manifest.json",
        "independent_verification":"outputs/proposal_gain_factorial_development_v1/verification.json",
        "completion_manifest":"outputs/proposal_gain_factorial_development_v1/completion_manifest.json",
        "previous_stage_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),"previous_stage_status_sha256":prior_sha,
        "previous_gain_target_alignment_diagnostic":{"stage":old["stage"],"report":old["report"],"summary":old["summary"],"diagnostic_scope":old["diagnostic_scope"]},
        "diagnostic_scope":{"generated_training_families":96,"actual_fitting_families":34,"actual_fitting_states":68,
            "calibration_families":24,"eligible_calibration_families":11,"test_families":36,"eligible_test_families":14,
            "all_population_states":312,"planning_futures":6,"independent_actual_futures":6,"source_inputs_unchanged":1601,
            "test_proposal_actual_classes":s["test_proposal_actual_classes"],"test_beneficial_recall_estimable":False},
        "primary_paired_effect":s["primary_paired_effect"],"prediction_quality":s["prediction_quality"],
        "gate_summary":s["gate_summary"],"margins_pp":s["margins_pp"],"qualification_thresholds_changed":False,
        "conclusion":"4个目标×模型对照在36新检验家庭均0覆盖，主差值0。新模型减少旧预测误差但均不如零预测；检验28提案起点仅26持平和2有害，无有益，不能估计有益召回。暂缓扩训，优先审计提案器是否漏掉候选收益。",
        "next_step":"冻结本轮模型，预先固定少量开发场景做完整17候选覆盖审计；已有GTEST若再使用必须标为已查看开发诊断。先判断候选库与提案器的机会覆盖，不扩大PPO或临时调门槛。",
        "research_limits":s["limits"]+["No actual beneficial test nominees; beneficial recall not estimable, not zero."],
        "targeted_tests":{"tests_passed":11,"exit_code":0,"full_project_tests_rerun":False},
        "report_sha256":sha(REPORT),"summary_sha256":sha(OUT/"summary.json")})
    write(STATUS,status);check_completion("gain_target_alignment_diagnostic_development_v1",HISTORY)
    artifacts=[REPORT,ROOT/"reports/收益标签与轻量模型四组对照开发方案_2026-10-03.md",STATUS,HISTORY,ROOT/"tests/test_proposal_gain_models.py"]
    artifacts+=[ROOT/"scripts"/n for n in ("proposal_gain_models.py","run_proposal_gain_factorial.py","verify_proposal_gain_factorial.py","write_proposal_gain_factorial_report.py","finalize_proposal_gain_factorial.py")]
    artifacts+=[p for p in OUT.rglob("*") if p.is_file() and p.name!="completion_manifest.json"]
    completion={"role":"family_disjoint_proposal_gain_factorial_development_completion_not_final_qualification","all_checks_passed":True,
        "created_at_utc":now,"new_full_rollouts":6180,"new_planning_rollouts":5160,"new_actual_rollouts":1020,
        "new_fits":4,"new_ppo_training":0,"new_llm_calls":0,"source_inputs_unchanged":len(m["source_input_sha256"]),
        "targeted_tests_passed":11,"actual_trace_steps_verified":25296,"prior_completion_files_checked":counts,
        "historical_unique_files_checked":len(unique),"legacy_core_checks_passed":len(core),
        "prior_completion_manifest_sha256":m["prior_completion_sha256"],"saved_prior_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),
        "saved_prior_status_sha256":prior_sha,"intentional_status_update_previous_exact_bytes_saved":True,
        "no_beneficial_test_nominees_explicitly_reported":True,"zero_reference_superior_mae_explicitly_reported":True,
        "old_sources_models_reports_preserved":True,"qualification_thresholds_changed":False,
        "files":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in artifacts}}
    write(OUT/"completion_manifest.json",completion)
    for p,h in completion["files"].items():assert sha(ROOT/p)==h,p
    print(json.dumps({"completed":True,"files_hashed":len(completion["files"]),"source_inputs_unchanged":len(m["source_input_sha256"]),
        "prior_completion_files_checked":counts,"status_history_preserved":True},ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
