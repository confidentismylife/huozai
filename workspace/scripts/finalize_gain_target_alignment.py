"""Verify preservation, report QA and package the completed diagnostic."""
from datetime import datetime,timezone
from pathlib import Path
import json
import re
import sys
import numpy as np
import matplotlib
from finalize_set_aware_distillation import sha,read,write,check_completion

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/gain_target_alignment_diagnostic_development_v1"
STATUS=ROOT/"outputs/third_round_development_status.json"
HISTORY=ROOT/"outputs/third_round_status_history/before_gain_target_alignment_diagnostic_20261003.json"
REPORT=ROOT/"reports/收益目标对齐与十二未来复核诊断报告_2026-10-03.md"


def main():
    assert not (OUT/"completion_manifest.json").exists() and not HISTORY.exists()
    m,s,v=(read(OUT/p) for p in ("development_manifest.json","summary.json","verification.json"))
    assert v["all_checks_passed"] and s["full_rollouts"]==772
    assert s["all_fits"]==s["new_ppo_training"]==s["new_llm_calls"]==0
    assert v["verifier_sha256"]==sha(ROOT/"scripts/verify_gain_target_alignment.py")
    assert v["analysis_manifest_sha256"]==sha(OUT/"analysis_manifest.json")
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    assert sha(STATUS)==m["previous_status_sha256"]==sha(OUT/"previous_status_exact_bytes.json")
    counts={
        "override_worth":check_completion("override_worth_evidence_development_v1",STATUS),
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
        target=ROOT/"outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if p==STATUS else p
        assert sha(target)==h,p
    for a in read(ROOT/"artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]:assert sha(a["path"])==a["sha256"]
    core=read(ROOT/"outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for e in core:assert sha(e["file"])==e["sha256"]
    # Preserve the first report draft; corrections clarify gate semantics and reused populations.
    assets=read(OUT/"report_assets_manifest.json");draft_assets=read(OUT/"report_assets_manifest_draft_1.json")
    assert draft_assets["report_sha256"]==sha(OUT/"report_editorial_draft_1.md")
    for p,h in assets["plots_sha256"].items():assert sha(OUT/p)==h
    assets.update({"report_sha256":sha(REPORT),"writer_sha256":sha(ROOT/"scripts/write_gain_target_alignment_report.py"),
        "editorial_draft_1_preserved":True,"editorial_repairs":["Lowering gain alone does not fix ranking; combined classifier safety requires new evidence.","Replication reuses original population starts."],
        "figure_visually_inspected":True,"figure_clipped_or_overlapping_essential_data":False})
    write(OUT/"report_assets_manifest.json",assets)
    text=REPORT.read_text(encoding="utf-8")
    links=re.findall(r"\]\((C:/[^)]+)\)",text);assert len(links)==12 and all(Path(p).is_file() for p in links)
    assert all(str(n) in text for n in (108,78,772,768,9895,877,1513))
    assert all(r["family"] in text and f"{r['actual_gain_pp']:.4f}" in text for r in s["targeted_families"])
    assert "不再重复" not in text and "不能代表自然场景出现率" in text
    write(OUT/"targeted_test_verification.json",{"tests_passed":8,"exit_code":0,
        "command":"D:/miniconda3/python.exe -m pytest -q tests/test_gain_target_diagnostic.py",
        "observed_result":"8 passed in 0.29s","observed_before_prepare_and_simulations":True,
        "test_sha256":sha(ROOT/"tests/test_gain_target_diagnostic.py"),"full_project_tests_rerun":False})
    write(OUT/"packaging_verification.json",{"all_checks_passed":True,"prior_completion_files_checked":counts,
        "historical_unique_files_checked":len(unique),"historical_report_md_docx_pdf_unchanged":True,"legacy_core_checks_passed":len(core),
        "valid_local_report_links":len(links),"report_data_values_checked":True,"figure_visually_inspected":True,
        "new_full_rollouts":0,"new_fits":0,"runtime_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "finalizer_sha256":sha(__file__)})
    old,prior_sha=read(STATUS),sha(STATUS);HISTORY.write_bytes(STATUS.read_bytes());assert sha(HISTORY)==prior_sha
    status=dict(old)
    for key in ("new_planning_rollouts","new_evaluation_rollouts","new_evidence_scope","primary_paired_effect","gate_summary","classifier_quality","prior_margin_pp","conditional_margin_pp"):
        status.pop(key,None)
    now=datetime.now(timezone.utc).isoformat()
    status.update({"stage":"gain_target_alignment_diagnostic_complete_prediction_error_and_world_paired_semantics_unresolved",
        "updated_date":"2026-10-03","updated_at_utc":now,"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "formal_round3_training_started":False,"formal_round3_final_bank_generated":False,"new_simulations":772,
        "new_world_pair_replication_rollouts":768,"new_true_scene_audit_rollouts":4,
        "new_supervised_training_runs":0,"new_lightweight_classifier_fits":0,"new_lightweight_linear_fits":0,
        "new_supervised_neural_training_runs":0,"new_ppo_training_seeds":0,"new_llm_calls":0,
        "actual_control_efficacy_evaluated_in_current_stage":False,"targeted_actual_counterfactual_diagnosis_in_current_stage":True,
        "observable_student_closed_loop_advantage_established":False,"llm_control_contribution_established":False,
        "report":str(REPORT.relative_to(ROOT)).replace("\\","/"),"summary":"outputs/gain_target_alignment_diagnostic_development_v1/summary.json",
        "manifest":"outputs/gain_target_alignment_diagnostic_development_v1/development_manifest.json",
        "integrity_audit":"outputs/gain_target_alignment_diagnostic_development_v1/verification.json",
        "analysis_manifest":"outputs/gain_target_alignment_diagnostic_development_v1/analysis_manifest.json",
        "independent_verification":"outputs/gain_target_alignment_diagnostic_development_v1/verification.json",
        "completion_manifest":"outputs/gain_target_alignment_diagnostic_development_v1/completion_manifest.json",
        "previous_stage_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),"previous_stage_status_sha256":prior_sha,
        "previous_override_worth_evidence":{"stage":old["stage"],"report":old["report"],"summary":old["summary"],
            "primary_paired_effect":old["primary_paired_effect"],"new_evidence_scope":old["new_evidence_scope"]},
        "diagnostic_scope":{"cached_states":108,"eligible_cached_states":78,"targeted_families":8,"replicated_original_population_states":16,
            "new_futures_per_state":12,"new_full_rollouts":772,"actual_trace_steps":9895,"source_inputs_unchanged":1513,
            "viewed_old_test_reclassified_as_development":True,"no_generalization_effect_estimate":True},
        "replication_family_results":s["targeted_families"],"fixed_gate_diagnostics":s["fixed_gate_diagnostics"],
        "qualification_thresholds_changed":False,
        "conclusion":"十二新未来复核保留四个较大长封闭收益、一个小收益短封闭、一个伤害及两个高分持平案例；预测对规划目标本身也有明显偏差。另发现两个起点R有益而某世界配对收益为负，最坏损失改善不能解释为每世界无伤害；未建立门控总体效益。",
        "next_step":"另存新家庭训练/校准/检验分离的小预算目标与拟合验证方案，明确R与世界内配对收益语义；固定提案与门槛，暂缓扩大PPO训练。",
        "research_limits":s["limits"],"targeted_tests":{"tests_passed":8,"exit_code":0,"full_project_tests_rerun":False},
        "report_sha256":sha(REPORT),"summary_sha256":sha(OUT/"summary.json")})
    write(STATUS,status);check_completion("override_worth_evidence_development_v1",HISTORY)
    artifacts=[REPORT,ROOT/"reports/收益目标对齐与十二未来复核开发方案_2026-10-03.md",STATUS,HISTORY,ROOT/"tests/test_gain_target_diagnostic.py"]
    artifacts+=[ROOT/"scripts"/name for name in ("gain_target_diagnostic.py","run_gain_target_alignment.py","verify_gain_target_alignment.py","write_gain_target_alignment_report.py","finalize_gain_target_alignment.py")]
    artifacts+=[p for p in OUT.rglob("*") if p.is_file() and p.name!="completion_manifest.json"]
    completion={"role":"viewed_development_target_alignment_diagnostic_completion_not_formal_qualification",
        "all_checks_passed":True,"created_at_utc":now,"new_full_rollouts":772,"new_fits":0,"new_ppo_training":0,"new_llm_calls":0,
        "source_inputs_unchanged":len(m["source_input_sha256"]),"targeted_tests_passed":8,"actual_trace_steps_verified":9895,
        "prior_completion_files_checked":counts,"historical_unique_files_checked":len(unique),"legacy_core_checks_passed":len(core),
        "prior_completion_manifest_sha256":m["prior_completion_sha256"],"saved_prior_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),
        "saved_prior_status_sha256":prior_sha,"intentional_status_update_previous_exact_bytes_saved":True,
        "old_sources_models_reports_preserved":True,"qualification_thresholds_changed":False,
        "files":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in artifacts}}
    write(OUT/"completion_manifest.json",completion)
    for p,h in completion["files"].items():assert sha(ROOT/p)==h,p
    print(json.dumps({"completed":True,"files_hashed":len(completion["files"]),"source_inputs_unchanged":len(m["source_input_sha256"]),"prior_completion_files_checked":counts,"status_history_preserved":True},ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
