"""Package fresh worth evidence and verify all earlier preservation chains."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sys
import numpy as np
import torch
import stable_baselines3

from finalize_set_aware_distillation import sha, read, write, check_completion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/override_worth_evidence_development_v1"
PRIOR = ROOT/"outputs/consistent_gain_crossfit_development_v1"
STATUS = ROOT/"outputs/third_round_development_status.json"
HISTORY = ROOT/"outputs/third_round_status_history/before_override_worth_evidence_20261003.json"
REPORT = ROOT/"reports/覆盖价值预测与新增条件校准实验报告_2026-10-03.md"


def main():
    assert not (OUT/"completion_manifest.json").exists() and not HISTORY.exists()
    m,s,v = (read(OUT/path) for path in ("development_manifest.json","summary.json","verification.json"))
    assert v["all_checks_passed"] and s["total_unique_full_rollouts"] == 1674
    assert s["new_ppo_training"] == s["new_llm_calls"] == s["new_deep_network_training"] == 0
    assert v["verifier_sha256"] == sha(ROOT/"scripts/verify_override_worth_evidence.py")
    assert v["analysis_manifest_sha256"] == sha(OUT/"analysis_manifest.json")
    for path,expected in m["source_input_sha256"].items(): assert sha(path) == expected
    assert sha(STATUS) == m["previous_status_sha256"] == sha(OUT/"previous_status_exact_bytes.json")
    assert sha(PRIOR/"completion_manifest.json") == m["prior_completion_sha256"]
    counts = {
        "consistent_gain":check_completion("consistent_gain_crossfit_development_v1",STATUS),
        "set_aware":check_completion("set_aware_distillation_development_v1",ROOT/"outputs/third_round_status_history/before_consistent_gain_crossfit_20261003.json"),
        "crossed":check_completion("crossed_teacher_rng_diagnostic_development_v1",ROOT/"outputs/third_round_status_history/before_set_aware_distillation_20261003.json"),
        "observable":check_completion("observable_teacher_label_audit_development_v1",ROOT/"outputs/third_round_status_history/before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot":check_completion("teacher_factorial_pilot_development_v1",ROOT/"outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"),
        "research_value":check_completion("research_value_quantification_20261003",STATUS)}
    for path,expected in read(ROOT/"outputs/research_value_quantification_20261003/completion_manifest.json")["sources_unchanged"].items(): assert sha(ROOT/path) == expected
    historical = read(ROOT/"artifacts/full_history_report_assets/experiment_preservation_audit.json")
    unique = {Path(r["path"]).resolve():r["current_sha256"] for r in historical["records"]}
    for path,expected in unique.items():
        effective = ROOT/"outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path
        assert sha(effective) == expected
    for artifact in read(ROOT/"artifacts/full_history_report_assets/report_qa_manifest.json")["artifacts"]: assert sha(artifact["path"]) == artifact["sha256"]
    core = read(ROOT/"outputs/set_aware_distillation_development_v1/packaging_verification.json")["legacy_core_checks"]
    for entry in core: assert sha(entry["file"]) == entry["sha256"]
    text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)",text)
    assert len(links) == 11 and all(Path(p).is_file() for p in links)
    assert all(str(n) in text for n in (1674,1116,558,13096,1373))
    for row in s["test_paired_effects"]:
        assert row["treatment"] in text and f"{row['delta_incap_pp']:+.4f}" in text
    write(OUT/"targeted_test_verification.json",{"tests_passed":8,"exit_code":0,
        "command":"D:/miniconda3/python.exe -m pytest -q tests/test_override_worth_model.py","observed_result":"8 passed in 3.75s",
        "observed_before_prepare_and_fitting":True,"test_sha256":sha(ROOT/"tests/test_override_worth_model.py"),"full_project_tests_rerun":False})
    write(OUT/"packaging_verification.json",{"all_checks_passed":True,"prior_completion_files_checked":counts,
        "historical_unique_files_checked":len(unique),"historical_report_md_docx_pdf_unchanged":True,"readme_unchanged":True,
        "legacy_core_checks_passed":len(core),"valid_local_report_links":len(links),"new_full_rollouts":0,"new_fits":0,
        "runtime_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"torch":str(torch.__version__),"stable_baselines3":stable_baselines3.__version__},
        "finalizer_sha256":sha(__file__)})
    previous,prior_sha = read(STATUS),sha(STATUS)
    HISTORY.write_bytes(STATUS.read_bytes()); assert sha(HISTORY) == prior_sha
    now = datetime.now(timezone.utc).isoformat()
    status = dict(previous)
    for field in ("cached_diagnostic_scope","additive_margin_pp","oof_consistent_cached_diagnostic"): status.pop(field,None)
    status.update({"stage":"override_worth_fresh_calibration_evidence_complete_guard_misses_benefits_no_added_incap_gain",
        "updated_date":"2026-10-03","updated_at_utc":now,"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "formal_round3_training_started":False,"formal_round3_final_bank_generated":False,
        "new_simulations":1674,"new_planning_rollouts":1116,"new_evaluation_rollouts":558,
        "new_supervised_training_runs":1,"new_lightweight_linear_fits":0,"new_lightweight_classifier_fits":1,
        "new_supervised_neural_training_runs":0,"new_ppo_training_seeds":0,"new_llm_calls":0,
        "actual_control_efficacy_evaluated_in_current_stage":True,"nomination_override_same_score_verified":True,
        "observable_student_closed_loop_advantage_established":False,"teacher_generalization_advantage_established":False,"llm_control_contribution_established":False,
        "report":str(REPORT.relative_to(ROOT)).replace("\\","/"),"summary":"outputs/override_worth_evidence_development_v1/summary.json",
        "manifest":"outputs/override_worth_evidence_development_v1/development_manifest.json",
        "integrity_audit":"outputs/override_worth_evidence_development_v1/verification.json",
        "analysis_manifest":"outputs/override_worth_evidence_development_v1/analysis_manifest.json",
        "independent_verification":"outputs/override_worth_evidence_development_v1/verification.json",
        "completion_manifest":"outputs/override_worth_evidence_development_v1/completion_manifest.json",
        "previous_stage_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),"previous_stage_status_sha256":prior_sha,
        "previous_consistent_gain_diagnostic":{"stage":previous["stage"],"report":previous["report"],"summary":previous["summary"],
            "cached_diagnostic_scope":previous["cached_diagnostic_scope"]},
        "new_evidence_scope":{"calibration_pool_families":180,"eligible_pool_families":48,"selected_calibration_families":30,
            "natural_test_families":24,"population_states":108,"fixed_execution_arms":5,"actual_trace_steps":13096,
            "actual_calibration_proposal_classes":{"harmful":2,"neutral_or_small":52,"beneficial":6},
            "actual_natural_proposal_classes":{"harmful":0,"neutral_or_small":14,"beneficial":4},
            "no_calibration_proposals_for_safe_exit_congested":True},
        "primary_paired_effect":s["primary_paired_effect"],"gate_summary":s["gate_summary"],"classifier_quality":s["classifier_quality"],
        "prior_margin_pp":s["prior_margin_pp"],"conditional_margin_pp":s["conditional_margin_pp"],"qualification_thresholds_changed":False,
        "conclusion":"30个新提案校准家庭补足条件证据，但误差余量从1.7557升到3.3954百分点。自然检验4个有益起点均被门控挡住，最终过滤0覆盖，主差值0；分类评分只在9提案家庭略优于频率参考，未建立伤害拒绝能力或稳定控制收益。",
        "next_step":"先另存面向独立实际反事实收益的目标与预测偏差诊断方案，严格隔离新训练/校准/检验；不临时降低阈值，不扩大PPO训练。",
        "research_limits":s["limits"],"targeted_tests":{"tests_passed":8,"exit_code":0,"full_project_tests_rerun":False},
        "report_sha256":sha(REPORT),"summary_sha256":sha(OUT/"summary.json")})
    write(STATUS,status); check_completion("consistent_gain_crossfit_development_v1",HISTORY)
    artifacts = [REPORT,ROOT/"reports/覆盖价值预测与新增条件校准开发方案_2026-10-03.md",STATUS,HISTORY,ROOT/"tests/test_override_worth_model.py"]
    artifacts += [ROOT/"scripts"/name for name in ("override_worth_model.py","run_override_worth_evidence.py","verify_override_worth_evidence.py","write_override_worth_report.py","finalize_override_worth_evidence.py")]
    artifacts += [p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json"]
    completion = {"role":"fresh_development_override_worth_evidence_completion_not_formal_qualification","all_checks_passed":True,"created_at_utc":now,
        "new_lightweight_classifier_fits":1,"new_ppo_training":0,"new_llm_calls":0,"new_full_rollouts":1674,
        "new_planning_rollouts":1116,"new_evaluation_rollouts":558,"calibration_families":30,"natural_test_families":24,
        "source_inputs_unchanged":len(m["source_input_sha256"]),"targeted_tests_passed":8,"actual_trace_steps_verified":13096,
        "prior_completion_files_checked":counts,"historical_unique_files_checked":len(unique),"legacy_core_checks_passed":len(core),
        "prior_completion_manifest_sha256":sha(PRIOR/"completion_manifest.json"),"saved_prior_status":str(HISTORY.relative_to(ROOT)).replace("\\","/"),
        "saved_prior_status_sha256":prior_sha,"intentional_status_update_previous_exact_bytes_saved":True,
        "no_added_incap_gain_reported":True,"qualification_thresholds_changed":False,
        "files":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in artifacts}}
    write(OUT/"completion_manifest.json",completion)
    for file,expected in completion["files"].items(): assert sha(ROOT/file) == expected
    print(json.dumps({"completed":True,"files_hashed":len(completion["files"]),"source_inputs_unchanged":len(m["source_input_sha256"]),"prior_status_preserved":True},ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
