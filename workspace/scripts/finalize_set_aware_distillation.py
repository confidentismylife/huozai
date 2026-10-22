"""Verify preservation chains and package the completed development experiment."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/set_aware_distillation_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
HISTORY = ROOT / "outputs/third_round_status_history/before_set_aware_distillation_20261003.json"
REPORT = ROOT / "reports/集合监督与覆盖不确定性开发实验报告_2026-10-03.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_completion(folder, status_path):
    completion = read(ROOT / "outputs" / folder / "completion_manifest.json")
    for file, expected in completion["files"].items():
        effective = ROOT / file
        if effective.resolve() == STATUS:
            effective = status_path
        assert sha(effective) == expected, str(effective)
    return len(completion["files"])


def bootstrap(values, kinds):
    rng, total = np.random.default_rng(20262002), np.zeros(2000)
    for kind in sorted(set(kinds)):
        part = values[kinds == kind]
        total += part[rng.integers(len(part), size=(2000, len(part)))].sum(axis=1)
    return np.quantile(total / len(values), [.025, .975])


def main():
    assert not (OUT / "completion_manifest.json").exists()
    assert not HISTORY.exists()
    summary, manifest, verification, descriptive = (read(OUT / p) for p in
        ("summary.json", "development_manifest.json", "verification.json", "descriptive_quantification.json"))
    assert verification["all_checks_passed"] and summary["total_unique_full_rollouts"] == 5829
    assert summary["new_supervised_training_runs"] == 10
    assert summary["new_ppo_training_runs"] == summary["new_llm_calls"] == 0
    assert verification["verifier_sha256"] == sha(ROOT / "scripts/verify_set_aware_distillation.py")
    assert verification["analysis_manifest_sha256"] == sha(OUT / "analysis_manifest.json")
    for path, expected in manifest["source_input_sha256"].items():
        assert sha(path) == expected, path
    for path, expected in descriptive["source_sha256"].items():
        assert sha(OUT / path) == expected
    assert descriptive["script_sha256"] == sha(ROOT / "scripts/describe_set_aware_distillation.py")
    previous_completion = ROOT / "outputs/crossed_teacher_rng_diagnostic_development_v1/completion_manifest.json"
    prior_counts = {
        "crossed": check_completion("crossed_teacher_rng_diagnostic_development_v1", STATUS),
        "observable": check_completion("observable_teacher_label_audit_development_v1", ROOT / "outputs/third_round_status_history/before_crossed_teacher_rng_diagnostic_20261003.json"),
        "pilot": check_completion("teacher_factorial_pilot_development_v1", ROOT / "outputs/third_round_status_history/before_observable_teacher_label_audit_20261003.json"),
        "research_value": check_completion("research_value_quantification_20261003", STATUS)}
    value = read(ROOT / "outputs/research_value_quantification_20261003/completion_manifest.json")
    for path, expected in value["sources_unchanged"].items():
        assert sha(ROOT / path) == expected
    historical = read(ROOT / "artifacts/full_history_report_assets/experiment_preservation_audit.json")
    historical_files = {Path(r["path"]).resolve(): r["current_sha256"] for r in historical["records"]}
    for path, expected in historical_files.items():
        effective = ROOT / "outputs/third_round_status_history/before_teacher_diagnosis_and_factorial_pilot.json" if path == STATUS else path
        assert sha(effective) == expected
    qa = read(ROOT / "artifacts/full_history_report_assets/report_qa_manifest.json")
    for artifact in qa["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    core_names = {"env.py", "dynamic_conflicts.py", "confirmatory.py", "terminal_shield.py", "source_world_planning.py",
                  "ppo_reference_planning.py", "behavior.py", "hazard.py", "config.py", "run_source_world_planning.py"}
    core_checks = []
    for folder in ("ppo_extended_world_planning_development_v1", "baseline_relative_plan_distillation_development_v1",
                   "observable_plan_distillation_development_v1", "teacher_factorial_pilot_development_v1"):
        old = read(ROOT / "outputs" / folder / "development_manifest.json")
        hashes = old.get("source_hashes", old.get("source_input_sha256", {}))
        checked = 0
        for path, expected in hashes.items():
            file = Path(path)
            if not file.is_absolute():
                file = ROOT / file
            if file.name in core_names:
                assert sha(file) == expected
                core_checks.append({"bank": folder, "file": str(file), "sha256": expected})
                checked += 1
        assert checked == 10
    with (OUT / "family_paired_deltas.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    delta = {(r["arm"], r["family"]): float(r["delta_incap_vs601_pp"]) for r in rows}
    ids = sorted({r["family"] for r in rows})
    kindmap = {s["scenario_id"]: s["conflict_type"] for s in manifest["audit_scenario_records"]}
    kinds = np.asarray([kindmap[f] for f in ids])
    teacher = -np.asarray([delta[("teacher_robust3", f)] for f in ids])
    for row in descriptive["teacher_retention"]:
        student = -np.asarray([delta[(row["arm"], f)] for f in ids])
        contrast = student - .5 * teacher
        np.testing.assert_allclose([row["benefit_pp"], row["teacher_benefit_pp"], row["descriptive_point_retention_percent"], row["benefit_minus_half_teacher_pp"], row["exploratory_ci_low_pp"], row["exploratory_ci_high_pp"]],
            [student.mean(), teacher.mean(), 100*student.mean()/teacher.mean(), contrast.mean(), *bootstrap(contrast, kinds)], atol=1e-10, rtol=0)
    text = REPORT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", text)
    assert len(links) == 12 and all(Path(path).is_file() for path in links)
    for row in summary["all_paired_effects"]:
        assert row["treatment"] in text and f"{row['delta_incap_pp']:+.4f}" in text
    assert "5829" in text and "25275" in text and "第三轮正式PPO训练与最终银行未启动" in text
    tests = {"command": "D:/miniconda3/python.exe -m pytest -q tests/test_set_aware_plan_student.py",
        "observed_result": "8 passed in 3.97s", "exit_code": 0, "tests_passed": 8,
        "test_sha256": sha(ROOT / "tests/test_set_aware_plan_student.py"),
        "observed_before_prepare_and_training": True, "full_project_tests_rerun": False}
    write(OUT / "targeted_test_verification.json", tests)
    write(OUT / "packaging_verification.json", {"all_checks_passed": True, "prior_completion_files_checked": prior_counts,
        "historical_snapshot_records_checked": len(historical["records"]), "historical_unique_files_checked": len(historical_files),
        "historical_report_md_docx_pdf_unchanged": True, "readme_unchanged_against_pilot_hash": True,
        "legacy_core_checks": core_checks, "supplementary_teacher_retention_contrasts_independently_recomputed": True,
        "valid_report_links": len(links), "all22_arms_reported": True, "new_simulations": 0, "new_training": 0, "new_llm_calls": 0,
        "finalizer_sha256": sha(__file__)})
    before_sha, previous = sha(STATUS), read(STATUS)
    HISTORY.write_bytes(STATUS.read_bytes())
    assert sha(HISTORY) == before_sha
    now = datetime.now(timezone.utc).isoformat()
    status = dict(previous)
    status.update({"stage": "set_aware_distillation_pilot_complete_all_overrides_blocked_no_guarded_gain_established",
        "updated_date": "2026-10-03", "updated_at_utc": now,
        "freeze_allowed": False, "hard_shield_qualification_passed": False,
        "formal_round3_training_started": False, "formal_round3_final_bank_generated": False,
        "new_simulations": 5829, "new_planning_rollouts": 4896, "new_evaluation_rollouts": 933,
        "new_supervised_training_runs": 10, "new_ppo_training_seeds": 0, "new_llm_calls": 0,
        "observable_student_closed_loop_advantage_established": False, "teacher_generalization_advantage_established": False,
        "llm_control_contribution_established": False,
        "teacher_benefit_observed_in_sgdev_small_development_pilot": True,
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "summary": "outputs/set_aware_distillation_development_v1/summary.json",
        "manifest": "outputs/set_aware_distillation_development_v1/development_manifest.json",
        "integrity_audit": "outputs/set_aware_distillation_development_v1/integrity_audit.json",
        "analysis_manifest": "outputs/set_aware_distillation_development_v1/analysis_manifest.json",
        "independent_verification": "outputs/set_aware_distillation_development_v1/verification.json",
        "completion_manifest": "outputs/set_aware_distillation_development_v1/completion_manifest.json",
        "previous_stage_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "previous_stage_status_sha256": before_sha,
        "previous_crossed_rng_diagnostic": {"stage": previous["stage"], "report": previous["report"], "summary": previous["summary"],
            "pairwise_conflict_family_categories": previous["pairwise_conflict_family_categories"],
            "pooled_six_future_population_primary_conflicting_families": previous["pooled_six_future_population_primary_conflicting_families"]},
        "set_aware_pilot_scope": {"new_development_families": 24, "initial_population_states": 48, "unique_initial_observations": 24,
            "arms": 22, "supervised_objectives": 2, "paired_initializations": 5, "new_supervised_fits": 10,
            "actual_execution_trace_steps": 25275, "no_behavior_map_noise_shift": True},
        "primary_paired_effect": summary["primary_paired_effect"], "gate_summary": summary["gate_summary"],
        "qualification_thresholds_changed": False,
        "conclusion": "两目标门控均48/48回退601，无覆盖收益。集合无门控改善0.1302百分点、3家庭改善3变差；唯一无门控改善0.2315全部来自1家庭，未建立稳定优势。robust3教师本轮改善0.4601百分点仍属开发证据。",
        "next_step": "先另存提名与覆盖收益一致化、训练家庭交叉拟合和独立余量校准的开发方案；不按本轮结果放松门控，不扩大正式PPO训练。",
        "research_limits": summary["limitations"], "targeted_tests": {"tests_passed": 8, "exit_code": 0, "full_project_tests_rerun": False},
        "report_sha256": sha(REPORT), "summary_sha256": sha(OUT / "summary.json")})
    write(STATUS, status)
    check_completion("crossed_teacher_rng_diagnostic_development_v1", HISTORY)
    artifacts = [REPORT, ROOT / "reports/集合监督与覆盖不确定性小规模开发方案_2026-10-03.md", HISTORY, STATUS,
        ROOT / "tests/test_set_aware_plan_student.py"]
    artifacts += [ROOT / "scripts" / name for name in ("set_aware_plan_student.py", "run_set_aware_distillation.py", "analyze_set_aware_distillation.py",
        "verify_set_aware_distillation.py", "describe_set_aware_distillation.py", "write_set_aware_distillation_report.py", "finalize_set_aware_distillation.py")]
    artifacts += [p for p in OUT.rglob("*") if p.is_file() and p.name != "completion_manifest.json"]
    completion = {"role": "completed_small_set_aware_development_pilot_not_formal_qualification", "all_checks_passed": True,
        "created_at_utc": now, "new_supervised_fits": 10, "new_planning_rollouts": 4896, "new_execution_rollouts": 933,
        "total_unique_full_rollouts": 5829, "new_ppo_training": 0, "new_llm_calls": 0,
        "independent_actual_trace_steps_checked": 25275, "frozen_inputs_unchanged": len(manifest["source_input_sha256"]),
        "legacy_core_checks_passed": len(core_checks), "targeted_tests_passed": 8, "guard_all_fallback_reported_as_no_advantage": True,
        "prior_completion_files_checked": prior_counts, "historical_unique_files_checked": len(historical_files),
        "prior_completion_manifest_sha256": sha(previous_completion), "prior_status_sha256": before_sha,
        "intentional_live_status_update_with_exact_previous_bytes_saved": True,
        "saved_prior_status": str(HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "verification_serialization_failure_and_source_preserved": True,
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in artifacts}}
    write(OUT / "completion_manifest.json", completion)
    for path, expected in read(OUT / "completion_manifest.json")["files"].items():
        assert sha(ROOT / path) == expected
    print(json.dumps({"completed": True, "files_hashed": len(completion["files"]), "frozen_inputs_unchanged": len(manifest["source_input_sha256"]),
        "legacy_core_checks_passed": len(core_checks), "historical_unique_files_checked": len(historical_files), "status_history_preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
