"""Prepare reporting links and close the bounded session only at its deadline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_two_hour_session_20261012"
DEADLINE = datetime.fromisoformat("2026-10-12T12:16:40+00:00")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    draft = ROOT / "reports/LLM辅助疏散研究_真人行为验证补充稿_2026-10-12.md"
    text = draft.read_text(encoding="utf-8")
    text = text.replace("显著的表达不稳定；这里“显著”仅描述变化幅度，不表示统计显著性。", "较大的表达不稳定。")
    draft.write_text(text, encoding="utf-8")
    plan = ROOT / "reports/Exit_choice下一阶段数据规范与验证方案_2026-10-12.md"
    text = plan.read_text(encoding="utf-8")
    text = text.replace("当前Qwen方案尚未获准成为已验证的人群模拟器。", "当前Qwen方案仍缺少作为已验证人群模拟器的证据。")
    text = text.replace("这里“获准”指研究证据是否支持使用，不是额外用户审批流程。可以继续做本地资料整理和代码，但应保持结论与实际证据相符。\n\n", "资料整理和代码可以继续推进，结论应与实际证据相符。\n\n")
    plan.write_text(text, encoding="utf-8")
    report = ROOT / "reports/Exit_choice两小时自主研究整合报告_2026-10-12.md"
    text = report.read_text(encoding="utf-8")
    text = text.replace("组会汇报_真人行为验证与控制收益_2026-10-12_校订版.pptx", "组会汇报_真人行为验证与控制收益_2026-10-12_组会版.pptx")
    text = text.replace("六页可编辑PPT校订版", "六页可编辑组会PPT")
    for script, folder in [
        ("exit_choice_initial_benchmark.py", "exit_choice_initial_benchmark_20261012"),
        ("audit_exit_choice_transitions.py", "exit_choice_transition_audit_20261012"),
        ("exit_choice_calibration_campaign.py", "exit_choice_calibration_campaign_20261012"),
        ("exit_choice_supervised_context.py", "exit_choice_supervised_context_20261012"),
        ("exit_choice_order_stability.py", "exit_choice_order_stability_20261012"),
        ("exit_choice_training_table_probe.py", "exit_choice_training_table_probe_20261012"),
        ("exit_choice_two_hour_evidence_audit.py", "exit_choice_two_hour_session_20261012"),
    ]:
        text = text.replace(f"| scripts/{script} | outputs/{folder} |",
            f"| [{script}]({ROOT.as_posix()}/scripts/{script}) | [结果目录]({ROOT.as_posix()}/outputs/{folder}) |")
    report.write_text(text, encoding="utf-8")
    sources = [ROOT / "reports/Exit_choice两小时自主研究整合报告_2026-10-12.md",
               ROOT / "reports/Exit_choice证据与论文主张对照_2026-10-12.md",
               ROOT / "reports/Exit_choice下一阶段数据规范与验证方案_2026-10-12.md",
               ROOT / "reports/Exit_choice缓存错误分解_2026-10-12.md",
               draft,
               ROOT / "reports/Exit_choice事件类型与信息可得性核查_2026-10-12.md",
               OUT / "references.bib",
               ROOT / "outputs/two_hour_group_meeting_20261012/final/组会汇报_真人行为验证与控制收益_2026-10-12_组会版.pptx",
               ROOT / "outputs/two_hour_group_meeting_20261012/final/五分钟组会讲稿_精简口语版_2026-10-12.md",
               ROOT / "scripts/exit_choice_observation_contract.py"]
    missing = [str(p) for p in sources if not p.exists()]
    assert not missing, missing
    manifest = {"at_utc": datetime.now(timezone.utc).isoformat(), "phase": "reporting_in_progress",
                "session_deadline_utc": DEADLINE.isoformat(),
                "files": [{"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sources]}
    (OUT / "delivery_manifest_preliminary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": manifest["phase"], "files": len(sources)}, ensure_ascii=False))


def close():
    now = datetime.now(timezone.utc)
    assert now >= DEADLINE, "The user requested continued work through the two-hour deadline."
    report = ROOT / "reports/Exit_choice两小时自主研究整合报告_2026-10-12.md"
    text = report.read_text(encoding="utf-8")
    marker = "## 14. 会话收尾状态\n\n"
    assert marker in text
    text = text.split(marker)[0] + marker + (
        f"约定两小时工作窗口已结束，实际收尾记录时间为{now.astimezone().isoformat()}。"
        "341次新本地LLM调用、缓存分析、独立指标复核、报告、论文补充稿、六页组会PPT及五分钟口语讲稿均已完成。"
        "最终PPT已重新导入、渲染并逐页检查；未声称在原生PowerPoint中运行。\n\n"
        "新增动态接入代码通过11项合成检查，真实时间标注模板仍为空。新的真人独立事件验证、动态改道模型、"
        "新地图运动网络与RL控制尚未执行，旧48个最终疏散场景继续封存。"
        "本轮没有新PPO训练、LLM权重训练、疏散回合或对外发送资料。\n\n"
        f"文件哈希、实验分母及完成情况见[最终完成清单]({OUT.as_posix()}/completion_manifest.json)。\n")
    report.write_text(text, encoding="utf-8")
    preliminary = read(OUT / "delivery_manifest_preliminary.json")
    audit = read(OUT / "evidence_audit.json")
    errors = read(OUT / "cached_error_decomposition.json")
    manifest = {"phase": "completed", "started_at_utc": "2026-10-12T10:16:40+00:00",
                "agreed_deadline_utc": DEADLINE.isoformat(), "completed_at_utc": now.isoformat(),
                "new_local_llm_calls": audit["total_new_local_llm_calls"], "strict_human_records": 103,
                "human_events": 4, "new_human_records": 0, "new_ppo_training_runs": 0,
                "new_evacuation_episodes": 0, "new_llm_weight_training_runs": 0,
                "reserved_final_scenarios_read": 0, "protected_files_unchanged": audit["protected_source_files_unchanged"],
                "recomputed_metric_rows": 114, "contract_synthetic_checks": 11,
                "error_decomposition_methods": len(errors["summaries"]),
                "uncompleted": ["new independent human-event validation", "time-verified dynamic labels",
                                "new-map motion calibration and RL experiments", "human manuscript verification"],
                "files": [{"path": f["path"], "bytes": Path(f["path"]).stat().st_size,
                           "sha256": sha(Path(f["path"]))} for f in preliminary["files"]],
                "automation_cleanup": "handled separately by the calling session after the agreed deadline"}
    (OUT / "completion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "completed", "at_utc": now.isoformat(), "calls": 341}, ensure_ascii=False))


def verify_documents():
    manifest = read(OUT / "delivery_manifest_preliminary.json")
    documents = [Path(f["path"]) for f in manifest["files"] if f["path"].endswith(".md")]
    local_targets = set()
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            cleaned = target.strip("<>")
            if cleaned.startswith("C:/"):
                local_targets.add(cleaned)
    missing = [target for target in sorted(local_targets) if not Path(re.sub(r":\d+$", "", target)).exists()]
    assert not missing, missing
    checked = read(OUT / "observation_contract_self_check_final.json")
    assert checked["passed"] and len(checked["checks"]) == 11
    ppt = ROOT / "outputs/two_hour_group_meeting_20261012/final/组会汇报_真人行为验证与控制收益_2026-10-12_组会版.pptx"
    receipt = read(ROOT / "outputs/two_hour_group_meeting_20261012/build/validation_final.json")
    assert sha(ppt) == receipt["finalSha256"]
    notes = ROOT / "outputs/two_hour_group_meeting_20261012/final/五分钟组会讲稿_精简口语版_2026-10-12.md"
    durations = [int(s) for s in re.findall(r"## 第\d页，(\d+)秒", notes.read_text(encoding="utf-8"))]
    assert len(durations) == 6 and sum(durations) == 300
    result = {"at_utc": datetime.now(timezone.utc).isoformat(), "documents": len(documents),
              "local_link_targets": len(local_targets), "missing_targets": missing,
              "ppt_matches_validation_receipt": True, "spoken_sections": 6,
              "nominal_speaking_seconds": sum(durations), "passed": True,
              "limits": "Links, file hashes and allocated timing only; no human manuscript approval or native PowerPoint test"}
    (OUT / "document_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "verify", "close"])
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify_documents, "close": close}[args.command]()
