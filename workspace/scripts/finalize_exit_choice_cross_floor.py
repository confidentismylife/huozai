"""Finalize the completed floor-transfer development report and manifest."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_cross_floor_transfer_20261012"
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

summary = read(OUT / "summary.json")
assert summary["complete"] and summary["saved"] == 142 and summary["valid"] == 142
check = read(OUT / "independent_verification.json")
assert check["passed"] and check["metric_rows_recomputed"] == 40
protected = read(OUT / "preservation_before.json")
unique_sources = {os.path.normcase(str(Path(r["path"]).resolve())): r for r in protected}
assert all(sha(Path(r["path"])) == r["sha256"] for r in unique_sources.values())
metrics = read(OUT / "metrics.json")
def metric(method, date):
    return next(r for r in metrics if r["method"] == method and r["date"] == date)

report = ROOT / "reports/Exit_choice跨楼层迁移实验_2026-10-12.md"
text = report.read_text(encoding="utf-8")
text = text.replace("完整=True", "完整队列已完成")
zero = metric("qwen_zero_same_settings", "pooled")
full = metric("qwen_source_training_records", "pooled")
utility = metric("portable_nonlinear", "pooled")
lines = ["", "## 本轮可以得出的结论", "",
         f"同设置零样本Qwen正确{round(zero['accuracy']*71)}/71；提供源域记录后正确{round(full['accuracy']*71)}/71。完整记录相对零样本纠正13例，也破坏15例，净少正确2例。其平均概率损失改善，但出口3召回从30/32降到15/32。因此增加源域资料没有产生一致的分类改善。", "",
         f"主要统计参照正确{round(utility['accuracy']*71)}/71，但这个总体成绩由出口1的36/36及出口3的31/32支撑；出口2仅有三例，全部漏判。最近出口在这三例全部正确，而总体较差。统计模型也不能凭94.37%的总体准确率称为完整、可靠的人群模型。", "",
         "32条出口3记录全部来自10-31这一个误报日期；演习日期集中选择出口1或出口2。常用出口代理、此前人群、几何和事件背景相关，不能把跨楼层结果当作独立验证心理熟悉程度，或视为多个独立事件的出口3泛化。", "",
         "本轮没有建立当前Qwen方案在跨楼层条件下的额外预测价值。后续保持候选和资料来源清晰，优先寻找新日期或新建筑的真实决策记录；不要继续在这71条上改提示并称修复已经独立确认。恢复PPO训练仍需动态行为和运动层依据。", "",
         "## 复核与工作量", "",
         f"完成142次新本地调用，71条目标记录的两种LLM条件，加六个统计或规则对照，共568条方法级预测。独立重算426条统计概率和40行汇总指标，最大差为0。请求的训练来源与当前属性逐条核对通过。{len(protected)}条保护哈希登记覆盖{len(unique_sources)}个唯一来源文件，全部保持不变。", "",
         "本轮没有新LLM权重训练、PPO训练或疏散回合，也没有读取48个最终场景。原两小时报告与完成清单保持原状，本轮另存结果。", "",
         f"[冻结方案]({(ROOT/'reports/Exit_choice跨楼层迁移冻结方案_2026-10-12.md').as_posix()})，"
         f"[独立复核]({(OUT/'independent_verification.json').as_posix()})，"
         f"[完整指标]({(OUT/'metrics.json').as_posix()})。"]
marker = "\n## 本轮可以得出的结论\n"
if marker in text: text = text.split(marker)[0]
report.write_text(text+"\n".join(lines)+"\n", encoding="utf-8")
files = [report, ROOT / "reports/Exit_choice跨楼层迁移冻结方案_2026-10-12.md",
         ROOT / "scripts/exit_choice_cross_floor_transfer.py", ROOT / "scripts/verify_exit_choice_cross_floor.py",
         OUT / "frozen_plan.json", OUT / "data_summary.json", OUT / "metrics.json", OUT / "effects.json",
         OUT / "independent_verification.json"]
manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "state": "completed",
            "development_only": True, "new_human_events": 0, "target_records": 71,
            "new_llm_calls": 142, "all_calls_valid": True, "method_predictions": 568,
            "protected_hash_entries": len(protected), "unique_protected_files": len(unique_sources),
            "protected_sources_unchanged": True, "new_ppo_training": 0, "sealed_scenarios_read": 0,
            "files": [{"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
(OUT / "completion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"completed": True, "new_llm_calls": 142, "target_records": 71,
                  "unique_protected_files": len(unique_sources), "report": report.as_posix()}, ensure_ascii=False))
