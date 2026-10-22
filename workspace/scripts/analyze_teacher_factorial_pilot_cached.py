"""Complete cached pilot analysis after a NumPy scalar JSON-export failure.

The frozen collector, policies, choice rules, caches, and statistics are intact.
Only NumPy scalar serialization is adapted; no environment rollout is executed.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np

import run_teacher_factorial_pilot as frozen
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from diagnose_plan_teacher import digest

ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def convert(item):
    if isinstance(item,np.generic):
        return item.item()
    raise TypeError(f"unsupported JSON object: {type(item)}")


def json_native(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert)+"\n",encoding="utf-8")


def run(args):
    args.output=args.output.resolve()
    manifest=json.loads((args.output/"development_manifest.json").read_text(encoding="utf-8"))
    progress=json.loads((args.output/"progress.json").read_text(encoding="utf-8"))
    assert progress["families_completed"]==progress["families_total"]==24
    assert len(list((args.output/"states").glob("*.npz")))==48
    for path,value in manifest["source_input_sha256"].items():
        assert digest(path)==value,f"frozen input changed: {path}"
    revision=args.output/"analysis_revisions/initial_serialization_failure"
    revision.mkdir(parents=True,exist_ok=True)
    if not (revision/"error.txt").exists():
        (revision/"error.txt").write_text(
            "All 24 families and 48 states completed. Original frozen analyzer failed "
            "when writing summary JSON: TypeError: Object of type int64 is not JSON serializable. "
            "NumPy counts in bootstrap summaries require conversion to Python integer. "
            "No simulation, candidate selection or numeric statistic is changed by the repair.\n",encoding="utf-8")
        shutil.copy2(ROOT/"scripts/run_teacher_factorial_pilot.py",revision/"frozen_collector_and_initial_analyzer.py")
        for path in args.output.glob("*.csv"):
            shutil.copy2(path,revision/path.name)
        shutil.copy2(args.output/"integrity_audit.json",revision/"integrity_audit.json")
    frozen.json_write=json_native
    frozen.json=SimpleNamespace(dumps=lambda value,**kwargs:json.dumps(value,default=convert,**kwargs))
    scenarios=tuple(DynamicConflictScenario(**record) for record in manifest["scenario_records"])
    frozen.analyze(args,manifest,scenarios,progress["new_planning_rollouts_this_invocation"],
                   progress["new_evaluation_rollouts_this_invocation"],progress["elapsed_seconds"])
    summary_path=args.output/"summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis_completion"]={
        "mode":"cached_analysis_only_numpy_scalar_json_export_repair",
        "new_rollouts_during_analysis_repair":0,
        "new_training_during_analysis_repair":0,
        "collector_and_selector_frozen_unchanged":True,
        "original_collection_counts_and_duration_retained":True,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "wrapper_code_sha256":digest(__file__)}
    json_native(summary_path,summary)
    with args.report.open("a",encoding="utf-8") as stream:
        stream.write("\n汇总导出首次因NumPy整数的JSON格式报错；48个采集缓存均完整，失败分析版本已保留。"
                     "补充入口仅转换数值序列化并从缓存完成同一分析，没有重新运行仿真或修改统计、候选选择规则。"
                     "汇总中的采集数量与耗时仍取原采集进度记录。\n")
    analysis=json.loads((args.output/"analysis_manifest.json").read_text(encoding="utf-8"))
    analysis.update(wrapper_code_sha256=digest(__file__),summary_sha256=digest(summary_path),
                    report_sha256=digest(args.report),serialization_repair_only=True,
                    new_rollouts_in_repair=0,
                    original_failed_analysis="analysis_revisions/initial_serialization_failure")
    json_native(args.output/"analysis_manifest.json",analysis)
    print(json.dumps({"completed":True,"unique_rollouts":summary["total_unique_full_rollouts"],
                      "new_rollouts_in_repair":0},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/teacher_factorial_pilot_development_v1")
    parser.add_argument("--report",type=Path,default=ROOT/"reports/教师世界聚合与规划预算全新开发验证报告.md")
    run(parser.parse_args())
