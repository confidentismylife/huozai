"""Freeze, acquire, and score the bounded DeepSeek congestion sensitivity gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.campaign_extension import BudgetedPlanner, write_json
from fire_evac.congestion_sensitivity import build_factorial_states, factor_sensitivity, plan_metrics


SOURCE = ROOT / "outputs" / "congestion_plan_coverage_20261022"
OUT = ROOT / "outputs" / "deepseek_congestion_sensitivity_20261022"
PROTOCOL = ROOT / "docs" / "deepseek_congestion_sensitivity_protocol_20261022.md"
REPORT = ROOT / "reports" / "DeepSeek拥堵计划敏感性验收_20261022.md"
DESIGN = OUT / "frozen_design.json"
MANIFEST = OUT / "freeze_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DESIGN.exists() or MANIFEST.exists():
        raise RuntimeError("Sensitivity design already frozen; refusing to overwrite.")
    all_states = json.loads((SOURCE / "frozen_congestion_states.json").read_text(encoding="utf-8"))
    by_original = {state["original_scenario_id"]: state for state in all_states}
    selected = [by_original["DSG0017"], by_original["DSG0018"]]
    design = build_factorial_states(selected, seed=20261022)
    DESIGN.write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "deepseek-congestion-sensitivity-2x2x2-v1",
        "blocks": ["DSG0017", "DSG0018"],
        "factors": {
            "junction_occupancy": [0.20, 0.45],
            "junction_change_per_step": [-0.05, 0.05],
            "forecast_steps_remaining": [2, 8],
        },
        "randomization_seed": 20261022,
        "state_count": len(design),
        "design_sha256": sha(DESIGN),
        "protocol_sha256": sha(PROTOCOL),
        "max_new_requests": 16,
        "max_cny": 0.50,
        "future_outcomes_supplied": False,
        "final_test_scenarios_loaded": False,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def acquire() -> None:
    if not DESIGN.exists() or not MANIFEST.exists():
        raise RuntimeError("Freeze the design before API acquisition.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(DESIGN) != manifest["design_sha256"]:
        raise RuntimeError("Frozen sensitivity design hash changed.")
    responses_path = OUT / "response_audit.json"
    if responses_path.exists():
        raise RuntimeError("Sensitivity responses already exist; refusing duplicate paid calls.")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    planner = BudgetedPlanner(OUT / "api", max_cny=0.50, max_requests=16, max_tokens=512)
    responses = []
    for row in design:
        state = row["state"]
        result = planner(state)
        first = result.plan.phases[0]
        target_capacity = (
            int(state["north_exit_capacity"]) if first.route == "north"
            else int(state["south_exit_capacity"]) if first.route == "south"
            else None
        )
        responses.append({
            "design_id": row["design_id"],
            "block_id": row["block_id"],
            "severity": row["severity"],
            "trend": row["trend"],
            "duration": row["duration"],
            "scenario_id": state["scenario_id"],
            "source": result.source,
            "valid_model_response": result.source in ("api", "cache"),
            "unsafe_first_route_closed": target_capacity is not None and target_capacity <= 0,
            "plan": asdict(result.plan),
            "usage": asdict(result.usage),
            "latency_seconds": planner.last_latency_seconds,
        })
    write_json(responses_path, responses)
    write_json(OUT / "acquisition.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": manifest["design_sha256"],
        "responses": len(responses),
        "ledger": planner.ledger(),
        "final_test_scenarios_loaded": False,
    })
    print(json.dumps(planner.ledger(), ensure_ascii=False, indent=2))


def analyze() -> None:
    response_path = OUT / "response_audit.json"
    if not response_path.exists():
        raise RuntimeError("Acquire frozen responses before analysis.")
    if (OUT / "complete.json").exists():
        raise RuntimeError("Sensitivity analysis already complete; refusing to overwrite.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    acquisition = json.loads((OUT / "acquisition.json").read_text(encoding="utf-8"))
    if sha(DESIGN) != manifest["design_sha256"] or acquisition["design_sha256"] != manifest["design_sha256"]:
        raise RuntimeError("Sensitivity design provenance mismatch.")
    responses = json.loads(response_path.read_text(encoding="utf-8"))
    scored = []
    for item in responses:
        from fire_evac.deepseek_advice import parse_plan
        plan = parse_plan(item["plan"])
        scored.append({**item, **plan_metrics(plan)})
    valid = sum(row["valid_model_response"] for row in scored)
    unsafe = sum(row["unsafe_first_route_closed"] for row in scored)
    sensitivity = factor_sensitivity(scored)
    unique_signatures = len({row["plan_signature"] for row in scored})
    criteria = {
        "valid_rate_at_least_95pct": valid / len(scored) >= 0.95,
        "no_closed_exit_first_route": unsafe == 0,
        "at_least_four_plan_signatures": unique_signatures >= 4,
        "severity_pair_change_at_least_50pct": sensitivity["severity"]["paired_plan_change_fraction"] >= 0.50,
        "trend_pair_change_at_least_50pct": sensitivity["trend"]["paired_plan_change_fraction"] >= 0.50,
        "duration_pair_change_at_least_50pct": sensitivity["duration"]["paired_plan_change_fraction"] >= 0.50,
        "severity_directional_capacity_or_wait": (
            sensitivity["severity"]["weighted_release_high_minus_low"] <= -0.05
            or sensitivity["severity"]["wait_steps_high_minus_low"] >= 1.0
        ),
        "trend_directional_capacity_or_wait": (
            sensitivity["trend"]["weighted_release_high_minus_low"] <= -0.05
            or sensitivity["trend"]["wait_steps_high_minus_low"] >= 1.0
        ),
    }
    passed = all(criteria.values())
    result = {
        "passed": passed,
        "valid": valid,
        "total": len(scored),
        "unsafe_first_route_closed": unsafe,
        "unique_plan_signatures": unique_signatures,
        "unique_confidences": sorted({row["confidence"] for row in scored}),
        "sensitivity": sensitivity,
        "criteria": criteria,
        "ledger": acquisition["ledger"],
        "design_sha256": manifest["design_sha256"],
        "new_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "scored_responses.json", scored)
    write_json(OUT / "analysis.json", result)
    lines = [
        "# DeepSeek拥堵计划敏感性验收",
        "",
        "本轮使用两个训练侧物理背景和完整2×2×2设计，共16个冻结提示。它是温度0的功能验收，不把16次响应当作独立统计样本。",
        "",
        "## 验收结论",
        "",
        f"**{'通过' if passed else '未通过'}。** 合法响应{valid}/{len(scored)}，关闭出口违规{unsafe}，完整计划签名{unique_signatures}种。",
        "",
        "| 因素 | 配对数 | 计划变化率 | 加权有效放行量：高−低 | 等待步：高−低 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"severity": "严重度", "trend": "上升趋势", "duration": "预计持续时间"}
    for factor in ("severity", "trend", "duration"):
        value = sensitivity[factor]
        lines.append(
            f"| {labels[factor]} | {int(value['pairs'])} | {100*value['paired_plan_change_fraction']:.1f}% | "
            f"{value['weighted_release_high_minus_low']:+.3f} | {value['wait_steps_high_minus_low']:+.2f} |"
        )
    lines += ["", "## 逐项标准", ""]
    for name, ok in criteria.items():
        lines.append(f"- {'通过' if ok else '失败'}：`{name}`")
    lines += [
        "",
        "## 响应摘要",
        "",
        "| 区组 | 严重度 | 趋势 | 持续 | 首阶段有效放行 | 加权有效放行 | 等待步 | 总步数 | 置信度 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(scored, key=lambda value: value["design_id"]):
        lines.append(
            f"| {row['block_id']} | {row['severity']} | {row['trend']} | {row['duration']} | "
            f"{row['first_effective_release']:.2f} | {row['weighted_effective_release']:.3f} | "
            f"{row['wait_steps']} | {row['total_plan_steps']} | {row['confidence']:.2f} |"
        )
    ledger = acquisition["ledger"]
    lines += [
        "",
        "## 边界与下一步",
        "",
        f"API尝试{ledger['attempted_requests']}次、成功{ledger['successful_requests']}次，保守费用{ledger['actual_cost_upper_cny']:.6f}元，预留余额{ledger['reserved_cny']:.6f}元。",
        "严重度和趋势具有预先规定的方向要求；持续时间只要求计划发生变化。任何标准失败都意味着当前接口停止进入MAPPO重训练。",
        "两个区组都是人工构造的训练侧压力状态。本轮只回答DeepSeek是否读取并响应新增字段，不证明这些字段可在真实火灾中准确预测，也不证明响应能改善疏散。",
        "最终48个场景未读取，MAPPO新增训练步数为0。",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "实验报告.md").write_text(report, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    complete = {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), **result}
    write_json(OUT / "complete.json", complete)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--acquire", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.acquire:
        acquire()
    else:
        analyze()


if __name__ == "__main__":
    main()
