"""Freeze, acquire and analyze the action-risk-ranking v2 admission gate."""

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

from fire_evac.action_ranking_client_v2 import BudgetedActionRanker
from fire_evac.action_risk_ranking_v2 import (
    build_ranking_factorial,
    numeric_action_ranking,
    parse_action_ranking,
    ranking_payload,
    ranking_safety_violations,
    ranking_sensitivity,
    ranking_signature,
    rule_action_ranking,
)
from fire_evac.campaign_extension import write_json


SOURCE = ROOT / "outputs" / "congestion_plan_coverage_20261022" / "frozen_congestion_states.json"
OUT = ROOT / "outputs" / "action_risk_ranking_v2_20261022"
PROTOCOL = ROOT / "docs" / "action_risk_ranking_v2_protocol_20261022.md"
CORE = ROOT / "src" / "fire_evac" / "action_risk_ranking_v2.py"
CLIENT = ROOT / "src" / "fire_evac" / "action_ranking_client_v2.py"
REPORT = ROOT / "reports" / "动作风险排序接口v2_功能验收_20261022.md"
DESIGN = OUT / "frozen_design.json"
MANIFEST = OUT / "freeze_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DESIGN.exists() or MANIFEST.exists():
        raise RuntimeError("v2 design already frozen; refusing to overwrite")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {row["original_scenario_id"]: row for row in source}
    selected = [by_id["DSG0004"], by_id["DSG0019"]]
    design = build_ranking_factorial(selected, seed=20261023)
    DESIGN.write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "action-risk-ranking-v2",
        "blocks": ["DSG0004", "DSG0019"],
        "factors": {
            "junction_occupancy": [0.20, 0.45],
            "junction_change_per_step": [-0.05, 0.05],
            "hazard_side": ["north", "south"],
            "blockage": ["none", "north_closed"],
        },
        "randomization_seed": 20261023,
        "state_count": len(design),
        "design_sha256": sha(DESIGN),
        "protocol_sha256": sha(PROTOCOL),
        "core_sha256": sha(CORE),
        "client_sha256": sha(CLIENT),
        "criteria": {
            "valid_rate_minimum": 0.95,
            "unsafe_closed_best_maximum": 0,
            "blockage_north_risk_increase_minimum": 0.30,
            "safe_route_flip_fraction_minimum": 0.75,
            "severity_release_gap_increase_minimum": 0.08,
            "trend_release_gap_increase_minimum": 0.05,
            "unique_ranking_signatures_minimum": 4,
        },
        "max_new_requests": 32,
        "max_cny": 0.35,
        "max_output_tokens": 512,
        "new_mappo_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def acquire() -> None:
    if not DESIGN.exists() or not MANIFEST.exists():
        raise RuntimeError("freeze the v2 design before acquisition")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(DESIGN) != manifest["design_sha256"] or sha(PROTOCOL) != manifest["protocol_sha256"]:
        raise RuntimeError("frozen design or protocol changed")
    destination = OUT / "response_audit.json"
    if destination.exists():
        raise RuntimeError("v2 responses already exist; refusing duplicate paid calls")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    ranker = BudgetedActionRanker(OUT / "api", max_cny=0.35, max_requests=32, max_tokens=512)
    records = []
    for row in design:
        state = row["state"]
        result = ranker(state)
        deepseek = result.ranking
        rule = rule_action_ranking(state)
        numeric = numeric_action_ranking(state)
        records.append({
            "design_id": row["design_id"],
            "block_id": row["block_id"],
            "severity": row["severity"],
            "trend": row["trend"],
            "hazard_side": row["hazard_side"],
            "blockage": row["blockage"],
            "scenario_id": state["scenario_id"],
            "source": result.source,
            "valid_model_response": result.source in ("api", "cache"),
            "deepseek": ranking_payload(deepseek),
            "rule": ranking_payload(rule),
            "numeric": ranking_payload(numeric),
            "deepseek_safety_violations": ranking_safety_violations(deepseek, state),
            "rule_safety_violations": ranking_safety_violations(rule, state),
            "numeric_safety_violations": ranking_safety_violations(numeric, state),
            "usage": asdict(result.usage),
            "latency_seconds": ranker.last_latency_seconds,
        })
    write_json(destination, records)
    write_json(OUT / "acquisition.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": manifest["design_sha256"],
        "responses": len(records),
        "ledger": ranker.ledger(),
        "new_mappo_training_steps": 0,
        "final_test_scenarios_loaded": False,
    })
    print(json.dumps(ranker.ledger(), ensure_ascii=False, indent=2))


def _method_records(raw_records, field):
    result = []
    for row in raw_records:
        result.append({
            "block_id": row["block_id"],
            "severity": row["severity"],
            "trend": row["trend"],
            "hazard_side": row["hazard_side"],
            "blockage": row["blockage"],
            "ranking": parse_action_ranking(row[field]),
        })
    return result


def analyze() -> None:
    response_path = OUT / "response_audit.json"
    if not response_path.exists():
        raise RuntimeError("acquire frozen responses before analysis")
    if (OUT / "complete.json").exists():
        raise RuntimeError("v2 analysis already complete; refusing to overwrite")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    acquisition = json.loads((OUT / "acquisition.json").read_text(encoding="utf-8"))
    if sha(DESIGN) != manifest["design_sha256"] or acquisition["design_sha256"] != manifest["design_sha256"]:
        raise RuntimeError("v2 design provenance mismatch")
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    valid = sum(bool(row["valid_model_response"]) for row in raw)
    unsafe = sum(bool(row["deepseek_safety_violations"]) for row in raw)
    analyses = {}
    for method in ("rule", "numeric"):
        rows = _method_records(raw, method)
        analyses[method] = {
            "sensitivity": ranking_sensitivity(rows),
            "unique_ranking_signatures": len({ranking_signature(row["ranking"]) for row in rows}),
            "unsafe_closed_best": sum(bool(item[f"{method}_safety_violations"]) for item in raw),
        }
    if valid == len(raw):
        rows = _method_records(raw, "deepseek")
        analyses["deepseek"] = {
            "sensitivity": ranking_sensitivity(rows),
            "unique_ranking_signatures": len({ranking_signature(row["ranking"]) for row in rows}),
            "unsafe_closed_best": unsafe,
        }
    else:
        analyses["deepseek"] = {
            "sensitivity": None,
            "unique_ranking_signatures": None,
            "unsafe_closed_best": unsafe,
            "note": "Sensitivity not scored because invalid API responses would contaminate it with numeric fallbacks.",
        }
    criteria_spec = manifest["criteria"]
    deepseek = analyses["deepseek"]
    sensitivity = deepseek["sensitivity"]
    criteria = {
        "valid_rate_at_least_95pct": valid / len(raw) >= criteria_spec["valid_rate_minimum"],
        "no_closed_exit_best_action": unsafe <= criteria_spec["unsafe_closed_best_maximum"],
        "complete_sensitivity_matrix": sensitivity is not None,
        "blockage_north_risk_increase": sensitivity is not None and sensitivity["blockage"]["north_risk_increase"] >= criteria_spec["blockage_north_risk_increase_minimum"],
        "safe_route_flip_fraction": sensitivity is not None and sensitivity["hazard_side"]["safe_route_flip_fraction_unblocked"] >= criteria_spec["safe_route_flip_fraction_minimum"],
        "severity_release_gap_increase": sensitivity is not None and sensitivity["severity"]["high_minus_low_release_gap"] >= criteria_spec["severity_release_gap_increase_minimum"],
        "trend_release_gap_increase": sensitivity is not None and sensitivity["trend"]["high_minus_low_release_gap"] >= criteria_spec["trend_release_gap_increase_minimum"],
        "at_least_four_ranking_signatures": deepseek["unique_ranking_signatures"] is not None and deepseek["unique_ranking_signatures"] >= criteria_spec["unique_ranking_signatures_minimum"],
    }
    passed = all(criteria.values())
    result = {
        "passed": passed,
        "valid": valid,
        "total": len(raw),
        "criteria": criteria,
        "methods": analyses,
        "ledger": acquisition["ledger"],
        "design_sha256": manifest["design_sha256"],
        "new_mappo_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "analysis.json", result)

    lines = [
        "# 动作风险排序接口 v2：功能验收",
        "",
        "本轮把三阶段计划改成九动作连续风险与唯一排序。DeepSeek、人工规则和透明数值评分器使用相同的32个冻结训练侧状态。响应不含奖励、失能率、未来轨迹或最优动作。",
        "",
        "## 验收结论",
        "",
        f"**{'通过' if passed else '未通过'}。** DeepSeek合法响应{valid}/{len(raw)}，关闭出口首选违规{unsafe}次。",
        "",
        "| 标准 | 结果 |",
        "|---|---|",
    ]
    for name, ok in criteria.items():
        lines.append(f"| `{name}` | {'通过' if ok else '失败'} |")
    lines += [
        "",
        "## 同信息三种转化器的功能指标",
        "",
        "这些是冻结因素的动作敏感性，不是疏散终点收益，也不把32次确定性响应当作独立统计样本。",
        "",
        "| 方法 | 排名签名 | 封闭首选违规 | 封闭使北侧风险增加 | 烟气安全侧翻转 | 严重度使高放行相对风险增加 | 上升趋势使高放行相对风险增加 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"deepseek": "DeepSeek", "rule": "人工规则", "numeric": "透明数值"}
    for method in ("deepseek", "rule", "numeric"):
        item = analyses[method]
        sens = item["sensitivity"]
        if sens is None:
            lines.append(f"| {labels[method]} | 未评分 | {item['unsafe_closed_best']} | 未评分 | 未评分 | 未评分 | 未评分 |")
        else:
            lines.append(
                f"| {labels[method]} | {item['unique_ranking_signatures']} | {item['unsafe_closed_best']} | "
                f"{sens['blockage']['north_risk_increase']:+.3f} | "
                f"{100*sens['hazard_side']['safe_route_flip_fraction_unblocked']:.1f}% | "
                f"{sens['severity']['high_minus_low_release_gap']:+.3f} | "
                f"{sens['trend']['high_minus_low_release_gap']:+.3f} |"
            )
    ledger = acquisition["ledger"]
    lines += [
        "",
        "## 解释边界与执行决定",
        "",
        "通过只表示输出真正响应了动作相关字段，可以进入另行冻结的小规模训练；不表示DeepSeek优于规则或数值评分，也不表示真实疏散有效。",
        "若任一标准失败，则保持MAPPO新增训练步数为0，不追加提示或降低阈值。若全部通过，下一步仍需先冻结四臂MAPPO训练和停止规则，不能直接读取最终测试集。",
        "",
        f"API尝试{ledger['attempted_requests']}次、成功{ledger['successful_requests']}次；保守费用{ledger['actual_cost_upper_cny']:.6f}元，未结预留{ledger['reserved_cny']:.6f}元。",
        "凭据未写入输出。最终48个场景未读取，MAPPO新增训练步数为0。",
        "",
        f"冻结设计SHA-256：`{manifest['design_sha256']}`。",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "实验报告.md").write_text(report, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    write_json(OUT / "complete.json", {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    })
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
