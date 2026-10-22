"""Read-only mechanism audit of the completed action-ranking v2 gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.action_risk_ranking_v2 import parse_action_ranking, ranking_signature
from fire_evac.campaign_extension import write_json


OUT = ROOT / "outputs" / "action_risk_ranking_v2_20261022"
REPORT = ROOT / "reports" / "动作风险排序接口v2_失败机制审计_20261022.md"


def release_class(action: str) -> str:
    return "wait" if action == "wait" else action.split("_")[1]


def route(action: str) -> str:
    return "wait" if action == "wait" else action.split("_")[0]


def main() -> None:
    rows = json.loads((OUT / "response_audit.json").read_text(encoding="utf-8"))
    summary = {}
    for method in ("deepseek", "rule", "numeric"):
        best = []
        signatures = []
        safe_correct = []
        by_factor = {name: defaultdict(Counter) for name in ("severity", "trend", "hazard_side", "blockage")}
        unique_scores = set()
        for row in rows:
            ranking = parse_action_ranking(row[method])
            action = ranking.best_action
            best.append(action)
            signatures.append(ranking_signature(ranking))
            unique_scores.update(round(item.risk_score, 8) for item in ranking.actions)
            for factor in by_factor:
                by_factor[factor][row[factor]][action] += 1
            if row["blockage"] == "none":
                safe = "south" if row["hazard_side"] == "north" else "north"
                safe_correct.append(route(action) == safe)
        summary[method] = {
            "best_action_counts": dict(Counter(best)),
            "best_release_counts": dict(Counter(release_class(action) for action in best)),
            "best_route_counts": dict(Counter(route(action) for action in best)),
            "unique_ranking_signatures": len(set(signatures)),
            "unique_risk_values": len(unique_scores),
            "unblocked_safe_route_accuracy": sum(safe_correct) / len(safe_correct),
            "unblocked_safe_route_correct": sum(safe_correct),
            "unblocked_safe_route_total": len(safe_correct),
            "best_actions_by_factor": {
                factor: {level: dict(counts) for level, counts in levels.items()}
                for factor, levels in by_factor.items()
            },
        }
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    result = {
        "source_design_sha256": analysis["design_sha256"],
        "read_only": True,
        "new_api_requests": 0,
        "new_mappo_training_steps": 0,
        "final_test_scenarios_loaded": False,
        "summary": summary,
    }
    write_json(OUT / "mechanism_audit.json", result)

    ds = summary["deepseek"]
    lines = [
        "# 动作风险排序接口 v2：失败机制审计",
        "",
        "本审计只读取已经完成的32个冻结响应，没有新增API请求、没有调整阈值、没有训练MAPPO，也没有读取最终测试场景。",
        "",
        "## 主要发现",
        "",
        f"DeepSeek的首选动作中，`south_025`为{ds['best_action_counts'].get('south_025', 0)}/32，"
        f"`north_025`为{ds['best_action_counts'].get('north_025', 0)}/32，"
        f"等待为{ds['best_action_counts'].get('wait', 0)}/32。它从未把0.50、0.75或1.00档列为第一。",
        "",
        f"在16个未封闭状态中，首选路线与当前较低烟气出口一致{ds['unblocked_safe_route_correct']}/"
        f"{ds['unblocked_safe_route_total']}（{100*ds['unblocked_safe_route_accuracy']:.1f}%）。"
        "按完整配对要求，8组危险侧翻转中没有一组同时正确翻转首选路线，因此冻结指标为0%。",
        "",
        "严重度和趋势的失败不是“输出完全不变”：DeepSeek产生11种完整排名。问题是变化方向没有落到预定的当前容量语义。高拥堵使高放行相对低放行的风险差变化为"
        f"{analysis['methods']['deepseek']['sensitivity']['severity']['high_minus_low_release_gap']:+.3f}，"
        "上升趋势相应为"
        f"{analysis['methods']['deepseek']['sensitivity']['trend']['high_minus_low_release_gap']:+.3f}；"
        "两者均为负，和预先要求的正向惩罚相反。",
        "",
        "## 三种转化器对照",
        "",
        "| 方法 | 首选0.25档 | 首选等待 | 未封闭安全路线正确 | 排名签名 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"deepseek": "DeepSeek", "rule": "人工规则", "numeric": "透明数值"}
    for method in ("deepseek", "rule", "numeric"):
        item = summary[method]
        low = item["best_release_counts"].get("025", 0)
        wait = item["best_release_counts"].get("wait", 0)
        lines.append(
            f"| {labels[method]} | {low}/32 | {wait}/32 | "
            f"{item['unblocked_safe_route_correct']}/{item['unblocked_safe_route_total']} "
            f"({100*item['unblocked_safe_route_accuracy']:.1f}%) | {item['unique_ranking_signatures']} |"
        )
    lines += [
        "",
        "人工规则通过全部动作敏感性标准，说明冻结设计能够产生预期方向的可执行变化。透明数值评分器正确处理封闭与烟气方向，但其冻结拥堵权重也低于0.08和0.05门槛；该结果同样保留，不能在本轮事后调权重。DeepSeek除拥堵幅度不足外，还没有可靠跟随当前烟气安全侧。",
        "",
        "## 研究判断",
        "",
        "把输出改成九动作风险表解决了旧接口中“理由码或后续阶段变化冒充当前动作变化”的测量问题。新接口因而更容易暴露真实失败：模型几乎总把最低放行档列在首位，形成固定保守排序；状态变化虽然改变若干分数和名次，却没有形成稳定的烟气方向翻转或拥堵单调性。",
        "",
        "因此当前版本不进入MAPPO。若继续研究，应先更换信息来源或任务角色，而不是继续改同一提示：例如让LLM处理数值控制器看不到的建筑语义、消防文字指令或历史案例，并把它输出的结构化约束交给数值优化器。任何新方向都应使用新的协议编号和独立准入集。",
        "",
        f"冻结设计SHA-256：`{analysis['design_sha256']}`。",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "机制审计.md").write_text(report, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
