from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "outputs" / "mappo_development_multiseed_20261021"
AFTER = ROOT / "outputs" / "mappo_independent_actor_multiseed_20261021"
AUDIT = ROOT / "outputs" / "mappo_plan_channel_audit_20261021"
DEEPSEEK = "mappo_50_deepseek_retrieval"
RAW = "mappo_34_raw"
RULE = "mappo_50_rule_plan"


def paired_effect(data: pd.DataFrame, candidate: str, baseline: str) -> dict:
    pivot = data.pivot(
        index=["training_seed", "scenario_id", "population_seed"],
        columns="arm",
        values="incapacitation_rate",
    )
    by_seed = (100 * (pivot[candidate] - pivot[baseline])).groupby("training_seed").mean()
    return {
        "mean_pp": float(by_seed.mean()),
        "seed_sd_pp": float(by_seed.std(ddof=1)),
        "by_seed_pp": {str(int(seed)): float(value) for seed, value in by_seed.items()},
    }


def arm_diagnostics(data: pd.DataFrame) -> dict:
    seed_means = data.groupby(["training_seed", "arm"], as_index=False).agg(
        incapacitation_rate=("incapacitation_rate", "mean"),
        release_fraction=("mean_release_fraction_nonwait", "mean"),
    )
    output = {}
    for arm, group in seed_means.groupby("arm"):
        episodes = data[data.arm == arm]
        output[arm] = {
            "incapacitation_mean_pp": float(100 * group.incapacitation_rate.mean()),
            "incapacitation_seed_sd_pp": float(100 * group.incapacitation_rate.std(ddof=1)),
            "release_mean": float(group.release_fraction.mean()),
            "release_seed_sd": float(group.release_fraction.std(ddof=1)),
            "full_release_episode_fraction": float(
                (episodes.mean_release_fraction_nonwait >= 0.999).mean()
            ),
            "waiting_episode_fraction": float((episodes.wait_fraction > 0).mean()),
        }
    return output


def main() -> int:
    before = pd.read_csv(BEFORE / "combined_episode_results.csv")
    after = pd.read_csv(AFTER / "combined_episode_results.csv")
    audit = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    before_effect = paired_effect(before, DEEPSEEK, RAW)
    after_effect = paired_effect(after, DEEPSEEK, RAW)
    after_rule_effect = paired_effect(after, DEEPSEEK, RULE)
    before_arms = arm_diagnostics(before)
    after_arms = arm_diagnostics(after)
    reduction = 1.0 - after_effect["seed_sd_pp"] / before_effect["seed_sd_pp"]
    payload = {
        "status": "complete",
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests": 0,
        "change": "separate PPO clipping ratios for route and flow actors",
        "deepseek_minus_raw_before": before_effect,
        "deepseek_minus_raw_after": after_effect,
        "deepseek_minus_rule_after": after_rule_effect,
        "relative_seed_sd_reduction": float(reduction),
        "arm_diagnostics_before": before_arms,
        "arm_diagnostics_after": after_arms,
        "plan_channel_audit": audit,
    }
    (AFTER / "actor_fix_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plan = audit["models"][DEEPSEEK]
    before_seed = ", ".join(
        f"{seed}: {value:+.3f}" for seed, value in before_effect["by_seed_pp"].items()
    )
    after_seed = ", ".join(
        f"{seed}: {value:+.3f}" for seed, value in after_effect["by_seed_pp"].items()
    )
    report = f"""# MAPPO独立actor裁剪修正与计划通道诊断

## 本轮做了什么

原实现把路线actor和流量actor的对数概率相加后计算一个联合PPO比率。这样一个actor的概率波动会改变另一个actor的裁剪结果。本轮改为两个actor分别计算PPO裁剪损失，共享同一团队优势和集中式critic；等待动作仍屏蔽流量actor。旧的九动作语义和 `act()` 接口保持不变。

修正后完整回归为417项通过、1条既有PyTorch警告。随后严格复跑原来的3个训练种子、每臂10,000步、12个开发场景和2个人口种子。训练与评估没有调用DeepSeek API，最终测试场景没有读取。

## 修正前后核心对比

| 指标 | 修正前 | 修正后 |
|---|---:|---:|
| DeepSeek计划MAPPO − 34维MAPPO失能率差 | {before_effect['mean_pp']:+.3f} pp | {after_effect['mean_pp']:+.3f} pp |
| 上述差值的种子标准差 | {before_effect['seed_sd_pp']:.3f} pp | {after_effect['seed_sd_pp']:.3f} pp |
| DeepSeek臂平均放行比例 | {before_arms[DEEPSEEK]['release_mean']:.3f} | {after_arms[DEEPSEEK]['release_mean']:.3f} |
| DeepSeek臂全量放行回合占比 | {100*before_arms[DEEPSEEK]['full_release_episode_fraction']:.1f}% | {100*after_arms[DEEPSEEK]['full_release_episode_fraction']:.1f}% |
| 34维臂全量放行回合占比 | {100*before_arms[RAW]['full_release_episode_fraction']:.1f}% | {100*after_arms[RAW]['full_release_episode_fraction']:.1f}% |

三个种子的DeepSeek相对34维差值，修正前为：{before_seed} pp；修正后为：{after_seed} pp。种子标准差下降了{100*reduction:.1f}%，说明联合比率确实制造了大量训练耦合和方向翻转。

但修正没有证明DeepSeek有效。修正后DeepSeek相对34维MAPPO平均失能率高{after_effect['mean_pp']:.3f}个百分点；相对规则计划MAPPO高{after_rule_effect['mean_pp']:.3f}个百分点。多数策略还退化到接近全量放行，34维臂所有评估回合都是全量放行。

## 计划通道是否真的被使用

对每个50维模型保持场景、人口种子和权重不变，只把输入从DeepSeek冻结计划切换为同接口规则计划。共完成{audit['paired_comparisons']}组配对、{audit['evaluation_episodes']}个评估回合。

DeepSeek计划训练模型中：

- 初始计划特征平均L1差异为{plan['initial_plan_l1_difference_mean']:.3f}，说明两类输入确实不同；
- 只有{100*plan['initial_action_change_rate']:.1f}%的回合首动作改变；
- 只有{100*plan['any_action_change_rate']:.1f}%的回合在任一步发生动作改变；
- 20261021和20261023两个种子的动作序列完全不受计划替换影响；只有20261022种子表现出通道敏感性；
- 对敏感的20261022种子，使用DeepSeek计划相对规则计划的失能率高1.528个百分点；三种子平均高{plan['deepseek_minus_rule_incapacitation_pp']:.3f}个百分点。

## 当前结论

独立actor裁剪修复是必要的，它把先前9.025个百分点的种子波动压到0.571个百分点。现在暴露出的主要问题不再是“训练结果随机翻转”，而是**流量策略饱和以及大多数种子忽略计划内容**。因此不能把稳定性改善解释为LLM带来收益。

下一步应在开发集上处理容量actor退化：记录每次更新的流量动作分布和熵，加入开发集检查点选择，并比较无计划、真实计划和置换计划的同模型表现。在策略能够稳定产生多档容量动作以前，不扩大DeepSeek调用、不增加正式训练量，也不读取最终测试集。
"""
    (AFTER / "MAPPO独立裁剪修正与计划通道诊断报告.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
