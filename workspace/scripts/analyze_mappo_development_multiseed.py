from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "mappo_development_multiseed_20261021"
ARMS = ("mappo_34_raw", "mappo_50_rule_plan", "mappo_50_deepseek_retrieval")
METRICS = (
    "incapacitation_rate",
    "mean_cumulative_fed",
    "evacuation_time_seconds",
    "cumulative_congestion",
    "mean_release_fraction_nonwait",
)
COMPARISONS = (
    ("mappo_50_deepseek_retrieval", "mappo_34_raw"),
    ("mappo_50_deepseek_retrieval", "mappo_50_rule_plan"),
)


def main() -> int:
    frames = []
    for seed_dir in sorted(path for path in SOURCE.glob("seed_*") if path.is_dir()):
        seed = int(seed_dir.name.split("_")[-1])
        frame = pd.read_csv(seed_dir / "episode_results.csv")
        frame.insert(0, "training_seed", seed)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(SOURCE / "combined_episode_results.csv", index=False, encoding="utf-8-sig")
    seed_summary = data.groupby(["training_seed", "arm"], as_index=False)[list(METRICS)].mean()
    seed_summary.to_csv(SOURCE / "seed_arm_summary.csv", index=False, encoding="utf-8-sig")

    effects = []
    for candidate, baseline in COMPARISONS:
        for metric in METRICS:
            pivot = data.pivot(
                index=["training_seed", "scenario_id", "population_seed"],
                columns="arm",
                values=metric,
            )
            paired = pivot[candidate] - pivot[baseline]
            by_seed = paired.groupby("training_seed").mean()
            effects.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_difference": float(by_seed.mean()),
                    "seed_sd": float(by_seed.std(ddof=1)),
                    "seed_min": float(by_seed.min()),
                    "seed_max": float(by_seed.max()),
                    "improving_seeds": int((by_seed < 0).sum()),
                    "worsening_seeds": int((by_seed > 0).sum()),
                    "seed_differences": {str(seed): float(value) for seed, value in by_seed.items()},
                }
            )
    effects_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "seed_differences"} for row in effects])
    effects_frame.to_csv(SOURCE / "paired_seed_effects.csv", index=False, encoding="utf-8-sig")

    rho, p_value = spearmanr(
        seed_summary["mean_release_fraction_nonwait"],
        seed_summary["incapacitation_rate"],
    )
    conflict = data.pivot(
        index=["training_seed", "scenario_id", "conflict_type", "population_seed"],
        columns="arm",
        values="incapacitation_rate",
    )
    conflict["deepseek_minus_raw_pp"] = 100 * (
        conflict["mappo_50_deepseek_retrieval"] - conflict["mappo_34_raw"]
    )
    conflict_summary = conflict.groupby("conflict_type")["deepseek_minus_raw_pp"].agg(["mean", "std", "count"])
    conflict_summary.to_csv(SOURCE / "conflict_effects.csv", encoding="utf-8-sig")
    payload = {
        "status": "analyzed",
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests_during_training": 0,
        "training_seeds": sorted(data.training_seed.unique().astype(int).tolist()),
        "episodes_per_arm": int((data.arm == ARMS[0]).sum()),
        "effects": effects,
        "release_incapacitation_spearman_across_seed_arms": {
            "rho": float(rho),
            "p_value_descriptive_only": float(p_value),
            "points": int(len(seed_summary)),
        },
        "conflict_effects_deepseek_minus_raw_pp": {
            index: {key: float(value) for key, value in row.items()}
            for index, row in conflict_summary.to_dict(orient="index").items()
        },
        "interpretation": (
            "Three seeds are insufficient for inference. DeepSeek-plan effects reverse sign across seeds; "
            "the current result does not establish stable benefit over raw-state MAPPO."
        ),
    }
    (SOURCE / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    effect = next(
        row for row in effects
        if row["candidate"] == "mappo_50_deepseek_retrieval"
        and row["baseline"] == "mappo_34_raw"
        and row["metric"] == "incapacitation_rate"
    )
    rule_effect = next(
        row for row in effects
        if row["candidate"] == "mappo_50_deepseek_retrieval"
        and row["baseline"] == "mappo_50_rule_plan"
        and row["metric"] == "incapacitation_rate"
    )
    report = f"""# MAPPO＋DeepSeek 三种子开发实验报告

本轮只使用12个开发场景、3个训练种子和每种子2个人群种子。每个方法训练10,000步，每臂评估72回合。训练与评估期间DeepSeek API调用为0，最终测试场景未读取。

## 主要结果

DeepSeek检索计划MAPPO相对34维原始MAPPO的平均失能率差为{100*effect['mean_difference']:.3f}个百分点，三个种子差分别为：{', '.join(f'{seed}: {100*value:+.3f} pp' for seed, value in effect['seed_differences'].items())}。{effect['improving_seeds']}个种子改善，{effect['worsening_seeds']}个种子恶化，种子间标准差为{100*effect['seed_sd']:.3f}个百分点。

相对同接口规则计划MAPPO，DeepSeek的平均差为{100*rule_effect['mean_difference']:.3f}个百分点；{rule_effect['improving_seeds']}个种子改善，{rule_effect['worsening_seeds']}个种子恶化。

跨9个“训练种子×方法”点，平均放行比例与失能率的Spearman相关为{rho:.3f}。该相关只用于机制诊断，不作为显著性证据。

## 解释

当前结果不能证明DeepSeek稳定改善疏散。三个训练种子只用于开发诊断，不能支持正式推断；效应方向和放行策略应结合上面的逐种子结果解释。

下一步应检查“计划采用程度”和检索距离，再决定是否扩大训练；现在不应读取最终测试集。
"""
    (SOURCE / "三种子开发实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
