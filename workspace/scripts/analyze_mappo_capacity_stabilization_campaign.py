from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "mappo_capacity_stabilization_20261022"
DEEPSEEK = "mappo_50_deepseek_retrieval"
RAW = "mappo_34_raw"
RULE = "mappo_50_rule_plan"


def interval(values: pd.Series, seed: int = 20261022) -> dict:
    array = values.to_numpy(dtype=float)
    if not len(array):
        return {"mean": None, "seed_sd": None, "ci95": [None, None], "n_seeds": 0}
    rng = np.random.default_rng(seed)
    boot = array[rng.integers(0, len(array), (10_000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "seed_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci95": [float(value) for value in np.quantile(boot, (0.025, 0.975))],
        "n_seeds": int(len(array)),
        "by_seed": {str(index): float(value) for index, value in values.items()},
    }


def paired_effects(data: pd.DataFrame, group_column: str, candidate: str, baseline: str):
    metrics = (
        "incapacitation_rate",
        "mean_cumulative_fed",
        "evacuation_time_seconds",
        "cumulative_congestion",
        "mean_release_fraction_nonwait",
    )
    output = {}
    for metric in metrics:
        pivot = data.pivot(
            index=["training_seed", "scenario_id", "population_seed"],
            columns=group_column,
            values=metric,
        )
        paired = pivot[candidate] - pivot[baseline]
        by_seed = paired.groupby("training_seed").mean()
        output[metric] = interval(by_seed)
    return output


def main() -> int:
    status = json.loads((SOURCE / "campaign_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError("campaign is not complete")
    selection = json.loads((SOURCE / "selection.json").read_text(encoding="utf-8"))

    screen_rows = []
    for path in sorted((SOURCE / "screen").glob("*/seed_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation"]
        screen_rows.append(
            {
                "config_id": path.parents[1].name,
                "training_seed": int(payload["seed"]),
                "selected_timesteps": int(payload["selected_checkpoint"]["timesteps"]),
                "flow_entropy_coefficient": float(payload["config"]["flow_entropy_coefficient"]),
                "plan_imitation_coefficient": float(payload["config"]["plan_imitation_coefficient"]),
                **validation,
            }
        )
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(SOURCE / "screen_seed_results.csv", index=False, encoding="utf-8-sig")
    screen_summary = screen.groupby("config_id", as_index=False).agg(
        seeds=("training_seed", "count"),
        incapacitation_mean=("mean_incapacitation_rate", "mean"),
        incapacitation_seed_sd=("mean_incapacitation_rate", "std"),
        release_mean=("mean_release_fraction_nonwait", "mean"),
        full_release_fraction=("full_release_episode_fraction", "mean"),
        route_plan_match=("mean_route_plan_match", "mean"),
        flow_plan_match=("mean_flow_plan_match", "mean"),
        selected_timesteps_mean=("selected_timesteps", "mean"),
    )
    screen_summary.to_csv(SOURCE / "screen_config_summary.csv", index=False, encoding="utf-8-sig")

    confirmation_frames = []
    for path in sorted((SOURCE / "confirmation").glob("seed_*/*/episode_results.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "training_seed", int(path.parents[1].name.split("_")[-1]))
        confirmation_frames.append(frame)
    confirmation = pd.concat(confirmation_frames, ignore_index=True)
    confirmation.to_csv(SOURCE / "confirmation_combined.csv", index=False, encoding="utf-8-sig")
    confirmation_summary = confirmation.groupby(["training_seed", "arm"], as_index=False).agg(
        incapacitation_rate=("incapacitation_rate", "mean"),
        mean_cumulative_fed=("mean_cumulative_fed", "mean"),
        evacuation_time_seconds=("evacuation_time_seconds", "mean"),
        cumulative_congestion=("cumulative_congestion", "mean"),
        mean_release_fraction_nonwait=("mean_release_fraction_nonwait", "mean"),
        full_release_episode_fraction=("full_release_episode", "mean"),
        route_plan_match=("route_plan_match", "mean"),
        flow_plan_match=("flow_plan_match", "mean"),
    )
    confirmation_summary.to_csv(
        SOURCE / "confirmation_seed_arm_summary.csv", index=False, encoding="utf-8-sig"
    )
    effects = {
        "deepseek_minus_raw": paired_effects(confirmation, "arm", DEEPSEEK, RAW),
        "deepseek_minus_rule": paired_effects(confirmation, "arm", DEEPSEEK, RULE),
    }

    audit = pd.read_csv(SOURCE / "channel_audit.csv")
    audit_effects = {}
    for variant in ("rule", "shuffled", "masked"):
        audit_effects[f"native_minus_{variant}"] = paired_effects(
            audit, "provider_variant", "native", variant
        )
    audit_summary = audit.groupby(["training_seed", "provider_variant"], as_index=False).agg(
        incapacitation_rate=("incapacitation_rate", "mean"),
        mean_cumulative_fed=("mean_cumulative_fed", "mean"),
        evacuation_time_seconds=("evacuation_time_seconds", "mean"),
        mean_release_fraction_nonwait=("mean_release_fraction_nonwait", "mean"),
        route_plan_match=("route_plan_match", "mean"),
        flow_plan_match=("flow_plan_match", "mean"),
    )
    audit_summary.to_csv(SOURCE / "channel_audit_seed_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "analyzed",
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests": 0,
        "campaign_elapsed_hours": status.get("elapsed_hours"),
        "selected_config": selection["selected"],
        "screen_configs": screen_summary.to_dict(orient="records"),
        "confirmation_training_seeds": sorted(
            confirmation.training_seed.unique().astype(int).tolist()
        ),
        "confirmation_effects": effects,
        "channel_effects": audit_effects,
        "inference_scope": (
            "development-only seed bootstrap; all scenarios come from the development generator "
            "and the final frozen test set remains unread"
        ),
    }
    (SOURCE / "final_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chosen = selection["selected"]
    ds_raw = effects["deepseek_minus_raw"]["incapacitation_rate"]
    ds_rule = effects["deepseek_minus_rule"]["incapacitation_rate"]
    native_rule = audit_effects["native_minus_rule"]["incapacitation_rate"]
    native_shuffle = audit_effects["native_minus_shuffled"]["incapacitation_rate"]
    native_mask = audit_effects["native_minus_masked"]["incapacitation_rate"]
    screen_md = "\n".join(
        f"| {row.config_id} | {100*row.incapacitation_mean:.3f}% | {row.release_mean:.3f} | "
        f"{100*row.full_release_fraction:.1f}% | {row.route_plan_match:.3f} | {row.flow_plan_match:.3f} |"
        for row in screen_summary.itertuples()
    )
    report = f"""# MAPPO容量策略稳定化与DeepSeek计划作用实验报告

## 实验边界

本轮连续运行约{float(status.get('elapsed_hours', 0)):.2f}小时，只使用预注册的开发场景、冻结DeepSeek计划和新训练种子。DeepSeek API调用为0，新增费用为0，最终测试场景未读取。筛选阶段每个设置使用{int(screen.groupby('config_id').training_seed.nunique().min())}个种子；确认阶段使用{confirmation.training_seed.nunique()}个全新训练种子。所有方法使用相同训练预算、动作空间和开发验证协议。

## 预注册筛选

| 设置 | 失能率 | 平均放行 | 全量放行回合 | 方向计划匹配 | 流量计划匹配 |
|---|---:|---:|---:|---:|---:|
{screen_md}

按预注册规则选中 `{chosen['config_id']}`：先排除平均全量放行比例超过85%的设置，再最小化失能率，以计划匹配率作为后续判据。选中设置的筛选失能率为{100*chosen['mean_incapacitation_rate']:.3f}%，流量计划匹配率为{chosen['mean_flow_plan_match']:.3f}。

## 新种子三臂确认

DeepSeek计划MAPPO相对34维MAPPO的失能率差为{100*ds_raw['mean']:+.3f}个百分点，种子重采样95%区间为[{100*ds_raw['ci95'][0]:+.3f}, {100*ds_raw['ci95'][1]:+.3f}]；相对规则计划MAPPO为{100*ds_rule['mean']:+.3f}个百分点，区间[{100*ds_rule['ci95'][0]:+.3f}, {100*ds_rule['ci95'][1]:+.3f}]。这些区间只描述开发种子，不替代最终封存评估。

## 计划内容通道审计

同一个DeepSeek训练模型保持权重、场景和人口随机数不变，只替换计划输入：

- 真实DeepSeek计划 − 规则计划：{100*native_rule['mean']:+.3f}个百分点失能；
- 真实DeepSeek计划 − 同事件类型置换计划：{100*native_shuffle['mean']:+.3f}个百分点；
- 真实DeepSeek计划 − 屏蔽全部计划特征：{100*native_mask['mean']:+.3f}个百分点。

若真实计划同时优于规则、置换和屏蔽版本，才说明模型利用了DeepSeek计划内容；只优于屏蔽而不优于规则或置换，说明有“计划通道效应”，但没有DeepSeek特异增益。反事实替换会产生输入分布偏移，因此这里只作机制证据。

## 可辩护结论

本轮首先检验并修正容量actor退化，再用独立开发种子检验DeepSeek计划是否产生增量。最终论文结论应严格依据上面的配对方向与区间：区间跨0时写“未检出稳定差异”；真实计划未同时优于三个通道对照时，不宣称LLM带来特异收益。最终测试集继续封存，只有开发方案冻结后才允许读取一次。
"""
    (SOURCE / "MAPPO容量策略稳定化与DeepSeek计划作用实验报告.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
