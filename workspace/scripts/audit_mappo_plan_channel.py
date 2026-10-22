from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import EvacuationPlan, rule_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_domain_randomized_trust_bank
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from fire_evac.mappo import FLOW_LEVELS, FactorizedMAPPO, encode_factorized_action


FORMAT_GATE = ROOT / "outputs" / "deepseek_format_gate_20261021"
DEFAULT_SOURCE = ROOT / "outputs" / "mappo_independent_actor_multiseed_20261021"
DEFAULT_OUTPUT = ROOT / "outputs" / "mappo_plan_channel_audit_20261021"
MODEL_ARMS = ("mappo_50_rule_plan", "mappo_50_deepseek_retrieval")


class RecordingLibrary:
    def __init__(self, library: FrozenPlanLibrary):
        self.library = library
        self.matches: list[dict[str, Any]] = []

    def __call__(self, state: Mapping[str, Any]) -> EvacuationPlan:
        plan = self.library(state)
        self.matches.append(dict(self.library.last_match or {}))
        return plan


def make_plan_env(scenarios, provider):
    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    return DeepSeekPlanObservationWrapper(base, provider, max_requests_per_episode=3)


def run_episode(model, env, scenario, population_seed: int) -> dict[str, Any]:
    observation, _ = env.reset(seed=population_seed, options={"scenario": scenario})
    initial_observation = np.asarray(observation, dtype=np.float32).copy()
    actions: list[int] = []
    release_total = 0.0
    nonwait = 0
    waits = 0
    while True:
        routes, flows, _, _ = model.act(observation, deterministic=True)
        action = encode_factorized_action(int(routes[0]), int(flows[0]))
        actions.append(action)
        if action == 8:
            waits += 1
        else:
            release_total += FLOW_LEVELS[action % len(FLOW_LEVELS)]
            nonwait += 1
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return {
        "initial_observation": initial_observation,
        "actions": actions,
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "evacuation_time_seconds": 5.0 * float(info["step"]),
        "mean_release_fraction_nonwait": release_total / max(1, nonwait),
        "wait_fraction": waits / max(1, len(actions)),
        "plan_requests": int(info.get("deepseek_requests_this_episode", 0)),
    }


def action_comparison(left: list[int], right: list[int]) -> tuple[bool, int, float]:
    common = min(len(left), len(right))
    first_difference = next((index for index in range(common) if left[index] != right[index]), common)
    identical = left == right
    return identical, first_difference, first_difference / max(1, common)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether trained MAPPO policies use plan content")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scenarios = make_domain_randomized_trust_bank(12, 20261021, "DSG")
    population_seeds = (31001, 31002)
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(path for path in args.source.glob("seed_*") if path.is_dir()):
        training_seed = int(seed_dir.name.split("_")[-1])
        for model_arm in MODEL_ARMS:
            model, _ = FactorizedMAPPO.load(seed_dir / f"{model_arm}.pt")
            library = FrozenPlanLibrary.from_files(
                FORMAT_GATE / "frozen_development_states.json",
                FORMAT_GATE / "response_audit.json",
            )
            recorder = RecordingLibrary(library)
            deepseek_env = make_plan_env(scenarios, recorder)
            rule_env = make_plan_env(scenarios, rule_plan)
            for scenario in scenarios:
                for population_seed in population_seeds:
                    recorder.matches.clear()
                    deepseek = run_episode(model, deepseek_env, scenario, population_seed)
                    matches = list(recorder.matches)
                    rule = run_episode(model, rule_env, scenario, population_seed)
                    identical, first_difference, prefix_fraction = action_comparison(
                        deepseek["actions"], rule["actions"]
                    )
                    distances = [float(match["squared_distance"]) for match in matches]
                    rows.append(
                        {
                            "training_seed": training_seed,
                            "model_arm": model_arm,
                            "scenario_id": scenario.scenario_id,
                            "conflict_type": scenario.conflict_type,
                            "population_seed": population_seed,
                            "initial_plan_l1_difference": float(
                                np.abs(
                                    deepseek["initial_observation"][28:49]
                                    - rule["initial_observation"][28:49]
                                ).sum()
                            ),
                            "initial_action_changed": int(
                                deepseek["actions"][0] != rule["actions"][0]
                            ),
                            "action_sequences_identical": int(identical),
                            "first_action_difference_step": first_difference,
                            "matching_prefix_fraction": prefix_fraction,
                            "deepseek_minus_rule_incapacitation_rate": (
                                deepseek["incapacitation_rate"] - rule["incapacitation_rate"]
                            ),
                            "deepseek_minus_rule_mean_cumulative_fed": (
                                deepseek["mean_cumulative_fed"] - rule["mean_cumulative_fed"]
                            ),
                            "deepseek_minus_rule_evacuation_time_seconds": (
                                deepseek["evacuation_time_seconds"] - rule["evacuation_time_seconds"]
                            ),
                            "deepseek_minus_rule_cumulative_congestion": (
                                deepseek["cumulative_congestion"] - rule["cumulative_congestion"]
                            ),
                            "deepseek_release_fraction": deepseek["mean_release_fraction_nonwait"],
                            "rule_release_fraction": rule["mean_release_fraction_nonwait"],
                            "deepseek_wait_fraction": deepseek["wait_fraction"],
                            "rule_wait_fraction": rule["wait_fraction"],
                            "deepseek_plan_requests": deepseek["plan_requests"],
                            "rule_plan_requests": rule["plan_requests"],
                            "retrieval_count": len(distances),
                            "retrieval_squared_distance_mean": float(np.mean(distances)),
                            "retrieval_squared_distance_max": float(np.max(distances)),
                            "matched_scenario_ids": ";".join(
                                str(match["matched_scenario_id"]) for match in matches
                            ),
                        }
                    )
            deepseek_env.close()
            rule_env.close()

    csv_path = args.output / "paired_plan_channel_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "status": "complete",
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests": 0,
        "paired_comparisons": len(rows),
        "evaluation_episodes": 2 * len(rows),
        "models": {},
    }
    for model_arm in MODEL_ARMS:
        subset = [row for row in rows if row["model_arm"] == model_arm]
        by_seed: dict[str, Any] = {}
        for seed in sorted({int(row["training_seed"]) for row in subset}):
            group = [row for row in subset if int(row["training_seed"]) == seed]
            by_seed[str(seed)] = {
                "initial_action_change_rate": float(np.mean([row["initial_action_changed"] for row in group])),
                "any_action_change_rate": float(np.mean([1 - row["action_sequences_identical"] for row in group])),
                "mean_matching_prefix_fraction": float(np.mean([row["matching_prefix_fraction"] for row in group])),
                "deepseek_minus_rule_incapacitation_pp": float(
                    100 * np.mean([row["deepseek_minus_rule_incapacitation_rate"] for row in group])
                ),
            }
        summary["models"][model_arm] = {
            "initial_plan_l1_difference_mean": float(
                np.mean([row["initial_plan_l1_difference"] for row in subset])
            ),
            "initial_action_change_rate": float(np.mean([row["initial_action_changed"] for row in subset])),
            "any_action_change_rate": float(
                np.mean([1 - row["action_sequences_identical"] for row in subset])
            ),
            "mean_matching_prefix_fraction": float(
                np.mean([row["matching_prefix_fraction"] for row in subset])
            ),
            "deepseek_minus_rule_incapacitation_pp": float(
                100 * np.mean([row["deepseek_minus_rule_incapacitation_rate"] for row in subset])
            ),
            "deepseek_minus_rule_mean_cumulative_fed": float(
                np.mean([row["deepseek_minus_rule_mean_cumulative_fed"] for row in subset])
            ),
            "retrieval_squared_distance_mean": float(
                np.mean([row["retrieval_squared_distance_mean"] for row in subset])
            ),
            "retrieval_squared_distance_max": float(
                np.max([row["retrieval_squared_distance_max"] for row in subset])
            ),
            "by_seed": by_seed,
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    deepseek_model = summary["models"]["mappo_50_deepseek_retrieval"]
    report = f"""# MAPPO计划通道采用度审计

本审计只使用开发场景、冻结的DeepSeek计划和已训练检查点。没有调用DeepSeek API，也没有读取最终测试场景。对每个50维模型保持场景、人口种子和权重不变，只切换DeepSeek计划与同接口规则计划，共完成{len(rows)}组配对、{2*len(rows)}个评估回合。

## DeepSeek计划训练模型

- 初始计划特征的平均L1差异：{deepseek_model['initial_plan_l1_difference_mean']:.3f}。
- 首动作随计划切换而改变的回合比例：{100*deepseek_model['initial_action_change_rate']:.1f}%。
- 任一动作随计划切换而改变的回合比例：{100*deepseek_model['any_action_change_rate']:.1f}%。
- 两个动作序列在首次分叉前的平均匹配比例：{100*deepseek_model['mean_matching_prefix_fraction']:.1f}%。
- 使用DeepSeek计划相对换成规则计划的失能率差：{deepseek_model['deepseek_minus_rule_incapacitation_pp']:+.3f}个百分点。
- 检索平方距离均值/最大值：{deepseek_model['retrieval_squared_distance_mean']:.3f} / {deepseek_model['retrieval_squared_distance_max']:.3f}。

## 解释边界

计划切换后动作是否变化，用于判断策略是否读取计划内容；它不等于计划带来净收益。把已训练策略切换到另一类计划属于反事实通道敏感性测试，可能存在分布偏移。正式有效性仍需同信息、同训练预算的独立训练对照与封存测试集。
"""
    (args.output / "计划通道采用度审计.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
