"""Finite fixed-weight evaluation of training-side congestion plan coverage."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mappo_six_hour_extension as previous
from analyze_mappo_six_hour_extension import paired_effect
from fire_evac.behavior import MixedLogitBehavior
from fire_evac.campaign_extension import write_json
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import parse_plan, rule_plan
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv
from fire_evac.event_plan_v3 import EventCorrectedPlanWrapper
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from fire_evac.mappo import FactorizedMAPPO, encode_factorized_action
from fire_evac.process_evidence import (
    active_plan_decision,
    cyclically_permute_plans,
    summarize_episode_progress,
)


SOURCE = ROOT / "outputs" / "congestion_plan_coverage_20261022"
OUT = ROOT / "outputs" / "congestion_plan_coverage_validation_20261022"
REPORT = ROOT / "reports" / "拥堵计划覆盖有限验证_20261022.md"
SEEDS = tuple(range(20261301, 20261306))
VARIANTS = ("old_library", "expanded_library", "shuffled_congestion", "rule_substitution")
METRICS = (
    "incapacitation_rate",
    "successful_evacuation_rate",
    "episode_end_time_seconds",
    "successful_evacuation_progress_auc",
    "mean_cumulative_fed",
    "cumulative_congestion",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(seed: int, arm: str) -> Path:
    folder = "shared_raw" if arm == previous.RAW else "e003_i002"
    return previous.OUT / "mechanism" / folder / f"seed_{seed}" / arm / "best.pt"


def libraries() -> tuple[FrozenPlanLibrary, FrozenPlanLibrary, FrozenPlanLibrary, list[dict]]:
    old = previous.expanded_library()
    states = json.loads((SOURCE / "frozen_congestion_states.json").read_text(encoding="utf-8"))
    audits = json.loads((SOURCE / "response_audit.json").read_text(encoding="utf-8"))
    by_id = {state["scenario_id"]: state for state in states}
    new_records = [
        (by_id[item["scenario_id"]], parse_plan(item["plan"]))
        for item in audits if item["valid_model_response"]
    ]
    if len(new_records) != len(states):
        raise RuntimeError("Coverage gate did not provide one valid plan per frozen state.")
    expanded = FrozenPlanLibrary(list(old.records) + new_records)
    shuffled_records = cyclically_permute_plans(new_records)
    shuffled = FrozenPlanLibrary(list(old.records) + shuffled_records)
    permutation = [
        {
            "state_scenario_id": state["scenario_id"],
            "assigned_plan_from": new_records[(index + 1) % len(new_records)][0]["scenario_id"],
        }
        for index, (state, _) in enumerate(new_records)
    ]
    return old, expanded, shuffled, permutation


def evaluate(model, scenarios, config, seed: int, method: str, provider=None) -> tuple[list[dict], list[dict]]:
    physical = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios, config=config), advice_mode="none"
    )
    env = physical if provider is None else EventCorrectedPlanWrapper(
        physical, provider, max_requests_per_episode=3
    )
    episodes, steps = [], []
    for scenario in scenarios:
        observation, info = env.reset(seed=45001, options={"scenario": scenario})
        trace = []
        index = 0
        while True:
            routes, flows, _, _ = model.act(observation, deterministic=True)
            route, flow = int(routes[0]), int(flows[0])
            action = encode_factorized_action(route, flow)
            plan = (
                {"plan_route": None, "plan_release_fraction": None, "plan_joint_action": None}
                if provider is None
                else active_plan_decision(env.current_plan, env.plan_age_steps)
            )
            observation, _, terminated, truncated, info = env.step(action)
            post = {
                "step": index + 1,
                "evacuated": int(info["evacuated"]),
                "incapacitated": int(info["incapacitated"]),
                "remaining": int(info["remaining"]),
            }
            trace.append(post)
            steps.append({
                "training_seed": seed,
                "scenario_id": scenario.scenario_id,
                "population_seed": 45001,
                "method": method,
                "decision_step": index,
                "requested_route_action": route,
                "requested_flow_action": flow,
                "requested_joint_action": action,
                "actual_joint_action": int(info["actual_joint_action"]),
                "requested_release_fraction": float(info["requested_release_fraction"]),
                "actual_release_fraction": float(info["actual_release_fraction"]),
                "actual_release_count": int(info["room_release_count"]),
                **plan,
            })
            index += 1
            if terminated or truncated:
                break
        progress = summarize_episode_progress(trace, config.population, step_seconds=5)
        episodes.append({
            "training_seed": seed,
            "scenario_id": scenario.scenario_id,
            "conflict_type": scenario.conflict_type,
            "population_seed": 45001,
            "method": method,
            "incapacitation_rate": progress["final_incapacitation_rate"],
            "successful_evacuation_rate": progress["final_successful_evacuation_rate"],
            "episode_end_time_seconds": progress["episode_end_time_seconds"],
            "successful_evacuation_progress_auc": progress["successful_evacuation_progress_auc"],
            "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
            "cumulative_congestion": float(info["cumulative_congestion"]),
            "plan_requests": int(info.get("deepseek_requests_this_episode", 0)),
        })
    env.close()
    return episodes, steps


def means(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("method", sort=True).agg(
        episodes=("scenario_id", "size"),
        incapacitation_rate=("incapacitation_rate", "mean"),
        successful_evacuation_rate=("successful_evacuation_rate", "mean"),
        episode_end_time_seconds=("episode_end_time_seconds", "mean"),
        successful_evacuation_progress_auc=("successful_evacuation_progress_auc", "mean"),
        mean_cumulative_fed=("mean_cumulative_fed", "mean"),
        cumulative_congestion=("cumulative_congestion", "mean"),
        plan_requests=("plan_requests", "mean"),
    ).reset_index()


def report_text(summary: pd.DataFrame, effects: dict, mechanism: pd.DataFrame, ledger: dict) -> str:
    lines = [
        "# 拥堵计划覆盖有限验证",
        "",
        "本轮使用训练侧9个拥堵快照生成DeepSeek计划，再用固定MAPPO权重在既有开发验证场景的人工瓶颈条件中做有限对照。没有重新训练，没有读取最终场景。",
        "",
        "## 方法均值",
        "",
        "时间为回合结束时间，不能脱离成功疏散率和失能率解释。",
        "",
        "| 方法 | 回合 | 失能率 | 成功疏散率 | 回合结束s | 成功进度AUC | FED | 拥堵 | 请求/回合 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.method} | {int(row.episodes)} | {100*row.incapacitation_rate:.3f}% | "
            f"{100*row.successful_evacuation_rate:.3f}% | {row.episode_end_time_seconds:.2f} | "
            f"{row.successful_evacuation_progress_auc:.4f} | {row.mean_cumulative_fed:.4f} | "
            f"{row.cumulative_congestion:.2f} | {row.plan_requests:.2f} |"
        )
    lines += ["", "## 固定DeepSeek权重的配对差异", "",
              "失能差负值较好；AUC差正值表示成功疏散进度更早。区间为训练种子×开发场景交叉重采样，属于探索证据。", "",
              "| 比较 | 失能差pp [95%区间] | 成功率差pp | 回合结束差s | AUC差 |", "|---|---:|---:|---:|---:|"]
    for name, values in effects.items():
        incap = values["incapacitation_rate"]
        success = values["successful_evacuation_rate"]
        end = values["episode_end_time_seconds"]
        auc = values["successful_evacuation_progress_auc"]
        lo, hi = incap["crossed_ci95"]
        lines.append(
            f"| {name} | {100*incap['mean']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] | "
            f"{100*success['mean']:+.3f} | {end['mean']:+.2f} | {auc['mean']:+.4f} |"
        )
    lines += ["", "## 采用机制", "",
              "| 方法 | 计划控制步 | 方向一致 | 容量一致 | 完全一致 | 平均实际放行人数 |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in mechanism.iterrows():
        lines.append(
            f"| {row.method} | {int(row.steps)} | {100*row.route_match:.2f}% | "
            f"{100*row.flow_match:.2f}% | {100*row.joint_match:.2f}% | {row.release_count:.2f} |"
        )
    expanded_old = effects["expanded_library - old_library"]["incapacitation_rate"]
    expanded_shuffle = effects["expanded_library - shuffled_congestion"]["incapacitation_rate"]
    lines += [
        "",
        "## 判断",
        "",
        f"扩充库减旧库的失能差为{100*expanded_old['mean']:+.3f}个百分点；扩充库减置换拥堵计划为{100*expanded_shuffle['mean']:+.3f}个百分点。",
        "只有当扩充库优于旧库，同时正确匹配优于置换，才能支持‘拥堵状态与计划内容的对应关系有用’。区间跨零时应写成证据不足。",
        "固定权重没有见过这些新拥堵计划，任何结果都只属于推理时分布变化诊断，不能替代重新训练后的公平比较。",
        "",
        "## 调用与边界",
        "",
        f"DeepSeek新请求{ledger['attempted_requests']}次，成功{ledger['successful_requests']}次，保守费用{ledger['actual_cost_upper_cny']:.6f}元。训练场景快照9个，时间和交汇占比高度集中，因此计划覆盖仍有限。",
        "本轮共360回合；最终48个场景未读取。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "complete.json").exists():
        raise RuntimeError("Coverage validation already complete; refusing to overwrite.")
    gate = json.loads((SOURCE / "gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise RuntimeError("DeepSeek congestion plan gate did not pass.")
    old, expanded, shuffled, permutation = libraries()
    scenarios = previous.base.make_domain_randomized_trust_bank(24, 20261021, "DSG")[12:]
    config = EnvConfig(junction_to_branch_capacity=3)
    paths = [Path(__file__), SOURCE / "frozen_congestion_states.json", SOURCE / "response_audit.json"]
    paths += [model_path(seed, arm) for seed in SEEDS for arm in (previous.RAW, previous.RULE, previous.EXPANDED)]
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fixed-weight-congestion-coverage-v1",
        "source_sha256": {str(path): sha(path) for path in paths},
        "training_seeds": list(SEEDS),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "population_seed": 45001,
        "variants": list(VARIANTS),
        "congestion_plan_permutation": permutation,
        "target_episodes": 360,
        "new_API_requests": 0,
        "new_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "manifest.json", manifest)
    episode_rows, step_rows = [], []
    for seed in SEEDS:
        ds_model, _ = FactorizedMAPPO.load(model_path(seed, previous.EXPANDED))
        providers = {
            "old_library": old,
            "expanded_library": expanded,
            "shuffled_congestion": shuffled,
            "rule_substitution": rule_plan,
        }
        for method, provider in providers.items():
            episodes, steps = evaluate(ds_model, scenarios, config, seed, method, provider)
            episode_rows.extend(episodes); step_rows.extend(steps)
        raw_model, _ = FactorizedMAPPO.load(model_path(seed, previous.RAW))
        episodes, steps = evaluate(raw_model, scenarios, config, seed, "raw34_system", None)
        episode_rows.extend(episodes); step_rows.extend(steps)
        rule_model, _ = FactorizedMAPPO.load(model_path(seed, previous.RULE))
        episodes, steps = evaluate(rule_model, scenarios, config, seed, "trained_rule_system", rule_plan)
        episode_rows.extend(episodes); step_rows.extend(steps)
        write_json(OUT / "status.json", {"status": "running", "completed_episodes": len(episode_rows), "target_episodes": 360})
    frame = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    if len(frame) != 360:
        raise AssertionError(f"expected 360 episodes, got {len(frame)}")
    identities = ["training_seed", "scenario_id", "population_seed", "method"]
    if frame.duplicated(identities).any():
        raise AssertionError("duplicate episode identities")
    fixed = frame[frame.method.isin(VARIANTS)]
    comparisons = {
        "expanded_library - old_library": ("expanded_library", "old_library"),
        "expanded_library - shuffled_congestion": ("expanded_library", "shuffled_congestion"),
        "expanded_library - rule_substitution": ("expanded_library", "rule_substitution"),
    }
    effects = {
        label: {metric: paired_effect(fixed, "method", candidate, baseline, metric) for metric in METRICS}
        for label, (candidate, baseline) in comparisons.items()
    }
    plan_steps = steps[steps.plan_joint_action.notna()].copy()
    plan_steps["plan_route_action"] = plan_steps.plan_route.map({"north": 0, "south": 1, "wait": 2})
    plan_steps["route_match"] = plan_steps.requested_route_action.eq(plan_steps.plan_route_action)
    flow_target = plan_steps.plan_release_fraction.map({0.25: 0, 0.5: 1, 0.75: 2, 1.0: 3})
    plan_steps["flow_match"] = plan_steps.requested_flow_action.eq(flow_target).astype(float)
    plan_steps.loc[plan_steps.plan_route.eq("wait"), "flow_match"] = np.nan
    plan_steps["joint_match"] = plan_steps.requested_joint_action.eq(plan_steps.plan_joint_action)
    mechanism = plan_steps.groupby("method", sort=True).agg(
        steps=("decision_step", "size"), route_match=("route_match", "mean"),
        flow_match=("flow_match", "mean"), joint_match=("joint_match", "mean"),
        release_count=("actual_release_count", "mean"),
    ).reset_index()
    summary = means(frame)
    frame.to_csv(OUT / "episode_results.csv", index=False)
    steps.to_csv(OUT / "step_results.csv", index=False)
    summary.to_csv(OUT / "method_means.csv", index=False)
    mechanism.to_csv(OUT / "mechanism.csv", index=False)
    write_json(OUT / "analysis.json", {"effects": effects, "method_means": summary.to_dict(orient="records")})
    text = report_text(summary, effects, mechanism, gate["ledger"])
    (OUT / "实验报告.md").write_text(text, encoding="utf-8")
    REPORT.write_text(text, encoding="utf-8")
    complete = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": len(frame), "control_steps": len(steps), "new_API_requests": 0,
        "new_training_steps": 0, "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "complete.json", complete)
    write_json(OUT / "status.json", complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
