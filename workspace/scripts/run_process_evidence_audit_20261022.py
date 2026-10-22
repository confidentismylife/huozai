"""Zero-cost replay separating successful evacuation progress from episode ending."""

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
from fire_evac.behavior import MixedLogitBehavior
from fire_evac.campaign_extension import ActivePhasePlanWrapper, write_json
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import rule_plan
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv
from fire_evac.event_plan_v3 import EventCorrectedPlanWrapper
from fire_evac.mappo import FactorizedMAPPO, encode_factorized_action
from fire_evac.process_evidence import active_plan_decision, summarize_episode_progress


OUT = ROOT / "outputs" / "process_evidence_audit_20261022"
PUBLIC_REPORT = ROOT / "reports" / "疏散过程证据与指标修正审计_20261022.md"
SEEDS = tuple(range(20261301, 20261306))
ARMS = (previous.RAW, previous.RULE, previous.EXPANDED)
POPULATION = EnvConfig().population


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(seed: int, arm: str) -> Path:
    folder = "shared_raw" if arm == previous.RAW else "e003_i002"
    return previous.OUT / "mechanism" / folder / f"seed_{seed}" / arm / "best.pt"


def plan_payload(plan) -> dict:
    return {
        "phases": [asdict(phase) for phase in plan.phases],
        "confidence": plan.confidence,
        "reason_codes": list(plan.reason_codes),
        "replan_triggers": list(plan.replan_triggers),
    }


def method_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["condition", "arm", "variant"]
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(
            episodes=len(group),
            final_successful_evacuation_rate=group.final_successful_evacuation_rate.mean(),
            final_incapacitation_rate=group.final_incapacitation_rate.mean(),
            final_remaining_rate=group.final_remaining_rate.mean(),
            episode_end_time_seconds=group.episode_end_time_seconds.mean(),
            successful_evacuation_progress_auc=group.successful_evacuation_progress_auc.mean(),
        )
        for percent in (50, 80, 90):
            column = f"time_to_{percent}pct_success_seconds"
            row[f"attained_{percent}pct_success_fraction"] = group[column].notna().mean()
            row[f"conditional_mean_{column}"] = group[column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(means: pd.DataFrame, steps: pd.DataFrame, events: list[dict], verification: dict) -> str:
    lines = [
        "# 疏散过程证据与指标修正审计",
        "",
        "本轮不训练、不调用DeepSeek API、不读取最终封存场景。它以独立回放补充原900回合结果。",
        "原字段 `evacuation_time_seconds` 现明确解释为**回合结束时间**：人员成功疏散或失能都会退出活跃队列，因此该时间不能单独代表疏散效率。",
        "",
        "## 方法均值",
        "",
        "AUC是每个控制步结束后的成功疏散比例的时间平均；越高表示成功疏散发生得越早。阈值时间只对实际达到该成功疏散比例的回合计算，因此同时报告达到比例。",
        "",
        "| 条件 | 方法 | 协议 | 回合 | 成功疏散率 | 失能率 | 滞留率 | 回合结束s | 成功进度AUC | 达到50% | 50%条件时间s | 达到80% | 80%条件时间s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in means.iterrows():
        lines.append(
            f"| {row.condition} | {row.arm} | {row.variant} | {int(row.episodes)} | "
            f"{100*row.final_successful_evacuation_rate:.3f}% | {100*row.final_incapacitation_rate:.3f}% | "
            f"{100*row.final_remaining_rate:.3f}% | {row.episode_end_time_seconds:.2f} | "
            f"{row.successful_evacuation_progress_auc:.4f} | "
            f"{100*row.attained_50pct_success_fraction:.1f}% | {row.conditional_mean_time_to_50pct_success_seconds:.2f} | "
            f"{100*row.attained_80pct_success_fraction:.1f}% | "
            f"{row.conditional_mean_time_to_80pct_success_seconds if pd.notna(row.conditional_mean_time_to_80pct_success_seconds) else '未达到'} |"
        )
    plan_steps = steps[steps.plan_joint_action.notna()].copy()
    by_plan = plan_steps.groupby(["condition", "arm", "variant"], sort=True).agg(
        control_steps=("decision_step", "size"),
        requested_plan_match=("requested_matches_plan", "mean"),
        actual_plan_match=("actual_matches_plan", "mean"),
        mean_actual_release_count=("actual_release_count", "mean"),
    ).reset_index()
    lines += [
        "",
        "## 计划到执行的链条",
        "",
        "一致率只表示MAPPO是否采用当前计划，不证明计划本身正确。实际联合动作还会受到指示牌最短保持时间影响。",
        "",
        "| 条件 | 方法 | 协议 | 控制步 | 请求动作=计划 | 实际动作=计划 | 平均实际放行人数/步 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in by_plan.iterrows():
        lines.append(
            f"| {row.condition} | {row.arm} | {row.variant} | {int(row.control_steps)} | "
            f"{100*row.requested_plan_match:.2f}% | {100*row.actual_plan_match:.2f}% | "
            f"{row.mean_actual_release_count:.2f} |"
        )
    ds_events = [event for event in events if event["arm"] == previous.EXPANDED]
    congestion = [event for event in ds_events if event["event_type"] == "congestion_threshold"]
    same = sum(bool(event["same_event_match"]) for event in congestion)
    lines += [
        "",
        "## 缓存检索覆盖",
        "",
        f"DeepSeek缓存共记录{len(ds_events)}次计划请求，其中拥堵请求{len(congestion)}次；同事件匹配{same}次，跨事件回退{len(congestion)-same}次。",
        "每条请求均保存查询状态、匹配案例、平方距离、候选数和完整计划。距离值只在同一冻结特征尺度内用于诊断，没有预先校准的安全阈值。",
        "",
        "## 结论边界",
        "",
        "1. 回合更早结束不能直接解释为更快疏散；必须与成功疏散率、失能率和成功进度一起阅读。",
        "2. 拥堵计划覆盖仍为空时，新增触发调用的是跨事件近邻计划，不能据此评价DeepSeek生成拥堵方案的能力。",
        "3. 计划采用率能够区分“建议未被采用”和“采用后仍无收益”，但不能独自证明LLM内容有效。",
        "4. 下一步如补充拥堵计划，应另行冻结训练侧状态和费用上限，并把旧库、扩充库、规则和置换计划放在同一请求预算下比较。",
        "",
        "## 核验",
        "",
        f"固定回放{verification['episodes']}回合、{verification['control_steps']}控制步；人口恒等式和单调性通过；与冻结900回合终点最大差{verification['max_endpoint_abs_difference']:.3g}。",
        "API请求0，训练步数0，最终场景未读取。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "complete.json").exists():
        raise RuntimeError("Audit already complete; refusing to overwrite frozen outputs.")
    scenarios = previous.base.make_domain_randomized_trust_bank(24, 20261021, "DSG")[12:]
    conditions = [
        ("normal", EnvConfig(), (42001, 42002)),
        ("bottleneck", EnvConfig(junction_to_branch_capacity=3), (45001,)),
    ]
    inputs = [
        ROOT / "src" / "fire_evac" / "process_evidence.py",
        ROOT / "src" / "fire_evac" / "event_plan_v3.py",
        ROOT / "docs" / "process_evidence_audit_20261022.md",
        Path(__file__),
        previous.OUT / "plan_bank" / "response_audit.json",
    ] + [checkpoint(seed, arm) for seed in SEEDS for arm in ARMS]
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "process-evidence-audit-v1",
        "source_sha256": {str(path): digest(path) for path in inputs},
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "conditions": [
            {"name": name, "config": asdict(config), "population_seeds": list(populations)}
            for name, config, populations in conditions
        ],
        "training_seeds": list(SEEDS),
        "target_episodes": 900,
        "new_API_requests": 0,
        "new_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "manifest.json", manifest)

    episodes: list[dict] = []
    step_rows: list[dict] = []
    event_rows: list[dict] = []
    for condition, config, populations in conditions:
        for seed in SEEDS:
            for arm in ARMS:
                model, _ = FactorizedMAPPO.load(checkpoint(seed, arm))
                variants = ("shared",) if arm == previous.RAW else ("legacy", "v3")
                for variant in variants:
                    physical = ConfirmatoryJointControlWrapper(
                        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios, config=config),
                        advice_mode="none",
                    )
                    library = previous.expanded_library() if arm == previous.EXPANDED else None
                    calls: list[dict] = []

                    def provider(state):
                        if library is None:
                            plan = rule_plan(state)
                            match = None
                            matched_event = None
                        else:
                            plan = library(state)
                            match = dict(library.last_match or {})
                            matched_id = match.get("matched_scenario_id")
                            matched_state = next(
                                stored for stored, _ in library.records
                                if str(stored["scenario_id"]) == str(matched_id)
                            )
                            matched_event = str(matched_state["event_type"])
                        calls.append({
                            "event_type": str(state["event_type"]),
                            "timestamp_step": int(state["timestamp_step"]),
                            "state": dict(state),
                            "matched_scenario_id": None if match is None else match["matched_scenario_id"],
                            "matched_event_type": matched_event,
                            "squared_distance": None if match is None else match["squared_distance"],
                            "candidate_count": None if match is None else match["candidate_count"],
                            "same_event_match": None if match is None else matched_event == state["event_type"],
                            "plan": plan_payload(plan),
                        })
                        return plan

                    wrapper = ActivePhasePlanWrapper if variant == "legacy" else EventCorrectedPlanWrapper
                    env = physical if arm == previous.RAW else wrapper(
                        physical, provider, max_requests_per_episode=3
                    )
                    for scenario in scenarios:
                        for population_seed in populations:
                            calls.clear()
                            identity = {
                                "condition": condition,
                                "training_seed": seed,
                                "arm": arm,
                                "variant": variant,
                                "scenario_id": scenario.scenario_id,
                                "conflict_type": scenario.conflict_type,
                                "population_seed": population_seed,
                            }
                            observation, info = env.reset(
                                seed=population_seed, options={"scenario": scenario}
                            )
                            episode_trace: list[dict] = []
                            decision_step = 0
                            while True:
                                routes, flows, _, _ = model.act(observation, deterministic=True)
                                route, flow = int(routes[0]), int(flows[0])
                                requested = encode_factorized_action(route, flow)
                                plan_decision = (
                                    {"plan_phase_index": None, "plan_route": None,
                                     "plan_release_fraction": None, "plan_joint_action": None}
                                    if arm == previous.RAW
                                    else active_plan_decision(env.current_plan, env.plan_age_steps)
                                )
                                observation, _, terminated, truncated, info = env.step(requested)
                                post = {
                                    "step": decision_step + 1,
                                    "evacuated": int(info["evacuated"]),
                                    "incapacitated": int(info["incapacitated"]),
                                    "remaining": int(info["remaining"]),
                                }
                                episode_trace.append(post)
                                actual_joint = int(info["actual_joint_action"])
                                row = {
                                    **identity,
                                    "decision_step": decision_step,
                                    **post,
                                    "requested_route_action": route,
                                    "requested_flow_action": flow,
                                    "requested_joint_action": requested,
                                    "actual_joint_action": actual_joint,
                                    "requested_release_fraction": float(info["requested_release_fraction"]),
                                    "actual_release_fraction": float(info["actual_release_fraction"]),
                                    "actual_release_capacity": int(info["room_release_capacity"]),
                                    "actual_release_count": int(info["room_release_count"]),
                                    **plan_decision,
                                }
                                row["requested_matches_plan"] = (
                                    None if plan_decision["plan_joint_action"] is None
                                    else requested == plan_decision["plan_joint_action"]
                                )
                                row["actual_matches_plan"] = (
                                    None if plan_decision["plan_joint_action"] is None
                                    else actual_joint == plan_decision["plan_joint_action"]
                                )
                                step_rows.append(row)
                                decision_step += 1
                                if terminated or truncated:
                                    break
                            progress = summarize_episode_progress(
                                episode_trace, population=config.population, step_seconds=5
                            )
                            episodes.append({
                                **identity,
                                **progress,
                                "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
                                "cumulative_congestion": float(info["cumulative_congestion"]),
                                "plan_requests": len(calls),
                            })
                            for index, call in enumerate(calls):
                                event_rows.append({**identity, "request_index": index, **call})
                    env.close()
                    write_json(OUT / "status.json", {
                        "status": "running",
                        "completed_episodes": len(episodes),
                        "target_episodes": 900,
                        "last_completed": {"condition": condition, "seed": seed,
                                           "arm": arm, "variant": variant},
                    })

    episode_frame = pd.DataFrame(episodes)
    step_frame = pd.DataFrame(step_rows)
    if len(episode_frame) != 900:
        raise AssertionError(f"expected 900 episodes, got {len(episode_frame)}")
    identities = ["condition", "training_seed", "arm", "variant", "scenario_id", "population_seed"]
    if episode_frame.duplicated(identities).any():
        raise AssertionError("duplicate episode identity")
    if (episode_frame.plan_requests > 3).any():
        raise AssertionError("request cap exceeded")
    if not np.allclose(
        episode_frame.final_successful_evacuation_rate
        + episode_frame.final_incapacitation_rate
        + episode_frame.final_remaining_rate,
        1.0,
    ):
        raise AssertionError("endpoint population balance failed")

    frozen = pd.read_csv(ROOT / "outputs" / "event_trigger_v3_validation_20261022" / "episode_results.csv")
    merged = episode_frame.merge(frozen, on=identities, validate="one_to_one", suffixes=("_new", "_old"))
    comparisons = {
        "final_incapacitation_rate": "incapacitation_rate",
        "episode_end_time_seconds": "evacuation_time_seconds",
        "mean_cumulative_fed": "mean_cumulative_fed",
        "cumulative_congestion": "cumulative_congestion",
    }
    differences = []
    for new, old in comparisons.items():
        new_column = f"{new}_new" if new in frozen.columns else new
        old_column = f"{old}_old" if old in episode_frame.columns else old
        differences.extend(np.abs(merged[new_column] - merged[old_column]).tolist())
    max_difference = float(max(differences))
    if max_difference > 1e-12:
        raise AssertionError(f"frozen endpoint replay mismatch: {max_difference}")

    means = method_table(episode_frame)
    episode_frame.to_csv(OUT / "episode_process_metrics.csv", index=False)
    step_frame.to_csv(OUT / "step_process.csv", index=False)
    means.to_csv(OUT / "method_means.csv", index=False)
    with (OUT / "request_audit.jsonl").open("w", encoding="utf-8") as handle:
        for event in event_rows:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    verification = {
        "episodes": len(episode_frame),
        "control_steps": len(step_frame),
        "requests": len(event_rows),
        "max_endpoint_abs_difference": max_difference,
        "population_balance_passed": True,
        "request_cap_passed": True,
        "new_API_requests": 0,
        "new_training_steps": 0,
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "analysis.json", {
        "verification": verification,
        "method_means": means.replace({np.nan: None}).to_dict(orient="records"),
    })
    report = render_report(means, step_frame, event_rows, verification)
    (OUT / "实验报告.md").write_text(report, encoding="utf-8")
    PUBLIC_REPORT.write_text(report, encoding="utf-8")
    complete = {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), **verification}
    write_json(OUT / "complete.json", complete)
    write_json(OUT / "status.json", complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
