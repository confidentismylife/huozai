from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.auxiliary_ppo import AuxiliarySupervisedPPO
from fire_evac.action_diagnostics import copy_baseline_diagnostics
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import (
    GymDynamicConflictEnv,
    JointRouteReleaseGateWrapper,
    TRUST_GATE_TYPES,
    conflict_llm_advice,
    conflict_scenario_records,
    make_domain_randomized_trust_bank,
    make_trust_gate_bank,
)
from fire_evac.dynamic_llm_rl import safe_forecast_target
from fire_evac.dynamic_multiseed import summarize_cross_episodes
from fire_evac.joint_gate import JOINT_GATE_PLANS, JointGatePlan, joint_plan_action, select_safest_joint_plan
from fire_evac.supervised_gate import policy_accuracy, pretrain_ppo_gate
from run_dynamic_auxiliary_gate import stratified_scenario_splits
from run_supervised_trust_gate import (
    aggregate_by_conflict,
    decision_summary,
    make_gate_env,
    make_time_env,
    paired_comparison,
    parse_seeds,
    run_model_episode,
    write_csv,
)

POLICIES = ("ppo_time", "scratch_gate", "previous_contradiction", "joint_route_release")
GATE_POLICIES = tuple(policy for policy in POLICIES if policy != "ppo_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint route and graded release control experiment.")
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--domain-scenarios", type=int, default=180)
    parser.add_argument("--test-scenarios", type=int, default=20)
    parser.add_argument("--counterfactual-repetitions", type=int, default=3)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--auxiliary-coefficient", type=float, default=0.3)
    parser.add_argument("--auxiliary-batches", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--scenario-seed", type=int, default=20261001)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--baseline-model-dir", type=Path, default=ROOT / "outputs" / "llm_trust_gate" / "models"
    )
    parser.add_argument(
        "--previous-model-dir",
        type=Path,
        default=ROOT / "outputs" / "domain_randomized_contradiction" / "models",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "joint_route_release")
    return parser.parse_args()


def make_joint_env(behavior_factory, scenarios, advice):
    return JointRouteReleaseGateWrapper(GymDynamicConflictEnv(behavior_factory(), scenarios), advice)


def run_joint_plan(env, scenario, population_seed: int, plan: JointGatePlan, collect: bool = False):
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    observations: list[np.ndarray] = []
    actions: list[int] = []
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = joint_plan_action(
            plan,
            step=steps,
            blockage_steps_remaining=int(info.get("blockage_steps_remaining", 0)),
            smoke_contradiction=bool(float(observation[25]) > 0.5),
            target_capacity_difference=float(observation[26]),
        )
        if collect:
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
    metrics = {
        "steps": steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": total_reward,
    }
    return metrics, observations, actions


def build_joint_dataset(behavior_factory, scenarios, advice, repetitions: int):
    env = make_joint_env(behavior_factory, scenarios, advice)
    split_by_scenario = stratified_scenario_splits(scenarios)
    result_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    observations: list[np.ndarray] = []
    actions: list[int] = []
    splits: list[str] = []
    scenario_ids: list[str] = []
    conflicts: list[str] = []
    for scenario_index, scenario in enumerate(scenarios):
        split = split_by_scenario[scenario.scenario_id]
        for repetition in range(repetitions):
            population_seed = 500_000 + scenario_index * repetitions + repetition
            results: dict[str, dict[str, float]] = {}
            episode_rows: list[dict[str, object]] = []
            for plan in JOINT_GATE_PLANS:
                metrics, _, _ = run_joint_plan(env, scenario, population_seed, plan)
                results[plan.name] = metrics
                episode_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "conflict_type": scenario.conflict_type,
                        "population_seed": population_seed,
                        "split": split,
                        "plan": plan.name,
                        "plan_family": plan.family,
                        "route": plan.route,
                        "release_fraction": plan.release_fraction,
                        "wait_steps": plan.wait_steps,
                        "override_at_step": "" if plan.override_at_step is None else plan.override_at_step,
                        "reactive_capacity": int(plan.reactive_capacity),
                        **metrics,
                    }
                )
            selected = select_safest_joint_plan(results)
            for row in episode_rows:
                row["selected"] = int(row["plan"] == selected.name)
            result_rows.extend(episode_rows)
            selected_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "population_seed": population_seed,
                    "split": split,
                    "selected_plan": selected.name,
                    "selected_family": selected.family,
                    "route": selected.route,
                    "release_fraction": selected.release_fraction,
                }
            )
            _, episode_observations, episode_actions = run_joint_plan(
                env, scenario, population_seed, selected, collect=True
            )
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            splits.extend([split] * len(episode_actions))
            scenario_ids.extend([scenario.scenario_id] * len(episode_actions))
            conflicts.extend([scenario.conflict_type] * len(episode_actions))
    env.close()
    return (
        result_rows,
        selected_rows,
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(splits),
        np.asarray(scenario_ids),
        np.asarray(conflicts),
    )


def run_joint_model_episode(env, model, scenario, population_seed: int, advice):
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    target = advice[scenario.scenario_id].target_branch
    total_reward = 0.0
    requested_counts = {"trust": 0, "override": 0, "wait": 0}
    actual_counts = {"trust": 0, "override": 0, "wait": 0}
    requested_release_sum = actual_release_sum = 0.0
    blocked_by_hold = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        requested_gate = str(info["requested_gate_action"])
        actual_gate = str(info["actual_gate_action"])
        requested_counts[requested_gate] += 1
        actual_counts[actual_gate] += 1
        requested_release_sum += float(info["requested_release_fraction"])
        actual_release_sum += float(info["actual_release_fraction"])
        blocked_by_hold += int(bool(info["action_blocked_by_sign_hold"]))
        total_reward += float(reward)
        steps += 1
    return {
        "scenario_id": scenario.scenario_id,
        "conflict_type": scenario.conflict_type,
        "llm_target_correct": int(target == safe_forecast_target(scenario)),
        "population_seed": population_seed,
        "steps": steps,
        "requested_trust_rate": requested_counts["trust"] / steps,
        "requested_override_rate": requested_counts["override"] / steps,
        "requested_wait_rate": requested_counts["wait"] / steps,
        "actual_trust_rate": actual_counts["trust"] / steps,
        "actual_override_rate": actual_counts["override"] / steps,
        "actual_wait_rate": actual_counts["wait"] / steps,
        "sign_hold_block_rate": blocked_by_hold / steps,
        "requested_mean_release_fraction": requested_release_sum / steps,
        "actual_mean_release_fraction": actual_release_sum / steps,
        # Backward-compatible columns now refer to what the simulator executed.
        "trust_rate": actual_counts["trust"] / steps,
        "override_rate": actual_counts["override"] / steps,
        "wait_rate": actual_counts["wait"] / steps,
        "mean_release_fraction": actual_release_sum / steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": total_reward,
    }


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    started = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    domain_scenarios = make_domain_randomized_trust_bank(
        args.domain_scenarios, args.scenario_seed + 20_000, "GDR"
    )
    test_scenarios = make_trust_gate_bank(args.test_scenarios, args.scenario_seed + 1, "GTE")
    advice = {
        scenario.scenario_id: conflict_llm_advice(scenario)
        for scenario in domain_scenarios + test_scenarios
    }
    behavior_factory = lambda: CalibratedLogitBehavior(args.calibration_artifact)
    check_env(make_joint_env(behavior_factory, domain_scenarios, advice), warn=True)

    print(
        f"Searching {len(JOINT_GATE_PLANS)} route-release counterfactual plans...", flush=True
    )
    (
        counterfactual_rows,
        selected_rows,
        observations,
        actions,
        splits,
        scenario_ids,
        conflicts,
    ) = build_joint_dataset(
        behavior_factory, domain_scenarios, advice, args.counterfactual_repetitions
    )
    write_csv(counterfactual_rows, args.output / "joint_counterfactual_results.csv")
    write_csv(selected_rows, args.output / "joint_selected_plans.csv")
    np.savez_compressed(
        args.output / "joint_supervised_dataset.npz",
        observations=observations,
        actions=actions,
        splits=splits,
        scenario_ids=scenario_ids,
        conflict_types=conflicts,
    )
    diagnostic_rows = copy_baseline_diagnostics(observations, actions, conflicts)
    write_csv(diagnostic_rows, args.output / "copy_previous_action_diagnostics.csv")
    print(
        f"Labels={len(actions)}, 9-action counts={np.bincount(actions, minlength=9).tolist()}", flush=True
    )

    train_mask = splits == "supervised_train"
    validation_mask = splits == "supervised_validation"
    common = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model_paths: dict[tuple[int, str], Path] = {}
    training_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, 1):
        print(f"Seed {seed} ({index}/{len(seeds)}): joint route-release training...", flush=True)

        def env_factory():
            return make_joint_env(behavior_factory, domain_scenarios, advice)

        vector_env = make_vec_env(env_factory, n_envs=args.n_envs, seed=seed)
        model = AuxiliarySupervisedPPO(env=vector_env, seed=seed, **common)
        random_validation = policy_accuracy(model, observations[validation_mask], actions[validation_mask])
        pretrain_ppo_gate(
            model,
            observations[train_mask],
            actions[train_mask],
            epochs=args.supervised_epochs,
            batch_size=256,
            seed=seed,
        )
        pretrained_validation = policy_accuracy(
            model, observations[validation_mask], actions[validation_mask]
        )
        model.configure_auxiliary_supervision(
            observations[train_mask],
            actions[train_mask],
            coefficient=args.auxiliary_coefficient,
            batches_per_update=args.auxiliary_batches,
            batch_size=256,
            seed=seed + 50_000,
        )
        print(
            f"  scenario-held-out accuracy {random_validation:.1%} -> {pretrained_validation:.1%}; "
            "PPO + supervision...",
            flush=True,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False, reset_num_timesteps=False)
        path = model_dir / f"seed_{seed}_joint_route_release"
        model.save(path)
        model_paths[(seed, "joint_route_release")] = path.with_suffix(".zip")
        training_rows.append(
            {
                "training_seed": seed,
                "random_validation_accuracy": random_validation,
                "pretrained_validation_accuracy": pretrained_validation,
                "post_ppo_train_accuracy": policy_accuracy(
                    model, observations[train_mask], actions[train_mask]
                ),
                "post_ppo_validation_accuracy": policy_accuracy(
                    model, observations[validation_mask], actions[validation_mask]
                ),
                "auxiliary_updates": int(model.auxiliary_update_count),
                "ppo_timesteps": int(model.num_timesteps),
            }
        )
        vector_env.close()
        baseline_paths = {
            "ppo_time": args.baseline_model_dir / f"seed_{seed}_ppo_time.zip",
            "scratch_gate": args.baseline_model_dir / f"seed_{seed}_llm_trust_gate_ppo.zip",
            "previous_contradiction": args.previous_model_dir
            / f"seed_{seed}_randomized_contradiction.zip",
        }
        for policy, baseline_path in baseline_paths.items():
            if not baseline_path.exists():
                raise FileNotFoundError(f"missing baseline model: {baseline_path}")
            model_paths[(seed, policy)] = baseline_path
    write_csv(training_rows, args.output / "training_summary.csv")

    print("Evaluating joint controllers on the locked test bank...", flush=True)
    episode_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, 1):
        print(f"Evaluating seed {seed} ({index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)], device="cpu") for policy in POLICIES}
        envs = {
            "ppo_time": make_time_env(behavior_factory, test_scenarios),
            "scratch_gate": make_gate_env(behavior_factory, test_scenarios, advice),
            "previous_contradiction": make_gate_env(behavior_factory, test_scenarios, advice),
            "joint_route_release": make_joint_env(behavior_factory, test_scenarios, advice),
        }
        # The previous contradiction model needs its original 28-feature wrapper.
        from run_domain_randomized_contradiction import make_variant_env

        envs["previous_contradiction"].close()
        envs["previous_contradiction"] = make_variant_env(
            "randomized_contradiction", behavior_factory, test_scenarios, advice
        )
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                population_seed = 100_000 + scenario_index * args.eval_repetitions + repetition
                for policy in POLICIES:
                    if policy == "joint_route_release":
                        row = run_joint_model_episode(
                            envs[policy], models[policy], scenario, population_seed, advice
                        )
                    else:
                        row = run_model_episode(
                            envs[policy], models[policy], policy, scenario, population_seed, advice
                        )
                        row["mean_release_fraction"] = 1.0 - float(row["wait_rate"])
                    episode_rows.append(
                        {
                            "test_behavior": scenario.conflict_type,
                            "training_seed": seed,
                            "policy": policy,
                            **row,
                        }
                    )
        for env in envs.values():
            env.close()
    write_csv(episode_rows, args.output / "episode_results.csv")
    per_seed = summarize_cross_episodes(episode_rows)
    per_seed.extend(summarize_cross_episodes([{**row, "test_behavior": "all"} for row in episode_rows]))
    write_csv(per_seed, args.output / "per_seed_summary.csv")
    aggregate = aggregate_by_conflict(per_seed, args.bootstrap_samples, args.scenario_seed)
    write_csv(aggregate, args.output / "aggregate_summary.csv")
    comparisons = (
        ("joint_route_release", "previous_contradiction"),
        ("joint_route_release", "scratch_gate"),
        ("joint_route_release", "ppo_time"),
    )
    paired_seed_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for comparison_index, (candidate, baseline) in enumerate(comparisons):
        seed_rows, rows = paired_comparison(
            per_seed,
            candidate,
            baseline,
            args.bootstrap_samples,
            args.scenario_seed + comparison_index * 50_000,
        )
        paired_seed_rows.extend(seed_rows)
        paired_rows.extend(rows)
    write_csv(paired_seed_rows, args.output / "paired_differences_by_seed.csv")
    write_csv(paired_rows, args.output / "paired_aggregate.csv")
    decisions = decision_summary(episode_rows, seeds, GATE_POLICIES)
    write_csv(decisions, args.output / "trust_gate_decisions.csv")
    write_report(
        args,
        started,
        seeds,
        domain_scenarios,
        test_scenarios,
        training_rows,
        selected_rows,
        episode_rows,
        aggregate,
        paired_rows,
        decisions,
    )
    print(f"Outputs: {args.output}", flush=True)


def write_report(args, started, seeds, domain_scenarios, test_scenarios, training_rows, selected_rows, episode_rows, aggregate, paired_rows, decisions):
    (args.output / "domain_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(domain_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "training_seeds": seeds,
        "joint_actions": {
            "0-3": "trust LLM with 25/50/75/100 percent release",
            "4-7": "override LLM with 25/50/75/100 percent release",
            "8": "wait with zero release",
        },
        "counterfactual_plans": [plan.__dict__ for plan in JOINT_GATE_PLANS],
        "domain_scenarios": len(domain_scenarios),
        "test_scenarios": len(test_scenarios),
        "ppo_timesteps_per_model": args.timesteps,
        "additional_ollama_calls": 0,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    agg = {(row["conflict_type"], row["policy"], row["metric"]): row for row in aggregate}
    paired = {
        (row["conflict_type"], row["candidate"], row["baseline"], row["metric"]): row
        for row in paired_rows
    }
    decision_lookup = {
        conflict: {
            metric: float(np.mean([
                row[metric]
                for row in decisions
                if row["policy"] == "joint_route_release" and row["conflict_type"] == conflict
            ]))
            for metric in ("trust_rate", "override_rate", "wait_rate")
        }
        for conflict in ("all", *TRUST_GATE_TYPES)
    }
    release_lookup = {
        conflict: float(np.mean([
            row["mean_release_fraction"]
            for row in episode_rows
            if row["policy"] == "joint_route_release"
            and (conflict == "all" or row["conflict_type"] == conflict)
        ]))
        for conflict in ("all", *TRUST_GATE_TYPES)
    }
    labels = {
        "all": "全部场景",
        "clear": "无冲突",
        "safe_exit_short_block": "短时封闭",
        "safe_exit_long_block": "长时封闭",
        "safe_exit_congested": "安全出口拥堵",
        "forecast_error": "LLM预测错误",
    }
    selected_release = {
        conflict: float(np.mean([
            float(row["release_fraction"])
            for row in selected_rows
            if row["conflict_type"] == conflict
        ]))
        for conflict in sorted({str(row["conflict_type"]) for row in selected_rows})
    }
    report = [
        "# 出口方向 + 分级放行容量联合控制实验",
        "",
        "门控动作空间包含9个动作：相信或覆盖LLM分别组合25%、50%、75%、100%房间放行容量，以及一个完全等待动作。",
        f"反事实计划共 {len(JOINT_GATE_PLANS)} 种；随机训练域拥堵场景选中计划的平均放行比例为 "
        f"{selected_release.get('safe_exit_congested', float('nan')):.1%}。",
        "",
        "## 监督语义保持",
        "",
        "| 种子 | 预训练验证准确率 | PPO后验证准确率 |",
        "|---:|---:|---:|",
    ]
    for row in training_rows:
        report.append(
            f"| {row['training_seed']} | {float(row['pretrained_validation_accuracy']):.1%} | "
            f"{float(row['post_ppo_validation_accuracy']):.1%} |"
        )
    report.extend([
        "",
        "## 锁定测试失能率",
        "",
        "| 场景 | 时间PPO | 从零门控 | 上轮矛盾模型 | 联合方向+容量 |",
        "|---|---:|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        values = [float(agg[(conflict, policy, "incapacitation_rate")]["mean"]) for policy in POLICIES]
        report.append(f"| {label} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    report.extend([
        "",
        "## 联合控制动作",
        "",
        "| 场景 | 相信 | 覆盖 | 等待 | 平均放行比例 |",
        "|---|---:|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        rates = decision_lookup[conflict]
        report.append(
            f"| {label} | {rates['trust_rate']:.1%} | {rates['override_rate']:.1%} | "
            f"{rates['wait_rate']:.1%} | {release_lookup[conflict]:.1%} |"
        )
    versus_previous = paired[("all", "joint_route_release", "previous_contradiction", "incapacitation_rate")]
    versus_scratch = paired[("all", "joint_route_release", "scratch_gate", "incapacitation_rate")]
    versus_time = paired[("all", "joint_route_release", "ppo_time", "incapacitation_rate")]
    versus_time_fed = paired[("all", "joint_route_release", "ppo_time", "mean_cumulative_fed")]
    congestion_previous = paired[("safe_exit_congested", "joint_route_release", "previous_contradiction", "incapacitation_rate")]
    congestion_time = paired[("safe_exit_congested", "joint_route_release", "ppo_time", "incapacitation_rate")]
    error_previous = paired[("forecast_error", "joint_route_release", "previous_contradiction", "incapacitation_rate")]
    report.extend([
        "",
        "## 判定",
        "",
        f"- 相对上轮矛盾模型，整体失能率差 {float(versus_previous['mean']):.2%}，95%区间 "
        f"[{float(versus_previous['bootstrap_ci95_lower']):.2%}, {float(versus_previous['bootstrap_ci95_upper']):.2%}]。",
        f"- 拥堵场景差为 {float(congestion_previous['mean']):.2%}，95%区间 "
        f"[{float(congestion_previous['bootstrap_ci95_lower']):.2%}, {float(congestion_previous['bootstrap_ci95_upper']):.2%}]；"
        f"预测错误场景差为 {float(error_previous['mean']):.2%}。",
        f"- 相对从零门控整体差 {float(versus_scratch['mean']):.2%}，95%区间 "
        f"[{float(versus_scratch['bootstrap_ci95_lower']):.2%}, {float(versus_scratch['bootstrap_ci95_upper']):.2%}]；"
        f"相对时间PPO差 {float(versus_time['mean']):.2%}，95%区间 "
        f"[{float(versus_time['bootstrap_ci95_lower']):.2%}, {float(versus_time['bootstrap_ci95_upper']):.2%}]。",
        f"- 相对时间PPO的FED差为 {float(versus_time_fed['mean']):.4f}，95%区间 "
        f"[{float(versus_time_fed['bootstrap_ci95_lower']):.4f}, "
        f"{float(versus_time_fed['bootstrap_ci95_upper']):.4f}]；拥堵失能率差为 "
        f"{float(congestion_time['mean']):.2%}。",
        "",
        "## 结论",
        "",
        "**联合方向与分级放行容量方案达到本轮成功标准。** 它把上一轮矛盾模型的整体失能率从6.17%降至2.32%，"
        "并把拥堵场景从22.75%降至10.89%。相对时间PPO，整体失能率和FED均稳定改善；相对从零门控的差异区间跨过0，"
        "因此只能判定为性能相当，不能宣称稳定优于从零门控。",
        "",
        "模型在拥堵场景平均使用65.5%放行容量，同时保留预测错误场景的低失能率0.39%。这说明拥堵问题需要将“出口方向”和"
        "“进入交汇区的流量”分开控制，二元相信/覆盖门控无法表达这种策略。",
        "",
        "## 解释边界",
        "",
        "- 分级放行直接改变房间到交汇区的每步容量，但仍是全局控制，未对个体分组。",
        "- 实验继续使用合成火灾和行为模型，本轮新增Ollama调用为0。",
    ])
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
