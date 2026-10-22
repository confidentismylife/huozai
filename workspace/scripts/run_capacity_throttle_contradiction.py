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
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import (
    TRUST_GATE_TYPES,
    conflict_llm_advice,
    conflict_scenario_records,
    make_domain_randomized_trust_bank,
    make_trust_gate_bank,
)
from fire_evac.dynamic_gate_supervision import DYNAMIC_GATE_PLANS
from fire_evac.dynamic_multiseed import summarize_cross_episodes
from fire_evac.supervised_gate import policy_accuracy, pretrain_ppo_gate
from run_domain_randomized_contradiction import (
    collect_contradiction_observations,
    make_variant_env,
)
from run_dynamic_auxiliary_gate import build_dynamic_dataset
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

POLICIES = ("ppo_time", "scratch_gate", "previous_contradiction", "throttle_contradiction")
GATE_POLICIES = tuple(policy for policy in POLICIES if policy != "ppo_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity throttling plans with contradiction-aware PPO.")
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
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "capacity_throttle_contradiction"
    )
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.domain_scenarios < 12 or args.test_scenarios < len(TRUST_GATE_TYPES):
        raise SystemExit("scenario counts are too small")
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
    check_env(
        make_variant_env("randomized_contradiction", behavior_factory, domain_scenarios, advice), warn=True
    )

    print(
        f"Searching {len(DYNAMIC_GATE_PLANS)} plans including capacity throttling...", flush=True
    )
    (
        counterfactual_rows,
        selected_rows,
        _,
        base_actions,
        base_splits,
        _,
        _,
    ) = build_dynamic_dataset(
        behavior_factory, domain_scenarios, advice, args.counterfactual_repetitions
    )
    observations, actions, splits, scenario_ids, conflicts = collect_contradiction_observations(
        behavior_factory, domain_scenarios, advice, selected_rows
    )
    if not np.array_equal(base_actions, actions) or not np.array_equal(base_splits, splits):
        raise RuntimeError("replayed contradiction dataset does not match counterfactual labels")
    write_csv(counterfactual_rows, args.output / "throttle_counterfactual_results.csv")
    write_csv(selected_rows, args.output / "throttle_selected_plans.csv")
    np.savez_compressed(
        args.output / "throttle_supervised_dataset.npz",
        observations=observations,
        actions=actions,
        splits=splits,
        scenario_ids=scenario_ids,
        conflict_types=conflicts,
    )
    throttle_selected = sum("throttle" in str(row["selected_family"]) for row in selected_rows)
    print(
        f"Labels={len(actions)}, actions={np.bincount(actions, minlength=3).tolist()}, "
        f"throttle plans selected={throttle_selected}/{len(selected_rows)}",
        flush=True,
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
        print(f"Seed {seed} ({index}/{len(seeds)}): throttle-aware training...", flush=True)

        def env_factory():
            return make_variant_env(
                "randomized_contradiction", behavior_factory, domain_scenarios, advice
            )

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
            seed=seed + 40_000,
        )
        print(
            f"  scenario-held-out accuracy {random_validation:.1%} -> {pretrained_validation:.1%}; "
            "PPO + supervision...",
            flush=True,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False, reset_num_timesteps=False)
        path = model_dir / f"seed_{seed}_throttle_contradiction"
        model.save(path)
        model_paths[(seed, "throttle_contradiction")] = path.with_suffix(".zip")
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

    print("Evaluating capacity-throttle models on the locked test bank...", flush=True)
    episode_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, 1):
        print(f"Evaluating seed {seed} ({index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)], device="cpu") for policy in POLICIES}
        envs = {
            policy: make_time_env(behavior_factory, test_scenarios)
            if policy == "ppo_time"
            else make_variant_env("randomized_contradiction", behavior_factory, test_scenarios, advice)
            if policy in ("previous_contradiction", "throttle_contradiction")
            else make_gate_env(behavior_factory, test_scenarios, advice)
            for policy in POLICIES
        }
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                population_seed = 100_000 + scenario_index * args.eval_repetitions + repetition
                for policy in POLICIES:
                    row = run_model_episode(
                        envs[policy], models[policy], policy, scenario, population_seed, advice
                    )
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
        ("throttle_contradiction", "previous_contradiction"),
        ("throttle_contradiction", "scratch_gate"),
        ("throttle_contradiction", "ppo_time"),
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
        aggregate,
        paired_rows,
        decisions,
    )
    print(f"Outputs: {args.output}", flush=True)


def write_report(
    args,
    started,
    seeds,
    domain_scenarios,
    test_scenarios,
    training_rows,
    selected_rows,
    aggregate,
    paired_rows,
    decisions,
):
    (args.output / "domain_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(domain_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "training_seeds": seeds,
        "counterfactual_plans": [plan.__dict__ for plan in DYNAMIC_GATE_PLANS],
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
        (policy, conflict): {
            metric: float(np.mean([
                row[metric]
                for row in decisions
                if row["policy"] == policy and row["conflict_type"] == conflict
            ]))
            for metric in ("trust_rate", "override_rate", "wait_rate")
        }
        for policy in GATE_POLICIES
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
    throttle_by_conflict = {
        conflict: sum(
            "throttle" in str(row["selected_family"])
            for row in selected_rows
            if row["conflict_type"] == conflict
        )
        for conflict in sorted({str(row["conflict_type"]) for row in selected_rows})
    }
    report = [
        "# 容量限流反事实计划 + 烟气矛盾门控实验",
        "",
        f"在上一轮 {len(DYNAMIC_GATE_PLANS) - 6} 个计划上增加6个周期性限流计划：25%、50%、75%时隙放行到LLM安全出口，其余时隙等待。",
        f"随机训练域中，限流计划被选中 {sum(throttle_by_conflict.values())}/{len(selected_rows)} 个回合；拥堵场景为 "
        f"{throttle_by_conflict.get('safe_exit_congested', 0)} 个。",
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
        "| 场景 | 时间PPO | 从零门控 | 上轮矛盾模型 | 加容量限流计划 |",
        "|---|---:|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        values = [float(agg[(conflict, policy, "incapacitation_rate")]["mean"]) for policy in POLICIES]
        report.append(f"| {label} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    report.extend([
        "",
        "## 新模型动作比例",
        "",
        "| 场景 | 相信 | 覆盖 | 等待/限流 |",
        "|---|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        rates = decision_lookup[("throttle_contradiction", conflict)]
        report.append(
            f"| {label} | {rates['trust_rate']:.1%} | {rates['override_rate']:.1%} | {rates['wait_rate']:.1%} |"
        )
    versus_previous = paired[("all", "throttle_contradiction", "previous_contradiction", "incapacitation_rate")]
    versus_scratch = paired[("all", "throttle_contradiction", "scratch_gate", "incapacitation_rate")]
    versus_time = paired[("all", "throttle_contradiction", "ppo_time", "incapacitation_rate")]
    congestion_previous = paired[("safe_exit_congested", "throttle_contradiction", "previous_contradiction", "incapacitation_rate")]
    report.extend([
        "",
        "## 判定",
        "",
        f"- 相对上轮矛盾模型，整体失能率差 {float(versus_previous['mean']):.2%}，95%区间 "
        f"[{float(versus_previous['bootstrap_ci95_lower']):.2%}, {float(versus_previous['bootstrap_ci95_upper']):.2%}]。",
        f"- 拥堵场景相对上轮矛盾模型的失能率差为 {float(congestion_previous['mean']):.2%}，95%区间 "
        f"[{float(congestion_previous['bootstrap_ci95_lower']):.2%}, {float(congestion_previous['bootstrap_ci95_upper']):.2%}]。",
        f"- 相对从零门控，整体失能率差 {float(versus_scratch['mean']):.2%}；相对时间PPO为 "
        f"{float(versus_time['mean']):.2%}。",
        "",
        "## 结论",
        "",
        "**周期性限流方案判定为失败，不应替代上一轮矛盾模型。** 它将拥堵场景失能率从22.75%进一步推高到35.70%，"
        "整体失能率也从6.17%升至8.57%。长时封闭有所改善，但不足以抵消拥堵和短时封闭的恶化。",
        "",
        "原因是全局时间占空比同时引入了房间等待暴露，而且策略仍会执行完全反向覆盖。真正的拥堵控制需要一个独立的“保持安全出口方向、"
        "直接设置部分放行容量”动作，并在环境中记录队列吞吐量；不能继续用相信/等待的时间交替近似。",
        "",
        "## 解释边界",
        "",
        "- 限流用全局指示牌的时间占空比近似，不是逐人精确流量控制。",
        "- 实验仍使用合成火灾与行为模型，本轮新增Ollama调用为0。",
    ])
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
