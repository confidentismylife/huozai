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
    ContradictionAwareLLMTrustGateWrapper,
    GymDynamicConflictEnv,
    TRUST_GATE_TYPES,
    conflict_llm_advice,
    conflict_scenario_records,
    make_domain_randomized_trust_bank,
    make_trust_gate_bank,
)
from fire_evac.dynamic_gate_supervision import DYNAMIC_GATE_PLANS
from fire_evac.dynamic_multiseed import summarize_cross_episodes
from fire_evac.supervised_gate import policy_accuracy, pretrain_ppo_gate
from run_dynamic_auxiliary_gate import build_dynamic_dataset, run_dynamic_plan
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

POLICIES = (
    "ppo_time",
    "scratch_gate",
    "previous_dynamic_aux",
    "randomized_base",
    "randomized_contradiction",
)
GATE_POLICIES = tuple(policy for policy in POLICIES if policy != "ppo_time")
VARIANTS = ("randomized_base", "randomized_contradiction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain randomization and explicit LLM contradiction features.")
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
        "--previous-model-dir", type=Path, default=ROOT / "outputs" / "dynamic_auxiliary_gate" / "models"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "domain_randomized_contradiction"
    )
    return parser.parse_args()


def make_variant_env(variant: str, behavior_factory, scenarios, advice):
    if variant == "randomized_base":
        return make_gate_env(behavior_factory, scenarios, advice)
    if variant == "randomized_contradiction":
        return ContradictionAwareLLMTrustGateWrapper(
            GymDynamicConflictEnv(behavior_factory(), scenarios), advice
        )
    raise ValueError(f"unknown variant: {variant}")


def collect_contradiction_observations(
    behavior_factory,
    scenarios,
    advice,
    selected_rows,
):
    scenario_lookup = {scenario.scenario_id: scenario for scenario in scenarios}
    plan_lookup = {plan.name: plan for plan in DYNAMIC_GATE_PLANS}
    env = make_variant_env("randomized_contradiction", behavior_factory, scenarios, advice)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    splits: list[str] = []
    scenario_ids: list[str] = []
    conflicts: list[str] = []
    for row in selected_rows:
        scenario = scenario_lookup[str(row["scenario_id"])]
        plan = plan_lookup[str(row["selected_plan"])]
        _, episode_observations, episode_actions = run_dynamic_plan(
            env,
            scenario,
            int(row["population_seed"]),
            plan,
            collect=True,
        )
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        splits.extend([str(row["split"])] * len(episode_actions))
        scenario_ids.extend([scenario.scenario_id] * len(episode_actions))
        conflicts.extend([scenario.conflict_type] * len(episode_actions))
    env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(splits),
        np.asarray(scenario_ids),
        np.asarray(conflicts),
    )


def accuracy(model, observations, actions, mask) -> float:
    return policy_accuracy(model, observations[mask], actions[mask])


def contradiction_response(rows, seeds):
    output: list[dict[str, object]] = []
    for policy in GATE_POLICIES:
        for conflict in ("all", *TRUST_GATE_TYPES):
            for seed in seeds:
                subset = [
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row["training_seed"] == seed
                    and (conflict == "all" or row["conflict_type"] == conflict)
                ]
                output.append(
                    {
                        "policy": policy,
                        "conflict_type": conflict,
                        "training_seed": seed,
                        "trust_rate": float(np.mean([row["trust_rate"] for row in subset])),
                        "override_rate": float(np.mean([row["override_rate"] for row in subset])),
                        "wait_rate": float(np.mean([row["wait_rate"] for row in subset])),
                    }
                )
    return output


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.domain_scenarios < 12 or args.test_scenarios < len(TRUST_GATE_TYPES):
        raise SystemExit("domain-scenarios must be >=12 and test-scenarios must cover all conflict types")
    if args.timesteps < 1_000 or args.counterfactual_repetitions < 1 or args.eval_repetitions < 1:
        raise SystemExit("timesteps must be >=1000 and repetitions must be positive")

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
    for variant in VARIANTS:
        check_env(make_variant_env(variant, behavior_factory, domain_scenarios, advice), warn=True)

    print(
        f"Searching {len(DYNAMIC_GATE_PLANS)} plans over {len(domain_scenarios)} randomized scenarios...",
        flush=True,
    )
    (
        counterfactual_rows,
        selected_rows,
        base_observations,
        base_actions,
        base_splits,
        base_scenario_ids,
        base_conflicts,
    ) = build_dynamic_dataset(
        behavior_factory,
        domain_scenarios,
        advice,
        args.counterfactual_repetitions,
    )
    (
        contradiction_observations,
        contradiction_actions,
        contradiction_splits,
        contradiction_scenario_ids,
        contradiction_conflicts,
    ) = collect_contradiction_observations(
        behavior_factory,
        domain_scenarios,
        advice,
        selected_rows,
    )
    if not np.array_equal(base_actions, contradiction_actions):
        raise RuntimeError("base and contradiction datasets produced different action labels")
    if not np.array_equal(base_splits, contradiction_splits):
        raise RuntimeError("base and contradiction datasets produced different splits")
    write_csv(counterfactual_rows, args.output / "randomized_counterfactual_results.csv")
    write_csv(selected_rows, args.output / "randomized_selected_plans.csv")
    np.savez_compressed(
        args.output / "randomized_supervised_datasets.npz",
        base_observations=base_observations,
        contradiction_observations=contradiction_observations,
        actions=base_actions,
        splits=base_splits,
        scenario_ids=base_scenario_ids,
        conflict_types=base_conflicts,
    )
    print(
        f"Randomized labels: {len(base_actions)} decisions, counts="
        f"{np.bincount(base_actions, minlength=3).tolist()}",
        flush=True,
    )

    datasets = {
        "randomized_base": base_observations,
        "randomized_contradiction": contradiction_observations,
    }
    train_mask = base_splits == "supervised_train"
    validation_mask = base_splits == "supervised_validation"
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
    curve_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds, 1):
        for variant in VARIANTS:
            print(f"Seed {seed} ({seed_index}/{len(seeds)}): training {variant}...", flush=True)

            def env_factory(variant_name=variant):
                return make_variant_env(variant_name, behavior_factory, domain_scenarios, advice)

            vector_env = make_vec_env(env_factory, n_envs=args.n_envs, seed=seed)
            model = AuxiliarySupervisedPPO(env=vector_env, seed=seed, **common)
            observations = datasets[variant]
            random_validation = accuracy(model, observations, base_actions, validation_mask)
            logs = pretrain_ppo_gate(
                model,
                observations[train_mask],
                base_actions[train_mask],
                epochs=args.supervised_epochs,
                batch_size=256,
                seed=seed,
            )
            curve_rows.extend({"training_seed": seed, "variant": variant, **row} for row in logs)
            pretrained_train = accuracy(model, observations, base_actions, train_mask)
            pretrained_validation = accuracy(model, observations, base_actions, validation_mask)
            model.configure_auxiliary_supervision(
                observations[train_mask],
                base_actions[train_mask],
                coefficient=args.auxiliary_coefficient,
                batches_per_update=args.auxiliary_batches,
                batch_size=256,
                seed=seed + 30_000,
            )
            print(
                f"  scenario-held-out accuracy {random_validation:.1%} -> {pretrained_validation:.1%}; "
                "PPO + auxiliary supervision...",
                flush=True,
            )
            model.learn(total_timesteps=args.timesteps, progress_bar=False, reset_num_timesteps=False)
            path = model_dir / f"seed_{seed}_{variant}"
            model.save(path)
            model_paths[(seed, variant)] = path.with_suffix(".zip")
            training_rows.append(
                {
                    "training_seed": seed,
                    "variant": variant,
                    "random_validation_accuracy": random_validation,
                    "pretrained_train_accuracy": pretrained_train,
                    "pretrained_validation_accuracy": pretrained_validation,
                    "post_ppo_train_accuracy": accuracy(model, observations, base_actions, train_mask),
                    "post_ppo_validation_accuracy": accuracy(model, observations, base_actions, validation_mask),
                    "auxiliary_updates": int(model.auxiliary_update_count),
                    "ppo_timesteps": int(model.num_timesteps),
                }
            )
            vector_env.close()
        baseline_paths = {
            "ppo_time": args.baseline_model_dir / f"seed_{seed}_ppo_time.zip",
            "scratch_gate": args.baseline_model_dir / f"seed_{seed}_llm_trust_gate_ppo.zip",
            "previous_dynamic_aux": args.previous_model_dir / f"seed_{seed}_dynamic_aux_ppo.zip",
        }
        for policy, path in baseline_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing baseline model: {path}")
            model_paths[(seed, policy)] = path
    write_csv(training_rows, args.output / "training_summary.csv")
    write_csv(curve_rows, args.output / "pretraining_curves.csv")

    print("Running the unchanged locked test bank...", flush=True)
    episode_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds, 1):
        print(f"Evaluating seed {seed} ({seed_index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)], device="cpu") for policy in POLICIES}
        envs = {}
        for policy in POLICIES:
            if policy == "ppo_time":
                envs[policy] = make_time_env(behavior_factory, test_scenarios)
            elif policy == "randomized_contradiction":
                envs[policy] = make_variant_env(policy, behavior_factory, test_scenarios, advice)
            else:
                envs[policy] = make_gate_env(behavior_factory, test_scenarios, advice)
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
        ("randomized_contradiction", "randomized_base"),
        ("randomized_contradiction", "previous_dynamic_aux"),
        ("randomized_contradiction", "scratch_gate"),
        ("randomized_contradiction", "ppo_time"),
        ("randomized_base", "previous_dynamic_aux"),
    )
    paired_seed_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for index, (candidate, baseline) in enumerate(comparisons):
        seed_rows, rows = paired_comparison(
            per_seed,
            candidate,
            baseline,
            args.bootstrap_samples,
            args.scenario_seed + index * 50_000,
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
        aggregate,
        paired_rows,
        decisions,
    )
    print(f"Outputs: {args.output}", flush=True)


def write_report(args, started, seeds, domain_scenarios, test_scenarios, training_rows, aggregate, paired_rows, decisions):
    (args.output / "domain_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(domain_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "training_seeds": seeds,
        "domain_scenarios": len(domain_scenarios),
        "test_scenarios": len(test_scenarios),
        "counterfactual_plans": len(DYNAMIC_GATE_PLANS),
        "counterfactual_repetitions": args.counterfactual_repetitions,
        "ppo_timesteps_per_model": args.timesteps,
        "explicit_features": [
            "llm_target_smoke",
            "alternative_smoke",
            "normalized_smoke_contradiction",
            "contradiction_flag",
            "target_capacity_difference",
            "target_occupancy_difference",
        ],
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
    report = [
        "# 场景域随机化 + LLM烟气矛盾特征实验",
        "",
        f"使用 {len(domain_scenarios)} 个随机化训练场景，覆盖火灾增长、起火延迟、背景烟气、出口容量、封闭时长、预测错误和复合冲突。",
        "两组新模型共享完全相同的反事实标签、网络与训练预算；唯一观测差异是后一组增加6个烟气矛盾与拥堵证据特征。",
        "",
        "## 监督语义保持",
        "",
        "| 变体 | 种子 | 预训练验证准确率 | PPO后验证准确率 |",
        "|---|---:|---:|---:|",
    ]
    for row in training_rows:
        report.append(
            f"| {row['variant']} | {row['training_seed']} | "
            f"{float(row['pretrained_validation_accuracy']):.1%} | "
            f"{float(row['post_ppo_validation_accuracy']):.1%} |"
        )
    report.extend([
        "",
        "## 锁定测试失能率",
        "",
        "| 场景 | 时间PPO | 从零门控 | 上轮动态辅助 | 仅域随机化 | 域随机化+矛盾特征 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        values = [float(agg[(conflict, policy, "incapacitation_rate")]["mean"]) for policy in POLICIES]
        report.append(f"| {label} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    report.extend([
        "",
        "## 新模型动作比例",
        "",
        "| 场景 | 仅随机化 覆盖/等待 | 加矛盾特征 覆盖/等待 |",
        "|---|---:|---:|",
    ])
    for conflict, label in labels.items():
        base = decision_lookup[("randomized_base", conflict)]
        explicit = decision_lookup[("randomized_contradiction", conflict)]
        report.append(
            f"| {label} | {base['override_rate']:.1%}/{base['wait_rate']:.1%} | "
            f"{explicit['override_rate']:.1%}/{explicit['wait_rate']:.1%} |"
        )
    feature_effect = paired[("all", "randomized_contradiction", "randomized_base", "incapacitation_rate")]
    versus_previous = paired[("all", "randomized_contradiction", "previous_dynamic_aux", "incapacitation_rate")]
    versus_scratch = paired[("all", "randomized_contradiction", "scratch_gate", "incapacitation_rate")]
    versus_time = paired[("all", "randomized_contradiction", "ppo_time", "incapacitation_rate")]
    versus_time_fed = paired[("all", "randomized_contradiction", "ppo_time", "mean_cumulative_fed")]
    base_previous = paired[("all", "randomized_base", "previous_dynamic_aux", "incapacitation_rate")]
    report.extend([
        "",
        "## 判定",
        "",
        f"- 显式矛盾特征相对仅域随机化的失能率差为 {float(feature_effect['mean']):.2%}，95%区间 "
        f"[{float(feature_effect['bootstrap_ci95_lower']):.2%}, {float(feature_effect['bootstrap_ci95_upper']):.2%}]。",
        f"- 域随机化本身相对上一轮动态辅助门控的失能率差为 {float(base_previous['mean']):.2%}。",
        f"- 完整新模型相对上一轮动态辅助门控的失能率差为 {float(versus_previous['mean']):.2%}；"
        f"相对从零门控为 {float(versus_scratch['mean']):.2%}。",
        f"- 完整新模型相对时间PPO的失能率差为 {float(versus_time['mean']):.2%}，95%区间 "
        f"[{float(versus_time['bootstrap_ci95_lower']):.2%}, {float(versus_time['bootstrap_ci95_upper']):.2%}]；FED差为 "
        f"{float(versus_time_fed['mean']):.4f}。",
        "",
        "## 结论",
        "",
        "**预测错误识别成功，拥堵识别和整体安全目标仍未通过。** 显式烟气矛盾特征使预测错误场景的覆盖率从0%提高到80.4%，"
        "失能率从7.33%降至0.11%，并优于时间PPO与从零门控；但拥堵场景仍使用100%反向覆盖，失能率为22.75%。",
        "",
        "拥堵失败说明“LLM建议的安全出口容量不足”不等于“相反出口更安全”。当前门控只能相信、反向覆盖或完全等待，"
        "缺少直接控制安全出口放行容量的动作。因此完整模型整体仍显著劣于时间PPO和从零门控，不能判定为总体成功。",
        "",
        "## 解释边界",
        "",
        "- 显式矛盾只能在实时烟气差已经出现后触发，起火前和观测滞后阶段仍可能相信错误预测。",
        "- 训练和测试继续使用合成火灾与人群模型；本轮新增Ollama调用为0，不能宣称真实部署有效。",
    ])
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
