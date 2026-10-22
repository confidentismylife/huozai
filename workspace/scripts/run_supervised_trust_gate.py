from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import BehaviorModel, CalibratedLogitBehavior
from fire_evac.conflict_multiseed import (
    aggregate_paired_differences,
    aggregate_seed_summaries,
    paired_seed_differences,
)
from fire_evac.dynamic_conflicts import (
    TRUST_GATE_TYPES,
    GymDynamicConflictEnv,
    HoldEnabledActionWrapper,
    LLMTrustGateWrapper,
    ReopenForecastObservationWrapper,
    conflict_llm_advice,
    conflict_scenario_records,
    make_trust_gate_bank,
)
from fire_evac.dynamic_llm_rl import NORTH, safe_forecast_target
from fire_evac.dynamic_multiseed import DYNAMIC_METRICS, summarize_cross_episodes
from fire_evac.supervised_gate import PLANS, plan_action, policy_accuracy, pretrain_ppo_gate, select_safest_plan


POLICIES = ("ppo_time", "scratch_gate", "supervised_only", "supervised_finetuned")
GATE_POLICIES = ("scratch_gate", "supervised_only", "supervised_finetuned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual supervision followed by PPO trust-gate fine-tuning.")
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--train-scenarios", type=int, default=40)
    parser.add_argument("--test-scenarios", type=int, default=20)
    parser.add_argument("--counterfactual-repetitions", type=int, default=10)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "supervised_trust_gate")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    return seeds


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_gate_env(factory: Callable[[], BehaviorModel], scenarios, advice):
    return LLMTrustGateWrapper(GymDynamicConflictEnv(factory(), scenarios), advice)


def make_time_env(factory: Callable[[], BehaviorModel], scenarios):
    return ReopenForecastObservationWrapper(HoldEnabledActionWrapper(GymDynamicConflictEnv(factory(), scenarios)))


def run_forced_plan(env, scenario, episode_seed: int, plan: str, collect: bool = False):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    observations: list[np.ndarray] = []
    actions: list[int] = []
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = plan_action(plan, int(info.get("blockage_steps_remaining", 0)))
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


def build_counterfactual_dataset(factory, scenarios, advice, repetitions: int):
    env = make_gate_env(factory, scenarios, advice)
    result_rows: list[dict[str, object]] = []
    sample_observations: list[np.ndarray] = []
    sample_actions: list[int] = []
    sample_splits: list[str] = []
    sample_scenarios: list[str] = []
    sample_conflicts: list[str] = []
    selected_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition in range(repetitions):
            episode_seed = 200_000 + scenario_index * repetitions + repetition
            results: dict[str, dict[str, float]] = {}
            per_plan: dict[str, dict[str, object]] = {}
            for plan in PLANS:
                metrics, _, _ = run_forced_plan(env, scenario, episode_seed, plan)
                results[plan] = metrics
                per_plan[plan] = metrics
            selected = select_safest_plan(results)
            split = "supervised_train" if repetition < max(1, int(np.floor(repetitions * 0.8))) else "supervised_validation"
            selected_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "population_seed": episode_seed,
                    "split": split,
                    "selected_plan": selected,
                }
            )
            for plan in PLANS:
                result_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "conflict_type": scenario.conflict_type,
                        "population_seed": episode_seed,
                        "split": split,
                        "plan": plan,
                        "selected": int(plan == selected),
                        **per_plan[plan],
                    }
                )
            _, observations, actions = run_forced_plan(env, scenario, episode_seed, selected, collect=True)
            sample_observations.extend(observations)
            sample_actions.extend(actions)
            sample_splits.extend([split] * len(actions))
            sample_scenarios.extend([scenario.scenario_id] * len(actions))
            sample_conflicts.extend([scenario.conflict_type] * len(actions))
    env.close()
    return (
        result_rows,
        selected_rows,
        np.asarray(sample_observations, dtype=np.float32),
        np.asarray(sample_actions, dtype=np.int64),
        np.asarray(sample_splits),
        np.asarray(sample_scenarios),
        np.asarray(sample_conflicts),
    )


def label_summary(selected_rows, actions, splits, conflicts):
    output: list[dict[str, object]] = []
    for split in ("all", "supervised_train", "supervised_validation"):
        for conflict in ("all", *TRUST_GATE_TYPES):
            episode_subset = [
                row
                for row in selected_rows
                if (split == "all" or row["split"] == split)
                and (conflict == "all" or row["conflict_type"] == conflict)
            ]
            mask = np.ones(len(actions), dtype=bool)
            if split != "all":
                mask &= splits == split
            if conflict != "all":
                mask &= conflicts == conflict
            step_actions = actions[mask]
            plan_counts = {plan: sum(row["selected_plan"] == plan for row in episode_subset) for plan in PLANS}
            action_counts = np.bincount(step_actions, minlength=3) if len(step_actions) else np.zeros(3, dtype=int)
            output.append(
                {
                    "split": split,
                    "conflict_type": conflict,
                    "episodes": len(episode_subset),
                    "steps": int(len(step_actions)),
                    "selected_trust": plan_counts["trust"],
                    "selected_override": plan_counts["override"],
                    "selected_wait_then_trust": plan_counts["wait_then_trust"],
                    "trust_labels": int(action_counts[0]),
                    "override_labels": int(action_counts[1]),
                    "wait_labels": int(action_counts[2]),
                }
            )
    return output


def classification_metrics(model, observations, actions, mask):
    if not np.any(mask):
        return float("nan")
    return policy_accuracy(model, observations[mask], actions[mask])


def run_model_episode(env, model, policy: str, scenario, episode_seed: int, advice):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    target = advice[scenario.scenario_id].target_branch
    trust_steps = override_steps = wait_steps = steps = 0
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if policy != "ppo_time":
            gate_action = str(info["trust_gate_action"])
            trust_steps += int(gate_action == "trust")
            override_steps += int(gate_action == "override")
            wait_steps += int(gate_action == "wait")
        else:
            hold = bool(info.get("hold_action_requested", False))
            wait_steps += int(hold)
            if not hold:
                trust_steps += int(int(info["last_action"]) == target)
                override_steps += int(int(info["last_action"]) != target)
    return {
        "scenario_id": scenario.scenario_id,
        "conflict_type": scenario.conflict_type,
        "llm_target_correct": int(target == safe_forecast_target(scenario)),
        "population_seed": episode_seed,
        "steps": steps,
        "trust_rate": trust_steps / steps,
        "override_rate": override_steps / steps,
        "wait_rate": wait_steps / steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": total_reward,
    }


def aggregate_by_conflict(per_seed, samples: int, seed: int):
    output: list[dict[str, object]] = []
    for index, conflict in enumerate(sorted({str(row["test_behavior"]) for row in per_seed})):
        subset = [row for row in per_seed if row["test_behavior"] == conflict]
        rows = aggregate_seed_summaries(
            subset, metrics=DYNAMIC_METRICS, bootstrap_samples=samples, bootstrap_seed=seed + index * 1000
        )
        output.extend({"conflict_type": conflict, **row} for row in rows)
    return output


def paired_comparison(per_seed, candidate: str, baseline: str, samples: int, seed: int):
    by_seed: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for index, conflict in enumerate(sorted({str(row["test_behavior"]) for row in per_seed})):
        subset = [row for row in per_seed if row["test_behavior"] == conflict]
        differences = paired_seed_differences(
            subset, candidate=candidate, baseline=baseline, metrics=DYNAMIC_METRICS
        )
        by_seed.extend({"conflict_type": conflict, **row} for row in differences)
        rows = aggregate_paired_differences(
            differences, bootstrap_samples=samples, bootstrap_seed=seed + index * 1000
        )
        aggregate.extend(
            {
                "conflict_type": conflict,
                "candidate": candidate,
                "baseline": baseline,
                **{key: value for key, value in row.items() if key != "policy"},
            }
            for row in rows
        )
    return by_seed, aggregate


def decision_summary(rows, seeds, gate_policies=GATE_POLICIES):
    output: list[dict[str, object]] = []
    for policy in gate_policies:
        for conflict in ("all", *TRUST_GATE_TYPES):
            for seed in seeds:
                subset = [
                    row
                    for row in rows
                    if row["training_seed"] == seed
                    and row["policy"] == policy
                    and (conflict == "all" or row["conflict_type"] == conflict)
                ]
                output.append(
                    {
                        "policy": policy,
                        "conflict_type": conflict,
                        "training_seed": seed,
                        "episodes": len(subset),
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
    if args.train_scenarios < len(TRUST_GATE_TYPES) or args.test_scenarios < len(TRUST_GATE_TYPES):
        raise SystemExit(f"scenario counts must be at least {len(TRUST_GATE_TYPES)}")
    if args.counterfactual_repetitions < 2 or args.eval_repetitions < 1:
        raise SystemExit("counterfactual repetitions must be >=2 and evaluation repetitions must be positive")
    if args.timesteps < 1_000 or args.n_envs < 1 or args.supervised_epochs < 1:
        raise SystemExit("timesteps must be >=1000; n-envs and supervised epochs must be positive")

    started = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    train_scenarios = make_trust_gate_bank(args.train_scenarios, args.scenario_seed, "GTR")
    test_scenarios = make_trust_gate_bank(args.test_scenarios, args.scenario_seed + 1, "GTE")
    all_scenarios = train_scenarios + test_scenarios
    advice = {scenario.scenario_id: conflict_llm_advice(scenario) for scenario in all_scenarios}
    behavior_factory = lambda: CalibratedLogitBehavior(args.calibration_artifact)
    check_env(make_gate_env(behavior_factory, train_scenarios, advice), warn=True)

    print("Generating counterfactual supervision from training scenarios...", flush=True)
    (
        counterfactual_rows,
        selected_rows,
        observations,
        actions,
        splits,
        sample_scenarios,
        sample_conflicts,
    ) = build_counterfactual_dataset(
        behavior_factory, train_scenarios, advice, args.counterfactual_repetitions
    )
    write_csv(counterfactual_rows, args.output / "counterfactual_plan_results.csv")
    write_csv(selected_rows, args.output / "counterfactual_episode_labels.csv")
    summaries = label_summary(selected_rows, actions, splits, sample_conflicts)
    write_csv(summaries, args.output / "counterfactual_label_summary.csv")
    np.savez_compressed(
        args.output / "supervised_dataset.npz",
        observations=observations,
        actions=actions,
        splits=splits,
        scenario_ids=sample_scenarios,
        conflict_types=sample_conflicts,
    )
    print(
        f"Counterfactual data: {len(selected_rows)} episodes, {len(actions)} labeled decisions; "
        f"action counts={np.bincount(actions, minlength=3).tolist()}",
        flush=True,
    )

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
    train_mask = splits == "supervised_train"
    validation_mask = splits == "supervised_validation"
    model_paths: dict[tuple[int, str], Path] = {}
    training_rows: list[dict[str, object]] = []
    pretraining_logs: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds, 1):
        print(f"Seed {seed} ({seed_index}/{len(seeds)}): supervised pretraining...", flush=True)

        def gate_factory():
            return make_gate_env(behavior_factory, train_scenarios, advice)

        vector_env = make_vec_env(gate_factory, n_envs=args.n_envs, seed=seed)
        model = PPO(env=vector_env, seed=seed, **common)
        random_train = classification_metrics(model, observations, actions, train_mask)
        random_validation = classification_metrics(model, observations, actions, validation_mask)
        logs = pretrain_ppo_gate(
            model,
            observations[train_mask],
            actions[train_mask],
            epochs=args.supervised_epochs,
            batch_size=256,
            seed=seed,
        )
        supervised_train = classification_metrics(model, observations, actions, train_mask)
        supervised_validation = classification_metrics(model, observations, actions, validation_mask)
        supervised_path = model_dir / f"seed_{seed}_supervised_only"
        model.save(supervised_path)
        model_paths[(seed, "supervised_only")] = supervised_path.with_suffix(".zip")
        training_rows.append(
            {
                "training_seed": seed,
                "random_train_accuracy": random_train,
                "random_validation_accuracy": random_validation,
                "supervised_train_accuracy": supervised_train,
                "supervised_validation_accuracy": supervised_validation,
                "supervised_epochs": args.supervised_epochs,
            }
        )
        pretraining_logs.extend({"training_seed": seed, **row} for row in logs)
        print(
            f"  validation accuracy {random_validation:.1%} -> {supervised_validation:.1%}; PPO fine-tuning...",
            flush=True,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False, reset_num_timesteps=False)
        finetuned_path = model_dir / f"seed_{seed}_supervised_finetuned"
        model.save(finetuned_path)
        model_paths[(seed, "supervised_finetuned")] = finetuned_path.with_suffix(".zip")
        training_rows[-1].update(
            {
                "post_ppo_train_accuracy": classification_metrics(model, observations, actions, train_mask),
                "post_ppo_validation_accuracy": classification_metrics(model, observations, actions, validation_mask),
                "ppo_timesteps": int(model.num_timesteps),
            }
        )
        vector_env.close()
        for policy, old_name in (("ppo_time", "ppo_time"), ("scratch_gate", "llm_trust_gate_ppo")):
            baseline_path = args.baseline_model_dir / f"seed_{seed}_{old_name}.zip"
            if not baseline_path.exists():
                raise FileNotFoundError(f"missing baseline model: {baseline_path}")
            model_paths[(seed, policy)] = baseline_path

    write_csv(training_rows, args.output / "supervised_training_summary.csv")
    write_csv(pretraining_logs, args.output / "supervised_training_curve.csv")

    print("Running locked test comparison...", flush=True)
    episode_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds, 1):
        print(f"Evaluating seed {seed} ({seed_index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)], device="cpu") for policy in POLICIES}
        envs = {
            policy: make_time_env(behavior_factory, test_scenarios)
            if policy == "ppo_time"
            else make_gate_env(behavior_factory, test_scenarios, advice)
            for policy in POLICIES
        }
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                episode_seed = 100_000 + scenario_index * args.eval_repetitions + repetition
                for policy in POLICIES:
                    row = run_model_episode(
                        envs[policy], models[policy], policy, scenario, episode_seed, advice
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
    comparison_specs = (
        ("supervised_finetuned", "scratch_gate"),
        ("supervised_finetuned", "ppo_time"),
        ("supervised_only", "scratch_gate"),
    )
    paired_seed_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for index, (candidate, baseline) in enumerate(comparison_specs):
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
    decisions = decision_summary(episode_rows, seeds)
    write_csv(decisions, args.output / "trust_gate_decisions.csv")

    (args.output / "train_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(train_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "training_seeds": seeds,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "counterfactual_repetitions": args.counterfactual_repetitions,
        "counterfactual_plans": list(PLANS),
        "counterfactual_selection": "Lexicographic minimum: incapacitation rate, mean FED, then maximum reward.",
        "supervised_epochs": args.supervised_epochs,
        "ppo_timesteps_per_model": args.timesteps,
        "baseline_models_reused_from": str(args.baseline_model_dir.resolve()),
        "reopen_forecast": "Exact remaining blockage steps, normalized by 12.",
        "llm_advice": "Replay of the previously audited qwen3:4b policy; no new model calls.",
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
            metric: float(
                np.mean(
                    [
                        row[metric]
                        for row in decisions
                        if row["policy"] == policy and row["conflict_type"] == conflict
                    ]
                )
            )
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
        "# 反事实监督预训练 + PPO微调的LLM信任门控实验",
        "",
        f"只用 {len(train_scenarios)} 个训练场景生成反事实标签：每个场景使用 "
        f"{args.counterfactual_repetitions} 个人群随机种子，分别完整模拟始终相信、始终覆盖、等待恢复后再相信。",
        "每个回合按失能率、平均累计FED、总奖励的字典序选择标签。最后20%的人群种子仅用于监督验证；20个锁定测试场景没有参与标签生成。",
        "",
        "## 监督学习是否学会标签",
        "",
        "| 训练种子 | 随机验证准确率 | 监督后 | PPO微调后 |",
        "|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        report.append(
            f"| {row['training_seed']} | {float(row['random_validation_accuracy']):.1%} | "
            f"{float(row['supervised_validation_accuracy']):.1%} | "
            f"{float(row['post_ppo_validation_accuracy']):.1%} |"
        )
    label_lookup = {
        str(row["conflict_type"]): row
        for row in summaries
        if row["split"] == "all" and row["conflict_type"] != "all"
    }
    report.extend(
        [
            "",
            "## 反事实标签实际表达的策略",
            "",
            "| 场景 | 相信计划 | 覆盖计划 | 等待后相信计划 |",
            "|---|---:|---:|---:|",
        ]
    )
    for conflict, label in {key: value for key, value in labels.items() if key != "all"}.items():
        row = label_lookup[conflict]
        report.append(
            f"| {label} | {int(row['selected_trust'])}/{int(row['episodes'])} | "
            f"{int(row['selected_override'])}/{int(row['episodes'])} | "
            f"{int(row['selected_wait_then_trust'])}/{int(row['episodes'])} |"
        )
    report.extend(
        [
            "",
            "## 锁定测试结果",
            "",
            "失能率越低越好；括号中是门控的相信/覆盖/等待比例。",
            "",
            "| 场景 | 时间PPO | 从零门控 | 仅监督门控 | 监督+PPO门控 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for conflict, label in labels.items():
        cells = []
        for policy in POLICIES:
            value = float(agg[(conflict, policy, "incapacitation_rate")]["mean"])
            if policy in GATE_POLICIES:
                rates = decision_lookup[(policy, conflict)]
                cells.append(
                    f"{value:.2%} ({rates['trust_rate']:.0%}/{rates['override_rate']:.0%}/{rates['wait_rate']:.0%})"
                )
            else:
                cells.append(f"{value:.2%}")
        report.append(f"| {label} | " + " | ".join(cells) + " |")

    fine_scratch = paired[("all", "supervised_finetuned", "scratch_gate", "incapacitation_rate")]
    fine_time = paired[("all", "supervised_finetuned", "ppo_time", "incapacitation_rate")]
    fine_time_fed = paired[("all", "supervised_finetuned", "ppo_time", "mean_cumulative_fed")]
    supervised_scratch = paired[("all", "supervised_only", "scratch_gate", "incapacitation_rate")]
    clear_rates = decision_lookup[("supervised_finetuned", "clear")]
    short_rates = decision_lookup[("supervised_finetuned", "safe_exit_short_block")]
    long_rates = decision_lookup[("supervised_finetuned", "safe_exit_long_block")]
    error_rates = decision_lookup[("supervised_finetuned", "forecast_error")]
    validation_before = float(np.mean([row["random_validation_accuracy"] for row in training_rows]))
    validation_supervised = float(np.mean([row["supervised_validation_accuracy"] for row in training_rows]))
    validation_after = float(np.mean([row["post_ppo_validation_accuracy"] for row in training_rows]))
    report.extend(
        [
            "",
            "## 主要判定",
            "",
            f"- 监督验证准确率从平均 {validation_before:.1%} 提高到 {validation_supervised:.1%}；PPO微调后为 {validation_after:.1%}。",
            f"- 监督+PPO相对从零门控的整体失能率差为 {float(fine_scratch['mean']):.2%}，95%区间 "
            f"[{float(fine_scratch['bootstrap_ci95_lower']):.2%}, {float(fine_scratch['bootstrap_ci95_upper']):.2%}]。",
            f"- 监督+PPO相对时间PPO的整体失能率差为 {float(fine_time['mean']):.2%}，95%区间 "
            f"[{float(fine_time['bootstrap_ci95_lower']):.2%}, {float(fine_time['bootstrap_ci95_upper']):.2%}]；FED差为 "
            f"{float(fine_time_fed['mean']):.4f}，95%区间 [{float(fine_time_fed['bootstrap_ci95_lower']):.4f}, "
            f"{float(fine_time_fed['bootstrap_ci95_upper']):.4f}]。",
            f"- 仅监督门控相对从零门控的整体失能率差为 {float(supervised_scratch['mean']):.2%}，95%区间 "
            f"[{float(supervised_scratch['bootstrap_ci95_lower']):.2%}, {float(supervised_scratch['bootstrap_ci95_upper']):.2%}]。",
            f"- 监督+PPO的动作语义：无冲突信任 {clear_rates['trust_rate']:.1%}；短时封闭等待 {short_rates['wait_rate']:.1%}；"
            f"长时封闭覆盖 {long_rates['override_rate']:.1%}；预测错误覆盖 {error_rates['override_rate']:.1%}。",
            "",
            "## 结论",
            "",
            "**本轮按预先设定的成功标准判定为失败。** 监督预训练确实学会了反事实标签，但这些标签没有形成“短时等待、长期覆盖”的预期语义；"
            "标准PPO微调又把等待动作降为0%，出现明显的监督语义遗忘。监督+PPO的失能率和FED都稳定劣于时间PPO与从零训练门控。",
            "",
            "失败并非只来自分类器：当前仿真中，短时堵塞时让人员先进入安全支路排队通常比留在房间更安全；"
            "长时堵塞时，等待恢复后走安全出口又经常比立即改走烟气侧出口更安全。因此，反事实最优标签本身就反驳了原先手工设想的动作对应关系。",
            "",
            "下一轮应同时处理两个问题：把监督损失持续混入PPO更新以防遗忘，并扩展反事实候选策略（限定等待时长、分流比例和动态覆盖时机），"
            "而不是强迫三个固定计划产生预期标签。",
            "",
            "## 解释边界",
            "",
            "- LLM建议是此前审计过的 qwen3:4b 策略回放，本轮新增Ollama调用为0。",
            "- 恢复时间是准确值，尚未加入预测噪声；火灾与人群行为仍为合成先导模型。",
            "- 反事实监督只比较三个固定高层计划，并不代表找到了每一步的全局最优动作。",
            "- 结果只能支持仿真中的算法验证，不能宣称真实建筑部署有效。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
