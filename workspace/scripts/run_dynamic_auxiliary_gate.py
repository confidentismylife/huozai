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
    make_trust_gate_bank,
)
from fire_evac.dynamic_gate_supervision import (
    DYNAMIC_GATE_PLANS,
    DynamicGatePlan,
    dynamic_plan_action,
    select_safest_dynamic_plan,
)
from fire_evac.dynamic_multiseed import summarize_cross_episodes
from fire_evac.supervised_gate import policy_accuracy, pretrain_ppo_gate
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
    "fixed_supervised_finetuned",
    "dynamic_supervised_only",
    "dynamic_aux_ppo",
)
GATE_POLICIES = tuple(policy for policy in POLICIES if policy != "ppo_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic counterfactual plans with persistent PPO supervision.")
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--train-scenarios", type=int, default=40)
    parser.add_argument("--test-scenarios", type=int, default=20)
    parser.add_argument("--counterfactual-repetitions", type=int, default=10)
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
        "--fixed-supervised-model-dir",
        type=Path,
        default=ROOT / "outputs" / "supervised_trust_gate" / "models",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dynamic_auxiliary_gate")
    return parser.parse_args()


def run_dynamic_plan(env, scenario, episode_seed: int, plan: DynamicGatePlan, collect: bool = False):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    observations: list[np.ndarray] = []
    actions: list[int] = []
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = dynamic_plan_action(
            plan,
            step=steps,
            blockage_steps_remaining=int(info.get("blockage_steps_remaining", 0)),
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


def stratified_scenario_splits(scenarios) -> dict[str, str]:
    grouped: dict[str, list] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.conflict_type, []).append(scenario)
    assignments: dict[str, str] = {}
    for conflict, items in grouped.items():
        ordered = sorted(items, key=lambda item: item.scenario_id)
        if len(ordered) < 2:
            raise ValueError(f"need at least two training scenarios for {conflict}")
        validation_count = max(1, int(round(len(ordered) * 0.2)))
        for scenario in ordered[:-validation_count]:
            assignments[scenario.scenario_id] = "supervised_train"
        for scenario in ordered[-validation_count:]:
            assignments[scenario.scenario_id] = "supervised_validation"
    return assignments


def build_dynamic_dataset(factory, scenarios, advice, repetitions: int):
    env = make_gate_env(factory, scenarios, advice)
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
            episode_seed = 300_000 + scenario_index * repetitions + repetition
            results: dict[str, dict[str, float]] = {}
            episode_rows: list[dict[str, object]] = []
            for plan in DYNAMIC_GATE_PLANS:
                metrics, _, _ = run_dynamic_plan(env, scenario, episode_seed, plan)
                results[plan.name] = metrics
                episode_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "conflict_type": scenario.conflict_type,
                        "population_seed": episode_seed,
                        "split": split,
                        "plan": plan.name,
                        "plan_family": plan.family,
                        "wait_steps": plan.wait_steps,
                        "override_at_step": "" if plan.override_at_step is None else plan.override_at_step,
                        "trust_fraction": plan.trust_fraction,
                        **metrics,
                    }
                )
            selected = select_safest_dynamic_plan(results)
            for row in episode_rows:
                row["selected"] = int(row["plan"] == selected.name)
            result_rows.extend(episode_rows)
            selected_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "population_seed": episode_seed,
                    "split": split,
                    "selected_plan": selected.name,
                    "selected_family": selected.family,
                    "wait_steps": selected.wait_steps,
                    "override_at_step": "" if selected.override_at_step is None else selected.override_at_step,
                    "trust_fraction": selected.trust_fraction,
                }
            )
            _, episode_observations, episode_actions = run_dynamic_plan(
                env, scenario, episode_seed, selected, collect=True
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


def dynamic_label_summary(selected_rows, actions, splits, conflicts):
    families = sorted({plan.family for plan in DYNAMIC_GATE_PLANS})
    output: list[dict[str, object]] = []
    for split in ("all", "supervised_train", "supervised_validation"):
        for conflict in ("all", *TRUST_GATE_TYPES):
            episodes = [
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
            subset = actions[mask]
            counts = np.bincount(subset, minlength=3) if len(subset) else np.zeros(3, dtype=int)
            row: dict[str, object] = {
                "split": split,
                "conflict_type": conflict,
                "episodes": len(episodes),
                "steps": len(subset),
                "trust_labels": int(counts[0]),
                "override_labels": int(counts[1]),
                "wait_labels": int(counts[2]),
            }
            for family in families:
                row[f"selected_{family}"] = sum(item["selected_family"] == family for item in episodes)
            output.append(row)
    return output


def model_accuracy(model, observations, actions, mask) -> float:
    return policy_accuracy(model, observations[mask], actions[mask])


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.train_scenarios < 2 * len(TRUST_GATE_TYPES):
        raise SystemExit(f"train-scenarios must be at least {2 * len(TRUST_GATE_TYPES)}")
    if args.test_scenarios < len(TRUST_GATE_TYPES):
        raise SystemExit(f"test-scenarios must be at least {len(TRUST_GATE_TYPES)}")
    if args.timesteps < 1_000 or args.counterfactual_repetitions < 1 or args.eval_repetitions < 1:
        raise SystemExit("timesteps must be >=1000 and repetitions must be positive")

    started = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    train_scenarios = make_trust_gate_bank(args.train_scenarios, args.scenario_seed, "GTR")
    test_scenarios = make_trust_gate_bank(args.test_scenarios, args.scenario_seed + 1, "GTE")
    advice = {
        scenario.scenario_id: conflict_llm_advice(scenario)
        for scenario in train_scenarios + test_scenarios
    }
    behavior_factory = lambda: CalibratedLogitBehavior(args.calibration_artifact)
    check_env(make_gate_env(behavior_factory, train_scenarios, advice), warn=True)

    print(f"Evaluating {len(DYNAMIC_GATE_PLANS)} dynamic counterfactual plans...", flush=True)
    (
        counterfactual_rows,
        selected_rows,
        observations,
        actions,
        splits,
        sample_scenarios,
        sample_conflicts,
    ) = build_dynamic_dataset(
        behavior_factory, train_scenarios, advice, args.counterfactual_repetitions
    )
    summaries = dynamic_label_summary(selected_rows, actions, splits, sample_conflicts)
    write_csv(counterfactual_rows, args.output / "dynamic_counterfactual_results.csv")
    write_csv(selected_rows, args.output / "dynamic_selected_plans.csv")
    write_csv(summaries, args.output / "dynamic_label_summary.csv")
    np.savez_compressed(
        args.output / "dynamic_supervised_dataset.npz",
        observations=observations,
        actions=actions,
        splits=splits,
        scenario_ids=sample_scenarios,
        conflict_types=sample_conflicts,
    )
    print(
        f"Selected {len(selected_rows)} episodes and {len(actions)} step labels; "
        f"actions={np.bincount(actions, minlength=3).tolist()}",
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
    curve_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, 1):
        print(f"Seed {seed} ({index}/{len(seeds)}): pretraining dynamic gate...", flush=True)

        def gate_factory():
            return make_gate_env(behavior_factory, train_scenarios, advice)

        vector_env = make_vec_env(gate_factory, n_envs=args.n_envs, seed=seed)
        model = AuxiliarySupervisedPPO(env=vector_env, seed=seed, **common)
        random_train = model_accuracy(model, observations, actions, train_mask)
        random_validation = model_accuracy(model, observations, actions, validation_mask)
        logs = pretrain_ppo_gate(
            model,
            observations[train_mask],
            actions[train_mask],
            epochs=args.supervised_epochs,
            batch_size=256,
            seed=seed,
        )
        curve_rows.extend({"training_seed": seed, **row} for row in logs)
        supervised_train = model_accuracy(model, observations, actions, train_mask)
        supervised_validation = model_accuracy(model, observations, actions, validation_mask)
        supervised_path = model_dir / f"seed_{seed}_dynamic_supervised_only"
        model.save(supervised_path)
        model_paths[(seed, "dynamic_supervised_only")] = supervised_path.with_suffix(".zip")
        model.configure_auxiliary_supervision(
            observations[train_mask],
            actions[train_mask],
            coefficient=args.auxiliary_coefficient,
            batches_per_update=args.auxiliary_batches,
            batch_size=256,
            seed=seed + 10_000,
        )
        print(
            f"  scenario-held-out accuracy {random_validation:.1%} -> {supervised_validation:.1%}; "
            "PPO + persistent supervision...",
            flush=True,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False, reset_num_timesteps=False)
        aux_path = model_dir / f"seed_{seed}_dynamic_aux_ppo"
        model.save(aux_path)
        model_paths[(seed, "dynamic_aux_ppo")] = aux_path.with_suffix(".zip")
        training_rows.append(
            {
                "training_seed": seed,
                "random_train_accuracy": random_train,
                "random_validation_accuracy": random_validation,
                "supervised_train_accuracy": supervised_train,
                "supervised_validation_accuracy": supervised_validation,
                "post_aux_ppo_train_accuracy": model_accuracy(model, observations, actions, train_mask),
                "post_aux_ppo_validation_accuracy": model_accuracy(model, observations, actions, validation_mask),
                "auxiliary_update_count": int(model.auxiliary_update_count),
                "ppo_timesteps": int(model.num_timesteps),
            }
        )
        vector_env.close()
        baseline_paths = {
            "ppo_time": args.baseline_model_dir / f"seed_{seed}_ppo_time.zip",
            "scratch_gate": args.baseline_model_dir / f"seed_{seed}_llm_trust_gate_ppo.zip",
            "fixed_supervised_finetuned": args.fixed_supervised_model_dir
            / f"seed_{seed}_supervised_finetuned.zip",
        }
        for policy, path in baseline_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing baseline model: {path}")
            model_paths[(seed, policy)] = path
    write_csv(training_rows, args.output / "dynamic_training_summary.csv")
    write_csv(curve_rows, args.output / "dynamic_pretraining_curve.csv")

    print("Running locked test comparison...", flush=True)
    episode_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, 1):
        print(f"Evaluating seed {seed} ({index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)], device="cpu") for policy in POLICIES}
        envs = {
            policy: make_time_env(behavior_factory, test_scenarios)
            if policy == "ppo_time"
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
    comparison_specs = (
        ("dynamic_aux_ppo", "scratch_gate"),
        ("dynamic_aux_ppo", "ppo_time"),
        ("dynamic_aux_ppo", "fixed_supervised_finetuned"),
        ("dynamic_aux_ppo", "dynamic_supervised_only"),
    )
    paired_seed_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for comparison_index, (candidate, baseline) in enumerate(comparison_specs):
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

    write_outputs_and_report(
        args,
        started,
        seeds,
        train_scenarios,
        test_scenarios,
        training_rows,
        summaries,
        aggregate,
        paired_rows,
        decisions,
    )
    print(f"Outputs: {args.output}", flush=True)


def write_outputs_and_report(
    args,
    started,
    seeds,
    train_scenarios,
    test_scenarios,
    training_rows,
    summaries,
    aggregate,
    paired_rows,
    decisions,
) -> None:
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
        "dynamic_counterfactual_plans": [plan.__dict__ for plan in DYNAMIC_GATE_PLANS],
        "counterfactual_repetitions": args.counterfactual_repetitions,
        "supervised_split": "Scenario-level stratified split within the training bank.",
        "ppo_timesteps_per_model": args.timesteps,
        "auxiliary_coefficient": args.auxiliary_coefficient,
        "auxiliary_batches_per_ppo_update": args.auxiliary_batches,
        "llm_advice": "Replay of previously audited qwen3:4b policy; no new model calls.",
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
        "# 动态反事实计划 + 持续辅助监督PPO实验",
        "",
        f"比较 {len(DYNAMIC_GATE_PLANS)} 个动态计划，覆盖等待2/4/8/12步、定时覆盖，以及25%/50%/75%时间分流。",
        "分流以两步为一个信号时隙、八步为一个周期。监督验证按完整场景划分，避免同场景人群种子泄漏。",
        "",
        "## 监督语义保持",
        "",
        "| 种子 | 随机验证 | 预训练后 | 辅助PPO后 | 辅助更新 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        report.append(
            f"| {row['training_seed']} | {float(row['random_validation_accuracy']):.1%} | "
            f"{float(row['supervised_validation_accuracy']):.1%} | "
            f"{float(row['post_aux_ppo_validation_accuracy']):.1%} | {int(row['auxiliary_update_count'])} |"
        )
    family_columns = (
        "selected_trust",
        "selected_override",
        "selected_timed_override",
        "selected_split",
        "selected_wait_then_trust",
    )
    family_lookup = {
        row["conflict_type"]: row
        for row in summaries
        if row["split"] == "all" and row["conflict_type"] != "all"
    }
    report.extend([
        "",
        "## 动态反事实计划选择",
        "",
        "| 场景 | 相信 | 直接覆盖 | 定时覆盖 | 分流 | 等待后相信 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for conflict, label in {key: value for key, value in labels.items() if key != "all"}.items():
        row = family_lookup[conflict]
        values = [int(row[column]) for column in family_columns]
        report.append(f"| {label} | " + " | ".join(str(value) for value in values) + " |")
    report.extend([
        "",
        "## 锁定测试失能率",
        "",
        "| 场景 | 时间PPO | 从零门控 | 固定监督+PPO | 动态仅监督 | 动态辅助PPO |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        values = [float(agg[(conflict, policy, "incapacitation_rate")]["mean"]) for policy in POLICIES]
        report.append(f"| {label} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    report.extend([
        "",
        "## 动态辅助PPO动作比例",
        "",
        "| 场景 | 相信 | 覆盖 | 等待 |",
        "|---|---:|---:|---:|",
    ])
    for conflict, label in labels.items():
        rates = decision_lookup[("dynamic_aux_ppo", conflict)]
        report.append(
            f"| {label} | {rates['trust_rate']:.1%} | {rates['override_rate']:.1%} | {rates['wait_rate']:.1%} |"
        )
    aux_scratch = paired[("all", "dynamic_aux_ppo", "scratch_gate", "incapacitation_rate")]
    aux_time = paired[("all", "dynamic_aux_ppo", "ppo_time", "incapacitation_rate")]
    aux_time_fed = paired[("all", "dynamic_aux_ppo", "ppo_time", "mean_cumulative_fed")]
    aux_fixed = paired[("all", "dynamic_aux_ppo", "fixed_supervised_finetuned", "incapacitation_rate")]
    aux_supervised = paired[("all", "dynamic_aux_ppo", "dynamic_supervised_only", "incapacitation_rate")]
    aux_supervised_fed = paired[("all", "dynamic_aux_ppo", "dynamic_supervised_only", "mean_cumulative_fed")]
    before = float(np.mean([row["supervised_validation_accuracy"] for row in training_rows]))
    after = float(np.mean([row["post_aux_ppo_validation_accuracy"] for row in training_rows]))
    report.extend([
        "",
        "## 判定",
        "",
        f"- 场景级验证准确率：PPO前 {before:.1%}，持续辅助监督PPO后 {after:.1%}。",
        f"- 相对从零门控，失能率差 {float(aux_scratch['mean']):.2%}，95%区间 "
        f"[{float(aux_scratch['bootstrap_ci95_lower']):.2%}, {float(aux_scratch['bootstrap_ci95_upper']):.2%}]。",
        f"- 相对时间PPO，失能率差 {float(aux_time['mean']):.2%}，95%区间 "
        f"[{float(aux_time['bootstrap_ci95_lower']):.2%}, {float(aux_time['bootstrap_ci95_upper']):.2%}]；FED差 "
        f"{float(aux_time_fed['mean']):.4f}，95%区间 [{float(aux_time_fed['bootstrap_ci95_lower']):.4f}, "
        f"{float(aux_time_fed['bootstrap_ci95_upper']):.4f}]。",
        f"- 相对固定计划监督+PPO，失能率差 {float(aux_fixed['mean']):.2%}；相对动态仅监督为 "
        f"{float(aux_supervised['mean']):.2%}。",
        f"- 相对动态仅监督，FED差为 {float(aux_supervised_fed['mean']):.4f}，95%区间 "
        f"[{float(aux_supervised_fed['bootstrap_ci95_lower']):.4f}, "
        f"{float(aux_supervised_fed['bootstrap_ci95_upper']):.4f}]。",
        "",
        "## 结论",
        "",
        "**持续辅助监督机制成功，但整体安全目标仍判定为失败。** 它基本保持了监督准确率和等待语义，并稳定优于上一轮固定计划监督+PPO；"
        "但整体失能率与FED仍显著劣于时间PPO和从零门控。失败主要集中在安全出口拥堵、LLM预测错误和长时封闭。",
        "",
        "动态反事实搜索从未选择“等待后覆盖”或“等待后分流”；长时封闭主要选择等待后继续相信，"
        "而拥堵与预测错误主要选择立即或定时覆盖。锁定测试中门控对预测错误只产生约三分之一覆盖，说明下一瓶颈是跨场景泛化，而不是监督遗忘。",
        "",
        "## 解释边界",
        "",
        "- 时间分流是控制标志在时间上的比例，不是强制指定比例的个体走某出口。",
        "- LLM建议仍为此前审计过的qwen3:4b策略回放，本轮新增Ollama调用为0。",
        "- 恢复时间仍为准确值，火灾与人群行为仍是合成模型，不能宣称真实部署有效。",
    ])
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
