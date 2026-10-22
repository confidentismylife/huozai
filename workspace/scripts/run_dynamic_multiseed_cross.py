from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import (
    BehaviorModel,
    CalibratedLogitBehavior,
    LLMSurrogateBehavior,
    MixedLogitBehavior,
    RuleBehavior,
)
from fire_evac.dynamic_llm_rl import (
    NORTH,
    DynamicLLMAdvice,
    LLMGuidedObservationWrapper,
    ScenarioBankHazardProvider,
    parse_dynamic_advice,
    safe_forecast_target,
)
from fire_evac.dynamic_multiseed import DYNAMIC_METRICS, cross_seed_statistics, summarize_cross_episodes
from fire_evac.gym_env import GymFireEvacuationEnv
from fire_evac.hazard import FireScenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed and cross-behavior dynamic LLM+PPO experiment.")
    parser.add_argument("--seeds", default="201,202,203,204,205")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "outputs" / "dynamic_llm_rl_v2",
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "dynamic_multiseed_cross",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    return seeds


def load_scenarios(path: Path) -> tuple[FireScenario, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = tuple(FireScenario(**item) for item in payload)
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError(f"duplicate scenario IDs in {path}")
    return scenarios


def load_advice(path: Path) -> dict[str, DynamicLLMAdvice]:
    advice: dict[str, DynamicLLMAdvice] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario_id = row["scenario_id"]
            if scenario_id in advice:
                raise ValueError(f"duplicate advice for {scenario_id}")
            advice[scenario_id] = parse_dynamic_advice(
                {
                    "target_exit": row["llm_target_exit"],
                    "confidence": float(row["confidence"]),
                    "reason_codes": row["reason_codes"].split("|"),
                }
            )
    return advice


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def behavior_factories(artifact: Path) -> dict[str, Callable[[], BehaviorModel]]:
    return {
        "rule": RuleBehavior,
        "mixed_logit": MixedLogitBehavior,
        "calibrated_logit": lambda: CalibratedLogitBehavior(artifact),
        "llm_surrogate_unvalidated": LLMSurrogateBehavior,
    }


def make_env(factory: Callable[[], BehaviorModel], scenarios, advice=None):
    base = GymFireEvacuationEnv(factory(), hazard_provider=ScenarioBankHazardProvider(scenarios))
    return LLMGuidedObservationWrapper(base, advice) if advice is not None else base


def run_episode(env, *, policy: str, scenario: FireScenario, episode_seed: int, model=None, advice=None):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        if policy in {"ppo", "llm_guided_ppo"}:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).item())
        elif policy == "hazard_aware_rule":
            occupancy = observation[:5]
            hazard = observation[5:10]
            action = int(3.0 * hazard[3] + occupancy[3] > 3.0 * hazard[4] + occupancy[4])
        elif policy == "local_llm_static":
            action = advice[scenario.scenario_id].target_branch
        else:
            raise ValueError(f"unknown policy: {policy}")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return {
        "scenario_id": scenario.scenario_id,
        "seed": episode_seed,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": float(total_reward),
    }


def summarize_baselines(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["test_behavior"]), str(row["policy"])), []).append(row)
    output: list[dict[str, object]] = []
    for (behavior, policy), items in sorted(groups.items()):
        record: dict[str, object] = {
            "test_behavior": behavior,
            "policy": policy,
            "episodes": len(items),
        }
        for metric in DYNAMIC_METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(record)
    return output


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("Install the project's optional RL dependencies first") from exc

    args = parse_args()
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.timesteps < 1_000 or args.n_envs < 1 or args.eval_repetitions < 1:
        raise SystemExit("timesteps must be >=1000; n-envs and eval-repetitions must be positive")
    required = (
        args.source_dir / "train_scenarios.json",
        args.source_dir / "test_scenarios.json",
        args.source_dir / "scenario_advice.csv",
        args.calibration_artifact,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required inputs are missing: {missing}")

    train_scenarios = load_scenarios(args.source_dir / "train_scenarios.json")
    test_scenarios = load_scenarios(args.source_dir / "test_scenarios.json")
    advice = load_advice(args.source_dir / "scenario_advice.csv")
    missing_advice = [s.scenario_id for s in train_scenarios + test_scenarios if s.scenario_id not in advice]
    if missing_advice:
        raise SystemExit(f"missing cached advice: {missing_advice}")
    accuracy = float(
        np.mean([advice[s.scenario_id].target_branch == safe_forecast_target(s) for s in train_scenarios + test_scenarios])
    )
    if accuracy < 1.0:
        raise SystemExit(f"v2 advice audit failed: safe forecast accuracy is {accuracy:.2%}")

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    factories = behavior_factories(args.calibration_artifact)
    training_factory = factories["calibrated_logit"]

    check_env(make_env(training_factory, train_scenarios), warn=True)
    check_env(make_env(training_factory, train_scenarios, advice), warn=True)
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
    training_records: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    for index, seed in enumerate(seeds, 1):
        for policy in ("ppo", "llm_guided_ppo"):
            print(f"Seed {seed} ({index}/{len(seeds)}): training {policy}...", flush=True)

            def factory(policy_name=policy):
                return make_env(
                    training_factory,
                    train_scenarios,
                    advice if policy_name == "llm_guided_ppo" else None,
                )

            vector_env = make_vec_env(factory, n_envs=args.n_envs, seed=seed)
            model = PPO(env=vector_env, seed=seed, **common)
            model.learn(total_timesteps=args.timesteps, progress_bar=False)
            path = model_dir / f"seed_{seed}_{policy}"
            model.save(path)
            model_paths[(seed, policy)] = path.with_suffix(".zip")
            training_records.append(
                {
                    "training_seed": seed,
                    "policy": policy,
                    "requested_timesteps": args.timesteps,
                    "actual_timesteps": int(model.num_timesteps),
                    "model_path": str(path.with_suffix('.zip').resolve()),
                }
            )
            vector_env.close()

    learned_rows: list[dict[str, object]] = []
    total_eval_groups = len(seeds) * len(factories)
    group_index = 0
    for seed in seeds:
        base_model = PPO.load(model_paths[(seed, "ppo")])
        guided_model = PPO.load(model_paths[(seed, "llm_guided_ppo")])
        for behavior_name, behavior_factory in factories.items():
            group_index += 1
            print(
                f"Evaluating seed {seed}, behavior {behavior_name} ({group_index}/{total_eval_groups})...",
                flush=True,
            )
            base_env = make_env(behavior_factory, test_scenarios)
            guided_env = make_env(behavior_factory, test_scenarios, advice)
            for scenario_index, scenario in enumerate(test_scenarios):
                for repetition in range(args.eval_repetitions):
                    episode_seed = 80_000 + scenario_index * args.eval_repetitions + repetition
                    for policy, env, model in (
                        ("ppo", base_env, base_model),
                        ("llm_guided_ppo", guided_env, guided_model),
                    ):
                        row = run_episode(
                            env,
                            policy=policy,
                            scenario=scenario,
                            episode_seed=episode_seed,
                            model=model,
                        )
                        row.update(
                            {
                                "test_behavior": behavior_name,
                                "training_seed": seed,
                                "policy": policy,
                            }
                        )
                        learned_rows.append(
                            {
                                "test_behavior": row.pop("test_behavior"),
                                "training_seed": row.pop("training_seed"),
                                "policy": row.pop("policy"),
                                **row,
                            }
                        )
            base_env.close()
            guided_env.close()

    baseline_rows: list[dict[str, object]] = []
    for behavior_name, behavior_factory in factories.items():
        env = make_env(behavior_factory, test_scenarios)
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                episode_seed = 80_000 + scenario_index * args.eval_repetitions + repetition
                for policy in ("hazard_aware_rule", "local_llm_static"):
                    row = run_episode(
                        env,
                        policy=policy,
                        scenario=scenario,
                        episode_seed=episode_seed,
                        advice=advice,
                    )
                    row.update({"test_behavior": behavior_name, "policy": policy})
                    baseline_rows.append(
                        {
                            "test_behavior": row.pop("test_behavior"),
                            "policy": row.pop("policy"),
                            **row,
                        }
                    )
        env.close()

    write_csv(learned_rows, args.output / "episode_results.csv")
    write_csv(baseline_rows, args.output / "baseline_episode_results.csv")
    per_seed = summarize_cross_episodes(learned_rows)
    write_csv(per_seed, args.output / "per_seed_cross_summary.csv")
    aggregate, paired, paired_aggregate = cross_seed_statistics(
        per_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=20260929,
    )
    write_csv(aggregate, args.output / "aggregate_cross_summary.csv")
    write_csv(paired, args.output / "paired_differences_by_seed.csv")
    write_csv(paired_aggregate, args.output / "paired_aggregate.csv")
    baseline_summary = summarize_baselines(baseline_rows)
    write_csv(baseline_summary, args.output / "baseline_summary.csv")

    static_lookup = {
        (str(row["test_behavior"]), str(row["scenario_id"]), int(row["seed"])): row
        for row in baseline_rows
        if row["policy"] == "local_llm_static"
    }
    equivalence_rows: list[dict[str, object]] = []
    for seed in seeds:
        guided = [
            row
            for row in learned_rows
            if int(row["training_seed"]) == seed and row["policy"] == "llm_guided_ppo"
        ]
        exact_matches = 0
        maximum_difference = 0.0
        for row in guided:
            reference = static_lookup[(str(row["test_behavior"]), str(row["scenario_id"]), int(row["seed"]))]
            difference = max(abs(float(row[metric]) - float(reference[metric])) for metric in DYNAMIC_METRICS)
            maximum_difference = max(maximum_difference, difference)
            exact_matches += int(difference < 1e-12)
        equivalence_rows.append(
            {
                "training_seed": seed,
                "guided_episodes": len(guided),
                "exact_static_matches": exact_matches,
                "exact_match_rate": exact_matches / len(guided),
                "maximum_absolute_metric_difference": maximum_difference,
            }
        )
    write_csv(equivalence_rows, args.output / "static_equivalence_audit.csv")
    guided_static_exact_match_rate = float(
        sum(int(row["exact_static_matches"]) for row in equivalence_rows)
        / sum(int(row["guided_episodes"]) for row in equivalence_rows)
    )

    metadata = {
        "training_seeds": seeds,
        "training_behavior": "calibrated_logit",
        "test_behaviors": list(factories),
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "eval_repetitions_per_scenario": args.eval_repetitions,
        "episodes_per_policy_seed_behavior": len(test_scenarios) * args.eval_repetitions,
        "timesteps_per_model": args.timesteps,
        "training_runs": training_records,
        "llm_advice_source": str((args.source_dir / 'scenario_advice.csv').resolve()),
        "additional_ollama_calls": 0,
        "llm_advice_safe_accuracy": accuracy,
        "guided_static_exact_match_rate": guided_static_exact_match_rate,
        "bootstrap_unit": "training seed",
        "bootstrap_samples": args.bootstrap_samples,
        "behavior_evidence_warning": "Calibration data and LLM-surrogate behavior are synthetic engineering placeholders.",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    aggregate_lookup = {
        (str(row["test_behavior"]), str(row["policy"]), str(row["metric"])): row for row in aggregate
    }
    paired_lookup = {(str(row["test_behavior"]), str(row["metric"])): row for row in paired_aggregate}
    behavior_labels = {
        "rule": "规则行为",
        "mixed_logit": "混合Logit",
        "calibrated_logit": "校准Logit",
        "llm_surrogate_unvalidated": "LLM代理行为",
    }
    report = [
        "# 动态LLM+PPO：多训练种子与跨行为模型验证",
        "",
        f"本实验固定 {len(train_scenarios)} 个训练火灾场景和 {len(test_scenarios)} 个锁定测试场景，"
        f"使用 {len(seeds)} 个独立PPO训练种子。每个种子分别训练基础PPO与LLM指导PPO各 {args.timesteps:,} 步。",
        f"每个学习策略在每类测试人群上运行 {len(test_scenarios) * args.eval_repetitions} 个配对回合。"
        "本轮直接复用已审计的LLM缓存，新增Ollama调用为0。",
        "",
        "| 测试人群 | 基础PPO失能率 | LLM+PPO失能率 | 配对差95%区间 | 基础PPO FED | LLM+PPO FED | FED配对差95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stable_incap = 0
    stable_fed = 0
    for behavior, label in behavior_labels.items():
        base_incap = aggregate_lookup[(behavior, "ppo", "incapacitation_rate")]
        guided_incap = aggregate_lookup[(behavior, "llm_guided_ppo", "incapacitation_rate")]
        diff_incap = paired_lookup[(behavior, "incapacitation_rate")]
        base_fed = aggregate_lookup[(behavior, "ppo", "mean_cumulative_fed")]
        guided_fed = aggregate_lookup[(behavior, "llm_guided_ppo", "mean_cumulative_fed")]
        diff_fed = paired_lookup[(behavior, "mean_cumulative_fed")]
        stable_incap += int(float(diff_incap["bootstrap_ci95_upper"]) < 0.0)
        stable_fed += int(float(diff_fed["bootstrap_ci95_upper"]) < 0.0)
        report.append(
            f"| {label} | {float(base_incap['mean']):.2%} ± {float(base_incap['between_seed_std']):.2%} | "
            f"{float(guided_incap['mean']):.2%} ± {float(guided_incap['between_seed_std']):.2%} | "
            f"[{float(diff_incap['bootstrap_ci95_lower']):.2%}, {float(diff_incap['bootstrap_ci95_upper']):.2%}] | "
            f"{float(base_fed['mean']):.4f} | {float(guided_fed['mean']):.4f} | "
            f"[{float(diff_fed['bootstrap_ci95_lower']):.4f}, {float(diff_fed['bootstrap_ci95_upper']):.4f}] |"
        )
    report.extend(
        [
            "",
            "## 稳定性判定",
            "",
            f"- 失能率配对差的95%区间在 {stable_incap}/4 类人群中完全低于0。",
            f"- 人均累计FED配对差的95%区间在 {stable_fed}/4 类人群中完全低于0。",
            "- 区间的重采样单位是5个PPO训练种子；同一模型产生的重复测试回合先汇总，未被当作独立训练证据。",
            f"- LLM指导PPO与LLM静态标志在 {guided_static_exact_match_rate:.2%} 的逐回合结果上完全一致。"
            "5个训练种子都退化为执行固定LLM目标，因此当前证据支持LLM降低训练波动，但不支持PPO学到了额外的动态修正。",
            "",
            "## 解释边界",
            "",
            "- 所有策略仍使用快速合成火灾模型；下一项外部有效性工作是接入FDS导出的时间序列。",
            "- 校准Logit来自合成演示数据，LLM代理行为也未经真人数据验证；跨行为结果目前表示软件层面的分布偏移压力测试。",
            "- LLM高层建议在这些场景中等价于安全方向预测。更复杂场景还需要加入出口阻塞、距离差和预测误差。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stable incapacity improvement: {stable_incap}/4 behaviors")
    print(f"Stable FED improvement: {stable_fed}/4 behaviors")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
