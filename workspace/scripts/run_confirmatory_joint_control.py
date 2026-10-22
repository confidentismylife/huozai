from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import (
    CalibratedLogitBehavior,
    EpisodeBehaviorEnsemble,
    MixedLogitBehavior,
    RuleBehavior,
)
from fire_evac.confirmatory import (
    NOISE_LEVELS,
    ConfirmatoryJointControlWrapper,
)
from fire_evac.dynamic_conflicts import (
    DynamicConflictScenario,
    GymDynamicConflictEnv,
    conflict_scenario_records,
    make_domain_randomized_trust_bank,
)
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice, NORTH, SOUTH, parse_dynamic_advice


ARMS = ("pure_ppo", "rule_advice_ppo", "llm_advice_ppo")
ARM_MODES = {"pure_ppo": "none", "rule_advice_ppo": "rule", "llm_advice_ppo": "llm"}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_exit": {"type": "string", "enum": ["north", "south"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "string",
                "enum": [
                    "avoid_predicted_smoke",
                    "early_guidance",
                    "reduce_exposure",
                    "balance_congestion",
                ],
            },
        },
    },
    "required": ["target_exit", "confidence", "reason_codes"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen fair comparison for nine-action joint control.")
    parser.add_argument("--seeds", default="501,502,503,504,505")
    parser.add_argument("--train-scenarios", type=int, default=120)
    parser.add_argument("--final-scenarios", type=int, default=60)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=5)
    parser.add_argument("--scenario-seed", type=int, default=20261117)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "confirmatory_joint_control"
    )
    return parser.parse_args()


def api_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) < 5 or len(seeds) != len(set(seeds)):
        raise ValueError("confirmatory evaluation requires at least five unique training seeds")
    return seeds


def advice_prompt(scenario: DynamicConflictScenario) -> str:
    source = "north" if scenario.forecast_source_branch == NORTH else "south"
    blocked = (
        "none"
        if scenario.blocked_branch not in (NORTH, SOUTH)
        else "north" if scenario.blocked_branch == NORTH else "south"
    )
    return (
        "You are a high-level advisor for a two-exit fire evacuation. Choose one initial target exit. "
        "The lower-level controller sees live sensors and may reject your advice. Minimize smoke exposure, "
        "incapacitation, and severe congestion. A forecast fire-source corridor is where smoke is predicted "
        "to originate. Use only the structured forecast below and return JSON matching the schema.\n"
        f"forecast_fire_source={source}\n"
        f"smoke_growth_rate={scenario.growth_rate:.6f}\n"
        f"smoke_onset_step={scenario.onset_step}\n"
        f"north_exit_capacity={scenario.north_exit_capacity}\n"
        f"south_exit_capacity={scenario.south_exit_capacity}\n"
        f"forecast_blocked_exit={blocked}\n"
        f"forecast_reopen_step={scenario.blockage_end_step}\n"
    )


def collect_llm_advice(
    scenarios: tuple[DynamicConflictScenario, ...],
    output: Path,
    base_url: str,
    model: str,
    timeout: int,
    seed: int,
) -> tuple[dict[str, DynamicLLMAdvice], str]:
    tags = api_json(f"{base_url}/api/tags", timeout=10)
    matching = [item for item in tags.get("models", []) if item.get("name") == model]
    if not matching:
        raise RuntimeError(f"Ollama model {model!r} is not installed")
    digest = str(matching[0].get("digest", ""))
    cache_dir = output / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    advice: dict[str, DynamicLLMAdvice] = {}
    for index, scenario in enumerate(scenarios, 1):
        prompt = advice_prompt(scenario)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{scenario.scenario_id}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if cached and cached.get("model_digest") == digest and cached.get("prompt_hash") == prompt_hash:
            advice[scenario.scenario_id] = parse_dynamic_advice(cached["parsed"])
            continue
        response = api_json(
            f"{base_url}/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": SCHEMA,
                "options": {"temperature": 0, "seed": seed, "num_predict": 96},
                "think": False,
                "stream": False,
            },
            timeout=timeout,
        )
        raw = str(response["message"]["content"])
        parsed_payload = json.loads(raw)
        advice[scenario.scenario_id] = parse_dynamic_advice(parsed_payload)
        cache_path.write_text(
            json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "model": model,
                    "model_digest": digest,
                    "prompt_version": "confirmatory_v1",
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed": parsed_payload,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"LLM advice {index}/{len(scenarios)}: {scenario.scenario_id}", flush=True)
    return advice, digest


def behavior_factory(name: str, artifact: Path):
    if name == "calibrated_logit":
        return lambda: CalibratedLogitBehavior(artifact)
    if name == "mixed_logit":
        return MixedLogitBehavior
    if name == "rule":
        return RuleBehavior
    if name == "episode_ensemble":
        return lambda: EpisodeBehaviorEnsemble(
            (lambda: CalibratedLogitBehavior(artifact), MixedLogitBehavior, RuleBehavior)
        )
    raise ValueError(f"unknown behavior model: {name}")


def make_env(
    behavior_name: str,
    artifact: Path,
    scenarios: tuple[DynamicConflictScenario, ...],
    arm: str,
    advice: dict[str, DynamicLLMAdvice],
    noise_name: str = "none",
):
    return ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(behavior_factory(behavior_name, artifact)(), scenarios),
        advice_mode=ARM_MODES[arm],
        advice_by_scenario=advice,
        noise=NOISE_LEVELS[noise_name],
    )


def rollout(env, model, scenario, population_seed: int) -> dict[str, object]:
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    total_reward = 0.0
    request_match = actual_match = hold_blocks = steps = 0
    requested_release = actual_release = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        total_reward += float(reward)
        request_match += int(info["requested_joint_action"] == info["actual_joint_action"])
        actual_match += int(info["requested_sign_action"] == info["actual_sign_action"])
        hold_blocks += int(bool(info["action_blocked_by_sign_hold"]))
        requested_release += float(info["requested_release_fraction"])
        actual_release += float(info["actual_release_fraction"])
        steps += 1
    return {
        "scenario_id": scenario.scenario_id,
        "conflict_type": scenario.conflict_type,
        "population_seed": population_seed,
        "steps": steps,
        "requested_actual_joint_match_rate": request_match / steps,
        "requested_actual_sign_match_rate": actual_match / steps,
        "sign_hold_block_rate": hold_blocks / steps,
        "requested_mean_release_fraction": requested_release / steps,
        "actual_mean_release_fraction": actual_release / steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": total_reward,
    }


def summarize_by_seed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "evacuation_rate",
        "incapacitation_rate",
        "mean_cumulative_fed",
        "cumulative_congestion",
        "total_reward",
        "sign_hold_block_rate",
        "actual_mean_release_fraction",
    )
    groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["arm"],
            int(row["training_seed"]),
            row["test_behavior"],
            row["noise_level"],
            row["conflict_type"],
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        record = dict(zip(("arm", "training_seed", "test_behavior", "noise_level", "conflict_type"), key))
        record["episodes"] = len(items)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in items])
            record[f"{metric}_mean"] = float(values.mean())
        result.append(record)
    return result


def paired_seed_effects(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ("incapacitation_rate_mean", "mean_cumulative_fed_mean")
    index = {
        (r["arm"], r["training_seed"], r["test_behavior"], r["noise_level"], r["conflict_type"]): r
        for r in seed_rows
    }
    output = []
    for candidate in ("rule_advice_ppo", "llm_advice_ppo"):
        for baseline in ("pure_ppo", "rule_advice_ppo"):
            if candidate == baseline or (candidate == "rule_advice_ppo" and baseline == "rule_advice_ppo"):
                continue
            for behavior in sorted({str(r["test_behavior"]) for r in seed_rows}):
                for noise in sorted({str(r["noise_level"]) for r in seed_rows}):
                    for conflict in sorted({str(r["conflict_type"]) for r in seed_rows}):
                        seeds = sorted(
                            {
                                int(r["training_seed"])
                                for r in seed_rows
                                if r["arm"] == candidate
                                and r["test_behavior"] == behavior
                                and r["noise_level"] == noise
                                and r["conflict_type"] == conflict
                            }
                        )
                        for metric in metrics:
                            differences = []
                            for seed in seeds:
                                a = index.get((candidate, seed, behavior, noise, conflict))
                                b = index.get((baseline, seed, behavior, noise, conflict))
                                if a is not None and b is not None:
                                    differences.append(float(a[metric]) - float(b[metric]))
                            if differences:
                                values = np.asarray(differences)
                                rng = np.random.default_rng(91_731 + len(output))
                                bootstrap = np.asarray(
                                    [rng.choice(values, size=len(values), replace=True).mean() for _ in range(10_000)]
                                )
                                output.append(
                                    {
                                        "candidate": candidate,
                                        "baseline": baseline,
                                        "test_behavior": behavior,
                                        "noise_level": noise,
                                        "conflict_type": conflict,
                                        "metric": metric.removesuffix("_mean"),
                                        "training_seeds": len(values),
                                        "mean_seed_difference": float(values.mean()),
                                        "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                                        "seed_bootstrap_ci_2_5": float(np.quantile(bootstrap, 0.025)),
                                        "seed_bootstrap_ci_97_5": float(np.quantile(bootstrap, 0.975)),
                                        "min_seed_difference": float(values.min()),
                                        "max_seed_difference": float(values.max()),
                                    }
                                )
    return output


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if not args.calibration_artifact.exists():
        raise FileNotFoundError(args.calibration_artifact)
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)

    train_scenarios = make_domain_randomized_trust_bank(
        args.train_scenarios, args.scenario_seed, "CTR"
    )
    final_scenarios = make_domain_randomized_trust_bank(
        args.final_scenarios, args.scenario_seed + 1, "CFE"
    )
    (args.output / "train_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(train_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "final_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(final_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    advice, model_digest = collect_llm_advice(
        train_scenarios + final_scenarios,
        args.output,
        args.base_url,
        args.model,
        args.timeout,
        args.scenario_seed,
    )
    advice_rows = []
    for split, scenarios in (("train", train_scenarios), ("final", final_scenarios)):
        for scenario in scenarios:
            item = advice[scenario.scenario_id]
            advice_rows.append(
                {
                    "split": split,
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "target_exit": "north" if item.target_branch == NORTH else "south",
                    "confidence": item.confidence,
                    "reason_codes": "|".join(item.reason_codes),
                }
            )
    write_csv(advice_rows, args.output / "llm_advice_audit.csv")
    for arm in ARMS:
        check_env(make_env("calibrated_logit", args.calibration_artifact, train_scenarios, arm, advice), warn=True)

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
    models: dict[tuple[int, str], object] = {}
    training_rows = []
    for seed in seeds:
        for arm in ARMS:
            path = model_dir / f"seed_{seed}_{arm}.zip"
            if args.reuse_models and path.exists():
                model = PPO.load(path, device="cpu")
            else:
                print(f"Training seed={seed}, arm={arm}...", flush=True)

                def factory(arm_name=arm):
                    return make_env(
                        "calibrated_logit", args.calibration_artifact, train_scenarios, arm_name, advice
                    )

                vector_env = make_vec_env(factory, n_envs=args.n_envs, seed=seed)
                model = PPO(env=vector_env, seed=seed, **common)
                model.learn(total_timesteps=args.timesteps, progress_bar=False)
                model.save(path.with_suffix(""))
                vector_env.close()
            models[(seed, arm)] = model
            training_rows.append(
                {
                    "training_seed": seed,
                    "arm": arm,
                    "requested_timesteps": args.timesteps,
                    "actual_timesteps": int(model.num_timesteps),
                    "model_path": str(path.resolve()),
                }
            )
    write_csv(training_rows, args.output / "training_summary.csv")

    factors = [("calibrated_logit", "none")]
    if not args.skip_robustness:
        factors.extend(
            (behavior, noise)
            for behavior in ("calibrated_logit", "mixed_logit", "rule", "episode_ensemble")
            for noise in NOISE_LEVELS
            if (behavior, noise) != ("calibrated_logit", "none")
        )
    episode_rows: list[dict[str, object]] = []
    checkpoint_path = args.output / "final_episode_results.csv"
    if args.reuse_models and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
            episode_rows = list(csv.DictReader(handle))
        print(f"Loaded {len(episode_rows)} checkpointed evaluation episodes.", flush=True)
    expected_factor_rows = len(final_scenarios) * args.eval_repetitions * len(seeds) * len(ARMS)
    for behavior, noise in factors:
        completed_rows = sum(
            1
            for row in episode_rows
            if row["test_behavior"] == behavior and row["noise_level"] == noise
        )
        if completed_rows == expected_factor_rows:
            print(f"Skipping completed factor: behavior={behavior}, noise={noise}", flush=True)
            continue
        if completed_rows:
            # A factor is atomic: discard a partial checkpoint and rerun all paired episodes.
            episode_rows = [
                row
                for row in episode_rows
                if not (row["test_behavior"] == behavior and row["noise_level"] == noise)
            ]
        print(f"Final evaluation: behavior={behavior}, noise={noise}", flush=True)
        envs = {
            arm: make_env(behavior, args.calibration_artifact, final_scenarios, arm, advice, noise)
            for arm in ARMS
        }
        for scenario_index, scenario in enumerate(final_scenarios):
            for repetition in range(args.eval_repetitions):
                population_seed = 800_000 + scenario_index * args.eval_repetitions + repetition
                for seed in seeds:
                    for arm in ARMS:
                        episode_rows.append(
                            {
                                "arm": arm,
                                "training_seed": seed,
                                "test_behavior": behavior,
                                "noise_level": noise,
                                **rollout(envs[arm], models[(seed, arm)], scenario, population_seed),
                            }
                        )
        for env in envs.values():
            env.close()
        write_csv(episode_rows, checkpoint_path)
        print(f"Checkpointed factor with {expected_factor_rows} paired episodes.", flush=True)
    write_csv(episode_rows, checkpoint_path)
    per_seed = summarize_by_seed(episode_rows)
    # Add the pre-registered all-conflict aggregation without treating episodes as independent seeds.
    all_conflict_rows = [{**row, "conflict_type": "all"} for row in episode_rows]
    per_seed.extend(summarize_by_seed(all_conflict_rows))
    write_csv(per_seed, args.output / "final_per_seed_summary.csv")
    effects = paired_seed_effects(per_seed)
    write_csv(effects, args.output / "paired_training_seed_effects.csv")
    metadata = {
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_bank": "outputs/joint_route_release/test_scenarios.json (renamed development)",
        "development_bank_used_for_final": False,
        "train_scenario_seed": args.scenario_seed,
        "final_scenario_seed": args.scenario_seed + 1,
        "training_seeds": seeds,
        "arms": ARMS,
        "action_space": "north/south x 25/50/75/100 percent release + wait",
        "observation_dimensions": 34,
        "requested_timesteps_per_seed_arm": args.timesteps,
        "ollama_model": args.model,
        "ollama_model_digest": model_digest,
        "prompt_version": "confirmatory_v1",
        "behavior_validity_warning": "calibrated_logit artifact was fitted to synthetic demonstration data",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed. Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
