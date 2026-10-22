from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import AdviceBudget, DeepSeekAdviceClient, fallback_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_domain_randomized_trust_bank
from fire_evac.mappo import FactorizedMAPPO, MAPPOConfig, encode_factorized_action


def make_env(scenarios, provider):
    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    return DeepSeekPlanObservationWrapper(
        base,
        provider,
        max_requests_per_episode=3,
        congestion_threshold=45 / 240,
    )


def evaluate(model: FactorizedMAPPO, env, episodes: int, seed: int) -> dict[str, float]:
    records = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + 10_000 + episode)
        reward_total = 0.0
        while True:
            routes, flows, _, _ = model.act(observation, deterministic=True)
            action = encode_factorized_action(int(routes[0]), int(flows[0]))
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            if terminated or truncated:
                break
        records.append(
            {
                "reward": reward_total,
                "incapacitation_rate": float(info["incapacitation_rate"]),
                "evacuation_rate": float(info["evacuation_rate"]),
                "steps": float(info["step"]),
                "deepseek_requests": float(info["deepseek_requests_this_episode"]),
            }
        )
    return {
        f"mean_{name}": float(np.mean([record[name] for record in records]))
        for name in records[0]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test factorized MAPPO with cached DeepSeek plans")
    parser.add_argument("--source", choices=("rule", "deepseek"), default="rule")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--evaluation-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--max-api-requests", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "mappo_deepseek_smoke_20261021",
    )
    args = parser.parse_args()
    if args.timesteps < 1 or args.rollout_steps < 1 or args.evaluation_episodes < 1:
        raise ValueError("timesteps, rollout steps and evaluation episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    scenarios = make_domain_randomized_trust_bank(12, args.seed, "MAPPO")
    client = None
    if args.source == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is not configured; no online requests were made")
        client = DeepSeekAdviceClient(
            args.output / "deepseek_cache",
            budget=AdviceBudget(
                max_requests=args.max_api_requests,
                max_input_tokens=10_000,
                max_output_tokens=max(160, args.max_api_requests * 160),
                max_estimated_cny=0.10,
            ),
            global_budget_path=ROOT / "outputs" / "deepseek_global_budget.json",
            global_max_cny=1.0,
        )
        provider = client.advise
    else:
        provider = fallback_plan

    env = make_env(scenarios, provider)
    config = MAPPOConfig(hidden_size=64, update_epochs=2, batch_size=64)
    model = FactorizedMAPPO(env.observation_space.shape[0], config=config, seed=args.seed)
    training = model.train(env, total_timesteps=args.timesteps, rollout_steps=args.rollout_steps)
    evaluation = evaluate(model, env, args.evaluation_episodes, args.seed)
    model_path = model.save(
        args.output / "mappo_smoke.pt",
        metadata={"source": args.source, "timesteps": args.timesteps, "observation_dim": 50},
    )
    env.close()
    if client is not None:
        client.write_ledger(args.output / "deepseek_usage.json")
    summary = {
        "status": "smoke_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "heterogeneous_factorized_action_MAPPO",
        "actors": ["route_north_south_wait", "release_025_050_075_100"],
        "centralized_critic": True,
        "observation_dim": 50,
        "original_joint_action_count": 9,
        "advice_source": args.source,
        "deepseek_model": None if client is None else client.model,
        "training_config": asdict(config),
        "training": training,
        "evaluation": evaluation,
        "checkpoint": str(model_path.resolve()),
        "api_requests": 0 if client is None else client.ledger.requests,
        "estimated_api_cost_cny": 0.0 if client is None else client.ledger.estimated_cny,
        "claim_scope": "engineering smoke test only; no comparative effectiveness claim",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
