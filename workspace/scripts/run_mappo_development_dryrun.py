from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import rule_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_domain_randomized_trust_bank
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from fire_evac.mappo import FactorizedMAPPO, MAPPOConfig, encode_factorized_action


ARMS = ("mappo_34_raw", "mappo_50_rule_plan", "mappo_50_deepseek_retrieval")
FORMAT_GATE = ROOT / "outputs" / "deepseek_format_gate_20261021"


def make_env(arm, scenarios, library=None):
    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    if arm == "mappo_34_raw":
        return base
    provider = rule_plan if arm == "mappo_50_rule_plan" else library
    if provider is None:
        raise ValueError("DeepSeek retrieval arm requires a frozen plan library")
    return DeepSeekPlanObservationWrapper(base, provider, max_requests_per_episode=3)


def evaluate(model, env, arm, scenarios, population_seeds):
    rows = []
    for scenario in scenarios:
        for population_seed in population_seeds:
            observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
            reward_total = 0.0
            waits = 0
            release_total = 0.0
            nonwait = 0
            while True:
                routes, flows, _, _ = model.act(observation, deterministic=True)
                action = encode_factorized_action(int(routes[0]), int(flows[0]))
                waits += action == 8
                if action != 8:
                    release_total += (0.25, 0.50, 0.75, 1.00)[action % 4]
                    nonwait += 1
                observation, reward, terminated, truncated, info = env.step(action)
                reward_total += float(reward)
                if terminated or truncated:
                    break
            rows.append(
                {
                    "arm": arm,
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "population_seed": population_seed,
                    "reward": reward_total,
                    "incapacitation_rate": float(info["incapacitation_rate"]),
                    "evacuation_rate": float(info["evacuation_rate"]),
                    "remaining_rate": float(info["remaining"]) / 240.0,
                    "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
                    "cumulative_congestion": float(info["cumulative_congestion"]),
                    "evacuation_time_seconds": float(info["step"]) * 5.0,
                    "wait_fraction": waits / max(1, int(info["step"])),
                    "mean_release_fraction_nonwait": release_total / max(1, nonwait),
                    "plan_requests": float(info.get("deepseek_requests_this_episode", 0)),
                }
            )
    return rows


def summarize(rows):
    metrics = (
        "reward", "incapacitation_rate", "evacuation_rate", "mean_cumulative_fed",
        "cumulative_congestion", "evacuation_time_seconds", "wait_fraction",
        "mean_release_fraction_nonwait", "plan_requests",
    )
    output = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        output[arm] = {f"mean_{metric}": float(np.mean([row[metric] for row in subset])) for metric in metrics}
        output[arm]["episodes"] = len(subset)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only MAPPO interface dry-run")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--scenario-seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mappo_development_dryrun_20261021")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library = FrozenPlanLibrary.from_files(
        FORMAT_GATE / "frozen_development_states.json",
        FORMAT_GATE / "response_audit.json",
    )
    scenarios = make_domain_randomized_trust_bank(12, args.scenario_seed, "DSG")
    population_seeds = (31001, 31002)
    config = MAPPOConfig(hidden_size=64, update_epochs=2, batch_size=64)
    all_rows = []
    training = {}
    for arm in ARMS:
        env = make_env(arm, scenarios, library)
        model = FactorizedMAPPO(env.observation_space.shape[0], config=config, seed=args.seed)
        training[arm] = model.train(env, args.timesteps, args.rollout_steps)
        all_rows.extend(evaluate(model, env, arm, scenarios, population_seeds))
        model.save(
            args.output / f"{arm}.pt",
            metadata={"arm": arm, "timesteps": args.timesteps, "development_only": True},
        )
        env.close()
    with (args.output / "episode_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "status": "dryrun_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests": 0,
        "algorithm": "heterogeneous_factorized_action_MAPPO",
        "arms": list(ARMS),
        "training_seed": args.seed,
        "scenario_seed": args.scenario_seed,
        "timesteps_per_arm": args.timesteps,
        "evaluation_scenarios": len(scenarios),
        "evaluation_population_seeds": list(population_seeds),
        "training_config": asdict(config),
        "training": training,
        "evaluation": summarize(all_rows),
        "claim_scope": "pipeline dry-run only; insufficient seeds and training for effectiveness claims",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
