"""Posthoc context check against existing frozen PPO; not a fair new RL trial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/source_world_planning_development_v1")
    parser.add_argument("--scenario-source", type=Path, default=ROOT / "outputs/terminal_shield_development_v1")
    parser.add_argument("--models", type=Path, default=ROOT / "outputs/guarded_confirmatory_v2/models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/source_world_planning_frozen_ppo_context_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose new output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    manifest = json.loads((args.scenario_source / "development_manifest.json").read_text(encoding="utf-8"))
    scenarios = tuple(DynamicConflictScenario(**s) for s in manifest["scenario_records"][:manifest["original_scenarios"]])
    by_id = {s.scenario_id: s for s in scenarios}
    with (args.source / "evaluation_episode_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        references = [r for r in csv.DictReader(handle) if r["arm"] == "minimax_world_planner" and int(r["step"]) == 0]
    env = ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), scenarios), "none")
    records = []
    summaries = []
    hashes = {}
    for seed in range(601, 606):
        path = args.models / f"seed_{seed}_pure_ppo.zip"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = PPO.load(path, device="cpu")
        rows = []
        for r in references:
            scenario = by_id[r["scenario_id"]]
            observation, info = env.reset(seed=int(r["population_seed"]), options={"scenario": scenario})
            future_seed = int(r["future_seed"])
            env.unwrapped.domain_env.rng = np.random.default_rng(future_seed)
            env.rng = np.random.default_rng(future_seed + 9_000_001)
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, term, trunc, info = env.step(int(np.asarray(action).item()))
                done = term or trunc
                steps += 1
            rows.append({"training_seed": seed, "scenario_id": scenario.scenario_id, "conflict_type": scenario.conflict_type,
                         "population_seed": int(r["population_seed"]), "future_seed": future_seed,
                         "steps": steps, "incapacitation_rate": float(info["incapacitation_rate"]),
                         "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
                         "reference_minimax_incapacitation": float(r["final_incapacitation"]),
                         "ppo_minus_minimax_delta_pp": 100 * (float(info["incapacitation_rate"]) - float(r["final_incapacitation"]))})
        records.extend(rows)
        summaries.append({"training_seed": seed, "episodes": len(rows), "actual_training_steps": int(model.num_timesteps),
                          "incapacitation_rate": float(np.mean([r["incapacitation_rate"] for r in rows])),
                          "mean_cumulative_fed": float(np.mean([r["mean_cumulative_fed"] for r in rows])),
                          "ppo_minus_minimax_delta_pp": float(np.mean([r["ppo_minus_minimax_delta_pp"] for r in rows]))})
        print(f"Frozen PPO context seed {seed}: {summaries[-1]['incapacitation_rate']:.4%}", flush=True)
    env.close()
    with (args.output / "episode_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"evaluation_role": "posthoc_frozen_controller_context_not_fair_training_ablation", "freeze_allowed": False,
               "episodes": len(records), "checkpoint_hashes": hashes, "per_seed": summaries,
               "mean_ppo_incapacitation_rate": float(np.mean([r["incapacitation_rate"] for r in summaries])),
               "mean_ppo_minus_minimax_delta_pp": float(np.mean([r["ppo_minus_minimax_delta_pp"] for r in summaries])),
               "limits": ["PPO trained earlier on a different bank and has 34 inputs; references have 49 plus privileged offline replay",
                          "no new training, no parameter changes", "same initial population and future RNG starts; action-dependent RNG consumption remains",
                          "this does not establish the causal value of advice or any new LLM/RL method"]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
