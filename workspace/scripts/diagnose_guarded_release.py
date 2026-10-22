"""Descriptive paired intervention on preserved round-2 models and scenarios."""
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
from fire_evac.advice_protocol_v2 import safety_validate_advice
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.dynamic_llm_rl import parse_dynamic_advice


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode(env, model, scenario, population_seed, training_seed, treatment, actions=None):
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    done = False
    step = 0
    commands = []
    rows = []
    rewards = 0.0
    while not done:
        if actions is None:
            proposed, _ = model.predict(observation, deterministic=True)
            proposed = int(np.asarray(proposed).item())
        else:
            proposed = actions[min(step, len(actions) - 1)]
        applied = proposed if treatment in ("baseline", "baseline_replay") or proposed == 8 else 4 * (proposed // 4) + 3
        domain = env.unwrapped.domain_env
        room_ids = domain.zones[0] + domain.zones[1]
        room_fed = float(np.mean([domain.people[i].fed for i in room_ids])) if room_ids else 0.0
        room_count = len(room_ids)
        observation, reward, term, trunc, info = env.step(applied)
        rewards += float(reward)
        commands.append(proposed)
        rows.append({"training_seed": training_seed, "scenario_id": scenario.scenario_id,
                     "conflict_type": scenario.conflict_type, "population_seed": population_seed,
                     "treatment": treatment, "step": step, "policy_or_replay_action": proposed,
                     "submitted_action": applied, "actual_joint_action": info["actual_joint_action"],
                     "hold_blocked": info["action_blocked_by_sign_hold"],
                     "room_count_before": room_count, "active_room_mean_fed_before": room_fed,
                     "actual_release_fraction": info["actual_release_fraction"],
                     "incapacitation_rate": info["incapacitation_rate"], "mean_cumulative_fed": info["mean_cumulative_fed"],
                     "evacuation_rate": info["evacuation_rate"], "congestion": info["cumulative_congestion"]})
        step += 1
        done = term or trunc
    summary = {"training_seed": training_seed, "scenario_id": scenario.scenario_id,
               "conflict_type": scenario.conflict_type, "population_seed": population_seed, "treatment": treatment,
               "steps": step, "incapacitation_rate": info["incapacitation_rate"],
               "mean_cumulative_fed": info["mean_cumulative_fed"], "evacuation_rate": info["evacuation_rate"],
               "cumulative_congestion": info["cumulative_congestion"], "total_reward": rewards,
               "mean_actual_release_fraction": float(np.mean([r["actual_release_fraction"] for r in rows])),
               "room_person_steps": sum(r["room_count_before"] for r in rows),
               "hold_block_rate": float(np.mean([r["hold_blocked"] for r in rows])),
               "replay_tail_steps": max(0, step - len(actions)) if actions is not None else 0}
    return summary, commands, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/guarded_confirmatory_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/guarded_seed605_diagnosis_v1")
    parser.add_argument("--scenarios-per-conflict", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new diagnosis output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    scenarios = tuple(DynamicConflictScenario(**v) for v in json.loads((args.source / "final_scenarios.json").read_text(encoding="utf-8")))
    selected = []
    for conflict in sorted({s.conflict_type for s in scenarios}):
        selected.extend(sorted((s for s in scenarios if s.conflict_type == conflict), key=lambda s: s.scenario_id)[:args.scenarios_per_conflict])
    advice = {}
    for s in selected:
        cached = json.loads((args.source / "llm_cache" / f"{s.scenario_id}.json").read_text(encoding="utf-8"))
        raw = parse_dynamic_advice(json.dumps(cached["parsed"]))
        advice[s.scenario_id], _ = safety_validate_advice(s, raw)
    env = ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), selected), "llm", advice)
    # Preserve original population seeds and check baseline reproduces saved results.
    with (args.source / "final_episode_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        originals = {(int(r["training_seed"]), r["scenario_id"], int(r["population_seed"])): r
                     for r in csv.DictReader(handle) if r["arm"] == "guarded_llm_ppo" and r["test_behavior"] == "calibrated_logit" and r["noise_level"] == "none"}
    results = []
    trajectories = []
    source_hashes = {}
    for training_seed in range(601, 606):
        path = args.source / "models" / f"seed_{training_seed}_guarded_llm_ppo.zip"
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = PPO.load(path, device="cpu")
        for scenario in selected:
            seeds = sorted(key[2] for key in originals if key[:2] == (training_seed, scenario.scenario_id))[:args.repetitions]
            for population_seed in seeds:
                baseline, actions, rows = episode(env, model, scenario, population_seed, training_seed, "baseline")
                prior = originals[(training_seed, scenario.scenario_id, population_seed)]
                for metric in ("incapacitation_rate", "mean_cumulative_fed", "evacuation_rate", "steps", "total_reward"):
                    np.testing.assert_allclose(baseline[metric], float(prior[metric]), atol=1e-8, rtol=0)
                replay, _, _ = episode(env, model, scenario, population_seed, training_seed, "baseline_replay", actions)
                for metric in ("incapacitation_rate", "mean_cumulative_fed", "evacuation_rate", "steps", "total_reward"):
                    np.testing.assert_allclose(replay[metric], baseline[metric], atol=1e-8, rtol=0)
                results.append(baseline)
                trajectories.extend(rows)
                for name, commands in (("fixed_route_full_release", actions), ("feedback_full_release", None)):
                    result, _, rows = episode(env, model, scenario, population_seed, training_seed, name, commands)
                    results.append(result)
                    trajectories.extend(rows)
        print(f"Diagnosed seed {training_seed}", flush=True)
    env.close()
    write_csv(results, args.output / "episode_results.csv")
    write_csv(trajectories, args.output / "step_trajectories.csv")
    paired = []
    for seed in range(601, 606):
        for conflict in ("all",) + tuple(sorted({s.conflict_type for s in selected})):
            rows = [r for r in results if r["training_seed"] == seed and (conflict == "all" or r["conflict_type"] == conflict)]
            base = [r for r in rows if r["treatment"] == "baseline"]
            for treatment in ("fixed_route_full_release", "feedback_full_release"):
                modified = [r for r in rows if r["treatment"] == treatment]
                assert [(r["scenario_id"], r["population_seed"]) for r in modified] == [(r["scenario_id"], r["population_seed"]) for r in base]
                differences = np.asarray([m["incapacitation_rate"] - b["incapacitation_rate"] for m, b in zip(modified, base)])
                paired.append({"training_seed": seed, "conflict_type": conflict, "treatment": treatment,
                               "pairs": len(base), "baseline_incapacitation_rate": float(np.mean([r["incapacitation_rate"] for r in base])),
                               "intervention_incapacitation_rate": float(np.mean([r["incapacitation_rate"] for r in modified])),
                               "incapacitation_delta_pp": float(differences.mean() * 100),
                               "improved_pairs": int(np.sum(differences < 0)), "worsened_pairs": int(np.sum(differences > 0)),
                               "fed_delta": float(np.mean([m["mean_cumulative_fed"] - b["mean_cumulative_fed"] for m, b in zip(modified, base)])),
                               "room_person_steps_delta": float(np.mean([m["room_person_steps"] - b["room_person_steps"] for m, b in zip(modified, base)])),
                               "congestion_delta": float(np.mean([m["cumulative_congestion"] - b["cumulative_congestion"] for m, b in zip(modified, base)]))})
    write_csv(paired, args.output / "paired_intervention_summary.csv")
    summary = {"evaluation_role": "posthoc_diagnostic_reuse_of_round2_final_bank_not_new_confirmatory_evidence",
               "selection": f"first {args.scenarios_per_conflict} sorted IDs per conflict, all five saved training seeds, first {args.repetitions} original population seeds",
               "scenarios": [s.scenario_id for s in selected], "recorded_episodes": len(results),
               "baseline_replay_checks": len(results) // 3, "checkpoint_hashes": source_hashes,
               "full_release_intervention": "25/50/75% become 100%; wait preserved; fixed routes replay original requests; after original trajectory ends repeat its last command",
               "randomness_limit": "paired initial population and RNG; interventions can alter later RNG consumption",
               "all_scenario_effects": [r for r in paired if r["conflict_type"] == "all"],
               "no_parameter_or_threshold_changes": True}
    (args.output / "diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["all_scenario_effects"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
