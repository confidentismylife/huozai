from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import EvacuationPlan, rule_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_domain_randomized_trust_bank
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from fire_evac.mappo import (
    FLOW_LEVELS,
    FactorizedMAPPO,
    MAPPOConfig,
    encode_factorized_action,
    plan_supervision_targets,
)


FORMAT_GATE = ROOT / "outputs" / "deepseek_format_gate_20261021"
DEFAULT_OUTPUT = ROOT / "outputs" / "mappo_capacity_stabilization_20261022"
ARMS = ("mappo_34_raw", "mappo_50_rule_plan", "mappo_50_deepseek_retrieval")
SCREEN_CONFIGS = (
    ("e001_i000", 0.01, 0.00),
    ("e003_i000", 0.03, 0.00),
    ("e006_i000", 0.06, 0.00),
    ("e010_i000", 0.10, 0.00),
    ("e003_i002", 0.03, 0.02),
    ("e003_i005", 0.03, 0.05),
    ("e006_i002", 0.06, 0.02),
    ("e006_i005", 0.06, 0.05),
)
SCREEN_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
CONFIRMATION_SEED_START = 20261201


class MaskPlanFeatures(gym.ObservationWrapper):
    def observation(self, observation):
        masked = np.asarray(observation, dtype=np.float32).copy()
        masked[28:49] = 0.0
        return masked


class ShuffledPlanProvider:
    """Deterministically substitute another frozen plan with the same event type."""

    def __init__(self, library: FrozenPlanLibrary):
        self.library = library

    def __call__(self, state: Mapping[str, Any]) -> EvacuationPlan:
        event = str(state.get("event_type", "initial"))
        candidates = [
            index for index, (stored, _) in enumerate(self.library.records)
            if str(stored.get("event_type")) == event
        ]
        if not candidates:
            candidates = list(range(len(self.library.records)))
        _ = self.library(state)
        matched = str((self.library.last_match or {}).get("matched_scenario_id", ""))
        winner_position = next(
            (position for position, index in enumerate(candidates)
             if str(self.library.records[index][0]["scenario_id"]) == matched),
            0,
        )
        return self.library.records[candidates[(winner_position + 1) % len(candidates)]][1]


def new_library() -> FrozenPlanLibrary:
    return FrozenPlanLibrary.from_files(
        FORMAT_GATE / "frozen_development_states.json",
        FORMAT_GATE / "response_audit.json",
    )


def make_env(arm: str, scenarios, provider_variant: str = "native"):
    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    if arm == "mappo_34_raw":
        return base
    if provider_variant == "rule" or arm == "mappo_50_rule_plan":
        provider = rule_plan
    elif provider_variant == "shuffled":
        provider = ShuffledPlanProvider(new_library())
    else:
        provider = new_library()
    wrapped = DeepSeekPlanObservationWrapper(base, provider, max_requests_per_episode=3)
    return MaskPlanFeatures(wrapped) if provider_variant == "masked" else wrapped


def evaluate(model, env, scenarios, population_seeds, arm: str, provider_variant: str = "native"):
    rows = []
    for scenario in scenarios:
        for population_seed in population_seeds:
            observation, _ = env.reset(seed=population_seed, options={"scenario": scenario})
            actions: list[int] = []
            releases: list[float] = []
            waits = 0
            route_matches: list[float] = []
            flow_matches: list[float] = []
            while True:
                routes, flows, _, _ = model.act(observation, deterministic=True)
                route, flow = int(routes[0]), int(flows[0])
                action = encode_factorized_action(route, flow)
                actions.append(action)
                if action == 8:
                    waits += 1
                else:
                    releases.append(float(FLOW_LEVELS[flow]))
                if len(observation) >= 50 and provider_variant != "masked":
                    tensor = torch.as_tensor(observation[None, :], dtype=torch.float32)
                    target_route, target_flow, route_weight, flow_weight = plan_supervision_targets(tensor)
                    if float(route_weight[0]) > 0:
                        route_matches.append(float(route == int(target_route[0])))
                    if float(flow_weight[0]) > 0 and route != 2:
                        flow_matches.append(float(flow == int(target_flow[0])))
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            rows.append(
                {
                    "arm": arm,
                    "provider_variant": provider_variant,
                    "scenario_id": scenario.scenario_id,
                    "conflict_type": scenario.conflict_type,
                    "population_seed": population_seed,
                    "incapacitation_rate": float(info["incapacitation_rate"]),
                    "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
                    "cumulative_congestion": float(info["cumulative_congestion"]),
                    "evacuation_time_seconds": 5.0 * float(info["step"]),
                    "wait_fraction": waits / max(1, len(actions)),
                    "mean_release_fraction_nonwait": float(np.mean(releases)) if releases else 0.0,
                    "full_release_episode": int(bool(releases) and min(releases) >= 0.999),
                    "route_plan_match": float(np.mean(route_matches)) if route_matches else np.nan,
                    "flow_plan_match": float(np.mean(flow_matches)) if flow_matches else np.nan,
                    "plan_requests": int(info.get("deepseek_requests_this_episode", 0)),
                }
            )
    return rows


def summarize(rows):
    def finite_mean(key: str) -> float:
        values = np.asarray([r[key] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if len(finite) else 0.0

    return {
        "episodes": len(rows),
        "mean_incapacitation_rate": float(np.mean([r["incapacitation_rate"] for r in rows])),
        "mean_cumulative_fed": float(np.mean([r["mean_cumulative_fed"] for r in rows])),
        "mean_evacuation_time_seconds": float(np.mean([r["evacuation_time_seconds"] for r in rows])),
        "mean_release_fraction_nonwait": float(np.mean([r["mean_release_fraction_nonwait"] for r in rows])),
        "full_release_episode_fraction": float(np.mean([r["full_release_episode"] for r in rows])),
        "mean_route_plan_match": finite_mean("route_plan_match"),
        "mean_flow_plan_match": finite_mean("flow_plan_match"),
    }


def checkpoint_key(summary: dict[str, float]):
    saturated = summary["full_release_episode_fraction"] > 0.85
    return (
        int(saturated),
        summary["mean_incapacitation_rate"],
        summary["mean_cumulative_fed"],
        summary["mean_evacuation_time_seconds"],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_run(
    output: Path,
    arm: str,
    seed: int,
    config: MAPPOConfig,
    train_scenarios,
    validation_scenarios,
    timesteps: int,
    checkpoint_interval: int,
):
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(arm, train_scenarios)
    model = FactorizedMAPPO(env.observation_space.shape[0], config=config, seed=seed)
    checkpoints: list[dict[str, Any]] = []
    best_key = None
    next_checkpoint = checkpoint_interval

    def callback(current_model, step, record):
        nonlocal next_checkpoint, best_key
        if step < next_checkpoint and step < timesteps:
            return
        validation_env = make_env(arm, validation_scenarios)
        rows = evaluate(current_model, validation_env, validation_scenarios, (42001,), arm)
        validation_env.close()
        validation = summarize(rows)
        checkpoint = {"timesteps": step, "validation": validation, "training": dict(record)}
        checkpoints.append(checkpoint)
        current_model.save(output / f"checkpoint_{step}.pt", metadata=checkpoint)
        key = checkpoint_key(validation)
        if best_key is None or key < best_key:
            best_key = key
            current_model.save(output / "best.pt", metadata=checkpoint)
        (output / "checkpoints.json").write_text(
            json.dumps(checkpoints, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        next_checkpoint += checkpoint_interval

    training = model.train(
        env,
        total_timesteps=timesteps,
        rollout_steps=256,
        rollout_callback=callback,
    )
    env.close()
    best_model, best_metadata = FactorizedMAPPO.load(output / "best.pt")
    validation_env = make_env(arm, validation_scenarios)
    final_rows = evaluate(
        best_model, validation_env, validation_scenarios, (42001, 42002), arm
    )
    validation_env.close()
    write_csv(output / "episode_results.csv", final_rows)
    payload = {
        "status": "complete",
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests": 0,
        "arm": arm,
        "seed": seed,
        "timesteps": timesteps,
        "config": asdict(config),
        "training": training,
        "selected_checkpoint": best_metadata,
        "validation": summarize(final_rows),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def select_screen_config(screen_summaries: list[dict[str, Any]]):
    rows = []
    for config_id, _, _ in SCREEN_CONFIGS:
        group = [s for s in screen_summaries if s["config_id"] == config_id]
        rows.append(
            {
                "config_id": config_id,
                "mean_incapacitation_rate": float(np.mean([s["validation"]["mean_incapacitation_rate"] for s in group])),
                "mean_full_release_fraction": float(np.mean([s["validation"]["full_release_episode_fraction"] for s in group])),
                "mean_flow_plan_match": float(np.mean([s["validation"]["mean_flow_plan_match"] for s in group])),
                "mean_route_plan_match": float(np.mean([s["validation"]["mean_route_plan_match"] for s in group])),
            }
        )
    eligible = [row for row in rows if row["mean_full_release_fraction"] <= 0.85]
    pool = eligible or rows
    selected = min(
        pool,
        key=lambda row: (
            row["mean_incapacitation_rate"],
            -row["mean_flow_plan_match"],
            -row["mean_route_plan_match"],
        ),
    )
    return selected, rows


def update_status(path: Path, **values):
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-hour MAPPO capacity stabilization campaign")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "campaign_status.json"
    started = time.monotonic()
    deadline = started + 3600.0 * args.hours
    all_scenarios = make_domain_randomized_trust_bank(24, 20261021, "DSG")
    train_scenarios = all_scenarios[:12]
    validation_scenarios = all_scenarios[12:]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "final_test_scenarios_loaded": False,
        "deepseek_api_requests_allowed": 0,
        "screen_configs": [
            {"config_id": name, "flow_entropy": entropy, "plan_imitation": imitation}
            for name, entropy, imitation in SCREEN_CONFIGS
        ],
        "screen_seeds": list(SCREEN_SEEDS),
        "screen_timesteps": 100_000,
        "confirmation_timesteps": 150_000,
        "checkpoint_interval": 20_000,
        "selection_rule": (
            "exclude mean full-release fraction >0.85 when possible; then minimize mean incapacitation; "
            "break ties by higher flow and route plan match"
        ),
        "confirmation_arms": list(ARMS),
        "channel_variants": ["native", "rule", "shuffled", "masked"],
    }
    manifest_path = args.output / "preregistered_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_status(
        status_path,
        status="running",
        phase="screen",
        pid=None,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        requested_hours=args.hours,
        final_test_scenarios_loaded=False,
        deepseek_api_requests=0,
    )

    screen_summaries = []
    for config_id, flow_entropy, imitation in SCREEN_CONFIGS:
        for seed in SCREEN_SEEDS:
            config = MAPPOConfig(
                hidden_size=64,
                update_epochs=2,
                batch_size=64,
                route_entropy_coefficient=0.01,
                flow_entropy_coefficient=flow_entropy,
                plan_imitation_coefficient=imitation,
            )
            result = train_run(
                args.output / "screen" / config_id / f"seed_{seed}",
                "mappo_50_deepseek_retrieval",
                seed,
                config,
                train_scenarios,
                validation_scenarios,
                100_000,
                20_000,
            )
            result["config_id"] = config_id
            screen_summaries.append(result)
            update_status(status_path, phase="screen", last_completed=f"{config_id}/seed_{seed}")

    selected, screen_table = select_screen_config(screen_summaries)
    selected_spec = next(item for item in SCREEN_CONFIGS if item[0] == selected["config_id"])
    selection = {"selected": selected, "all_configs": screen_table}
    (args.output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_status(status_path, phase="confirmation", selected_config=selected["config_id"])

    _, selected_entropy, selected_imitation = selected_spec
    config = MAPPOConfig(
        hidden_size=64,
        update_epochs=2,
        batch_size=64,
        route_entropy_coefficient=0.01,
        flow_entropy_coefficient=selected_entropy,
        plan_imitation_coefficient=selected_imitation,
    )
    confirmation_summaries = []
    confirmation_seed = CONFIRMATION_SEED_START
    minimum_confirmation_seeds = 5
    while confirmation_seed < CONFIRMATION_SEED_START + minimum_confirmation_seeds or time.monotonic() < deadline - 900:
        for arm in ARMS:
            result = train_run(
                args.output / "confirmation" / f"seed_{confirmation_seed}" / arm,
                arm,
                confirmation_seed,
                config,
                train_scenarios,
                validation_scenarios,
                150_000,
                30_000,
            )
            confirmation_summaries.append(result)
            update_status(
                status_path,
                phase="confirmation",
                last_completed=f"seed_{confirmation_seed}/{arm}",
            )
        confirmation_seed += 1

    update_status(status_path, phase="channel_audit")
    audit_rows = []
    completed_seeds = sorted(
        int(path.name.split("_")[-1])
        for path in (args.output / "confirmation").glob("seed_*")
        if path.is_dir()
    )
    for seed in completed_seeds:
        checkpoint = args.output / "confirmation" / f"seed_{seed}" / "mappo_50_deepseek_retrieval" / "best.pt"
        if not checkpoint.exists():
            continue
        model, _ = FactorizedMAPPO.load(checkpoint)
        for variant in ("native", "rule", "shuffled", "masked"):
            env = make_env("mappo_50_deepseek_retrieval", validation_scenarios, variant)
            rows = evaluate(
                model,
                env,
                validation_scenarios,
                (42001, 42002),
                "mappo_50_deepseek_retrieval",
                variant,
            )
            env.close()
            for row in rows:
                row["training_seed"] = seed
            audit_rows.extend(rows)
    if audit_rows:
        write_csv(args.output / "channel_audit.csv", audit_rows)

    update_status(
        status_path,
        status="complete",
        phase="complete",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_hours=(time.monotonic() - started) / 3600.0,
        confirmation_seeds=completed_seeds,
        final_test_scenarios_loaded=False,
        deepseek_api_requests=0,
    )
    print(json.dumps(json.loads(status_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
