"""Freeze training-side congestion states and acquire a bounded DeepSeek plan bank."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mappo_six_hour_extension as previous
from fire_evac.behavior import MixedLogitBehavior
from fire_evac.campaign_extension import BudgetedPlanner, write_json
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import parse_plan, rule_plan
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv
from fire_evac.event_plan_v3 import EventCorrectedPlanWrapper
from fire_evac.mappo import FactorizedMAPPO, encode_factorized_action
from fire_evac.process_evidence import select_training_event_states


OUT = ROOT / "outputs" / "congestion_plan_coverage_20261022"
PROTOCOL = ROOT / "docs" / "congestion_plan_coverage_protocol_20261022.md"
STATE_PATH = OUT / "frozen_congestion_states.json"
MANIFEST_PATH = OUT / "state_freeze_manifest.json"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def checkpoint() -> Path:
    return (
        previous.OUT / "mechanism" / "e003_i002" / "seed_20261301"
        / previous.RULE / "best.pt"
    )


def collect_states() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists() or MANIFEST_PATH.exists():
        raise RuntimeError("Frozen congestion states already exist; refusing to overwrite.")
    train = previous.base.make_domain_randomized_trust_bank(24, 20261021, "DSG")[:12]
    config = EnvConfig(junction_to_branch_capacity=3)
    model, _ = FactorizedMAPPO.load(checkpoint())
    observed: list[dict] = []
    physical = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), train, config=config), advice_mode="none"
    )

    def recorder(state):
        if state["event_type"] == "congestion_threshold":
            observed.append(dict(state))
        return rule_plan(state)

    env = EventCorrectedPlanWrapper(physical, recorder, max_requests_per_episode=3)
    for population_seed in (45001, 45002, 45003):
        for scenario in train:
            observation, _ = env.reset(seed=population_seed, options={"scenario": scenario})
            while True:
                routes, flows, _, _ = model.act(observation, deterministic=True)
                action = encode_factorized_action(int(routes[0]), int(flows[0]))
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    env.close()
    selected = select_training_event_states(observed, maximum=12)
    if len(selected) < 8:
        raise RuntimeError(f"Only {len(selected)} distinct training congestion states; acquisition stopped.")
    frozen = []
    for state in selected:
        original = str(state["scenario_id"])
        payload = dict(state)
        payload["original_scenario_id"] = original
        payload["semantic_context"] = {
            "control_period_seconds": 5,
            "observation_source": "training-only actual simulator congestion crossing",
            "stress_condition": "artificial junction-to-branch capacity 3 people per side per step",
            "future_outcomes_supplied": False,
        }
        provisional = {**payload, "scenario_id": "pending"}
        state_id = "CONGEST_" + digest_bytes(canonical_json(provisional))[:16]
        payload["scenario_id"] = state_id
        frozen.append(payload)
    STATE_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_count": len(frozen),
        "states_sha256": digest_bytes(STATE_PATH.read_bytes()),
        "event_counts": {"congestion_threshold": len(frozen)},
        "original_scenario_ids": [state["original_scenario_id"] for state in frozen],
        "selection": "first congestion crossing per training scenario, sorted by scenario id and time",
        "population_seeds_observed": [45001, 45002, 45003],
        "source_checkpoint": str(checkpoint()),
        "source_checkpoint_sha256": digest_bytes(checkpoint().read_bytes()),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "validation_outcomes_used_for_selection": False,
        "final_test_scenarios_loaded": False,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def acquire_plans() -> None:
    if not STATE_PATH.exists() or not MANIFEST_PATH.exists():
        raise RuntimeError("Run --collect-only before any API acquisition.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if digest_bytes(STATE_PATH.read_bytes()) != manifest["states_sha256"]:
        raise RuntimeError("Frozen congestion state hash changed.")
    audit_path = OUT / "response_audit.json"
    if audit_path.exists():
        raise RuntimeError("Response audit already exists; refusing duplicate paid acquisition.")
    states = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    planner = BudgetedPlanner(OUT / "api", max_cny=0.50, max_requests=len(states), max_tokens=512)
    audits = []
    for state in states:
        result = planner(state)
        payload = asdict(result.plan)
        valid_model_response = result.source in ("api", "cache")
        first = result.plan.phases[0]
        target_capacity = (
            int(state["north_exit_capacity"]) if first.route == "north"
            else int(state["south_exit_capacity"]) if first.route == "south"
            else None
        )
        audits.append({
            "scenario_id": state["scenario_id"],
            "original_scenario_id": state["original_scenario_id"],
            "event_type": state["event_type"],
            "source": result.source,
            "valid_model_response": valid_model_response,
            "unsafe_first_route_closed": target_capacity is not None and target_capacity <= 0,
            "plan": payload,
            "usage": asdict(result.usage),
            "latency_seconds": planner.last_latency_seconds,
        })
    write_json(audit_path, audits)
    valid = sum(item["valid_model_response"] for item in audits)
    unsafe = sum(item["unsafe_first_route_closed"] for item in audits)
    gate = {
        "passed": valid / len(audits) >= 0.95,
        "valid": valid,
        "total": len(audits),
        "valid_rate": valid / len(audits),
        "unsafe_first_route_closed": unsafe,
        "criterion": "schema-valid API/cache plan rate >=95%; no replacement sampling",
        "ledger": planner.ledger(),
        "states_sha256": manifest["states_sha256"],
        "final_test_scenarios_loaded": False,
    }
    write_json(OUT / "gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-only", action="store_true")
    group.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if args.collect_only:
        collect_states()
    else:
        acquire_plans()


if __name__ == "__main__":
    main()
