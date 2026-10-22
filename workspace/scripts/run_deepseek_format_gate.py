from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.deepseek_advice import AdviceBudget, DeepSeekAdviceClient
from fire_evac.deepseek_format_gate import build_format_gate_states, plan_safety_violations


DEFAULT_OUT = ROOT / "outputs" / "deepseek_format_gate_20261021"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    states_path = out / "frozen_development_states.json"
    states = build_format_gate_states(12, seed=20261021)
    write_json(states_path, states)
    write_json(
        out / "freeze_manifest.json",
        {
            "protocol": "deepseek_three_phase_plan_format_gate_v1",
            "development_only": True,
            "final_test_scenarios_loaded": False,
            "state_count": len(states),
            "event_types": sorted({state["event_type"] for state in states}),
            "states_sha256": sha256(states_path),
            "maximum_api_requests": 12,
            "maximum_estimated_cost_cny": 0.10,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Prepared {len(states)} frozen development states at {out}")


def run(out: Path) -> bool:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; no online requests were made")
    states_path = out / "frozen_development_states.json"
    manifest_path = out / "freeze_manifest.json"
    if not states_path.exists() or not manifest_path.exists():
        raise RuntimeError("run prepare before the online format gate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(states_path) != manifest["states_sha256"]:
        raise RuntimeError("frozen development states changed after preparation")
    states = json.loads(states_path.read_text(encoding="utf-8"))
    client = DeepSeekAdviceClient(
        out / "cache",
        budget=AdviceBudget(
            max_requests=12,
            max_input_tokens=20_000,
            max_output_tokens=12 * 160,
            max_estimated_cny=0.10,
        ),
        global_budget_path=ROOT / "outputs" / "deepseek_global_budget.json",
        global_max_cny=1.0,
    )
    rows = []
    for state in states:
        result = client.advise(state)
        violations = plan_safety_violations(result.plan, state)
        rows.append(
            {
                "scenario_id": state["scenario_id"],
                "event_type": state["event_type"],
                "source": result.source,
                "valid_model_response": result.source in {"api", "cache"},
                "safety_violations": violations,
                "plan": {
                    "phases": [asdict(phase) for phase in result.plan.phases],
                    "confidence": result.plan.confidence,
                    "reason_codes": list(result.plan.reason_codes),
                    "replan_triggers": list(result.plan.replan_triggers),
                },
                "usage": asdict(result.usage),
            }
        )
    client.write_ledger(out / "usage_ledger.json")
    valid = sum(bool(row["valid_model_response"]) for row in rows)
    violations = sum(len(row["safety_violations"]) for row in rows)
    passed = valid == len(rows) and violations == 0
    summary = {
        "status": "passed" if passed else "failed",
        "model": client.model,
        "states": len(rows),
        "valid_model_responses": valid,
        "valid_response_rate": valid / len(rows),
        "safety_violations": violations,
        "new_api_requests": client.ledger.requests,
        "input_tokens": client.ledger.input_tokens,
        "output_tokens": client.ledger.output_tokens,
        "cache_hit_tokens": client.ledger.cache_hit_tokens,
        "estimated_cost_cny_conservative": client.ledger.estimated_cny,
        "final_test_scenarios_loaded": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "response_audit.json", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
        return 0
    return 0 if run(args.output) else 2


if __name__ == "__main__":
    raise SystemExit(main())
