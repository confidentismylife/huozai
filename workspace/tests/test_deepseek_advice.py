from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.deepseek_advice import (
    AdviceBudget,
    DeepSeekAdviceClient,
    DeepSeekUsage,
    encode_plan_features,
    rule_plan,
)


SCENARIO = {
    "scenario_id": "S001",
    "forecast_source_branch": 0,
    "growth_rate": 0.02,
    "onset_step": 3,
    "north_exit_capacity": 8,
    "south_exit_capacity": 4,
    "blocked_branch": -1,
    "blockage_end_step": 0,
}


def valid_response() -> tuple[str, DeepSeekUsage]:
    return (
        json.dumps(
            {
                "phases": [
                    {"route": "south", "release_fraction": 0.5, "duration_steps": 2},
                    {"route": "south", "release_fraction": 0.75, "duration_steps": 3},
                    {"route": "north", "release_fraction": 0.25, "duration_steps": 2},
                ],
                "confidence": 0.74,
                "reason_codes": ["avoid_predicted_smoke", "balance_congestion"],
                "replan_triggers": ["smoke_reversal", "congestion_threshold"],
            }
        ),
        DeepSeekUsage(input_tokens=120, output_tokens=24, cache_hit_tokens=100),
    )


def test_cache_hit_does_not_call_transport(tmp_path: Path) -> None:
    calls = 0

    def transport(_messages, _max_tokens):
        nonlocal calls
        calls += 1
        return valid_response()

    client = DeepSeekAdviceClient(tmp_path, transport=transport)
    first = client.advise(SCENARIO)
    second = client.advise(SCENARIO)

    assert first.source == "api"
    assert second.source == "cache"
    assert calls == 1
    assert client.ledger.requests == 1


def test_exhausted_budget_uses_safe_rule_without_network(tmp_path: Path) -> None:
    def transport(_messages, _max_tokens):
        raise AssertionError("transport must not be called after the request budget is exhausted")

    client = DeepSeekAdviceClient(
        tmp_path,
        budget=AdviceBudget(max_requests=0, max_input_tokens=0, max_output_tokens=0),
        transport=transport,
    )
    result = client.advise(SCENARIO)

    assert result.source == "fallback_budget"
    assert result.plan.phases[0].route == "south"
    assert client.ledger.requests == 0


def test_cache_and_ledger_never_store_api_key(tmp_path: Path, monkeypatch) -> None:
    secret = "sentinel-secret-that-must-not-be-written"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    client = DeepSeekAdviceClient(tmp_path, transport=lambda *_: valid_response())
    client.advise(SCENARIO)
    client.write_ledger(tmp_path / "usage.json")

    contents = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))
    assert secret not in contents


def test_same_state_and_prompt_version_share_cache_key(tmp_path: Path) -> None:
    first = DeepSeekAdviceClient(tmp_path, transport=lambda *_: valid_response())
    second = DeepSeekAdviceClient(tmp_path, transport=lambda *_: (_ for _ in ()).throw(AssertionError()))
    first.advise(dict(reversed(list(SCENARIO.items()))))
    result = second.advise(SCENARIO)
    assert result.source == "cache"


def test_plan_features_encode_capacity_timing_and_triggers(tmp_path: Path) -> None:
    client = DeepSeekAdviceClient(tmp_path, transport=lambda *_: valid_response())
    plan = client.advise(SCENARIO).plan
    features = encode_plan_features(plan, age_steps=2, contradiction=True)
    assert features.shape == (21,)
    assert features[3] == 0.5
    assert features[4] == 2 / 12
    assert features[-2] == 2 / 12
    assert features[-1] == 1.0


def test_invalid_paid_response_still_consumes_request_budget(tmp_path: Path) -> None:
    calls = 0

    def invalid_transport(_messages, _max_tokens):
        nonlocal calls
        calls += 1
        return "{}", DeepSeekUsage(input_tokens=80, output_tokens=2)

    client = DeepSeekAdviceClient(
        tmp_path,
        budget=AdviceBudget(max_requests=1),
        transport=invalid_transport,
    )
    first = client.advise(SCENARIO)
    second = client.advise({**SCENARIO, "scenario_id": "S002"})

    assert first.source == "fallback_error"
    assert second.source == "fallback_budget"
    assert calls == 1
    assert client.ledger.requests == 1
    assert client.ledger.input_tokens == 80
    diagnostics = list((tmp_path / "invalid_responses").glob("*.json"))
    assert len(diagnostics) == 1
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert payload["raw_response"] == "{}"
    assert "schema" in payload["parse_error"].lower()


def test_rule_plan_avoids_blocked_nominally_safe_exit() -> None:
    state = {
        **SCENARIO,
        "north_exit_capacity": 8,
        "south_exit_capacity": 0,
        "blockage_active": True,
        "smoke": [0.0, 0.0, 0.0, 0.01, 0.0],
        "occupancy": [0.4, 0.4, 0.1, 0.05, 0.05],
    }
    plan = rule_plan(state)
    assert plan.phases[0].route == "north"
    assert plan.phases[0].release_fraction in (0.5, 0.75, 1.0)


def test_global_budget_persists_across_client_instances(tmp_path: Path) -> None:
    budget_path = tmp_path / "global_budget.json"
    first = DeepSeekAdviceClient(
        tmp_path / "first",
        transport=lambda *_: valid_response(),
        global_budget_path=budget_path,
        global_max_cny=1.0,
    )
    first.advise(SCENARIO)
    spent = json.loads(budget_path.read_text(encoding="utf-8"))["estimated_cny"]
    calls = 0

    def should_not_run(*_):
        nonlocal calls
        calls += 1
        return valid_response()

    second = DeepSeekAdviceClient(
        tmp_path / "second",
        transport=should_not_run,
        global_budget_path=budget_path,
        global_max_cny=spent,
    )
    result = second.advise({**SCENARIO, "scenario_id": "S002"})
    assert result.source == "fallback_budget"
    assert calls == 0
