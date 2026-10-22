from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.action_risk_ranking_v2 import ACTION_IDS
from fire_evac.action_ranking_client_v2 import BudgetedActionRanker
from fire_evac.deepseek_advice import DeepSeekUsage


STATE = {
    "scenario_id": "R1",
    "occupancy": [0.25, 0.25, 0.40, 0.04, 0.04],
    "smoke": [0.002, 0.002, 0.002, 0.01, 0.06],
    "north_exit_capacity": 6,
    "south_exit_capacity": 8,
    "blockage_active": False,
    "blockage_steps_remaining": 0,
    "semantic_context": {"junction_occupancy_change_per_step": 0.05},
}


def response() -> tuple[str, DeepSeekUsage]:
    actions = []
    for rank, action in enumerate(ACTION_IDS, start=1):
        actions.append({"action_id": action, "risk_score": rank / 10, "rank": rank})
    return json.dumps({"actions": actions, "reason_codes": ["congestion_risk"]}), DeepSeekUsage(300, 180, 0)


def test_exact_query_cache_avoids_second_paid_transport(tmp_path: Path) -> None:
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return response()

    client = BudgetedActionRanker(tmp_path, transport=transport, max_cny=0.35, max_requests=2)
    first = client(STATE)
    second = client(dict(reversed(list(STATE.items()))))

    assert first.source == "api"
    assert second.source == "cache"
    assert calls == 1
    assert client.ledger()["attempted_requests"] == 1


def test_invalid_response_consumes_budget_and_uses_numeric_fallback(tmp_path: Path) -> None:
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return "{}", DeepSeekUsage(100, 2, 0)

    client = BudgetedActionRanker(tmp_path, transport=transport, max_requests=1)
    first = client(STATE)
    second = client({**STATE, "scenario_id": "R2"})

    assert first.source == "fallback_schema"
    assert second.source == "fallback_budget"
    assert calls == 1
    assert first.ranking.best_action in ACTION_IDS
    assert client.ledger()["attempted_requests"] == 1


def test_outputs_and_logs_never_contain_environment_secret(tmp_path: Path, monkeypatch) -> None:
    secret = "sentinel-secret-never-write"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    client = BudgetedActionRanker(tmp_path, transport=lambda *_: response())
    result = client(STATE)
    assert result.source == "api"

    text = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.*"))
    assert secret not in text
    assert "Authorization" not in text
