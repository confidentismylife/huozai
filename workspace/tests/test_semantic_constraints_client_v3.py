import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.deepseek_advice import DeepSeekUsage
from fire_evac.semantic_constraints_client_v3 import BudgetedSemanticExtractor
from fire_evac.semantic_constraints_v3 import (
    build_semantic_gate_cases,
    constraint_payload,
)


CASE = next(case for case in build_semantic_gate_cases(seed=1) if case["style"] == "direct")


def response():
    return json.dumps(constraint_payload(CASE["truth"])), DeepSeekUsage(250, 80, 0)


def test_semantic_extractor_caches_exact_report(tmp_path: Path) -> None:
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return response()

    client = BudgetedSemanticExtractor(tmp_path, transport=transport, max_requests=2)
    first = client(CASE["incident_report"])
    second = client(CASE["incident_report"])
    assert first.source == "api"
    assert second.source == "cache"
    assert first.constraints == CASE["truth"]
    assert calls == 1


def test_invalid_response_consumes_budget_and_uses_rule_fallback(tmp_path: Path) -> None:
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return "{}", DeepSeekUsage(100, 2, 0)

    client = BudgetedSemanticExtractor(tmp_path, transport=transport, max_requests=1)
    first = client(CASE["incident_report"])
    second = client(CASE["incident_report"] + " ")
    assert first.source == "fallback_schema"
    assert second.source == "fallback_budget"
    assert calls == 1
    assert first.constraints == CASE["truth"]


def test_semantic_extractor_never_logs_secret_or_header(tmp_path: Path, monkeypatch) -> None:
    secret = "sentinel-secret-never-write"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    client = BudgetedSemanticExtractor(tmp_path, transport=lambda *_: response())
    client(CASE["incident_report"])
    text = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.*"))
    assert secret not in text
    assert "Authorization" not in text
