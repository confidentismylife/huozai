from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


ALLOWED_REASON_CODES = {
    "familiar_route",
    "follow_sign",
    "follow_majority",
    "avoid_smoke",
    "avoid_congestion",
    "help_companion",
    "wait_for_information",
    "other",
}


class LLMDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class LLMDecision:
    start_evacuation: bool
    target_exit: str | None
    follow_sign: bool
    follow_crowd: bool
    decision_delay_s: float
    confidence: float
    reason_codes: tuple[str, ...]


def canonical_state_hash(state: dict[str, Any]) -> str:
    """Stable key for response caching and audit logs."""

    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_llm_decision(payload: str | dict[str, Any]) -> LLMDecision:
    """Parse and validate the minimal high-level evacuation decision schema."""

    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMDecisionError("response is not valid JSON") from exc

    required = {
        "start_evacuation",
        "target_exit",
        "follow_sign",
        "follow_crowd",
        "decision_delay_s",
        "confidence",
        "reason_codes",
    }
    if set(data) != required:
        missing = sorted(required - set(data))
        extra = sorted(set(data) - required)
        raise LLMDecisionError(f"schema mismatch; missing={missing}, extra={extra}")
    for field in ("start_evacuation", "follow_sign", "follow_crowd"):
        if type(data[field]) is not bool:
            raise LLMDecisionError(f"{field} must be boolean")
    if data["target_exit"] is not None and not isinstance(data["target_exit"], str):
        raise LLMDecisionError("target_exit must be a string or null")

    delay = float(data["decision_delay_s"])
    confidence = float(data["confidence"])
    if not 0.0 <= delay <= 300.0:
        raise LLMDecisionError("decision_delay_s must be between 0 and 300")
    if not 0.0 <= confidence <= 1.0:
        raise LLMDecisionError("confidence must be between 0 and 1")
    reasons = tuple(data["reason_codes"])
    if len(reasons) > 5 or any(code not in ALLOWED_REASON_CODES for code in reasons):
        raise LLMDecisionError("reason_codes contains an unsupported value")

    return LLMDecision(
        start_evacuation=data["start_evacuation"],
        target_exit=data["target_exit"],
        follow_sign=data["follow_sign"],
        follow_crowd=data["follow_crowd"],
        decision_delay_s=delay,
        confidence=confidence,
        reason_codes=reasons,
    )

