from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


MODEL_NAME = "deepseek-flash"
PROMPT_VERSION = "evacuation-plan-v2"
ROUTES = ("north", "south", "wait")
RELEASE_LEVELS = (0.25, 0.50, 0.75, 1.00)
REASON_CODES = (
    "avoid_predicted_smoke",
    "balance_congestion",
    "respect_blockage",
    "limit_release",
    "wait_for_information",
)
REPLAN_TRIGGERS = ("blockage_change", "smoke_reversal", "congestion_threshold")

SYSTEM_PROMPT = (
    "You plan a two-exit fire evacuation for a lower-level MAPPO controller. "
    "Return one JSON object only, without markdown or commentary. Use exactly four top-level keys: "
    "phases, confidence, reason_codes, replan_triggers. Produce exactly three short phases. "
    "Each phase must use exactly the keys route, release_fraction, duration_steps. "
    "Each phase selects north, south, or wait; "
    "a release fraction from 0.25, 0.50, 0.75, 1.00; and a duration from 1 to 12 control steps. "
    "Use capacity, blockage, smoke, crowd and any supplied semantic context. Do not mechanically choose "
    "the opposite of the forecast source. confidence must be between 0 and 1. "
    "reason_codes must contain 1 to 3 values chosen only from: avoid_predicted_smoke, "
    "balance_congestion, respect_blockage, limit_release, wait_for_information. "
    "replan_triggers must contain unique values chosen only from: blockage_change, smoke_reversal, "
    "congestion_threshold. Do not invent any other enum value. "
    'Example JSON: {"phases":[{"route":"south","release_fraction":0.5,"duration_steps":2},'
    '{"route":"south","release_fraction":0.75,"duration_steps":3},'
    '{"route":"north","release_fraction":0.25,"duration_steps":2}],"confidence":0.7,'
    '"reason_codes":["avoid_predicted_smoke"],"replan_triggers":["smoke_reversal"]}.'
)


@dataclass(frozen=True)
class PlanPhase:
    route: str
    release_fraction: float
    duration_steps: int


@dataclass(frozen=True)
class EvacuationPlan:
    phases: tuple[PlanPhase, PlanPhase, PlanPhase]
    confidence: float
    reason_codes: tuple[str, ...]
    replan_triggers: tuple[str, ...]


@dataclass(frozen=True)
class DeepSeekUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0


@dataclass(frozen=True)
class AdviceBudget:
    max_requests: int = 48
    max_input_tokens: int = 50_000
    max_output_tokens: int = 8_000
    max_estimated_cny: float = 0.50
    input_cny_per_million: float = 2.5
    cached_input_cny_per_million: float = 0.5
    output_cny_per_million: float = 9.0


@dataclass
class UsageLedger:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    estimated_cny: float = 0.0


@dataclass(frozen=True)
class AdviceResult:
    plan: EvacuationPlan
    source: str
    usage: DeepSeekUsage


def parse_plan(payload: str | Mapping[str, Any]) -> EvacuationPlan:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    required = {"phases", "confidence", "reason_codes", "replan_triggers"}
    if set(data) != required:
        raise ValueError("DeepSeek plan schema mismatch")
    raw_phases = data["phases"]
    if not isinstance(raw_phases, list) or len(raw_phases) != 3:
        raise ValueError("plan must contain exactly three phases")
    phases: list[PlanPhase] = []
    for raw in raw_phases:
        if set(raw) != {"route", "release_fraction", "duration_steps"}:
            raise ValueError("phase schema mismatch")
        route = str(raw["route"])
        release = float(raw["release_fraction"])
        duration = int(raw["duration_steps"])
        if route not in ROUTES or release not in RELEASE_LEVELS or not 1 <= duration <= 12:
            raise ValueError("unsupported route, release fraction, or phase duration")
        phases.append(PlanPhase(route, release, duration))
    confidence = float(data["confidence"])
    reasons = tuple(str(value) for value in data["reason_codes"])
    triggers = tuple(str(value) for value in data["replan_triggers"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")
    if not 1 <= len(reasons) <= 3 or any(value not in REASON_CODES for value in reasons):
        raise ValueError("invalid reason codes")
    if len(triggers) > 3 or len(set(triggers)) != len(triggers) or any(value not in REPLAN_TRIGGERS for value in triggers):
        raise ValueError("invalid replanning triggers")
    return EvacuationPlan(tuple(phases), confidence, reasons, triggers)  # type: ignore[arg-type]


def encode_plan_features(
    plan: EvacuationPlan, age_steps: int = 0, contradiction: bool = False
) -> np.ndarray:
    """Encode capacity, timing and event semantics into 21 controller features."""
    features: list[float] = []
    for phase in plan.phases:
        features.extend(float(phase.route == route) for route in ROUTES)
        features.append(float(phase.release_fraction))
        features.append(min(1.0, phase.duration_steps / 12.0))
    features.append(float(plan.confidence))
    features.extend(float(trigger in plan.replan_triggers) for trigger in REPLAN_TRIGGERS)
    features.append(min(1.0, max(0, age_steps) / 12.0))
    features.append(float(bool(contradiction)))
    return np.asarray(features, dtype=np.float32)


def rule_plan(state: Mapping[str, Any]) -> EvacuationPlan:
    """Same-interface numeric baseline and safe fallback for failed API calls."""
    capacities = [
        int(state.get("north_exit_capacity", 0)),
        int(state.get("south_exit_capacity", 0)),
    ]
    smoke = list(state.get("smoke", (0.0,) * 5))
    occupancy = list(state.get("occupancy", (0.0,) * 5))
    if capacities[0] <= 0 and capacities[1] <= 0:
        route = "wait"
        release = 0.25
        confidence = 0.9
        reasons = ("respect_blockage", "wait_for_information")
    else:
        forecast_source = int(state.get("forecast_source_branch", 0))
        scores = []
        for branch in (0, 1):
            capacity = capacities[branch]
            branch_smoke = float(smoke[3 + branch]) if len(smoke) >= 5 else float(branch == forecast_source) * 0.02
            branch_occupancy = float(occupancy[3 + branch]) if len(occupancy) >= 5 else 0.0
            score = 100.0 * branch_smoke + 2.0 * branch_occupancy + 1.0 / max(1, capacity)
            score += float(branch == forecast_source)
            if capacity <= 0:
                score += 100.0
            scores.append(score)
        target = int(np.argmin(scores))
        route = "north" if target == 0 else "south"
        junction_occupancy = float(occupancy[2]) if len(occupancy) >= 3 else 0.0
        target_smoke = float(smoke[3 + target]) if len(smoke) >= 5 else 0.0
        if junction_occupancy >= 45 / 240:
            release = 0.25
        elif capacities[target] <= 3 or target_smoke > 0.02:
            release = 0.50
        else:
            release = 0.75
        confidence = float(np.clip(0.5 + abs(scores[0] - scores[1]) / 4.0, 0.5, 0.95))
        reasons = ("respect_blockage", "balance_congestion")
    next_release = RELEASE_LEVELS[min(RELEASE_LEVELS.index(release) + 1, len(RELEASE_LEVELS) - 1)]
    phases = (
        PlanPhase(route, release, 2),
        PlanPhase(route, next_release, 3),
        PlanPhase(route, 1.00, 4),
    )
    return EvacuationPlan(phases, confidence, reasons, REPLAN_TRIGGERS)


def fallback_plan(state: Mapping[str, Any]) -> EvacuationPlan:
    return rule_plan(state)


Transport = Callable[[Sequence[Mapping[str, str]], int], tuple[str, DeepSeekUsage]]


class DeepSeekAdviceClient:
    """Cached, budgeted DeepSeek planner; the API key is read only in memory."""

    def __init__(
        self,
        cache_dir: str | Path,
        budget: AdviceBudget | None = None,
        transport: Transport | None = None,
        model: str = MODEL_NAME,
        max_output_tokens: int = 160,
        global_budget_path: str | Path | None = None,
        global_max_cny: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget or AdviceBudget()
        self.transport = transport
        self.model = str(model)
        self.max_output_tokens = int(max_output_tokens)
        self.global_budget_path = None if global_budget_path is None else Path(global_budget_path)
        self.global_max_cny = float(global_max_cny)
        self.ledger = UsageLedger()

    def _messages(self, state: Mapping[str, Any]) -> list[dict[str, str]]:
        state_text = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"observable_state={state_text}"},
        ]

    def _cache_path(self, messages: Sequence[Mapping[str, str]]) -> Path:
        canonical = json.dumps(
            {"model": self.model, "prompt_version": PROMPT_VERSION, "messages": messages},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self.cache_dir / f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}.json"

    def _estimated_input_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return max(1, sum(len(message["content"]) for message in messages) // 3)

    def _estimate_cost(self, usage: DeepSeekUsage) -> float:
        uncached = max(0, usage.input_tokens - usage.cache_hit_tokens)
        return (
            uncached * self.budget.input_cny_per_million
            + usage.cache_hit_tokens * self.budget.cached_input_cny_per_million
            + usage.output_tokens * self.budget.output_cny_per_million
        ) / 1_000_000.0

    def _has_budget(self, estimated_input: int) -> bool:
        if self.ledger.requests >= self.budget.max_requests:
            return False
        if self.ledger.input_tokens + estimated_input > self.budget.max_input_tokens:
            return False
        if self.ledger.output_tokens + self.max_output_tokens > self.budget.max_output_tokens:
            return False
        conservative = DeepSeekUsage(estimated_input, self.max_output_tokens, 0)
        if self.ledger.estimated_cny + self._estimate_cost(conservative) > self.budget.max_estimated_cny:
            return False
        if self.global_budget_path is not None:
            global_usage = self._read_global_budget()
            if float(global_usage["estimated_cny"]) + self._estimate_cost(conservative) > self.global_max_cny:
                return False
        return True

    def _read_global_budget(self) -> dict[str, float | int]:
        if self.global_budget_path is None or not self.global_budget_path.exists():
            return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_hit_tokens": 0, "estimated_cny": 0.0}
        try:
            data = json.loads(self.global_budget_path.read_text(encoding="utf-8"))
            return {
                "requests": int(data["requests"]),
                "input_tokens": int(data["input_tokens"]),
                "output_tokens": int(data["output_tokens"]),
                "cache_hit_tokens": int(data["cache_hit_tokens"]),
                "estimated_cny": float(data["estimated_cny"]),
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("global DeepSeek budget ledger is unreadable; refusing new requests") from exc

    def _record_global_usage(self, usage: DeepSeekUsage) -> None:
        if self.global_budget_path is None:
            return
        current = self._read_global_budget()
        current["requests"] = int(current["requests"]) + 1
        current["input_tokens"] = int(current["input_tokens"]) + usage.input_tokens
        current["output_tokens"] = int(current["output_tokens"]) + usage.output_tokens
        current["cache_hit_tokens"] = int(current["cache_hit_tokens"]) + usage.cache_hit_tokens
        current["estimated_cny"] = float(current["estimated_cny"]) + self._estimate_cost(usage)
        payload = {
            **current,
            "maximum_cny": self.global_max_cny,
            "last_model": self.model,
            "pricing_is_conservative_estimate": True,
        }
        self.global_budget_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.global_budget_path.with_suffix(self.global_budget_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.global_budget_path)

    def _online_transport(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int
    ) -> tuple[str, DeepSeekUsage]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured in the process environment")
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cache_hits = int(getattr(details, "cached_tokens", 0) or 0)
        if not cache_hits:
            cache_hits = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        return (
            response.choices[0].message.content or "",
            DeepSeekUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cache_hit_tokens=cache_hits,
            ),
        )

    def advise(self, state: Mapping[str, Any]) -> AdviceResult:
        messages = self._messages(state)
        cache_path = self._cache_path(messages)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return AdviceResult(parse_plan(cached["plan"]), "cache", DeepSeekUsage())
        if not self._has_budget(self._estimated_input_tokens(messages)):
            return AdviceResult(fallback_plan(state), "fallback_budget", DeepSeekUsage())
        usage = DeepSeekUsage()
        try:
            raw, usage = (self.transport or self._online_transport)(messages, self.max_output_tokens)
        except Exception:
            self.ledger.requests += 1
            self._record_global_usage(usage)
            return AdviceResult(fallback_plan(state), "fallback_error", DeepSeekUsage())
        self.ledger.requests += 1
        self.ledger.input_tokens += usage.input_tokens
        self.ledger.output_tokens += usage.output_tokens
        self.ledger.cache_hit_tokens += usage.cache_hit_tokens
        self.ledger.estimated_cny += self._estimate_cost(usage)
        self._record_global_usage(usage)
        try:
            plan = parse_plan(raw)
        except Exception as exc:
            diagnostic_dir = self.cache_dir / "invalid_responses"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            (diagnostic_dir / cache_path.name).write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "prompt_version": PROMPT_VERSION,
                        "cache_key": cache_path.stem,
                        "raw_response": raw,
                        "parse_error": str(exc),
                        "usage": asdict(usage),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return AdviceResult(fallback_plan(state), "fallback_error", usage)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "plan": {
                        "phases": [asdict(phase) for phase in plan.phases],
                        "confidence": plan.confidence,
                        "reason_codes": list(plan.reason_codes),
                        "replan_triggers": list(plan.replan_triggers),
                    },
                    "usage": asdict(usage),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return AdviceResult(plan, "api", usage)

    def write_ledger(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "budget": asdict(self.budget),
                    "usage": asdict(self.ledger),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target
