"""Budgeted DeepSeek client for frozen semantic-constraint extraction v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

from .campaign_extension import saved_credential, write_json
from .deepseek_advice import DeepSeekUsage, MODEL_NAME
from .semantic_constraints_v3 import (
    SemanticConstraints,
    constraint_payload,
    parse_semantic_constraints,
    rule_extract_constraints,
)


PROMPT_VERSION = "semantic-constraint-extraction-v3"
SYSTEM_PROMPT = (
    "Extract the latest actionable constraints from an incident-command report for a two-exit evacuation. "
    "Return one JSON object only with exactly north, south, valid_for_steps, reason_codes. north and south "
    "must each contain exactly status, max_release, smoke_sensor_reliable. status is open, restricted, or "
    "closed. open requires max_release 1.0; closed requires 0.0; restricted requires 0.25, 0.50, or 0.75. "
    "smoke_sensor_reliable is true or false. valid_for_steps is an integer 1 through 12. reason_codes must "
    "contain one to four unique values selected only from explicit_closure, capacity_limit, sensor_fault, "
    "explicit_reopening, latest_update, no_restriction. A latest update overrides every earlier bulletin. "
    "Handle paraphrase and negation literally. Extract stated constraints only; do not recommend an action, "
    "invent a restriction, estimate outcomes, or output confidence or prose."
)


@dataclass(frozen=True)
class SemanticExtractionResult:
    constraints: SemanticConstraints
    source: str
    usage: DeepSeekUsage


Transport = Callable[[Sequence[Mapping[str, str]], int], tuple[str, DeepSeekUsage | Mapping[str, Any]]]


class BudgetedSemanticExtractor:
    def __init__(
        self,
        output: str | Path,
        max_cny: float = 0.25,
        max_requests: int = 24,
        max_tokens: int = 320,
        transport: Transport | None = None,
    ) -> None:
        self.output = Path(output)
        self.cache = self.output / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget_path = self.output / "budget.json"
        self.records_path = self.output / "requests.jsonl"
        self.max_cny = float(max_cny)
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self.transport = transport
        self._key: str | None = None
        self.last_latency_seconds = 0.0
        if not self.budget_path.exists():
            self._save({
                "attempted_requests": 0,
                "successful_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hit_tokens": 0,
                "actual_cost_upper_cny": 0.0,
                "reserved_cny": 0.0,
            })

    def ledger(self) -> dict[str, Any]:
        return json.loads(self.budget_path.read_text(encoding="utf-8"))

    def _save(self, ledger: Mapping[str, Any]) -> None:
        write_json(self.budget_path, {
            **dict(ledger),
            "maximum_cny": self.max_cny,
            "maximum_requests": self.max_requests,
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "input_cny_per_million_upper": 2.5,
            "output_cny_per_million_upper": 9.0,
            "credential_logged": False,
        })

    def messages(self, report: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "incident_report=" + str(report)},
        ]

    def _cache_key(self, messages: Sequence[Mapping[str, str]]) -> str:
        canonical = json.dumps({
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "messages": list(messages),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _online(self, messages: Sequence[Mapping[str, str]], maximum: int):
        if self._key is None:
            self._key = saved_credential()
        body = json.dumps({
            "model": MODEL_NAME,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": maximum,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"DeepSeek HTTP {exc.code}") from None
        usage = payload.get("usage", {})
        return payload["choices"][0]["message"]["content"], DeepSeekUsage(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            int(usage.get("prompt_cache_hit_tokens", 0)),
        )

    @staticmethod
    def _usage(value: DeepSeekUsage | Mapping[str, Any]) -> DeepSeekUsage:
        if isinstance(value, DeepSeekUsage):
            return value
        return DeepSeekUsage(
            int(value.get("prompt_tokens", value.get("input_tokens", 0))),
            int(value.get("completion_tokens", value.get("output_tokens", 0))),
            int(value.get("prompt_cache_hit_tokens", value.get("cache_hit_tokens", 0))),
        )

    def __call__(self, report: str) -> SemanticExtractionResult:
        messages = self.messages(report)
        key = self._cache_key(messages)
        cache_path = self.cache / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.last_latency_seconds = float(cached.get("latency_seconds", 0.0))
            return SemanticExtractionResult(
                parse_semantic_constraints(cached["constraints"]), "cache", DeepSeekUsage()
            )

        fallback = rule_extract_constraints(report)
        ledger = self.ledger()
        conservative_input = sum(len(item["content"].encode("utf-8")) for item in messages) + 256
        reservation = (conservative_input * 2.5 + self.max_tokens * 9.0) / 1_000_000.0
        if (int(ledger["attempted_requests"]) >= self.max_requests
                or float(ledger["actual_cost_upper_cny"]) + float(ledger["reserved_cny"]) + reservation > self.max_cny):
            return SemanticExtractionResult(fallback, "fallback_budget", DeepSeekUsage())

        ledger["attempted_requests"] = int(ledger["attempted_requests"]) + 1
        ledger["reserved_cny"] = float(ledger["reserved_cny"]) + reservation
        self._save(ledger)
        started = time.perf_counter()
        raw: str | None = None
        usage = DeepSeekUsage()
        constraints = fallback
        status = "fallback_transport"
        error_type: str | None = None
        try:
            raw, reported = (self.transport or self._online)(messages, self.max_tokens)
            usage = self._usage(reported)
            ledger["reserved_cny"] = max(0.0, float(ledger["reserved_cny"]) - reservation)
            ledger["actual_cost_upper_cny"] = float(ledger["actual_cost_upper_cny"]) + (
                usage.input_tokens * 2.5 + usage.output_tokens * 9.0
            ) / 1_000_000.0
            ledger["input_tokens"] = int(ledger["input_tokens"]) + usage.input_tokens
            ledger["output_tokens"] = int(ledger["output_tokens"]) + usage.output_tokens
            ledger["cache_hit_tokens"] = int(ledger["cache_hit_tokens"]) + usage.cache_hit_tokens
            try:
                constraints = parse_semantic_constraints(raw)
                status = "api"
                ledger["successful_requests"] = int(ledger["successful_requests"]) + 1
                write_json(cache_path, {
                    "model": MODEL_NAME,
                    "prompt_version": PROMPT_VERSION,
                    "constraints": constraint_payload(constraints),
                    "usage": asdict(usage),
                    "incident_report": report,
                    "latency_seconds": time.perf_counter() - started,
                })
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                status = "fallback_schema"
                error_type = type(exc).__name__
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            self.last_latency_seconds = time.perf_counter() - started
            self._save(ledger)
            record: dict[str, Any] = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "cache_key": key,
                "source": status,
                "error_type": error_type,
                "incident_report": report,
                "constraints": constraint_payload(constraints),
                "usage": asdict(usage),
                "latency_seconds": self.last_latency_seconds,
            }
            if raw is not None:
                record["raw_response"] = raw
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return SemanticExtractionResult(constraints, status, usage)
