"""Semantic incident-report constraints for the v3 LLM/MAPPO interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from .action_risk_ranking_v2 import ACTION_IDS, ActionRanking, ActionRisk


STATUSES = ("open", "restricted", "closed")
LIMITS = (0.0, 0.25, 0.50, 0.75, 1.0)
REASON_CODES = (
    "explicit_closure",
    "capacity_limit",
    "sensor_fault",
    "explicit_reopening",
    "latest_update",
    "no_restriction",
)
STYLES = ("direct", "operational", "negation", "correction")


@dataclass(frozen=True)
class RouteConstraint:
    status: str
    max_release: float
    smoke_sensor_reliable: bool


@dataclass(frozen=True)
class SemanticConstraints:
    north: RouteConstraint
    south: RouteConstraint
    valid_for_steps: int
    reason_codes: tuple[str, ...]


def _validate_route(route: RouteConstraint) -> None:
    if route.status not in STATUSES:
        raise ValueError("invalid route status")
    if route.max_release not in LIMITS:
        raise ValueError("max_release must use a supported level")
    if route.status == "closed" and route.max_release != 0.0:
        raise ValueError("closed route must have zero max_release")
    if route.status == "open" and route.max_release != 1.0:
        raise ValueError("open route must have full max_release")
    if route.status == "restricted" and route.max_release not in (0.25, 0.50, 0.75):
        raise ValueError("restricted route must use a partial max_release")
    if not isinstance(route.smoke_sensor_reliable, bool):
        raise ValueError("smoke_sensor_reliable must be boolean")


def parse_semantic_constraints(payload: str | Mapping[str, Any]) -> SemanticConstraints:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if set(data) != {"north", "south", "valid_for_steps", "reason_codes"}:
        raise ValueError("semantic constraint schema mismatch")

    def parse_route(value: Any) -> RouteConstraint:
        if not isinstance(value, Mapping) or set(value) != {
            "status", "max_release", "smoke_sensor_reliable"
        }:
            raise ValueError("route constraint schema mismatch")
        route = RouteConstraint(
            status=str(value["status"]),
            max_release=float(value["max_release"]),
            smoke_sensor_reliable=value["smoke_sensor_reliable"],
        )
        _validate_route(route)
        return route

    north, south = parse_route(data["north"]), parse_route(data["south"])
    steps = int(data["valid_for_steps"])
    if not 1 <= steps <= 12:
        raise ValueError("valid_for_steps must be between one and twelve")
    reasons = tuple(str(value) for value in data["reason_codes"])
    if (not 1 <= len(reasons) <= 4 or len(set(reasons)) != len(reasons)
            or any(value not in REASON_CODES for value in reasons)):
        raise ValueError("invalid reason_codes")
    return SemanticConstraints(north, south, steps, reasons)


def constraint_payload(value: SemanticConstraints) -> dict[str, Any]:
    return {
        "north": {
            "status": value.north.status,
            "max_release": value.north.max_release,
            "smoke_sensor_reliable": value.north.smoke_sensor_reliable,
        },
        "south": {
            "status": value.south.status,
            "max_release": value.south.max_release,
            "smoke_sensor_reliable": value.south.smoke_sensor_reliable,
        },
        "valid_for_steps": value.valid_for_steps,
        "reason_codes": list(value.reason_codes),
    }


def _semantic_reasons(north: RouteConstraint, south: RouteConstraint, latest: bool) -> tuple[str, ...]:
    values: list[str] = []
    if north.status == "closed" or south.status == "closed":
        values.append("explicit_closure")
    if north.status == "restricted" or south.status == "restricted":
        values.append("capacity_limit")
    if not north.smoke_sensor_reliable or not south.smoke_sensor_reliable:
        values.append("sensor_fault")
    if latest:
        values.append("latest_update")
    if not values:
        values.append("no_restriction")
    return tuple(values[:4])


def _route_direct(name: str, value: RouteConstraint) -> str:
    sensor = "RELIABLE" if value.smoke_sensor_reliable else "UNRELIABLE"
    return (f"{name.title()} status {value.status.upper()}; maximum release "
            f"{value.max_release:.2f}; {name} smoke sensor {sensor}.")


def _limit_words(limit: float) -> str:
    return {0.0: "zero", 0.25: "one quarter", 0.50: "one half",
            0.75: "three quarters", 1.0: "full"}[limit]


def _route_operational(name: str, value: RouteConstraint) -> str:
    if value.status == "closed":
        status = f"The {name} stair door hardware is jammed; send no one through the {name} route."
    elif value.status == "open":
        status = f"The {name} route can operate at full release."
    else:
        status = (f"Meter the {name} route at no more than {_limit_words(value.max_release)} "
                  "of normal release.")
    trust = "trustworthy" if value.smoke_sensor_reliable else "untrustworthy"
    return status + f" Treat the {name} smoke reading as {trust}."


def _route_negation(name: str, value: RouteConstraint) -> str:
    if value.status == "closed":
        status = f"{name.title()} is not open and no flow is permitted there."
    elif value.status == "open":
        status = f"{name.title()} is not closed or restricted; full release is permitted."
    else:
        status = (f"{name.title()} is not closed, but no more than {_limit_words(value.max_release)} "
                  "release is permitted.")
    sensor = (f"The {name} sensor is not faulty." if value.smoke_sensor_reliable
              else f"Do not rely on the {name} smoke sensor.")
    return status + " " + sensor


def _opposite(value: RouteConstraint) -> RouteConstraint:
    if value.status == "closed":
        return RouteConstraint("open", 1.0, not value.smoke_sensor_reliable)
    return RouteConstraint("closed", 0.0, not value.smoke_sensor_reliable)


def _render_report(
    north: RouteConstraint, south: RouteConstraint, steps: int, style: str
) -> tuple[str, SemanticConstraints]:
    latest = style == "correction"
    reasons = _semantic_reasons(north, south, latest)
    truth = SemanticConstraints(north, south, steps, reasons)
    if style == "direct":
        report = ("Current incident command. " + _route_direct("north", north) + " "
                  + _route_direct("south", south) + f" Valid for {steps} control steps.")
    elif style == "operational":
        report = (_route_operational("north", north) + " " + _route_operational("south", south)
                  + f" Follow this for the next {steps} control cycles.")
    elif style == "negation":
        report = (_route_negation("north", north) + " " + _route_negation("south", south)
                  + f" This instruction is not valid beyond {steps} control steps.")
    elif style == "correction":
        report = (
            "Earlier bulletin: " + _route_direct("north", _opposite(north)) + " "
            + _route_direct("south", _opposite(south))
            + " That bulletin is superseded. Latest update: "
            + _route_direct("north", north) + " " + _route_direct("south", south)
            + f" Apply the latest update for {steps} control steps."
        )
    else:
        raise ValueError("unknown language style")
    return report, truth


def build_semantic_gate_cases(seed: int = 20261024) -> list[dict[str, Any]]:
    truths = {
        "north_closed": (RouteConstraint("closed", 0.0, True), RouteConstraint("open", 1.0, True), 4),
        "south_closed": (RouteConstraint("open", 1.0, False), RouteConstraint("closed", 0.0, True), 8),
        "north_restricted": (RouteConstraint("restricted", 0.25, True), RouteConstraint("open", 1.0, True), 2),
        "south_restricted": (RouteConstraint("open", 1.0, True), RouteConstraint("restricted", 0.50, False), 4),
        "dual_restricted": (RouteConstraint("restricted", 0.75, False), RouteConstraint("restricted", 0.25, True), 8),
        "open_sensor_fault": (RouteConstraint("open", 1.0, True), RouteConstraint("open", 1.0, False), 2),
    }
    rows: list[dict[str, Any]] = []
    for truth_id, (north, south, steps) in truths.items():
        for style in STYLES:
            report, truth = _render_report(north, south, steps, style)
            opaque = hashlib.sha256(f"{truth_id}|{style}".encode("utf-8")).hexdigest()[:16]
            rows.append({
                "case_id": f"SEM_{opaque}",
                "truth_id": truth_id,
                "style": style,
                "incident_report": report,
                "truth": truth,
            })
    rng = np.random.default_rng(seed)
    return [rows[int(index)] for index in rng.permutation(len(rows))]


def _parse_number(text: str) -> float | None:
    for phrase, value in (("three quarters", 0.75), ("one quarter", 0.25),
                          ("one half", 0.50), ("half", 0.50), ("full", 1.0), ("zero", 0.0)):
        if phrase in text:
            return value
    match = re.search(r"(?:maximum release|release)\s+([01](?:\.\d+)?)", text)
    return None if match is None else float(match.group(1))


def _rule_route(text: str, name: str) -> RouteConstraint:
    lowered = text.lower()
    direct = re.search(
        rf"{name} status (open|restricted|closed); maximum release ([01](?:\.\d+)?); "
        rf"{name} smoke sensor (reliable|unreliable)", lowered,
    )
    if direct:
        return RouteConstraint(direct.group(1), float(direct.group(2)), direct.group(3) == "reliable")

    sentences = [value.strip() for value in re.split(r"[.!?]", lowered) if name in value]
    joined = " ".join(sentences)
    if "hardware is jammed" in joined or f"{name} is not open" in joined:
        status, limit = "closed", 0.0
    elif "operate at full release" in joined or "not closed or restricted" in joined:
        status, limit = "open", 1.0
    elif f"meter the {name} route" in joined or f"{name} is not closed, but" in joined:
        status = "restricted"
        parsed = _parse_number(joined)
        limit = 0.50 if parsed is None else parsed
    else:
        status, limit = "open", 1.0
    if f"do not rely on the {name} smoke sensor" in joined or "untrustworthy" in joined:
        reliable = False
    elif f"{name} sensor is not faulty" in joined or "trustworthy" in joined:
        reliable = True
    else:
        reliable = True
    return RouteConstraint(status, limit, reliable)


def rule_extract_constraints(report: str) -> SemanticConstraints:
    text = str(report)
    latest = "latest update:" in text.lower()
    if latest:
        text = re.split(r"latest update:", text, flags=re.IGNORECASE, maxsplit=1)[1]
    north = _rule_route(text, "north")
    south = _rule_route(text, "south")
    match = re.search(r"(?:valid for|next|beyond|for)\s+(\d+)\s+control", text.lower())
    if match is None:
        raise ValueError("rule parser could not find validity duration")
    steps = int(match.group(1))
    return SemanticConstraints(north, south, steps, _semantic_reasons(north, south, latest))


def compile_ranking_constraints(
    ranking: ActionRanking, constraints: SemanticConstraints
) -> ActionRanking:
    scores: dict[str, float] = {}
    for item in ranking.actions:
        action = item.action_id
        score = float(item.risk_score)
        if action != "wait":
            route_name, suffix = action.split("_")
            release = int(suffix) / 100.0
            constraint = constraints.north if route_name == "north" else constraints.south
            if constraint.status == "closed" or release > constraint.max_release + 1e-12:
                score = 1.0
            elif not constraint.smoke_sensor_reliable:
                score = min(0.98, score + 0.15)
        scores[action] = score
    ordered = sorted(ACTION_IDS, key=lambda action: (scores[action], action))
    ranks = {action: index + 1 for index, action in enumerate(ordered)}
    return ActionRanking(
        tuple(ActionRisk(action, scores[action], ranks[action]) for action in ACTION_IDS),
        ranking.reason_codes,
    )


def score_constraint_predictions(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("at least one prediction is required")
    exact: list[bool] = []
    statuses: list[bool] = []
    releases: list[bool] = []
    sensors: list[bool] = []
    durations: list[bool] = []
    correction: list[bool] = []
    for row in records:
        truth: SemanticConstraints = row["truth"]
        prediction: SemanticConstraints = row["prediction"]
        status_values = [prediction.north.status == truth.north.status,
                         prediction.south.status == truth.south.status]
        release_values = [prediction.north.max_release == truth.north.max_release,
                          prediction.south.max_release == truth.south.max_release]
        sensor_values = [prediction.north.smoke_sensor_reliable == truth.north.smoke_sensor_reliable,
                         prediction.south.smoke_sensor_reliable == truth.south.smoke_sensor_reliable]
        duration_value = prediction.valid_for_steps == truth.valid_for_steps
        row_exact = all(status_values + release_values + sensor_values + [duration_value])
        exact.append(row_exact)
        statuses.extend(status_values)
        releases.extend(release_values)
        sensors.extend(sensor_values)
        durations.append(duration_value)
        if str(row["style"]) == "correction":
            correction.append(row_exact)
    return {
        "cases": float(len(records)),
        "exact_accuracy": float(np.mean(exact)),
        "route_status_accuracy": float(np.mean(statuses)),
        "max_release_accuracy": float(np.mean(releases)),
        "sensor_reliability_accuracy": float(np.mean(sensors)),
        "validity_duration_accuracy": float(np.mean(durations)),
        "correction_cases": float(len(correction)),
        "correction_exact_accuracy": float(np.mean(correction)) if correction else float("nan"),
    }
