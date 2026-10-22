from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deepseek_advice import EvacuationPlan, parse_plan


def _state_vector(state: Mapping[str, Any]) -> np.ndarray:
    occupancy = [float(value) for value in state.get("occupancy", [0.0] * 5)]
    smoke = [float(value) for value in state.get("smoke", [0.0] * 5)]
    if len(occupancy) != 5 or len(smoke) != 5:
        raise ValueError("frozen-plan states require five occupancy and five smoke values")
    return np.asarray(
        [
            float(state.get("forecast_source_branch", 0)),
            float(state.get("growth_rate", 0.0)) / 0.05,
            float(state.get("onset_step", 0)) / 12.0,
            *occupancy,
            *(min(1.0, value / 0.10) for value in smoke),
            float(state.get("north_exit_capacity", 0)) / 10.0,
            float(state.get("south_exit_capacity", 0)) / 10.0,
            float(state.get("blockage_steps_remaining", 0)) / 12.0,
            float(state.get("last_release_fraction", 1.0)),
        ],
        dtype=np.float32,
    )


class FrozenPlanLibrary:
    """Nearest-case retrieval over valid, already-paid DeepSeek plans."""

    def __init__(self, records: list[tuple[dict[str, Any], EvacuationPlan]]):
        if not records:
            raise ValueError("frozen DeepSeek plan library cannot be empty")
        self.records = records
        self.vectors = np.stack([_state_vector(state) for state, _ in records])
        self.last_match: dict[str, Any] | None = None

    @classmethod
    def from_files(cls, states_path: str | Path, audit_path: str | Path) -> "FrozenPlanLibrary":
        states = json.loads(Path(states_path).read_text(encoding="utf-8"))
        audits = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        by_id = {str(state["scenario_id"]): dict(state) for state in states}
        if len(by_id) != len(states):
            raise ValueError("frozen state identifiers must be unique")
        records: list[tuple[dict[str, Any], EvacuationPlan]] = []
        for audit in audits:
            if not bool(audit.get("valid_model_response", False)):
                raise ValueError("plan library accepts only valid model responses")
            scenario_id = str(audit["scenario_id"])
            if scenario_id not in by_id:
                raise ValueError(f"audit scenario missing from frozen states: {scenario_id}")
            state = by_id[scenario_id]
            if str(audit["event_type"]) != str(state["event_type"]):
                raise ValueError(f"event mismatch for {scenario_id}")
            records.append((state, parse_plan(audit["plan"])))
        if len(records) != len(states):
            raise ValueError("every frozen state must have one valid DeepSeek response")
        return cls(records)

    def __call__(self, state: Mapping[str, Any]) -> EvacuationPlan:
        event_type = str(state.get("event_type", "initial"))
        candidates = [
            index for index, (stored, _) in enumerate(self.records)
            if str(stored.get("event_type")) == event_type
        ]
        if not candidates:
            candidates = list(range(len(self.records)))
        query = _state_vector(state)
        distances = [float(np.square(query - self.vectors[index]).sum()) for index in candidates]
        winner = candidates[int(np.argmin(distances))]
        matched_state, plan = self.records[winner]
        self.last_match = {
            "query_scenario_id": str(state.get("scenario_id", "unknown")),
            "matched_scenario_id": str(matched_state["scenario_id"]),
            "event_type": event_type,
            "squared_distance": distances[int(np.argmin(distances))],
            "candidate_count": len(candidates),
        }
        return plan
