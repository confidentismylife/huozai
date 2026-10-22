from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Protocol

import numpy as np


ZONE_NAMES = (
    "room_a",
    "room_b",
    "junction",
    "north_corridor",
    "south_corridor",
)


@dataclass(frozen=True)
class FireScenario:
    """Compact scenario description used by the synthetic hazard provider."""

    scenario_id: str
    source_branch: int  # 0 north, 1 south
    growth_rate: float
    onset_step: int
    background_hazard: float = 0.002


class HazardProvider(Protocol):
    def sample_scenario(self, rng: np.random.Generator, prefix: str = "train") -> FireScenario: ...

    def values(self, scenario: FireScenario, step: int) -> np.ndarray: ...


class SyntheticHazardProvider:
    """Fast hazard approximation for software and RL development.

    Hazard is expressed as FED increment per simulation step. This provider is
    intentionally simple and must be replaced by FDS-derived values for the
    scientific experiment.
    """

    def sample_scenario(self, rng: np.random.Generator, prefix: str = "train") -> FireScenario:
        branch = int(rng.integers(0, 2))
        growth = float(rng.uniform(0.010, 0.030))
        onset = int(rng.integers(0, 8))
        sid = f"{prefix}-b{branch}-g{growth:.4f}-o{onset}"
        return FireScenario(sid, branch, growth, onset)

    def values(self, scenario: FireScenario, step: int) -> np.ndarray:
        hazard = np.full(len(ZONE_NAMES), scenario.background_hazard, dtype=np.float64)
        active_time = max(0, step - scenario.onset_step)
        source_level = min(0.25, scenario.growth_rate * active_time)
        other_level = min(0.12, 0.35 * source_level)

        source_zone = 3 if scenario.source_branch == 0 else 4
        other_zone = 4 if scenario.source_branch == 0 else 3
        hazard[source_zone] += source_level
        hazard[other_zone] += other_level
        hazard[2] += 0.25 * source_level
        hazard[0] += 0.08 * source_level
        hazard[1] += 0.08 * source_level
        return np.clip(hazard, 0.0, 0.25)


class FDSCsvHazardProvider:
    """Adapter for a future FDS preprocessing pipeline.

    Expected long-form CSV columns are: scenario_id, step, zone, fed_increment.
    The adapter is implemented now so later fire-model integration does not
    require changes to the evacuation environment.
    """

    def __init__(self, csv_path: str | Path, require_complete_zones: bool = True):
        self._values: dict[tuple[str, int], np.ndarray] = {}
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"scenario_id", "step", "zone", "fed_increment"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"FDS CSV must contain columns: {sorted(required)}")
            grouped: dict[tuple[str, int], dict[str, float]] = {}
            for row in reader:
                scenario_id = row["scenario_id"].strip()
                if not scenario_id:
                    raise ValueError("scenario_id cannot be empty")
                step = int(row["step"])
                if step < 0:
                    raise ValueError("FDS step must be non-negative")
                zone = row["zone"].strip()
                if zone not in ZONE_NAMES:
                    raise ValueError(f"unknown FDS zone {zone!r}; expected one of {ZONE_NAMES}")
                fed_increment = float(row["fed_increment"])
                if not math.isfinite(fed_increment) or fed_increment < 0:
                    raise ValueError("fed_increment must be finite and non-negative")
                key = (scenario_id, step)
                if zone in grouped.setdefault(key, {}):
                    raise ValueError(f"duplicate FDS row for scenario={scenario_id!r}, step={step}, zone={zone!r}")
                grouped[key][zone] = fed_increment
        if not grouped:
            raise ValueError("FDS CSV contains no hazard records")
        for key, by_zone in grouped.items():
            missing_zones = set(ZONE_NAMES) - set(by_zone)
            if require_complete_zones and missing_zones:
                raise ValueError(f"missing FDS zones for scenario={key[0]!r}, step={key[1]}: {sorted(missing_zones)}")
            self._values[key] = np.array([by_zone.get(z, 0.0) for z in ZONE_NAMES], dtype=float)
        self.scenario_ids = tuple(sorted({scenario_id for scenario_id, _ in self._values}))
        self.max_step_by_scenario: dict[str, int] = {}
        for scenario_id in self.scenario_ids:
            steps = sorted(step for sid, step in self._values if sid == scenario_id)
            if steps != list(range(steps[-1] + 1)):
                raise ValueError(f"FDS steps for scenario={scenario_id!r} must be contiguous and start at zero")
            self.max_step_by_scenario[scenario_id] = steps[-1]

    def sample_scenario(self, rng: np.random.Generator, prefix: str = "train") -> FireScenario:
        del prefix
        scenario_id = str(rng.choice(self.scenario_ids))
        return FireScenario(scenario_id, source_branch=-1, growth_rate=0.0, onset_step=0, background_hazard=0.0)

    def values(self, scenario: FireScenario, step: int) -> np.ndarray:
        key = (scenario.scenario_id, step)
        if key not in self._values:
            maximum = self.max_step_by_scenario.get(scenario.scenario_id)
            raise KeyError(
                f"No FDS hazard row for scenario={scenario.scenario_id!r}, step={step}; available max step={maximum}"
            )
        return self._values[key].copy()

