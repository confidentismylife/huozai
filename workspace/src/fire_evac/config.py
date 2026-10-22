from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the graph-based minimum viable simulator."""

    population: int = 240
    dt_seconds: float = 5.0
    max_steps: int = 120
    start_to_junction_capacity: int = 20
    junction_to_branch_capacity: int = 12
    branch_to_exit_capacity: int = 8
    density_scale: float = 45.0
    incapacitation_fed: float = 1.0

    reward_exit: float = 1.0
    reward_incapacitation: float = 20.0
    reward_fed: float = 0.2
    reward_congestion: float = 0.1
    reward_switch: float = 0.01

    min_sign_hold_steps: int = 2

