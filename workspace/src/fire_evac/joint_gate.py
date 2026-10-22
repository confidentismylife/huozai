from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .dynamic_conflicts import JOINT_RELEASE_LEVELS


WAIT_ACTION = 8


def encode_joint_action(route: str, release_fraction: float) -> int:
    if route not in ("trust", "override"):
        raise ValueError("route must be trust or override")
    try:
        level = JOINT_RELEASE_LEVELS.index(float(release_fraction))
    except ValueError as exc:
        raise ValueError(f"unsupported release fraction: {release_fraction}") from exc
    return level if route == "trust" else 4 + level


@dataclass(frozen=True)
class JointGatePlan:
    name: str
    family: str
    route: str = "trust"
    release_fraction: float = 1.0
    wait_steps: int = 0
    override_at_step: int | None = None
    reactive_capacity: bool = False

    def __post_init__(self) -> None:
        if self.route not in ("trust", "override"):
            raise ValueError("route must be trust or override")
        if self.release_fraction not in JOINT_RELEASE_LEVELS:
            raise ValueError("unsupported release fraction")
        if self.wait_steps < 0:
            raise ValueError("wait_steps must be non-negative")


def make_joint_gate_plans() -> tuple[JointGatePlan, ...]:
    plans: list[JointGatePlan] = []
    for route in ("trust", "override"):
        for release in JOINT_RELEASE_LEVELS:
            plans.append(
                JointGatePlan(
                    f"{route}_release_{int(release * 100)}",
                    f"constant_{route}",
                    route=route,
                    release_fraction=release,
                )
            )
    for wait_steps in (2, 4):
        for release in JOINT_RELEASE_LEVELS:
            plans.append(
                JointGatePlan(
                    f"wait_{wait_steps}_trust_release_{int(release * 100)}",
                    "wait_then_trust",
                    release_fraction=release,
                    wait_steps=wait_steps,
                )
            )
    for wait_steps in (2, 4):
        plans.append(
            JointGatePlan(
                f"wait_{wait_steps}_override_full",
                "wait_then_override",
                route="override",
                wait_steps=wait_steps,
            )
        )
    for step in (2, 4, 8):
        plans.append(
            JointGatePlan(
                f"trust_then_override_{step}",
                "timed_override",
                override_at_step=step,
            )
        )
    for release in (0.25, 0.5, 0.75):
        plans.append(
            JointGatePlan(
                f"reactive_capacity_{int(release * 100)}",
                "reactive_evidence",
                release_fraction=release,
                reactive_capacity=True,
            )
        )
    return tuple(plans)


JOINT_GATE_PLANS = make_joint_gate_plans()


def joint_plan_action(
    plan: JointGatePlan,
    *,
    step: int,
    blockage_steps_remaining: int,
    smoke_contradiction: bool,
    target_capacity_difference: float,
) -> int:
    if step < 0 or blockage_steps_remaining < 0:
        raise ValueError("step and blockage remaining must be non-negative")
    if plan.reactive_capacity:
        if smoke_contradiction:
            return encode_joint_action("override", 1.0)
        if blockage_steps_remaining > 0:
            return WAIT_ACTION
        release = plan.release_fraction if target_capacity_difference < -0.05 else 1.0
        return encode_joint_action("trust", release)
    if blockage_steps_remaining > 0 and step < plan.wait_steps:
        return WAIT_ACTION
    if plan.override_at_step is not None and step >= plan.override_at_step:
        return encode_joint_action("override", 1.0)
    return encode_joint_action(plan.route, plan.release_fraction)


def select_safest_joint_plan(
    results: Mapping[str, Mapping[str, float]],
    plans: Sequence[JointGatePlan] = JOINT_GATE_PLANS,
) -> JointGatePlan:
    names = [plan.name for plan in plans]
    if not names or len(names) != len(set(names)) or set(results) != set(names):
        raise ValueError("results must match a non-empty unique plan bank")
    required = {"incapacitation_rate", "mean_cumulative_fed", "total_reward"}
    for plan in plans:
        if not required.issubset(results[plan.name]):
            raise ValueError(f"missing metrics for {plan.name}")
    return min(
        plans,
        key=lambda plan: (
            float(results[plan.name]["incapacitation_rate"]),
            float(results[plan.name]["mean_cumulative_fed"]),
            -float(results[plan.name]["total_reward"]),
            names.index(plan.name),
        ),
    )
