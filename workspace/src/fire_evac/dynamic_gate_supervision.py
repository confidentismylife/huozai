from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DynamicGatePlan:
    """A compact schedule that emits trust, override, or wait gate actions.

    trust_fraction is implemented as two-step sign slots in an eight-step cycle,
    matching the simulator's minimum sign hold time while approximating a route
    split over time.
    """

    name: str
    family: str
    wait_steps: int = 0
    override_at_step: int | None = None
    trust_fraction: float = 1.0
    release_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.wait_steps < 0:
            raise ValueError("wait_steps must be non-negative")
        if self.override_at_step is not None and self.override_at_step < 0:
            raise ValueError("override_at_step must be non-negative")
        if self.trust_fraction not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("trust_fraction must use a supported quarter-step split")
        if self.release_fraction not in (0.25, 0.5, 0.75, 1.0):
            raise ValueError("release_fraction must use a supported positive quarter-step split")
        if self.release_fraction < 1.0 and self.trust_fraction != 1.0:
            raise ValueError("throttling cannot be combined with trust/override splitting")


def make_dynamic_gate_plans() -> tuple[DynamicGatePlan, ...]:
    plans: list[DynamicGatePlan] = [
        DynamicGatePlan("trust", "trust"),
        DynamicGatePlan("override", "override", trust_fraction=0.0),
    ]
    plans.extend(
        DynamicGatePlan(
            f"trust_then_override_{step}",
            "timed_override",
            override_at_step=step,
        )
        for step in (2, 4, 8)
    )
    plans.extend(
        DynamicGatePlan(f"wait_{step}_then_trust", "wait_then_trust", wait_steps=step)
        for step in (2, 4, 8, 12)
    )
    plans.extend(
        DynamicGatePlan(
            f"wait_{step}_then_override",
            "wait_then_override",
            wait_steps=step,
            trust_fraction=0.0,
        )
        for step in (2, 4, 8, 12)
    )
    plans.extend(
        DynamicGatePlan(
            f"split_trust_{int(fraction * 100)}",
            "split",
            trust_fraction=fraction,
        )
        for fraction in (0.25, 0.5, 0.75)
    )
    plans.extend(
        DynamicGatePlan(
            f"wait_{step}_split_trust_{int(fraction * 100)}",
            "wait_then_split",
            wait_steps=step,
            trust_fraction=fraction,
        )
        for step in (2, 4)
        for fraction in (0.25, 0.5, 0.75)
    )
    plans.extend(
        DynamicGatePlan(
            f"throttle_release_{int(fraction * 100)}",
            "throttle",
            release_fraction=fraction,
        )
        for fraction in (0.25, 0.5, 0.75)
    )
    plans.extend(
        DynamicGatePlan(
            f"wait_{step}_throttle_release_{int(fraction * 100)}",
            "wait_then_throttle",
            wait_steps=step,
            release_fraction=fraction,
        )
        for step in (2,)
        for fraction in (0.25, 0.5, 0.75)
    )
    return tuple(plans)


DYNAMIC_GATE_PLANS = make_dynamic_gate_plans()


def dynamic_plan_action(
    plan: DynamicGatePlan,
    *,
    step: int,
    blockage_steps_remaining: int,
) -> int:
    """Return 0=trust, 1=override, 2=wait for one control step."""
    if step < 0 or blockage_steps_remaining < 0:
        raise ValueError("step and blockage_steps_remaining must be non-negative")
    if blockage_steps_remaining > 0 and step < plan.wait_steps:
        return 2
    if plan.override_at_step is not None and step >= plan.override_at_step:
        return 1
    if plan.release_fraction < 1.0:
        slot = (step // 2) % 4
        release_slots = int(round(plan.release_fraction * 4))
        return 0 if slot < release_slots else 2
    if plan.trust_fraction == 1.0:
        return 0
    if plan.trust_fraction == 0.0:
        return 1
    # Four two-step sign slots give an eight-step cycle and respect the
    # simulator's two-step minimum sign hold time.
    slot = (step // 2) % 4
    trust_slots = int(round(plan.trust_fraction * 4))
    return 0 if slot < trust_slots else 1


def select_safest_dynamic_plan(
    results: Mapping[str, Mapping[str, float]],
    plans: Sequence[DynamicGatePlan] = DYNAMIC_GATE_PLANS,
) -> DynamicGatePlan:
    """Choose minimum incapacity, then FED, then maximum reward and simplicity."""
    if not plans:
        raise ValueError("at least one plan is required")
    plan_names = [plan.name for plan in plans]
    if len(plan_names) != len(set(plan_names)):
        raise ValueError("plan names must be unique")
    if set(results) != set(plan_names):
        raise ValueError("results must match the supplied plans exactly")
    required = {"incapacitation_rate", "mean_cumulative_fed", "total_reward"}
    for plan in plans:
        if not required.issubset(results[plan.name]):
            raise ValueError(f"missing counterfactual metrics for {plan.name}")
    return min(
        plans,
        key=lambda plan: (
            float(results[plan.name]["incapacitation_rate"]),
            float(results[plan.name]["mean_cumulative_fed"]),
            -float(results[plan.name]["total_reward"]),
            plan_names.index(plan.name),
        ),
    )
