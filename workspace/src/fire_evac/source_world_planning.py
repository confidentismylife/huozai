"""Development reference planning over observable-compatible source hypotheses."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np

from .confirmatory import encode_absolute_action
from .hazard import SyntheticHazardProvider


@dataclass(frozen=True)
class DynamicReferencePlan:
    name: str
    kind: str
    wait_limit: int = 0
    target_fraction: float = .5
    split_release: float = 1.0


PLANS = (
    DynamicReferencePlan("live_graded", "live"),
    DynamicReferencePlan("live_full", "full"),
    DynamicReferencePlan("wait1_then_live", "wait", wait_limit=1),
    DynamicReferencePlan("wait3_then_live", "wait", wait_limit=3),
    DynamicReferencePlan("wait6_then_live", "wait", wait_limit=6),
    DynamicReferencePlan("fixed_forecast_full", "forecast"),
) + tuple(DynamicReferencePlan(f"split{int(fraction * 100)}_release{int(release * 100)}", "split", target_fraction=fraction, split_release=release)
          for fraction in (.25, .5, .75) for release in (.5, 1.0))


def source_world_scenarios(template):
    """Uses forecast/map parameters only; the template's actual source is ignored."""
    return tuple(replace(template, source_branch=source) for source in (0, 1))


def compatible_source_worlds(hypotheses, observation, step, tolerance=1e-6):
    """Exact smoke compatibility for the explicitly noiseless development model.

    Assumes forecast growth/onset/background/map are correct; only source can
    be wrong. Does not use source labels, individual FED, or conflict-type labels.
    If neither hypothesis fits, retain both and flag model mismatch.
    """
    sensed = np.asarray(observation)[5:10]
    provider = SyntheticHazardProvider()
    errors = np.asarray([np.max(np.abs(provider.values(s, step) - sensed)) for s in hypotheses])
    supported = errors <= tolerance
    mismatch = not np.any(supported)
    if mismatch:
        supported[:] = True
    return supported, errors, mismatch


def smoke_source_visible(observation, tolerance=1e-6):
    return bool(abs(float(observation[8]) - float(observation[9])) > tolerance)


def live_action(observation, full_release=False):
    x = np.asarray(observation)
    capacity = x[14:16]
    if np.all(capacity <= 1e-6):
        return 8
    cost = 6 * x[8:10] + 1.5 * x[3:5] + 2 * (capacity <= 1e-6)
    # Directional ties use the forecast's opposite corridor, preserving reflection.
    target = 1 - int(np.argmax(x[19:21]))
    branch = target if abs(float(cost[0] - cost[1])) <= 1e-8 else int(np.argmin(cost))
    if full_release:
        release = 1.0
    elif capacity[branch] < .45:
        release = .25
    elif x[3 + branch] > .75:
        release = .5
    elif x[3 + branch] > .45:
        release = .75
    else:
        release = 1.0
    return encode_absolute_action(branch, release)


def plan_action(plan, observation, elapsed, minimum_sign_hold=2):
    if elapsed < 0:
        raise ValueError("elapsed must be nonnegative")
    visible = smoke_source_visible(observation)
    target = 1 - int(np.argmax(np.asarray(observation)[19:21]))
    if plan.kind == "forecast":
        return encode_absolute_action(target, 1.0)
    if plan.kind == "wait" and elapsed < plan.wait_limit and not visible:
        return 8
    if plan.kind == "split" and not visible:
        # Time-division signs, not direct control of each person's route.
        block = max(1, minimum_sign_hold + 1)
        segment = (elapsed // block) % 4
        branch = target if segment < int(round(plan.target_fraction * 4)) else 1 - target
        if observation[14 + branch] <= 1e-6:
            return live_action(observation)
        return encode_absolute_action(branch, plan.split_release)
    return live_action(observation, full_release=plan.kind == "full")


@dataclass(frozen=True)
class PlanOutcome:
    final_incapacitation: float
    final_fed: float
    final_evacuation: float
    final_remaining: float
    additional_congestion: float
    steps: int
    mean_release: float
    wait_requests: int
    hold_block_rate: float
    first_requested_action: int
    first_actual_joint_action: int


OUTCOME_FIELDS = tuple(PlanOutcome.__dataclass_fields__)


def evaluate_plan(env, observation, plan, future_seed):
    """Clone a historically consistent world; follow a full closed-loop plan."""
    clone = copy.deepcopy(env)
    clone.unwrapped.domain_env.rng = np.random.default_rng(future_seed)
    layer = clone
    while isinstance(layer, gym.Wrapper):
        if "rng" in vars(layer):
            layer.rng = np.random.default_rng(future_seed + 9_000_001)
        layer = layer.env
    config = clone.unwrapped.domain_env.config
    start_info = clone.unwrapped.domain_env._info()
    x = np.asarray(observation).copy()
    total_release = waits = hold_blocks = 0
    elapsed = 0
    term = trunc = False
    first_requested = first_actual = -1
    while not (term or trunc):
        action = plan_action(plan, x, elapsed, config.min_sign_hold_steps)
        x, _, term, trunc, info = clone.step(action)
        if elapsed == 0:
            first_requested = action
            first_actual = int(info["actual_joint_action"])
        total_release += float(info["actual_release_fraction"])
        waits += action == 8
        hold_blocks += bool(info["action_blocked_by_sign_hold"])
        elapsed += 1
        if elapsed > config.max_steps:
            raise RuntimeError("reference plan exceeded episode boundary")
    outcome = PlanOutcome(float(info["incapacitation_rate"]), float(info["mean_cumulative_fed"]),
                          float(info["evacuation_rate"]), float(info["remaining"] / config.population),
                          float(info["cumulative_congestion"] - start_info["cumulative_congestion"]),
                          elapsed, total_release / elapsed, int(waits), hold_blocks / elapsed,
                          first_requested, first_actual)
    clone.close()
    return outcome


def select_reference_plan(risks, supported, forecast_source, mode, true_source=None):
    """Lexicographic finite-library reference, not a global optimum or bound.

    risks: [source, plan, future, metric]. Mean over futures in each world;
    robust uses the maximum world mean, mean uses equal world weights.
    Oracle is explicitly privileged and is for diagnostic comparison only.
    """
    risks = np.asarray(risks)
    supported = np.asarray(supported, dtype=bool)
    if risks.ndim != 4 or risks.shape[0] != 2 or supported.shape != (2,) or not supported.any():
        raise ValueError("two-source plan risks and a nonempty support required")
    sources = np.flatnonzero(supported)
    if mode == "point":
        sources = np.asarray([forecast_source]) if supported[forecast_source] else sources
    elif mode == "oracle":
        if true_source not in (0, 1):
            raise ValueError("oracle requires a true source")
        sources = np.asarray([true_source])
    elif mode not in ("mean", "robust"):
        raise ValueError("unknown planning mode")
    means = np.mean(risks[sources], axis=2)
    if not np.all(np.isfinite(means)):
        raise ValueError("selected world risks are missing")
    average = means.mean(axis=0)
    worst = means.max(axis=0)
    primary = worst if mode == "robust" else average
    order = sorted(range(risks.shape[1]), key=lambda p: (
        primary[p, 0], average[p, 0], primary[p, 1], average[p, 1],
        primary[p, 3], primary[p, 4], p,
    ))
    return order[0]
