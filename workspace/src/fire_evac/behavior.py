from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass
class Evacuee:
    evacuee_id: int
    familiarity_branch: int
    compliance: float
    herding: float
    risk_sensitivity: float
    congestion_sensitivity: float
    previous_choice: int | None = None
    fed: float = 0.0
    active: bool = True


@dataclass(frozen=True)
class DecisionContext:
    branch_hazard: np.ndarray
    branch_occupancy: np.ndarray
    branch_flow_last_step: np.ndarray
    sign_action: int  # 0 north, 1 south, 2 neutral


class BehaviorModel(Protocol):
    name: str

    def choose_branch(
        self,
        evacuee: Evacuee,
        context: DecisionContext,
        rng: np.random.Generator,
    ) -> int: ...


def sample_population(n: int, rng: np.random.Generator) -> list[Evacuee]:
    """Sample heterogeneous personas used by all behavior models.

    These distributions are placeholders. They must later be estimated from
    the development split of real human data.
    """

    population: list[Evacuee] = []
    for i in range(n):
        population.append(
            Evacuee(
                evacuee_id=i,
                familiarity_branch=int(rng.integers(0, 2)),
                compliance=float(rng.beta(4.0, 2.5)),
                herding=float(rng.beta(2.5, 3.0)),
                risk_sensitivity=float(rng.lognormal(mean=0.0, sigma=0.25)),
                congestion_sensitivity=float(rng.lognormal(mean=-0.15, sigma=0.25)),
            )
        )
    return population


class RuleBehavior:
    name = "rule"

    def choose_branch(self, evacuee: Evacuee, context: DecisionContext, rng: np.random.Generator) -> int:
        del rng
        cost = 3.0 * context.branch_hazard + 0.7 * context.branch_occupancy
        if context.sign_action in (0, 1):
            cost[context.sign_action] -= 0.35 * evacuee.compliance
        cost[evacuee.familiarity_branch] -= 0.20
        return int(np.argmin(cost))


class MixedLogitBehavior:
    name = "mixed_logit"

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    def choose_branch(self, evacuee: Evacuee, context: DecisionContext, rng: np.random.Generator) -> int:
        utility = np.zeros(2, dtype=float)
        utility -= 3.2 * evacuee.risk_sensitivity * context.branch_hazard
        utility -= 0.9 * evacuee.congestion_sensitivity * context.branch_occupancy
        utility[evacuee.familiarity_branch] += 0.45
        if context.sign_action in (0, 1):
            utility[context.sign_action] += 0.95 * evacuee.compliance
        majority = int(np.argmax(context.branch_flow_last_step))
        if context.branch_flow_last_step.sum() > 0:
            utility[majority] += 0.55 * evacuee.herding
        logits = utility / self.temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(rng.choice(2, p=probs))


class LLMSurrogateBehavior:
    """Placeholder for a model distilled from frozen LLM responses.

    The nonlinear interactions mimic the shape of a possible learned student
    policy, but the coefficients below are synthetic. Research use requires
    replacing them with a fitted artifact and recording its provenance.
    """

    name = "llm_surrogate_unvalidated"

    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature

    def choose_branch(self, evacuee: Evacuee, context: DecisionContext, rng: np.random.Generator) -> int:
        utility = np.zeros(2, dtype=float)
        risk = np.sqrt(np.maximum(context.branch_hazard, 0.0))
        crowd = np.tanh(2.0 * context.branch_occupancy)
        utility -= 2.7 * evacuee.risk_sensitivity * risk
        utility -= 0.8 * evacuee.congestion_sensitivity * crowd
        utility[evacuee.familiarity_branch] += 0.55 + 0.25 * (1.0 - evacuee.compliance)

        if context.sign_action in (0, 1):
            sign_bonus = 1.15 * evacuee.compliance
            if context.branch_hazard[context.sign_action] > 0.10:
                sign_bonus *= 0.25
            utility[context.sign_action] += sign_bonus

        if context.branch_flow_last_step.sum() > 0:
            majority = int(np.argmax(context.branch_flow_last_step))
            utility[majority] += 0.85 * evacuee.herding

        if evacuee.previous_choice is not None:
            utility[evacuee.previous_choice] += 0.35

        logits = utility / self.temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        choice = int(rng.choice(2, p=probs))
        evacuee.previous_choice = choice
        return choice


class CalibratedLogitBehavior:
    """Behavior policy loaded from a participant-level calibration artifact.

    Unseen simulated evacuees use the population-level coefficients. Training
    participant random effects are deliberately not transferred into RL tests.
    """

    name = "calibrated_logit"

    def __init__(self, artifact_path: str | Path, temperature: float = 1.0):
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        expected = [
            "intercept",
            "relative_safety_north",
            "relative_space_north",
            "sign_points_north",
            "familiarity_north",
            "crowd_moves_north",
        ]
        if payload.get("feature_names") != expected:
            raise ValueError("calibration artifact uses an incompatible feature definition")
        self.coefficients = np.asarray(payload["coefficients"], dtype=float)
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        self.feature_scale = np.asarray(payload["feature_scale"], dtype=float)
        if self.coefficients.shape != (6,) or self.feature_mean.shape != (6,) or self.feature_scale.shape != (6,):
            raise ValueError("calibration artifact must contain six coefficients and scaling values")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def choose_branch(self, evacuee: Evacuee, context: DecisionContext, rng: np.random.Generator) -> int:
        total_flow = float(context.branch_flow_last_step.sum())
        features = np.array(
            [
                1.0,
                context.branch_hazard[1] - context.branch_hazard[0],
                context.branch_occupancy[1] - context.branch_occupancy[0],
                1.0 if context.sign_action == 0 else -1.0 if context.sign_action == 1 else 0.0,
                1.0 if evacuee.familiarity_branch == 0 else -1.0,
                (context.branch_flow_last_step[0] - context.branch_flow_last_step[1]) / max(1.0, total_flow),
            ],
            dtype=float,
        )
        features[1:] = (features[1:] - self.feature_mean[1:]) / self.feature_scale[1:]
        logit_north = float(features @ self.coefficients) / self.temperature
        probability_north = 1.0 / (1.0 + np.exp(-np.clip(logit_north, -30.0, 30.0)))
        return 0 if rng.random() < probability_north else 1


class EpisodeBehaviorEnsemble:
    """Randomize the active behavior model once per episode."""

    name = "behavior_ensemble"

    def __init__(self, factories: Sequence[Callable[[], BehaviorModel]], weights: Sequence[float] | None = None):
        if not factories:
            raise ValueError("behavior ensemble requires at least one member")
        self.factories = tuple(factories)
        if weights is None:
            self.weights = np.full(len(factories), 1.0 / len(factories), dtype=float)
        else:
            parsed = np.asarray(weights, dtype=float)
            if parsed.shape != (len(factories),) or np.any(parsed < 0) or parsed.sum() <= 0:
                raise ValueError("ensemble weights must be non-negative and match the member count")
            self.weights = parsed / parsed.sum()
        self.active_model = self.factories[0]()
        self.active_model_name = self.active_model.name

    def reset_episode(self, rng: np.random.Generator) -> None:
        index = int(rng.choice(len(self.factories), p=self.weights))
        self.active_model = self.factories[index]()
        self.active_model_name = self.active_model.name

    def choose_branch(self, evacuee: Evacuee, context: DecisionContext, rng: np.random.Generator) -> int:
        return self.active_model.choose_branch(evacuee, context, rng)

