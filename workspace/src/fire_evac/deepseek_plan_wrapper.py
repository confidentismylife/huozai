from __future__ import annotations

from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np

from .deepseek_advice import AdviceResult, EvacuationPlan, encode_plan_features, fallback_plan


PlanProvider = Callable[[Mapping[str, Any]], EvacuationPlan | AdviceResult]


class DeepSeekPlanObservationWrapper(gym.Wrapper):
    """Replace the five legacy advice flags with an executable 21-feature plan.

    The wrapped confirmatory environment keeps its physical 18 features, reopen
    estimate, nine raw forecast features and release memory.  DeepSeek supplies
    three route/capacity/time phases plus observable replanning triggers, giving
    a 50-dimensional state without duplicating the old direction hint.
    """

    def __init__(
        self,
        env: gym.Env,
        provider: PlanProvider,
        max_requests_per_episode: int = 3,
        congestion_threshold: float = 45 / 240,
        smoke_margin: float = 0.01,
        semantic_context: Mapping[str, Any] | None = None,
    ):
        super().__init__(env)
        if tuple(env.observation_space.shape) != (34,):
            raise ValueError("DeepSeek plan wrapper requires the 34-feature confirmatory environment")
        if max_requests_per_episode < 1:
            raise ValueError("max_requests_per_episode must be positive")
        self.provider = provider
        self.max_requests_per_episode = int(max_requests_per_episode)
        self.congestion_threshold = float(congestion_threshold)
        self.smoke_margin = float(smoke_margin)
        self.semantic_context = dict(semantic_context or {})
        self.current_plan: EvacuationPlan | None = None
        self.plan_source = "uninitialized"
        self.plan_age_steps = 0
        self.requests_this_episode = 0
        self._previous_blockage: bool | None = None
        self._previous_safer_branch: int | None = None
        self._previous_congested = False
        low = np.concatenate(
            [env.observation_space.low[:28], np.zeros(21, dtype=np.float32), env.observation_space.low[33:34]]
        )
        high = np.concatenate(
            [env.observation_space.high[:28], np.ones(21, dtype=np.float32), env.observation_space.high[33:34]]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def _safer_branch(self, observation: np.ndarray) -> int | None:
        north_hazard, south_hazard = float(observation[8]), float(observation[9])
        if abs(north_hazard - south_hazard) <= self.smoke_margin:
            return None
        return 0 if north_hazard < south_hazard else 1

    def _contradiction(self, observation: np.ndarray) -> bool:
        if self.current_plan is None:
            return False
        route = self.current_plan.phases[0].route
        if route == "wait":
            return False
        target = 0 if route == "north" else 1
        alternative = 1 - target
        return float(observation[8 + target]) > float(observation[8 + alternative]) + self.smoke_margin

    def _snapshot(self, observation: np.ndarray, info: Mapping[str, Any], event_type: str) -> dict[str, Any]:
        scenario = dict(info.get("scenario") or {})
        return {
            "scenario_id": str(scenario.get("scenario_id", "unknown")),
            "event_type": event_type,
            "timestamp_step": int(info.get("step", 0)),
            "forecast_source_branch": int(info.get("forecast_source_branch", scenario.get("forecast_source_branch", 0))),
            "growth_rate": float(scenario.get("growth_rate", 0.0)),
            "onset_step": int(scenario.get("onset_step", 0)),
            "occupancy": [float(value) for value in observation[0:5]],
            "smoke": [float(value) for value in observation[5:10]],
            "north_exit_capacity": int(info.get("north_exit_capacity", 0)),
            "south_exit_capacity": int(info.get("south_exit_capacity", 0)),
            "blockage_active": bool(info.get("blockage_active", False)),
            "blockage_steps_remaining": int(info.get("blockage_steps_remaining", 0)),
            "last_release_fraction": float(observation[33]),
            "semantic_context": self.semantic_context,
        }

    def _request_plan(self, observation: np.ndarray, info: Mapping[str, Any], event_type: str) -> None:
        if self.requests_this_episode >= self.max_requests_per_episode:
            return
        state = self._snapshot(observation, info, event_type)
        result = self.provider(state)
        if isinstance(result, AdviceResult):
            self.current_plan = result.plan
            self.plan_source = result.source
        elif isinstance(result, EvacuationPlan):
            self.current_plan = result
            self.plan_source = "provider"
        else:
            self.current_plan = fallback_plan(state)
            self.plan_source = "fallback_invalid_provider"
        self.requests_this_episode += 1
        self.plan_age_steps = 0

    def _encode(self, observation: np.ndarray) -> np.ndarray:
        if self.current_plan is None:
            raise RuntimeError("reset must initialize a DeepSeek plan")
        plan_features = encode_plan_features(
            self.current_plan,
            age_steps=self.plan_age_steps,
            contradiction=self._contradiction(observation),
        )
        return np.concatenate([observation[:28], plan_features, observation[33:34]]).astype(np.float32, copy=False)

    def _enrich_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(info)
        enriched.update(
            {
                "deepseek_plan_source": self.plan_source,
                "deepseek_plan_age_steps": self.plan_age_steps,
                "deepseek_requests_this_episode": self.requests_this_episode,
            }
        )
        return enriched

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.current_plan = None
        self.plan_source = "uninitialized"
        self.plan_age_steps = 0
        self.requests_this_episode = 0
        self._previous_blockage = bool(info.get("blockage_active", False))
        self._previous_safer_branch = self._safer_branch(observation)
        self._previous_congested = float(observation[2]) >= self.congestion_threshold
        self._request_plan(observation, info, "initial")
        return self._encode(observation), self._enrich_info(info)

    def _event(self, observation: np.ndarray, info: Mapping[str, Any]) -> str | None:
        if self.current_plan is None:
            return None
        triggers = set(self.current_plan.replan_triggers)
        blocked = bool(info.get("blockage_active", False))
        safer = self._safer_branch(observation)
        congested = float(observation[2]) >= self.congestion_threshold
        event: str | None = None
        if "blockage_change" in triggers and self._previous_blockage is not None and blocked != self._previous_blockage:
            event = "blockage_change"
        elif "smoke_reversal" in triggers and safer is not None:
            if self._previous_safer_branch is not None and safer != self._previous_safer_branch:
                event = "smoke_reversal"
            elif self._previous_safer_branch is None and self._contradiction(observation):
                event = "smoke_reversal"
        elif "congestion_threshold" in triggers and congested and not self._previous_congested:
            event = "congestion_threshold"
        self._previous_blockage = blocked
        if safer is not None:
            self._previous_safer_branch = safer
        self._previous_congested = congested
        return event

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.plan_age_steps += 1
        event = self._event(observation, info)
        if event is not None:
            self._request_plan(observation, info, event)
        return self._encode(observation), reward, terminated, truncated, self._enrich_info(info)
