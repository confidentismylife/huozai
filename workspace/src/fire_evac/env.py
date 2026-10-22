from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .behavior import BehaviorModel, DecisionContext, Evacuee, sample_population
from .config import EnvConfig
from .hazard import FireScenario, HazardProvider, SyntheticHazardProvider


class FireEvacuationEnv:
    """A small graph-based fire evacuation control environment.

    Zones 0-1 are rooms, zone 2 is a controlled junction, and zones 3-4 are
    north/south exit corridors. Actions are 0=north, 1=south, 2=neutral.
    """

    action_count = 3

    def __init__(
        self,
        behavior_model: BehaviorModel,
        config: EnvConfig | None = None,
        hazard_provider: HazardProvider | None = None,
    ):
        self.behavior_model = behavior_model
        self.config = config or EnvConfig()
        self.hazard_provider = hazard_provider or SyntheticHazardProvider()
        self.rng = np.random.default_rng()
        self.scenario: FireScenario | None = None
        self.people: list[Evacuee] = []
        self.zones: list[list[int]] = [[] for _ in range(5)]
        self.evacuated = 0
        self.incapacitated = 0
        self.step_index = 0
        self.last_action = 2
        self.steps_since_switch = self.config.min_sign_hold_steps
        self.last_branch_flow = np.zeros(2, dtype=float)
        self.cumulative_fed = 0.0
        self.cumulative_congestion = 0.0
        self.sign_switches = 0
        self.last_room_release_capacity = self.config.start_to_junction_capacity
        self.last_room_release_count = 0
        self.last_requested_action = 2
        self.last_applied_action = 2
        self.last_action_blocked_by_hold = False

    def reset(
        self,
        seed: int | None = None,
        scenario: FireScenario | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario or self.hazard_provider.sample_scenario(self.rng)
        reset_behavior = getattr(self.behavior_model, "reset_episode", None)
        if callable(reset_behavior):
            reset_behavior(self.rng)
        self.people = sample_population(self.config.population, self.rng)
        self.zones = [[] for _ in range(5)]
        ids = np.arange(self.config.population)
        self.rng.shuffle(ids)
        split = self.config.population // 2
        self.zones[0] = ids[:split].tolist()
        self.zones[1] = ids[split:].tolist()
        self.evacuated = 0
        self.incapacitated = 0
        self.step_index = 0
        self.last_action = 2
        self.steps_since_switch = self.config.min_sign_hold_steps
        self.last_branch_flow = np.zeros(2, dtype=float)
        self.cumulative_fed = 0.0
        self.cumulative_congestion = 0.0
        self.sign_switches = 0
        self.last_room_release_capacity = self.config.start_to_junction_capacity
        self.last_room_release_count = 0
        self.last_requested_action = 2
        self.last_applied_action = 2
        self.last_action_blocked_by_hold = False
        return self._observation(), self._info()

    def _active_count(self) -> int:
        return sum(len(z) for z in self.zones)

    def _hazard(self) -> np.ndarray:
        assert self.scenario is not None
        return self.hazard_provider.values(self.scenario, self.step_index)

    def _observation(self) -> np.ndarray:
        occupancy = np.array([len(z) for z in self.zones], dtype=np.float32)
        occupancy /= max(1.0, float(self.config.population))
        hazard = self._hazard().astype(np.float32)
        action_one_hot = np.zeros(3, dtype=np.float32)
        action_one_hot[self.last_action] = 1.0
        remaining = np.array([self._active_count() / self.config.population], dtype=np.float32)
        return np.concatenate([occupancy, hazard, action_one_hot, remaining])

    def _move_limited(self, source: int, target: int, capacity: int) -> int:
        count = min(capacity, len(self.zones[source]))
        if count <= 0:
            return 0
        selected = self.zones[source][:count]
        del self.zones[source][:count]
        self.zones[target].extend(selected)
        return count

    def _branch_to_exit_capacities(self) -> np.ndarray:
        """Per-branch downstream capacities for the current control step.

        Subclasses may override this hook for time-varying blockages or route
        length effects. The base simulator retains its original symmetric
        capacities.
        """
        return np.full(2, self.config.branch_to_exit_capacity, dtype=int)

    def _junction_to_branch_capacities(self) -> np.ndarray:
        """Per-branch junction capacities for the current control step."""
        return np.full(2, self.config.junction_to_branch_capacity, dtype=int)

    def _start_to_junction_capacity(self) -> int:
        """Number of people released from rooms during the current step."""
        return self.config.start_to_junction_capacity

    def _apply_fed(self, hazard: np.ndarray) -> tuple[int, float]:
        newly_incapacitated = 0
        fed_increment_sum = 0.0
        for zone_idx, ids in enumerate(self.zones):
            survivors: list[int] = []
            for person_id in ids:
                person = self.people[person_id]
                increment = float(hazard[zone_idx])
                person.fed += increment
                fed_increment_sum += increment
                if person.fed >= self.config.incapacitation_fed:
                    person.active = False
                    newly_incapacitated += 1
                else:
                    survivors.append(person_id)
            self.zones[zone_idx] = survivors
        self.incapacitated += newly_incapacitated
        self.cumulative_fed += fed_increment_sum
        return newly_incapacitated, fed_increment_sum

    def _enforce_sign_hold(self, requested_action: int) -> int:
        if requested_action not in (0, 1, 2):
            raise ValueError("action must be 0=north, 1=south, or 2=neutral")
        if requested_action != self.last_action and self.steps_since_switch < self.config.min_sign_hold_steps:
            return self.last_action
        return requested_action

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        requested_action = int(action)
        applied_action = self._enforce_sign_hold(requested_action)
        self.last_requested_action = requested_action
        self.last_applied_action = applied_action
        self.last_action_blocked_by_hold = bool(applied_action != requested_action)
        switched = int(applied_action != self.last_action)
        if switched:
            self.sign_switches += 1
            self.steps_since_switch = 0
        else:
            self.steps_since_switch += 1
        self.last_action = applied_action

        hazard = self._hazard()
        newly_incapacitated, fed_increment_sum = self._apply_fed(hazard)

        # People already in branch corridors leave first, creating downstream capacity.
        newly_evacuated = 0
        downstream_capacities = self._branch_to_exit_capacities()
        for branch_index, branch_zone in enumerate((3, 4)):
            count = min(int(downstream_capacities[branch_index]), len(self.zones[branch_zone]))
            if count > 0:
                del self.zones[branch_zone][:count]
                newly_evacuated += count
        self.evacuated += newly_evacuated

        # At the junction, behavior models choose a branch under local information.
        branch_occ = np.array([len(self.zones[3]), len(self.zones[4])], dtype=float)
        branch_occ /= self.config.density_scale
        context = DecisionContext(
            branch_hazard=hazard[[3, 4]],
            branch_occupancy=branch_occ,
            branch_flow_last_step=self.last_branch_flow.copy(),
            sign_action=applied_action,
        )
        waiting = self.zones[2]
        self.zones[2] = []
        capacity_left = self._junction_to_branch_capacities().astype(int, copy=True)
        flow = np.zeros(2, dtype=int)
        still_waiting: list[int] = []
        for person_id in waiting:
            person = self.people[person_id]
            choice = self.behavior_model.choose_branch(person, context, self.rng)
            if capacity_left[choice] > 0:
                self.zones[3 + choice].append(person_id)
                capacity_left[choice] -= 1
                flow[choice] += 1
            else:
                still_waiting.append(person_id)
        self.zones[2].extend(still_waiting)
        self.last_branch_flow = flow.astype(float)

        # Rooms feed the junction after route decisions for this control interval.
        total_room_capacity = self._start_to_junction_capacity()
        room_capacity = total_room_capacity // 2
        released = self._move_limited(0, 2, room_capacity)
        released += self._move_limited(1, 2, total_room_capacity - room_capacity)
        self.last_room_release_capacity = total_room_capacity
        self.last_room_release_count = released

        congestion = sum(max(0.0, len(z) / self.config.density_scale - 1.0) for z in self.zones)
        self.cumulative_congestion += congestion

        n = float(self.config.population)
        reward = (
            self.config.reward_exit * newly_evacuated / n
            - self.config.reward_incapacitation * newly_incapacitated / n
            - self.config.reward_fed * fed_increment_sum / n
            - self.config.reward_congestion * congestion
            - self.config.reward_switch * switched
        )

        self.step_index += 1
        terminated = self._active_count() == 0
        truncated = self.step_index >= self.config.max_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        scenario = asdict(self.scenario) if self.scenario is not None else None
        return {
            "scenario": scenario,
            "behavior_model": self.behavior_model.name,
            "active_behavior_model": getattr(self.behavior_model, "active_model_name", self.behavior_model.name),
            "step": self.step_index,
            "evacuated": self.evacuated,
            "incapacitated": self.incapacitated,
            "remaining": self._active_count(),
            "evacuation_rate": self.evacuated / self.config.population,
            "incapacitation_rate": self.incapacitated / self.config.population,
            "mean_cumulative_fed": self.cumulative_fed / self.config.population,
            "cumulative_congestion": self.cumulative_congestion,
            "sign_switches": self.sign_switches,
            "room_release_capacity": self.last_room_release_capacity,
            "room_release_count": self.last_room_release_count,
            "last_action": self.last_action,
            "requested_sign_action": self.last_requested_action,
            "actual_sign_action": self.last_applied_action,
            "action_blocked_by_sign_hold": self.last_action_blocked_by_hold,
            "zone_counts": [len(z) for z in self.zones],
        }

