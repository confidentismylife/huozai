from __future__ import annotations

import numpy as np


class StaticNorthController:
    name = "static_north"

    def act(self, observation: np.ndarray) -> int:
        del observation
        return 0


class NeutralController:
    name = "neutral"

    def act(self, observation: np.ndarray) -> int:
        del observation
        return 2


class HazardAwareController:
    name = "hazard_aware_rule"

    def act(self, observation: np.ndarray) -> int:
        # observation = 5 occupancies + 5 hazards + 3 action bits + remaining
        occupancy = observation[:5]
        hazard = observation[5:10]
        north_cost = 3.0 * hazard[3] + occupancy[3]
        south_cost = 3.0 * hazard[4] + occupancy[4]
        return int(north_cost > south_cost)

