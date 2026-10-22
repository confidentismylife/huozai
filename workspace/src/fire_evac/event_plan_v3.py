"""Corrected event protocol, separate from the frozen six-hour experiment."""
from typing import Any, Mapping

import numpy as np

from .campaign_extension import ActivePhasePlanWrapper


class EventCorrectedPlanWrapper(ActivePhasePlanWrapper):
    """Test actual event changes before applying single-request priority.

    Priority remains blockage, smoke reversal, then congestion. Stable smoke
    cannot consume that priority. The inherited episode request cap is unchanged.
    Legacy wrappers remain available for exact historical replay.
    """

    def _event(self, observation: np.ndarray, info: Mapping[str, Any]) -> str | None:
        if self.current_plan is None:
            return None
        triggers = set(self.current_plan.replan_triggers)
        blocked = bool(info.get('blockage_active', False))
        safer = self._safer_branch(observation)
        congested = float(observation[2]) >= self.congestion_threshold
        blockage_changed = (self._previous_blockage is not None
                            and blocked != self._previous_blockage)
        smoke_reversed = safer is not None and (
            (self._previous_safer_branch is not None and safer != self._previous_safer_branch)
            or (self._previous_safer_branch is None and self._contradiction(observation)))
        congestion_crossed = congested and not self._previous_congested
        event = None
        if 'blockage_change' in triggers and blockage_changed:
            event = 'blockage_change'
        elif 'smoke_reversal' in triggers and smoke_reversed:
            event = 'smoke_reversal'
        elif 'congestion_threshold' in triggers and congestion_crossed:
            event = 'congestion_threshold'
        self._previous_blockage = blocked
        if safer is not None:
            self._previous_safer_branch = safer
        self._previous_congested = congested
        return event
