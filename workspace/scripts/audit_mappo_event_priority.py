"""Reproduce a frozen-protocol trigger limitation without changing training."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper


def observed_event(triggers):
    # Use the real event method with precisely its required instance fields.
    wrapper = object.__new__(DeepSeekPlanObservationWrapper)
    wrapper.current_plan = EvacuationPlan(
        (PlanPhase('north', .5, 3),) * 3, .7, ('balance_congestion',), triggers)
    wrapper.smoke_margin = .01
    wrapper.congestion_threshold = 45 / 240
    wrapper._previous_blockage = False
    wrapper._previous_safer_branch = 0
    wrapper._previous_congested = False
    observation = np.zeros(34, dtype=np.float32)
    observation[2] = 46 / 240  # Junction occupancy crosses the threshold.
    observation[9] = .03     # North remains safer; no smoke reversal.
    return wrapper._event(observation, {'blockage_active': False})


def main():
    combined = observed_event(('smoke_reversal', 'congestion_threshold'))
    congestion_only = observed_event(('congestion_threshold',))
    assert combined is None and congestion_only == 'congestion_threshold'
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_method': 'DeepSeekPlanObservationWrapper._event',
        'case': 'unchanged safer north and newly congested junction, no blockage change',
        'junction_occupants': 46, 'threshold_occupants': 45,
        'with_smoke_and_congestion_triggers': combined,
        'with_congestion_trigger_only': congestion_only,
        'suppression_reproduced': True,
        'scope': 'Both original and ActivePhase wrappers inherit the frozen event method.',
        'not_established': 'This constructed case does not quantify frequency or endpoint effect in existing runs.',
        'training_modified': False,
        'decision': 'Disclose limitation; fix only in a separate future frozen protocol, never relabel current runs.'}
    target = ROOT / 'outputs/mappo_six_hour_extension_20261022/event_trigger_diagnostic.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
