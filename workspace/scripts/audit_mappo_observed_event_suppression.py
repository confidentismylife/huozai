"""Observe suppressed congestion events while exactly replaying saved evaluations.

This never repairs the trigger, changes actions, trains, or calls an API.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mappo_six_hour_extension as runner
from fire_evac.campaign_extension import write_json
from fire_evac.mappo import FactorizedMAPPO


def main():
    paths = sorted((runner.SOURCE / 'confirmation').glob('seed_*/*retrieval/summary.json'))
    paths += sorted((runner.OUT / 'mechanism').glob('e003_*/seed_*/*deepseek*/summary.json'))
    assert len(paths) == 29, 'Require all frozen 9 original and 20 new LLM models.'
    scenarios = runner.base.make_domain_randomized_trust_bank(24, 20261021, 'DSG')[12:]
    metrics = ['incapacitation_rate', 'mean_cumulative_fed', 'evacuation_time_seconds',
               'cumulative_congestion', 'wait_fraction', 'mean_release_fraction_nonwait']
    rows = []
    maximum_replay_error = 0.
    for path in paths:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        arm = metadata['arm']
        original = 'confirmation' in path.parts
        protocol = 'original' if original else path.parents[2].name
        reference = pd.read_csv(path.parent / 'episode_results.csv').set_index(
            ['scenario_id', 'population_seed'])
        model, _ = FactorizedMAPPO.load(path.parent / 'best.pt')
        env = (runner.ORIGINAL_FACTORY if original else runner.factory)(arm, scenarios)
        original_event = env._event
        counters = {}

        def observe_event(observation, info):
            triggers = set(env.current_plan.replan_triggers)
            congested = float(observation[2]) >= env.congestion_threshold
            crossing = ('congestion_threshold' in triggers and congested
                        and not env._previous_congested)
            under_cap = env.requests_this_episode < env.max_requests_per_episode
            event = original_event(observation, info)
            counters['steps'] += 1
            counters['congestion_crossings_with_trigger'] += int(crossing)
            suppressed = crossing and event is None and 'smoke_reversal' in triggers
            # The real method has updated history; its returned None is the key.
            suppressed = suppressed and env._safer_branch(observation) is not None
            counters['suppressed_crossings'] += int(suppressed)
            counters['suppressed_under_request_cap'] += int(suppressed and under_cap)
            counters['returned_congestion_events'] += int(event == 'congestion_threshold')
            counters['maximum_junction_occupants'] = max(
                counters['maximum_junction_occupants'], float(observation[2]) * 240)
            return event

        env._event = observe_event
        for scenario in scenarios:
            for population in (42001, 42002):
                counters.clear()
                counters.update(steps=0, congestion_crossings_with_trigger=0,
                    suppressed_crossings=0, suppressed_under_request_cap=0,
                    returned_congestion_events=0, maximum_junction_occupants=0.)
                evaluated = runner.base.evaluate(model, env, [scenario], (population,), arm)[0]
                expected = reference.loc[(scenario.scenario_id, population)]
                actual_values = np.asarray([evaluated[m] for m in metrics], dtype=float)
                expected_values = expected[metrics].to_numpy(dtype=float)
                np.testing.assert_allclose(actual_values, expected_values, rtol=1e-10, atol=1e-12)
                maximum_replay_error = max(maximum_replay_error,
                    float(np.abs(actual_values - expected_values).max()))
                rows.append(dict(protocol=protocol, arm=arm, training_seed=metadata['seed'],
                    scenario_id=scenario.scenario_id, population_seed=population, **counters))
        env.close()
    data = pd.DataFrame(rows)
    target = runner.OUT / 'event_suppression_replay'
    target.mkdir(exist_ok=True)
    data.to_csv(target / 'episode_diagnostics.csv', index=False, encoding='utf-8-sig')
    summary = data.groupby(['protocol', 'arm'], as_index=False).agg(
        episodes=('scenario_id', 'count'), control_steps=('steps', 'sum'),
        crossings=('congestion_crossings_with_trigger', 'sum'),
        suppressed=('suppressed_crossings', 'sum'),
        suppressed_under_cap=('suppressed_under_request_cap', 'sum'),
        returned_congestion=('returned_congestion_events', 'sum'),
        episodes_with_suppression=('suppressed_crossings', lambda v: int((v > 0).sum())))
    summary.to_csv(target / 'summary.csv', index=False, encoding='utf-8-sig')
    result = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        model_count=len(paths), episodes=len(data), API_requests=0, training_modified=False,
        all_replayed_endpoints_match=True, maximum_replay_error=maximum_replay_error,
        maximum_junction_occupants=float(data.maximum_junction_occupants.max()),
        groups=summary.to_dict(orient='records'),
        interpretation='Observed frequency only; no repaired-trigger counterfactual effect is estimated.',
        final_test_scenarios_loaded=False)
    write_json(target / 'audit.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
