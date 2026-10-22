"""Independent identity-grid and immutable-training-input audit."""
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mappo_six_hour_extension as runner
from fire_evac.campaign_extension import write_json


def verify_grid(frame, variants, scenarios, populations, trace=None):
    columns = ['provider_variant', 'scenario_id', 'population_seed']
    assert not frame.duplicated(columns).any(), 'Duplicate episode identity'
    actual = set(frame[columns].itertuples(index=False, name=None))
    expected = set(product(variants, scenarios, populations))
    assert actual == expected, 'Incomplete identity grid'
    steps = 0
    if trace is not None:
        lengths = frame.set_index(columns).evacuation_time_seconds / 5
        assert np.equal(lengths, np.floor(lengths)).all()
        observed = trace.groupby(columns).size().reindex(lengths.index)
        assert np.array_equal(observed.to_numpy(), lengths.to_numpy()), 'Per-episode trace length mismatch'
        for _, values in trace.groupby(columns, sort=False):
            assert np.array_equal(values.decision_step.to_numpy(), np.arange(len(values))), 'Step gaps or duplicates'
        assert len(trace) == int(lengths.sum())
        steps = len(trace)
    return dict(episodes=len(frame), steps=steps, exact_unique_grid=True)


def main():
    scenarios = [s.scenario_id for s in
        runner.base.make_domain_randomized_trust_bank(24, 20261021, 'DSG')[12:]]
    variants = ('native', 'rule', 'shuffled', 'masked')
    audited = []
    original = pd.read_csv(runner.SOURCE / 'channel_audit.csv')
    for seed, grouped in original.groupby('training_seed'):
        audited.append(dict(stage='original', seed=int(seed),
            **verify_grid(grouped, variants, scenarios, (42001, 42002))))
    paths = sorted((runner.OUT / 'mechanism_audit').glob('*/seed_*/*/episode_results.csv'))
    assert len(paths) == 20
    for path in paths:
        values = pd.read_csv(path)
        trace = pd.read_csv(path.parent / 'step_results.csv')
        audited.append(dict(stage='new_mechanism', path=str(path),
            **verify_grid(values, variants, scenarios, (42001, 42002), trace)))
    online = sorted((runner.OUT / 'online_pilot').glob('*/episode_results.csv'))
    assert len(online) == 4
    for path in online:
        audited.append(dict(stage='online', path=str(path),
            **verify_grid(pd.read_csv(path), variants + ('online',), scenarios[:6], (42003,),
                pd.read_csv(path.parent / 'step_results.csv'))))
    latency = pd.read_csv(runner.OUT / 'online_latency_pilot/episode_results.csv')
    audited.append(dict(stage='timing', **verify_grid(latency, variants + (
        'instant_unchecked', 'instant_checked', 'delayed_unchecked', 'delayed_checked'),
        scenarios[:6], (42004,))))
    frozen = json.loads((runner.OUT / 'training_runtime_and_source_manifest.json').read_text(encoding='utf-8'))
    for name, expected_hash in frozen['source_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name
    result = dict(passed=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
        grids=audited, immutable_training_inputs_checked=len(frozen['source_files']),
        all_training_inputs_unchanged=True, final_test_scenarios_loaded=False,
        scope='Complements finalizer verification with exact identity and per-episode step grids.')
    write_json(runner.OUT / 'independent_grid_and_source_audit.json', result)
    print(json.dumps(dict(passed=True, grids=len(audited),
        episodes=sum(x['episodes'] for x in audited),
        immutable_inputs=len(frozen['source_files']))))


if __name__ == '__main__':
    main()
