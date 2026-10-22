"""Frozen prompts and paired-input factorials; no inference at import."""
import json
import math
import numpy as np

OLD_PREFIX = 'You advise a research evacuation simulator, not a real emergency. Branch codes: 0=north, 1=south, -1=none. Choose north or south using only the forecast, initial equal room occupancy120 each, equal initial smoke at background_hazard, and map capacities. Forecast source can be wrong. Capacity is persons per control step. Fire growth begins after onset_step. Consider smoke risk, throughput and blockage. Return JSON target north/south and confidence0..1. Do not invent observations. Data: '
NEW_PREFIX = (
    'You advise a research evacuation simulator, not a real emergency. '
    'Choose the recommended evacuation EXIT: north or south. '
    'predicted_fire_location means the corridor where fire and smoke are forecast to originate. '
    'It describes a hazard location, not where the people start and not a recommended destination. '
    'recommended_exit means the exit toward which evacuees should initially be guided. '
    'The forecast can be wrong. Initial room occupancy is120 in each room; initial smoke is equal at background_hazard. '
    'Branch codes:0=north,1=south,-1=none. Capacity is persons per control step. '
    'Fire growth begins after onset_step. Consider smoke risk, throughput and blockage using only the supplied information. '
    'Return JSON recommended_exit north/south and confidence0..1. Do not invent observations. Data: '
)
PHYSICAL_KEYS = ('growth_rate', 'onset_step', 'background_hazard', 'north_exit_capacity',
                 'south_exit_capacity', 'blocked_branch', 'blockage_end_step')
FACTORIAL_ARMS = ('north_c50', 'north_c95', 'south_c50', 'south_c95')
DIAGNOSTIC_ARMS = FACTORIAL_ARMS + ('none', 'repaired_native')
FRESH_ARMS = ('none', 'rule_native', 'original_native', 'repaired_native')


def prompt(scene, version):
    data = {key: scene[key] for key in PHYSICAL_KEYS}
    if version == 'original':
        data['forecast_source'] = ('north', 'south')[scene['forecast_source_branch']]
        return OLD_PREFIX + json.dumps(data, sort_keys=True)
    if version == 'repaired':
        data['predicted_fire_location'] = ('north', 'south')[scene['forecast_source_branch']]
        return NEW_PREFIX + json.dumps(data, sort_keys=True)
    raise ValueError(version)


def schema(version):
    key = 'target' if version == 'original' else 'recommended_exit'
    return {'type': 'object', 'properties': {key: {'type': 'string', 'enum': ['north', 'south']},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
            'required': [key, 'confidence'], 'additionalProperties': False}


def parse(raw, version):
    value = json.loads(raw)
    key = 'target' if version == 'original' else 'recommended_exit'
    if set(value) != {key, 'confidence'} or value[key] not in ('north', 'south'):
        raise ValueError('advice schema mismatch')
    confidence = value['confidence']
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('invalid confidence')
    return {'target_branch': ('north', 'south').index(value[key]), 'confidence': float(confidence)}


def probes():
    result = []
    for pair, (capacity, growth, background) in enumerate(
            ((6, .020, .001), (8, .020, .003), (10, .020, .001),
             (6, .035, .003), (8, .035, .001), (10, .035, .003))):
        for fire in (0, 1):
            result.append({'scenario_id': f'SCP{pair:02d}{fire}', 'source_branch': fire,
                'forecast_source_branch': fire, 'growth_rate': growth, 'onset_step': 0,
                'background_hazard': background, 'north_exit_capacity': capacity, 'south_exit_capacity': capacity,
                'blocked_branch': -1, 'blockage_start_step': 0, 'blockage_end_step': 0,
                'blockage_capacity': 0, 'conflict_type': 'semantic_probe'})
    return result


def fixed_advice(arm):
    if arm not in FACTORIAL_ARMS:
        raise ValueError(arm)
    return {'target_branch': 0 if arm.startswith('north') else 1,
            'confidence': .50 if arm.endswith('c50') else .95}


def feature_advice(observation, advice):
    """Only intended input changes; contradiction follows observed smoke."""
    x = np.asarray(observation, dtype=np.float32).copy()
    if x.shape != (49,):
        raise ValueError('raw49 required')
    if advice is None:
        x[28:33] = 0
    else:
        target = advice['target_branch']
        if target not in (0, 1):
            raise ValueError('invalid target')
        x[28:33] = [target == 0, target == 1, advice['confidence'], 1,
                    x[8 + target] > x[8 + (1 - target)] + .01]
    return x


def stratified_ci(family_effects, kinds, seed=2026100571):
    values, kinds = np.asarray(family_effects, float), np.asarray(kinds)
    rng = np.random.default_rng(seed)
    samples = np.zeros(2000)
    for kind in sorted(set(kinds)):
        v = values[kinds == kind]
        samples += v[rng.integers(len(v), size=(2000, len(v)))].sum(axis=1)
    return np.quantile(samples / len(values), [.025, .975])
