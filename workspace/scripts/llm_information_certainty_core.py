"""Fixed one-step probability problems; no model calls or simulations."""
import json

from llm_output_mechanism_core import SCHEMA, OPTIONS

CONDITIONS = (
    {'id': 'confirmed_100', 'information_type': 'confirmed', 'probability': 1.0},
    {'id': 'forecast_100', 'information_type': 'forecast', 'probability': 1.0},
    {'id': 'forecast_90', 'information_type': 'forecast', 'probability': 0.9},
    {'id': 'forecast_60', 'information_type': 'forecast', 'probability': 0.6},
)
PREFIX = (
    'This is a synthetic one-step research test, not a real emergency. '
    'People start in two rooms with equal occupancy. Both exit routes are open '
    'and have equal distance and equal capacity. Current measured smoke is zero in both corridors. '
    'During the upcoming traversal exactly one exit corridor will have smoke exposure cost 1 '
    'and the other will have cost 0. Choosing an exit does not change which corridor has smoke. '
    'hazard_location is the reported dangerous corridor, not a starting room or a recommended destination. '
    'information_type is either a confirmed report or a forecast. '
    'probability_report_correct is the calibrated probability that the dangerous corridor '
    'is hazard_location. If the report is wrong, the dangerous corridor is the other corridor. '
    'A forecast with probability 1 is an idealized mathematical case. '
    'Choose one initial evacuation exit to minimize expected smoke exposure using only the supplied information. '
    'recommended_exit means the evacuation destination. Return ONLY one JSON object '
    'with exactly recommended_exit (north or south) and confidence (a number from 0 to 1). '
    'Do not add other fields. Data: '
)


def expected_exposure(location, probability, exit_value):
    if location not in ('north', 'south') or exit_value not in ('north', 'south'):
        raise ValueError('Unknown direction')
    if isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError('Invalid report probability')
    return probability if exit_value == location else 1 - probability


def reference(location, probability):
    north = expected_exposure(location, probability, 'north')
    south = expected_exposure(location, probability, 'south')
    if north == south:
        raise ValueError('No unique reference at probability 0.5')
    return 'north' if north < south else 'south'


def cases():
    result = []
    for family, (capacity, distance, occupancy) in enumerate(zip((4, 8, 12, 16), (18, 24, 36, 48), (60, 90, 150, 180))):
        for location in ('north', 'south'):
            result.append({'case_id': f'IC{family:02d}_{location}', 'pair_id': f'ICP{family:02d}',
                'inputs': {'hazard_location': location, 'both_routes_open': True,
                    'north_exit_capacity': capacity, 'south_exit_capacity': capacity,
                    'north_route_distance': distance, 'south_route_distance': distance,
                    'initial_room_occupancy': [occupancy, occupancy],
                    'measured_smoke_north': 0, 'measured_smoke_south': 0}})
    return result


def inputs(case, condition):
    return {**case['inputs'], 'information_type': condition['information_type'],
            'probability_report_correct': condition['probability']}


def payload(case, condition):
    text = PREFIX + json.dumps(inputs(case, condition), sort_keys=True)
    return {'model': 'qwen3:4b', 'messages': [{'role': 'user', 'content': text}],
            'options': OPTIONS.copy(), 'format': SCHEMA, 'think': False, 'stream': False}


def private_reference(case, condition):
    location, q = case['inputs']['hazard_location'], condition['probability']
    costs = {direction: expected_exposure(location, q, direction) for direction in ('north', 'south')}
    return {'reference_exit': reference(location, q), 'expected_cost_by_exit': costs,
            'minimum_expected_cost': min(costs.values())}
