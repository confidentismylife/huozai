"""Fixed goal reversal and equivalent-role encoding tests; no inference."""
import json

from llm_output_mechanism_core import OPTIONS, SCHEMA

CONDITIONS = tuple({'id': f'{objective}_{role}', 'objective': objective, 'role': role}
                   for role in ('hazard', 'clear') for objective in ('minimize', 'maximize'))
PREFIX = (
    'This is a synthetic numerical comparison test, not a real emergency or a routing instruction. '
    'Both exit routes are open and have equal distance and equal capacity. '
    'There are two equally occupied rooms and current measured smoke is zero in both corridors. '
    'During one upcoming traversal exactly one exit corridor has smoke exposure cost 1 '
    'and the other has cost 0. The selected label does not affect smoke. '
    'reported_location is a corridor named in a report, not a starting room or a destination instruction. '
    'report_role=hazard means the report names the corridor with cost 1; '
    'report_role=clear means the report names the corridor with cost 0. '
    'probability_report_correct is the calibrated probability that the role applies to reported_location. '
    'If the report is wrong, that role applies to the other corridor. '
    'Follow objective exactly: minimize asks for the smaller expected smoke exposure; '
    'maximize asks for the larger expected smoke exposure. The maximize task is only an inverted '
    'numerical comparison, not advice to enter smoke. '
    'For this test recommended_exit is the selected label under the specified objective. '
    'Use only the supplied information. Return ONLY one JSON object with exactly '
    'recommended_exit (north or south) and confidence (a number from 0 to 1). '
    'Do not add other fields. Data: '
)


def opposite(location):
    if location not in ('north', 'south'):
        raise ValueError('Unknown corridor')
    return 'south' if location == 'north' else 'north'


def decoded_costs(location, role, probability):
    if role not in ('hazard', 'clear'):
        raise ValueError('Unknown report role')
    if isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError('Invalid probability')
    other = opposite(location)
    at_location = probability if role == 'hazard' else 1-probability
    return {location: at_location, other: 1-at_location}


def choose(costs, objective):
    if costs['north'] == costs['south']:
        raise ValueError('No unique reference')
    if objective == 'minimize':
        return min(costs, key=costs.get)
    if objective == 'maximize':
        return max(costs, key=costs.get)
    raise ValueError('Unknown objective')


def regret(selected_cost, optimal_cost, objective):
    if objective == 'minimize':
        return selected_cost-optimal_cost
    if objective == 'maximize':
        return optimal_cost-selected_cost
    raise ValueError('Unknown objective')


def cases():
    result = []
    for i, (capacity, distance, occupancy, q) in enumerate(zip(
            (5, 9, 13, 17), (20, 28, 40, 52), (70, 110, 140, 170), (0.9, 0.6, 0.9, 0.6))):
        for anchor in ('north', 'south'):
            result.append({'case_id': f'GR{i:02d}_{anchor}', 'pair_id': f'GRP{i:02d}',
                'hazard_anchor': anchor, 'probability': q,
                'physical_inputs': {'both_routes_open': True,
                    'north_exit_capacity': capacity, 'south_exit_capacity': capacity,
                    'north_route_distance': distance, 'south_route_distance': distance,
                    'initial_room_occupancy': [occupancy, occupancy],
                    'measured_smoke_north': 0, 'measured_smoke_south': 0}})
    return result


def inputs(case, condition):
    role = condition['role']
    return {**case['physical_inputs'], 'reported_location': case['hazard_anchor'] if role == 'hazard' else opposite(case['hazard_anchor']),
        'report_role': role, 'probability_report_correct': case['probability'], 'objective': condition['objective']}


def payload(case, condition):
    return {'model': 'qwen3:4b', 'messages': [{'role': 'user', 'content': PREFIX+json.dumps(inputs(case, condition), sort_keys=True)}],
        'format': SCHEMA, 'options': OPTIONS.copy(), 'think': False, 'stream': False}


def private_reference(case, condition):
    q, anchor = case['probability'], case['hazard_anchor']
    costs = {anchor: q, opposite(anchor): 1-q}
    direction = choose(costs, condition['objective'])
    return {'reference_exit': direction, 'expected_cost_by_exit': costs,
        'optimal_expected_cost': costs[direction], 'objective': condition['objective']}
