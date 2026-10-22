"""Fixed comprehension cases and outcome classification, no inference."""
import json
import math
import re

SCHEMA = {'type': 'object', 'properties': {
    'recommended_exit': {'type': 'string', 'enum': ['north', 'south']},
    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
    'required': ['recommended_exit', 'confidence'], 'additionalProperties': False}
OPTIONS = {'temperature': 0, 'seed': 2026100601, 'num_predict': 1024, 'num_ctx': 4096}
CONDITIONS = ({'id': 'schema_on_think_off', 'schema': True, 'think': False},
              {'id': 'schema_off_think_off', 'schema': False, 'think': False},
              {'id': 'schema_on_think_on', 'schema': True, 'think': True},
              {'id': 'schema_off_think_on', 'schema': False, 'think': True})
PREFIX = (
    'This is a synthetic research comprehension test, not a real emergency. '
    'People start in two rooms. An active fire has been CONFIRMED in one exit corridor; '
    'that corridor is hazardous. The other exit corridor is confirmed clear of smoke. '
    'Both routes are open, have equal distance and equal capacity. '
    'Choose the initial evacuation exit to minimize smoke exposure. '
    'confirmed_fire_location describes the hazard location, not the people starting location '
    'and not a recommended destination. recommended_exit means the evacuation destination. '
    'Return ONLY one JSON object with exactly recommended_exit (north or south) and '
    'confidence (a number from0 to1). Do not add other fields. Data: '
)


def cases():
    result = []
    for pair, capacity in enumerate((7, 9, 11, 13)):
        for fire in ('north', 'south'):
            result.append({'case_id': f'OM{pair:02d}_{fire}', 'pair_id': f'OMP{pair:02d}',
                           'inputs': {'confirmed_fire_location': fire, 'fire_active': True,
                                      'other_exit_corridor_clear_of_smoke': True, 'both_routes_open': True,
                                      'route_distances_equal': True, 'north_exit_capacity': capacity,
                                      'south_exit_capacity': capacity, 'initial_room_occupancy': [120, 120]},
                           'reference_exit': 'south' if fire == 'north' else 'north'})
    return result


def prompt(case):
    return PREFIX + json.dumps(case['inputs'], sort_keys=True)


def payload(case, condition):
    value = {'model': 'qwen3:4b', 'messages': [{'role': 'user', 'content': prompt(case)}],
             'options': OPTIONS.copy(), 'think': condition['think'], 'stream': False}
    if condition['schema']:
        value['format'] = SCHEMA
    return value


def unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = child
    return value


def finite_confidence(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def grade(response, reference):
    content = response.get('message', {}).get('content', '')
    if not isinstance(content, str):
        content = ''
    thinking = response.get('message', {}).get('thinking', '')
    strict = False
    direction = None
    confidence = None
    method = 'none'
    parse_error = None
    try:
        value = json.loads(content, object_pairs_hook=unique_object)
        strict = (isinstance(value, dict) and set(value) == {'recommended_exit', 'confidence'}
                  and value['recommended_exit'] in ('north', 'south') and finite_confidence(value['confidence']))
        if isinstance(value, dict):
            exit_value = value.get('recommended_exit')
            if isinstance(exit_value, str) and exit_value.lower() in ('north', 'south'):
                direction, method = exit_value.lower(), 'json_field'
            if finite_confidence(value.get('confidence')):
                confidence = float(value['confidence'])
    except (ValueError, TypeError) as exc:
        parse_error = str(exc)
    if direction is None:
        fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', content, flags=re.IGNORECASE)
        if fenced:
            try:
                value = json.loads(fenced.group(1), object_pairs_hook=unique_object)
                candidate = value.get('recommended_exit') if isinstance(value, dict) else None
                if isinstance(candidate, str) and candidate.lower() in ('north', 'south'):
                    direction, method = candidate.lower(), 'fenced_json_field'
            except (ValueError, TypeError):
                pass
    if direction is None:
        matches = {v.lower() for v in re.findall(r'"recommended_exit"\s*:\s*"(north|south)"', content, flags=re.IGNORECASE)}
        if len(matches) == 1:
            direction, method = next(iter(matches)).lower(), 'unique_named_field'
    truncated = response.get('done_reason') == 'length'
    correct = None if direction is None else direction == reference
    if truncated:
        category = 'truncated'
    elif strict:
        category = 'usable_correct' if correct else 'valid_wrong_direction'
    else:
        category = 'invalid_format'
    return {'category': category, 'strict_format_valid': bool(strict), 'semantic_direction': direction,
            'semantic_correct': correct, 'semantic_extraction': method, 'confidence': confidence,
            'truncated': truncated, 'usable_correct': bool(strict and correct and not truncated),
            'thinking_present': isinstance(thinking, str) and bool(thinking.strip()),
            'thinking_characters': len(thinking) if isinstance(thinking, str) else 0,
            'final_content_characters': len(content), 'done_reason': response.get('done_reason'),
            'prompt_eval_tokens': response.get('prompt_eval_count'), 'generated_tokens': response.get('eval_count'),
            'parse_error': parse_error}
