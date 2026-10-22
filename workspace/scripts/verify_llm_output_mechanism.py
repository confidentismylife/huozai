"""Reconstruct format, direction, truncation, paired counts and provenance."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_output_mechanism_20261006'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def object_unique(pairs):
    if len(pairs) != len({p[0] for p in pairs}):
        raise ValueError('duplicate key')
    return dict(pairs)


def classify(response, expected):
    content = response.get('message', {}).get('content')
    content = content if isinstance(content, str) else ''
    final = None
    is_strict = False
    try:
        value = json.loads(content, object_pairs_hook=object_unique)
        if isinstance(value, dict):
            confidence = value.get('confidence')
            exit_value = value.get('recommended_exit')
            valid_confidence = type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
            is_strict = set(value) == {'recommended_exit', 'confidence'} and exit_value in ('north', 'south') and valid_confidence
            if isinstance(exit_value, str) and exit_value.lower() in ('north', 'south'):
                final = exit_value.lower()
    except (ValueError, TypeError):
        pass
    if final is None:
        fence = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', content, re.IGNORECASE)
        if fence:
            try:
                value = json.loads(fence.group(1), object_pairs_hook=object_unique)
                exit_value = value.get('recommended_exit') if isinstance(value, dict) else None
                if isinstance(exit_value, str) and exit_value.lower() in ('north', 'south'):
                    final = exit_value.lower()
            except (ValueError, TypeError):
                pass
    if final is None:
        candidates = {m.lower() for m in re.findall('"recommended_exit"\\s*:\\s*"(north|south)"', content, re.IGNORECASE)}
        if len(candidates) == 1:
            final = candidates.pop()
    truncated = response.get('done_reason') == 'length'
    correct = None if final is None else final == expected
    reasoning = response.get('message', {}).get('thinking')
    category = 'truncated' if truncated else ('usable_correct' if is_strict and correct else 'valid_wrong_direction' if is_strict else 'invalid_format')
    return {'strict_format_valid': bool(is_strict), 'semantic_direction': final, 'semantic_correct': correct,
            'truncated': truncated, 'category': category, 'usable_correct': bool(is_strict and correct and not truncated),
            'thinking_present': isinstance(reasoning, str) and bool(reasoning.strip()),
            'generated_tokens': response.get('eval_count')}


def verify():
    manifest, budget = read(OUT / 'manifest.json'), read(OUT / 'budget.json')
    cases = {s['case_id']: s for s in manifest['cases']}
    assert len(cases) == 8 and len({s['pair_id'] for s in cases.values()}) == 4
    assert len(manifest['request_order']) == 32
    for name, digest in manifest['frozen_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in manifest['parent_completion_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(OUT / 'model_service_snapshot.json') == manifest['model_snapshot_sha256']
    assert sha(ROOT / 'outputs/third_round_development_status.json') == manifest['old_global_status_sha256']
    request_ids, verified, bodies_by_case = set(), {}, {}
    raw_responses_checked = 0
    for request in manifest['request_order']:
        request_id, body = request['request_id'], request['payload']
        assert request_id not in request_ids
        request_ids.add(request_id)
        assert payload_sha(body) == request['payload_sha256']
        condition = next(c for c in manifest['conditions'] if c['id'] == request['condition_id'])
        assert body['think'] == condition['think'] and ('format' in body) == condition['schema']
        assert body['options'] == {'temperature': 0, 'seed': 2026100601, 'num_predict': 1024, 'num_ctx': 4096}
        assert body['stream'] is False and body['model'] == 'qwen3:4b'
        assert len(body['messages']) == 1 and body['messages'][0]['role'] == 'user'
        normalized = {k: v for k, v in body.items() if k not in ('format', 'think')}
        if request['case_id'] in bodies_by_case:
            assert bodies_by_case[request['case_id']] == normalized
        else:
            bodies_by_case[request['case_id']] = normalized
        case = cases[request['case_id']]
        input_json = json.loads(body['messages'][0]['content'].split('Data: ')[1])
        assert input_json == case['inputs'] and 'reference_exit' not in input_json
        expected = 'south' if case['inputs']['confirmed_fire_location'] == 'north' else 'north'
        assert case['reference_exit'] == expected
        result = read(OUT / 'results' / f'{request_id}.json')
        assert result['payload_sha256'] == request['payload_sha256']
        if result['raw_path'] is not None:
            raw = read(ROOT / result['raw_path'])
            assert raw['payload'] == body and raw['payload_sha256'] == request['payload_sha256']
            assert not raw.get('service_error') and raw['response']['done'] is True
            assert raw['response']['model'] == 'qwen3:4b'
            independent = classify(raw['response'], expected)
            for key, value in independent.items():
                assert value == result['outcome'][key], (request_id, key)
            assert raw['latency_seconds'] == result['latency_seconds']
            verified[(request['case_id'], request['condition_id'])] = (independent, result['latency_seconds'])
            raw_responses_checked += 1
            assert not (OUT / 'attempts' / f"{request_id}_attempt{result['attempt'] + 1}.started.json").exists()
        else:
            assert result['outcome']['category'] == 'transport_failure'
            assert all((OUT / 'attempts' / f'{request_id}_attempt{a}.started.json').exists() for a in (1, 2))
            verified[(request['case_id'], request['condition_id'])] = (result['outcome'], None)
    assert len(verified) == 32 and len(list((OUT / 'results').glob('*.json'))) == 32
    result_summary = read(OUT / 'summary.json')
    group_passes = []
    for cond in manifest['conditions']:
        pairs = [verified[(s['case_id'], cond['id'])] for s in cases.values()]
        outcomes = [p[0] for p in pairs]
        published = result_summary['groups'][cond['id']]
        for category in ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure'):
            assert published[category] == sum(o['category'] == category for o in outcomes)
        assert sum(published[k] for k in ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure')) == 8
        for field in ('strict_format_valid', 'thinking_present'):
            assert published[field] == sum(o[field] for o in outcomes)
        assert published['semantic_correct_nontruncated'] == sum(o.get('semantic_correct') is True and not o['truncated'] for o in outcomes)
        assert published['generated_tokens'] == sum(o.get('generated_tokens') or 0 for o in outcomes)
        latencies = [p[1] for p in pairs if p[1] is not None]
        if latencies:
            assert abs(published['mean_response_latency_seconds'] - statistics.mean(latencies)) < 1e-9
        swapped = 0
        for pair in sorted({s['pair_id'] for s in cases.values()}):
            a, b = [s for s in cases.values() if s['pair_id'] == pair]
            aa, bb = verified[(a['case_id'], cond['id'])][0]['semantic_direction'], verified[(b['case_id'], cond['id'])][0]['semantic_direction']
            swapped += aa is not None and bb is not None and aa != bb
        assert published['mirror_pairs_swapped'] == swapped
        passed = sum(o['usable_correct'] for o in outcomes) == 8 and swapped == 4
        assert published['basic_function_passed'] == passed
        group_passes.append(passed)
    assert result_summary['all_settings_pass'] == all(group_passes)
    assert result_summary['any_setting_pass'] == any(group_passes)
    comparisons = [('schema_effect_think_off', 'schema_off_think_off', 'schema_on_think_off'),
                   ('schema_effect_think_on', 'schema_off_think_on', 'schema_on_think_on'),
                   ('thinking_effect_schema_off', 'schema_off_think_off', 'schema_off_think_on'),
                   ('thinking_effect_schema_on', 'schema_on_think_off', 'schema_on_think_on')]
    for name, a, b in comparisons:
        published = next(r for r in result_summary['paired_comparisons'] if r['comparison'] == name)
        delta = [int(verified[(s['case_id'], b)][0]['usable_correct']) - int(verified[(s['case_id'], a)][0]['usable_correct']) for s in cases.values()]
        assert published['b_minus_a_correct_cases'] == sum(delta)
        assert published['improved_cases'] == delta.count(1) and published['worsened_cases'] == delta.count(-1)
    markers = list((OUT / 'attempts').glob('*.started.json'))
    assert len(markers) == budget['api_attempts_started'] <= 64
    assert budget['successful_http_responses'] == raw_responses_checked
    assert budget['logical_requests_completed'] == 32
    assert budget['simulations'] == budget['ppo_updates'] == budget['fits'] == 0
    unused = ROOT / 'outputs/llm_semantic_channel_diagnostic_20261005'
    assert not (unused / 'fresh_actual').exists() and not (unused / 'fresh_advice_frozen.json').exists()
    return {'all_checks_passed': True, 'logical_requests_rechecked': 32, 'raw_responses_reclassified': raw_responses_checked,
            'four_cell_summary_reconstructed': True, 'paired_setting_effects_reconstructed': True,
            'prompts_and_options_identical_except_two_treatments': True, 'thinking_not_used_to_infer_final_answer': True,
            'direction_format_truncation_separated': True, 'prior_source_and_status_hashes_unchanged': True,
            'reserved48_scenarios_untouched': True, 'new_inference_for_verification': 0,
            'new_simulations': 0, 'new_ppo_updates': 0, 'new_fits': 0, 'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(verify(), ensure_ascii=False))
