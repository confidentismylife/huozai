"""Independent raw-response and probability audit; no inference or simulation."""
from pathlib import Path
import csv
import hashlib
import json
import random
import statistics

from verify_llm_output_mechanism import classify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_information_certainty_20261006'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def verify():
    m, budget, summary = read(OUT/'manifest.json'), read(OUT/'budget.json'), read(OUT/'summary.json')
    assert len(m['cases']) == 8 and len(m['request_order']) == 32
    expected_conditions = {'confirmed_100': ('confirmed', 1.0), 'forecast_100': ('forecast', 1.0),
                           'forecast_90': ('forecast', 0.9), 'forecast_60': ('forecast', 0.6)}
    assert {c['id']: (c['information_type'], c['probability']) for c in m['conditions']} == expected_conditions
    for name, digest in m['frozen_sha256'].items():
        assert sha(ROOT/name) == digest, name
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    parents = {}
    for name, digest in m['parent_completion_sha256'].items():
        path = ROOT/name
        assert sha(path) == digest, name
        parents[name] = ledger_check(path, HISTORY if path.parent == MAIN else None)
        assert parents[name] == m['parent_checks'][name]
    assert sha(OUT/'model_service_snapshot.json') == m['model_snapshot_sha256']
    assert sha(OUT/'pre_inference_checks.json') == m['pre_inference_checks_sha256']
    assert read(OUT/'pre_inference_checks.json')['all_checks_passed']
    assert sha(ROOT/'outputs/third_round_development_status.json') == m['old_global_status_sha256']
    cases = {s['case_id']: s for s in m['cases']}
    families = sorted({s['pair_id'] for s in m['cases']})
    assert len(families) == 4
    for pair in families:
        a, b = [s['inputs'] for s in m['cases'] if s['pair_id'] == pair]
        assert {a['hazard_location'], b['hazard_location']} == {'north', 'south'}
        assert {k: v for k, v in a.items() if k != 'hazard_location'} == {k: v for k, v in b.items() if k != 'hazard_location'}
    rng = random.Random(2026100603)
    rng.shuffle(families)
    planned_order = []
    for pair in families:
        block = [(s, c) for s in m['cases'] if s['pair_id'] == pair for c in m['conditions']]
        rng.shuffle(block)
        planned_order += [f"{s['case_id']}_{c['id']}" for s, c in block]
    assert planned_order == [r['request_id'] for r in m['request_order']]
    rows, bodies, prefixes, raw_count = {}, {}, set(), 0
    for req in m['request_order']:
        body, case = req['payload'], cases[req['case_id']]
        assert hash_value(body) == req['payload_sha256']
        assert body['model'] == 'qwen3:4b' and body['think'] is False and body['stream'] is False
        assert body['format'] == {'type': 'object', 'properties': {
            'recommended_exit': {'type': 'string', 'enum': ['north', 'south']},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
            'required': ['recommended_exit', 'confidence'], 'additionalProperties': False}
        assert body['options'] == {'temperature': 0, 'seed': 2026100601, 'num_predict': 1024, 'num_ctx': 4096}
        assert len(body['messages']) == 1 and body['messages'][0]['role'] == 'user'
        normalized = {k: v for k, v in body.items() if k != 'messages'}
        if bodies:
            assert normalized == next(iter(bodies.values()))
        bodies[req['request_id']] = normalized
        prefix, data_text = body['messages'][0]['content'].split('Data: ')
        prefixes.add(prefix)
        data = json.loads(data_text)
        kind, q = expected_conditions[req['condition_id']]
        assert data == {**case['inputs'], 'information_type': kind, 'probability_report_correct': q}
        assert not {'reference_exit', 'minimum_expected_cost', 'expected_cost_by_exit'} & set(data)
        assert data['north_exit_capacity'] == data['south_exit_capacity']
        assert data['north_route_distance'] == data['south_route_distance']
        assert data['initial_room_occupancy'][0] == data['initial_room_occupancy'][1]
        assert data['measured_smoke_north'] == data['measured_smoke_south'] == 0
        posterior = {'north': q if data['hazard_location'] == 'north' else 1-q,
                     'south': q if data['hazard_location'] == 'south' else 1-q}
        expected = min(posterior, key=posterior.get)
        ref = {'reference_exit': expected, 'expected_cost_by_exit': posterior, 'minimum_expected_cost': min(posterior.values())}
        assert req['private_reference'] == ref and q > 0.5
        result = read(OUT/'results'/f"{req['request_id']}.json")
        assert result['payload_sha256'] == req['payload_sha256']
        if result['raw_path'] is not None:
            raw = read(ROOT/result['raw_path'])
            assert raw['payload'] == body and raw['payload_sha256'] == req['payload_sha256']
            assert not raw.get('service_error') and raw['response']['done'] is True
            assert raw['response']['model'] == 'qwen3:4b'
            o = classify(raw['response'], expected)
            assert all(result['outcome'][k] == v for k, v in o.items())
            assert raw['latency_seconds'] == result['latency_seconds']
            assert not (OUT/'attempts'/f"{req['request_id']}_attempt{result['attempt']+1}.started.json").exists()
            raw_count += 1
        else:
            o = result['outcome']
            assert o['category'] == 'transport_failure'
            for attempt in (1, 2):
                assert (OUT/'attempts'/f"{req['request_id']}_attempt{attempt}.started.json").exists()
                assert (OUT/'transport_failures'/f"{req['request_id']}_attempt{attempt}.json").exists()
        valid = o['strict_format_valid'] and not o['truncated'] and o.get('semantic_direction') in ('north', 'south')
        cost = posterior[o['semantic_direction']] if valid else None
        rows[(req['case_id'], req['condition_id'])] = {'outcome': o, 'cost': cost,
            'regret': cost-min(posterior.values()) if cost is not None else None,
            'latency': result['latency_seconds'], 'request_id': req['request_id'], 'reference': expected}
    assert len(prefixes) == 1 and len(rows) == 32
    with (OUT/'case_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = {r['request_id']: r for r in csv.DictReader(stream)}
    assert len(table) == 32
    for value in rows.values():
        row = table[value['request_id']]
        assert row['category'] == value['outcome']['category'] and row['reference_exit'] == value['reference']
        for column, field in [('selected_expected_cost_if_valid', 'cost'), ('expected_regret_if_valid', 'regret')]:
            assert (row[column] == '') if value[field] is None else abs(float(row[column])-value[field]) < 1e-12
    passed = []
    for cid in expected_conditions:
        values = [v for (case, cond), v in rows.items() if cond == cid]
        g = summary['groups'][cid]
        for category in ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure'):
            assert g[category] == sum(v['outcome']['category'] == category for v in values)
        swaps = 0
        for pair in families:
            a, b = [s for s in m['cases'] if s['pair_id'] == pair]
            da = rows[(a['case_id'], cid)]['outcome']['semantic_direction']
            db = rows[(b['case_id'], cid)]['outcome']['semantic_direction']
            swaps += da is not None and db is not None and da != db
        assert g['mirror_pairs_swapped'] == swaps
        basic = sum(v['outcome']['usable_correct'] for v in values) == 8 and swaps == 4
        assert g['basic_function_passed'] == basic
        passed.append(basic)
        assert g['thinking_present'] == sum(v['outcome']['thinking_present'] for v in values)
        assert g['generated_tokens'] == sum(v['outcome'].get('generated_tokens') or 0 for v in values)
        costs = [v['cost'] for v in values if v['cost'] is not None]
        regrets = [v['regret'] for v in values if v['regret'] is not None]
        times = [v['latency'] for v in values if v['latency'] is not None]
        assert g['valid_final_cost_count'] == len(costs)
        assert g['mean_expected_cost_if_valid'] == (statistics.mean(costs) if costs else None)
        assert g['mean_expected_regret_if_valid'] == (statistics.mean(regrets) if regrets else None)
        assert g['mean_latency_seconds'] == (statistics.mean(times) if times else None)
        assert abs(g['total_latency_seconds']-sum(times)) < 1e-9
    assert summary['all_conditions_passed'] == all(passed)
    for comparison in summary['paired_comparisons']:
        a, b = comparison['a'], comparison['b']
        delta = [int(rows[(s['case_id'], b)]['outcome']['usable_correct'])-int(rows[(s['case_id'], a)]['outcome']['usable_correct']) for s in m['cases']]
        assert comparison['b_minus_a_correct_cases'] == sum(delta)
        assert comparison['improved_cases'] == delta.count(1) and comparison['worsened_cases'] == delta.count(-1)
        assert comparison['unchanged_cases'] == delta.count(0)
    assert budget['api_attempts_started'] == len(list((OUT/'attempts').glob('*.started.json'))) <= 64
    assert budget['successful_http_responses'] == raw_count
    assert budget['logical_requests_completed'] == len(list((OUT/'results').glob('*.json'))) == 32
    assert budget['simulations'] == budget['ppo_updates'] == budget['fits'] == 0
    reserved = ROOT/'outputs/llm_semantic_channel_diagnostic_20261005'
    assert not (reserved/'fresh_actual').exists() and not (reserved/'fresh_advice_frozen.json').exists()
    return {'all_checks_passed': True, 'logical_requests_rechecked': 32, 'raw_responses_reclassified': raw_count,
        'probability_references_reconstructed': 32, 'same_prompt_and_output_settings': True,
        'only_predeclared_information_fields_vary': True, 'randomized_order_reconstructed': True,
        'private_answer_not_sent_to_model': True, 'expected_cost_and_regret_reconstructed': True,
        'no_fallback_imputed_for_missing_answers': True, 'ancestor_records_checked': parents,
        'old_global_status_unchanged': True, 'reserved48_scenarios_untouched': True,
        'new_inference_for_verification': 0, 'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0,
        'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
