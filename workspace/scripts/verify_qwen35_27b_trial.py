"""Reconstruct references and outcomes from raw 27B responses, without inference."""
from pathlib import Path
import json

from verify_llm_output_mechanism import classify
from run_llm_output_mechanism import hash_value

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def verify(out):
    out = Path(out)
    m = read(out / 'manifest.json')
    base = read(ROOT / m['base_manifest'])
    assert len(m['request_order']) == len(base['request_order']) == 32
    rows = {}
    for a, b in zip(base['request_order'], m['request_order']):
        assert set(a) == set(b)
        assert {k for k in a if a[k] != b[k]} == {'payload', 'payload_sha256'}
        assert {k for k in a['payload'] if a['payload'][k] != b['payload'][k]} == {'model'}
        assert b['payload']['model'] == 'qwen3.5:27b'
        assert hash_value(b['payload']) == b['payload_sha256']
        assert 'private_reference' not in b['payload']
        data = json.loads(b['payload']['messages'][0]['content'].split('Data: ')[1])
        p = data['probability_report_correct']
        location = data['reported_location']
        at_report = p if data['report_role'] == 'hazard' else 1 - p
        costs = {location: at_report, ('south' if location == 'north' else 'north'): 1 - at_report}
        ref = min(costs, key=costs.get) if data['objective'] == 'minimize' else max(costs, key=costs.get)
        assert ref == b['private_reference']['reference_exit']
        assert all(abs(costs[k] - b['private_reference']['expected_cost_by_exit'][k]) < 1e-10 for k in costs)
        raw = read(ROOT / read(out / 'results' / (b['request_id'] + '.json'))['raw_path'])
        assert raw['payload'] == b['payload'] and raw['http_status'] == 200
        assert raw['response'].get('done') is True and raw['response']['model'] == 'qwen3.5:27b'
        result = read(out / 'results' / (b['request_id'] + '.json'))
        independently_scored = classify(raw['response'], ref)
        assert all(result['outcome'][k] == v for k, v in independently_scored.items())
        assert result['latency_seconds'] == raw['latency_seconds']
        rows[(b['case_id'], b['condition_id'])] = result['outcome']
    summary = read(out / 'summary.json')
    for cond in m['conditions']:
        values = [rows[(case['case_id'], cond['id'])] for case in m['cases']]
        for category in ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure'):
            assert summary['groups'][cond['id']][category] == sum(v['category'] == category for v in values)
    pairs = {p['comparison']: p for p in summary['paired_diagnostics']}
    for objective in ('minimize', 'maximize'):
        values = [(rows[(s['case_id'], objective + '_hazard')], rows[(s['case_id'], objective + '_clear')]) for s in m['cases']]
        usable = [(a, b) for a, b in values if a['strict_format_valid'] and b['strict_format_valid'] and not a['truncated'] and not b['truncated']]
        assert pairs['role_invariance_' + objective]['expected_relation_observed'] == sum(a['semantic_direction'] == b['semantic_direction'] for a, b in usable)
        assert pairs['role_invariance_' + objective]['both_answers_correct'] == sum(a['usable_correct'] and b['usable_correct'] for a, b in values)
    for role in ('hazard', 'clear'):
        values = [(rows[(s['case_id'], 'minimize_' + role)], rows[(s['case_id'], 'maximize_' + role)]) for s in m['cases']]
        usable = [(a, b) for a, b in values if a['strict_format_valid'] and b['strict_format_valid'] and not a['truncated'] and not b['truncated']]
        assert pairs['goal_flip_' + role]['expected_relation_observed'] == sum(a['semantic_direction'] != b['semantic_direction'] for a, b in usable)
        assert pairs['goal_flip_' + role]['both_answers_correct'] == sum(a['usable_correct'] and b['usable_correct'] for a, b in values)
    budget = read(out / 'budget.json')
    assert budget['api_attempts_started'] == budget['successful_http_responses'] == budget['logical_requests_completed'] == 32
    assert len(list((out / 'attempts').glob('*.json'))) == 32
    assert len(list((out / 'raw_responses').glob('*.json'))) == 32
    return {'all_checks_passed': True, 'raw_responses_reclassified': 32, 'references_reconstructed': 32,
            'goal_and_role_pairs_reconstructed': True, 'only_request_model_changed': True,
            'new_inference_for_verification': 0}
