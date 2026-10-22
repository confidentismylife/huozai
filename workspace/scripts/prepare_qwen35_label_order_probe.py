"""Freeze 32 matched label/order probes without contacting a model service."""
from pathlib import Path
import copy
import itertools
import json
import random
import sys

from llm_goal_role_core import PREFIX, inputs, decoded_costs, choose
from run_llm_output_mechanism import save, hash_value
from run_qwen35_27b_trial import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_label_order_probe_20261006'
PROTOCOL = ROOT / 'reports/27B出口名称与候选顺序诊断固定方案_2026-10-06.md'
BASE = ROOT / 'outputs/llm_goal_role_20261006/manifest.json'
PARENT = ROOT / 'outputs/qwen35_27b_text_final_audit_20261006/completion_manifest.json'


def run():
    assert not OUT.exists(), 'Preserve an existing preparation.'
    base = json.loads(BASE.read_text(encoding='utf-8'))
    cases = [s for s in base['cases'] if s['pair_id'] == 'GRP00']
    assert len(cases) == 2
    requests = []
    for case, role, objective, naming, reverse in itertools.product(
            cases, ('hazard', 'clear'), ('minimize', 'maximize'), ('compass', 'abstract'), (False, True)):
        names = ('north', 'south') if naming == 'compass' else ('A', 'B')
        mapping = dict(zip(('north', 'south'), names))
        data = inputs(case, {'role': role, 'objective': objective})
        costs = decoded_costs(data['reported_location'], role, data['probability_report_correct'])
        ref = choose(costs, objective)
        renamed = {}
        for key, value in data.items():
            new_key = key.replace('north', names[0]).replace('south', names[1])
            renamed[new_key] = mapping[value] if key == 'reported_location' else value
        ordered = list(reversed(names)) if reverse else list(names)
        prefix = PREFIX.replace('(north or south)', '(' + ' or '.join(ordered) + ')')
        schema = copy.deepcopy(base['request_order'][0]['payload']['format'])
        schema['properties']['recommended_exit']['enum'] = ordered
        payload = {'model': 'qwen35-27b-text-q4:latest',
            'messages': [{'role': 'user', 'content': prefix + json.dumps(renamed, sort_keys=True)}],
            'format': schema, 'options': {'temperature': 0, 'seed': 2026100601,
                'num_predict': 1024, 'num_ctx': 2048, 'num_batch': 32},
            'think': False, 'stream': False}
        # Reconstruct the reference from the actual message, using a separate label-agnostic calculation.
        actual = json.loads(payload['messages'][0]['content'].split('Data: ')[1])
        location = actual['reported_location']
        other = next(x for x in names if x != location)
        v = actual['probability_report_correct'] if actual['report_role'] == 'hazard' else 1 - actual['probability_report_correct']
        recovered = {location: v, other: 1 - v}
        expected = (min if objective == 'minimize' else max)(recovered, key=recovered.get)
        assert expected == mapping[ref]
        assert not {'private_reference', 'reference_exit', 'hazard_anchor', 'expected_cost_by_exit'} & set(actual)
        requests.append({'request_id': f"LO_{case['case_id']}_{role}_{objective}_{naming}_{int(reverse)}",
            'family': 'GRP00', 'mirror': case['hazard_anchor'], 'role': role, 'objective': objective,
            'naming': naming, 'candidate_order': ordered, 'reversed_order': reverse,
            'payload': payload, 'payload_sha256': hash_value(payload),
            'private_reference': {'label_to_physical': {names[0]: 0, names[1]: 1},
                'reference_exit': expected, 'reference_physical_exit': 0 if ref == 'north' else 1,
                'expected_cost_by_label': recovered}})
    assert len(requests) == len({r['request_id'] for r in requests}) == 32
    for a in requests:
        for b in requests:
            if (a['mirror'], a['role'], a['objective']) == (b['mirror'], b['role'], b['objective']):
                assert a['private_reference']['reference_physical_exit'] == b['private_reference']['reference_physical_exit']
            if (a['mirror'], a['objective'], a['naming'], a['reversed_order']) == (b['mirror'], b['objective'], b['naming'], b['reversed_order']):
                assert a['private_reference']['expected_cost_by_label'] == b['private_reference']['expected_cost_by_label']
            if (a['role'], a['objective'], a['naming'], a['reversed_order']) == (b['role'], b['objective'], b['naming'], b['reversed_order']) and a['mirror'] != b['mirror']:
                assert a['private_reference']['reference_physical_exit'] != b['private_reference']['reference_physical_exit']
    random.Random(2026100607).shuffle(requests)
    for label in ('compass', 'abstract'):
        for reverse in (False, True):
            group = [r for r in requests if r['naming'] == label and r['reversed_order'] == reverse]
            assert len(group) == 8
            assert sum(r['private_reference']['reference_physical_exit'] == 0 for r in group) == 4
    sources = [Path(__file__), PROTOCOL, ROOT / 'scripts/llm_goal_role_core.py',
        ROOT / 'outputs/local_llm_selection.json', ROOT / 'outputs/third_round_development_status.json']
    OUT.mkdir()
    save(OUT / 'manifest.json', {'state': 'prepared_not_run', 'planned_requests': 32,
        'model': 'qwen35-27b-text-q4:latest', 'request_order_seed': 2026100607,
        'independent_physical_families': 1, 'development_diagnostic_only': True,
        'base_manifest_sha256': sha(BASE), 'parent_completion_sha256': sha(PARENT),
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'request_order': requests, 'inference_started': False})
    save(OUT / 'pre_inference_checks.json', {'all_checks_passed': True,
        'label_order_reference_invariant': True, 'equivalent_role_costs_equal': True,
        'mirror_reference_swaps': True, 'each_naming_order_group_balanced': True,
        'reference_reconstructed_from_actual_messages': True, 'requests': 32,
        'model_service_calls': 0, 'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    save(OUT / 'budget.json', {'api_attempts_started': 0, 'logical_requests_completed': 0,
        'api_attempt_cap': 32, 'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    save(OUT / 'campaign_status.json', {'state': 'prepared_not_run', 'pid': None})
    files = [p for p in OUT.iterdir() if p.is_file()] + sources
    save(OUT / 'preparation_manifest.json', {'state': 'prepared_not_run',
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}, 'new_inference': 0})
    for name, digest in json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))['files'].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps({'state': 'prepared_not_run', 'requests': 32,
        'checks_passed': True, 'new_inference': 0, 'protocol': str(PROTOCOL)}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
