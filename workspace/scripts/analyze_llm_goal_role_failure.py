"""Post hoc behavioral audit using sealed response caches; no model requests."""
from pathlib import Path
import csv
import json
import sys
from collections import Counter

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_goal_role_failure_analysis_20261006'


def audit(model):
    folder = ROOT / f'outputs/llm_goal_role_{model}_20261006'
    checked = ledger_check(folder / 'completion_manifest.json')
    m = c.read(folder / 'manifest.json')
    records = []
    for req in m['request_order']:
        result = c.read(folder / 'results' / f"{req['request_id']}.json")
        raw = c.read(ROOT / result['raw_path'])
        data = json.loads(req['payload']['messages'][0]['content'].split('Data: ')[1])
        response = json.loads(raw['response']['message']['content'])
        location = data['reported_location']
        other = 'south' if location == 'north' else 'north'
        q = data['probability_report_correct']
        location_cost = q if data['report_role'] == 'hazard' else 1 - q
        costs = {location: location_cost, other: 1 - location_cost}
        reference = min(costs, key=costs.get) if data['objective'] == 'minimize' else max(costs, key=costs.get)
        assert reference == req['private_reference']['reference_exit']
        assert all(abs(costs[k] - req['private_reference']['expected_cost_by_exit'][k]) < 1e-12 for k in costs)
        direction = response['recommended_exit']
        prediction = other if data['objective'] == 'minimize' else location
        assert result['outcome']['usable_correct'] == (direction == reference)
        records.append({'model': model, 'request_id': req['request_id'], 'case_id': req['case_id'],
            'condition': req['condition_id'], 'objective': data['objective'], 'role': data['report_role'],
            'reported_location': location, 'probability': q, 'direction': direction, 'reference': reference,
            'role_ignoring_rule_prediction': prediction, 'matches_role_ignoring_rule': direction == prediction,
            'correct': direction == reference, 'confidence': response['confidence'],
            'confidence_equals_report_reliability': response['confidence'] == q,
            'output_tokens': raw['response'].get('eval_count'), 'done_reason': raw['response'].get('done_reason')})
    by = {(r['case_id'], r['condition']): r for r in records}
    goal_pairs = [(by[(s['case_id'], f'minimize_{role}')], by[(s['case_id'], f'maximize_{role}')])
                  for s in m['cases'] for role in ('hazard', 'clear')]
    role_pairs = [(by[(s['case_id'], f'{objective}_hazard')], by[(s['case_id'], f'{objective}_clear')])
                  for s in m['cases'] for objective in ('minimize', 'maximize')]
    summary = {'model': model, 'requests': len(records), 'sealed_records_checked': checked,
        'correct': sum(r['correct'] for r in records),
        'matches_role_ignoring_rule': sum(r['matches_role_ignoring_rule'] for r in records),
        'north_outputs': sum(r['direction'] == 'north' for r in records),
        'confidence_equals_input_report_reliability': sum(r['confidence_equals_report_reliability'] for r in records),
        'goal_flips_observed_of16': sum(a['direction'] != b['direction'] for a, b in goal_pairs),
        'equivalent_role_encodings_same_answer_of16': sum(a['direction'] == b['direction'] for a, b in role_pairs),
        'output_token_counts': dict(Counter(r['output_tokens'] for r in records)),
        'stop_reasons': dict(Counter(r['done_reason'] for r in records)),
        'all_input_probabilities_above_one_half': all(r['probability'] > .5 for r in records),
        'probability_values': sorted({r['probability'] for r in records}),
        'confidence_semantics_defined_in_prompt': False,
        'posthoc_rule_fit_is_not_internal_mechanism_identification': True}
    return summary, records


def run():
    assert not OUT.exists()
    OUT.mkdir()
    summaries, rows = [], []
    for model in ['8b', '14b']:
        summary, records = audit(model)
        summaries.append(summary)
        rows.extend(records)
    io.save(OUT / 'summary.json', {'models': summaries, 'new_llm_requests': 0, 'new_simulations': 0,
        'new_fits': 0, 'new_ppo_updates': 0, 'internal_root_cause_established': False})
    c.csv_write(OUT / 'behavioral_rule_audit.csv', rows)
    io.save(OUT / 'provenance.json', {'at_utc': c.stamp(), 'parent_completion_sha256': {
        f'outputs/llm_goal_role_{model}_20261006/completion_manifest.json': c.sha(ROOT / f'outputs/llm_goal_role_{model}_20261006/completion_manifest.json')
        for model in ['8b', '14b']}, 'global_status_sha256': c.sha(c.STATUS)})
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
