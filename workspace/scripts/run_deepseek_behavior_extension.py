"""Fixed cheap API transfer of the existing 34 tasks, with a reserved-cost cap."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from deepseek_private_client import request
from run_bristol_calibrated_behavior import read_csv, metrics, cond, family
from run_bristol_prior_mechanism_probe import canonical, object_sha

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'schemas/deepseek_behavior_extension_20261014.json'
OUT = ROOT / 'outputs/deepseek_behavior_extension_20261014'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(name, rows):
    import csv
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    local = json.loads((ROOT / cfg['source_manifest']).read_text(encoding='utf-8'))
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text(encoding='utf-8'))
    tasks = []
    for old in local['tasks']:
        payload = {'model': cfg['model'], 'messages': old['payload']['messages'], **cfg['request_parameters']}
        arm = 'deepseek_human_context' if old['arm'] == 'llm_human_context' else 'deepseek_scene_only'
        input_bound = sum(len(m['content'].encode('utf-8')) for m in payload['messages']) + 512
        reserve = (input_bound * cfg['pricing_upper_usd_per_million']['input_cache_miss'] +
                   cfg['request_parameters']['max_tokens'] * cfg['pricing_upper_usd_per_million']['output']) / 1e6
        tasks.append({'key': old['key'].replace('llm_', 'deepseek_'), 'original_key': old['key'],
            'condition': old['condition'], 'family': old['family'], 'arm': arm,
            'request_order': old['request_order'], 'payload': payload, 'payload_sha256': object_sha(payload),
            'prompt_sha256': old['prompt_sha256'], 'reserved_cost_usd': reserve,
            'input_token_reserve': input_bound})
    assert len(tasks) == cfg['expected_requests'] and sum(t['reserved_cost_usd'] for t in tasks) < cfg['budget_usd']
    protected = [cfg['source_manifest'], cfg['source_predictions'], cfg['source_config'],
        'scripts/deepseek_private_client.py', 'scripts/run_bristol_calibrated_behavior.py',
        'scripts/run_bristol_prior_mechanism_probe.py',
        'outputs/bristol_calibrated_behavior_20261014/summary.json',
        'outputs/bristol_calibrated_behavior_20261014/responses.jsonl',
        'outputs/bristol_calibrated_behavior_20261014/verification.json']
    plan = {'config_sha256': sha(CFG), 'runner_sha256': sha(__file__), 'tasks': tasks,
        'expected_requests': len(tasks), 'selected_case_keys': local['selected_case_keys'],
        'protected_files': {p: sha(ROOT / p) for p in protected},
        'inherited_gate': source_cfg['continuation_gate'],
        'budget_usd': cfg['budget_usd'], 'total_reserved_cost_usd': sum(t['reserved_cost_usd'] for t in tasks)}
    OUT.mkdir(exist_ok=True)
    path = OUT / 'frozen_manifest.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert {k: v for k, v in saved.items() if k != 'frozen_at_utc'} == plan
        return cfg, saved
    plan['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(path, plan)
    print(json.dumps({'status': 'api_tasks_frozen', 'requests': 34,
        'budget_usd': cfg['budget_usd'], 'maximum_reserved_usd': plan['total_reserved_cost_usd']}), flush=True)
    return cfg, plan


def ledger(plan):
    path = OUT / 'responses.jsonl'
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    saved = {v['key']: v for v in values}
    assert len(values) == len(saved)
    tasks = {t['key']: t for t in plan['tasks']}
    for key, row in saved.items():
        assert row['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
        assert row['payload_sha256'] == tasks[key]['payload_sha256']
    return saved


def run(cfg, plan):
    saved = ledger(plan)
    spent = sum(row['budget_charge_usd'] for row in saved.values())
    elapsed = sum(row['wall_seconds'] for row in saved.values())
    failures = 0
    for task in plan['tasks']:
        if task['key'] in saved:
            failures = 0 if saved[task['key']]['valid'] else failures + 1
            if saved[task['key']].get('usage_unknown'): break
            continue
        if failures >= cfg['max_consecutive_errors'] or spent + task['reserved_cost_usd'] > cfg['budget_usd']:
            break
        if elapsed >= cfg['total_request_wall_budget_seconds']: break
        row = {k: task[k] for k in ['key', 'request_order', 'payload_sha256', 'prompt_sha256']}
        row.update(manifest_sha256=sha(OUT / 'frozen_manifest.json'),
                   started_at_utc=datetime.now(timezone.utc).isoformat(),
                   reserved_cost_usd=task['reserved_cost_usd'], budget_charge_usd=task['reserved_cost_usd'])
        before = time.monotonic()
        try:
            response = request('/chat/completions', task['payload'], timeout=cfg['per_request_timeout_seconds'])
            row['response'] = response
            usage = response.get('usage', {})
            if not all(isinstance(usage.get(k), int) for k in ['prompt_tokens', 'completion_tokens']):
                row['usage_unknown'] = True
                raise ValueError('Usage unavailable; stop to preserve cost audit')
            actual_upper = (usage['prompt_tokens'] * cfg['pricing_upper_usd_per_million']['input_cache_miss'] +
                            usage['completion_tokens'] * cfg['pricing_upper_usd_per_million']['output']) / 1e6
            row.update(estimated_peak_all_miss_usd=actual_upper, budget_charge_usd=actual_upper)
            if actual_upper > task['reserved_cost_usd']:
                row['usage_unknown'] = True
                raise ValueError('Actual usage exceeds reserved token bound; stop')
            choice = response['choices'][0]
            assert choice['finish_reason'] == 'stop', 'Incomplete output'
            assert not choice['message'].get('reasoning_content'), 'Unexpected reasoning output'
            value = json.loads(choice['message']['content'])
            assert set(value) == {'p_left'} and not isinstance(value['p_left'], bool)
            p = value['p_left']
            assert isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
            row.update(valid=True, p_left=float(p))
            failures = 0
        except Exception as exc:
            row.update(valid=False, p_left=None, error=str(exc))
            # If the server accepted but usage was not recovered, reserve the
            # whole amount and stop rather than silently issuing another call.
            if 'response' not in row: row['usage_unknown'] = True
            failures += 1
        row['wall_seconds'] = time.monotonic() - before
        elapsed += row['wall_seconds']
        spent += row['budget_charge_usd']
        with (OUT / 'responses.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        saved[task['key']] = row
        print(json.dumps({'completed': len(saved), 'planned': 34, 'valid': row['valid'],
            'budget_used_usd_upper': round(spent, 6)}), flush=True)
        if row.get('usage_unknown'): break
    dump(OUT / 'run_status.json', {'completed': len(saved), 'valid': sum(r['valid'] for r in saved.values()),
        'budget_used_usd_upper': spent, 'budget_usd': cfg['budget_usd'], 'wall_seconds': elapsed})


def analyze(cfg, plan):
    saved = ledger(plan)
    if len(saved) != 34 or not all(r['valid'] for r in saved.values()):
        dump(OUT / 'summary.json', {'status': 'inconclusive_partial_api_queue', 'completed': len(saved),
            'valid': sum(r['valid'] for r in saved.values()),
            'budget_used_usd_upper': sum(r['budget_charge_usd'] for r in saved.values())})
        return
    records = read_csv(ROOT / cfg['source_predictions'])
    for row in records:
        for arm in cfg['arms']:
            row[arm] = saved['_'.join(map(str, cond(row))) + '__' + arm]['p_left']
    table('all_predictions.csv', records)
    aggregate = {m: metrics(records, m) for m in cfg['comparisons']}
    families = sorted({family(r) for r in records})
    per_family, per_condition = [], []
    for fam in families:
        subset = [r for r in records if family(r) == fam]
        for method in cfg['comparisons']:
            per_family.append({'family': canonical(fam), 'method': method, **metrics(subset, method)})
    for c in sorted({cond(r) for r in records}):
        subset = [r for r in records if cond(r) == c]
        for method in cfg['comparisons']:
            per_condition.append({'condition': canonical(c), 'method': method, 'p_left': float(subset[0][method]),
                'observed_left': sum(int(r['observed_exit']) == 1 for r in subset), **metrics(subset, method)})
    table('per_condition_metrics.csv', per_condition)
    table('per_family_metrics.csv', per_family)
    candidate = aggregate['deepseek_human_context']
    comparisons = {}
    for ref in ['clean_frequency', 'clean_less_crowded_rule', 'clean_logit', 'deepseek_scene_only', 'llm_human_context']:
        base = aggregate[ref]
        diffs = [metrics([r for r in records if family(r) == f], 'deepseek_human_context')['condition_macro_NLL'] -
                 metrics([r for r in records if family(r) == f], ref)['condition_macro_NLL'] for f in families]
        comparisons[ref] = {'NLL_delta': candidate['condition_macro_NLL'] - base['condition_macro_NLL'],
            'NLL_relative_improvement': 1 - candidate['condition_macro_NLL'] / base['condition_macro_NLL'],
            'Brier_delta': candidate['condition_macro_Brier'] - base['condition_macro_Brier'],
            'accuracy_delta_pp': 100 * (candidate['participant_accuracy'] - base['participant_accuracy']),
            'families_lower_NLL': sum(d < -1e-12 for d in diffs), 'families_higher_NLL': sum(d > 1e-12 for d in diffs)}
    passed = all(comparisons[r]['NLL_relative_improvement'] >= .02 and comparisons[r]['Brier_delta'] < 0
        for r in ['clean_frequency', 'clean_less_crowded_rule', 'clean_logit', 'deepseek_scene_only'])
    passed &= comparisons['clean_logit']['families_lower_NLL'] >= 6
    usage = {'prompt_tokens': sum(r['response']['usage']['prompt_tokens'] for r in saved.values()),
             'completion_tokens': sum(r['response']['usage']['completion_tokens'] for r in saved.values())}
    subgroups = {label: {m: metrics(rs, m) for m in cfg['comparisons']} for label, rs in {
        'no_map': [r for r in records if cond(r)[2] == 0],
        'equal_map_nonbalanced': [r for r in records if cond(r)[2] == 3],
        'busy': [r for r in records if r['crowd_state_design'] == 'busy'],
        'free': [r for r in records if r['crowd_state_design'] == 'free']}.items()}
    summary = {'status': 'completed_cheap_api_development_extension', 'requests': 34, 'valid': 34,
        'records': 246, 'conditions': 17, 'families': 9, 'methods': aggregate,
        'comparisons': comparisons, 'subgroups': subgroups, 'continuation_gate_passed': bool(passed),
        'decision': 'Development signal requiring independent confirmation.' if passed else
                    'No added-value gate pass; close this fixed API setup on these outcomes.',
        'usage': usage, 'estimated_peak_all_miss_usd': sum(r['estimated_peak_all_miss_usd'] for r in saved.values()),
        'budget_usd': cfg['budget_usd'], 'cost_disclosure': 'Conservative estimate at documented peak cache-miss rates, not an account invoice; caching/off-peak may lower actual charges.',
        'total_request_seconds': sum(r['wall_seconds'] for r in saved.values()),
        'reported_models': sorted({r['response']['model'] for r in saved.values()}),
        'system_fingerprints': sorted({r['response'].get('system_fingerprint', '') or '' for r in saved.values()}),
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'prediction_sha256': sha(OUT / 'all_predictions.csv'),
        'PPO_updates': 0, 'old_final_scenarios_accessed': False, 'Liao_confirmation_scores_computed': False,
        'limitations': [cfg['timing_disclosure'], 'Hosted model alias is not a frozen model digest.',
                        'Stateless text probability generation; static virtual task, no dynamic guidance efficacy.']}
    dump(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['status', 'methods', 'comparisons', 'usage',
        'estimated_peak_all_miss_usd', 'reported_models', 'continuation_gate_passed', 'decision']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    mode = parser.parse_args().mode
    cfg, plan = prepare()
    if mode == 'run': run(cfg, plan)
    if mode in ['run', 'analyze']: analyze(cfg, plan)
