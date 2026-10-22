"""One frozen, source-filtered, training-context experiment; no prompt search."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

from run_liao_motion_pilot import fit_logit
from run_bristol_prior_mechanism_probe import api, canonical, object_sha, validate_content

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'schemas/bristol_calibrated_behavior_20261014.json'
OUT = ROOT / 'outputs/bristol_calibrated_behavior_20261014'
FIELDS = ['assigned_left_crowd', 'assigned_right_crowd', 'knowledge_code']
SYSTEM = (
    'Predict the probability that a human chooses the LEFT room door in a first-person virtual evacuation task. '
    'Predict observed human choice, rather than recommend an optimal route. The room is a 20m square with two '
    'available 1.5m-wide doors. The participant starts equidistant from the doors and is told that a fire alarm '
    'has sounded and to go to the final exit quickly. Simulated people move through each door. Crowd counts '
    'are experimental assignments, not measured queues. Counts 2 or 4 indicate free-flow conditions; '
    'counts 20 or 40 indicate congested conditions. Map code 0 means no map and unknown onward exit location. '
    'Map code 3 means a map showing equal onward distance to the final exit. No smoke or staff observations '
    'are supplied. Training rows, when present, summarize other experimental conditions: '
    '[assigned_left,assigned_right,map_code,human_chose_left,human_chose_right]. '
    'Use those observed human choices to estimate how behavior varies with context. Each training condition '
    'has equal importance; counts also show sampling uncertainty. Do not treat the training samples as '
    'the target condition or assume optimal human behavior. With no training rows use only the target scene. '
    'Return only JSON {"p_left":number} with a finite number between 0 and 1.'
)
FORMAT = {'type': 'object', 'properties': {'p_left': {'type': 'number', 'minimum': 0, 'maximum': 1}},
          'required': ['p_left'], 'additionalProperties': False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cond(row):
    return tuple(int(row[k]) for k in FIELDS)


def family(row):
    a, b, k = cond(row)
    return min(a, b), max(a, b), k


def features(row):
    a, b, k = cond(row)
    adv = (b - a) / (a + b)
    busy = int(max(a, b) >= 20)
    mapped = int(k == 3)
    return [mapped, adv, int(a == 0), int(b == 0), busy, adv * busy, adv * mapped]


def weights(rows):
    counts = Counter(cond(r) for r in rows)
    result = np.array([1 / (len(counts) * counts[cond(r)]) for r in rows])
    assert np.isclose(result.sum(), 1)
    return result


def preference(row):
    a, b, _ = cond(row)
    return 1 if a < b else -1 if a > b else 0


def check_saved(plan):
    assert plan['config_sha256'] == sha(CONFIG)
    assert plan['runner_sha256'] == sha(__file__)
    for path, digest in plan['protected_previous_files'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(OUT / 'baseline_predictions.csv') == plan['baseline_predictions_sha256']


def prepare():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    saved_path = OUT / 'frozen_manifest.json'
    if saved_path.exists():
        plan = json.loads(saved_path.read_text(encoding='utf-8'))
        check_saved(plan)
        return cfg, plan
    records = read_csv(ROOT / cfg['source_cases'])
    source_plan = json.loads((ROOT / cfg['source_selection_manifest']).read_text(encoding='utf-8'))
    gates = read_csv(ROOT / cfg['source_gates'])
    eligible = {r['case_key'] for r in gates if r['source_gate'] == 'comparatively_clear_under_README'}
    assert {r['case_key'] for r in records} == eligible == set(source_plan['selected_case_keys'])
    assert len(records) == len(eligible) == cfg['expected_records']
    assert all(k == 0 or (k == 3 and a != b) for a, b, k in map(cond, records))
    conditions = sorted({cond(r) for r in records})
    families = sorted({family(r) for r in records})
    assert len(conditions) == 17 and len(families) == 9
    # Historical prediction fields never enter a new learner or target input.
    records = [{key: row[key] for key in ['case_key', 'observed_exit', *FIELDS,
                'map_information', 'nearer_to_final_exit', 'crowd_state_design']} for row in records]
    folds, tasks = [], []
    for fam in families:
        train = [r for r in records if family(r) != fam]
        test = [r for r in records if family(r) == fam]
        train_conditions = sorted({cond(r) for r in train})
        held_conditions = sorted({cond(r) for r in test})
        assert not set(train_conditions) & set(held_conditions)
        w = weights(train)
        y = np.array([int(r['observed_exit']) == 1 for r in train], dtype=float)
        p0 = float((w @ y * len(train) + .5) / (len(train) + 1))
        rec = np.array([preference(r) for r in train])
        nonzero = rec != 0
        agrees = ((y == 1) & (rec == 1)) | ((y == 0) & (rec == -1))
        q = float((np.sum(w[nonzero] * agrees[nonzero]) * len(train) + 1) /
                  (np.sum(w[nonzero]) * len(train) + 2))
        x = np.array([features(r) for r in train])
        center, scale, beta, opt = fit_logit(x, y, w, p0, cfg['logit']['l2'], cfg['logit']['maxiter'])
        logits = expit(np.column_stack([np.ones(len(test)),
                       (np.array([features(r) for r in test]) - center) / scale]) @ beta)
        summaries = []
        for c in train_conditions:
            rs = [r for r in train if cond(r) == c]
            left = sum(int(r['observed_exit']) == 1 for r in rs)
            summaries.append([*c, left, len(rs) - left])
        folds.append({'family': list(fam), 'train_case_keys': [r['case_key'] for r in train],
            'held_case_keys': [r['case_key'] for r in test], 'train_conditions': train_conditions,
            'held_conditions': held_conditions, 'human_training_summary': summaries,
            'train_records': len(train), 'held_records': len(test), 'p0': p0, 'q': q,
            'center': center.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist(), 'optimizer': opt})
        for i, row in enumerate(test):
            direction = preference(row)
            row.update(clean_frequency=p0,
                clean_less_crowded_rule=q if direction == 1 else 1 - q if direction == -1 else .5,
                clean_logit=float(logits[i]))
        for c in held_conditions:
            for arm in cfg['arms']:
                user = {'training_columns': ['assigned_left', 'assigned_right', 'map_code',
                        'human_chose_left', 'human_chose_right'],
                    'training_conditions': summaries if arm == 'llm_human_context' else [],
                    'target': {'assigned_left': c[0], 'assigned_right': c[1], 'map_code': c[2],
                               'crowd_state': 'busy' if max(c[:2]) >= 20 else 'free'}}
                content = canonical(user)
                payload = {'model': cfg['LLM']['model'], 'messages': [
                    {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
                    'format': FORMAT, 'think': cfg['LLM']['think'], 'stream': False,
                    'keep_alive': cfg['LLM']['keep_alive'], 'options': cfg['LLM']['options']}
                tasks.append({'key': '_'.join(map(str, c)) + '__' + arm, 'condition': list(c),
                    'family': list(fam), 'arm': arm, 'payload': payload, 'payload_sha256': object_sha(payload),
                    'prompt_sha256': hashlib.sha256((SYSTEM + '\n' + content).encode()).hexdigest()})
    assert all('clean_logit' in r for r in records)
    assert len(tasks) == cfg['expected_requests']
    random.Random(cfg['random_order_seed']).shuffle(tasks)
    for i, task in enumerate(tasks, 1):
        task['request_order'] = i
    OUT.mkdir(exist_ok=True)
    table('baseline_predictions.csv', records)
    protected = [cfg['source_cases'], cfg['source_selection_manifest'], cfg['source_gates'],
        'scripts/run_liao_motion_pilot.py', 'scripts/audit_liao_dynamic_data.py',
        'scripts/run_bristol_prior_mechanism_probe.py']
    for folder in ['bristol_context_choice_benchmark_20261013', 'bristol_prior_mechanism_probe_20261013',
                   'bristol_no_prior_choice_benchmark_20261013', 'bristol_source_semantics_audit_20261014',
                   'clear_context_cache_review_20261014']:
        protected.extend(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'outputs' / folder).iterdir()
                         if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv'})
    plan = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(CONFIG),
        'runner_sha256': sha(__file__), 'protected_previous_files': {p: sha(ROOT / p) for p in sorted(set(protected))},
        'baseline_predictions_sha256': sha(OUT / 'baseline_predictions.csv'),
        'selected_case_keys': [r['case_key'] for r in records], 'folds': folds, 'tasks': tasks,
        'model_digest': cfg['LLM']['digest'], 'expected_requests': len(tasks),
        'PPO_updates': 0, 'old_final_scenarios_accessed': False, 'Liao_confirmation_scores_computed': False}
    dump(saved_path, plan)
    print(json.dumps({'status': 'frozen', 'records': 246, 'conditions': 17, 'folds': 9, 'requests': 34}), flush=True)
    return cfg, plan


def ledger(plan):
    path = OUT / 'responses.jsonl'
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    saved = {r['key']: r for r in values}
    assert len(saved) == len(values)
    tasks = {t['key']: t for t in plan['tasks']}
    for key, row in saved.items():
        assert key in tasks and row['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
        assert row['model_digest'] == plan['model_digest']
        for field in ['payload_sha256', 'prompt_sha256']:
            assert row[field] == tasks[key][field]
    return saved


def run(cfg, plan):
    saved = ledger(plan)
    tags = api('/api/tags', timeout=5)
    actual = next(m for m in tags['models'] if m['name'] == cfg['LLM']['model'])
    assert actual['digest'] == plan['model_digest'], 'Local model digest changed'
    spent = sum(r['wall_seconds'] for r in saved.values())
    errors = 0
    for task in plan['tasks']:
        if task['key'] in saved:
            errors = 0 if saved[task['key']]['valid'] else errors + 1
            continue
        if errors >= cfg['max_consecutive_errors'] or spent >= cfg['total_request_wall_budget_seconds']:
            break
        started = time.monotonic()
        row = {k: task[k] for k in ['key', 'request_order', 'payload_sha256', 'prompt_sha256']}
        row.update(manifest_sha256=sha(OUT / 'frozen_manifest.json'), model_digest=plan['model_digest'],
                   started_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            response = api('/api/chat', task['payload'], timeout=min(cfg['per_request_timeout_seconds'],
                max(1, cfg['total_request_wall_budget_seconds'] - spent)))
            row['response'] = response
            p = validate_content(response)
            if response.get('done_reason') not in [None, 'stop']:
                raise ValueError('Generation ended by a token limit')
            tokens = response.get('prompt_eval_count')
            if tokens is None or tokens >= cfg['LLM']['options']['num_ctx'] - cfg['LLM']['options']['num_predict']:
                raise ValueError('Prompt budget cannot be verified')
            row.update(valid=True, p_left=p)
            errors = 0
        except Exception as exc:
            row.update(valid=False, p_left=None, error=f'{type(exc).__name__}: {exc}')
            errors += 1
        row['wall_seconds'] = time.monotonic() - started
        spent += row['wall_seconds']
        with (OUT / 'responses.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        saved[row['key']] = row
        print(json.dumps({'completed': len(saved), 'planned': 34, 'arm': task['arm'],
            'valid': row['valid'], 'seconds': round(spent, 1)}), flush=True)
    dump(OUT / 'run_status.json', {'completed': len(saved), 'valid': sum(r['valid'] for r in saved.values()),
        'request_wall_seconds': spent, 'budget_seconds': cfg['total_request_wall_budget_seconds'],
        'model_digest': plan['model_digest'], 'PPO_updates': 0})


def metrics(rows, name):
    groups = defaultdict(list)
    correct = ties = 0
    total_loss = 0.0
    for row in rows:
        raw = float(row[name])
        assert math.isfinite(raw) and 0 <= raw <= 1
        p = min(1 - 1e-9, max(1e-9, raw))
        y = int(row['observed_exit']) == 1
        loss = -math.log(p if y else 1 - p)
        groups[cond(row)].append((loss, (p - y) ** 2, raw))
        correct += (raw >= .5) == y
        ties += raw == .5
        total_loss += loss
    assert all(max(v[2] for v in cell) - min(v[2] for v in cell) <= 1e-12 for cell in groups.values())
    return {'records': len(rows), 'conditions': len(groups),
        'condition_macro_NLL': statistics.mean(statistics.mean(v[0] for v in cell) for cell in groups.values()),
        'condition_macro_Brier': statistics.mean(statistics.mean(v[1] for v in cell) for cell in groups.values()),
        'participant_NLL': total_loss / len(rows), 'participant_accuracy': correct / len(rows),
        'correct_records': int(correct), 'tied_records': int(ties),
        'tied_conditions': sum(cell[0][2] == .5 for cell in groups.values())}


def analyze(cfg, plan):
    saved = ledger(plan)
    if len(saved) != cfg['expected_requests'] or not all(r['valid'] for r in saved.values()):
        dump(OUT / 'summary.json', {'status': 'inconclusive_incomplete_or_invalid_inference',
            'completed': len(saved), 'valid': sum(r['valid'] for r in saved.values()),
            'decision': 'No predictive conclusion; no fallbacks or replacement calls.'})
        return
    rows = read_csv(OUT / 'baseline_predictions.csv')
    for row in rows:
        key = '_'.join(map(str, cond(row)))
        for arm in cfg['arms']:
            row[arm] = saved[key + '__' + arm]['p_left']
    table('all_predictions.csv', rows)
    aggregate = {m: metrics(rows, m) for m in cfg['methods']}
    conditions = sorted({cond(r) for r in rows})
    families = sorted({family(r) for r in rows})
    cells, groups = [], []
    for c in conditions:
        subset = [r for r in rows if cond(r) == c]
        for method in cfg['methods']:
            cells.append({'condition': canonical(c), 'method': method, 'p_left': float(subset[0][method]),
                'observed_left_count': sum(int(r['observed_exit']) == 1 for r in subset), **metrics(subset, method)})
    for f in families:
        subset = [r for r in rows if family(r) == f]
        for method in cfg['methods']:
            groups.append({'family': canonical(f), 'method': method, **metrics(subset, method)})
    table('per_condition_metrics.csv', cells)
    table('per_family_metrics.csv', groups)
    candidate = aggregate['llm_human_context']
    comparisons = {}
    sensitivity = []
    for ref in cfg['continuation_gate']['references']:
        base = aggregate[ref]
        deltas = [metrics([r for r in rows if family(r) == f], 'llm_human_context')['condition_macro_NLL'] -
                  metrics([r for r in rows if family(r) == f], ref)['condition_macro_NLL'] for f in families]
        comparisons[ref] = {'NLL_delta': candidate['condition_macro_NLL'] - base['condition_macro_NLL'],
            'NLL_relative_improvement': 1 - candidate['condition_macro_NLL'] / base['condition_macro_NLL'],
            'Brier_delta': candidate['condition_macro_Brier'] - base['condition_macro_Brier'],
            'accuracy_delta_pp': 100 * (candidate['participant_accuracy'] - base['participant_accuracy']),
            'families_lower_NLL': sum(d < -1e-12 for d in deltas),
            'families_higher_NLL': sum(d > 1e-12 for d in deltas)}
        for f in families:
            subset = [r for r in rows if family(r) != f]
            sensitivity.append({'omitted_scoring_family': canonical(f), 'reference': ref,
                'NLL_delta': metrics(subset, 'llm_human_context')['condition_macro_NLL'] -
                             metrics(subset, ref)['condition_macro_NLL']})
    table('leave_one_scoring_family_sensitivity.csv', sensitivity)
    passed = all(v['NLL_relative_improvement'] >= cfg['continuation_gate']['min_NLL_relative_improvement_over_each_reference']
                 and v['Brier_delta'] < 0 for v in comparisons.values())
    passed &= comparisons['clean_logit']['families_lower_NLL'] >= cfg['continuation_gate']['min_families_lower_NLL_than_clean_logit']
    subgroups = {label: {m: metrics(rs, m) for m in cfg['methods']} for label, rs in {
        'no_map': [r for r in rows if cond(r)[2] == 0],
        'equal_map_nonbalanced': [r for r in rows if cond(r)[2] == 3],
        'busy': [r for r in rows if r['crowd_state_design'] == 'busy'],
        'free': [r for r in rows if r['crowd_state_design'] == 'free']}.items()}
    summary = {'status': 'completed_bounded_development_experiment', 'records': 246, 'conditions': 17,
        'mirror_families': 9, 'LLM_requests': len(saved), 'valid_responses': len(saved), 'logit_fits': 9,
        'methods': aggregate, 'comparisons': comparisons, 'subgroups': subgroups,
        'continuation_gate_passed': bool(passed),
        'decision': 'Development signal only; independent confirmation needed.' if passed else
                    'Close this fixed human-context configuration; no further prompt search on these outcomes.',
        'total_request_seconds': sum(r['wall_seconds'] for r in saved.values()),
        'prompt_tokens_range': [min(r['response']['prompt_eval_count'] for r in saved.values()),
                               max(r['response']['prompt_eval_count'] for r in saved.values())],
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'),
        'prediction_sha256': sha(OUT / 'all_predictions.csv'),
        'limitations': [cfg[k] for k in ['learning_disclosure', 'development_visibility', 'scope']],
        'PPO_updates': 0, 'old_final_scenarios_accessed': False, 'Liao_confirmation_scores_computed': False}
    check_saved(plan)
    dump(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    mode = parser.parse_args().mode
    cfg, plan = prepare()
    if mode == 'run':
        run(cfg, plan)
    if mode in ['run', 'analyze']:
        analyze(cfg, plan)
