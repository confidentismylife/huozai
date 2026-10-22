"""Independent source, held-family leakage, probability and scoring audit."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bristol_calibrated_behavior_20261014'
CFG = ROOT / 'schemas/bristol_calibrated_behavior_20261014.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def c(row): return tuple(int(row[k]) for k in ['assigned_left_crowd', 'assigned_right_crowd', 'knowledge_code'])


def family(c): return min(c[:2]), max(c[:2]), c[2]


def close(a, b): assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), (a, b)


def cell_metrics(rs, method):
    cells = defaultdict(list)
    for r in rs: cells[c(r)].append(r)
    losses, briers, weighted_loss = [], [], 0.0
    correct = ties = tied_conditions = 0
    for cell in cells.values():
        p = float(cell[0][method])
        assert all(abs(float(r[method]) - p) <= 1e-12 for r in cell)
        assert math.isfinite(p) and 0 <= p <= 1
        left = sum(int(r['observed_exit']) == 1 for r in cell)
        n = len(cell)
        q = left / n
        pc = max(1e-9, min(1 - 1e-9, p))
        loss = -q * math.log(pc) - (1 - q) * math.log1p(-pc)
        losses.append(loss)
        briers.append(q * (1 - pc) ** 2 + (1 - q) * pc ** 2)
        weighted_loss += n * loss
        correct += left if p >= .5 else n - left
        ties += n if p == .5 else 0
        tied_conditions += p == .5
    return {'records': len(rs), 'conditions': len(cells), 'condition_macro_NLL': sum(losses) / len(cells),
        'condition_macro_Brier': sum(briers) / len(cells), 'participant_NLL': weighted_loss / len(rs),
        'participant_accuracy': correct / len(rs), 'correct_records': correct,
        'tied_records': ties, 'tied_conditions': tied_conditions}


def vector(condition):
    left, right, k = condition
    a = (right - left) / (left + right)
    busy = int(max(left, right) >= 20)
    return [int(k == 3), a, int(left == 0), int(right == 0), busy, a * busy, a * int(k == 3)]


def main(preflight):
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    assert sha(CFG) == plan['config_sha256']
    assert sha(ROOT / 'scripts/run_bristol_calibrated_behavior.py') == plan['runner_sha256']
    for p, digest in plan['protected_previous_files'].items(): assert sha(ROOT / p) == digest, p
    assert sha(OUT / 'baseline_predictions.csv') == plan['baseline_predictions_sha256']
    source = rows(ROOT / cfg['source_cases'])
    original = {r['case_key']: r for r in source}
    gates = rows(ROOT / cfg['source_gates'])
    eligible = {r['case_key'] for r in gates if int(r['knowledge_code']) == 0 or
                (int(r['knowledge_code']) == 3 and int(r['assigned_left']) != int(r['assigned_right']))}
    assert set(original) == eligible == set(plan['selected_case_keys']) and len(source) == 246
    conditions = {c(r) for r in source}
    families = {family(v) for v in conditions}
    assert len(conditions) == 17 and len(families) == 9
    baselines = rows(OUT / 'baseline_predictions.csv')
    for r in baselines:
        orig = original[r['case_key']]
        assert c(r) == c(orig) and r['observed_exit'] == orig['observed_exit']
    folds = {tuple(f['family']): f for f in plan['folds']}
    for fam, fold in folds.items():
        train = [r for r in source if family(c(r)) != fam]
        test = [r for r in source if family(c(r)) == fam]
        assert set(fold['train_case_keys']) == {r['case_key'] for r in train}
        assert set(fold['held_case_keys']) == {r['case_key'] for r in test}
        assert not set(fold['train_case_keys']) & set(fold['held_case_keys'])
        assert {tuple(v) for v in fold['train_conditions']} == {c(r) for r in train}
        assert {tuple(v) for v in fold['held_conditions']} == {c(r) for r in test}
        actual = []
        for cond in sorted({c(r) for r in train}):
            cell = [r for r in train if c(r) == cond]
            left = sum(int(r['observed_exit']) == 1 for r in cell)
            actual.append([*cond, left, len(cell) - left])
        assert actual == fold['human_training_summary']
        n = len(train)
        y = np.array([left / (left + right) for _, _, _, left, right in actual])
        x = np.array([vector(tuple(row[:3])) for row in actual])
        p0 = (y.mean() * n + .5) / (n + 1)
        close(p0, fold['p0'])
        agreements = []
        for row in actual:
            l, r, _, nl, nr = row
            if l < r: agreements.append(nl / (nl + nr))
            if l > r: agreements.append(nr / (nl + nr))
        q = ((sum(agreements) / len(actual)) * n + 1) / ((len(agreements) / len(actual)) * n + 2)
        close(q, fold['q'])
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        assert np.allclose(center, fold['center'], atol=1e-12)
        assert np.allclose(scale, fold['scale'], atol=1e-12)
        design = np.column_stack([np.ones(len(x)), (x - center) / scale])
        beta = np.array(fold['beta'])
        gradient = design.T @ (expit(design @ beta) - y) / len(y)
        gradient[1:] += cfg['logit']['l2'] * beta[1:]
        assert np.max(np.abs(gradient)) < 1e-5
        for row in baselines:
            if family(c(row)) != fam: continue
            close(row['clean_frequency'], p0)
            l, r, _ = c(row)
            close(row['clean_less_crowded_rule'], q if l < r else 1 - q if l > r else .5)
            design_row = np.r_[1, (np.array(vector(c(row))) - center) / scale]
            close(row['clean_logit'], expit(design_row @ beta))
    tasks = {t['key']: t for t in plan['tasks']}
    assert len(tasks) == 34 and len({t['payload']['messages'][0]['content'] for t in tasks.values()}) == 1
    for task in tasks.values():
        payload = task['payload']
        target = tuple(task['condition'])
        obs = json.loads(payload['messages'][1]['content'])
        assert set(obs) == {'target', 'training_columns', 'training_conditions'}
        assert set(obs['target']) == {'assigned_left', 'assigned_right', 'map_code', 'crowd_state'}
        assert tuple(obs['target'][k] for k in ['assigned_left', 'assigned_right', 'map_code']) == target
        assert obs['target']['crowd_state'] == ('busy' if max(target[:2]) >= 20 else 'free')
        fold = folds[family(target)]
        assert obs['training_conditions'] == (fold['human_training_summary'] if task['arm'] == 'llm_human_context' else [])
        assert all(family(tuple(row[:3])) != family(target) for row in obs['training_conditions'])
        assert payload['options'] == cfg['LLM']['options'] and payload['model'] == cfg['LLM']['model']
        assert task['payload_sha256'] == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        system, user = [m['content'] for m in payload['messages']]
        assert hashlib.sha256((system + '\n' + user).encode()).hexdigest() == task['prompt_sha256']
    result = {'status': 'preflight_verified' if preflight else 'completed_experiment_verified',
        'records': 246, 'conditions': 17, 'families': 9, 'fold_leakage_check_passed': True,
        'clean_training_and_scoring_selection_verified': True, 'training_summary_counts_verified': True,
        'baseline_probabilities_and_logit_stationarity_verified': True, 'previous_files_unchanged': True,
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'verifier_sha256': sha(__file__)}
    if not preflight:
        cached = [json.loads(line) for line in (OUT / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(cached) == len({r['key'] for r in cached}) == 34
        ledger = {r['key']: r for r in cached}
        predictions = rows(OUT / 'all_predictions.csv')
        baseline_by_key = {r['case_key']: r for r in baselines}
        assert {r['case_key'] for r in predictions} == eligible and len(predictions) == 246
        for response in cached:
            task = tasks[response['key']]
            assert response['valid'] and response['manifest_sha256'] == result['manifest_sha256']
            assert response['model_digest'] == cfg['LLM']['digest']
            assert response['payload_sha256'] == task['payload_sha256']
            raw = json.loads(response['response']['message']['content'])
            assert set(raw) == {'p_left'} and not isinstance(raw['p_left'], bool)
            close(raw['p_left'], response['p_left'])
            assert response['response']['done'] and response['response'].get('done_reason') in [None, 'stop']
            assert response['response']['prompt_eval_count'] < cfg['LLM']['options']['num_ctx'] - 32
        for row in predictions:
            assert row['observed_exit'] == original[row['case_key']]['observed_exit'] and c(row) == c(original[row['case_key']])
            for method in cfg['methods'][:3]: close(row[method], baseline_by_key[row['case_key']][method])
            for arm in cfg['arms']: close(row[arm], ledger['_'.join(map(str, c(row))) + '__' + arm]['p_left'])
        summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
        assert summary['manifest_sha256'] == result['manifest_sha256']
        assert summary['prediction_sha256'] == sha(OUT / 'all_predictions.csv')
        for method in cfg['methods']:
            for key, value in cell_metrics(predictions, method).items(): close(summary['methods'][method][key], value)
        subsets = {
            'no_map': [r for r in predictions if c(r)[2] == 0],
            'equal_map_nonbalanced': [r for r in predictions if c(r)[2] == 3],
            'busy': [r for r in predictions if r['crowd_state_design'] == 'busy'],
            'free': [r for r in predictions if r['crowd_state_design'] == 'free']}
        for label, subset in subsets.items():
            for method in cfg['methods']:
                for key, value in cell_metrics(subset, method).items(): close(summary['subgroups'][label][method][key], value)
        close(summary['total_request_seconds'], sum(r['wall_seconds'] for r in cached))
        assert summary['prompt_tokens_range'] == [min(r['response']['prompt_eval_count'] for r in cached),
                                                  max(r['response']['prompt_eval_count'] for r in cached)]
        for name, field in [('per_condition_metrics.csv', 'condition'), ('per_family_metrics.csv', 'family')]:
            for detail in rows(OUT / name):
                key = tuple(json.loads(detail[field]))
                subset = [r for r in predictions if (c(r) if field == 'condition' else family(c(r))) == key]
                for metric, value in cell_metrics(subset, detail['method']).items(): close(detail[metric], value)
        candidate = summary['methods']['llm_human_context']
        gate_passed = True
        for ref in cfg['continuation_gate']['references']:
            base = summary['methods'][ref]
            comparison = summary['comparisons'][ref]
            improvement = 1 - candidate['condition_macro_NLL'] / base['condition_macro_NLL']
            brier_delta = candidate['condition_macro_Brier'] - base['condition_macro_Brier']
            close(comparison['NLL_relative_improvement'], improvement)
            close(comparison['NLL_delta'], candidate['condition_macro_NLL'] - base['condition_macro_NLL'])
            close(comparison['Brier_delta'], brier_delta)
            close(comparison['accuracy_delta_pp'], 100 * (candidate['participant_accuracy'] - base['participant_accuracy']))
            wins = sum(cell_metrics([r for r in predictions if family(c(r)) == fam], 'llm_human_context')['condition_macro_NLL'] <
                cell_metrics([r for r in predictions if family(c(r)) == fam], ref)['condition_macro_NLL'] - 1e-12 for fam in families)
            assert wins == comparison['families_lower_NLL']
            losses = sum(cell_metrics([r for r in predictions if family(c(r)) == fam], 'llm_human_context')['condition_macro_NLL'] >
                cell_metrics([r for r in predictions if family(c(r)) == fam], ref)['condition_macro_NLL'] + 1e-12 for fam in families)
            assert losses == comparison['families_higher_NLL']
            gate_passed &= improvement >= .02 and brier_delta < 0
        gate_passed &= summary['comparisons']['clean_logit']['families_lower_NLL'] >= 6
        assert bool(gate_passed) == summary['continuation_gate_passed']
        for item in rows(OUT / 'leave_one_scoring_family_sensitivity.csv'):
            omitted = tuple(json.loads(item['omitted_scoring_family']))
            subset = [r for r in predictions if family(c(r)) != omitted]
            close(item['NLL_delta'], cell_metrics(subset, 'llm_human_context')['condition_macro_NLL'] -
                cell_metrics(subset, item['reference'])['condition_macro_NLL'])
        result.update(all_metrics_independently_recomputed=True, raw_responses_match_predictions=True,
            continuation_gate_verified=True, summary_sha256=sha(OUT / 'summary.json'), LLM_requests=34,
            PPO_updates=0, old_final_scenarios_accessed=False, Liao_confirmation_scores_computed=False)
    (OUT / ('preflight_verification.json' if preflight else 'verification.json')).write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
