"""Verify identical public inputs, billed-token estimates and independent scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from verify_bristol_calibrated_behavior import rows, c, family, cell_metrics, close

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/deepseek_behavior_extension_20261014'
CONFIG = ROOT / 'schemas/deepseek_behavior_extension_20261014.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(preflight):
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    local = json.loads((ROOT / cfg['source_manifest']).read_text(encoding='utf-8'))
    assert plan['config_sha256'] == sha(CONFIG)
    assert plan['runner_sha256'] == sha(ROOT / 'scripts/run_deepseek_behavior_extension.py')
    for path, digest in plan['protected_files'].items(): assert sha(ROOT / path) == digest, path
    original = {t['key']: t for t in local['tasks']}
    assert len(plan['tasks']) == len({t['key'] for t in plan['tasks']}) == 34
    for t in plan['tasks']:
        source = original[t['original_key']]
        payload = t['payload']
        assert payload['messages'] == source['payload']['messages']
        assert t['condition'] == source['condition'] and t['family'] == source['family']
        assert t['request_order'] == source['request_order'] and t['prompt_sha256'] == source['prompt_sha256']
        assert payload['thinking'] == {'type': 'disabled'} and payload['max_tokens'] == 32
        assert payload['temperature'] == 0 and payload['response_format'] == {'type': 'json_object'}
        assert payload['model'] == cfg['model']
        assert t['payload_sha256'] == hashlib.sha256(json.dumps(payload, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        token_reserve = sum(len(m['content'].encode('utf-8')) for m in payload['messages']) + 512
        assert token_reserve == t['input_token_reserve']
        close(t['reserved_cost_usd'], (token_reserve * .3 + 32 * 1.2) / 1e6)
    close(plan['total_reserved_cost_usd'], sum(t['reserved_cost_usd'] for t in plan['tasks']))
    assert plan['total_reserved_cost_usd'] < cfg['budget_usd'] == .05
    result = {'status': 'api_preflight_verified' if preflight else 'completed_api_extension_verified',
        'identical_scene_and_training_messages_verified': True, 'same_folds_and_request_order_verified': True,
        'cost_reservation_verified': True, 'previous_files_unchanged': True,
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'verifier_sha256': sha(__file__)}
    if not preflight:
        ledger = [json.loads(line) for line in (OUT / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        tasks = {t['key']: t for t in plan['tasks']}
        assert len(ledger) == len({r['key'] for r in ledger}) == 34 and all(r['valid'] for r in ledger)
        cached = {r['key']: r for r in ledger}
        total_fee = 0.0
        for r in ledger:
            task = tasks[r['key']]
            assert r['manifest_sha256'] == result['manifest_sha256'] and r['payload_sha256'] == task['payload_sha256']
            assert r['response']['choices'][0]['finish_reason'] == 'stop'
            message = r['response']['choices'][0]['message']
            assert not message.get('reasoning_content')
            obj = json.loads(message['content'])
            assert set(obj) == {'p_left'} and isinstance(obj['p_left'], (int, float)) and not isinstance(obj['p_left'], bool)
            assert math.isfinite(obj['p_left']) and 0 <= obj['p_left'] <= 1
            close(r['p_left'], obj['p_left'])
            use = r['response']['usage']
            fee = (use['prompt_tokens'] * .3 + use['completion_tokens'] * 1.2) / 1e6
            close(r['estimated_peak_all_miss_usd'], fee)
            close(r['budget_charge_usd'], fee)
            assert fee <= task['reserved_cost_usd'] and use['completion_tokens'] <= 32
            total_fee += fee
        assert total_fee < .05
        preds = rows(OUT / 'all_predictions.csv')
        source_rows = {r['case_key']: r for r in rows(ROOT / cfg['source_predictions'])}
        assert len(preds) == 246 and {r['case_key'] for r in preds} == set(plan['selected_case_keys'])
        for row in preds:
            source = source_rows[row['case_key']]
            assert all(row[field] == value for field, value in source.items())
            for arm in cfg['arms']: close(row[arm], cached['_'.join(map(str, c(row))) + '__' + arm]['p_left'])
        summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
        assert summary['manifest_sha256'] == result['manifest_sha256'] and summary['prediction_sha256'] == sha(OUT / 'all_predictions.csv')
        for method in cfg['comparisons']:
            for metric, value in cell_metrics(preds, method).items(): close(summary['methods'][method][metric], value)
        subsets = {'no_map': [r for r in preds if c(r)[2] == 0],
            'equal_map_nonbalanced': [r for r in preds if c(r)[2] == 3],
            'busy': [r for r in preds if r['crowd_state_design'] == 'busy'],
            'free': [r for r in preds if r['crowd_state_design'] == 'free']}
        for label, subset in subsets.items():
            for method in cfg['comparisons']:
                for metric, value in cell_metrics(subset, method).items(): close(summary['subgroups'][label][method][metric], value)
        for filename, field in [('per_condition_metrics.csv', 'condition'), ('per_family_metrics.csv', 'family')]:
            for detail in rows(OUT / filename):
                group = tuple(json.loads(detail[field]))
                subset = [r for r in preds if (c(r) if field == 'condition' else family(c(r))) == group]
                for metric, value in cell_metrics(subset, detail['method']).items(): close(detail[metric], value)
        fams = sorted({family(c(r)) for r in preds})
        candidate = summary['methods']['deepseek_human_context']
        for ref, comp in summary['comparisons'].items():
            base = summary['methods'][ref]
            close(comp['NLL_delta'], candidate['condition_macro_NLL'] - base['condition_macro_NLL'])
            close(comp['NLL_relative_improvement'], 1 - candidate['condition_macro_NLL'] / base['condition_macro_NLL'])
            close(comp['Brier_delta'], candidate['condition_macro_Brier'] - base['condition_macro_Brier'])
            close(comp['accuracy_delta_pp'], 100 * (candidate['participant_accuracy'] - base['participant_accuracy']))
            differences = [cell_metrics([r for r in preds if family(c(r)) == f], 'deepseek_human_context')['condition_macro_NLL'] -
                           cell_metrics([r for r in preds if family(c(r)) == f], ref)['condition_macro_NLL'] for f in fams]
            assert comp['families_lower_NLL'] == sum(d < -1e-12 for d in differences)
            assert comp['families_higher_NLL'] == sum(d > 1e-12 for d in differences)
        passed = all(summary['comparisons'][ref]['NLL_relative_improvement'] >= .02 and
            summary['comparisons'][ref]['Brier_delta'] < 0 for ref in
            ['clean_frequency', 'clean_less_crowded_rule', 'clean_logit', 'deepseek_scene_only'])
        passed &= summary['comparisons']['clean_logit']['families_lower_NLL'] >= 6
        assert bool(passed) == summary['continuation_gate_passed']
        for k in ['prompt_tokens', 'completion_tokens']: assert summary['usage'][k] == sum(r['response']['usage'][k] for r in ledger)
        close(summary['estimated_peak_all_miss_usd'], total_fee)
        close(summary['total_request_seconds'], sum(r['wall_seconds'] for r in ledger))
        result.update(all_metrics_independently_recomputed=True, raw_predictions_and_usage_verified=True,
            estimated_peak_upper_usd=total_fee, summary_sha256=sha(OUT / 'summary.json'),
            requests=34, continuation_gate_verified=True)
    (OUT / ('preflight_verification.json' if preflight else 'verification.json')).write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
