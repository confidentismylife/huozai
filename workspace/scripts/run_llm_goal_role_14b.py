"""Run the prepared, fixed 14B replay, preserving all sealed earlier rounds."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
import llm_goal_role_analysis as analysis
from llm_output_mechanism_core import grade
from llm_14b_runtime import Monitor, write_report

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/llm_goal_role_20261006'
OUT = ROOT / 'outputs/llm_goal_role_14b_20261006'
PLAN = ROOT / 'outputs/llm_goal_role_14b_runtime_20261006/pending_request_plan.json'
PROTOCOL = ROOT / 'reports/LLM_14B固定复测执行方案_2026-10-06.md'
REPORT = ROOT / 'reports/LLM_14B模型复测实验报告_2026-10-06.md'
MODEL = 'qwen3:14b'


def prepare():
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    assert not OUT.exists(), 'Preserve existing replay.'
    base, plan = c.read(BASE / 'manifest.json'), c.read(PLAN)
    assert c.sha(BASE / 'manifest.json') == plan['base_manifest_sha256']
    for a, b in zip(base['request_order'], plan['request_order']):
        assert {k for k in a if a[k] != b[k]} == {'payload', 'payload_sha256'}
        assert {k for k in a['payload'] if a['payload'][k] != b['payload'][k]} == {'model'}
        assert b['payload']['model'] == MODEL and io.hash_value(b['payload']) == b['payload_sha256']
    tags = io.api('/api/tags', timeout=10)
    model = next(x for x in tags['models'] if x['name'] == MODEL)
    assert model['digest'] == plan['expected_registry_manifest_digest']
    snapshot = {'tags': tags, 'version': io.api('/api/version', timeout=10),
                'show': io.api('/api/show', {'model': MODEL}, 10)}
    old8 = c.read(ROOT / 'outputs/llm_goal_role_8b_20261006/model_service_snapshot.json')
    assert snapshot['version'] == old8['version'] == c.read(BASE / 'model_service_snapshot.json')['version']
    assert snapshot['show']['model_info']['general.parameter_count'] == 14_768_307_200
    assert snapshot['show']['details']['quantization_level'] == 'Q4_K_M'
    assert snapshot['show']['template'] == old8['show']['template']
    assert snapshot['show']['parameters'] == old8['show']['parameters']
    assert not io.api('/api/ps', timeout=10)['models']
    parents = [ROOT / x for x in base['parent_completion_sha256']]
    parents += [BASE / 'completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_runtime_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_8b_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_8b_runtime_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_14b_runtime_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_14b_activation_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_goal_role_14b_app_path_fix_20261006/completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p, HISTORY if p.parent == MAIN else None) for p in parents}
    paths = [Path(__file__), ROOT / 'scripts/llm_14b_runtime.py', ROOT / 'scripts/verify_llm_goal_role_14b.py',
             ROOT / 'scripts/llm_goal_role_analysis.py', ROOT / 'scripts/llm_goal_role_core.py',
             ROOT / 'scripts/run_llm_output_mechanism.py', ROOT / 'scripts/llm_output_mechanism_core.py',
             ROOT / 'scripts/verify_llm_output_mechanism.py', ROOT / 'scripts/auto_evacuation_campaign.py',
             ROOT / 'scripts/finalize_auto_evacuation_campaign_review.py', PROTOCOL, PLAN,
             ROOT / 'outputs/local_llm_selection.json']
    OUT.mkdir()
    preflight = {'all_checks_passed': True, 'parent_checks': parent_checks,
                 'same_template_and_default_parameters_as_8b': True,
                 'checked_source_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
                 'memory_before_bytes': Monitor.sample()['available_memory_bytes']}
    io.save(OUT / 'pre_inference_checks.json', preflight)
    io.save(OUT / 'model_service_snapshot.json', snapshot)
    io.save(OUT / 'manifest.json', {'created_at_utc': c.stamp(), 'model_name': MODEL,
        'model_digest': model['digest'], 'cases': base['cases'], 'conditions': base['conditions'],
        'request_order': plan['request_order'], 'request_order_seed': base['request_order_seed'],
        'base_manifest': (BASE / 'manifest.json').relative_to(ROOT).as_posix(),
        'base_manifest_sha256': c.sha(BASE / 'manifest.json'), 'parent_checks': parent_checks,
        'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in parents},
        'frozen_sha256': preflight['checked_source_sha256'],
        'model_snapshot_sha256': c.sha(OUT / 'model_service_snapshot.json'),
        'pre_inference_checks_sha256': c.sha(OUT / 'pre_inference_checks.json'),
        'old_global_status_sha256': c.sha(c.STATUS), 'logical_request_count': 32, 'api_attempt_cap': 32,
        'max_retry_only_transport_errors': 0, 'fail_fast_on_transport_or_loading_error': True,
        'only_api_model_field_changed_from_baseline': True, 'not_a_parameter_count_only_comparison': True,
        'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    io.save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0,
        'logical_requests_completed': 0, 'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    io.save(OUT / 'campaign_status.json', {'state': 'ready', 'pid': None, 'run_attempts': 0})
    print('14B identity and frozen 32 requests verified; inference not yet started.', flush=True)


def fetch(request):
    rid = request['request_id']
    target = OUT / 'results' / f'{rid}.json'
    if target.exists():
        result = c.read(target)
        assert result['payload_sha256'] == request['payload_sha256']
        return result
    marker = OUT / 'attempts' / f'{rid}_attempt1.started.json'
    assert not marker.exists(), 'An interrupted request is not automatically repeated.'
    budget = c.read(OUT / 'budget.json')
    assert budget['api_attempts_started'] < 32
    io.save(marker, {'request_id': rid, 'attempt': 1, 'started_at_utc': c.stamp(),
                    'payload': request['payload'], 'payload_sha256': request['payload_sha256']})
    budget['api_attempts_started'] += 1
    io.save(OUT / 'budget.json', budget)
    start = time.monotonic()
    try:
        response = io.api('/api/chat', request['payload'], 180)
    except Exception as exc:
        io.save(OUT / 'transport_failures' / f'{rid}_attempt1.json',
                {'request_id': rid, 'error': repr(exc), 'elapsed_seconds': time.monotonic() - start})
        raise
    raw_path = OUT / 'raw_responses' / f'{rid}_attempt1.json'
    raw = {'request_id': rid, 'attempt': 1, 'payload': request['payload'],
           'payload_sha256': request['payload_sha256'], 'response': response,
           'latency_seconds': time.monotonic() - start, 'received_at_utc': c.stamp(),
           'service_error': response.get('error') or (None if response.get('done') is True else 'incomplete response')}
    io.save(raw_path, raw)
    assert not raw['service_error'], raw['service_error']
    result = {'request_id': rid, 'case_id': request['case_id'], 'condition_id': request['condition_id'],
              'payload_sha256': request['payload_sha256'], 'attempt': 1,
              'raw_path': raw_path.relative_to(ROOT).as_posix(), 'latency_seconds': raw['latency_seconds'],
              'outcome': grade(response, request['private_reference']['reference_exit']), 'created_at_utc': c.stamp()}
    io.save(target, result)
    return result


def comparison(summary):
    previous = c.read(BASE / 'summary.json')
    rows = [{'condition': key, 'baseline_correct': previous['groups'][key]['usable_correct'],
             '14b_correct': g['usable_correct'], 'correct_count_change': g['usable_correct'] - previous['groups'][key]['usable_correct'],
             'baseline_mean_latency_seconds': previous['groups'][key]['mean_latency_seconds'],
             '14b_mean_latency_seconds': g['mean_latency_seconds']} for key, g in summary['groups'].items()]
    io.save(OUT / 'comparison.json', {'groups': rows, 'not_parameter_count_only_effect': True,
                                     'independent_evacuation_effect_established': False, 'held_out_replication': False})
    c.csv_write(OUT / 'model_comparison.csv', rows)


def run(limit=32):
    from finalize_auto_evacuation_campaign_review import ledger_check
    m, status = c.read(OUT / 'manifest.json'), c.read(OUT / 'campaign_status.json')
    if status['state'] == 'complete':
        return
    assert not c.pid_live(status.get('pid')), 'Existing worker is still running.'
    io.OUT, io.DIGEST, analysis.OUT = OUT, m['model_digest'], OUT
    io.check_frozen(m)
    assert next(x for x in io.api('/api/tags', timeout=10)['models'] if x['name'] == MODEL)['digest'] == m['model_digest']
    status.update(state='running', pid=os.getpid(), run_attempts=status['run_attempts'] + 1)
    io.save(OUT / 'campaign_status.json', status)
    monitor = Monitor(OUT / f"resource_run_{status['run_attempts']:03d}.json")
    monitor.start()
    try:
        for i, request in enumerate(m['request_order'][:limit]):
            assert Monitor.sample()['available_memory_bytes'] >= 512 * 1024**2, 'Low available memory; stop remaining requests.'
            result = fetch(request)
            io.reconcile_budget()
            if i == 0 and not (OUT / 'loaded_model_snapshot.json').exists():
                io.save(OUT / 'loaded_model_snapshot.json', io.api('/api/ps', timeout=10))
            status.update(completed_logical_requests=i + 1, last_request=request['request_id'], updated_at_utc=c.stamp())
            io.save(OUT / 'campaign_status.json', status)
            print(json.dumps({'completed': i + 1, 'total': 32, 'condition': request['condition_id'],
                              'category': result['outcome']['category'], 'latency_seconds': result['latency_seconds']}), flush=True)
        monitor.stop()
        if limit < 32:
            status.update(state='pilot_complete', pid=None)
            io.save(OUT / 'campaign_status.json', status)
            return
        io.check_frozen(m)
        summary = analysis.summarize(m)
        comparison(summary)
        from verify_llm_goal_role_14b import verify
        io.save(OUT / 'verification.json', verify())
        write_report(OUT, REPORT, m, summary)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('completion_manifest.json', 'campaign_status.json')]
        paths += [ROOT / p for p in m['frozen_sha256']] + [REPORT]
        io.save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
                'budget': c.read(OUT / 'budget.json'), 'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
                'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0})
        ledger_check(OUT / 'completion_manifest.json')
        status.update(state='complete', pid=None, completed_at_utc=c.stamp())
        io.save(OUT / 'campaign_status.json', status)
        print('14B replay independently verified, reported, and sealed.', flush=True)
    except Exception:
        status.update(state='failed', failed_at_utc=c.stamp())
        io.save(OUT / 'campaign_status.json', status)
        io.save(OUT / 'failures' / f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc()})
        raise
    finally:
        monitor.stop()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    if args.pilot: run(1)
    if args.run: run()
