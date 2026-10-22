"""Frozen goal-role experiment; lifecycle derived from preserved prior runner."""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import traceback

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
from llm_goal_role_core import CONDITIONS, cases, payload, private_reference
from llm_goal_role_analysis import summarize, write_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_goal_role_20261006'
RUNTIME = ROOT / 'outputs/llm_goal_role_runtime_20261006'
PROTOCOL = ROOT / 'reports/LLM任务目标与字段角色对照固定方案_2026-10-06.md'
REPORT = ROOT / 'reports/LLM任务目标与字段角色对照实验报告_2026-10-06.md'
PREVIOUS = ROOT / 'outputs/llm_information_certainty_20261006'
io.OUT = OUT
assert io.OUT != PREVIOUS

def prepare():
    assert not OUT.exists()
    preflight = c.read(RUNTIME/'pre_inference_checks.json')
    assert preflight['all_checks_passed']
    for name, digest in preflight['checked_source_sha256'].items():
        assert c.sha(ROOT/name) == digest, name
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    parents = [ROOT / p for p in c.read(PREVIOUS / 'manifest.json')['parent_completion_sha256']]
    parents += [PREVIOUS / 'completion_manifest.json',
                ROOT / 'outputs/llm_information_certainty_review_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_information_certainty_runtime_20261006/completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p, HISTORY if p.parent == MAIN else None) for p in parents}
    tags = io.api('/api/tags', timeout=10)
    model = next(v for v in tags['models'] if v['name'] == 'qwen3:4b')
    assert model['digest'] == io.DIGEST
    snapshot = {'tags': tags, 'version': io.api('/api/version', timeout=10),
                'show': io.api('/api/show', {'model': 'qwen3:4b'}, 10)}
    records = cases()
    requests = []
    rng = random.Random(2026100604)
    families = sorted({s['pair_id'] for s in records})
    rng.shuffle(families)
    for family in families:
        block = [(s, cond) for s in records if s['pair_id'] == family for cond in CONDITIONS]
        rng.shuffle(block)
        for s, cond in block:
            body = payload(s, cond)
            requests.append({'request_id': f"{s['case_id']}_{cond['id']}", 'case_id': s['case_id'],
                'pair_id': family, 'condition_id': cond['id'], 'payload': body,
                'payload_sha256': io.hash_value(body), 'private_reference': private_reference(s, cond)})
    assert len(requests) == 32 and len(records) == 8 and len(families) == 4
    paths = [Path(__file__), ROOT/'scripts/llm_goal_role_core.py',
        ROOT/'scripts/verify_llm_goal_role.py', ROOT/'scripts/llm_goal_role_analysis.py', ROOT/'tests/test_llm_goal_role_core.py', PROTOCOL,
        ROOT/'scripts/run_llm_output_mechanism.py', ROOT/'scripts/llm_output_mechanism_core.py',
        ROOT/'scripts/verify_llm_output_mechanism.py', ROOT/'scripts/auto_evacuation_campaign.py',
        ROOT/'scripts/finalize_auto_evacuation_campaign_review.py']
    OUT.mkdir()
    io.save(OUT/'pre_inference_checks.json', preflight)
    io.save(OUT/'model_service_snapshot.json', snapshot)
    io.save(OUT/'manifest.json', {'role': 'fixed_goal_and_equivalent_role_probe', 'created_at_utc': c.stamp(),
        'local_date': '2026-10-06', 'cases': records, 'conditions': list(CONDITIONS), 'request_order': requests,
        'model_digest': io.DIGEST, 'logical_request_count': 32, 'api_attempt_cap': 64,
        'request_order_seed': 2026100604, 'parent_checks': parent_checks,
        'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
        'model_snapshot_sha256': c.sha(OUT/'model_service_snapshot.json'),
        'pre_inference_checks_sha256': c.sha(OUT/'pre_inference_checks.json'),
        'old_global_status_sha256': c.sha(c.STATUS), 'same_output_settings_as_prior_rounds': True,
        'process_local_io_destination': OUT.relative_to(ROOT).as_posix(),
        'single_user_message_per_request': True, 'wrong_invalid_empty_truncated_are_not_retried': True,
        'simulations': 0, 'ppo_updates': 0, 'fits': 0, 'max_retry_only_transport_errors': 1,
        'mathematical_cost_not_real_evacuations': True, 'historical_root_cause_not_established_by_before_after': True})
    io.save(OUT/'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0,
        'logical_requests_completed': 0, 'simulations': 0, 'ppo_updates': 0, 'fits': 0})
    io.save(OUT/'campaign_status.json', {'state': 'ready', 'stage': 'prepared', 'pid': None,
        'run_attempts': 0, 'created_at_utc': c.stamp(), 'completed_logical_requests': 0})
    print('32 goal-role requests frozen; no inference yet.', flush=True)


def run():
    m, status = c.read(OUT/'manifest.json'), c.read(OUT/'campaign_status.json')
    if status['state'] == 'complete':
        return
    if status.get('pid') != os.getpid() and c.pid_live(status.get('pid')):
        print('Active worker preserved; no duplicate requests.', flush=True)
        return
    io.check_frozen(m)
    assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == 'qwen3:4b')['digest'] == io.DIGEST
    status.update(state='running', stage='inference', pid=os.getpid(), run_attempts=status['run_attempts']+1)
    io.save(OUT/'campaign_status.json', status)
    try:
        for i, req in enumerate(m['request_order']):
            result = io.fetch(req, req['private_reference']['reference_exit'])
            io.reconcile_budget()
            status.update(completed_logical_requests=i+1, last_request=req['request_id'], updated_at_utc=c.stamp())
            io.save(OUT/'campaign_status.json', status)
            print(json.dumps({'completed': i+1, 'total': 32, 'condition': req['condition_id'],
                'category': result['outcome']['category']}), flush=True)
        io.check_frozen(m)
        assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == 'qwen3:4b')['digest'] == io.DIGEST
        summary = summarize(m)
        from verify_llm_goal_role import verify
        io.save(OUT/'verification.json', verify())
        write_report(m, summary)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('completion_manifest.json', 'campaign_status.json') and not p.name.endswith('.tmp')]
        paths += [ROOT/p for p in m['frozen_sha256']] + [REPORT]
        io.save(OUT/'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
            'budget': c.read(OUT/'budget.json'), 'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(set(paths))},
            'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0})
        from finalize_auto_evacuation_campaign_review import ledger_check
        ledger_check(OUT/'completion_manifest.json')
        status.update(state='complete', stage='complete', completed_at_utc=c.stamp())
        io.save(OUT/'campaign_status.json', status)
        print('Goal-role experiment sealed and complete.', flush=True)
    except Exception:
        status.update(state='failed', failed_at_utc=c.stamp())
        io.save(OUT/'campaign_status.json', status)
        io.save(OUT/'failures'/f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc(), 'budget': c.read(OUT/'budget.json')})
        raise

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
