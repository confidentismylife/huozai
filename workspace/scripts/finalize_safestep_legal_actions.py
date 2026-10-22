"""Complete the fixed-weight audit report without altering any source artifact."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_legal_action_audit_20261014'
REPORTS = ['reports/SafeStep失败定位与合法动作修正_2026-10-14.md',
           'reports/SafeStep原始资料申请清单与邮件草稿_2026-10-14.md']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    verified = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    assert verified['summary_sha256'] == sha(OUT / 'summary.json')
    assert verified['episodes_recomputed'] == 264
    assert summary['paired']['rescued'] == 8 and summary['paired']['successes_lost'] == 0
    assert summary['paired']['original_success_action_chains_changed'] == 0
    assert summary['variants']['geometric_legal_greedy']['invalid_moves'] == 0
    for path, digest in plan['protected_previous_files'].items(): assert sha(ROOT / path) == digest, path
    links = 0
    for relative in REPORTS:
        text = (ROOT / relative).read_text(encoding='utf-8')
        for target in re.findall(r'\]\((C:/[^)]+)\)', text):
            assert Path(target).is_file(), target
            links += 1
    report = (ROOT / REPORTS[0]).read_text(encoding='utf-8')
    for pair in summary['paired']['recovered_cases']: assert f"{pair['masked_FED']:.6f}" in report
    files = [*REPORTS, 'schemas/safestep_legal_action_audit_20261014.json',
        'src/fire_evac/safestep_execution_checks.py', 'scripts/audit_safestep_legal_actions.py',
        'scripts/verify_safestep_legal_actions.py', 'tests/test_safestep_execution_checks.py']
    files += [str(p.relative_to(ROOT)).replace('\\', '/') for p in OUT.iterdir()
              if p.is_file() and p.name != 'completion_manifest.json']
    result = {'status': 'fixed_weight_execution_repair_and_data_request_draft_completed',
        'paired_cases': 132, 'episodes': 264, 'original_successes': 124, 'masked_successes': 132,
        'failures_recovered': 8, 'original_success_traces_changed': 0, 'new_tests_passed': 4,
        'training_visitation_counts_unavailable': True, 'data_request_email_drafted_locally': True,
        'emails_sent': 0, 'new_DQN_training': 0, 'PPO_updates': 0, 'LLM_requests': 0,
        'old_final_scenarios_accessed': False, 'report_links_checked': links,
        'original_artifacts_and_weights_unchanged': True, 'reports': REPORTS,
        'artifact_sha256': {p: sha(ROOT / p) for p in sorted(set(files))}}
    (OUT / 'completion_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['artifact_sha256', 'reports']}, indent=2))


if __name__ == '__main__': main()
