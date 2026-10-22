"""Seal a completed diagnostic without modifying prior models or results."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_q_cache_audit_20261015'
REPORT = ROOT/'reports/SafeStep循环Q值缓存审计_证据表与下一步决定_2026-10-15.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read(OUT/'frozen_manifest.json')
    summary = read(OUT/'summary.json')
    verified = read(OUT/'verification.json')
    assert verified['summary_sha256'] == sha(OUT/'summary.json')
    for group in ['code_sha256', 'protected_previous_files']:
        for path, digest in plan[group].items():
            assert sha(ROOT/path) == digest, path
    aggregates = summary['aggregates']
    total = lambda name: sum(row[name] for row in aggregates.values())
    assert total('post_reveal_steps') == 7233
    assert total('positive_selected_Q_post_reveal') == 6204
    assert total('offline_nonoptimal_actions_post_reveal') == 3130
    training = read(OUT/'training_coverage_and_final_TD.json')
    assert sum(row['positive_selected_Q_all_training'] for row in training.values()) == 85916
    with (OUT/'cycle_Q_coverage_cases.csv').open(encoding='utf-8-sig', newline='') as handle:
        cases = list(csv.DictReader(handle))
    assert len(cases) == 96
    assert all(int(row['training_chosen_action_at_cell']) > 0 and int(row['training_reference_action_at_cell']) > 0 for row in cases)
    links = re.findall(r'\]\((C:/[^)]+)\)', REPORT.read_text(encoding='utf-8'))
    assert links and all(Path(link).exists() for link in links)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']
    files += [ROOT/p for p in plan['code_sha256']]
    files += [Path(__file__), REPORT]
    value = {'status': 'completed_cache_only_cycle_Q_coverage_audit',
        'cached_cases': 96, 'cached_decisions': 7473, 'training_transitions_read': 120000,
        'representative_evidence_rows': 96, 'post_reveal_decisions': 7233,
        'positive_selected_Q_post_reveal': 6204, 'offline_nonoptimal_actions_post_reveal': 3130,
        'training_inputs_with_positive_selected_Q': 85916,
        'independent_forced_action_references_checked': 250, 'new_unit_checks_passed': 3,
        'old_artifacts_verified': len(plan['protected_previous_files']), 'report_links_checked': len(links),
        'historical_TD_targets_not_imputed': True, 'training_cause_not_causally_isolated': True,
        'new_training': 0, 'new_rollouts': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False,
        'decision': 'Audit closed. Value-range and ranking errors established; no directly confirmed implementation bug or repair training started.',
        'report': str(REPORT.relative_to(ROOT)),
        'artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}}
    payload = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    target = OUT/'completion_manifest.json'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
