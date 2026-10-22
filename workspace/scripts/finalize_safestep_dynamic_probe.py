"""Seal completed bounded probe and report, preserving all previous artifacts."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_dynamic_probe_20261014'
REPORT = ROOT / 'reports/SafeStep动态观测最小验证_方案结果与后续决定_2026-10-14.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    plan = read(OUT / 'frozen_manifest.json')
    summary = read(OUT / 'summary.json')
    verified = read(OUT / 'verification.json')
    assert verified['summary_sha256'] == sha(OUT / 'summary.json')
    assert all(summary['frozen_decisions'].values())
    assert summary['episodes'] == verified['episodes_recomputed'] == 144
    for group in ['protected_previous_files', 'code_sha256', 'field_sha256']:
        for path, digest in plan[group].items():
            assert sha(ROOT / path) == digest, path
    links = re.findall(r'\]\((C:/[^)]+)\)', REPORT.read_text(encoding='utf-8'))
    assert links and all(Path(link).exists() for link in links)
    traces = [json.loads(line) for line in (OUT / 'traces.jsonl').read_text(encoding='utf-8').splitlines()]
    indexed = {(row['scenario_id'], row['method']): row for row in traces}
    # Check the report's key behavioral equivalence from every actual action.
    for scenario in plan['scenarios']:
        sid = scenario['scenario_id']
        fixed = [r['action'] for r in indexed[(sid, 'fixed_left')]['trace']]
        for seed in [41, 42, 43]:
            assert fixed == [r['action'] for r in indexed[(sid, f'legacy_DQN_seed{seed}')]['trace']]
    files = [path for path in OUT.rglob('*') if path.is_file() and path.name != 'completion_manifest.json']
    files += [ROOT / path for path in plan['code_sha256']]
    files += [Path(__file__), REPORT]
    result = {'status': 'completed_frozen_synthetic_dynamic_observation_probe',
        'planned_and_completed_rollouts': 144, 'scenarios': 16,
        'unique_relative_hazard_patterns': 8, 'new_unit_checks_passed': 4,
        'independently_recomputed_steps': verified['steps_recomputed'],
        'frozen_mechanism_criteria_passed': True,
        'all_legacy_main_actions_identical_to_fixed_left': True,
        'report_links_checked': len(links), 'previous_files_verified': len(plan['protected_previous_files']),
        'new_training': 0, 'LLM_requests': 0, 'PPO_updates': 0,
        'old_final_scenarios_accessed': False, 'future_work_started': False,
        'decision': 'Round closed. Keep rolling hazard-weighted planning as a serious baseline; define common observations before considering new RL training.',
        'report': str(REPORT.relative_to(ROOT)),
        'artifact_sha256': {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(files))}}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    target = OUT / 'completion_manifest.json'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
