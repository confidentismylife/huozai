"""Close the bounded round including its failed preregistered criteria."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_shared_observation_20261014'
REPORT = ROOT / 'reports/SafeStep统一观测DQN有限对照_结果与停止决定_2026-10-14.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read(OUT / 'frozen_manifest.json')
    summary = read(OUT / 'summary.json')
    verified = read(OUT / 'verification.json')
    diagnosis = read(OUT / 'cached_behavior_diagnosis.json')
    assert verified['summary_sha256'] == sha(OUT / 'summary.json')
    assert diagnosis['traces_sha256'] == sha(OUT / 'evaluation_traces.jsonl')
    for group in ['protected_previous_files', 'code_sha256', 'episode_schedules_sha256']:
        for relative, digest in plan[group].items():
            assert sha(ROOT / relative) == digest, relative
    assert verified['training_transitions_reconstructed'] == 240000
    assert verified['evaluation_episodes_reconstructed'] == 288
    assert len(diagnosis['failed_confirmation_episodes']) == 3
    assert all(r['method'] == 'hazard_full_seed143' for r in diagnosis['failed_confirmation_episodes'])
    # Preserve negative outcomes as completed findings, not as failed execution.
    assert summary['frozen_decisions'] == {'observation_signal_supported': False,
        'practical_RL_added_value': False, 'near_planner': False}
    links = re.findall(r'\]\((C:/[^)]+)\)', REPORT.read_text(encoding='utf-8'))
    assert links and all(Path(link).exists() for link in links)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']
    files += [ROOT / relative for relative in plan['code_sha256']]
    files += [Path(__file__), ROOT / 'scripts/analyze_safestep_shared_observation_cache.py', REPORT]
    result = {'status': 'completed_one_fixed_budget_round_with_negative_acceptance_results',
        'trained_models': 6, 'training_transitions': 240000, 'gradient_updates': 59238,
        'evaluation_episodes': 288, 'evaluation_steps': verified['evaluation_steps_reconstructed'],
        'new_unit_checks_passed': 4, 'confirmation_full_model_successes_by_seed': [24, 24, 21],
        'confirmation_planner_successes': 24, 'confirmation_failures_not_dropped': 3,
        'frozen_acceptance_criteria': summary['frozen_decisions'],
        'posthoc_diagnosis_from_cache_only': True, 'report_links_checked': len(links),
        'old_artifacts_verified': len(plan['protected_previous_files']),
        'LLM_requests': 0, 'PPO_updates': 0, 'old_final_scenarios_accessed': False,
        'additional_training_after_results': 0,
        'decision': 'Stop this round. Keep snapshot replanning as the engineering baseline; no RL or LLM added-value claim.',
        'report': str(REPORT.relative_to(ROOT)),
        'artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}}
    payload = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    target = OUT / 'completion_manifest.json'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
