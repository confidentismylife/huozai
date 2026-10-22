"""Close the fixed-budget exact-label diagnostic, preserving negative results."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_exact_supervision_20261015'
REPORT = ROOT/'reports/SafeStep准确Q监督诊断_固定预算结果与解释_2026-10-15.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan, summary = read(OUT/'frozen_manifest.json'), read(OUT/'summary.json')
    verified = read(OUT/'verification.json')
    repair = read(OUT/'verification_repair.json')
    diagnosis = read(OUT/'cached_scale_and_ranking_diagnosis.json')
    assert verified['summary_sha256'] == sha(OUT/'summary.json')
    assert diagnosis['source_summary_sha256'] == sha(OUT/'summary.json')
    assert repair['adapter_sha256'] == verified['verifier_sha256'] == sha(ROOT/'scripts/verify_safestep_exact_supervision_v2.py')
    for group in ['protected_previous_files', 'code_sha256', 'dataset_sha256']:
        for relative, digest in plan[group].items():
            assert sha(ROOT/relative) == digest, relative
    assert summary['frozen_decisions'] == {'training_fit': False, 'new_state_generalization': False, 'control_usable': False}
    assert verified['control_episodes_recomputed'] == 168
    seeds = [141, 142, 143]
    updates = sum(read(OUT/'models'/f'seed{seed}/training_summary.json')['updates'] for seed in seeds)
    assert updates == summary['supervised_optimizer_updates'] == 29619
    links = re.findall(r'\]\((C:/[^)]+)\)', REPORT.read_text(encoding='utf-8'))
    assert links and all(Path(link).exists() for link in links)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']
    files += [ROOT/p for p in plan['code_sha256']]
    files += [Path(__file__), ROOT/'scripts/verify_safestep_exact_supervision_v2.py',
              ROOT/'scripts/analyze_safestep_exact_supervision_cache.py', REPORT]
    value = {'status': 'completed_exact_Q_supervision_diagnostic_with_failed_acceptance',
        'supervised_models': 3, 'supervised_optimizer_updates': updates,
        'training_states': verified['training_states'], 'new_validation_profiles': 12,
        'new_validation_states': verified['new_validation_states'],
        'independently_checked_legal_Q_labels': verified['legal_Q_labels_checked_by_forward_min_plus_paths'],
        'control_episodes': 168, 'control_steps': verified['control_steps_recomputed'],
        'new_unit_checks_passed': 3, 'acceptance_results': summary['frozen_decisions'],
        'all_three_supervised_control_successes': [summary['control'][f'supervised_Q_seed{seed}']['successes'] for seed in seeds],
        'verifier_path_type_error_repaired_in_separate_adapter': True,
        'frozen_source_verifier_models_labels_and_scores_unchanged': True,
        'report_links_checked': len(links), 'previous_files_verified': len(plan['protected_previous_files']),
        'RL_updates': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False,
        'extra_updates_after_scoring': 0,
        'decision': 'Round closed. Exact labels helped calibration but did not establish fit, ranking generalization or efficient control; no claim of unique bootstrap cause or fundamental network incapacity.',
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
