"""Report/file closure only; no training or model requests."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_reproduction_20261014'
REPORTS = ['reports/SafeStep基线复现_来源核对与冻结方案_2026-10-14.md',
           'reports/SafeStep基线复现_首轮结果与代码说明_2026-10-14.md']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    verified = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    assert verified['summary_sha256'] == sha(OUT / 'summary.json')
    assert verified['all_start_diagnostics_independently_recomputed'] == 132
    assert summary['episodes'] == 12000 and not summary['paper_numeric_targets_reproduced']
    checked = 0
    for rel in REPORTS:
        content = (ROOT / rel).read_text(encoding='utf-8')
        for target in re.findall(r'\]\((C:/[^)]+)\)', content):
            assert Path(target).is_file(), target
            checked += 1
    content = (ROOT / REPORTS[1]).read_text(encoding='utf-8')
    for row in summary['RoomB_results']:
        if row['personal_FED'] is not None: assert f"{row['personal_FED']:.6f}" in content
    assert '124' in content and '132' in content and '8条' in content
    historical_path = ROOT / 'outputs/behavior_research_closure_20261014/completion_manifest.json'
    protected = json.loads(historical_path.read_text(encoding='utf-8'))['artifact_sha256']
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    files = list(REPORTS) + ['schemas/safestep_reproduction_20261014.json',
        'src/fire_evac/safestep_baseline.py', 'scripts/run_safestep_reproduction.py',
        'scripts/verify_safestep_reproduction.py', 'scripts/plot_safestep_reproduction.py',
        'tests/test_safestep_baseline.py']
    files += [str(p.relative_to(ROOT)).replace('\\', '/') for p in OUT.rglob('*') if p.is_file()
              and p.name != 'completion_manifest.json']
    result = {'status': 'completed_first_method_reimplementation_stage', 'reports': REPORTS,
        'training_runs': 6, 'training_episodes': 12000, 'unit_checks_passed': 7,
        'main_start_successes': 6, 'main_start_total': 6, 'all_start_successes': 124, 'all_start_total': 132,
        'report_links_checked': checked, 'previous_behavior_research_artifacts_unchanged': True,
        'original_FDS_numeric_reproduction': False, 'remaining_external_dependency': 'Original paper data and code-level implementation details',
        'LLM_requests': 0, 'PPO_updates': 0, 'old_final_scenarios_accessed': False,
        'artifact_sha256': {p: sha(ROOT / p) for p in sorted(set(files))}}
    (OUT / 'completion_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['reports', 'artifact_sha256']}, indent=2))


if __name__ == '__main__': main()
