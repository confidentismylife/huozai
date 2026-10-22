"""Cache-only artifact checks and completion manifest; never performs inference."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    'reports/研究收敛整理_方法结果讨论初稿_2026-10-14.md',
    'reports/真人数据校准行为预测_冻结方案_2026-10-14.md',
    'reports/真人数据校准行为预测_实验结果_2026-10-14.md',
    'reports/DeepSeek低成本行为预测补充_冻结方案_2026-10-14.md',
    'reports/DeepSeek低成本行为预测补充_实验结果_2026-10-14.md']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    missing, checked_links = [], 0
    for relative in REPORTS:
        path = ROOT / relative
        text = path.read_text(encoding='utf-8')
        assert text.startswith('# ') and '2026-10-14' in text
        # Credentials do not belong in manuscript or plan artifacts.
        assert re.search(r'sk-[A-Za-z0-9]{20,}', text) is None
        for target in re.findall(r'\]\((C:/[^)]+)\)', text):
            checked_links += 1
            if not Path(target).is_file(): missing.append(target)
    assert not missing, missing
    folders = ['bristol_calibrated_behavior_20261014', 'deepseek_behavior_extension_20261014']
    summaries, files = [], list(REPORTS)
    for folder in folders:
        base = ROOT / 'outputs' / folder
        summary = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
        verified = json.loads((base / 'verification.json').read_text(encoding='utf-8'))
        plan = json.loads((base / 'frozen_manifest.json').read_text(encoding='utf-8'))
        assert verified['summary_sha256'] == sha(base / 'summary.json')
        assert verified['manifest_sha256'] == sha(base / 'frozen_manifest.json')
        assert verified['all_metrics_independently_recomputed']
        for relative, digest in plan.get('protected_previous_files', plan.get('protected_files')).items():
            assert sha(ROOT / relative) == digest, relative
        ledger = [json.loads(line) for line in (base / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(ledger) == len({r['key'] for r in ledger}) == 34 and all(r['valid'] for r in ledger)
        assert not summary['continuation_gate_passed'] and summary['PPO_updates'] == 0
        assert not summary['old_final_scenarios_accessed'] and not summary['Liao_confirmation_scores_computed']
        summaries.append(summary)
        files.extend(str(p.relative_to(ROOT)).replace('\\', '/') for p in base.iterdir() if p.is_file())
    local, api = summaries
    local_report = (ROOT / REPORTS[2]).read_text(encoding='utf-8')
    api_report = (ROOT / REPORTS[4]).read_text(encoding='utf-8')
    for m in local['methods'].values():
        assert f"{m['condition_macro_NLL']:.6f}" in local_report
        assert f"{m['condition_macro_Brier']:.6f}" in local_report
        assert f"{m['participant_accuracy']*100:.2f}%" in local_report
    for m in api['methods'].values():
        assert f"{m['condition_macro_NLL']:.6f}" in api_report
        assert f"{m['condition_macro_Brier']:.6f}" in api_report
        assert f"{m['participant_accuracy']*100:.2f}%" in api_report
    fee = api['estimated_peak_all_miss_usd']
    assert f'{fee:.7f}' in api_report and fee < api['budget_usd']
    files.extend(['schemas/bristol_calibrated_behavior_20261014.json',
        'schemas/deepseek_behavior_extension_20261014.json',
        'scripts/run_bristol_calibrated_behavior.py', 'scripts/verify_bristol_calibrated_behavior.py',
        'scripts/run_deepseek_behavior_extension.py', 'scripts/verify_deepseek_behavior_extension.py',
        'scripts/deepseek_private_client.py', 'scripts/finalize_behavior_reports_20261014.py'])
    result = {'status': 'research_consolidation_and_bounded_trials_completed',
        'reports': REPORTS, 'report_local_links_checked': checked_links,
        'report_core_metrics_match_verified_summaries': True, 'new_generation_requests': 68,
        'local_requests': 34, 'DeepSeek_requests': 34, 'new_Logit_fits': 9,
        'DeepSeek_estimated_peak_all_miss_usd': fee, 'both_fixed_configurations_closed': True,
        'no_further_prompt_search_planned_on_these_outcomes': True,
        'PPO_updates': 0, 'old_final_scenarios_accessed': False, 'Liao_confirmation_scores_computed': False,
        'credential_in_report_or_manifest': False,
        'artifact_sha256': {p: sha(ROOT / p) for p in sorted(set(files))}}
    destination = ROOT / 'outputs/behavior_research_closure_20261014'
    destination.mkdir(exist_ok=True)
    (destination / 'completion_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
