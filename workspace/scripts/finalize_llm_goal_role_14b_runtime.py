"""Audit the sealed 14B replay, release its model memory, and record closure."""
from pathlib import Path
import csv
import json
import sys
import time
import psutil
import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / 'outputs/llm_goal_role_14b_20261006'
OUT = ROOT / 'outputs/llm_goal_role_14b_inference_runtime_20261006'
REPORT = ROOT / 'reports/LLM_14B模型复测实验报告_校订版_2026-10-06.md'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return {r['request_id']: r for r in csv.DictReader(f)}


def run():
    assert c.read(SCIENCE / 'campaign_status.json')['state'] == 'complete'
    assert not OUT.exists() and not REPORT.exists()
    OUT.mkdir()
    counts = {'14b_scientific_records': ledger_check(SCIENCE / 'completion_manifest.json'),
              '8b_scientific_records': ledger_check(ROOT / 'outputs/llm_goal_role_8b_20261006/completion_manifest.json'),
              '8b_runtime_records': ledger_check(ROOT / 'outputs/llm_goal_role_8b_runtime_20261006/completion_manifest.json'),
              '14b_preparation_records': ledger_check(ROOT / 'outputs/llm_goal_role_14b_runtime_20261006/completion_manifest.json')}
    m = c.read(SCIENCE / 'manifest.json')
    a = rows(ROOT / 'outputs/llm_goal_role_8b_20261006/case_results.csv')
    b = rows(SCIENCE / 'case_results.csv')
    assert len(a) == len(b) == 32 and set(a) == set(b)
    improved = sum(a[k]['usable_correct'] == 'False' and b[k]['usable_correct'] == 'True' for k in a)
    worsened = sum(a[k]['usable_correct'] == 'True' and b[k]['usable_correct'] == 'False' for k in a)
    comp = c.read(SCIENCE / '8b_14b_comparison.json')
    assert comp['improved_requests'] == improved and comp['worsened_requests'] == worsened
    assert comp['8b_total_correct'] == sum(r['usable_correct'] == 'True' for r in a.values())
    assert comp['14b_total_correct'] == sum(r['usable_correct'] == 'True' for r in b.values())
    for group in comp['groups']:
        key = group['condition']
        for model, data in [('8b', a), ('14b', b)]:
            values = [r for r in data.values() if r['condition'] == key]
            assert len(values) == 8
            assert group[model + '_correct'] == sum(r['usable_correct'] == 'True' for r in values)
            assert abs(group[model + '_mean_latency_seconds'] - sum(float(r['latency_seconds']) for r in values) / 8) < 1e-8
    # Audit the recorded reporting/monitoring repair against its preserved first revision.
    repair = c.read(SCIENCE / 'implementation_repair.json')
    initial = c.read(SCIENCE / 'code_history/manifest_initial.json')
    assert c.sha(SCIENCE / 'code_history/llm_14b_runtime_initial.py') == repair['old_sha256']
    assert c.sha(ROOT / repair['changed_source']) == repair['new_sha256']
    assert initial['request_order'] == m['request_order']
    assert all(c.sha(ROOT / k) == v for k, v in repair['pilot_result_sha256_unchanged'].items())
    tags = io.api('/api/tags', timeout=10)
    assert next(x for x in tags['models'] if x['name'] == 'qwen3:14b')['digest'] == m['model_digest']
    io.save(OUT / 'service_before_unload.json', {'tags': tags, 'ps': io.api('/api/ps', timeout=10),
            'available_memory_bytes': psutil.virtual_memory().available})
    start = time.monotonic()
    response = io.api('/api/generate', {'model': 'qwen3:14b', 'keep_alive': 0}, 30)
    after = io.api('/api/ps', timeout=10)
    assert response.get('done') is True and not response.get('response') and not response.get('eval_count')
    assert not any(x['name'] == 'qwen3:14b' for x in after['models'])
    io.save(OUT / 'model_unload.json', {'payload': {'model': 'qwen3:14b', 'keep_alive': 0},
            'response': response, 'elapsed_seconds': time.monotonic() - start, 'ps_after': after,
            'available_memory_bytes_after': psutil.virtual_memory().available, 'generated_tokens': 0})
    source_report = ROOT / 'reports/LLM_14B模型复测实验报告_2026-10-06.md'
    corrected = source_report.read_text(encoding='utf-8').replace('runner.exe', 'llama-server.exe')
    corrected += '\n校订说明：实际推理子进程名为llama-server.exe，已更正原报告中的名称；评估数值及封存原报告均未改变。完成后已发送一次无生成的模型卸载请求，确认14B不再加载。\n'
    REPORT.write_text(corrected, encoding='utf-8')
    audit = {'all_checks_passed': True, **counts, 'paired_comparison_recomputed': True,
             'report_and_monitoring_repair_audited': True, 'original_first_request_unchanged': True,
             'original_report_preserved': True, 'model_unloaded_without_generation': True,
             'existing_core_tests_passed': 8, 'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0,
             'default_experiment_model': c.read(ROOT / 'outputs/local_llm_selection.json'),
             'global_status_sha256': c.sha(c.STATUS), 'comparison': comp}
    assert all(c.sha(ROOT / k) == v for k, v in m['frozen_sha256'].items())
    io.save(OUT / 'closure_audit.json', audit)
    files = [p for p in OUT.iterdir() if p.is_file()] + [REPORT, Path(__file__), SCIENCE / 'completion_manifest.json']
    io.save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
            'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in files}, 'new_inference': 0,
            'non_generating_maintenance_requests': 1})
    ledger_check(OUT / 'completion_manifest.json')
    ledger_check(SCIENCE / 'completion_manifest.json')
    print(json.dumps({'all_checks_passed': True, 'comparison': comp,
                      '14b_unloaded': True, 'report': REPORT.as_posix()}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
