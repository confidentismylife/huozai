"""Finish the remaining fixed cases using smaller batches and per-case unload."""
from pathlib import Path
import copy
import json
import os
import sys
import traceback

import psutil
import auto_evacuation_campaign as c
import run_qwen35_27b_trial as helper
from run_qwen35_27b_text_continuation import independently_score
from run_llm_output_mechanism import save, hash_value
from verify_llm_output_mechanism import classify
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/llm_goal_role_20261006'
OLD = ROOT / 'outputs/qwen35_27b_trial_20261006'
PREVIOUS = ROOT / 'outputs/qwen35_27b_text_continuation_20261006'
CREATION = ROOT / 'outputs/qwen35_27b_text_creation_20261006'
SERIAL = ROOT / 'outputs/qwen35_27b_text_serial_unload_20261006'
OUT = ROOT / 'outputs/qwen35_27b_text_low_memory_20261006'
MODEL = 'qwen35-27b-text-q4:latest'
PROTOCOL = ROOT / 'reports/Qwen3.5_27B小上下文续测方案_2026-10-06.md'
REPORT = ROOT / 'reports/Qwen3.5_27B最终运行情况_2026-10-06.md'


def prepare():
    assert not OUT.exists()
    assert c.read(OLD / 'budget.json')['logical_requests_completed'] == 3
    assert c.read(PREVIOUS / 'budget.json')['logical_requests_completed'] == 13
    assert c.read(SERIAL / 'budget.json')['logical_requests_completed'] == 1
    assert not helper.api('/api/ps')['models']
    parents = [OLD / 'completion_manifest.json', PREVIOUS / 'completion_manifest.json', CREATION / 'completion_manifest.json', SERIAL / 'completion_manifest.json']
    checks = {p.relative_to(ROOT).as_posix(): ledger_check(p) for p in parents}
    base = c.read(BASE / 'manifest.json')
    order = copy.deepcopy(base['request_order'][17:])
    assert len(order) == 15
    for req in order:
        req['payload']['model'] = MODEL
        req['payload']['options']['num_batch'] = 32
        req['payload']['options']['num_ctx'] = 2048
        req['payload_sha256'] = hash_value(req['payload'])
    OUT.mkdir()
    snapshot = {'show': helper.api('/api/show', {'model': MODEL}), 'tags': helper.api('/api/tags'), 'version': helper.api('/api/version')}
    save(OUT / 'model_service_snapshot.json', snapshot)
    sources = [Path(__file__), ROOT / 'scripts/run_qwen35_27b_trial.py', ROOT / 'scripts/run_qwen35_27b_text_continuation.py',
        ROOT / 'scripts/llm_14b_runtime.py', ROOT / 'scripts/llm_output_mechanism_core.py',
        ROOT / 'scripts/verify_llm_output_mechanism.py', PROTOCOL, ROOT / 'outputs/local_llm_selection.json']
    save(OUT / 'manifest.json', {'model_name': MODEL, 'request_order': order, 'base_manifest': (BASE / 'manifest.json').relative_to(ROOT).as_posix(),
        'base_manifest_sha256': helper.sha(BASE / 'manifest.json'), 'parent_checks': checks,
        'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in sources},
        'model_snapshot_sha256': helper.sha(OUT / 'model_service_snapshot.json'), 'old_global_status_sha256': helper.sha(c.STATUS),
        'previous_covered': 17, 'attempt_cap': 15, 'memory_floor_before_request_bytes': 1024**3,
        'sampled_low_memory_stop_after_unload_bytes': 256*1024**2, 'num_batch': 32, 'num_ctx': 2048, 'no_automatic_retries': True})
    save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0, 'logical_requests_completed': 0,
                             'simulations': 0, 'fits': 0, 'ppo_updates': 0})


def unload(rid):
    before = helper.api('/api/ps')
    response = None
    if any(x['name'] == MODEL for x in before['models']):
        response = helper.api('/api/generate', {'model': MODEL, 'keep_alive': 0})
        assert response.get('done') and not response.get('eval_count') and not response.get('response')
    after = helper.api('/api/ps')
    assert not any(x['name'] == MODEL for x in after['models'])
    save(OUT / 'unloads' / (rid + '.json'), {'ps_before': before, 'response': response, 'ps_after': after,
        'available_memory_bytes_after': psutil.virtual_memory().available, 'generated_tokens': 0})
    return before


def finish(m, status, error):
    helper.frozen(m)
    rows = []
    for folder, phase in [(OLD, 'original_definition'), (PREVIOUS, 'text_resident'), (SERIAL, 'text_batch64_unload'), (OUT, 'text_batch32_ctx2048_unload')]:
        plan = c.read(folder / 'manifest.json')
        for req in plan['request_order']:
            p = folder / 'results' / (req['request_id'] + '.json')
            if not p.exists():
                continue
            result = c.read(p)
            raw = c.read(ROOT / result['raw_path'])
            score = classify(raw['response'], req['private_reference']['reference_exit'])
            assert all(result['outcome'][k] == v for k, v in score.items())
            if folder == OUT:
                independently_score(req, result)
            rows.append({'request_id': req['request_id'], 'condition': req['condition_id'], 'phase': phase,
                         'correct': score['usable_correct'], 'latency_seconds': result['latency_seconds']})
    assert len({x['request_id'] for x in rows}) == len(rows)
    assert len(rows) == 17+c.read(OUT / 'budget.json')['logical_requests_completed']
    c.csv_write(OUT / 'all_phases_results.csv', rows)
    metrics = c.read(OUT / 'resource_samples.json')
    summary = {'covered': len(rows), 'correct': sum(x['correct'] for x in rows), 'planned': 32,
        'all_32_covered': len(rows) == 32, 'single_configuration_benchmark': False, 'rows': rows,
        'stop_reason': error, 'minimum_available_memory_bytes': metrics['minimum_available_memory_bytes'],
        'maximum_system_pagefile_used_bytes': metrics['maximum_system_pagefile_used_bytes'],
        'new_fits': 0, 'ppo_updates': 0}
    save(OUT / 'summary.json', summary)
    lines = ['# Qwen3.5 27B分阶段完成情况', '', '日期：2026-10-06（北京时间）。', '',
        f"状态：{status['state']}。共覆盖{len(rows)}/32条，其中符合数学目标{summary['correct']}条。", '',
        '使用同一27B Q4_K_M文本权重。第一阶段3条使用原定义，第二阶段13条使用无视觉组件的纯文本定义，第三阶段1条使用num_batch64逐条卸载；第四阶段剩余15条使用num_batch32、num_ctx2048逐条卸载。没有重复已交付题目。', '',
        '纯文本模型名qwen35-27b-text-q4:latest；没有projector。去掉视觉组件不改变文本权重精度。第三阶段更改计算批次与驻留策略；第四阶段进一步将num_ctx设为2048、num_batch设为32。题目与schema未删减，temperature0、seed2026100601、num_predict1024和think=false保持原计划。', '',
        '这是一套固定题目的分阶段覆盖，不是同一配置下的32题基准。计算批次、缓存和后台内存状态不同，不能将结果差异直接归因于视觉移除或参数规模。', '',
        f"第四阶段采样最低可用内存{metrics['minimum_available_memory_bytes']/1e6:.1f}MB；整机分页文件使用量最高{metrics['maximum_system_pagefile_used_bytes']/1e9:.2f}GB。", '',
        '| 条件 | 已交付 | 符合目标 |', '| --- | --- | --- |']
    for cid in ['minimize_hazard', 'maximize_hazard', 'minimize_clear', 'maximize_clear']:
        group = [r for r in rows if r['condition'] == cid]
        lines.append(f"| {cid} | {len(group)} | {sum(r['correct'] for r in group)} |")
    if error:
        lines += ['', '停止原因：', '', '```text', error, '```']
    lines += ['', '各阶段使用独立原始响应、预算、内存采样、卸载记录和完整性台账。无付费API、拟合、疏散模拟或PPO更新。原8B、14B记录、默认模型选择与全局研究状态不变。', '',
        f'[第四阶段固定方案]({PROTOCOL.as_posix()})；[全阶段逐题结果]({(OUT / "all_phases_results.csv").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']+[REPORT]
    paths += [ROOT / x for x in m['frozen_sha256']]
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'state': status['state'],
        'files': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in paths}, 'all_32_covered': len(rows) == 32})
    ledger_check(OUT / 'completion_manifest.json')
    print(json.dumps({'state': status['state'], 'covered': len(rows), 'correct': summary['correct'], 'report': str(REPORT)}, ensure_ascii=False), flush=True)


def run():
    prepare()
    helper.OUT, helper.MODEL = OUT, MODEL
    m = c.read(OUT / 'manifest.json')
    helper.frozen(m)
    resource = helper.Resources()
    resource.start()
    status = {'state': 'running', 'pid': os.getpid(), 'completed_logical_requests': 0}
    save(OUT / 'campaign_status.json', status)
    error = None
    try:
        for i, req in enumerate(m['request_order']):
            assert psutil.virtual_memory().available >= m['memory_floor_before_request_bytes'], 'Available memory below 1GiB before reload.'
            start_sample = len(resource.samples)
            result = helper.fetch(req)
            independently_score(req, result)
            observed = unload(req['request_id'])
            if i == 0:
                save(OUT / 'loaded_model_snapshot.json', observed)
            status.update(completed_logical_requests=i+1, last_request=req['request_id'], at_utc=c.stamp())
            save(OUT / 'campaign_status.json', status)
            print(json.dumps({'new_completed': i+1, 'new_total': 15, 'covered_with_previous': i+18,
                'category': result['outcome']['category'], 'latency_seconds': result['latency_seconds'],
                'memory_after_unload_GB': psutil.virtual_memory().available/1e9}), flush=True)
            window = resource.samples[start_sample:]
            if window and min(s['available_memory_bytes'] for s in window) < m['sampled_low_memory_stop_after_unload_bytes']:
                raise RuntimeError('Available memory sampled below 256MiB during request; stop after unload.')
        status['state'] = 'complete'
    except Exception as exc:
        error = str(exc)
        status['state'] = 'stopped_after_failure'
        save(OUT / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        print('Serial continuation stopped: '+error, flush=True)
    finally:
        unload('final_maintenance')
        resource.stop()
        status.update(pid=None, finished_at_utc=c.stamp())
        save(OUT / 'campaign_status.json', status)
        finish(m, status, error)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
