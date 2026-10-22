"""Continue 29 requests with identical Q4 text weights and no vision projector."""
from pathlib import Path
import copy
import json
import os
import sys
import time
import traceback

import psutil
import auto_evacuation_campaign as c
import run_qwen35_27b_trial as helper
from run_llm_output_mechanism import save, hash_value
from verify_llm_output_mechanism import classify
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_27b_text_continuation_20261006'
BASE = ROOT / 'outputs/llm_goal_role_20261006'
OLD = ROOT / 'outputs/qwen35_27b_trial_20261006'
CREATION = ROOT / 'outputs/qwen35_27b_text_creation_20261006'
MODEL = 'qwen35-27b-text-q4:latest'
PROTOCOL = ROOT / 'reports/Qwen3.5_27B纯文本续测方案_2026-10-06.md'
REPORT = ROOT / 'reports/Qwen3.5_27B纯文本续测报告_2026-10-06.md'


def prepare():
    assert not OUT.exists(), 'Preserve any prior continuation.'
    assert c.read(OLD / 'budget.json')['logical_requests_completed'] == 3
    assert not helper.api('/api/ps')['models']
    identity = c.read(CREATION / 'text_model_identity.json')
    assert identity['model'] == MODEL and identity['projector_absent'] and identity['text_weights_unchanged']
    parents = [CREATION / 'completion_manifest.json', OLD / 'completion_manifest.json', BASE / 'completion_manifest.json']
    checks = {p.relative_to(ROOT).as_posix(): ledger_check(p) for p in parents}
    base = c.read(BASE / 'manifest.json')
    order = copy.deepcopy(base['request_order'][3:])
    assert len(order) == 29
    for req in order:
        req['payload']['model'] = MODEL
        req['payload_sha256'] = hash_value(req['payload'])
    snapshot = {'show': helper.api('/api/show', {'model': MODEL}), 'version': helper.api('/api/version'), 'tags': helper.api('/api/tags')}
    tag = next(x for x in snapshot['tags']['models'] if x['name'] == MODEL)
    assert tag['digest'] == identity['manifest_digest']
    OUT.mkdir()
    save(OUT / 'model_service_snapshot.json', snapshot)
    sources = [Path(__file__), ROOT / 'scripts/run_qwen35_27b_trial.py', ROOT / 'scripts/llm_14b_runtime.py',
        ROOT / 'scripts/llm_output_mechanism_core.py', ROOT / 'scripts/verify_llm_output_mechanism.py',
        ROOT / 'scripts/run_llm_output_mechanism.py', PROTOCOL, ROOT / 'outputs/local_llm_selection.json']
    save(OUT / 'manifest.json', {'model_name': MODEL, 'model_digest': tag['digest'], 'request_order': order,
        'cases': base['cases'], 'conditions': base['conditions'], 'base_manifest': (BASE / 'manifest.json').relative_to(ROOT).as_posix(),
        'base_manifest_sha256': helper.sha(BASE / 'manifest.json'), 'parent_checks': checks,
        'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in sources},
        'model_snapshot_sha256': helper.sha(OUT / 'model_service_snapshot.json'), 'old_global_status_sha256': helper.sha(c.STATUS),
        'api_attempt_cap': 29, 'previous_completed': 3, 'available_memory_floor_bytes': 1024**3,
        'old_3_plus_new_29_is_two_execution_configurations': True, 'new_fits': 0, 'ppo_updates': 0})
    save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0, 'logical_requests_completed': 0,
                             'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    save(OUT / 'campaign_status.json', {'state': 'ready', 'pid': None, 'completed_logical_requests': 0})


def independently_score(req, result):
    data = json.loads(req['payload']['messages'][0]['content'].split('Data: ')[1])
    p = data['probability_report_correct']
    location = data['reported_location']
    v = p if data['report_role'] == 'hazard' else 1-p
    costs = {location: v, ('south' if location == 'north' else 'north'): 1-v}
    expected = (min if data['objective'] == 'minimize' else max)(costs, key=costs.get)
    assert expected == req['private_reference']['reference_exit']
    raw = c.read(ROOT / result['raw_path'])
    assert raw['payload'] == req['payload'] and raw['response']['model'] == MODEL
    score = classify(raw['response'], expected)
    assert all(result['outcome'][k] == v for k, v in score.items())
    return score


def finish(m, status, error):
    helper.frozen(m)
    rows = []
    old = c.read(OLD / 'manifest.json')
    for req in old['request_order'][:3]:
        result = c.read(OLD / 'results' / (req['request_id'] + '.json'))
        raw = c.read(ROOT / result['raw_path'])
        score = classify(raw['response'], req['private_reference']['reference_exit'])
        assert all(result['outcome'][k] == v for k, v in score.items())
        rows.append({'request_id': req['request_id'], 'condition': req['condition_id'], 'phase': 'original_definition',
                     'usable_correct': score['usable_correct'], 'latency_seconds': result['latency_seconds']})
    for req in m['request_order']:
        p = OUT / 'results' / (req['request_id'] + '.json')
        if not p.exists():
            continue
        result = c.read(p)
        score = independently_score(req, result)
        rows.append({'request_id': req['request_id'], 'condition': req['condition_id'], 'phase': 'text_only_definition',
                     'usable_correct': score['usable_correct'], 'latency_seconds': result['latency_seconds']})
    assert len({x['request_id'] for x in rows}) == len(rows)
    budget = c.read(OUT / 'budget.json')
    assert budget['api_attempts_started'] == len(list((OUT / 'attempts').glob('*.json'))) <= 29
    assert budget['logical_requests_completed'] == len(rows)-3
    metrics = c.read(OUT / 'resource_samples.json')
    summary = {'covered_total_of_32': len(rows), 'original_phase_answers': 3,
        'text_phase_answers': len(rows)-3, 'correct_in_two_phase_coverage': sum(x['usable_correct'] for x in rows),
        'all_32_covered': len(rows) == 32, 'pure_text_all_32_tested': False,
        'mixed_phase_results_not_a_single_configuration_benchmark': True, 'rows': rows,
        'stop_reason': error, 'new_inference_attempts': budget['api_attempts_started'], 'new_fits': 0, 'ppo_updates': 0}
    save(OUT / 'summary.json', summary)
    c.csv_write(OUT / 'two_phase_coverage.csv', rows)
    loaded = c.read(OUT / 'loaded_model_snapshot.json') if (OUT / 'loaded_model_snapshot.json').exists() else {'models': []}
    original_loaded = c.read(OLD / 'loaded_model_snapshot.json')['models'][0]
    actual = loaded['models'][0] if loaded['models'] else None
    lines = ['# Qwen3.5 27B Q4_K_M纯文本续测报告', '', '日期：2026-10-06（北京时间）。', '',
        f"状态：{status['state']}。本次交付{len(rows)-3}/29条，连同原批次3条，共覆盖{len(rows)}/32条，其中符合数学目标{summary['correct_in_two_phase_coverage']}条。", '',
        '这是两阶段覆盖结果：前3条使用原模型定义，后续使用纯文本定义；不能冒充同一纯文本配置下32条复测。', '',
        '纯文本模型名：qwen35-27b-text-q4:latest。无projector，仍使用同一份16,547,399,008字节的27B Q4_K_M文本权重。模板、参数映射、renderer、parser和thinking设置与原模型相同。', '',
        '本次从第4条开始，无重复前三条，无自动推理重试。消息、schema、temperature0、seed2026100601、num_predict1024、num_ctx4096、think=false保持原计划。', '']
    if actual:
        lines += [f"Ollama报告本次加载大小{actual['size']/1e9:.2f}GB，原定义为{original_loaded['size']/1e9:.2f}GB；本次size_vram={actual['size_vram']}。", '',
            '加载大小是服务报告值。原纯文字请求未必加载视觉编码器，后台占用和缓存也不同，不把运行内存差异直接归因于去掉视觉组件。', '']
    lines += [f"采样最低可用内存{metrics['minimum_available_memory_bytes']/1e6:.1f}MB，系统分页文件使用量最高{metrics['maximum_system_pagefile_used_bytes']/1e9:.2f}GB。", '',
        '每次请求间保留1GiB可用内存门槛；一秒采样不保证捕获精确峰值。分页使用量为整机数值，不能全部归因于模型。', '']
    if error:
        lines += ['停止原因：', '', '```text', error, '```', '']
    lines += ['| 条件 | 两阶段交付数 | 符合目标数 |', '| --- | --- | --- |']
    for cond in m['conditions']:
        values = [r for r in rows if r['condition'] == cond['id']]
        lines.append(f"| {cond['id']} | {len(values)} | {sum(r['usable_correct'] for r in values)} |")
    lines += ['', '8个固定案例来自4个镜像家庭；数学标签比较不是实际疏散控制或伤亡结果，不能据此扩大PPO训练。原报告、模型、默认研究选择与全局状态保持不变。', '',
        f'[固定方案]({PROTOCOL.as_posix()})；[逐题记录]({(OUT / "two_phase_coverage.csv").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json'] + [REPORT]
    files += [ROOT / p for p in m['frozen_sha256']]
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'trial_state': status['state'],
        'files': {p.relative_to(ROOT).as_posix(): helper.sha(p) for p in files},
        'record_audit_passed': True, 'all_32_covered': summary['all_32_covered'], 'pure_text_all_32_tested': False})
    ledger_check(OUT / 'completion_manifest.json')
    print(json.dumps({'state': status['state'], 'covered': len(rows), 'correct': summary['correct_in_two_phase_coverage'],
                      'report': str(REPORT)}, ensure_ascii=False), flush=True)


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
            assert psutil.virtual_memory().available >= m['available_memory_floor_bytes'], 'Available memory below 1GiB; stop remaining requests.'
            result = helper.fetch(req)
            independently_score(req, result)
            if i == 0:
                save(OUT / 'loaded_model_snapshot.json', helper.api('/api/ps'))
            status.update(completed_logical_requests=i+1, last_request=req['request_id'], at_utc=c.stamp())
            save(OUT / 'campaign_status.json', status)
            print(json.dumps({'new_completed': i+1, 'new_total': 29, 'covered_with_previous': i+4,
                'category': result['outcome']['category'], 'latency_seconds': result['latency_seconds']}), flush=True)
        status['state'] = 'complete'
    except Exception as exc:
        error = str(exc)
        status['state'] = 'stopped_after_failure'
        save(OUT / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        print('Continuation stopped: ' + error, flush=True)
    finally:
        resource.stop()
        before = helper.api('/api/ps')
        response = None
        if any(x['name'] == MODEL for x in before['models']):
            response = helper.api('/api/generate', {'model': MODEL, 'keep_alive': 0})
            assert not response.get('eval_count') and not response.get('response')
        after = helper.api('/api/ps')
        assert not any(x['name'] == MODEL for x in after['models'])
        save(OUT / 'unload.json', {'ps_before': before, 'response': response, 'ps_after': after, 'generated_tokens': 0})
        status.update(pid=None, finished_at_utc=c.stamp())
        save(OUT / 'campaign_status.json', status)
        finish(m, status, error)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
