"""Try the pinned Qwen3.5 27B model, then replay at most 32 sealed requests."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time
import traceback

import psutil
import requests
import auto_evacuation_campaign as c
import llm_goal_role_analysis as analysis
from llm_14b_runtime import Monitor
from llm_output_mechanism_core import grade
from run_llm_output_mechanism import save, hash_value
from verify_qwen35_27b_trial import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/llm_goal_role_20261006'
DOWNLOAD = ROOT / 'outputs/qwen35_27b_download_resume_20261006'
OUT = ROOT / 'outputs/qwen35_27b_trial_20261006'
PROTOCOL = ROOT / 'reports/Qwen3.5_27B本机试运行固定方案_2026-10-06.md'
REPORT = ROOT / 'reports/Qwen3.5_27B本机试运行报告_2026-10-06.md'
MODEL = 'qwen3.5:27b'
URL = 'http://127.0.0.1:11434'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def api(route, payload=None, timeout=30):
    r = requests.get(URL + route, timeout=timeout) if payload is None else requests.post(URL + route, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


class Resources:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop_event.is_set():
            try:
                item = Monitor.sample()
                item['system_pagefile_used_bytes'] = psutil.swap_memory().used
                self.samples.append(item)
            except (OSError, psutil.Error):
                pass
            self.stop_event.wait(1)

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=3)
        self.samples.append({**Monitor.sample(), 'system_pagefile_used_bytes': psutil.swap_memory().used})
        save(OUT / 'resource_samples.json', {'interval_seconds': 1, 'samples': self.samples,
             'minimum_available_memory_bytes': min(x['available_memory_bytes'] for x in self.samples),
             'maximum_system_pagefile_used_bytes': max(x['system_pagefile_used_bytes'] for x in self.samples),
             'maximum_ollama_rss_sum_bytes': max(sum(p['rss_bytes'] for p in x['ollama_processes']) for x in self.samples),
             'rss_sum_may_double_count_shared_pages': True, 'sampling_is_not_exact_peak': True})


def prepare():
    assert not OUT.exists(), 'Preserve existing trial.'
    assert c.read(DOWNLOAD / 'status.json')['state'] == 'complete'
    identity = c.read(DOWNLOAD / 'model_identity.json')
    tags = api('/api/tags')
    model = next(x for x in tags['models'] if x['name'] == MODEL)
    assert model['digest'] == identity['model']['digest']
    show = api('/api/show', {'model': MODEL})
    assert show['details']['quantization_level'] == 'Q4_K_M'
    assert 26_000_000_000 <= show['model_info']['general.parameter_count'] <= 28_000_000_000
    assert not api('/api/ps')['models'], 'Do not unload someone else\'s active model.'
    from finalize_auto_evacuation_campaign_review import ledger_check
    parent_paths = [BASE / 'completion_manifest.json', ROOT / 'outputs/llm_goal_role_8b_20261006/completion_manifest.json',
                    ROOT / 'outputs/llm_goal_role_14b_20261006/completion_manifest.json', DOWNLOAD / 'completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p) for p in parent_paths}
    base = c.read(BASE / 'manifest.json')
    order = copy.deepcopy(base['request_order'])
    for req in order:
        req['payload']['model'] = MODEL
        req['payload_sha256'] = hash_value(req['payload'])
    OUT.mkdir()
    snapshot = {'version': api('/api/version'), 'tags': tags, 'show': show}
    assert snapshot['version'] == c.read(BASE / 'model_service_snapshot.json')['version']
    save(OUT / 'model_service_snapshot.json', snapshot)
    sources = [Path(__file__), ROOT / 'scripts/verify_qwen35_27b_trial.py', ROOT / 'scripts/llm_14b_runtime.py',
               ROOT / 'scripts/llm_goal_role_analysis.py', ROOT / 'scripts/llm_output_mechanism_core.py',
               ROOT / 'scripts/verify_llm_output_mechanism.py', ROOT / 'scripts/llm_goal_role_core.py',
               ROOT / 'scripts/run_llm_output_mechanism.py', ROOT / 'scripts/auto_evacuation_campaign.py', PROTOCOL,
               ROOT / 'outputs/local_llm_selection.json']
    save(OUT / 'manifest.json', {'created_at_utc': c.stamp(), 'model_name': MODEL, 'model_digest': model['digest'],
         'cases': base['cases'], 'conditions': base['conditions'], 'request_order': order,
         'base_manifest': (BASE / 'manifest.json').relative_to(ROOT).as_posix(), 'base_manifest_sha256': sha(BASE / 'manifest.json'),
         'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in parent_paths}, 'parent_checks': parent_checks,
         'frozen_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
         'model_snapshot_sha256': sha(OUT / 'model_service_snapshot.json'), 'old_global_status_sha256': sha(c.STATUS),
         'logical_request_cap': 32, 'api_attempt_cap': 32, 'automatic_retries': 0,
         'same_prompt_schema_order_options_as_baseline': True, 'only_api_model_field_changed': True,
         'comparison_is_not_parameter_scale_only': True, 'between_request_available_memory_floor_bytes': 1024**3,
         'first_request_is_pilot_and_part_of_32': True, 'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0, 'logical_requests_completed': 0,
                             'simulations': 0, 'fits': 0, 'ppo_updates': 0})
    save(OUT / 'campaign_status.json', {'state': 'ready', 'pid': None, 'completed_logical_requests': 0})
    print('27B identity checked; 32 requests frozen; no inference yet.', flush=True)


def frozen(m):
    for name, digest in {**m['frozen_sha256'], **m['parent_completion_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    assert sha(c.STATUS) == m['old_global_status_sha256']
    assert sha(BASE / 'manifest.json') == m['base_manifest_sha256']
    assert sha(OUT / 'model_service_snapshot.json') == m['model_snapshot_sha256']


def fetch(req):
    rid = req['request_id']
    assert not (OUT / 'attempts' / (rid + '.json')).exists(), 'No automatic repeated inference.'
    save(OUT / 'attempts' / (rid + '.json'), {'at_utc': c.stamp(), 'payload': req['payload'], 'payload_sha256': req['payload_sha256']})
    b = c.read(OUT / 'budget.json')
    b['api_attempts_started'] += 1
    save(OUT / 'budget.json', b)
    start = time.monotonic()
    r = requests.post(URL + '/api/chat', json=req['payload'], timeout=(10, 240))
    try:
        response = r.json()
    except ValueError:
        response = {'error': r.text}
    raw = {'at_utc': c.stamp(), 'request_id': rid, 'payload': req['payload'], 'payload_sha256': req['payload_sha256'],
           'response': response, 'http_status': r.status_code, 'latency_seconds': time.monotonic()-start}
    rawpath = OUT / 'raw_responses' / (rid + '.json')
    save(rawpath, raw)
    if r.status_code != 200 or response.get('error') or response.get('done') is not True:
        raise RuntimeError(json.dumps({'http_status': r.status_code, 'response': response}, ensure_ascii=False))
    result = {'request_id': rid, 'case_id': req['case_id'], 'condition_id': req['condition_id'],
              'payload_sha256': req['payload_sha256'], 'attempt': 1,
              'raw_path': rawpath.relative_to(ROOT).as_posix(), 'latency_seconds': raw['latency_seconds'],
              'outcome': grade(response, req['private_reference']['reference_exit'])}
    save(OUT / 'results' / (rid + '.json'), result)
    b['successful_http_responses'] += 1
    b['logical_requests_completed'] += 1
    save(OUT / 'budget.json', b)
    return result


def write_report(m, status, failure, summary):
    budget = c.read(OUT / 'budget.json')
    resource = c.read(OUT / 'resource_samples.json')
    snapshot = c.read(OUT / 'model_service_snapshot.json')
    old8 = c.read(ROOT / 'outputs/llm_goal_role_8b_20261006/summary.json')
    old14 = c.read(ROOT / 'outputs/llm_goal_role_14b_20261006/summary.json')
    lines = ['# Qwen3.5 27B Q4_K_M本机试运行报告', '', '日期：2026-10-06（北京时间）。', '',
             f"状态：{status['state']}。本地聊天请求开始{budget['api_attempts_started']}次，成功交付{budget['logical_requests_completed']}/32。", '',
             f"模型：{MODEL}；实际参数量{snapshot['show']['model_info']['general.parameter_count']:,}；量化Q4_K_M；Ollama {snapshot['version']['version']}。", '',
             f"清单digest：`{m['model_digest']}`。使用官方GGUF包，文本权重16,547,399,008字节，视觉组件931,145,984字节；仅文本测试。", '',
             '本机为Ryzen 7 8845HS、24GB内存、Radeon 780M核显。运行设备、加载大小和上下文以loaded_model_snapshot.json为准。', '',
             '题目、消息、JSON schema、temperature0、seed2026100601、1024输出预算、4096上下文与旧轮次保持一致；只更改请求model。Qwen3.5的架构、训练、模板和内置参数可能不同，不能归因于纯参数量效果。最大化条件只是数学诊断，不是疏散建议。', '',
             f"采样最低可用内存{resource['minimum_available_memory_bytes']/1e9:.2f}GB；最大系统分页使用量{resource['maximum_system_pagefile_used_bytes']/1e9:.2f}GB；Ollama进程及子进程RSS之和最大{resource['maximum_ollama_rss_sum_bytes']/1e9:.2f}GB。RSS可能重复计算共享页，采样不保证精确峰值；系统分页使用量不等于全部由模型引起。", '']
    if failure:
        lines += ['本次停止原因：', '', '```text', failure, '```', '',
                  '该结果只说明本次服务版本、模型文件、运行设置和当时资源条件下的尝试结果。未交付的题目不记为语义答错，也不输出伪造的全套正确率。']
    if summary:
        lines += ['| 条件 | Qwen3 8B | Qwen3 14B | Qwen3.5 27B |', '| --- | --- | --- | --- |']
        for key, g in summary['groups'].items():
            lines.append(f"| {key} | {old8['groups'][key]['usable_correct']}/8 | {old14['groups'][key]['usable_correct']}/8 | {g['usable_correct']}/8 |")
        total = sum(g['usable_correct'] for g in summary['groups'].values())
        elapsed = sum(g['total_latency_seconds'] for g in summary['groups'].values())
        lines += ['', f'27B合计可用且符合目标{total}/32，累计请求等待{elapsed:.2f}秒（含首次加载）。', '',
                  '| 配对检查 | 方向关系符合/8 | 两答案均正确/8 |', '| --- | --- | --- |']
        for pair in summary['paired_diagnostics']:
            lines.append(f"| {pair['comparison']} | {pair['expected_relation_observed']} | {pair['both_answers_correct']} |")
        lines += ['', '这32条来自8个固定案例、4个镜像家庭，不是32个独立场景。通过基础题不能证明真实疏散性能或PPO收益；自报confidence没有明确定义为校准后的答案正确率。']
    lines += ['', '旧实验、默认模型选择、全局研究状态和保留疏散场景保持原字节。无付费API、模拟、拟合或PPO更新。', '',
              f'[固定方案]({PROTOCOL.as_posix()})；[原始记录]({OUT.as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def run():
    m = c.read(OUT / 'manifest.json')
    frozen(m)
    status = c.read(OUT / 'campaign_status.json')
    assert status['state'] == 'ready', 'Preserve prior execution; no automatic retry.'
    status.update(state='running', pid=os.getpid(), started_at_utc=c.stamp())
    save(OUT / 'campaign_status.json', status)
    resource = Resources()
    resource.start()
    failure = None
    summary = None
    try:
        for i, req in enumerate(m['request_order']):
            assert psutil.virtual_memory().available >= m['between_request_available_memory_floor_bytes'], 'Available memory below 1GiB; stop remaining requests.'
            result = fetch(req)
            if i == 0:
                save(OUT / 'loaded_model_snapshot.json', api('/api/ps'))
            status.update(completed_logical_requests=i+1, last_request=req['request_id'], updated_at_utc=c.stamp())
            save(OUT / 'campaign_status.json', status)
            print(json.dumps({'completed': i+1, 'total': 32, 'category': result['outcome']['category'],
                              'latency_seconds': result['latency_seconds']}), flush=True)
        frozen(m)
        analysis.OUT = OUT
        summary = analysis.summarize(m)
        save(OUT / 'verification.json', verify(OUT))
        status['state'] = 'complete'
    except Exception as exc:
        failure = str(exc)
        status['state'] = 'stopped_after_failure'
        save(OUT / 'failure.json', {'at_utc': c.stamp(), 'error': repr(exc), 'traceback': traceback.format_exc(),
                                  'budget': c.read(OUT / 'budget.json')})
        print('Trial stopped: ' + failure, flush=True)
    finally:
        resource.stop()
        try:
            before = api('/api/ps')
            response = None
            if any(x['name'] == MODEL for x in before['models']):
                response = api('/api/generate', {'model': MODEL, 'keep_alive': 0})
                assert response.get('done') and not response.get('response') and not response.get('eval_count')
            after = api('/api/ps')
            assert not any(x['name'] == MODEL for x in after['models'])
            save(OUT / 'unload.json', {'ps_before': before, 'unload_response': response, 'ps_after': after,
                 'generated_tokens': 0, 'available_memory_bytes_after': psutil.virtual_memory().available})
        except Exception as exc:
            status['unload_error'] = repr(exc)
            save(OUT / 'unload_failure.json', {'error': repr(exc)})
        frozen(m)
        status.update(pid=None, finished_at_utc=c.stamp())
        save(OUT / 'campaign_status.json', status)
        write_report(m, status, failure, summary)
        files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']
        files += [REPORT] + [ROOT / p for p in m['frozen_sha256']]
        save(OUT / 'completion_manifest.json', {'trial_state': status['state'], 'at_utc': c.stamp(),
             'record_integrity_checked': True, 'full_functional_test_completed': status['state'] == 'complete',
             'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}, 'budget': c.read(OUT / 'budget.json')})
        print(json.dumps({'state': status['state'], 'report': str(REPORT), 'budget': c.read(OUT / 'budget.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
