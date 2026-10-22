"""Try three real short requests with bounded residency, or record a memory stop."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import sys
import threading
import time

import psutil
import requests
import auto_evacuation_campaign as c
from run_qwen35_27b_trial import sha, api
from run_llm_output_mechanism import save, hash_value
from verify_llm_output_mechanism import classify

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen35-27b-text-q4:latest'
PROTOCOL = ROOT / 'reports/27B短批次驻留提速方案_2026-10-06.md'
BASE = ROOT / 'outputs/llm_goal_role_20261006/manifest.json'
REQUEST_IDS = ('GR00_north_minimize_hazard', 'GR00_south_minimize_hazard', 'GR00_north_minimize_clear')
OPTIONS = {'temperature': 0, 'seed': 2026100601, 'num_ctx': 1024,
    'num_predict': 128, 'num_batch': 32, 'num_thread': 8, 'num_gpu': 0}


def run():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'outputs' / ('qwen35_speed_probe_' + stamp)
    out.mkdir()
    protected = {str(p.relative_to(ROOT)): sha(p) for p in (
        c.STATUS, ROOT / 'outputs/local_llm_selection.json',
        ROOT / 'outputs/qwen35_label_order_probe_20261006/preparation_manifest.json',
        ROOT / 'outputs/qwen35_27b_text_final_audit_20261006/completion_manifest.json')}
    base = c.read(BASE)
    chosen = {r['request_id']: r for r in base['request_order']}
    order = []
    for rid in REQUEST_IDS:
        req = copy.deepcopy(chosen[rid])
        req['payload']['model'] = MODEL
        req['payload']['options'] = OPTIONS.copy()
        req['payload']['keep_alive'] = '60s'
        req['payload_sha256'] = hash_value(req['payload'])
        order.append(req)
    save(out / 'manifest.json', {'purpose': 'small_representative_runtime_probe',
        'model': MODEL, 'request_order': order, 'request_cap': 3,
        'available_memory_before_load_floor_bytes': 17 * 1024**3,
        'available_memory_before_followup_floor_bytes': 1024**3,
        'sampled_low_memory_floor_bytes': 256 * 1024**2,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (BASE, PROTOCOL, Path(__file__))},
        'protected_sha256': protected, 'not_a_new_evacuations_or_accuracy_benchmark': True})
    budget = {'api_attempts_started': 0, 'successful_http_responses': 0,
        'logical_requests_completed': 0, 'simulations': 0, 'fits': 0, 'ppo_updates': 0}
    save(out / 'budget.json', budget)
    status = {'state': 'preflight', 'stop_reason': None}
    samples, rows = [], []
    event = threading.Event()
    loaded_by_this_probe = False
    worker = None

    def sample():
        return {'at_utc': c.stamp(), 'available_memory_bytes': psutil.virtual_memory().available,
            'system_pagefile_used_bytes': psutil.swap_memory().used}

    def collect():
        while not event.is_set():
            samples.append(sample())
            event.wait(0.5)

    try:
        ps = api('/api/ps')
        before = sample()
        save(out / 'preflight.json', {'ps': ps, 'resource_measurement': before,
            'memory_required_before_load_bytes': 17 * 1024**3})
        if ps['models']:
            status.update(state='not_run_model_already_loaded', stop_reason='Existing active model preserved.')
        elif before['available_memory_bytes'] < 17 * 1024**3:
            status.update(state='not_run_insufficient_memory',
                stop_reason='Available memory below the predeclared 17GiB startup floor.')
        else:
            identity = c.read(ROOT / 'outputs/qwen35_27b_text_creation_20261006/text_model_identity.json')
            tag = next(x for x in api('/api/tags')['models'] if x['name'] == MODEL)
            assert tag['digest'] == identity['manifest_digest']
            worker = threading.Thread(target=collect, daemon=True)
            worker.start()
            status['state'] = 'running'
            for index, req in enumerate(order):
                if index and psutil.virtual_memory().available < 1024**3:
                    status.update(state='stopped_memory_floor', stop_reason='Available memory below 1GiB before followup.')
                    break
                start_index = len(samples)
                loaded_by_this_probe = True
                budget['api_attempts_started'] += 1
                save(out / 'budget.json', budget)
                save(out / 'attempts' / (req['request_id'] + '.json'), req['payload'])
                begin = time.monotonic()
                http = requests.post('http://127.0.0.1:11434/api/chat', json=req['payload'], timeout=(10, 240))
                latency = time.monotonic() - begin
                try:
                    response = http.json()
                except ValueError:
                    response = {'error': http.text}
                save(out / 'raw_responses' / (req['request_id'] + '.json'),
                    {'payload': req['payload'], 'response': response, 'http_status': http.status_code,
                        'latency_seconds': latency})
                http.raise_for_status()
                assert response.get('done') is True and not response.get('error')
                assert response['model'] == MODEL
                score = classify(response, req['private_reference']['reference_exit'])
                row = {'request_id': req['request_id'], 'execution': 'first_load' if index == 0 else 'resident_followup',
                    'latency_seconds': latency, 'load_seconds': response.get('load_duration', 0)/1e9,
                    'prompt_seconds': response.get('prompt_eval_duration', 0)/1e9,
                    'generate_seconds': response.get('eval_duration', 0)/1e9,
                    'prompt_tokens': response.get('prompt_eval_count'), 'output_tokens': response.get('eval_count'),
                    'format_valid': score['strict_format_valid'], 'correct': score['usable_correct'],
                    'truncated': score['truncated']}
                row['generate_tps'] = row['output_tokens']/row['generate_seconds'] if row['generate_seconds'] else None
                rows.append(row)
                budget['successful_http_responses'] += 1
                budget['logical_requests_completed'] += 1
                save(out / 'budget.json', budget)
                loaded = api('/api/ps')
                save(out / 'loaded_snapshots' / (req['request_id'] + '.json'), loaded)
                actual = next(x for x in loaded['models'] if x['name'] == MODEL)
                assert actual['size_vram'] == 0 and actual['context_length'] == 1024
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if samples[start_index:] and min(x['available_memory_bytes'] for x in samples[start_index:]) < 256*1024**2:
                    status.update(state='stopped_sampled_low_memory', stop_reason='Sampled memory below 256MiB.')
                    break
            else:
                status['state'] = 'complete'
    except Exception as exc:
        status.update(state='stopped_error', stop_reason=repr(exc))
    finally:
        if loaded_by_this_probe:
            try:
                response = api('/api/generate', {'model': MODEL, 'keep_alive': 0})
                after = api('/api/ps')
                assert not response.get('eval_count') and not response.get('response')
                assert not any(x['name'] == MODEL for x in after['models'])
                save(out / 'unload.json', {'response': response, 'ps_after': after, 'generated_tokens': 0})
            except Exception as exc:
                status['unload_error'] = repr(exc)
        event.set()
        if worker:
            worker.join(timeout=2)
        samples.append(sample())
        save(out / 'resource_samples.json', {'samples': samples,
            'minimum_sampled_available_memory_bytes': min(x['available_memory_bytes'] for x in samples),
            'sampling_is_not_exact_peak': True})
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    save(out / 'campaign_status.json', status)
    save(out / 'summary.json', {'state': status['state'], 'rows': rows, 'attempts': budget['api_attempts_started'],
        'finished': len(rows), 'runtime_settings_not_yet_validated': len(rows) < 3,
        'new_evacuations': 0, 'ppo_updates': 0, 'protected_research_unchanged': True})
    lines = ['# 27B提速检查结果', '', '日期：2026-10-06。', '',
        f"状态：{status['state']}；实际推理请求{budget['api_attempts_started']}次，完成{len(rows)}/3条。", '',
        '保留纯文本27B Q4_K_M权重。新配置为ctx1024、输出上限128、batch32、CPU线程8、keep_alive60s，仅用于预定三条短题。其他应用与服务设置不变。', '',
        f"执行前可用内存{c.read(out / 'preflight.json')['resource_measurement']['available_memory_bytes']/1e9:.2f}GB；本测试加载前门槛17GiB（约18.25GB）。", '',
        f"停止原因：{status['stop_reason'] or '无'}。", '',
        '旧数据已显示：逐题卸载时约50秒，早先驻留时约26秒，但早先内存不足，不能直接采用长期驻留。这次若未执行，不能宣称新设置已得到实测提速。', '',
        '| 请求 | 类型 | 总秒 | 加载秒 | 输入秒 | 输出秒 | 生成token/秒 | 正确 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['request_id']} | {r['execution']} | {r['latency_seconds']:.2f} | {r['load_seconds']:.2f} | {r['prompt_seconds']:.2f} | {r['generate_seconds']:.2f} | {r['generate_tps']:.2f} | {r['correct']} |")
    lines += ['', '当前没有新结果时，减少加载与共用前缀缓存只作为候选优化。CPU生成吞吐能提升多少需要独立测量；降低最大输出上限不代表实际22-token答案自动变快。', '',
        f'[固定方案]({PROTOCOL.as_posix()})。内存充足后可再次运行脚本，每次另建目录；原研究选择、封存结果与标签诊断准备文件保持不变。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    files = [p for p in out.rglob('*') if p.is_file()] + [Path(__file__), PROTOCOL]
    save(out / 'completion_manifest.json', {'record_integrity_passed': True,
        'functional_test_complete': len(rows) == 3 and status['state'] == 'complete',
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}})
    for name, digest in c.read(out / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / name) == digest
    print(json.dumps({'state': status['state'], 'inference_attempts': budget['api_attempts_started'],
        'report': str(out / 'report.md'), 'out': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
