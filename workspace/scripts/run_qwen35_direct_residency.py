"""User-authorized representative 27B trial below the old 17GiB floor.

Preserves old plans. Serial requests only, at most three. Recovery-stage probe after automatic OS memory reclamation. Never starts a service.
"""
from pathlib import Path
import copy
import json
import sys
import threading
import time

import psutil
import requests

from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output, verify_preparation
from run_qwen35_timestamped_live import api, score, MODEL, URL


def run():
    verify_preparation()
    out = unique_output('qwen35_direct_residency')
    old = read(ROOT / 'outputs/llm_goal_role_20261006/manifest.json')['request_order']
    chosen = {r['request_id']: r for r in old}
    ids = ('GR00_north_minimize_hazard', 'GR00_south_minimize_hazard', 'GR00_north_minimize_clear')
    order = []
    for threads in (8,):
        for i, rid in enumerate(ids):
            r = copy.deepcopy(chosen[rid])
            r['request_id'] = f'DIRECT_t{threads}_{rid}'
            r['candidate_order'] = ['north', 'south']
            r['private_reference']['label_to_physical'] = {'north': 0, 'south': 1}
            r['payload']['model'] = MODEL
            r['payload']['options'] = {'temperature': 0, 'seed': 2026100601, 'num_ctx': 1024,
                                      'num_predict': 128, 'num_batch': 32, 'num_thread': threads, 'num_gpu': 0}
            r['payload']['keep_alive'] = '60s'
            r['cpu_threads'] = threads
            r['phase'] = 'cold' if i == 0 else 'resident_shared_prefix'
            order.append(r)
    save(out / 'manifest.json', {'user_instruction': '直接测试',
         'explicit_new_resource_override': 'New recovery-stage startup estimate: observed model size 16773388695 bytes plus 512MiB reserve; warm floor 512MiB. Old plans unchanged.',
         'old_preparations_unchanged': True, 'request_cap': 3, 'seconds_cap': 600,
         'startup_floor_bytes': 16773388695 + 512*1024**2, 'followup_floor_bytes': 512*1024**2, 'sampled_pressure_floor_bytes': 256 * 1024**2,
         'pressure_guard_applied_after_current_request_then_unload': True, 'read_timeout_seconds': 180,
         'http_timeout_does_not_cancel_server_generation': True, 'no_retries': True,
         'no_ppo_training_or_reserved_scenarios': True, 'request_order': order,
         'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                           'outputs/qwen35_direct_speed_20261006T192010814038Z/loaded_1.json': sha(ROOT / 'outputs/qwen35_direct_speed_20261006T192010814038Z/loaded_1.json'), 'outputs/timestamped_pipeline_preparation_20261007/preparation_manifest.json': sha(ROOT / 'outputs/timestamped_pipeline_preparation_20261007/preparation_manifest.json')}})
    rows, samples, attempts = [], [], 0
    state, reason = 'preflight', None
    stop = threading.Event()
    owned = False
    start = time.monotonic()
    worker = None
    def collect():
        while not stop.is_set():
            samples.append({'elapsed_s': time.monotonic()-start, 'available_bytes': psutil.virtual_memory().available,
                            'pagefile_used_bytes': psutil.swap_memory().used})
            stop.wait(.5)
    def unload():
        nonlocal owned
        if owned:
            r = api('/api/generate', {'model': MODEL, 'keep_alive': 0}, timeout=60)
            ps = api('/api/ps')
            save(out / f'unload_{attempts}.json', {'response': r, 'ps_after': ps})
            assert not any(x['name'] == MODEL for x in ps['models'])
            owned = False
    try:
        ps = api('/api/ps')
        tags = api('/api/tags')
        tag = next(x for x in tags['models'] if x['name'] == MODEL)
        identity = read(ROOT / 'outputs/qwen35_27b_text_creation_20261006/text_model_identity.json')
        assert tag['digest'] == identity['manifest_digest']
        before = {'ps': ps, 'model_tag': tag, 'available_bytes': psutil.virtual_memory().available,
                  'pagefile_used_bytes': psutil.swap_memory().used}
        save(out / 'preflight.json', before)
        assert not ps['models'], 'Existing active models preserved.'
        assert before['available_bytes'] >= 16773388695 + 512*1024**2, 'Available memory below observed model size plus 512MiB reserve.'
        worker = threading.Thread(target=collect, daemon=True)
        worker.start()
        for req in order:
            if time.monotonic()-start > 600:
                state, reason = 'stopped_time_cap', '600-second trial cap'
                break
            if psutil.virtual_memory().available < 512*1024**2:
                state, reason = 'stopped_followup_memory', 'Available memory below 512MiB before next request'
                break
            if req['phase'] == 'cold' and owned:
                unload()
            begin, sample_begin = time.monotonic(), len(samples)
            attempts += 1
            owned = True
            save(out / 'budget.json', {'attempts': attempts, 'completed': len(rows), 'request_cap': 3})
            print(json.dumps({'state': 'inference_started', 'attempt': attempts, 'phase': req['phase'],
                              'threads': req['cpu_threads'], 'available_GiB': round(psutil.virtual_memory().available/1024**3,2)}), flush=True)
            http = requests.post(URL + '/api/chat', json=req['payload'], timeout=(5, 180))
            latency = time.monotonic()-begin
            response = http.json()
            save(out / 'responses' / f'{attempts:02d}.json', {'payload': req['payload'], 'response': response,
                                                          'latency_seconds': latency, 'http_status': http.status_code})
            http.raise_for_status()
            assert response.get('model') == MODEL and response.get('done')
            row = {'request_id': req['request_id'], 'threads': req['cpu_threads'], 'phase': req['phase'],
                   'latency_seconds': latency, 'load_seconds': response.get('load_duration', 0)/1e9,
                   'prompt_seconds': response.get('prompt_eval_duration', 0)/1e9,
                   'generate_seconds': response.get('eval_duration', 0)/1e9,
                   'prompt_tokens': response.get('prompt_eval_count'), 'output_tokens': response.get('eval_count'),
                   **score(response, req)}
            row['generate_tps'] = row['output_tokens']/row['generate_seconds'] if row['generate_seconds'] else None
            window = samples[sample_begin:]
            row['min_sampled_available_bytes'] = min(x['available_bytes'] for x in window) if window else None
            rows.append(row)
            save(out / 'rows.json', rows)
            loaded = api('/api/ps')
            save(out / f'loaded_{attempts}.json', loaded)
            match = next(x for x in loaded['models'] if x['name'] == MODEL)
            assert match['size_vram'] == 0 and match['context_length'] == 1024
            print(json.dumps({'state': 'response_completed', **row}, ensure_ascii=False), flush=True)
            if row['min_sampled_available_bytes'] is not None and row['min_sampled_available_bytes'] < 256*1024**2:
                state, reason = 'stopped_sampled_pressure', 'Sampled available memory below 256MiB; current response retained.'
                break
        else:
            state = 'complete'
    except Exception as exc:
        state, reason = 'stopped_error', repr(exc)
        save(out / 'failure.json', {'error': reason, 'attempts': attempts})
    finally:
        try:
            unload()
        except Exception as exc:
            state, reason = 'unload_error', repr(exc)
        stop.set()
        if worker:
            worker.join(timeout=2)
        save(out / 'resource_samples.json', {'samples': samples, 'sampling_is_not_exact_peak': True})
        save(out / 'budget.json', {'attempts': attempts, 'completed': len(rows), 'request_cap': 3})
        save(out / 'summary.json', {'state': state, 'reason': reason, 'attempts': attempts, 'completed': len(rows), 'rows': rows,
             'new_ppo_updates': 0, 'reserved_scenarios_opened': 0,
             'minimum_available_bytes': min((x['available_bytes'] for x in samples), default=None),
             'maximum_pagefile_used_bytes': max((x['pagefile_used_bytes'] for x in samples), default=None),
             'available_after_unload_bytes': psutil.virtual_memory().available})
        verify_preparation()
        lines = ['# 27B直接实测', '', f'状态：{state}；尝试{attempts}次，完成{len(rows)}次。', '',
                 '本轮为首次内存压力停止、卸载后系统回收内存的恢复阶段。启动门槛依据实测模型大小加512MiB，后续驻留至少512MiB；低于256MiB仍停止。原方案封存。使用既有27B纯文本Q4_K_M、CPU、ctx1024、batch32。', '',
                 '|线程|阶段|总秒|加载秒|输入秒|输出秒|生成token/秒|正确|最低可用MiB|',
                 '|---:|---|---:|---:|---:|---:|---:|---|---:|']
        for r in rows:
            lines.append(f"|{r['threads']}|{r['phase']}|{r['latency_seconds']:.2f}|{r['load_seconds']:.2f}|{r['prompt_seconds']:.2f}|{r['generate_seconds']:.2f}|{r['generate_tps']:.2f}|{r['correct']}|{r['min_sampled_available_bytes']/1024**2:.1f}|")
        lines += ['', f'停止原因：{reason or "无"}。', '',
                  '旧逐题卸载总等待中位49.617秒、生成2.890token/秒仅为历史参照，后台占用不同不能作受控因果结论。驻留含共用前缀复用效果。未完成两组时不比较线程优劣。', '']
        (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        save(out / 'completion_manifest.json', {'state': state, 'old_sources_unchanged': True,
             'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}})
        print(json.dumps({'state': state, 'completed': len(rows), 'reason': reason, 'output': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
