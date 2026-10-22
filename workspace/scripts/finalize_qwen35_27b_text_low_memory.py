"""Read-only audit of the four 27B phases; write a separate closure only."""
from pathlib import Path
import copy
import json
import statistics
import sys

import requests
import auto_evacuation_campaign as c
from run_qwen35_27b_trial import sha
from run_llm_output_mechanism import save, hash_value
from verify_llm_output_mechanism import classify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen35_27b_text_final_audit_20261006'
REPORT = ROOT / 'reports/Qwen3.5_27B纯文本压缩与完整测试说明_2026-10-06.md'
MODEL = 'qwen35-27b-text-q4:latest'
BASE = ROOT / 'outputs/llm_goal_role_20261006'
STAGES = [
    ('qwen35_27b_trial_20261006', '原模型定义', 0, 3, 'qwen3.5:27b', None, 4096),
    ('qwen35_27b_text_continuation_20261006', '纯文本驻留', 3, 13, MODEL, None, 4096),
    ('qwen35_27b_text_serial_unload_20261006', '纯文本逐题卸载', 16, 1, MODEL, 64, 4096),
    ('qwen35_27b_text_low_memory_20261006', '纯文本小缓存逐题卸载', 17, None, MODEL, 32, 2048),
]


def ledger(path):
    data = c.read(path)
    for name, digest in data['files'].items():
        assert sha(ROOT / name) == digest, name
    return len(data['files'])


def mathematical_reference(req):
    data = json.loads(req['payload']['messages'][0]['content'].split('Data: ')[1])
    p = data['probability_report_correct']
    v = p if data['report_role'] == 'hazard' else 1 - p
    location = data['reported_location']
    costs = {location: v, ('south' if location == 'north' else 'north'): 1 - v}
    return (min if data['objective'] == 'minimize' else max)(costs, key=costs.get)


def run():
    assert not OUT.exists() and not REPORT.exists(), 'Preserve any existing closure.'
    base = c.read(BASE / 'manifest.json')
    checks, rows, stages = {}, [], []
    for name, label, offset, expected_n, model, batch, ctx in STAGES:
        folder = ROOT / 'outputs' / name
        status = c.read(folder / 'campaign_status.json')
        assert status['state'] in ('complete', 'stopped_after_failure')
        checks[name] = ledger(folder / 'completion_manifest.json')
        m, b = c.read(folder / 'manifest.json'), c.read(folder / 'budget.json')
        n = b['logical_requests_completed']
        assert expected_n is None or n == expected_n
        assert status['completed_logical_requests'] == n
        assert b['api_attempts_started'] == len(list((folder / 'attempts').glob('*.json')))
        assert b['successful_http_responses'] == n == len(list((folder / 'results').glob('*.json')))
        assert n <= b['api_attempts_started'] <= n + 1
        assert len(list((folder / 'raw_responses').glob('*.json'))) <= b['api_attempts_started']
        assert all(b[k] == 0 for k in ('simulations', 'fits', 'ppo_updates'))
        assert sha(BASE / 'manifest.json') == m['base_manifest_sha256']
        assert sha(c.STATUS) == m['old_global_status_sha256']
        for p, digest in {**m['frozen_sha256'], **m['parent_completion_sha256']}.items():
            assert sha(ROOT / p) == digest, p
        assert sha(folder / 'model_service_snapshot.json') == m['model_snapshot_sha256']
        for req, original in zip(m['request_order'], base['request_order'][offset:]):
            expected = copy.deepcopy(original)
            expected['payload']['model'] = model
            if batch is not None:
                expected['payload']['options']['num_batch'] = batch
            expected['payload']['options']['num_ctx'] = ctx
            expected['payload_sha256'] = hash_value(expected['payload'])
            assert req == expected, req['request_id']
        for req in m['request_order'][:n]:
            rid = req['request_id']
            result = c.read(folder / 'results' / (rid + '.json'))
            raw = c.read(folder / 'raw_responses' / (rid + '.json'))
            attempt = c.read(folder / 'attempts' / (rid + '.json'))
            assert attempt['payload'] == raw['payload'] == req['payload']
            assert attempt['payload_sha256'] == raw['payload_sha256'] == req['payload_sha256']
            assert raw['http_status'] == 200 and raw['response']['done'] is True
            assert raw['response']['model'] == model and raw['request_id'] == rid
            assert result['request_id'] == rid and result['attempt'] == 1
            assert ROOT / result['raw_path'] == folder / 'raw_responses' / (rid + '.json')
            ref = mathematical_reference(req)
            assert ref == req['private_reference']['reference_exit']
            score = classify(raw['response'], ref)
            assert all(result['outcome'][k] == v for k, v in score.items())
            rows.append({'request_id': rid, 'condition': req['condition_id'], 'phase': label,
                'reference_exit': ref, 'selected_exit': score['semantic_direction'],
                'correct': score['usable_correct'], 'format_valid': score['strict_format_valid'],
                'category': score['category'], 'latency_seconds': raw['latency_seconds'],
                'prompt_tokens': raw['response']['prompt_eval_count'],
                'output_tokens': raw['response']['eval_count']})
        unload_paths = list((folder / 'unloads').glob('*.json')) if (folder / 'unloads').exists() else [folder / 'unload.json']
        assert len(unload_paths) == (n + 1 if batch is not None else 1)
        for path in unload_paths:
            u = c.read(path)
            assert not u['ps_after']['models'] and u['generated_tokens'] == 0
            response = u.get('response', u.get('unload_response'))
            assert response is None or (response['done'] and not response.get('eval_count') and not response.get('response'))
        resources = c.read(folder / 'resource_samples.json')
        assert resources['minimum_available_memory_bytes'] == min(s['available_memory_bytes'] for s in resources['samples'])
        assert resources['maximum_system_pagefile_used_bytes'] == max(s['system_pagefile_used_bytes'] for s in resources['samples'])
        loaded = c.read(folder / 'loaded_model_snapshot.json')['models'][0]
        assert loaded['name'] == model and loaded['size_vram'] == 0 and loaded['context_length'] == ctx
        group = rows[-n:] if n else []
        stages.append({'phase': label, 'completed': n, 'correct': sum(r['correct'] for r in group),
            'state': status['state'], 'num_ctx': ctx, 'num_batch': batch, 'loaded_size_bytes': loaded['size'],
            'median_latency_seconds': statistics.median(r['latency_seconds'] for r in group) if group else None,
            'minimum_sampled_available_memory_bytes': resources['minimum_available_memory_bytes'],
            'maximum_system_pagefile_used_bytes': resources['maximum_system_pagefile_used_bytes']})
    ids = [r['request_id'] for r in rows]
    canonical = [r['request_id'] for r in base['request_order']]
    assert len(ids) == len(set(ids)) and ids == canonical[:len(ids)]
    assert len(ids) <= 32
    creation = ROOT / 'outputs/qwen35_27b_text_creation_20261006'
    checks['text_creation'] = ledger(creation / 'completion_manifest.json')
    for name in ('llm_goal_role_8b_20261006', 'llm_goal_role_14b_20261006'):
        checks[name] = ledger(ROOT / 'outputs' / name / 'completion_manifest.json')
    identity = c.read(creation / 'text_model_identity.json')
    source = c.read(creation / 'source_snapshot.json')
    store = Path('D:/LLMResearch/OllamaModels14B/manifests/registry.ollama.ai/library')
    text_path = store / 'qwen35-27b-text-q4/latest'
    assert sha(text_path) == identity['manifest_digest']
    assert sha(store / 'qwen3.5/27b') == source['manifest_sha256']
    live_ps = requests.get('http://127.0.0.1:11434/api/ps', timeout=10)
    live_ps.raise_for_status()
    assert not live_ps.json()['models']
    live_show = requests.post('http://127.0.0.1:11434/api/show', json={'model': MODEL}, timeout=10)
    live_show.raise_for_status()
    show = live_show.json()
    assert 'vision' not in show['capabilities'] and not show.get('projector_info')
    assert show['model_info'] == identity['show']['model_info']
    assert show['details']['quantization_level'] == 'Q4_K_M'
    assert identity['text_weights_unchanged'] and identity['projector_absent']
    groups = {k: {'delivered': sum(r['condition'] == k for r in rows),
        'correct': sum(r['condition'] == k and r['correct'] for r in rows)}
        for k in ('minimize_hazard', 'maximize_hazard', 'minimize_clear', 'maximize_clear')}
    audit = {'record_audit_passed': True, 'all_32_covered': ids == canonical,
        'single_configuration_benchmark': False, 'covered': len(rows),
        'correct': sum(r['correct'] for r in rows), 'format_valid': sum(r['format_valid'] for r in rows),
        'ledgers': checks, 'groups': groups, 'stages': stages, 'rows': rows,
        'prompt_tokens_min': min(r['prompt_tokens'] for r in rows),
        'prompt_tokens_max': max(r['prompt_tokens'] for r in rows),
        'model_unloaded': True, 'text_weight_bytes': identity['text_weight_bytes'],
        'new_inference': 0, 'simulations': 0, 'fits': 0, 'ppo_updates': 0,
        'global_status_and_default_selection_unchanged': True}
    OUT.mkdir()
    save(OUT / 'audit.json', audit)
    save(OUT / 'live_service_checks.json', {'ps': live_ps.json(), 'show': show})
    c.csv_write(OUT / 'verified_answers.csv', rows)
    lines = ['# Qwen3.5 27B纯文本压缩与测试说明', '', '日期：2026-10-06。', '',
        f"已创建纯文本模型 `{MODEL}`。四阶段共完成{audit['covered']}/32条固定题，其中{audit['correct']}条符合数学目标，{audit['format_valid']}条格式有效。模型已卸载。", '',
        '## 实际减少了什么', '',
        '纯文本定义不包含视觉projector，服务能力列表不含vision。仍使用同一份27B Q4_K_M文本权重，大小16.55GB（十进制）；没有进一步降低位宽、裁剪语言模型或重新下载文本权重。排除的视觉组件约0.93GB。原模型和共享文件保留，因此这不是磁盘清理操作。', '',
        '后续将上下文从4096降至2048、计算批次降至32，每题结束后卸载。减少的是运行缓存和批次占用，语言模型文本权重精度不变；可容纳的上下文变短。', '',
        '## 分阶段实测', '',
        '| 阶段 | 完成 | 符合目标 | 上下文 | 批次 | 服务加载GB | 每题中位秒 | 最低采样可用MB |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for stage in stages:
        lines.append(f"| {stage['phase']} | {stage['completed']} | {stage['correct']} | {stage['num_ctx']} | {stage['num_batch'] or '服务默认'} | {stage['loaded_size_bytes']/1e9:.2f} | {stage['median_latency_seconds']:.2f} | {stage['minimum_sampled_available_memory_bytes']/1e6:.1f} |")
    lines += ['', '以上加载大小为Ollama报告值，并非整机全部内存占用。实测均为CPU推理。可用内存最低值来自离散采样，不保证捕获瞬时最低值；分页使用量属于整台电脑。后台状态与缓存不同，不能把阶段间差异全归因于视觉移除。', '',
        '逐题卸载会重复加载模型，代价是每题耗时增加。没有重试已完成题，也没有因答错而重跑。第四阶段保留每题开始前1GiB可用内存门槛；若题内采样低于256MiB，在回答完成并卸载后停止后续请求。', '',
        '## 固定题结果与解释', '', '| 条件 | 已完成 | 符合目标 |', '| --- | --- | --- |']
    for key, values in groups.items():
        lines.append(f"| {key} | {values['delivered']} | {values['correct']} |")
    lines += ['', '这32条由8个案例、4个镜像家庭构成，测试的是角色与目标的数学理解。最大化危险等条件只用于诊断。它们不是32个独立火灾场景，不能证明实际疏散收益，也不能凭此恢复PPO训练。', '',
        '第一阶段3条使用原定义；第二阶段13条使用纯文本驻留；第三阶段1条使用batch64逐题卸载；第四阶段使用ctx2048、batch32逐题卸载。结果是分阶段覆盖，不能称为同一运行配置下的32题基准，更不能归因于纯参数量提升。', '',
        '## 输入token数更正与记录完整性', '',
        f"早先续测方案和进度说明写过输入约五六百token。根据保存的响应，实际范围为{audit['prompt_tokens_min']}–{audit['prompt_tokens_max']}个输入token。原方案已封存，因此在此单独更正。2048仍高于这些短题的输入与1024输出预算之和；题目和schema均未删减。", '',
        '逐题原始响应、独立数学参考、评分、请求预算、卸载记录和四阶段完整性台账均已复核。默认研究模型、全局研究状态和旧8B/14B记录的哈希未变；本次复核没有推理、拟合、模拟或PPO更新。', '',
        '启动模型：`ollama run qwen35-27b-text-q4:latest`。该命令只指定模型；本轮小缓存设置由实验请求显式传入，普通交互运行不自动继承这些设置。', '',
        f'[独立核验记录]({(OUT / "audit.json").as_posix()})；[逐题结果]({(OUT / "verified_answers.csv").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    files = [p for p in OUT.iterdir() if p.is_file()] + [REPORT, Path(__file__)]
    files += [ROOT / 'outputs' / name / 'completion_manifest.json' for name, *_ in STAGES]
    save(OUT / 'completion_manifest.json', {'record_audit_passed': True,
        'all_32_covered': audit['all_32_covered'], 'new_inference': 0,
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}})
    ledger(OUT / 'completion_manifest.json')
    print(json.dumps({'covered': audit['covered'], 'correct': audit['correct'],
        'groups': groups, 'stages': stages, 'model_unloaded': True,
        'report': str(REPORT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
