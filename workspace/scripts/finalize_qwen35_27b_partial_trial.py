"""Audit the actual three answers and resource stop, preserving sealed trial files."""
from pathlib import Path
import json
import sys

import requests
import auto_evacuation_campaign as c
from run_qwen35_27b_trial import sha
from run_llm_output_mechanism import save, hash_value
from verify_llm_output_mechanism import classify
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'outputs/qwen35_27b_trial_20261006'
OUT = ROOT / 'outputs/qwen35_27b_partial_closure_20261006'
REPORT = ROOT / 'reports/Qwen3.5_27B本机试运行结果说明_2026-10-06.md'


def run():
    assert not OUT.exists(), 'Preserve the previous closure.'
    manifest = c.read(TRIAL / 'manifest.json')
    base = c.read(ROOT / manifest['base_manifest'])
    status = c.read(TRIAL / 'campaign_status.json')
    budget = c.read(TRIAL / 'budget.json')
    assert status['state'] == 'stopped_after_failure'
    assert budget['api_attempts_started'] == budget['successful_http_responses'] == budget['logical_requests_completed'] == 3
    assert len(list((TRIAL / 'results').glob('*.json'))) == 3
    assert len(list((TRIAL / 'attempts').glob('*.json'))) == 3
    assert len(list((TRIAL / 'raw_responses').glob('*.json'))) == 3
    ledgers = {}
    for name in ['outputs/qwen35_27b_download_resume_20261006/completion_manifest.json',
                 'outputs/qwen35_27b_trial_20261006/completion_manifest.json',
                 'outputs/llm_goal_role_8b_20261006/completion_manifest.json',
                 'outputs/llm_goal_role_14b_20261006/completion_manifest.json']:
        ledgers[name] = ledger_check(ROOT / name)
    for name, digest in {**manifest['frozen_sha256'], **manifest['parent_completion_sha256']}.items():
        assert sha(ROOT / name) == digest
    assert sha(c.STATUS) == manifest['old_global_status_sha256']
    for a, b in zip(base['request_order'], manifest['request_order']):
        assert {k for k in a if a[k] != b[k]} == {'payload', 'payload_sha256'}
        assert {k for k in a['payload'] if a['payload'][k] != b['payload'][k]} == {'model'}
        assert b['payload']['model'] == 'qwen3.5:27b'
        assert hash_value(b['payload']) == b['payload_sha256']
    rows = []
    for request in manifest['request_order'][:3]:
        rid = request['request_id']
        data = json.loads(request['payload']['messages'][0]['content'].split('Data: ')[1])
        p, location = data['probability_report_correct'], data['reported_location']
        at_report = p if data['report_role'] == 'hazard' else 1-p
        costs = {location: at_report, ('south' if location == 'north' else 'north'): 1-at_report}
        ref = (min if data['objective'] == 'minimize' else max)(costs, key=costs.get)
        assert ref == request['private_reference']['reference_exit']
        raw = c.read(TRIAL / 'raw_responses' / (rid + '.json'))
        result = c.read(TRIAL / 'results' / (rid + '.json'))
        assert raw['payload'] == request['payload'] and raw['http_status'] == 200
        assert raw['response']['done'] and raw['response']['model'] == 'qwen3.5:27b'
        score = classify(raw['response'], ref)
        assert all(result['outcome'][k] == v for k, v in score.items())
        assert score['usable_correct']
        rows.append({'request_id': rid, 'condition': request['condition_id'], 'reference_exit': ref,
            'selected_exit': score['semantic_direction'], 'strict_format_valid': score['strict_format_valid'],
            'usable_correct': score['usable_correct'], 'latency_seconds': raw['latency_seconds'],
            'generated_tokens': raw['response'].get('eval_count'), 'final_content': raw['response']['message']['content']})
    failure = c.read(TRIAL / 'failure.json')
    assert 'Available memory below 1GiB' in failure['error']
    resource = c.read(TRIAL / 'resource_samples.json')
    samples = resource['samples']
    assert resource['minimum_available_memory_bytes'] == min(s['available_memory_bytes'] for s in samples)
    assert resource['minimum_available_memory_bytes'] < 1024**3
    loaded = c.read(TRIAL / 'loaded_model_snapshot.json')['models'][0]
    assert loaded['name'] == 'qwen3.5:27b' and loaded['size_vram'] == 0 and loaded['context_length'] == 4096
    unloaded = c.read(TRIAL / 'unload.json')
    assert not unloaded['ps_after']['models'] and unloaded['generated_tokens'] == 0
    assert unloaded['unload_response']['done_reason'] == 'unload'
    live = requests.get('http://127.0.0.1:11434/api/ps', timeout=10).json()
    assert not live['models']
    OUT.mkdir()
    save(OUT / 'audit.json', {'record_audit_passed': True, 'ledgers': ledgers, 'attempted': 3, 'delivered': 3,
         'independently_reclassified_correct': 3, 'unattempted': 29, 'planned': 32, 'rows': rows,
         'stop_reason': 'available_memory_below_predeclared_floor', 'sample_count': len(samples),
         'samples_below_1GiB': sum(s['available_memory_bytes'] < 1024**3 for s in samples),
         'minimum_sampled_available_memory_bytes': resource['minimum_available_memory_bytes'],
         'maximum_system_pagefile_used_bytes': resource['maximum_system_pagefile_used_bytes'],
         'model_load_snapshot': loaded, 'unloaded_without_generation': True,
         'global_status_and_default_selection_unchanged': True, 'new_inference': 0,
         'full_32_request_correctness_not_established': True})
    lines = ['# Qwen3.5 27B本机试运行结果说明', '', '日期：2026-10-06（北京时间）。', '',
        '27B Q4_K_M已下载到D盘，实际成功加载并返回3条正确答案。第三条完成后，下一条请求开始前可用内存低于预设1GiB门槛，资源保护停止了后续测试。随后已卸载模型。', '',
        '这次结果是“可以勉强运行”，尚不能认定适合长期批量实验；没有完成整套32题，也不能判断它是否比8B或14B整体更准确。', '',
        f"实际模型26,895,998,464参数，GGUF Q4_K_M；Ollama报告加载大小{loaded['size']/1e9:.2f}GB，CPU推理（size_vram=0），上下文4096。", '',
        f"采样最低可用内存约{resource['minimum_available_memory_bytes']/1e6:.1f}MB；系统分页文件使用量最高约{resource['maximum_system_pagefile_used_bytes']/1e9:.2f}GB。这是整机观测，不把分页使用量全部归因于模型。采样不保证精确峰值。", '',
        '| 已交付题目 | 条件 | 参考方向 | 实际方向 | 耗时秒 |', '| --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['request_id']} | {row['condition']} | {row['reference_exit']} | {row['selected_exit']} | {row['latency_seconds']:.2f} |")
    lines += ['', '首条58.25秒包含加载，其后两条分别25.88和26.27秒；三条各生成22个token。仅包含两条minimize_hazard和一条maximize_clear，没有覆盖全部目标与角色条件。未执行的29条不算答错。', '',
        '提示、schema、顺序、temperature0、seed2026100601、1024输出预算与4096上下文保持原计划，仅model字段变化。模型代际、架构、训练与模板不同，不能作纯参数量因果比较。无模拟、PPO更新、付费API或额外推理。', '',
        '原试运行记录与报告已经封存，本说明作为单独的补充记录；旧8B、14B台账、全局研究状态和默认模型选择均已核验未改。', '',
        '如果继续测试27B，应先释放更多后台内存，再另行冻结从第4条起的续测记录；不能降低已声明的内存门槛来强行扩大运行。当前配置下，日常批量实验继续使用已验证的较小模型更合适。', '',
        f'[原始试运行报告]({(ROOT / "reports/Qwen3.5_27B本机试运行报告_2026-10-06.md").as_posix()})；[独立核验记录]({(OUT / "audit.json").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    files = [OUT / 'audit.json', REPORT, Path(__file__)] + [ROOT / p for p in ledgers]
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}, 'new_inference': 0,
         'partial_trial_scope': '3 completed, 29 unattempted; resource stop'})
    ledger_check(OUT / 'completion_manifest.json')
    print(json.dumps({'audited_answers': 3, 'correct': 3, 'unattempted': 29, 'model_unloaded': True,
                      'minimum_available_memory_MB': resource['minimum_available_memory_bytes']/1e6,
                      'report': str(REPORT)}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
