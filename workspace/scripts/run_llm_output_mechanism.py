"""Frozen 32-request output study. No simulator imports, fits or PPO updates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
import traceback
import urllib.request

import auto_evacuation_campaign as c
from llm_output_mechanism_core import cases, CONDITIONS, payload, grade

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_output_mechanism_20261006'
RUNTIME = ROOT / 'outputs/llm_output_mechanism_runtime_20261006'
PROTOCOL = ROOT / 'reports/LLM输出机制四组对照固定方案_2026-10-06.md'
REPORT = ROOT / 'reports/LLM输出机制四组对照实验报告_2026-10-06.md'
DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def save(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def api(path, value=None, timeout=180):
    request = urllib.request.Request('http://127.0.0.1:11434' + path,
        data=None if value is None else json.dumps(value).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def prepare():
    assert not OUT.exists()
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, LLM, MANAGER, HISTORY
    parents = [MAIN / 'completion_manifest_v2.json', LLM / 'completion_manifest.json', MANAGER / 'completion_manifest.json',
               ROOT / 'outputs/evacuation_auto_final_review_20261005/completion_manifest.json',
               ROOT / 'outputs/llm_semantic_channel_diagnostic_20261005/completion_manifest.json',
               ROOT / 'outputs/llm_semantic_channel_review_20261005/completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p, HISTORY if p.parent == MAIN else None) for p in parents}
    tags = api('/api/tags', timeout=10)
    model = next(x for x in tags['models'] if x['name'] == 'qwen3:4b')
    assert model['digest'] == DIGEST and 'thinking' in model['capabilities']
    snapshot = {'tags': tags, 'version': api('/api/version', timeout=10), 'show': api('/api/show', {'model': 'qwen3:4b'}, 10)}
    records = cases()
    assert len(records) == 8 and len({r['pair_id'] for r in records}) == 4
    requests = []
    rng = random.Random(2026100602)
    pairs = sorted({r['pair_id'] for r in records})
    rng.shuffle(pairs)
    for pair in pairs:
        block = [(case, cond) for case in records if case['pair_id'] == pair for cond in CONDITIONS]
        rng.shuffle(block)
        for case, cond in block:
            body = payload(case, cond)
            requests.append({'request_id': f"{case['case_id']}_{cond['id']}", 'case_id': case['case_id'],
                             'pair_id': pair, 'condition_id': cond['id'], 'payload': body, 'payload_sha256': hash_value(body)})
    assert len(requests) == 32
    paths = [Path(__file__), ROOT / 'scripts/llm_output_mechanism_core.py', ROOT / 'scripts/verify_llm_output_mechanism.py',
             ROOT / 'tests/test_llm_output_mechanism_core.py', PROTOCOL, ROOT / 'scripts/auto_evacuation_campaign.py']
    OUT.mkdir()
    save(OUT / 'model_service_snapshot.json', snapshot)
    save(OUT / 'manifest.json', {'role': 'fixed_2x2_local_llm_output_mechanism_comprehension_probe', 'created_at_utc': c.stamp(),
        'local_date': '2026-10-06', 'cases': records, 'conditions': list(CONDITIONS), 'request_order': requests,
        'model_digest': DIGEST, 'logical_request_count': 32, 'api_attempt_cap': 64,
        'parent_checks': parent_checks, 'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
        'model_snapshot_sha256': c.sha(OUT / 'model_service_snapshot.json'),
        'old_global_status_sha256': c.sha(c.STATUS), 'simulations': 0, 'ppo_updates': 0, 'fits': 0,
        'single_user_message_per_request': True, 'wrong_or_invalid_outputs_are_not_retried': True,
        'case_knowledge_changed_from_previous_probe': True, 'max_retry_only_transport_errors': 1,
        'pre_inference_tests': {'tests': 9, 'passed': 9, 'pytest_seconds': .06}})
    save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0, 'logical_requests_completed': 0,
                             'simulations': 0, 'ppo_updates': 0, 'fits': 0})
    save(OUT / 'campaign_status.json', {'state': 'ready', 'stage': 'prepared', 'pid': None, 'run_attempts': 0,
                                       'created_at_utc': c.stamp(), 'completed_logical_requests': 0})
    print('32 requests frozen; no model inference performed.', flush=True)


def check_frozen(m):
    for name, digest in m['frozen_sha256'].items():
        assert c.sha(ROOT / name) == digest, name
    for name, digest in m['parent_completion_sha256'].items():
        assert c.sha(ROOT / name) == digest, name
    assert c.sha(OUT / 'model_service_snapshot.json') == m['model_snapshot_sha256']
    assert c.sha(c.STATUS) == m['old_global_status_sha256']


def fetch(request, reference):
    request_id = request['request_id']
    result_path = OUT / 'results' / f'{request_id}.json'
    if result_path.exists():
        result = c.read(result_path)
        assert result['payload_sha256'] == request['payload_sha256']
        return result
    successful = None
    for attempt in (1, 2):
        raw_path = OUT / 'raw_responses' / f'{request_id}_attempt{attempt}.json'
        if raw_path.exists():
            raw = c.read(raw_path)
            if not raw.get('service_error'):
                successful = raw
                break
        marker = OUT / 'attempts' / f'{request_id}_attempt{attempt}.started.json'
        if marker.exists():
            continue
        budget = c.read(OUT / 'budget.json')
        assert budget['api_attempts_started'] < 64
        budget['api_attempts_started'] += 1
        save(OUT / 'budget.json', budget)
        save(marker, {'request_id': request_id, 'attempt': attempt, 'started_at_utc': c.stamp(),
                      'payload': request['payload'], 'payload_sha256': request['payload_sha256']})
        start = time.monotonic()
        try:
            response = api('/api/chat', request['payload'], 180)
        except Exception as exc:
            save(OUT / 'transport_failures' / f'{request_id}_attempt{attempt}.json', {'request_id': request_id,
                 'attempt': attempt, 'error': repr(exc), 'at_utc': c.stamp(), 'elapsed_seconds': time.monotonic() - start})
            continue
        service_error = response.get('error') or (None if response.get('done') is True else 'incomplete service response')
        raw = {'request_id': request_id, 'attempt': attempt, 'payload': request['payload'],
               'payload_sha256': request['payload_sha256'], 'response': response,
               'latency_seconds': time.monotonic() - start, 'received_at_utc': c.stamp(), 'service_error': service_error}
        # A local file-write error stops the worker; it never triggers an inference retry.
        save(raw_path, raw)
        if service_error:
            save(OUT / 'transport_failures' / f'{request_id}_attempt{attempt}.json', {'request_id': request_id,
                 'attempt': attempt, 'error': str(service_error), 'at_utc': c.stamp(), 'elapsed_seconds': time.monotonic() - start})
            continue
        successful = raw
        break
    if successful is not None:
        outcome = grade(successful['response'], reference)
        result = {'request_id': request_id, 'case_id': request['case_id'], 'condition_id': request['condition_id'],
                  'payload_sha256': request['payload_sha256'], 'outcome': outcome, 'attempt': successful['attempt'],
                  'raw_path': (OUT / 'raw_responses' / f"{request_id}_attempt{successful['attempt']}.json").relative_to(ROOT).as_posix(),
                  'latency_seconds': successful['latency_seconds'], 'created_at_utc': c.stamp()}
    else:
        result = {'request_id': request_id, 'case_id': request['case_id'], 'condition_id': request['condition_id'],
                  'payload_sha256': request['payload_sha256'], 'outcome': {'category': 'transport_failure',
                  'strict_format_valid': False, 'semantic_direction': None, 'semantic_correct': None,
                  'usable_correct': False, 'thinking_present': False, 'truncated': False},
                  'attempt': 2, 'raw_path': None, 'latency_seconds': None, 'created_at_utc': c.stamp()}
    save(result_path, result)
    return result


def reconcile_budget():
    budget = c.read(OUT / 'budget.json')
    markers = list((OUT / 'attempts').glob('*.started.json'))
    assert budget['api_attempts_started'] == len(markers)
    budget['successful_http_responses'] = sum(not c.read(p).get('service_error') for p in (OUT / 'raw_responses').glob('*.json'))
    budget['logical_requests_completed'] = len(list((OUT / 'results').glob('*.json')))
    save(OUT / 'budget.json', budget)
    return budget


def summary(m):
    results = {r['request_id']: c.read(OUT / 'results' / f"{r['request_id']}.json") for r in m['request_order']}
    groups, rows = {}, []
    for cond in m['conditions']:
        values = [r for r in results.values() if r['condition_id'] == cond['id']]
        outcomes = [r['outcome'] for r in values]
        by_case = {r['case_id']: r for r in values}
        mirror_pairs = 0
        for pair in sorted({s['pair_id'] for s in m['cases']}):
            a, b = [s for s in m['cases'] if s['pair_id'] == pair]
            oa, ob = by_case[a['case_id']]['outcome'], by_case[b['case_id']]['outcome']
            mirror_pairs += oa.get('semantic_direction') is not None and ob.get('semantic_direction') is not None and oa['semantic_direction'] != ob['semantic_direction']
        counts = {category: sum(o['category'] == category for o in outcomes) for category in
                  ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure')}
        latencies = [r['latency_seconds'] for r in values if r['latency_seconds'] is not None]
        groups[cond['id']] = {'cases': 8, **counts, 'strict_format_valid': sum(o['strict_format_valid'] for o in outcomes),
            'semantic_correct_including_invalid_or_truncated': sum(o.get('semantic_correct') is True for o in outcomes),
            'semantic_correct_nontruncated': sum(o.get('semantic_correct') is True and not o['truncated'] for o in outcomes),
            'thinking_present': sum(o['thinking_present'] for o in outcomes), 'mirror_pairs_swapped': int(mirror_pairs),
            'basic_function_passed': counts['usable_correct'] == 8 and mirror_pairs == 4,
            'mean_response_latency_seconds': statistics.mean(latencies) if latencies else None,
            'generated_tokens': sum(o.get('generated_tokens') or 0 for o in outcomes)}
        for r in values:
            rows.append({'case_id': r['case_id'], 'condition': cond['id'], 'schema': cond['schema'], 'think_requested': cond['think'],
                         **{key: r['outcome'].get(key) for key in ('category', 'usable_correct', 'strict_format_valid', 'semantic_direction',
                               'semantic_correct', 'truncated', 'thinking_present', 'done_reason', 'generated_tokens')},
                         'latency_seconds': r['latency_seconds']})
    pairs = []
    by_case_condition = {(r['case_id'], r['condition_id']): r['outcome'] for r in results.values()}
    comparisons = [('schema_effect_think_off', 'schema_off_think_off', 'schema_on_think_off'),
                   ('schema_effect_think_on', 'schema_off_think_on', 'schema_on_think_on'),
                   ('thinking_effect_schema_off', 'schema_off_think_off', 'schema_off_think_on'),
                   ('thinking_effect_schema_on', 'schema_on_think_off', 'schema_on_think_on')]
    for name, a, b in comparisons:
        delta = [int(by_case_condition[(s['case_id'], b)]['usable_correct']) - int(by_case_condition[(s['case_id'], a)]['usable_correct']) for s in m['cases']]
        pairs.append({'comparison': name, 'b_minus_a_correct_cases': sum(delta), 'improved_cases': delta.count(1), 'worsened_cases': delta.count(-1), 'unchanged_cases': delta.count(0)})
    c.csv_write(OUT / 'case_results.csv', rows)
    c.csv_write(OUT / 'paired_setting_effects.csv', pairs)
    result = {'groups': groups, 'paired_comparisons': pairs,
              'all_settings_pass': all(g['basic_function_passed'] for g in groups.values()),
              'any_setting_pass': any(g['basic_function_passed'] for g in groups.values()),
              'sample_scope': '8 fixed cases in4 mirror families, not32 independent scenarios',
              'previous_failure_root_cause_established': False, 'new_simulations': 0, 'ppo_updates': 0, 'fits': 0}
    save(OUT / 'summary.json', result)
    return result


def report(m, result):
    budget = c.read(OUT / 'budget.json')
    labels = {'schema_on_think_off': 'JSON约束开、思考关', 'schema_off_think_off': 'JSON约束关、思考关',
              'schema_on_think_on': 'JSON约束开、思考开', 'schema_off_think_on': 'JSON约束关、思考开'}
    lines = ['# LLM输出机制四组对照实验报告', '', '日期：2026-10-06（北京时间）。本轮固定32请求功能实验已完成，无模拟、PPO训练或模型拟合。', '',
             '8个新简单案例、4组北/南镜像家庭，火源已确认，另一侧无烟，两路开放且等容量等距离。相同模型、问题、信息和1024输出预算，仅改变服务端JSON schema约束与think开关。无schema时仍要求相同JSON回答，不能解释为任意自然语言问答。', '',
             '| 设置 | 严格可用且方向正确/8 | 合法但方向错误 | 格式错误 | 截断 | 传输失败 | 有thinking回包 | 镜像互换/4 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, g in result['groups'].items():
        lines.append(f"| {labels[name]} | {g['usable_correct']} | {g['valid_wrong_direction']} | {g['invalid_format']} | {g['truncated']} | {g['transport_failure']} | {g['thinking_present']} | {g['mirror_pairs_swapped']} |")
    lines += ['', '严格格式、语义方向与输出截断分开统计。思考字段不参与最终答案判断；格式不合法但可识别方向的回答另记在原始结果表，不能当作已可接入PPO的合法输入。', '',
              '| 设置 | 宽松识别方向正确且未截断/8 | 平均成功回包耗时（秒） | 生成tokens | 基础功能通过 |', '| --- | --- | --- | --- | --- |']
    for name, g in result['groups'].items():
        latency = '不可估计' if g['mean_response_latency_seconds'] is None else f"{g['mean_response_latency_seconds']:.3f}"
        lines.append(f"| {labels[name]} | {g['semantic_correct_nontruncated']} | {latency} | {g['generated_tokens']} | {g['basic_function_passed']} |")
    lines += ['', '配对变化只比较本轮同一案例；正数表示右侧设置增加严格可用且正确的案例数。它不是疏散效果量。', '',
              '| 设置比较（右减左） | 正确案例净变化 | 改善案例 | 变差案例 |', '| --- | --- | --- | --- |']
    compare_labels = {'schema_effect_think_off': '思考关：JSON开减关', 'schema_effect_think_on': '思考开：JSON开减关',
                      'thinking_effect_schema_off': 'JSON关：思考开减关', 'thinking_effect_schema_on': 'JSON开：思考开减关'}
    for p in result['paired_comparisons']:
        lines.append(f"| {compare_labels[p['comparison']]} | {p['b_minus_a_correct_cases']:+d} | {p['improved_cases']} | {p['worsened_cases']} |")
    if result['all_settings_pass']:
        interpretation = '四组均通过当前基础功能检查，未发现这些案例下JSON约束或think开关造成正确率差异。不能据此定位历史失败：确认信息与旧轮的不确定预测不同，预算也从96增至1024、上下文固定4096。下一步应在独立固定方案中复核确认信息与预测信息的差异，不自动放行48个疏散场景。'
    elif result['any_setting_pass']:
        interpretation = '至少一组通过基础功能检查，设置之间的配对结果提示输出机制相关影响。需同时检查该组是否实际产生思考字段、是否因格式或截断而失败，不能单凭请求的think开关推断内部机制。成功设置仍未经过复杂疏散或新场景验证。'
    else:
        interpretation = '四组均未达到8/8严格可用且方向正确的基础功能条件。请根据表中方向、格式、截断和传输结果区分原因；这限于当前模型、提示和预算，不证明所有LLM无效。若输出截断占主因，本轮不能据此断言理解失败。'
    lines += ['', interpretation, '',
        f"本轮完成{budget['logical_requests_completed']}个逻辑请求，实际开始{budget['api_attempts_started']}次聊天调用，收到{budget['successful_http_responses']}个成功服务响应；仅传输失败允许一次重试，错误方向/格式/截断没有重试。无付费API调用、模拟0、PPO更新0、拟合0。全部原始请求、最终内容、thinking字段、结束原因和耗时保存。", '',
        '模型qwen3:4b，固定digest `' + DIGEST + '`。temperature0、seed2026100601、num_predict1024、num_ctx4096。随机家庭与组内顺序提前冻结。32回答属于8固定案例的重复设置对照，不能当成32独立场景，不进行总体显著性或原门控资格认证。', '',
        '旧全局状态、模型、数据、报告和完成台账保持原字节；本轮没有修改历史提示或阈值。48个保留新疏散场景没有调用或模拟。', '',
        f'[固定方案]({PROTOCOL.as_posix()})；[逐案例原始结果表]({(OUT/"case_results.csv").as_posix()})；[配对设置结果]({(OUT/"paired_setting_effects.csv").as_posix()})；[摘要]({(OUT/"summary.json").as_posix()})；[独立验证]({(OUT/"verification.json").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def run():
    m = c.read(OUT / 'manifest.json')
    status = c.read(OUT / 'campaign_status.json')
    if status['state'] == 'complete':
        return
    if status.get('pid') != os.getpid() and c.pid_live(status.get('pid')):
        print('Worker active; no duplicate requests.', flush=True)
        return
    check_frozen(m)
    assert next(x for x in api('/api/tags', timeout=10)['models'] if x['name'] == 'qwen3:4b')['digest'] == DIGEST
    status.update(state='running', stage='inference', pid=os.getpid(), run_attempts=status['run_attempts'] + 1)
    save(OUT / 'campaign_status.json', status)
    try:
        case_map = {s['case_id']: s for s in m['cases']}
        for i, request in enumerate(m['request_order']):
            value = fetch(request, case_map[request['case_id']]['reference_exit'])
            reconcile_budget()
            status.update(completed_logical_requests=i + 1, total_logical_requests=32, last_request=request['request_id'], updated_at_utc=c.stamp())
            save(OUT / 'campaign_status.json', status)
            print(json.dumps({'completed': i + 1, 'total': 32, 'condition': request['condition_id'], 'category': value['outcome']['category']}), flush=True)
        assert next(x for x in api('/api/tags', timeout=10)['models'] if x['name'] == 'qwen3:4b')['digest'] == DIGEST
        check_frozen(m)
        result = summary(m)
        from verify_llm_output_mechanism import verify
        save(OUT / 'verification.json', verify())
        report(m, result)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('campaign_status.json', 'completion_manifest.json') and not p.name.endswith('.tmp')]
        paths += [PROTOCOL, REPORT, Path(__file__), ROOT / 'scripts/llm_output_mechanism_core.py', ROOT / 'scripts/verify_llm_output_mechanism.py',
                  ROOT / 'tests/test_llm_output_mechanism_core.py']
        save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(), 'budget': c.read(OUT / 'budget.json'),
             'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(set(paths))}, 'new_simulations': 0, 'new_ppo_updates': 0, 'new_fits': 0})
        for name, digest in c.read(OUT / 'completion_manifest.json')['files'].items():
            assert c.sha(ROOT / name) == digest, name
        status.update(state='complete', stage='complete', completed_at_utc=c.stamp())
        save(OUT / 'campaign_status.json', status)
        print('Output mechanism study sealed and complete.', flush=True)
    except Exception:
        status.update(state='failed', failed_at_utc=c.stamp())
        save(OUT / 'campaign_status.json', status)
        save(OUT / 'failures' / f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc(), 'budget': c.read(OUT / 'budget.json')})
        raise


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
