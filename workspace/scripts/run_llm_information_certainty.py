"""Frozen information-condition experiment, using preserved output-study I/O."""
from pathlib import Path
import argparse
import json
import os
import random
import statistics
import sys
import traceback

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
from llm_information_certainty_core import CONDITIONS, cases, payload, private_reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_information_certainty_20261006'
RUNTIME = ROOT / 'outputs/llm_information_certainty_runtime_20261006'
PROTOCOL = ROOT / 'reports/LLM信息确定性对照固定方案_2026-10-06.md'
REPORT = ROOT / 'reports/LLM信息确定性对照实验报告_2026-10-06.md'
PREVIOUS = ROOT / 'outputs/llm_output_mechanism_20261006'
io.OUT = OUT  # Process-local destination; old files and code are never edited.
assert io.OUT != PREVIOUS


def prepare():
    assert not OUT.exists()
    preflight = c.read(RUNTIME/'pre_inference_checks.json')
    assert preflight['all_checks_passed']
    for name, digest in preflight['checked_source_sha256'].items():
        assert c.sha(ROOT/name) == digest, name
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    parents = [ROOT / p for p in c.read(PREVIOUS / 'manifest.json')['parent_completion_sha256']]
    parents += [PREVIOUS / 'completion_manifest.json',
                ROOT / 'outputs/llm_output_mechanism_review_20261006/completion_manifest.json',
                ROOT / 'outputs/llm_output_mechanism_runtime_20261006/completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p, HISTORY if p.parent == MAIN else None) for p in parents}
    tags = io.api('/api/tags', timeout=10)
    model = next(v for v in tags['models'] if v['name'] == 'qwen3:4b')
    assert model['digest'] == io.DIGEST
    snapshot = {'tags': tags, 'version': io.api('/api/version', timeout=10),
                'show': io.api('/api/show', {'model': 'qwen3:4b'}, 10)}
    records = cases()
    requests = []
    rng = random.Random(2026100603)
    families = sorted({s['pair_id'] for s in records})
    rng.shuffle(families)
    for family in families:
        block = [(s, cond) for s in records if s['pair_id'] == family for cond in CONDITIONS]
        rng.shuffle(block)
        for s, cond in block:
            body = payload(s, cond)
            requests.append({'request_id': f"{s['case_id']}_{cond['id']}", 'case_id': s['case_id'],
                'pair_id': family, 'condition_id': cond['id'], 'payload': body,
                'payload_sha256': io.hash_value(body), 'private_reference': private_reference(s, cond)})
    assert len(requests) == 32 and len(records) == 8 and len(families) == 4
    paths = [Path(__file__), ROOT/'scripts/llm_information_certainty_core.py',
        ROOT/'scripts/verify_llm_information_certainty.py', ROOT/'tests/test_llm_information_certainty_core.py', PROTOCOL,
        ROOT/'scripts/run_llm_output_mechanism.py', ROOT/'scripts/llm_output_mechanism_core.py',
        ROOT/'scripts/verify_llm_output_mechanism.py', ROOT/'scripts/auto_evacuation_campaign.py',
        ROOT/'scripts/finalize_auto_evacuation_campaign_review.py']
    OUT.mkdir()
    io.save(OUT/'pre_inference_checks.json', preflight)
    io.save(OUT/'model_service_snapshot.json', snapshot)
    io.save(OUT/'manifest.json', {'role': 'fixed_one_step_information_certainty_probe', 'created_at_utc': c.stamp(),
        'local_date': '2026-10-06', 'cases': records, 'conditions': list(CONDITIONS), 'request_order': requests,
        'model_digest': io.DIGEST, 'logical_request_count': 32, 'api_attempt_cap': 64,
        'request_order_seed': 2026100603, 'parent_checks': parent_checks,
        'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
        'model_snapshot_sha256': c.sha(OUT/'model_service_snapshot.json'),
        'pre_inference_checks_sha256': c.sha(OUT/'pre_inference_checks.json'),
        'old_global_status_sha256': c.sha(c.STATUS), 'same_output_settings_as_previous_successful_cell': True,
        'process_local_io_destination': OUT.relative_to(ROOT).as_posix(),
        'single_user_message_per_request': True, 'wrong_invalid_empty_truncated_are_not_retried': True,
        'simulations': 0, 'ppo_updates': 0, 'fits': 0, 'max_retry_only_transport_errors': 1,
        'mathematical_cost_not_real_evacuations': True, 'historical_root_cause_not_established_by_before_after': True})
    io.save(OUT/'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0,
        'logical_requests_completed': 0, 'simulations': 0, 'ppo_updates': 0, 'fits': 0})
    io.save(OUT/'campaign_status.json', {'state': 'ready', 'stage': 'prepared', 'pid': None,
        'run_attempts': 0, 'created_at_utc': c.stamp(), 'completed_logical_requests': 0})
    print('32 information-condition requests frozen; no inference yet.', flush=True)


def summarize(m):
    rows, groups, outcomes = [], {}, {}
    for req in m['request_order']:
        result = c.read(OUT/'results'/f"{req['request_id']}.json")
        o, ref = result['outcome'], req['private_reference']
        usable = o['strict_format_valid'] and not o['truncated'] and o.get('semantic_direction') in ('north', 'south')
        expected = ref['expected_cost_by_exit'][o['semantic_direction']] if usable else None
        regret = expected - ref['minimum_expected_cost'] if usable else None
        row = {'request_id': req['request_id'], 'case_id': req['case_id'], 'pair_id': req['pair_id'],
            'condition': req['condition_id'], 'category': o['category'], 'usable_correct': o['usable_correct'],
            'strict_format_valid': o['strict_format_valid'], 'direction': o.get('semantic_direction'),
            'reference_exit': ref['reference_exit'], 'minimum_expected_cost': ref['minimum_expected_cost'],
            'selected_expected_cost_if_valid': expected, 'expected_regret_if_valid': regret,
            'truncated': o['truncated'], 'thinking_present': o['thinking_present'],
            'confidence': o.get('confidence'), 'latency_seconds': result['latency_seconds'],
            'generated_tokens': o.get('generated_tokens'), 'prompt_eval_tokens': o.get('prompt_eval_tokens')}
        rows.append(row)
        outcomes[(req['case_id'], req['condition_id'])] = row
    for cond in m['conditions']:
        values = [r for r in rows if r['condition'] == cond['id']]
        counts = {k: sum(r['category'] == k for r in values) for k in
            ('usable_correct', 'valid_wrong_direction', 'invalid_format', 'truncated', 'transport_failure')}
        swaps = 0
        for pair in sorted({s['pair_id'] for s in m['cases']}):
            a, b = [s for s in m['cases'] if s['pair_id'] == pair]
            da, db = outcomes[(a['case_id'], cond['id'])]['direction'], outcomes[(b['case_id'], cond['id'])]['direction']
            swaps += da is not None and db is not None and da != db
        times = [r['latency_seconds'] for r in values if r['latency_seconds'] is not None]
        costs = [r['selected_expected_cost_if_valid'] for r in values if r['selected_expected_cost_if_valid'] is not None]
        regrets = [r['expected_regret_if_valid'] for r in values if r['expected_regret_if_valid'] is not None]
        groups[cond['id']] = {'cases': 8, **counts, 'mirror_pairs_swapped': int(swaps),
            'basic_function_passed': counts['usable_correct'] == 8 and swaps == 4,
            'thinking_present': sum(r['thinking_present'] for r in values),
            'mean_latency_seconds': statistics.mean(times) if times else None, 'total_latency_seconds': sum(times),
            'generated_tokens': sum(r['generated_tokens'] or 0 for r in values),
            'prompt_eval_tokens': sum(r['prompt_eval_tokens'] or 0 for r in values),
            'valid_final_cost_count': len(costs), 'mean_expected_cost_if_valid': statistics.mean(costs) if costs else None,
            'mean_expected_regret_if_valid': statistics.mean(regrets) if regrets else None}
    comparisons = []
    for name, a, b in [('forecast_label', 'confirmed_100', 'forecast_100'),
                        ('uncertainty_90', 'forecast_100', 'forecast_90'),
                        ('uncertainty_60', 'forecast_100', 'forecast_60')]:
        changes = [int(outcomes[(s['case_id'], b)]['usable_correct']) - int(outcomes[(s['case_id'], a)]['usable_correct']) for s in m['cases']]
        comparisons.append({'comparison': name, 'a': a, 'b': b, 'b_minus_a_correct_cases': sum(changes),
            'improved_cases': changes.count(1), 'worsened_cases': changes.count(-1), 'unchanged_cases': changes.count(0)})
    c.csv_write(OUT/'case_results.csv', rows)
    c.csv_write(OUT/'paired_information_effects.csv', comparisons)
    summary = {'groups': groups, 'paired_comparisons': comparisons,
        'all_conditions_passed': all(g['basic_function_passed'] for g in groups.values()),
        'previous_failure_root_cause_established': False, 'paid_api_requests': 0, 'new_simulations': 0,
        'new_ppo_updates': 0, 'new_fits': 0, 'scope': '8 fixed mathematical cases in4 mirrored families'}
    io.save(OUT/'summary.json', summary)
    return summary


def write_report(m, summary):
    labels = {'confirmed_100': '已确认，100%', 'forecast_100': '预测，100%（理想桥接）',
              'forecast_90': '预测，90%', 'forecast_60': '预测，60%'}
    lines = ['# LLM信息确定性对照实验报告', '', '日期：2026-10-06（北京时间）。冻结的32个本地聊天请求已完成。', '',
        '8道新题、4个镜像家庭，同题四条件配对。模型qwen3:4b，JSON约束开、请求think=false，temperature0、seed2026100601、num_predict1024、num_ctx4096。每个请求仅有一个用户消息，提示正文相同，只有报告类型和预定概率变化。', '',
        '这是一个单步数学世界：一次通行中恰有一侧暴露代价1、另一侧0，两出口的距离、容量和初始人数相同。危险位置报告的正确概率q由题目给定且按定义校准；若报告错，危险在另一侧。模型依据这些概率选出口，不看到随机实现、计算好的期望损失或参考出口。它不是火灾物理模拟，不测FED、失能率或疏散时间。', '',
        '| 信息条件 | 可用且正确/8 | 合法但方向错误 | 格式错误 | 截断 | 传输失败 | 镜像互换/4 | 平均回包秒数 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for cid, label in labels.items():
        g = summary['groups'][cid]
        latency = '不可估计' if g['mean_latency_seconds'] is None else f"{g['mean_latency_seconds']:.3f}"
        lines.append(f"| {label} | {g['usable_correct']} | {g['valid_wrong_direction']} | {g['invalid_format']} | {g['truncated']} | {g['transport_failure']} | {g['mirror_pairs_swapped']} | {latency} |")
    lines += ['', '每条件需8/8严格可用且方向正确、4/4镜像互换才称本轮基础功能通过。仅判断最终content；thinking不用于补答案，截断优先单列，未按答案表现重试。', '',
        '| 信息条件 | 理论最小期望暴露 | 有合法完整答案的数量 | 这些答案的平均期望暴露 | 平均期望遗憾 | thinking回包数 |',
        '| --- | --- | --- | --- | --- | --- |']
    for cid, label in labels.items():
        g = summary['groups'][cid]
        p = next(c['probability'] for c in m['conditions'] if c['id'] == cid)
        cost = '未交付，不计损失' if g['mean_expected_cost_if_valid'] is None else f"{g['mean_expected_cost_if_valid']:.3f}"
        regret = '未交付，不计损失' if g['mean_expected_regret_if_valid'] is None else f"{g['mean_expected_regret_if_valid']:.3f}"
        lines.append(f"| {label} | {1-p:.3f} | {g['valid_final_cost_count']} | {cost} | {regret} | {g['thinking_present']} |")
    lines += ['', '暴露与遗憾是题目定义的0–1抽象代价，只在合法且未截断的最终答案上计算。没有为缺失答案设置隐藏回退政策。90%与60%条件的理论最小暴露0.1和0.4是信息不确定性本身留下的风险，不能称为模型多造成的损失。LLM自报confidence没有经过校准验证，也不等于报告可靠性。', '',
        '| 同题配对变化（右减左） | 正确案例净变化 | 改善 | 变差 | 不变 |', '| --- | --- | --- | --- | --- |']
    for p in summary['paired_comparisons']:
        lines.append(f"| {labels[p['b']]} 减 {labels[p['a']]} | {p['b_minus_a_correct_cases']:+d} | {p['improved_cases']} | {p['worsened_cases']} | {p['unchanged_cases']} |")
    groups = summary['groups']
    if summary['all_conditions_passed']:
        interpretation = '四条件均通过这组基础功能题。把已确认改称理想100%预测，或把预测可靠性降至90%和60%，都没有降低这8题的可用且正确率。这没有重现旧轮跟随预测危险方向的行为；当前提示和明确概率条件下，模型可以完成这项简单期望暴露选择。'
    elif groups['confirmed_100']['basic_function_passed'] and not groups['forecast_100']['basic_function_passed']:
        interpretation = '已确认条件通过，而理想100%预测未通过。两者数学危险概率相同，仅报告标签不同，提示标签相关信号；仍需按表区分方向错误与交付失败，不能据此认定内部机理或现实预测能力。'
    elif groups['forecast_100']['basic_function_passed'] and (not groups['forecast_90']['basic_function_passed'] or not groups['forecast_60']['basic_function_passed']):
        interpretation = '理想100%预测通过，但至少一个较低概率条件未通过；这是明确概率条件相关信号。参考方向在这些条件下相同，应按表区分实际方向变化、格式、截断与传输失败。'
    else:
        interpretation = '结果未满足全部基础功能条件，应保留各条件的具体失败类型；当前范围内的配对结果不足以确定历史错误或模型内部原因。'
    lines += ['', interpretation, '',
        '不能把这轮与历史差异直接当成根因证据。旧题没有给出校准概率和完整单步概率生成规则；本轮也明确了期望暴露目标，并使用新题及1024上限。相较上一轮，本轮改成统一的概率世界，跨轮不是单因素对照。100%预测是理想数学桥接，不代表任何真实预测器已达到该可靠性。8题属于4个固定镜像家庭，32回包不是32独立场景；没有总体显著性、复杂疏散、拥堵取舍或LLM优于规则/PPO的证据。', '']
    budget = c.read(OUT/'budget.json')
    latency = sum(g['total_latency_seconds'] for g in groups.values())
    generated = sum(g['generated_tokens'] for g in groups.values())
    prompt_tokens = sum(g['prompt_eval_tokens'] for g in groups.values())
    lines += [f"成本：32个逻辑请求，实际开始{budget['api_attempts_started']}次本地聊天调用，成功服务响应{budget['successful_http_responses']}个，累计回包等待{latency:.2f}秒；提示评估token{prompt_tokens}、生成token{generated}。无付费API、模拟0、PPO更新0、拟合0。", '',
        '全部实际payload、开始标记、原始响应和分类保存；输出目录与旧轮隔离。独立核验重算概率参考、同题控制变量、镜像、配对差异、暴露与预算，并逐文件核验全部祖先台账。旧全局状态保持原字节，保留48个疏散场景未使用。', '',
        '下一步仍不扩大PPO训练。若本轮全部通过，优先做独立冻结的历史提示结构对照，隔离字段名称、给定概率和目标说明；不要把全部提示同时重写后宣称定位根因。若有失败，先按预定分类解释失败，不修改本轮提示或增加输出预算反复刷题。复杂疏散验证需要新的固定方案，不能直接放行保留场景。', '',
        f'[固定方案]({PROTOCOL.as_posix()})；[逐题结果]({(OUT/"case_results.csv").as_posix()})；[配对差异]({(OUT/"paired_information_effects.csv").as_posix()})；[独立验证]({(OUT/"verification.json").as_posix()})；[摘要]({(OUT/"summary.json").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def run():
    m, status = c.read(OUT/'manifest.json'), c.read(OUT/'campaign_status.json')
    if status['state'] == 'complete':
        return
    if status.get('pid') != os.getpid() and c.pid_live(status.get('pid')):
        print('Active worker preserved; no duplicate requests.', flush=True)
        return
    io.check_frozen(m)
    assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == 'qwen3:4b')['digest'] == io.DIGEST
    status.update(state='running', stage='inference', pid=os.getpid(), run_attempts=status['run_attempts']+1)
    io.save(OUT/'campaign_status.json', status)
    try:
        for i, req in enumerate(m['request_order']):
            result = io.fetch(req, req['private_reference']['reference_exit'])
            io.reconcile_budget()
            status.update(completed_logical_requests=i+1, last_request=req['request_id'], updated_at_utc=c.stamp())
            io.save(OUT/'campaign_status.json', status)
            print(json.dumps({'completed': i+1, 'total': 32, 'condition': req['condition_id'],
                'category': result['outcome']['category']}), flush=True)
        io.check_frozen(m)
        assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == 'qwen3:4b')['digest'] == io.DIGEST
        summary = summarize(m)
        from verify_llm_information_certainty import verify
        io.save(OUT/'verification.json', verify())
        write_report(m, summary)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('completion_manifest.json', 'campaign_status.json') and not p.name.endswith('.tmp')]
        paths += [ROOT/p for p in m['frozen_sha256']] + [REPORT]
        io.save(OUT/'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
            'budget': c.read(OUT/'budget.json'), 'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(set(paths))},
            'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0})
        from finalize_auto_evacuation_campaign_review import ledger_check
        ledger_check(OUT/'completion_manifest.json')
        status.update(state='complete', stage='complete', completed_at_utc=c.stamp())
        io.save(OUT/'campaign_status.json', status)
        print('Information certainty experiment sealed and complete.', flush=True)
    except Exception:
        status.update(state='failed', failed_at_utc=c.stamp())
        io.save(OUT/'campaign_status.json', status)
        io.save(OUT/'failures'/f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc(), 'budget': c.read(OUT/'budget.json')})
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
