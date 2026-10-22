"""Post-completion, read-only audit and explanatory report; no new inference.

This reviewer is separate from the pre-inference frozen worker and does not
change its labels, pass criterion, cases, or report.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/llm_output_mechanism_20261006'
OUT = ROOT / 'outputs/llm_output_mechanism_review_20261006'
REPORT = ROOT / 'reports/LLM输出机制结果复核与通俗说明_2026-10-06.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def old_response_audit(directory):
    responses = [read(p)['response'] for p in (ROOT / directory / 'llm_raw_responses').glob('*.json')]
    return {'responses': len(responses),
            'done_reasons': dict(Counter(v.get('done_reason') for v in responses)),
            'empty_final_content': sum(not v.get('message', {}).get('content', '').strip() for v in responses),
            'nonempty_thinking': sum(bool(v.get('message', {}).get('thinking', '').strip()) for v in responses)}


def link(path, label):
    return f'[{label}]({Path(path).as_posix()})'


def run():
    assert not OUT.exists() and not REPORT.exists(), 'Do not overwrite completed reviews.'
    status = read(STUDY / 'campaign_status.json')
    assert status['state'] == 'complete'
    from auto_evacuation_campaign import pid_live
    assert not pid_live(status.get('pid')), 'Wait for worker exit before closure.'
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    manifest = read(STUDY / 'manifest.json')
    ancestor_counts = {}
    for name, digest in manifest['parent_completion_sha256'].items():
        path = ROOT / name
        assert sha(path) == digest
        ancestor_counts[name] = ledger_check(path, HISTORY if path.parent == MAIN else None)
    study_records = ledger_check(STUDY / 'completion_manifest.json')
    from verify_llm_output_mechanism import verify
    verification = verify()
    summary, budget = read(STUDY / 'summary.json'), read(STUDY / 'budget.json')
    labels = {'schema_on_think_off': 'JSON约束开、思考关', 'schema_off_think_off': 'JSON约束关、思考关',
              'schema_on_think_on': 'JSON约束开、思考开', 'schema_off_think_on': 'JSON约束关、思考开'}
    rows, examples = [], {}
    for req in manifest['request_order']:
        result = read(STUDY / 'results' / f"{req['request_id']}.json")
        if result['raw_path'] is None:
            continue
        raw = read(ROOT / result['raw_path'])
        response = raw['response']
        message = response['message']
        row = {'request_id': req['request_id'], 'condition': req['condition_id'],
               'category': result['outcome']['category'], 'latency_seconds': raw['latency_seconds'],
               'generated_tokens': response.get('eval_count', 0), 'prompt_tokens': response.get('prompt_eval_count', 0),
               'done_reason': response.get('done_reason'), 'final_characters': len(message.get('content', '')),
               'thinking_characters': len(message.get('thinking', ''))}
        rows.append(row)
        examples.setdefault(req['condition_id'], {'request_id': req['request_id'], 'raw_path': result['raw_path'],
            'category': row['category'], 'done_reason': row['done_reason'],
            'final_content_prefix': message.get('content', '')[:700],
            'final_content_suffix': message.get('content', '')[-300:]})
    costs = {}
    for condition in labels:
        values = [r for r in rows if r['condition'] == condition]
        times = [r['latency_seconds'] for r in values]
        costs[condition] = {'responses': len(values), 'total_latency_seconds': sum(times),
            'mean_latency_seconds': statistics.mean(times) if times else None,
            'median_latency_seconds': statistics.median(times) if times else None,
            'p95_latency_seconds': percentile(times, .95),
            'generated_tokens': sum(r['generated_tokens'] for r in values),
            'prompt_tokens': sum(r['prompt_tokens'] for r in values),
            'empty_final_content': sum(r['final_characters'] == 0 for r in values),
            'nonempty_final_without_thinking_field': sum(r['final_characters'] > 0 and r['thinking_characters'] == 0 for r in values)}
    totals = {'successful_response_latency_seconds': sum(r['latency_seconds'] for r in rows),
        'generated_tokens': sum(r['generated_tokens'] for r in rows),
        'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
        'usable_correct': sum(g['usable_correct'] for g in summary['groups'].values()),
        'valid_wrong_direction': sum(g['valid_wrong_direction'] for g in summary['groups'].values()),
        'truncated': sum(g['truncated'] for g in summary['groups'].values()),
        'invalid_format': sum(g['invalid_format'] for g in summary['groups'].values()),
        'transport_failure': sum(g['transport_failure'] for g in summary['groups'].values()),
        'paid_api_requests': 0, 'new_simulations': 0, 'new_ppo_updates': 0, 'new_fits': 0}
    old = {directory: old_response_audit(directory) for directory in (
        'outputs/llm_semantic_channel_diagnostic_20261005', 'outputs/evacuation_auto_llm_independent_20261005')}
    for directory, expected_count in [('outputs/llm_semantic_channel_diagnostic_20261005', 30),
                                       ('outputs/evacuation_auto_llm_independent_20261005', 48)]:
        assert old[directory]['responses'] == expected_count
        assert old[directory]['done_reasons'] == {'stop': expected_count}
        assert old[directory]['empty_final_content'] == 0
    checks = {**verification, 'sealed_study_records_checked': study_records, 'ancestor_records_checked': ancestor_counts,
        'worker_exited_before_review': True, 'old_responses_audited_read_only': old,
        'reviewer_sha256': sha(__file__), 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    lines = ['# LLM输出机制：结果复核与通俗说明', '',
        '日期：2026-10-06。本文件是在冻结的32请求实验完成后追加的解释与只读复核，不改变本轮分类、通过条件或历史结论。', '',
        '这轮问的是：同一个小模型面对“北边已确认着火、南边无烟”这样的简单题，能否按时交付正确出口的JSON？每道题分别打开或关闭服务端JSON约束、打开或关闭请求中的思考开关。模型、问题、输入、生成上限和上下文一致。', '',
        '所谓JSON约束，是服务端限制输出的语法；无约束组仍然收到同样的“只回答JSON”要求。它不是随意自由问答对照。思考开关是请求参数；返回的thinking字段与最终content分别保存，不用思考文字代替最终答案。', '',
        '| 设置 | 可用且正确/8 | 合法JSON选错方向 | 截断 | 其他格式错误 | 传输失败 | thinking回包/8 | 平均回包秒数 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for condition, label in labels.items():
        g, cost = summary['groups'][condition], costs[condition]
        mean = '—' if cost['mean_latency_seconds'] is None else f"{cost['mean_latency_seconds']:.2f}"
        lines.append(f"| {label} | {g['usable_correct']} | {g['valid_wrong_direction']} | {g['truncated']} | {g['invalid_format']} | {g['transport_failure']} | {g['thinking_present']} | {mean} |")
    passes = [labels[c] for c, g in summary['groups'].items() if g['basic_function_passed']]
    lines += ['', '达到预设基础功能条件的设置：' + ('、'.join(passes) if passes else '无') + '。条件为8/8严格格式且方向正确，以及4/4镜像方向互换；这是功能检查，不能代替复杂疏散评估。', '',
        f"32个逻辑请求合计：可用且正确{totals['usable_correct']}个，合法但方向错误{totals['valid_wrong_direction']}个，截断{totals['truncated']}个，其他格式错误{totals['invalid_format']}个，传输失败{totals['transport_failure']}个。截断优先单列，即便某段文字能暗示出口，也不重新计成可用回答。", '']
    if totals['truncated']:
        lines += ['本轮的截断说明模型在1024个生成token以内没有完成交付。开启思考时，思考文字也占这一预算；最终content可能为空。关闭JSON约束时也可能在content里生成很长的分析。这是原始回包的可观察现象，不足以断言模型内部是否彻底停止了思考，也不能把它说成模型一定选错出口。', '']
    off = [r for r in rows if r['condition'] == 'schema_off_think_off']
    if any(r['final_characters'] > 100 and r['thinking_characters'] == 0 for r in off):
        lines += ['特别需要留意：部分“请求思考关、JSON约束关”的回包在content里仍有长段分析。请求参数已经原样保存并核验，但仅凭开关名称不能认证实际内部执行方式。后续服务诊断应核对模式与模板的实际处理，再谈内部机理；本轮没有改模板或插入新的控制指令。', '']
    lines += ['这轮能说明什么', '',
        '可直接比较同一案例下两种设置带来的交付率变化。例如“JSON约束开减关”的配对数反映可用且正确答案的净变化。如果差别主要来自截断，应描述为交付和预算问题，而不能表述成结构化输出让模型更懂火灾。', '',
        '这轮不能说明什么', '',
        '旧轮30个诊断回包和48个独立LLM回包均为正常stop结束，并没有空最终内容。旧轮的错误方向是真实最终回答中的方向，与本轮截断不同。本轮采用已确认火源及明确无烟的另一通道，旧轮采用可能错误的预测；生成上限也从96变为1024。因此，前后成功率差不能单独归因于JSON设置、思考模式或信息确定性。', '',
        '8个题属于4个镜像家庭，32个回答是这些题的重复设置；容量的4个数值没有改变“走无烟侧”的参考逻辑。这是小范围工程功能测试，没有测试拥堵与烟气的取舍、错误预测、临时封闭或在线动态纠正，也不证明LLM比规则或纯PPO更好。', '',
        '成本与保存情况', '',
        f"实际开始{budget['api_attempts_started']}次本地聊天调用，成功回包{budget['successful_http_responses']}次。成功请求累计等待{totals['successful_response_latency_seconds']:.2f}秒（{totals['successful_response_latency_seconds']/60:.2f}分钟），提示评估token合计{totals['prompt_tokens']}，生成token合计{totals['generated_tokens']}。这是实测请求等待时间，包含模型加载等开销；没有测量电费或完整资源成本。付费API请求0，疏散模拟0，PPO更新0，拟合0。", '',
        '方向错误、格式错误和截断没有触发重试。旧模型、数据、报告、全局状态以及六份祖先完成台账逐文件核验；本轮完成台账也全部匹配。48个保留疏散场景仍未使用。', '',
        '下一步', '',
        '保持当前训练暂停。优先在另一个提前固定的小实验里，把“已确认火源”与“只有可能错误的预测”分开比较：使用相同输出设置、相同预算、相同物理条件，明确预测概率和实时烟气信息，只改变事先定义的信息条件。若模型仍跟随危险方向，需要先解释这项基础决策问题；若能稳定输出，再制定新的独立场景评估方案。不要直接把这一轮成功设置放行到48个保留场景，也不要靠增加PPO训练掩盖建议错误。', '',
        '模型qwen3:4b，digest为冻结的359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7。原始实验仍以其冻结方案、逐案例数据和独立验证为准。', '',
        link(ROOT/'reports/LLM输出机制四组对照实验报告_2026-10-06.md', '原始实验报告') + '；' + link(STUDY/'case_results.csv', '逐案例表') + '；' + link(STUDY/'paired_setting_effects.csv', '配对设置差异') + '；' + link(OUT/'review.json', '成本与复核明细') + '。', '']
    OUT.mkdir()
    save(OUT/'review.json', {'verification': checks, 'costs': costs, 'totals': totals, 'examples': examples,
        'predefined_summary_sha256': sha(STUDY/'summary.json'), 'previous_failure_root_cause_established': False})
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    files = [OUT/'review.json', REPORT, Path(__file__)]
    save(OUT/'completion_manifest.json', {'all_checks_passed': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'new_inference': 0, 'new_simulations': 0, 'new_ppo_updates': 0, 'new_fits': 0,
        'base_study_completion_sha256': sha(STUDY/'completion_manifest.json')})
    ledger_check(OUT/'completion_manifest.json')
    print(json.dumps({'review_complete': True, 'totals': totals, 'basic_function_passed_settings': passes,
                      'ancestor_counts': ancestor_counts, 'study_records': study_records}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
