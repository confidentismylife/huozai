"""Frozen descriptive analysis; goal sensitivity and encoding invariance."""
from pathlib import Path
import statistics
import json

import auto_evacuation_campaign as c
from run_llm_output_mechanism import save
from llm_goal_role_core import regret

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/llm_goal_role_20261006'
REPORT = ROOT/'reports/LLM任务目标与字段角色对照实验报告_2026-10-06.md'
PROTOCOL = ROOT/'reports/LLM任务目标与字段角色对照固定方案_2026-10-06.md'


def paired_metrics(rows, cases):
    by = {(r['case_id'],r['condition']):r for r in rows}
    results = []
    for role in ('hazard','clear'):
        values = [(by[(s['case_id'],f'minimize_{role}')],by[(s['case_id'],f'maximize_{role}')]) for s in cases]
        delivered = [(a,b) for a,b in values if a['valid_final'] and b['valid_final']]
        results.append({'comparison':f'goal_flip_{role}','kind':'goal_flip','pairs':8,
            'fully_delivered_pairs':len(delivered), 'expected_relation_observed':sum(a['direction']!=b['direction'] for a,b in delivered),
            'both_answers_correct':sum(a['usable_correct'] and b['usable_correct'] for a,b in values)})
    for objective in ('minimize','maximize'):
        values = [(by[(s['case_id'],f'{objective}_hazard')],by[(s['case_id'],f'{objective}_clear')]) for s in cases]
        delivered = [(a,b) for a,b in values if a['valid_final'] and b['valid_final']]
        results.append({'comparison':f'role_invariance_{objective}','kind':'role_invariance','pairs':8,
            'fully_delivered_pairs':len(delivered), 'expected_relation_observed':sum(a['direction']==b['direction'] for a,b in delivered),
            'both_answers_correct':sum(a['usable_correct'] and b['usable_correct'] for a,b in values)})
    return results


def summarize(m):
    rows, groups = [], {}
    for req in m['request_order']:
        result = c.read(OUT/'results'/f"{req['request_id']}.json")
        o, ref = result['outcome'], req['private_reference']
        data = json.loads(req['payload']['messages'][0]['content'].split('Data: ')[1])
        valid = bool(o['strict_format_valid'] and not o['truncated'] and o.get('semantic_direction') in ('north','south'))
        cost = ref['expected_cost_by_exit'][o['semantic_direction']] if valid else None
        rows.append({'request_id':req['request_id'],'case_id':req['case_id'],'pair_id':req['pair_id'],
            'condition':req['condition_id'],'objective':data['objective'],'role':data['report_role'],
            'reported_location':data['reported_location'],'probability':data['probability_report_correct'],
            'category':o['category'],'valid_final':valid,'usable_correct':o['usable_correct'],
            'direction':o.get('semantic_direction'),'reference_exit':ref['reference_exit'],
            'follows_reported_location':bool(valid and o['semantic_direction']==data['reported_location']),
            'selected_expected_cost_if_valid':cost,'optimal_expected_cost':ref['optimal_expected_cost'],
            'objective_regret_if_valid':regret(cost,ref['optimal_expected_cost'],data['objective']) if valid else None,
            'truncated':o['truncated'],'thinking_present':o['thinking_present'],'confidence':o.get('confidence'),
            'latency_seconds':result['latency_seconds'],'generated_tokens':o.get('generated_tokens'),
            'prompt_tokens':o.get('prompt_eval_tokens')})
    by = {(r['case_id'],r['condition']):r for r in rows}
    for condition in m['conditions']:
        cid = condition['id']
        values = [r for r in rows if r['condition']==cid]
        counts = {k:sum(r['category']==k for r in values) for k in
                  ('usable_correct','valid_wrong_direction','invalid_format','truncated','transport_failure')}
        swaps = 0
        for pair in sorted({s['pair_id'] for s in m['cases']}):
            a,b = [s for s in m['cases'] if s['pair_id']==pair]
            da,db = by[(a['case_id'],cid)]['direction'],by[(b['case_id'],cid)]['direction']
            swaps += da is not None and db is not None and da!=db
        valid = [r for r in values if r['valid_final']]
        times = [r['latency_seconds'] for r in values if r['latency_seconds'] is not None]
        groups[cid] = {'cases':8,**counts,'valid_final':len(valid),'mirror_pairs_swapped':int(swaps),
            'basic_function_passed':counts['usable_correct']==8 and swaps==4,
            'follows_reported_location':sum(r['follows_reported_location'] for r in values),
            'mean_expected_cost_if_valid':statistics.mean(r['selected_expected_cost_if_valid'] for r in valid) if valid else None,
            'mean_objective_regret_if_valid':statistics.mean(r['objective_regret_if_valid'] for r in valid) if valid else None,
            'mean_latency_seconds':statistics.mean(times) if times else None,'total_latency_seconds':sum(times),
            'thinking_present':sum(r['thinking_present'] for r in values),
            'generated_tokens':sum(r['generated_tokens'] or 0 for r in values),
            'prompt_tokens':sum(r['prompt_tokens'] or 0 for r in values)}
    pairs = paired_metrics(rows,m['cases'])
    c.csv_write(OUT/'case_results.csv',rows)
    c.csv_write(OUT/'paired_goal_role_results.csv',pairs)
    result = {'groups':groups,'paired_diagnostics':pairs,
        'all_cells_passed':all(g['basic_function_passed'] for g in groups.values()),
        'all_primary_pairs_correct':all(p['both_answers_correct']==8 and p['expected_relation_observed']==8 for p in pairs),
        'always_follows_reported_location_in_valid_answers':sum(r['follows_reported_location'] for r in rows)==32,
        'reported_location_match_count':sum(r['follows_reported_location'] for r in rows),
        'previous_failure_root_cause_established':False,'new_simulations':0,'new_fits':0,'new_ppo_updates':0}
    save(OUT/'summary.json',result)
    return result


def write_report(m, result):
    labels = {'minimize_hazard':'最小化 × 危险侧报告','maximize_hazard':'最大化 × 危险侧报告',
              'minimize_clear':'最小化 × 无烟侧报告','maximize_clear':'最大化 × 无烟侧报告'}
    lines = ['# LLM任务目标与字段角色对照报告','','日期：2026-10-06（北京时间）。冻结的32个请求已完成并独立核验。','',
        '这轮检查：同一概率世界中，改变数学任务目标是否会正确反转出口；用危险侧或无烟侧等价表示同一信息，是否保持答案。8道新题、4个镜像家庭、2×2完整对照，所有组使用同一提示正文与JSON schema、think=false请求、qwen3:4b、temperature0、seed2026100601、num_predict1024、num_ctx4096。','',
        '最大化组只问期望暴露较大的标签，是数学诊断条件，绝不是疏散策略。reported_location可表示危险侧或无烟侧，角色同时改变时该字段方向也反转；解码后的两出口危险概率完全相同。角色对照不能单独归因为字段名称。','',
        '| 条件 | 可用且符合数学目标/8 | 合法但目标错误 | 格式错误 | 截断 | 传输失败 | 跟随输入位置/8 | 镜像互换/4 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for cid,label in labels.items():
        g = result['groups'][cid]
        lines.append(f"| {label} | {g['usable_correct']} | {g['valid_wrong_direction']} | {g['invalid_format']} | {g['truncated']} | {g['transport_failure']} | {g['follows_reported_location']} | {g['mirror_pairs_swapped']} |")
    lines += ['','主判读依靠同题配对，不以某两组表面成功宣布修复。最小化与最大化应反转出口；危险侧与无烟侧两种等价表示应保持同一目标的出口。只有两答案都交付合法未截断JSON时，才统计观察到的方向关系。','',
        '| 配对检查 | 完整交付/8对 | 满足预定方向关系/8对 | 两答案均正确/8对 |', '| --- | --- | --- | --- |']
    pair_labels = {'goal_flip_hazard':'危险侧报告：目标改变应翻转', 'goal_flip_clear':'无烟侧报告：目标改变应翻转',
                   'role_invariance_minimize':'最小化：等价表示应不变', 'role_invariance_maximize':'最大化：等价表示应不变'}
    for p in result['paired_diagnostics']:
        lines.append(f"| {pair_labels[p['comparison']]} | {p['fully_delivered_pairs']} | {p['expected_relation_observed']} | {p['both_answers_correct']} |")
    if result['always_follows_reported_location_in_valid_answers']:
        interpretation = '32/32最终出口跟随reported_location，与目标无关。最小化无烟侧与最大化危险侧可能表面通过，但目标翻转和等价表示不变检查均失败。这个行为模式支持继续调查输入位置被映射为输出位置、目标或角色未被正确处理；它不证明模型内部仅复制字段，也不区分具体提示、模型或结构化生成机制。'
    elif result['all_cells_passed'] and result['all_primary_pairs_correct']:
        interpretation = '四组及全部主要配对检查均通过这组基础功能题。模型在当前明确任务与角色定义下能按目标翻转，并保持等价表示的答案。这是功能证据，不能据此前后差值确定历史失败根因或直接进入复杂疏散训练。'
    else:
        interpretation = '结果需同时看单组和配对检查。出口翻转未必正确、表示不变也未必正确，均需结合参考；合法交付与截断也必须分开。当前数据尚不能确定具体内部机理。'
    lines += ['',interpretation,'',
        '参考概率为每家庭预定的0.9或0.6。危险锚点暴露概率q，另一侧1-q；最大化与最小化唯一参考相反。题目仅提供角色、报告位置与可靠性，不提供算好的目标答案或实际火源实现。参考和代价是单步数学量，不能称为真实伤亡、FED或疏散效果。LLM自报confidence未认证为校准概率。','',
        '| 条件 | 合法完整答案数 | 这些答案平均期望暴露 | 平均目标遗憾 | 平均回包秒数 |', '| --- | --- | --- | --- | --- |']
    for cid,label in labels.items():
        g = result['groups'][cid]
        show = lambda v:'未交付' if v is None else f'{v:.3f}'
        lines.append(f"| {label} | {g['valid_final']} | {show(g['mean_expected_cost_if_valid'])} | {show(g['mean_objective_regret_if_valid'])} | {show(g['mean_latency_seconds'])} |")
    budget = c.read(OUT/'budget.json')
    latency = sum(g['total_latency_seconds'] for g in result['groups'].values())
    prompt_tokens = sum(g['prompt_tokens'] for g in result['groups'].values())
    tokens = sum(g['generated_tokens'] for g in result['groups'].values())
    lines += ['', '目标遗憾：最小化为所选代价减最小值；最大化为最大值减所选代价。没有为缺失答案设置回退政策或填入损失。','',
        f"预算：32逻辑请求，实际开始{budget['api_attempts_started']}次本地聊天调用、收到{budget['successful_http_responses']}个成功服务响应，累计回包等待{latency:.2f}秒。提示评估token{prompt_tokens}，生成token{tokens}。无付费API、模拟0、拟合0、PPO更新0。方向、格式和截断均没有触发重试。",'',
        '4个固定功能家庭的镜像变化不是独立真实场景；32回答不能当成32独立样本，不作总体显著性或泛化认证。新提示较上一轮同时增加了角色与目标定义，跨轮不是单因素归因。全部旧台账、源文件与全局状态保持原字节，保留48个疏散场景没有使用。','',
        '下一步保持扩大PPO训练暂停。若主要配对失败，应先检查目标与信息角色的处理，并用独立冻结对照区分直接JSON生成、先形成风险判断以及任务表述的影响。若全部通过，也应先用独立任务复现，不能挑通过的一组包装为LLM提升。','',
        f'[冻结方案]({PROTOCOL.as_posix()})；[逐题表]({(OUT/"case_results.csv").as_posix()})；[配对检查]({(OUT/"paired_goal_role_results.csv").as_posix()})；[独立验证]({(OUT/"verification.json").as_posix()})；[摘要]({(OUT/"summary.json").as_posix()})。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
