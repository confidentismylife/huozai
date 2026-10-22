"""Evidence-linked narrative for the fixed campaign, without winner selection."""
from pathlib import Path
import json
import pandas as pd

RAW = 'mappo_34_raw'
RULE = 'mappo_50_rule_plan'
OLD = 'mappo_50_deepseek_retrieval'
NEW = 'mappo_50_deepseek_expanded'
MASK = 'mappo_50_masked_control'
LABEL = {RAW: '34维纯MAPPO', RULE: '规则计划', OLD: '旧DeepSeek库',
         NEW: '新DeepSeek库', MASK: '50维全零计划通道'}


def interval(value, scale=100):
    lo, hi = value['crossed_ci95']
    return f"{scale*value['mean']:+.3f} [{scale*lo:+.3f}, {scale*hi:+.3f}]"


def reading(value):
    lo, hi = value['crossed_ci95']
    independent = value.get('independent_training_crossed_ci95')
    if independent and independent[0] <= 0 <= independent[1] and (hi < 0 or lo > 0):
        return '配对开发区间未跨0，但独立训练重采样敏感性区间跨0，判断依赖统计口径'
    if hi < 0:
        return '该开发集区间全部为负，支持候选的失能率较低；属于探索证据'
    if lo > 0:
        return '该开发集区间全部为正，支持候选的失能率较高；属于探索证据'
    return '该开发集区间跨0，尚未检出稳定的失能率差异'


def independent_interval(value):
    bounds = value.get('independent_training_crossed_ci95')
    return f'[{100*bounds[0]:+.3f}, {100*bounds[1]:+.3f}]' if bounds else '不适用'


def brief_judgement(payload):
    if payload['partial']:
        return '矩阵尚未齐全，本次预览不作总体优劣判断。'
    values = payload['primary_effects'].get(OLD + '_minus_' + RULE)
    parts = []
    if values:
        parts.append('原设置相对规则：' + reading(values['incapacitation_rate']) + '。')
    for setting, comparisons in payload['mechanism_effects'].items():
        value = comparisons.get(NEW + '_minus_' + RULE)
        if value:
            parts.append(setting + '的新库相对规则：' + reading(value['incapacitation_rate']) + '。')
    parts.append('这些结果不合并成一个总效应，也不能替代计划内容审计和独立最终评估。')
    return '\n\n'.join(parts)


def render(payload, output_root: Path):
    lines = ['## 14. 依据本轮数值作出的研究判断', '',
        '本节不重选配置、不重算另一套更有利的统计口径。失能差以候选减基线计，负值较好；'
        '时间差单位秒，负值较快。方括号为训练种子×开发场景交叉重采样95%区间，'
        '未校正探索比较的多重性。', '', brief_judgement(payload), '',
        '### 原设置：系统收益与通道依赖分开', '']
    for baseline in (RAW, RULE):
        values = payload['primary_effects'].get(OLD + '_minus_' + baseline)
        if values:
            lines += [f"- 相对{LABEL[baseline]}：失能{interval(values['incapacitation_rate'])}个百分点；"
                f"时间{interval(values['evacuation_time_seconds'], 1)}秒。"
                + reading(values['incapacitation_rate']) + '。']
    lines += ['', '同权重屏蔽计划造成的损失不等于LLM的独有贡献。'
        '原方案的三臂训练、同权重规则替换和置换是不同问题，各自区间已在前文列出。', '',
        '### 新协议：两种监督设置全部呈现', '',
        '| 设置 | 新库减基线 | 种子 | 失能差pp [配对95%区间] | 独立训练敏感性区间pp | 时间差s [95%区间] |',
        '|---|---|---:|---:|---:|---:|']
    for setting, comparisons in payload['mechanism_effects'].items():
        for baseline in (RAW, MASK, RULE, OLD):
            values = comparisons.get(NEW + '_minus_' + baseline)
            if values:
                lines.append(f"| {setting} | {LABEL[baseline]} | "
                    f"{values['incapacitation_rate']['n_training_seeds']} | "
                    f"{interval(values['incapacitation_rate'])} | "
                    f"{independent_interval(values['incapacitation_rate'])} | "
                    f"{interval(values['evacuation_time_seconds'], 1)} |")
    lines += ['', '两个设置共用同五个RAW和MASKED基线，不是十次独立复现。'
        '新库同时改变状态覆盖与提示单位说明，新协议还改了阶段矛盾标志及训练回合随机数；'
        '不能把与原设置的差异全部归为DeepSeek模型能力。独立训练的敏感性区间保存在analysis.json中。', '',
        '### 内容审计：给同一权重换计划后怎样', '',
        '| 设置/计划库 | 真实减规则：失能差pp | 真实减置换：失能差pp |',
        '|---|---:|---:|']
    for setting, comparisons in payload['channel_effects'].items():
        rule = comparisons.get('native_minus_rule')
        shuffled = comparisons.get('native_minus_shuffled')
        if rule and shuffled:
            lines.append(f"| {setting} | {interval(rule['incapacitation_rate'])} | "
                f"{interval(shuffled['incapacitation_rate'])} |")
    shuffled_results = [c['native_minus_shuffled']['incapacitation_rate']
        for c in payload['channel_effects'].values() if 'native_minus_shuffled' in c]
    if shuffled_results and all(v['crossed_ci95'][0] <= 0 <= v['crossed_ci95'][1]
                                for v in shuffled_results):
        lines += ['', '本轮四组新模型的真实计划减置换计划区间均跨0。'
            '因此，当前还没有稳定证据表明匹配当前观测的DeepSeek计划内容，'
            '比置换后的同类计划内容更有效；同时，有限置换强度和小开发集使这一零结果仍有边界。']
    lines += ['', '上述差异包括闭环状态变化；置换与屏蔽也可能造成分布偏移。'
        '应把这张表与独立训练基线一起读。不能只凭采用率、模仿率或屏蔽后的下降，'
        '声称模型理解了LLM计划或LLM内容不可替代。', '',
        '### 真正在线API的小样本诊断', '',
        '| 固定模型 | 在线减缓存：失能差pp | 在线减缓存：时间差s |',
        '|---|---:|---:|']
    for label, comparisons in payload['online_effects'].items():
        values = comparisons.get('online_minus_native')
        if values:
            lines.append(f"| {label} | {100*values['incapacitation_rate']['mean']:+.3f} | "
                f"{values['evacuation_time_seconds']['mean']:+.3f} |")
    online_results = [c['online_minus_native']['incapacitation_rate']
        for c in payload['online_effects'].values() if 'online_minus_native' in c]
    if online_results and all(v['mean'] > 0 for v in online_results):
        lines += ['', '这四个固定模型的即时在线版本平均失能率均高于缓存版本。'
            '这个小诊断没有支持立即用实时API替代当前缓存库；'
            '它也不能证明所有在线LLM方案必然更差。训练时检索计划与运行时新生成计划的分布差异，'
            '仍是需要单独检验的解释。']
    lines += ['', '这里每个模型只有六个开发场景和一个人口种子；仿真在API等待时暂停，'
        '而且输入另含时间戳。它回答在线计划是否有不同的局部表现，不构成多种子部署优势证据。', '',
        '### 本轮之后优先做什么', '',
        '先停止扩大当前协议的训练。先修复已复现的拥堵触发器分支错误，'
        '为封闭变化、稳定烟气下的拥堵越线、烟气反转及请求上限逐项做功能验收；'
        '另行冻结修改后的协议。当前批次完整保留，修复后不能与之合并或追溯改名。', '',
        '随后只选择一个有明确终点判据、信息对齐的有限确认实验，区分在线生成、'
        '缓存检索和人工规则。是否解封最终场景，应以冻结方案和完整核验为前提；'
        '不能为得到正结果反复修改计划、监督权重或评价条件。当前结果首先是单布局开发证据。', '']
    location = output_root / ('progress_analysis' if payload['partial'] else 'analysis')
    mechanism_path = location / 'mechanism_episode_results.csv'
    if mechanism_path.exists():
        data = pd.read_csv(mechanism_path)
        descriptive = []
        lines += ['### 场景差异：完整列出，避免只看总均值', '',
            '下表逐场景平均训练种子和人口重复，属于描述性拆分；'
            '没有为每行检验显著性，也不把同类一两个场景当作该类问题的总体。'
            '所有场景都来自已使用的开发集。', '',
            '| 设置 | 场景 | 冲突类型 | 新库减纯MAPPO失能pp | 新库减规则失能pp | 新库减旧库失能pp |',
            '|---|---|---|---:|---:|---:|']
        for setting, grouped in data.groupby('config_id'):
            pivot = grouped.pivot(index=['training_seed', 'scenario_id', 'population_seed'],
                columns='arm', values='incapacitation_rate')
            if NEW not in pivot:
                continue
            metadata = grouped[['scenario_id', 'conflict_type']].drop_duplicates().set_index('scenario_id')
            means = {}
            for baseline in (RAW, RULE, OLD):
                if baseline in pivot:
                    means[baseline] = 100*(pivot[NEW]-pivot[baseline]).groupby('scenario_id').mean()
            for scene in sorted(metadata.index):
                values = [float(means[b].get(scene, float('nan'))) if b in means else float('nan')
                    for b in (RAW, RULE, OLD)]
                conflict = metadata.loc[scene, 'conflict_type']
                lines.append(f'| {setting} | {scene} | {conflict} | '
                    + ' | '.join(f'{v:+.3f}' for v in values) + ' |')
                descriptive.append(dict(config_id=setting, scenario_id=scene,
                    conflict_type=conflict, new_minus_raw_pp=values[0],
                    new_minus_rule_pp=values[1], new_minus_old_pp=values[2]))
        pd.DataFrame(descriptive).to_csv(location/'scenario_effect_descriptions.csv',
            index=False, encoding='utf-8-sig')
        lines += ['', '训练种子均值方向一致，也不保证每个场景都受益。'
            '总平均的降低应与发生损失的场景同时报告；不能根据此表只保留有利场景，'
            '也不能用扩大同场景人口重复替代新场景的验证。', '']
    diagnostic = output_root / 'event_suppression_replay/audit.json'
    if diagnostic.exists():
        result = json.loads(diagnostic.read_text(encoding='utf-8'))
        groups = result['groups']
        lines += ['### 拥堵触发缺陷在既有轨迹中的实际频率', '',
            f"已对{result['model_count']}个保存模型、{result['episodes']}个既有评估回合做只读回放，"
            f"全部终点与原CSV一致，最大绝对差{result['maximum_replay_error']:.3g}。没有修复触发器、改变动作或调用API。", '',
            '| 协议/臂 | 回合 | 启用拥堵触发的越阈次数 | 被抑制次数 | 尚有请求额度时被抑制 |',
            '|---|---:|---:|---:|---:|']
        for row in groups:
            lines.append(f"| {row['protocol']}/{row['arm']} | {row['episodes']} | "
                f"{row['crossings']} | {row['suppressed']} | {row['suppressed_under_cap']} |")
        peak = result.get('maximum_junction_occupants')
        if peak is not None:
            lines += ['', f'这些回放中交汇区人数最高约{peak:.1f}人，未达到45人的触发阈值。'
                '因此在这个回放范围内，没有证据认为该触发器缺陷改变了既有结果；'
                '这也说明当前轨迹没有实际检验拥堵越阈后的重规划。']
        lines += ['', '这里只量化发生频率，未计算修复触发器之后的反事实终点收益；'
            '即使漏触发，也不能假定新增LLM调用一定有益。', '']
    return '\n'.join(lines)
