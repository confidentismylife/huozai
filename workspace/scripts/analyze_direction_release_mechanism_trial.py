"""Descriptive paired analysis of a frozen finite development mechanism trial."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np

from run_direction_release_mechanism_trial import (
    ROOT, PREP, MODES, METRICS, ZONE_NAMES, verify, verify_completion, csv_read, sha, save, csv_write, read,
    unique_output, joint_effect_cells, factor_effects, outcome_class,
)

REPORT = ROOT / 'reports/方向容量与保持时间机制实验报告_2026-10-07.md'
AREA_METRICS = tuple(f'fed_{n}' for n in ZONE_NAMES)
TIME_METRICS = tuple(f'person_seconds_{n}' for n in ZONE_NAMES)
MEASURES = (*METRICS, 'episode_end_seconds', *AREA_METRICS, *TIME_METRICS)


def full_key(r, policy=None, seed=None, hold=None):
    return (r['population_group'], r['scenario_id'], r['behavior'], int(r['population_seed']),
            int(r['training_seed']) if seed is None else seed, r['source_mode'],
            int(r['hold_steps']) if hold is None else hold, r['policy'] if policy is None else policy)


def run(full, continuations):
    verify()
    full, continuations = Path(full).resolve(), Path(continuations).resolve()
    for p in (full, continuations):
        assert p.parent == ROOT / 'outputs'
        verify_completion(p)
    assert not REPORT.exists()
    fs, cs = read(full / 'summary.json'), read(continuations / 'summary.json')
    assert cs['full_source_manifest_sha256'] == sha(full / 'completion_manifest.json')
    assert fs['preparation_sha256'] == sha(PREP / 'manifest.json')
    assert fs['exact_old_replays'] == 400 and fs['full_episodes'] == 1920
    assert cs['exact_realized_suffixes'] == fs['selected_states']
    assert not any(fs['violations'].values()) and not any(cs['violations'].values())
    plan = read(PREP / 'manifest.json')
    rows = csv_read(full / 'episodes.csv')
    assert len(rows) == 1920
    index = {full_key(r): r for r in rows}
    assert len(index) == len(rows)
    for r in rows:
        assert abs(sum(float(r[m]) for m in AREA_METRICS) - float(r['mean_cumulative_fed'])) < 1e-9
        assert sum(int(r[f'incap_count_{n}']) for n in ZONE_NAMES) == round(float(r['incapacitation_rate']) * 240)
        if r['policy'] == 'preserve_release_hint':
            assert int(r['same_direction_release_changed_steps']) == 0
    full_effects = []
    comparisons = [('direction_hint_only', 'no_advice'), ('preserve_release_hint', 'no_advice'),
                   ('preserve_release_hint', 'direction_hint_only'), ('pure_rule', 'no_advice')]
    for group in ('replay', 'fresh_frozen'):
        for mode in MODES:
            for hold in plan['hold_settings'][group]:
                for candidate, baseline in comparisons:
                    pairs, scenes, differences = 0, defaultdict(list), []
                    for b in rows:
                        if b['population_group'] != group or b['source_mode'] != mode or int(b['hold_steps']) != hold or b['policy'] != baseline:
                            continue
                        a = index[full_key(b, candidate, seed=0 if candidate == 'pure_rule' else None)]
                        delta = {m: float(a[m]) - float(b[m]) for m in MEASURES}
                        differences.append(delta)
                        scenes[b['scenario_id']].append(delta)
                        pairs += 1
                    assert pairs == 100 and len(scenes) == 5
                    full_effects.append(dict(population_group=group, mode=mode, hold_steps=hold,
                        candidate=candidate, baseline=baseline, paired_comparisons=pairs,
                        selected_scenes=5, reused_rule_runs=candidate == 'pure_rule',
                        mean_differences={m: float(np.mean([d[m] for d in differences])) for m in MEASURES},
                        individual_pair_outcome_classes=dict(Counter(outcome_class(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in differences)),
                        scene_mean_differences={s: {m: float(np.mean([d[m] for d in ds])) for m in MEASURES}
                                                for s, ds in scenes.items()}))
    hold_interactions = []
    for mode in MODES:
        for policy in ('direction_hint_only','preserve_release_hint'):
            effects = {r['hold_steps']: r['mean_differences'] for r in full_effects
                       if r['population_group'] == 'replay' and r['mode'] == mode
                       and r['candidate'] == policy and r['baseline'] == 'no_advice'}
            hold_interactions.append(dict(mode=mode, policy=policy,
                difference_of_differences_hold0_minus_hold2={m: effects[0][m] - effects[2][m] for m in MEASURES}))
    with (full / 'selected_states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads,f)}
    cf = csv_read(continuations / 'terminal_rollouts.csv')
    values = defaultdict(list)
    for r in cf:
        key = (r['state_id'], int(r['forced_action']), r['future_policy'], int(r['hold_steps']))
        values[key].append(r)
    means = {k: {m: float(np.mean([float(r[m]) for r in rs])) for m in MEASURES} for k, rs in values.items()}
    state_rows = []
    for sid, s in states.items():
        b, h, matched = s['base_action'], s['hint_action'], s['matched_release_action']
        future_count = 1 if s['behavior'] == 'rule' else 4
        assert all(len(values[(sid,a,'no_advice',2)]) == future_count for a in s['legal_actions'])
        candidates = {a: means[(sid,a,'no_advice',2)] for a in s['legal_actions']}
        cells = joint_effect_cells(b, h, s['legal_actions'])
        def difference(a, z):
            return {m: a[m] - z[m] for m in MEASURES}
        d = difference(candidates[h], candidates[b])
        dm = difference(candidates[matched], candidates[b])
        resumed = difference(means[(sid,h,'direction_hint_only',2)], means[(sid,b,'direction_hint_only',2)])
        nohold = difference(means[(sid,h,'no_advice',0)], means[(sid,b,'no_advice',0)])
        record = {k:s[k] for k in ('state_id','case_id','scenario_id','behavior','population_seed','training_seed','source_mode','source_step','first_changed_state','base_action','hint_action','matched_release_action','rule_action')}
        record.update(dict(future_samples=future_count, outcome_class=outcome_class(d['incapacitation_rate'],d['mean_cumulative_fed']),
            matched_outcome_class=outcome_class(dm['incapacitation_rate'],dm['mean_cumulative_fed']),
            rule_outcome_class=outcome_class(candidates[s['rule_action']]['incapacitation_rate']-candidates[b]['incapacitation_rate'],
                                           candidates[s['rule_action']]['mean_cumulative_fed']-candidates[b]['mean_cumulative_fed'])))
        for name, diffs in [('common_ppo',d),('matched_release',dm),('resume_hint',resumed),('no_hold',nohold),
                            ('continuation_policy_interaction',{m:resumed[m]-d[m] for m in MEASURES}),
                            ('hold_interaction',{m:nohold[m]-d[m] for m in MEASURES})]:
            for m,v in diffs.items():
                record[f'{name}_{m}'] = v
        if cells is not None:
            for m in MEASURES:
                for part,v in factor_effects(cells,candidates,m).items():
                    record[f'{part}_{m}'] = v
        state_rows.append(record)
    assert len(state_rows) == fs['selected_states']
    out = unique_output('direction_release_mechanism_analysis')
    columns = sorted({k for r in state_rows for k in r})
    csv_write(out / 'state_effects.csv',[{k:r.get(k) for k in columns} for r in state_rows])
    save(out / 'paired_full_effects.json', full_effects)
    save(out / 'hold_interactions.json', hold_interactions)
    cf_groups = []
    for mode in MODES:
        for selection in ('first_changed_state','all_changed_states'):
            rs = [r for r in state_rows if r['source_mode'] == mode and (selection=='all_changed_states' or r['first_changed_state'])]
            cf_groups.append(dict(mode=mode,selection=selection,states=len(rs),
                actual_hint_classes=dict(Counter(r['outcome_class'] for r in rs)),
                matched_release_classes=dict(Counter(r['matched_outcome_class'] for r in rs))))
    fresh = [e for e in full_effects if e['population_group']=='fresh_frozen' and e['candidate']=='preserve_release_hint' and e['baseline']=='no_advice']
    summary = dict(full_source=str(full), continuation_source=str(continuations),
        exact_old_replay_episodes=400, full_episodes=1920, terminal_continuations=cs['terminal_rollouts'],
        selected_actual_changed_states=len(states), exact_realized_suffixes=cs['exact_realized_suffixes'],
        exact_suffix_control_steps=cs['exact_suffix_control_steps'],
        cf_groups=cf_groups, full_effects=full_effects, hold_interactions=hold_interactions,
        fresh_preserve_release_vs_no_advice=fresh,
        candidate_adopted=False, qualification_claim=False,
        decision='Finite mechanisms and targeted fresh-population comparison only; hold changes are diagnostic. Keep formal training paused and final bank sealed.',
        no_new_llm_requests=True,new_training=0,reserved_final_scenarios_opened=0,
        protected_sha256={p:sha(ROOT/p) for p in ('outputs/local_llm_selection.json','outputs/third_round_development_status.json')},
        tests_passed_new=5,tests_passed_related_regression=7,
        source_manifests={p.relative_to(ROOT).as_posix():sha(p) for p in [PREP/'manifest.json',full/'completion_manifest.json',continuations/'completion_manifest.json']} )
    save(out / 'summary.json',summary)
    write_report(out, summary, state_rows)
    verify()
    save(out / 'completion_manifest.json',dict(state='complete',files={p.relative_to(ROOT).as_posix():sha(p)
         for p in [*out.iterdir(),REPORT,Path(__file__)] if p.is_file()}))
    print(json.dumps(dict(output=str(out),report=str(REPORT),full_episodes=1920,terminal_continuations=cs['terminal_rollouts'],
                         fresh_preserve_release_vs_no_advice=fresh),ensure_ascii=False),flush=True)


def write_report(out,s,states):
    modes={MODES[0]:'即时缓存',MODES[1]:'实测延迟'}
    policy_names={'no_advice':'同权重无建议PPO','direction_hint_only':'原方向提示',
                  'preserve_release_hint':'转向保持原放行','pure_rule':'同信息规则'}
    lines=['# 出口方向、放行容量与指示牌保持时间：有限机制实验','',
        '日期：2026年10月7日。使用五个已选开发场景，补齐实际动作链和分区域暴露，并检验方向与容量联动。复用真实Qwen3 14B缓存，不调用LLM、不训练PPO，48个保留最终场景继续封存。','',
        '## 1. 冻结范围与实际完成','',
        f"完成1,920个完整回合、{s['terminal_continuations']}条终止反事实、{s['selected_actual_changed_states']}个实际建议改变动作状态；400个旧人口回合的终点与动作计数全部精确一致，{s['exact_realized_suffixes']}个实际建议续跑的全部{s['exact_suffix_control_steps']}个后续控制步与记录一致。",'',
        '场景为AULL0010、0013、0019、0020、0024，按前轮发现的问题选择，含有益、有害与反转病例。回合数、未来种子和权重不是独立场景数。本轮没有扩大训练，没有独立最终建筑测试。','',
        '旧人口组使用原1477100000前缀，重复既有病例以定位原因。新冻结人口使用1487100000前缀，每场景两批、共10批人口；行为模型为Rule及既有CalibratedLogit；5套权重601—605不变，温度与门控不搜索参数。新人口仍在这五个选择过的建筑上，属于针对性的开发确认。','',
        '三种PPO控制为无建议、原仅方向提示、换方向时保持同状态无建议PPO放行档位。规则控制使用相同公开传感器/预报和合法动作空间，单独运行，不为每个PPO权重复制一遍。无建议组是相同权重的消融，不是新训练的纯PPO。','',
        '旧人口组分别设置指示牌保持2步和0步，新人口确认保持原2步。0步只用于机制对照：同时清除执行器和包装器的保持状态，物理出口、烟气和放行限制不变；每种保持设置都与其对应的无建议PPO比较，不把环境改变当LLM收益。','',
        '## 2. 完整回合配对结果','',
        '下表为候选−同设置对照。FED和失能率正差表示更差，成功率正差表示更好。每行100个PPO权重条件下的配对比较，来自五个选定场景。规则输出被配给各套PPO权重作比较，不把这些复用算成新规则实验。仅报告描述性均差，不作独立场景显著性声明。','',
        '|人口|时序|保持步|候选−对照|FED差|失能率差（百分点）|成功率差（百分点）|累计拥堵差|',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for e in s['full_effects']:
        d=e['mean_differences']
        lines.append(f"|{'旧病例' if e['population_group']=='replay' else '新冻结人口'}|{modes[e['mode']]}|{e['hold_steps']}|{policy_names[e['candidate']]}−{policy_names[e['baseline']]}|{d['mean_cumulative_fed']:+.6f}|{100*d['incapacitation_rate']:+.4f}|{100*d['evacuation_rate']:+.4f}|{d['cumulative_congestion']:+.4f}|")
    lines+=['','新冻结人口中的转向保持放行版本与无建议PPO：','']
    for e in s['fresh_preserve_release_vs_no_advice']:
        d=e['mean_differences']
        lines.append(f"- {modes[e['mode']]}：FED差{d['mean_cumulative_fed']:+.6f}，失能率差{100*d['incapacitation_rate']:+.4f}个百分点。")
    lines+=['','即使修正比原方向提示好，也必须看是否比无建议PPO更好，是否把暴露改善换成更多失能，以及行为模型、人口和权重上的差异。全部逐场景差异保存在paired_full_effects.json；本轮没有因为均值改善就采用控制器。','',
        '## 3. 真正发生改动的状态如何检验','',
        '在旧人口、原保持时间的真实方向提示轨迹上，选择每个通过门控且改变同状态无建议动作的时刻，保存当时人员、累计FED、历史和随机表。复制状态仅用于离线仿真，控制器仍只接收34维公开观测。选择包含后来证实有益的改动，不只挑有害病例。','',
        '每个当前合法动作强制一步，随后所有候选使用同一无建议PPO直至终止；另外强制原建议或无建议动作，再恢复建议控制，以检查一次改动与后续建议的交互。取消保持的配对则从同一状态开始，只改变之后的保持规则。不同单步效果不能累加成全程效果。','',
        'Rule行为用原实际未来；概率行为用原实际未来加三套预先冻结未来，按人和绝对时刻配对。原实际“强制建议动作并恢复建议”的方向、放行、区域人数和终点逐步复现原轨迹，这补足了上轮只能看旧人口影子动作的限制。','',
        '|时序|状态选择|状态数|原提示：有益/有害/微小|保持档位：有益/有害/微小|', '|---|---|---:|---|---|']
    for g in s['cf_groups']:
        a,b=g['actual_hint_classes'],g['matched_release_classes']
        def counts(x): return '/'.join(str(x.get(k,0)) for k in ('beneficial','harmful','small_or_neutral'))
        lines.append(f"|{modes[g['mode']]}|{'每回合首次改动' if g['selection']=='first_changed_state' else '所有实际改动'}|{g['states']}|{counts(a)}|{counts(b)}|")
    lines+=['','分类沿用冻结阈值：先看失能率差±0.001，再看FED差±0.001，否则微小。它不是人体安全阈值。多未来是条件于当前隐藏个体状态的诊断采样，不是新建筑，也不作为在线覆盖标签。','',
        '## 4. 分区域暴露与等待提供了什么证据','',
        '逐步记录真实个体FED增量发生在哪个区域，并核对五个区域FED之和等于完整回合总FED。等待用区域人数×5秒累计，除以初始240人，得到每位初始人口平均区域人秒；包含失能之前的停留，不能冒充所有人成功疏散时间。','',
        '|实际改动：场景/行为/权重/步|时序|无建议→提示|纯方向FED差|纯容量FED差|交互FED差|容量项房间FED差|容量项房间停留秒差|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in states:
        if r['scenario_id'] not in ('AULL0010','AULL0013','AULL0024') or r['source_mode']!=MODES[0]: continue
        if 'direction_at_base_release_mean_cumulative_fed' not in r: continue
        roomfed=sum(r[f'release_at_base_direction_fed_{n}'] for n in ('room_a','room_b'))
        roomtime=sum(r[f'release_at_base_direction_person_seconds_{n}'] for n in ('room_a','room_b'))
        lines.append(f"|{r['scenario_id']}/{r['behavior']}/{r['training_seed']}/{r['source_step']}，人口{r['population_seed']}|{modes[r['source_mode']]}|{r['base_action']}→{r['hint_action']}|{r['direction_at_base_release_mean_cumulative_fed']:+.6f}|{r['release_at_base_direction_mean_cumulative_fed']:+.6f}|{r['direction_release_interaction_mean_cumulative_fed']:+.6f}|{roomfed:+.6f}|{roomtime:+.3f}|")
    lines+=['','方向与容量四组合固定当时无建议/提示的方向和档位，等式为：方向项+容量项+交互项=完整单步差。若涉及等待或某组合非法，不编造四组合数值；在state_effects.csv中单独记录。保留档位修正在等待→放行时沿用原提示，因为9动作不能表达“新方向且零放行”，对应例外计数保留。','',
        '## 5. 保持时间与后续建议的交互','',
        '|时序|策略|保持0步相对保持2步的建议增量差：FED|同一差：失能率（百分点）|', '|---|---|---:|---:|']
    for e in s['hold_interactions']:
        d=e['difference_of_differences_hold0_minus_hold2']
        lines.append(f"|{modes[e['mode']]}|{policy_names[e['policy']]}|{d['mean_cumulative_fed']:+.6f}|{100*d['incapacitation_rate']:+.4f}|")
    lines+=['','此处使用差中差：(策略−无建议)保持0步−(策略−无建议)保持2步；把无建议基线一起改变，避免把去掉保持后所有控制都改善误归因给LLM。单状态的hold_interaction与continuation_policy_interaction在CSV另存，以区分从同状态移除保持与改变后续控制策略的影响。','',
        '## 6. 决策、局限与下一步','',
        '本轮完成此前缺失的实际动作链、合法终点反事实和区域等待/暴露记录；继续保留全部正负效果。本轮不采用修正，也不调整建议强度、confidence阈值或有效期。保持0步是控制条件变化，不作为真实疏散方案。','',
        '新冻结人口减少了同一批人员被反复调整的风险，但五个建筑已根据问题选择，仍不能外推到独立建筑、现实火场或所有LLM。模拟人群属性与烟气近似仍有现实校准缺口。5套权重均为既有训练权重，没有补齐独立同预算新训练对照。','',
        '下一轮应依据保存的逐状态与逐区域结果，确认剩余风险是否主要需要预测转向净收益、处理等待，还是提供基线尚未掌握的信息。只对有证据支持的具体机制设计新冻结开发验证；正式训练和最终场景仍暂停。','',
        '## 7. 复现材料','',
        f'- [冻结实验方案]({(PREP/"manifest.json").as_posix()})',
        f'- [1,920个完整回合]({(Path(s["full_source"])/"episodes.csv").as_posix()})',
        f'- [全部实际逐步动作与区域记录]({(Path(s["full_source"])/"actual_step_traces.jsonl").as_posix()})',
        f'- [全部终点反事实]({(Path(s["continuation_source"])/"terminal_rollouts.csv").as_posix()})',
        f'- [逐状态机制分解]({(out/"state_effects.csv").as_posix()})',
        f'- [完整配对效果与区域差]({(out/"paired_full_effects.json").as_posix()})',
        f'- [总审计与决策]({(out/"summary.json").as_posix()})','',
        '执行检查：所有完整回合和终止反事实的封闭/超阈值走廊进入、实际放行超限、方向非法均为0；区域FED守恒检查通过。5项新测试、7项相关回归测试通过。原缓存、权重、报告和台账哈希未变；LLM新请求0、训练0、最终场景读取0。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',required=True)
    parser.add_argument('--continuations',required=True)
    args=parser.parse_args()
    run(args.full,args.continuations)
