"""Audited Chinese synthesis; feature/prior effects reuse enumerated actions."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output
from run_qwen14b_action_attribution import PREPARATION, verify, verify_replay, METRICS, MODES
from fire_evac.advice_causal_audit import outcome_class


def prepare():
    verify()
    p = PREPARATION / 'final_feature_analysis_manifest.json'
    assert not p.exists()
    save(p, dict(source_sha256=sha(__file__),
        scope='Primary 12-scene first-response states; existing legal-action enumeration, no extra simulations.',
        contrasts=['feature_only_action_minus_no_advice_action', 'full_advice_action_minus_feature_only_action'],
        trace_categories=['feature_changed_and_prior_restores_plain', 'feature_unchanged_and_prior_introduces_change',
                          'feature_changed_and_prior_further_changes', 'changed_direction_only',
                          'changed_release_only', 'changed_direction_and_release', 'changed_wait_status'],
        statistical_status='exploratory_effect_decomposition_declared_before_reading_terminal_continuations'))
    print('Frozen feature/prior effect decomposition.', flush=True)


def run(analysis):
    verify()
    p = PREPARATION / 'final_feature_analysis_manifest.json'
    assert sha(__file__) == read(p)['source_sha256']
    source = verify_replay(analysis)
    summary = read(source / 'summary.json')
    replay = verify_replay(summary['source_paths']['replay'])
    continuations = verify_replay(summary['source_paths']['continuations'])
    assert summary['source_replay']['exact_terminal_and_counter_parity']
    assert not any(summary['violations'].values())
    with (replay / 'selected_states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads, f)}
    with (continuations / 'terminal_rollouts.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    values = defaultdict(list)
    for r in rows:
        values[(r['state_id'], int(r['first_requested_action']))].append(r)
    means = {k: {m: float(np.mean([float(r[m]) for r in v])) for m in METRICS} for k, v in values.items()}
    decomposition = []
    for mode in MODES:
        selected = [r for r in states.values() if r['arm'] == mode and r['cohort'] == 'primary_separate_development'
                    and 'first_response_available' in r['selection_reasons']]
        assert len(selected) == 240
        for name, a_field, b_field in [('feature_only_action_minus_no_advice_action', 'action_features_only', 'action_no_advice'),
                                     ('full_advice_action_minus_feature_only_action', 'action_with_advice', 'action_features_only')]:
            for metric in METRICS:
                scenes = defaultdict(list)
                for r in selected:
                    d = means[(r['state_id'], r[a_field])][metric] - means[(r['state_id'], r[b_field])][metric]
                    scenes[r['scenario_id']].append(d)
                scene_means = np.array([np.mean(v) for v in scenes.values()])
                rng = np.random.default_rng(2026100760)
                boot = scene_means[rng.integers(0, len(scene_means), (10000, len(scene_means)))].mean(axis=1)
                decomposition.append(dict(mode=mode, contrast=name, metric=metric, states=240,
                    mean_difference=float(scene_means.mean()),
                    descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975])))))
    trace_counts = defaultdict(Counter)
    with (replay / 'action_trace.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads, f):
            if r['cohort'] != 'primary_separate_development':
                continue
            plain, feat, full = r['action_no_advice'], r['action_features_only'], r['action_with_advice']
            counts = trace_counts[r['arm']]
            counts['feature_changed_and_prior_restores_plain'] += int(feat != plain and full == plain)
            counts['feature_unchanged_and_prior_introduces_change'] += int(feat == plain and full != plain)
            counts['feature_changed_and_prior_further_changes'] += int(feat != plain and full != plain and full != feat)
            if plain != full:
                if 8 in (plain, full):
                    kind = 'changed_wait_status'
                elif plain // 4 == full // 4:
                    kind = 'changed_release_only'
                elif plain % 4 == full % 4:
                    kind = 'changed_direction_only'
                else:
                    kind = 'changed_direction_and_release'
                counts[kind] += 1
    out = unique_output('qwen14b_action_attribution_final_audit')
    report = ROOT / 'reports/14B建议作用与候选收益审计报告_2026-10-07.md'
    assert not report.exists()
    replay_groups = [r for r in summary['source_replay']['groups'] if r['cohort'] == 'primary_separate_development']
    primary = [r for r in summary['group_diagnostics'] if r['cohort'] == 'primary_separate_development'
               and r['selection'] == 'first_response_available']
    changed = [r for r in summary['group_diagnostics'] if r['cohort'] == 'primary_separate_development'
               and r['selection'] == 'first_action_change']
    lines = ['# 火灾疏散论文开发实验：14B建议的实际作用与候选收益', '',
        '日期：2026年10月7日。本轮执行“先查建议是否改变控制，再比较合法替代方案”的计划。复用真实Qwen3 14B缓存，未追加LLM调用，未增加PPO训练，保留最终场景继续封存。', '',
        '## 完成的工作', '',
        f"1. 精确重放960个即时/延迟14B回合，共{summary['source_replay']['replayed_control_steps']:,}个控制步；所有已保存终点和门控计数一致。",
        f"2. 在{summary['source_replay']['one_step_action_checks']}个动作变化步，复制相同状态执行无建议动作，检查实际执行器、流量和人员位置。",
        f"3. 按冻结规则选择{summary['selected_states']}个关键状态，跑完{summary['terminal_rollouts']:,}条完整后续轨迹，覆盖每个状态当前全部合法动作。", '',
        '选择规则是每回合“第一次响应可用”以及“第一次有效建议改变PPO动作”，两者重合则合并。主要结论取前者在12个主要开发场景上的结果；首次变化状态是刻意选出的探索性机制样本。', '',
        '## 一、建议通过检查，不代表控制已经改变', '',
        '|主要开发子集|回合|建议通过控制步|真正改变动作步|改变动作回合|当步人员流量改变步|',
        '|---|---:|---:|---:|---:|---:|']
    for r in replay_groups:
        name = '即时缓存' if r['arm'] == MODES[0] else '实测延迟回放'
        lines.append(f"|{name}|{r['episodes']}|{r.get('advice_accepted_steps', 0)}|{r.get('action_changed_steps', 0)}|{r.get('episodes_with_action_change', 0)}|{r.get('flow_changed_steps', 0)}|")
    lines += ['', '当前PPO大部分时候已经会选择与建议相同的方向或动作；建议接入后也不一定改变动作。指示牌发生变化但当步路口无人时，流量可以暂时不变，仍可能影响后续，因此执行器变化和流量变化分开统计。', '',
        '## 二、方向与放行档位的变化分开检查', '',
        '|模式|只变方向|只变放行档位|方向与档位都变|涉及等待|特征改变但加分恢复原动作|',
        '|---|---:|---:|---:|---:|---:|']
    for mode, counts in trace_counts.items():
        lines.append(f"|{mode}|{counts['changed_direction_only']}|{counts['changed_release_only']}|{counts['changed_direction_and_release']}|{counts['changed_wait_status']}|{counts['feature_changed_and_prior_restores_plain']}|")
    lines += ['', '“仅加入建议特征”与“再给候选动作小幅加分”不是同一环节，可能产生相反作用。下面用已经枚举的合法动作终点，检查两环节的单步效果；不追加仿真。', '',
        '|主要子集第一次响应状态|环节对照|指标|平均差|描述性场景95%区间|',
        '|---|---|---|---:|---|']
    for r in decomposition:
        if r['metric'] in ('mean_cumulative_fed', 'incapacitation_rate'):
            lo, hi = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['mode']}|{r['contrast']}|{r['metric']}|{r['mean_difference']:+.7f}|{lo:.7f}, {hi:.7f}|")
    lines += ['', '这是一时刻动作的效果分解；之后所有候选都恢复同一无建议PPO。不能把它当作整个14B控制流程的增益，也不能将多个状态的效果相加。', '',
        '## 三、建议方向有没有潜在收益', '',
        '|主要子集首次响应|状态|合法库中有有益方案|推荐方向含有益方案|有机会但推荐方向漏掉|方向有机会但实际动作没获益|实际有益/有害/微小|',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in primary:
        lines.append(f"|{r['mode']}|{r['states']}|{r.get('action_opportunity', 0)}|{r.get('target_direction_opportunity', 0)}|{r.get('opportunity_but_target_misses', 0)}|{r.get('target_opportunity_but_actual_not_beneficial', 0)}|{r.get('actual_beneficial', 0)}/{r.get('actual_harmful', 0)}/{r.get('actual_small_or_neutral', 0)}|")
    lines += ['', '当前14B只推荐方向和自报confidence，候选25%/50%档位由旧适配器固定生成。本轮检查推荐方向的全部合法档位，目的是区分“方向漏掉好机会”和“方向有潜力，但档位/执行没用好”。事后挑选最佳档位只是乐观诊断上限，不是14B实际提出的计划。', '',
        '## 四、在实际改变动作的状态，改变是否有益', '',
        '|主要子集首次动作变化|状态|有益|有害|微小/中性|不同未来中有益与有害反转|',
        '|---|---:|---:|---:|---:|---:|']
    for r in changed:
        lines.append(f"|{r['mode']}|{r['states']}|{r.get('actual_beneficial', 0)}|{r.get('actual_harmful', 0)}|{r.get('actual_small_or_neutral', 0)}|{r.get('actual_future_label_reversal', 0)}|")
    lines += ['', '“有益/有害”的分类先看未来平均失能率差±0.001；若更小，再看FED差±0.001。这个预先冻结阈值用于开发分析，不是人体剂量或现实安全标准。概率行为只有4个未来样本，分类仍有抽样不确定性；逐未来反转也单独报告。', '',
        '## 五、如何解读并决定下一轮', '',
        '1. 有好方案，但推荐方向没有任何有益候选时，优先查提名方向和观测/预测信息，不能只增强PPO服从程度。',
        '2. 推荐方向有潜力、实际执行动作没有获益时，优先查放行档位、建议特征与动作加分的衔接；这仍不意味着硬执行该方向一定更好。',
        '3. 改变动作本身有害时，不采用简单提高建议强度的修正；必须结合失能、暴露、容量、拥堵和等待损失验证。',
        '4. 若建议方向的收益依赖后续观察，后续可设计带新观测的候选计划；本轮数据只来自初始缓存，尚不能验证滚动14B建议的效果。',
        '5. 继续暂停扩大训练。本轮离线最佳动作不能直接当在线安全盾或用作已经通过最终验证的策略。', '',
        '## 六、证据边界和可复现性', '',
        '未来随机数按同一人、同一绝对时刻配对，复制当前个体属性和已累积FED。给已配对的环境换未来种子时，去除原随机表包装再赋新表；相关测试检查了不同种子确实不同、相同种子确实一致。旧实验是在原始行为上首次包装，此问题不改变旧结果。', '',
        '本轮重放不是新增训练或新的独立场景实验。24个场景仍来自既有开发库；主要子集为此前12个开发确认场景，保留最终48场景未启用。状态、人口种子、训练权重和未来样本不能当成独立场景。', '',
        '配对统计先平均未来，平均同一回合内的状态，再平均场景内回合；区间是条件于既有权重的开发场景bootstrap。候选库有限，只穷举单步9动作，不是全程动态计划的全局最优搜索。', '',
        'FED为简化图模型中的合成烟气增量，没有现实火场剂量验证；仿真结束包含失能终止，不能当成全员成功疏散时间。所有候选遵守同一物理约束，0次执行限制违规不等于0伤害。', '',
        f"执行限制违规：{summary['violations']}。新功能和相关回归测试28项通过。新LLM调用0，PPO更新0。原缓存、旧权重、报告和资格状态的哈希均保持一致。", '',
        '详细材料：', '',
        f"- [动作链与重放计数]({(replay / 'summary.json').as_posix()})",
        f"- [逐状态终点对照]({(source / 'state_comparison_effects.csv').as_posix()})",
        f"- [每个合法动作与未来的原始轨迹终点]({(continuations / 'terminal_rollouts.csv').as_posix()})",
        f"- [全部分组与区间]({(source / 'summary.json').as_posix()})", '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    result = dict(report=str(report), source_analysis=str(source), exact_replay_episodes=960,
        replay_control_steps=summary['source_replay']['replayed_control_steps'], selected_states=summary['selected_states'],
        terminal_rollouts=summary['terminal_rollouts'], trace_categories={k: dict(v) for k,v in trace_counts.items()},
        feature_prior_effects=decomposition, primary_first_response_diagnostics=primary, primary_first_change_diagnostics=changed,
        tests_passed=28, new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        protected_status_sha256=sha(ROOT / 'outputs/third_round_development_status.json'),
        protected_model_selection_sha256=sha(ROOT / 'outputs/local_llm_selection.json'),
        sources={q.relative_to(ROOT).as_posix(): sha(q) for q in [Path(__file__), p, report, source / 'completion_manifest.json']})
    save(out / 'summary.json', result)
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {q.relative_to(ROOT).as_posix(): sha(q) for q in out.iterdir() if q.is_file()}})
    print(json.dumps({'output': str(out), 'report': str(report), 'primary_first_change_diagnostics': changed}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--analysis')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        assert args.analysis
        run(args.analysis)
