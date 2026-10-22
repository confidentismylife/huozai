"""Descriptive attribution and offline candidate-quality diagnostics."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from run_qwen14b_action_attribution import verify, verify_replay, MODES, METRICS, PREPARATION
from fire_evac.advice_causal_audit import outcome_class, ranked_outcomes


def prepare():
    verify()
    p = PREPARATION / 'analysis_manifest.json'
    assert not p.exists()
    save(p, dict(sources={Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)},
        planned_comparators=['actual_advised_action', 'rule_action', 'matched_target_action',
                            'offline_best_target_direction', 'offline_best_legal_action'],
        groups=['first_response_available', 'first_action_change', 'all_selected'],
        cohorts=['primary_separate_development', 'exploratory_mechanism_development', 'all_development'],
        modes=MODES, outcome_thresholds={'incapacitation_rate': .001, 'fed': .001},
        offline_best='Expected incapacitation first, FED second, congestion third. Optimistic diagnostic only; no deployable gain estimate.',
        opportunities='Report any legal beneficial action, beneficial action within LLM target, target misses, adapter misses, actual harmful/beneficial and per-future reversals.',
        aggregation='Future means per state, then state means per episode, then episode means per scenario. Descriptive scenario bootstrap.',
        primary_claim_scope='First-response states on the prior primary 12-scene development cohort. Action-change selections are outcome-independent but action-selected exploratory strata.'))
    print('Frozen attribution analysis before reading continuation outcomes.', flush=True)


def group_selected(states, cohort, mode, reason):
    return [r for r in states.values() if r['arm'] == mode and
            (cohort == 'all_development' or r['cohort'] == cohort) and
            (reason == 'all_selected' or reason in r['selection_reasons'])]


def analysis(simulation, replay):
    verify()
    p = PREPARATION / 'analysis_manifest.json'
    for name, h in read(p)['sources'].items():
        assert sha(ROOT / name) == h
    sim = verify_replay(simulation)
    replay = verify_replay(replay)
    with (replay / 'selected_states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads, f)}
    with (sim / 'terminal_rollouts.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected = read(replay / 'summary.json')['planned_terminal_rollouts']
    assert len(rows) == expected
    groups = defaultdict(list)
    future_rows = defaultdict(dict)
    violations = Counter()
    for r in rows:
        a = int(r['first_requested_action'])
        future = int(r['future_seed'])
        assert a in states[r['state_id']]['legal_actions']
        assert int(r['first_actual_action']) == a
        groups[(r['state_id'], a)].append(r)
        assert a not in future_rows[(r['state_id'], future)]
        future_rows[(r['state_id'], future)][a] = {m: float(r[m]) for m in METRICS}
        for name in ('unavailable_entries', 'release_violations', 'direction_violations'):
            violations[name] += int(r[name])
    means = defaultdict(dict)
    for (sid, a), values in groups.items():
        assert len(values) == (1 if states[sid]['behavior'] == 'rule' else 4)
        means[sid][a] = {m: float(np.mean([float(r[m]) for r in values])) for m in METRICS}
    effects, diagnostics = [], {}
    for sid, source in states.items():
        candidates = means[sid]
        assert sorted(candidates) == source['legal_actions']
        b_action = source['action_no_advice']
        baseline = candidates[b_action]
        labels = {a: outcome_class(candidates[a]['incapacitation_rate'] - baseline['incapacitation_rate'],
                                  candidates[a]['mean_cumulative_fed'] - baseline['mean_cumulative_fed']) for a in candidates}
        target_actions = source['target_direction_actions']
        target_order = ranked_outcomes({a: candidates[a] for a in target_actions}) if target_actions else []
        best = ranked_outcomes(candidates)[0]
        comparisons = dict(actual_advised_action=source['action_with_advice'], rule_action=source['rule_action'],
                           offline_best_legal_action=best)
        if source['matched_target_action'] is not None:
            comparisons['matched_target_action'] = source['matched_target_action']
        if target_order:
            comparisons['offline_best_target_direction'] = target_order[0]
        futures = sorted(f for state, f in future_rows if state == sid)
        actual_future_classes = []
        winning_actions = []
        for future in futures:
            fr = future_rows[(sid, future)]
            winning_actions.append(ranked_outcomes(fr)[0])
            a = source['action_with_advice']
            actual_future_classes.append(outcome_class(fr[a]['incapacitation_rate'] - fr[b_action]['incapacitation_rate'],
                                                       fr[a]['mean_cumulative_fed'] - fr[b_action]['mean_cumulative_fed']))
        opportunity = any(v == 'beneficial' for v in labels.values())
        target_opportunity = any(labels[a] == 'beneficial' for a in target_actions)
        actual_class = labels[source['action_with_advice']]
        diagnostics[sid] = dict(state_id=sid, action_opportunity=opportunity,
            target_direction_executable=bool(target_actions), target_direction_opportunity=target_opportunity,
            opportunity_but_target_misses=opportunity and not target_opportunity,
            target_opportunity_but_actual_not_beneficial=target_opportunity and actual_class != 'beneficial',
            gate_rejected_but_target_opportunity=not source['gate_accepted'] and target_opportunity,
            actual_class=actual_class, best_legal_action=best,
            best_target_action=target_order[0] if target_order else None,
            actual_future_label_reversal='beneficial' in actual_future_classes and 'harmful' in actual_future_classes,
            future_winning_action_varies=len(set(winning_actions)) > 1,
            actual_future_classes=actual_future_classes,
            future_winning_actions=winning_actions)
        for comparison, a in comparisons.items():
            for metric in METRICS:
                effects.append(dict(state_id=sid, scenario_id=source['scenario_id'], cohort=source['cohort'],
                    behavior=source['behavior'], population_seed=source['population_seed'], training_seed=source['training_seed'],
                    source_mode=source['arm'], source_step=source['step'], comparison=comparison, metric=metric,
                    action=a, baseline_action=b_action, outcome_class=labels[a],
                    difference=float(candidates[a][metric] - baseline[metric])))
    report_groups, summaries = [], []
    for cohort in ('primary_separate_development', 'exploratory_mechanism_development', 'all_development'):
        for mode in MODES:
            for reason in ('first_response_available', 'first_action_change', 'all_selected'):
                subset = group_selected(states, cohort, mode, reason)
                ids = {r['state_id'] for r in subset}
                if not ids:
                    continue
                counts = Counter()
                for r in subset:
                    d = diagnostics[r['state_id']]
                    for name in ('action_opportunity', 'target_direction_executable', 'target_direction_opportunity',
                                 'opportunity_but_target_misses', 'target_opportunity_but_actual_not_beneficial',
                                 'gate_rejected_but_target_opportunity', 'actual_future_label_reversal', 'future_winning_action_varies'):
                        counts[name] += int(d[name])
                    counts['actual_' + d['actual_class']] += 1
                    counts['gate_accepted'] += r['gate_accepted']
                    counts['changed_action'] += r['action_changed']
                report_groups.append(dict(cohort=cohort, mode=mode, selection=reason, states=len(subset),
                    scenario_clusters=len({r['scenario_id'] for r in subset}), **dict(counts)))
                for comparison in ('actual_advised_action', 'rule_action', 'matched_target_action',
                                   'offline_best_target_direction', 'offline_best_legal_action'):
                    for metric in METRICS:
                        values = [r for r in effects if r['state_id'] in ids and r['comparison'] == comparison and r['metric'] == metric]
                        if not values:
                            continue
                        episodes = defaultdict(list)
                        for r in values:
                            episodes[(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'])].append(r['difference'])
                        scenes = defaultdict(list)
                        for k, v in episodes.items():
                            scenes[k[0]].append(float(np.mean(v)))
                        scene_means = np.array([np.mean(v) for v in scenes.values()])
                        rng = np.random.default_rng(2026100759)
                        boot = scene_means[rng.integers(0, len(scene_means), (10000, len(scene_means)))].mean(axis=1)
                        summaries.append(dict(cohort=cohort, mode=mode, selection=reason, comparison=comparison, metric=metric,
                            states=len(values), episode_clusters=len(episodes), scenario_clusters=len(scenes),
                            mean_difference=float(scene_means.mean()),
                            descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975])))))
    out = unique_output('qwen14b_action_attribution_analysis')
    csv_write(out / 'state_comparison_effects.csv', effects)
    with (out / 'state_diagnostics.jsonl').open('w', encoding='utf-8') as f:
        for r in diagnostics.values():
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = dict(terminal_rollouts=len(rows), selected_states=len(states),
        source_replay=read(replay / 'summary.json'), group_diagnostics=report_groups, paired_effects=summaries,
        violations=dict(violations), new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        optimistic_offline_choices_not_deployed=True, statistical_status='conditional_descriptive_development_audit',
        no_sum_of_single_step_effects_as_full_episode_gain=True,
        source_paths={'replay': str(replay), 'continuations': str(sim)})
    save(out / 'summary.json', summary)
    lines = ['# 14B建议为何没有带来收益：动作作用与合法替代方案审计', '',
        f"重放960个既有14B回合，选择{len(states)}个关键状态，新增{len(rows):,}条完整后续反事实轨迹。新LLM请求0，新训练0，保留最终场景未打开。", '',
        '## 设计与口径', '',
        '首先重放即时缓存和实测延迟缓存两个模式，逐项核对既有回合终点和门控计数。每个状态分别计算无建议、仅建议特征、建议特征＋有限动作加分三个PPO动作；在动作不同处复制状态执行一步，核查实际指示牌、放行、流量和人员位置。', '',
        '关键状态选择事先冻结：每回合第一次收到响应的时刻，以及第一次有效建议改变动作的时刻，两者重合时只算一个状态。主要结论取12个主要开发场景的第一次响应状态；首次改变动作状态是有意挑选的探索性诊断，不能代表全部控制步。', '',
        '每个关键状态枚举原9动作mask允许的全部动作，强制执行一步，再恢复同一个无建议PPO直至终止。概率行为使用4个新冻结未来，规则行为1个；配对随机数按人和绝对时刻索引。复制已配对的状态时先去除旧行为包装，再赋予新随机表，避免新种子被内层旧表忽略。', '',
        '当前14B只输出出口方向和confidence，25%/50%候选来自固定适配器。本审计也检查同一推荐方向其他合法放行档位；事后挑出的最好档位是诊断上限，不是LLM实际输出或可部署策略。', '',
        '收益分类先看平均失能：差≤−0.001为有益，差≥＋0.001为有害；其余再看FED，差≤−0.001为有益、≥＋0.001为有害。阈值只用于开发分类，不是现实人体剂量或安全阈值。所有连续差值与各未来标签均单独保存。', '',
        '## 建议通过以后，是否改变动作', '',
        '|主要开发子集模式|回合|有效建议控制步|建议改变动作步|其中仅特征改变动作步|候选加分改变动作步|改变动作的回合|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['source_replay']['groups']:
        if r['cohort'] == 'primary_separate_development':
            lines.append(f"|{r['arm']}|{r['episodes']}|{r.get('advice_accepted_steps', 0)}|{r.get('action_changed_steps', 0)}|{r.get('feature_changes_action_steps', 0)}|{r.get('prior_changes_action_steps', 0)}|{r.get('episodes_with_action_change', 0)}|")
    lines += ['', '特征和动作加分是两个逐状态对照，作用可能抵消，不能将改变步数直接相加。请求动作不同、执行器状态不同、当步人员流量不同也分别记录；改变指示牌时，当步路口还无人也可能影响后续。', '',
        '## 第一次响应状态的候选机会', '',
        '|主要子集模式|状态|建议改变动作|库中有有益动作|推荐方向有有益动作|有机会但推荐方向漏掉|方向有机会但实际动作没获益|实际有益/有害/微小|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in report_groups:
        if r['cohort'] == 'primary_separate_development' and r['selection'] == 'first_response_available':
            lines.append(f"|{r['mode']}|{r['states']}|{r.get('changed_action', 0)}|{r.get('action_opportunity', 0)}|{r.get('target_direction_opportunity', 0)}|{r.get('opportunity_but_target_misses', 0)}|{r.get('target_opportunity_but_actual_not_beneficial', 0)}|{r.get('actual_beneficial', 0)}/{r.get('actual_harmful', 0)}/{r.get('actual_small_or_neutral', 0)}|")
    lines += ['', '“推荐方向有有益动作”允许事后选择该方向的放行档位，因此只能说明这个方向存在潜力；不能把未实现的收益全部归因于PPO没有服从LLM。收益分类来自有限未来样本，不能当作真实标签确定无误。', '',
        '## 第一次响应状态的终点差值', '',
        '|主要子集模式|对照相对同状态无建议PPO|指标|均差|描述性场景95%区间|覆盖状态|',
        '|---|---|---|---:|---|---:|']
    for r in summaries:
        if r['cohort'] == 'primary_separate_development' and r['selection'] == 'first_response_available' and r['metric'] in ('mean_cumulative_fed', 'incapacitation_rate'):
            lo, hi = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['mode']}|{r['comparison']}|{r['metric']}|{r['mean_difference']:+.7f}|{lo:.7f}, {hi:.7f}|{r['states']}|")
    lines += ['', '先平均未来，再平均回合中的状态，再平均同一场景的回合。区间为复用开发场景的描述性bootstrap，条件于五套权重；没有把控制状态或未来种子当成独立场景。', '',
        '本轮只改变当前一步，之后统一恢复无建议PPO。它与原始全程14B控制的终点效果是不同问题，不能把各状态的单步收益相加成全程收益。不同对照的可执行状态子集可能不同，均值不能直接作为公平排名。', '',
        '真实未来仅用于离线仿真与事后审计，任何“最好动作”都未输入实时控制。操作限制符合不等于低剂量或现实安全；FED仍是简化图环境中的合成增量。', '',
        '原始动作链、逐状态差值、每个未来的候选终点和分组结果均保存，原缓存、旧报告和权重未改写。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()},
         'source_sha256': {q.relative_to(ROOT).as_posix(): sha(q) for q in [Path(__file__), p, sim / 'completion_manifest.json', replay / 'completion_manifest.json']}})
    print(json.dumps({'output': str(out), 'primary_first_response_diagnostics': [r for r in report_groups if r['cohort'] == 'primary_separate_development' and r['selection'] == 'first_response_available']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'analyze'])
    parser.add_argument('--simulation')
    parser.add_argument('--replay')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        assert args.simulation and args.replay
        analysis(args.simulation, args.replay)
