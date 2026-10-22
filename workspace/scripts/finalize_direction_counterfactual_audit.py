"""Predeclared matched-release contrasts and independent source audit.

Write a new analysis directory; never rewrite the simulation result files.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from run_direction_counterfactual_audit import verify, verify_audit, FIRST, PREPARATION, METRICS
from fire_evac.counterfactual_direction_audit import episode_key, IMMEDIATE_ARMS


def contrast_summary(rows, name):
    result = []
    for metric in METRICS:
        selected = [r for r in rows if r['metric'] == metric]
        if not selected:
            continue
        episodes = defaultdict(list)
        for r in selected:
            episodes[episode_key(r)].append(r['difference'])
        scenes = defaultdict(list)
        for key, values in episodes.items():
            scenes[key[0]].append(float(np.mean(values)))
        scene_means = np.array([np.mean(v) for v in scenes.values()])
        rng = np.random.default_rng(2026100730)
        boot = scene_means[rng.integers(0, len(scene_means), (10000, len(scene_means)))].mean(axis=1)
        deltas = np.array([r['difference'] for r in selected])
        beneficial = deltas if metric == 'evacuation_rate' else -deltas
        result.append(dict(contrast=name, metric=metric, states=len(selected),
                           episode_clusters=len(episodes), scenario_clusters=len(scenes),
                           equal_scene_episode_state_mean_difference=float(scene_means.mean()),
                           descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975]))),
                           improved_states=int(np.sum(beneficial > 1e-12)),
                           worsened_states=int(np.sum(beneficial < -1e-12)),
                           unchanged_states=int(np.sum(abs(beneficial) <= 1e-12))))
    return result


def prepare():
    verify()
    destination = PREPARATION / 'matched_contrast_analysis_manifest.json'
    assert not destination.exists(), 'No overwrite of frozen secondary analysis.'
    save(destination, dict(sources={Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)},
         contrasts=['lower_direction_same_release_all', 'lower_direction_same_release_rule',
                    'lower_direction_same_release_calibrated_logit',
                    'lower_direction_same_release_lower_capacity_ge_higher',
                    'lower_direction_same_release_lower_capacity_lt_higher',
                    'wait_until_lower_hold_limited', 'wait_one_hold_limited'],
         purpose='Isolate direction from release-volume changes; inspect capacity interaction and hold-limited waiting.',
         caveat='Exploratory development mechanism analyses. All state/future/episode/scene averaging is declared before reading terminal results.',
         endpoint_check='Rule behavior original continuation must exactly equal historical terminal results at every target state.'))
    print('Matched direction-only contrasts frozen.', flush=True)


def run(simulation, feasibility):
    verify()
    freeze = read(PREPARATION / 'matched_contrast_analysis_manifest.json')
    for p, h in freeze['sources'].items():
        assert sha(ROOT / p) == h, p
    sim, audit = verify_audit(simulation), verify_audit(feasibility)
    with (audit / 'states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads, f)}
    with (sim / 'terminal_rollouts.csv').open(encoding='utf-8-sig') as f:
        rollouts = list(csv.DictReader(f))
    with (sim / 'state_mean_effects.csv').open(encoding='utf-8-sig') as f:
        effects = list(csv.DictReader(f))
    for r in effects:
        r['difference'] = float(r['difference'])
        r['population_seed'], r['training_seed'] = int(r['population_seed']), int(r['training_seed'])
    with (FIRST / 'episodes.csv').open(encoding='utf-8-sig') as f:
        endpoints = {episode_key(r): r for r in csv.DictReader(f) if r['arm'] == 'wrong_advice_ppo'}
    checks = 0
    for r in rollouts:
        if r['behavior'] == 'rule' and r['arm'] == 'original_ppo':
            b = endpoints[episode_key(r)]
            assert all(abs(float(r[m]) - float(b[m])) < 1e-12 for m in METRICS), r['state_id']
            checks += 1
    same = [r for r in effects if r['arm'] == IMMEDIATE_ARMS[states[r['state_id']]['original_requested_action'] % 4]]
    assert len({r['state_id'] for r in same}) == 486
    summaries = contrast_summary(same, 'lower_direction_same_release_all')
    for behavior in ('rule', 'calibrated_logit'):
        summaries += contrast_summary([r for r in same if r['behavior'] == behavior], 'lower_direction_same_release_' + behavior)
    for criterion, comparison in [('ge', lambda a, b: a >= b), ('lt', lambda a, b: a < b)]:
        selected = []
        for r in same:
            state = states[r['state_id']]
            if comparison(state['capacities'][state['lower_branch']], state['capacities'][state['higher_branch']]):
                selected.append(r)
        summaries += contrast_summary(selected, 'lower_direction_same_release_lower_capacity_' + criterion + '_higher')
    for arm, label in [('wait_until_lower_then_ppo', 'wait_until_lower_hold_limited'), ('wait_one_then_ppo', 'wait_one_hold_limited')]:
        selected = [r for r in effects if r['arm'] == arm and states[r['state_id']]['category'] == 'sign_hold_prevents_switch']
        summaries += contrast_summary(selected, label)
    def transitions(selected):
        fed = {r['state_id']: r['difference'] for r in selected if r['metric'] == 'mean_cumulative_fed'}
        incap = {r['state_id']: r['difference'] for r in selected if r['metric'] == 'incapacitation_rate'}
        return dict(states=len(fed), fed_improves_without_more_incapacitation=sum(d < -1e-12 and incap[s] <= 1e-12 for s, d in fed.items()),
                    fed_improves_but_more_incapacitation=sum(d < -1e-12 and incap[s] > 1e-12 for s, d in fed.items()),
                    incapacitation_reduced=sum(d < -1e-12 for d in incap.values()),
                    incapacitation_increased=sum(d > 1e-12 for d in incap.values()))
    same_joint = transitions(same)
    out = unique_output('direction_counterfactual_final_audit')
    csv_write(out / 'matched_direction_effects.csv', same)
    result = dict(same_release_contrasts=summaries, same_release_joint_outcomes=same_joint,
                  exact_deterministic_rule_terminal_checks=checks, new_simulations=0,
                  simulation_output=str(sim), feasibility_output=str(audit),
                  original_frozen_sources_unchanged=True,
                  protected_selection_sha256=sha(ROOT / 'outputs/local_llm_selection.json'),
                  protected_research_status_sha256=sha(ROOT / 'outputs/third_round_development_status.json'),
                  no_global_qualification_change=True, reserved_final_scenarios_opened=0,
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__),
                      PREPARATION / 'matched_contrast_analysis_manifest.json', sim / 'completion_manifest.json', audit / 'completion_manifest.json']})
    save(out / 'summary.json', result)
    labels = {'mean_cumulative_fed': '人均累积FED', 'incapacitation_rate': '失能率',
              'evacuation_rate': '疏散率', 'cumulative_congestion': '累计拥堵'}
    audit_counts = read(audit / 'summary.json')
    cf = read(sim / 'summary.json')
    lines = ['# PPO较高烟气方向动作：机制审计与配对反事实结果', '',
             '本轮停止27B提速，继续研究“门控拒绝错误建议后，PPO为什么仍选较高烟气方向”。没有调用新LLM，没有新增训练，也没有打开保留最终场景。', '',
             '## 先确认发生了什么', '',
             '原先的596次记录来自11个开发场景、129个回合，不是596个独立场景。我们按旧日志逐步恢复全部240个错误建议回合，10,483步的应用动作、实际方向、放行人数、容量限制和终止结果均与原记录一致。', '',
             '其中486个状态允许立即换到较低烟气方向，110个状态受指示牌最短保持时间限制。这110次不能归为“可以立即纠错却没有纠错”。出口开放和指示牌现在能切换是两个条件。', '',
             '## 如何比较替代动作', '',
             f"对全部596个状态共运行{cf['terminal_rollouts']:,}条完整后续轨迹。复制同一批人的属性、当时位置与已吸入烟气；使用相同PPO权重。即时替代动作只取原mask允许的低烟气方向放行档位。另比较等待一步，以及等待1—3步、合法时换向再恢复PPO。", '',
             '校准概率行为用4个冻结未来种子，并按同一人、同一时刻配对随机数，避免路线变化后随机抽样序列错位。确定性规则行为只用1个未来种子。原PPO的确定性后续结果也与历史终点逐一核对。', '',
             '干预后继续同一PPO和同一种可观察错误建议生成逻辑。这里只改变当前一步或短计划，不能把单步效果直接相加成全程控制器效果。', '',
             '## 把换方向和减少放行分开', '',
             '下面专门比较“换到较低烟气方向，但保持原来的放行档位”。这能更清楚地说明方向本身的作用。先在未来种子内平均，再在回合内平均，再在场景内平均；区间是开发场景的描述性bootstrap，条件于既有权重。', '',
             '|比较范围|指标|状态|场景|平均差（替代−原PPO）|描述性95%区间|改善/恶化/不变|',
             '|---|---|---:|---:|---:|---|---|']
    for r in summaries:
        ci = r['descriptive_scene_bootstrap_95_interval']
        lines.append(f"|{r['contrast']}|{labels[r['metric']]}|{r['states']}|{r['scenario_clusters']}|{r['equal_scene_episode_state_mean_difference']:.7f}|{ci[0]:.7f}, {ci[1]:.7f}|{r['improved_states']}/{r['worsened_states']}/{r['unchanged_states']}|")
    lines += ['', 'FED、失能率和拥堵的负差通常表示改善，疏散率的正差表示改善。率的差乘100才是百分点。低烟气出口容量较小和容量不小于高烟气出口分别列出，便于检查是否存在烟气与通行能力的权衡。', '',
              '“等待”只停止房间放行，路口现有人群仍会按实际指示牌选择路线；等待不等于冻结所有人的运动。仿真结束时间包含失能后结束，不能称为全员疏散时间。', '',
              '同放行档位的联合结果：', '```json', json.dumps(same_joint, ensure_ascii=False, indent=2), '```', '',
              '完整候选库的事后挑选结果：', '```json', json.dumps(cf['offline_library_diagnostics'], ensure_ascii=False, indent=2), '```', '',
              '## 能说明什么，不能说明什么', '',
              '这一步把“错误建议拒绝”“物理可执行纠错”“纠错后的实际收益”分开了。操作限制通过不代表动作最优；烟气更低也必须结合容量、等待时间、人群响应和最终失能一起看。', '',
              '开发结果不能证明27B或LLM＋PPO已有稳定优势。此次没有新LLM推理；此前缓存来自Qwen3 4B。同权重对照里建议收益仍很小，区间跨零。这次反事实只审计下游控制机制。', '',
              '事后用真实未来选择最优候选属于乐观诊断上限，不能拿它当实时策略。新的动作校验若要采用，必须只依赖现场观测，统一约束所有控制器，并在另行冻结的开发确认场景上跑完整回合；不能把本轮最优动作直接作为已经验证的安全盾。', '',
              '所有结果仍基于简化图结构和合成烟气FED增量；尚不能外推为真实火场的剂量或安全保证。正式第三轮训练和最终资格审查状态均未修改。', '',
              f"仿真原始结果：{sim.relative_to(ROOT).as_posix()}",
              f"可执行性重放：{audit.relative_to(ROOT).as_posix()}",
              '本目录保存同放行对照数据、分析脚本来源哈希及最终审计台账。', '']
    report = out / 'report.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'same_release_joint_outcomes': same_joint,
                      'contrasts': [r for r in summaries if r['contrast'] == 'lower_direction_same_release_all']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--simulation')
    p.add_argument('--feasibility')
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        assert args.simulation and args.feasibility
        run(args.simulation, args.feasibility)
