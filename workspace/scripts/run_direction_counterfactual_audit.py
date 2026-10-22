"""Append-only state reconstruction and paired terminal counterfactuals."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import json
import sys
import time
import numpy as np

from run_timestamped_development_campaign import ROOT, PREP, read, save, sha, csv_write, unique_output
from run_timestamped_pipeline_followup import verify as verify_previous
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv
from fire_evac.counterfactual_direction_audit import (
    episode_key, state_id, lower_branch, available_arms, intervention_action,
    clone_for_future, paired_step, faulty_policy_action, terminal_metrics)

FIRST = ROOT / 'outputs/timestamped_pipeline_development_20261006T172554364147Z'
RECOVERED = ROOT / 'outputs/timestamped_pipeline_followup_recovered_20261006T173442494950Z'
PREPARATION = ROOT / 'outputs/direction_counterfactual_preparation_20261007'
FUTURE_SEEDS = [2026100721, 2026100722, 2026100723, 2026100724]
METRICS = ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate')


def source_files():
    return [Path(__file__), ROOT / 'src/fire_evac/counterfactual_direction_audit.py',
            ROOT / 'tests/test_counterfactual_direction_audit.py', FIRST / 'execution_trace.jsonl',
            FIRST / 'episodes.csv', RECOVERED / 'error_trace.jsonl', RECOVERED / 'completion_manifest.json',
            PREP / 'manifest.json', ROOT / 'outputs/local_llm_selection.json',
            ROOT / 'outputs/third_round_development_status.json']


def verify():
    verify_previous()
    if PREPARATION.exists():
        for name, digest in read(PREPARATION / 'manifest.json')['sources'].items():
            assert sha(ROOT / name) == digest, name


def prepare():
    if PREPARATION.exists():
        verify()
        return
    verify_previous()
    PREPARATION.mkdir()
    save(PREPARATION / 'manifest.json', dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), scope='development_only',
        targets_expected=596, exact_replay_episodes=240, new_training=0, llm_requests=0,
        reserved_final_scenarios_opened=0, calibrated_future_seeds=FUTURE_SEEDS,
        deterministic_rule_future_seeds=[FUTURE_SEEDS[0]],
        terminal_rollout_cap=12000, wall_seconds_cap=1800,
        arms=['original_ppo', 'lower_release_25', 'lower_release_50', 'lower_release_75',
              'lower_release_100', 'wait_one_then_ppo', 'wait_until_lower_then_ppo'],
        feasibility='Immediate lower-direction arms run only if the frozen mask permits them; never override holds or capacity limits.',
        wait_plan='Wait 1..3 steps until lower-smoke branch is mask-admissible; release at the greatest legal level no higher than the original; then resume fixed PPO. If still infeasible, resume PPO.',
        continuation='Same existing raw_llm_ppo checkpoint and observable erroneous-proposal generator in all arms; intervention lasts at most 4 steps.',
        randomization='Copy individuals and history. Pair calibrated future uniforms by original evacuee_id and absolute step; no stream-index mismatch. Rule behavior has no random continuation.',
        estimand='Local forced intervention at the 596 flagged historical states; NOT performance of an online deployed shield or full-episode controller.',
        outcomes='Paired terminal FED and incapacitation primary, evacuation and congestion secondary. Episode end time is not successful evacuation time.',
        aggregation='Average future seeds within state; average states within episode; average episode means within scenario. Descriptive scenario bootstrap conditional on fixed checkpoints. Do not treat 596 states as independent scenes.',
        safety='Offline hidden state only for simulation; intervention selectors receive public snapshot/mask only. No policy fitted to counterfactual labels.',
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files()}))
    save(PREPARATION / 'preparation_manifest.json', {'files': {
        (PREPARATION / 'manifest.json').relative_to(ROOT).as_posix(): sha(PREPARATION / 'manifest.json')}})
    print('Frozen counterfactual protocol.', flush=True)


def load_trace():
    traces = defaultdict(list)
    with (FIRST / 'execution_trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['arm'] == 'wrong_advice_ppo':
                traces[episode_key(r)].append(r)
    for rows in traces.values():
        rows.sort(key=lambda r: r['step'])
        assert [r['step'] for r in rows] == list(range(len(rows)))
    assert len(traces) == 240
    flags = {}
    with (RECOVERED / 'error_trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['rejected_contradiction_but_same_direction']:
                flags[state_id(r)] = r
    assert len(flags) == 596
    return traces, flags


def make_env(key, scenes):
    scene = DynamicConflictScenario(**scenes[key[0]])
    behavior = RuleBehavior() if key[1] == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
    env = TimestampedControlEnv(behavior, (scene,))
    env.reset(seed=key[2], options={'scenario': scene})
    return env


def validate_transition(r, env):
    assert env.snapshot.step == r['step']
    _, _, term, trunc, info = env.step(int(r['requested_action']))
    pairs = [('applied_action', 'constraint_applied_action'), ('actual_joint_action', 'actual_joint_action'),
             ('actual_sign', 'actual_sign_action'), ('actual_room_count', 'actual_room_release_count'),
             ('room_limit', 'pretransition_room_limit'), ('unavailable_entries', 'unavailable_corridor_entries')]
    assert all(int(r[a]) == int(info[b]) for a, b in pairs), state_id(r)
    return info, bool(term or trunc)


def audit():
    verify()
    traces, flags = load_trace()
    plan = read(PREP / 'manifest.json')
    scenes = {s['scenario_id']: s for s in plan['scenarios']}
    endpoints = {}
    with (FIRST / 'episodes.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['arm'] == 'wrong_advice_ppo':
                endpoints[episode_key(r)] = r
    out = unique_output('direction_state_feasibility')
    rows, counts = [], Counter()
    replay_steps = 0
    start = time.monotonic()
    for key in sorted(traces):
        env = make_env(key, scenes)
        for r in traces[key]:
            sid = state_id(r)
            target = sid in flags
            s = env.snapshot
            if target:
                flag = flags[sid]
                assert s.smoke[3 + flag['proposed_branch']] > s.smoke[4 - flag['proposed_branch']] + .01
                mask = env.constraints.action_mask(s)
                b = lower_branch(s)
                legal = [a for a in range(b * 4, b * 4 + 4) if mask[a]]
                category = ('immediate_lower_feasible' if legal else
                            'sign_hold_prevents_switch' if s.sign_hold_remaining and s.sign != b else
                            'release_budget_below_discrete_minimum')
                counts[category] += 1
                rows.append(dict(state_id=sid, scenario_id=key[0], behavior=key[1],
                                 population_seed=key[2], training_seed=key[3], step=s.step,
                                 original_requested_action=int(r['requested_action']),
                                 higher_branch=1-b, lower_branch=b, smoke=list(s.smoke[3:]),
                                 capacities=list(s.capacities), sign=s.sign, hold_remaining=s.sign_hold_remaining,
                                 junction_people=int(round(s.occupancy[2] * s.population)),
                                 room_people=int(round(sum(s.occupancy[:2]) * s.population)),
                                 room_limit=env.constraints.room_limit(s), legal_lower_actions=legal,
                                 feasible_arms=available_arms(s, env.constraints), category=category))
            info, ended = validate_transition(r, env)
            replay_steps += 1
            if target:
                rows[-1].update(actual_room_count=int(info['actual_room_release_count']),
                                branch_entries=list(map(int, env.unwrapped.domain_env.last_branch_flow)),
                                actual_sign=int(info['actual_sign_action']))
                assert int(info['actual_sign_action']) == flags[sid]['proposed_branch']
        assert ended
        assert all(abs(float(info[m]) - float(endpoints[key][m])) < 1e-12 for m in METRICS)
        env.close()
    assert len(rows) == 596
    with (out / 'states.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    budget = sum(len(r['feasible_arms']) * (1 if r['behavior'] == 'rule' else 4) for r in rows)
    summary = dict(target_states=len(rows), source_episodes=240,
                   target_episodes=len({episode_key(r) for r in rows}),
                   target_scenario_clusters=len({r['scenario_id'] for r in rows}),
                   counts=dict(counts), exact_replayed_steps=replay_steps,
                   exact_all_transition_and_terminal_checks=True,
                   planned_terminal_rollouts=budget,
                   inactive_room_and_junction_states=sum(r['room_people'] == 0 and r['junction_people'] == 0 for r in rows),
                   elapsed_seconds=time.monotonic() - start, sources_unchanged=True)
    save(out / 'summary.json', summary)
    verify()
    save(out / 'completion_manifest.json', {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), **summary}, ensure_ascii=False), flush=True)


def verify_audit(path):
    out = Path(path)
    for p, digest in read(out / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == digest, p
    return out


def run(audit_path):
    import torch
    from stable_baselines3 import PPO
    verify()
    audit_out = verify_audit(audit_path)
    with (audit_out / 'states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads, f)}
    expected = read(audit_out / 'summary.json')['planned_terminal_rollouts']
    assert expected <= read(PREPARATION / 'manifest.json')['terminal_rollout_cap']
    traces, flags = load_trace()
    assert set(states) == set(flags)
    plan = read(PREP / 'manifest.json')
    scenes = {s['scenario_id']: s for s in plan['scenarios']}
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['ppo_models'][str(s)]['raw_llm_ppo'], device='cpu') for s in plan['training_seeds']}
    out = unique_output('direction_counterfactual')
    save(out / 'run_manifest.json', dict(preparation_sha256=sha(PREPARATION / 'manifest.json'),
         audit_manifest_sha256=sha(audit_out / 'completion_manifest.json'), audit_path=str(audit_out),
         planned_terminal_rollouts=expected, future_seeds=FUTURE_SEEDS,
         new_training=0, llm_requests=0, reserved_final_scenarios_opened=0))
    started = time.monotonic()
    count = done_states = steps = replay_steps = 0
    path = out / 'terminal_rollouts.csv'
    writer = None
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        for key in sorted(traces):
            env = make_env(key, scenes)
            for old in traces[key]:
                sid = state_id(old)
                if sid in states:
                    rec = states[sid]
                    assert rec['feasible_arms'] == available_arms(env.snapshot, env.constraints)
                    # PPO recomputation checks policy replay as well as physical replay.
                    assert faulty_policy_action(env, models[key[3]]) == old['requested_action'], sid
                    futures = FUTURE_SEEDS[:1] if key[1] == 'rule' else FUTURE_SEEDS
                    for future in futures:
                        for arm in rec['feasible_arms']:
                            if time.monotonic() - started > 1800:
                                raise RuntimeError('Frozen wall-clock budget exhausted; retain partial CSV.')
                            branch_env = clone_for_future(env, future)
                            n = 0
                            active_plan = True
                            violations = Counter()
                            first = []
                            switched_lower = False
                            while True:
                                action = intervention_action(arm, branch_env, n, old['requested_action'] % 4) if active_plan else None
                                if action is None:
                                    active_plan = False
                                    action = faulty_policy_action(branch_env, models[key[3]])
                                elif action != 8:
                                    active_plan = False
                                    switched_lower = True
                                _, _, term, trunc, info = paired_step(branch_env, action)
                                if n < 4:
                                    first.append(int(info['actual_joint_action']))
                                violations['unavailable_entries'] += int(info['unavailable_corridor_entries'])
                                violations['release_violations'] += int(not info['actual_release_within_limit'])
                                violations['direction_violations'] += int(info['actual_direction_violation'])
                                n += 1
                                if term or trunc:
                                    break
                            row = dict(state_id=sid, scenario_id=key[0], behavior=key[1],
                                       population_seed=key[2], training_seed=key[3], step=old['step'],
                                       category=rec['category'], future_seed=future, arm=arm,
                                       continuation_steps=n, intervention_lower_executed=switched_lower,
                                       first_actual_actions=json.dumps(first),
                                       **terminal_metrics(info, branch_env.unwrapped.domain_env.config),
                                       **dict(violations))
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            count += 1
                            steps += n
                            branch_env.close()
                    done_states += 1
                    if done_states % 40 == 0:
                        f.flush()
                        print(json.dumps({'states': done_states, 'terminal_rollouts': count,
                                          'elapsed_seconds': round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
                validate_transition(old, env)
                replay_steps += 1
            env.close()
    assert count == expected and done_states == 596
    save(out / 'execution_summary.json', dict(terminal_rollouts=count, target_states=done_states,
         continuation_steps=steps, exact_replay_steps=replay_steps,
         elapsed_seconds=time.monotonic() - started))
    verify()
    analyze(out)
    save(out / 'completion_manifest.json', {'state': 'complete', 'previous_sources_unchanged': True,
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'completed': count, 'output': str(out)}, ensure_ascii=False), flush=True)


def aggregate_scene(pairs):
    episodes = defaultdict(list)
    for r in pairs:
        episodes[episode_key(r)].append(r['difference'])
    scenes = defaultdict(list)
    for key, values in episodes.items():
        scenes[key[0]].append(float(np.mean(values)))
    return {k: float(np.mean(v)) for k, v in scenes.items()}


def analyze(out):
    with (out / 'terminal_rollouts.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    base = {(r['state_id'], r['future_seed']): r for r in rows if r['arm'] == 'original_ppo'}
    by_state = defaultdict(list)
    paired = []
    for r in rows:
        if r['arm'] == 'original_ppo':
            continue
        b = base[(r['state_id'], r['future_seed'])]
        for metric in METRICS:
            pair = dict(state_id=r['state_id'], scenario_id=r['scenario_id'], behavior=r['behavior'],
                        population_seed=int(r['population_seed']), training_seed=int(r['training_seed']),
                        step=int(r['step']), category=r['category'], future_seed=int(r['future_seed']),
                        arm=r['arm'], metric=metric, difference=float(r[metric]) - float(b[metric]))
            by_state[(r['state_id'], r['arm'], metric)].append(pair)
            paired.append(pair)
    state_pairs = []
    for items in by_state.values():
        r = dict(items[0])
        r.pop('future_seed')
        r['difference'] = float(np.mean([x['difference'] for x in items]))
        r['future_replications'] = len(items)
        state_pairs.append(r)
    csv_write(out / 'state_mean_effects.csv', state_pairs)
    summaries = []
    for arm in sorted({r['arm'] for r in state_pairs}):
        for metric in METRICS:
            for behavior in ('all', 'rule', 'calibrated_logit'):
                selected = [r for r in state_pairs if r['arm'] == arm and r['metric'] == metric and
                            (behavior == 'all' or r['behavior'] == behavior)]
                if not selected:
                    continue
                scene = aggregate_scene(selected)
                values = np.array(list(scene.values()))
                rng = np.random.default_rng(2026100729)
                boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                effects = np.array([r['difference'] for r in selected])
                summaries.append(dict(arm=arm, metric=metric, behavior=behavior, states=len(selected),
                    episode_clusters=len({episode_key(r) for r in selected}), scenario_clusters=len(scene),
                    equal_scene_episode_state_mean_difference=float(values.mean()),
                    descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975]))),
                    improved_states=int(np.sum(effects < -1e-12)) if metric in METRICS[:2] + ('incapacitation_rate',) else int(np.sum(effects > 1e-12)),
                    worsened_states=int(np.sum(effects > 1e-12)) if metric in METRICS[:2] + ('incapacitation_rate',) else int(np.sum(effects < -1e-12)),
                    unchanged_states=int(np.sum(np.abs(effects) <= 1e-12)), scene_means=scene))
    # Offline best-of-library is an optimistic diagnostic upper bound, never a controller.
    fed_map = defaultdict(dict)
    incap_map = {}
    for r in state_pairs:
        if r['metric'] == 'mean_cumulative_fed':
            fed_map[r['state_id']][r['arm']] = r['difference']
        if r['metric'] == 'incapacitation_rate':
            incap_map[(r['state_id'], r['arm'])] = r['difference']
    diagnostics = Counter()
    for sid, candidates in fed_map.items():
        best_arm = min(candidates, key=candidates.get)
        best = candidates[best_arm]
        diagnostics['at_least_one_fed_improvement'] += int(best < -1e-12)
        diagnostics['no_fed_improvement_in_library'] += int(best >= -1e-12)
        diagnostics['all_candidate_fed_worse'] += int(all(d > 1e-12 for d in candidates.values()))
        diagnostics['fed_improvement_without_more_incapacitation'] += int(any(d < -1e-12 and incap_map[(sid, a)] <= 1e-12 for a, d in candidates.items()))
        diagnostics['best_fed_arm_increases_incapacitation'] += int(best < -1e-12 and incap_map[(sid, best_arm)] > 1e-12)
    summary = dict(**read(out / 'execution_summary.json'), paired_effect_summaries=summaries,
        offline_library_diagnostics=dict(diagnostics),
        violations={k: sum(int(r[k]) for r in rows) for k in ('unavailable_entries', 'release_violations', 'direction_violations')},
        new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        statistical_status='descriptive_development_audit_conditional_on_frozen_weights',
        success_time_warning='End time can shorten through incapacitation; all_evacuated_seconds only exists when all people evacuated.',
        local_interventions_not_a_deployed_shield=True)
    save(out / 'summary.json', summary)
    lines = ['# 较高烟气方向动作的可执行性与反事实审计', '',
             f"完成596个历史控制状态的重放，新增{len(rows)}条完整终止反事实轨迹。保留最终场景未读取；新训练0，新LLM请求0。", '',
             '本实验在相同人群属性、历史、PPO权重下，只改变当前动作或最多四步等待/切换计划。之后继续同一PPO与同一种错误建议生成规则。', '',
             '校准行为的未来随机数按“同一人、同一绝对时刻”配对，使用4个预先冻结的未来种子。确定性规则行为只跑1个未来，避免伪造重复信息。', '',
             '先对未来种子取均值，再对同一回合中的目标状态取均值，再对同一场景中的回合取均值。下面是各场景等权描述性结果；不是596个独立场景的显著性检验。', '',
             '|干预相对原PPO|指标|覆盖状态|场景|均差|描述性95%区间|改善/恶化/不变状态|',
             '|---|---|---:|---:|---:|---|---|']
    for r in summaries:
        if r['behavior'] == 'all':
            ci = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['arm']}|{r['metric']}|{r['states']}|{r['scenario_clusters']}|{r['equal_scene_episode_state_mean_difference']:.6f}|{ci[0]:.6f}, {ci[1]:.6f}|{r['improved_states']}/{r['worsened_states']}/{r['unchanged_states']}|")
    lines += ['', '即时换方向仅评估原动作mask允许的动作；出口开放但指示牌不能切换的状态没有强行执行换向。等待只停止房间放行，路口人员仍按当前有效指示牌行动。', '',
              '各即时放行档位的可行状态子集不同，不能直接把它们的组均值作公平排名。离线事后挑最优候选属于乐观上限，不允许作为实时控制器，更不证明LLM收益。', '',
              '候选库诊断：', '```json', json.dumps(dict(diagnostics), ensure_ascii=False, indent=2), '```', '',
              'FED越低一般越好，但必须同时检查失能率；更早失能终止不能当作更快疏散。时间戳建议与mask只保证本仿真的执行限制，不保证低剂量或现实火场安全。', '',
              '本轮属于重复开发场景的机制审计。新约束如要采用，需在另行冻结的开发确认场景中做完整回合比较；最终场景仍封存。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'audit', 'run', 'verify'])
    parser.add_argument('--audit-path')
    args = parser.parse_args()
    if args.stage == 'run':
        assert args.audit_path
        run(args.audit_path)
    else:
        {'prepare': prepare, 'audit': audit, 'verify': verify}[args.stage]()
