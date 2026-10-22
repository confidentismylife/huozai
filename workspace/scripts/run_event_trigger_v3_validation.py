"""Frozen 900-episode, zero-training, zero-API event repair diagnostic."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mappo_six_hour_extension as previous
from analyze_mappo_six_hour_extension import paired_effect
from fire_evac.behavior import MixedLogitBehavior
from fire_evac.campaign_extension import ActivePhasePlanWrapper, write_json
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import rule_plan
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv
from fire_evac.event_plan_v3 import EventCorrectedPlanWrapper
from fire_evac.mappo import FactorizedMAPPO, encode_factorized_action

OUT = ROOT / 'outputs/event_trigger_v3_validation_20261022'
SEEDS = tuple(range(20261301, 20261306))
ARMS = (previous.RAW, previous.RULE, previous.EXPANDED)
METRICS = ('incapacitation_rate', 'mean_cumulative_fed', 'evacuation_time_seconds',
           'cumulative_congestion', 'plan_requests')


def checkpoint(seed, arm):
    folder = 'shared_raw' if arm == previous.RAW else 'e003_i002'
    return previous.OUT / 'mechanism' / folder / f'seed_{seed}' / arm / 'best.pt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'complete.json').exists():
        raise RuntimeError('Already complete; preserve this frozen run rather than overwrite.')
    scenarios = previous.base.make_domain_randomized_trust_bank(24, 20261021, 'DSG')[12:]
    conditions = [('normal', EnvConfig(), (42001, 42002)),
                  ('bottleneck', EnvConfig(junction_to_branch_capacity=3), (45001,))]
    inputs = [ROOT / 'src/fire_evac/event_plan_v3.py', Path(__file__),
        ROOT / 'docs/event_trigger_v3_validation_20261022.md',
        previous.OUT / 'plan_bank/response_audit.json']
    inputs += [checkpoint(seed, arm) for seed in SEEDS for arm in ARMS]
    source_hashes = {str(p): digest(p) for p in inputs}
    old = json.loads((previous.OUT / 'training_runtime_and_source_manifest.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT / name) == h for name, h in old['source_files'].items())
    manifest = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        conditions=[dict(name=n, config=asdict(c), population_seeds=list(p)) for n, c, p in conditions],
        scenarios=[asdict(s) for s in scenarios], training_seeds=SEEDS, source_sha256=source_hashes,
        target_episodes=900, new_training_steps=0, new_API_requests=0,
        known_before_run='Old saved evaluation trajectories stayed below congestion threshold.',
        development_diagnostic_only=True, final_test_scenarios_loaded=False)
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding='utf-8'))['source_sha256'] == source_hashes
    else:
        write_json(manifest_path, manifest)
    episodes, steps, events = [], [], []
    for condition, config, populations in conditions:
        for seed in SEEDS:
            for arm in ARMS:
                model, _ = FactorizedMAPPO.load(checkpoint(seed, arm))
                variants = ('shared',) if arm == previous.RAW else ('legacy', 'v3')
                for variant in variants:
                    context = dict(condition=condition, training_seed=seed, arm=arm, variant=variant)
                    physical = ConfirmatoryJointControlWrapper(
                        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios, config=config), advice_mode='none')
                    library = previous.expanded_library() if arm == previous.EXPANDED else None
                    calls = []
                    def provider(state):
                        plan = library(state) if library is not None else rule_plan(state)
                        calls.append(dict(event_type=state['event_type'], timestamp_step=state['timestamp_step'],
                            has_same_event_library_case=(any(s['event_type'] == state['event_type']
                                for s, _ in library.records) if library is not None else None),
                            state=state))
                        return plan
                    wrapper = ActivePhasePlanWrapper if variant == 'legacy' else EventCorrectedPlanWrapper
                    env = physical if arm == previous.RAW else wrapper(physical, provider, max_requests_per_episode=3)
                    for scenario in scenarios:
                        for population in populations:
                            calls.clear()
                            identity = dict(**context, scenario_id=scenario.scenario_id,
                                conflict_type=scenario.conflict_type, population_seed=population)
                            observation, info = env.reset(seed=population, options={'scenario': scenario})
                            index, peak = 0, float(observation[2]) * config.population
                            while True:
                                routes, flows, _, _ = model.act(observation, deterministic=True)
                                action = encode_factorized_action(int(routes[0]), int(flows[0]))
                                record = dict(**identity, decision_step=index,
                                    junction_occupants=float(observation[2]) * config.population,
                                    requested_joint_action=action)
                                observation, _, terminated, truncated, info = env.step(action)
                                record.update(actual_joint_action=int(info['actual_joint_action']),
                                    actual_release_fraction=float(info['actual_release_fraction']))
                                steps.append(record)
                                index += 1
                                peak = max(peak, float(observation[2]) * config.population)
                                if terminated or truncated:
                                    break
                            assert len(calls) <= 3
                            for call in calls:
                                events.append(dict(**identity, **call,
                                    at_terminal_step=call['timestamp_step'] >= index))
                            episodes.append(dict(**identity,
                                incapacitation_rate=float(info['incapacitation_rate']),
                                mean_cumulative_fed=float(info['mean_cumulative_fed']),
                                evacuation_time_seconds=index * 5,
                                cumulative_congestion=float(info['cumulative_congestion']),
                                plan_requests=len(calls), maximum_junction_occupants=peak,
                                congestion_requests=sum(c['event_type'] == 'congestion_threshold' for c in calls)))
                    env.close()
                    write_json(OUT / 'status.json', dict(status='running', last_completed=context,
                        completed_episodes=len(episodes), target_episodes=900))
    frame = pd.DataFrame(episodes)
    assert len(frame) == 900
    identities = ['condition', 'training_seed', 'arm', 'variant', 'scenario_id', 'population_seed']
    assert not frame.duplicated(identities).any()
    assert (frame.plan_requests <= 3).all()
    for condition, _, populations in conditions:
        for seed in SEEDS:
            for arm in ARMS:
                for variant in ('shared',) if arm == previous.RAW else ('legacy', 'v3'):
                    subset = frame[(frame.condition == condition) & (frame.training_seed == seed)
                        & (frame.arm == arm) & (frame.variant == variant)]
                    assert set(zip(subset.scenario_id, subset.population_seed)) == {
                        (s.scenario_id, p) for s in scenarios for p in populations}
    replay_error = 0.
    for seed in SEEDS:
        for arm in ARMS:
            ref = pd.read_csv(checkpoint(seed, arm).parent / 'episode_results.csv').set_index(['scenario_id', 'population_seed'])
            observed = frame[(frame.condition == 'normal') & (frame.training_seed == seed)
                & (frame.arm == arm) & (frame.variant.isin(['legacy', 'shared']))]
            observed = observed.set_index(['scenario_id', 'population_seed']).reindex(ref.index)
            columns = list(METRICS[:4])
            delta = observed[columns].to_numpy() - ref[columns].to_numpy()
            np.testing.assert_allclose(delta, 0., atol=1e-12)
            replay_error = max(replay_error, float(np.abs(delta).max()))
    trace = pd.DataFrame(steps)
    lengths = frame.set_index(identities).evacuation_time_seconds / 5
    np.testing.assert_array_equal(trace.groupby(identities).size().reindex(lengths.index), lengths)
    effects = {}
    for (condition, arm), grouped in frame[frame.variant != 'shared'].groupby(['condition', 'arm']):
        effects[condition + '/' + arm] = {m: paired_effect(grouped, 'variant', 'v3', 'legacy', m)
            for m in METRICS}
    assert all(digest(Path(p)) == h for p, h in source_hashes.items())
    frame.to_csv(OUT / 'episode_results.csv', index=False, encoding='utf-8-sig')
    trace.to_csv(OUT / 'step_results.csv', index=False, encoding='utf-8-sig')
    with (OUT / 'request_events.jsonl').open('w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    means = frame.groupby(['condition', 'arm', 'variant'], as_index=False).agg(
        episodes=('scenario_id', 'count'), incapacitation=('incapacitation_rate', 'mean'),
        time=('evacuation_time_seconds', 'mean'), fed=('mean_cumulative_fed', 'mean'),
        congestion=('cumulative_congestion', 'mean'), requests=('plan_requests', 'mean'),
        congestion_requests=('congestion_requests', 'sum'), peak=('maximum_junction_occupants', 'max'))
    means.to_csv(OUT / 'method_means.csv', index=False, encoding='utf-8-sig')
    result = dict(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        episodes=len(frame), control_steps=len(trace), new_API_requests=0, training_steps=0,
        original_validation_replay_max_abs_error=replay_error, exact_episode_grids_passed=True,
        source_and_weight_hashes_unchanged=True, effects=effects, method_means=means.to_dict(orient='records'),
        congestion_library_fallback_requests=sum(e['event_type'] == 'congestion_threshold'
            and e['has_same_event_library_case'] is False for e in events),
        terminal_step_requests=sum(e['at_terminal_step'] for e in events),
        final_test_scenarios_loaded=False)
    write_json(OUT / 'analysis.json', result)
    lines = ['# 拥堵事件触发v3：修复与有限诊断结果', '',
        '本轮不训练、不调用付费API，使用五个保存种子和十二个开发场景。'
        'DeepSeek臂仍是既有48条计划的检索。修复代码独立保存，原六小时结果不覆写。', '',
        '## 功能修复', '',
        '旧烟气分支即使没有反转也会阻止拥堵检查。新版先判断反转是否发生，再按封闭、烟气、拥堵排序。'
        '新行为测试在旧实现上3失败、4通过；新版7通过，连同旧相关测试25通过。', '',
        '## 所有条件与方法均值', '',
        '| 条件 | 方法 | 协议 | 回合 | 失能率 | 时间s | 平均请求数 | 拥堵请求总数 | 最高交汇人数 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in means.to_dict(orient='records'):
        lines.append(f"| {row['condition']} | {row['arm']} | {row['variant']} | {row['episodes']} | "
            f"{100*row['incapacitation']:.3f}% | {row['time']:.2f} | {row['requests']:.3f} | "
            f"{row['congestion_requests']} | {row['peak']:.1f} |")
    lines += ['', '## 同权重v3减旧协议', '',
        '失能差负值较好，时间差负值较快。区间为开发种子×场景交叉重采样95%区间，未校正探索多重性。', '',
        '| 条件/方法 | 失能差pp [95%区间] | 时间差s | 暴露差 | 拥堵差 |',
        '|---|---:|---:|---:|---:|']
    for label, values in effects.items():
        v = values['incapacitation_rate']; lo, hi = v['crossed_ci95']
        lines.append(f"| {label} | {100*v['mean']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] | "
            f"{values['evacuation_time_seconds']['mean']:+.3f} | "
            f"{values['mean_cumulative_fed']['mean']:+.4f} | {values['cumulative_congestion']['mean']:+.3f} |")
    lines += ['', '## 如何解释', '',
        '正常条件用于回归核对：它的旧协议终点与原CSV逐回合一致，最大绝对差'
        f'{replay_error:.3g}。瓶颈条件只把每侧交汇通行容量从12人/步改为3人/步，'
        '是训练外的人工动力学压力诊断，不能代表真实建筑或最终测试。', '',
        f"新增识别的拥堵事件中有{result['congestion_library_fallback_requests']}次DeepSeek检索"
        '没有同类事件案例，只能跨事件检索；因此修复触发并不等于获得了合适的新计划。', '',
        f"请求计数中有{result['terminal_step_requests']}次发生在终止步；这是继承的请求时序，"
        '当前修复不改变它。这些终止步建议无法影响本回合后续动作，不应被算作有效干预。', '',
        '代码功能正确与终点收益是两项不同证据。若修复后无改善或恶化，也必须保留；'
        '不得为让LLM胜出调整瓶颈强度、阈值或筛选场景。后续若继续，先解决拥堵事件的计划覆盖、'
        '请求额度分配与终止步无效调用，再另行冻结有限的在线验证，而不是立即扩大训练。', '',
        f"执行完成{len(frame)}回合、{len(trace)}控制步；新API费用0元，训练0步，最终场景未读取。"]
    report = '\n'.join(lines)
    (OUT / '实验报告.md').write_text(report, encoding='utf-8')
    target = ROOT / 'reports/拥堵触发器v3修复与小规模对照_20261022.md'
    target.write_text(report, encoding='utf-8')
    write_json(OUT / 'complete.json', {k: v for k, v in result.items() if k not in ['effects', 'method_means']})
    write_json(OUT / 'status.json', dict(status='complete', completed_episodes=len(frame), report=str(target)))
    print(json.dumps(dict(episodes=len(frame), steps=len(trace), report=str(target), new_API_requests=0)))


if __name__ == '__main__':
    main()
