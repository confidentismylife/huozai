"""Recover completed CSVs after NumPy counter JSON error; no new rollouts."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, PREP, read, save, sha, unique_output
from run_timestamped_pipeline_followup import FIRST, FREEZE, verify


def run():
    verify()
    candidates = [p for p in (ROOT / 'outputs').glob('timestamped_pipeline_followup_*') if (p / 'episodes.csv').exists() and not (p / 'completion_manifest.json').exists()]
    assert len(candidates) == 1, 'Identify the exact failed postprocessing run before recovery.'
    failed = candidates[0]
    with (failed / 'episodes.csv').open(encoding='utf-8-sig') as f:
        fresh = list(csv.DictReader(f))
    assert len(fresh) == 480
    assert Counter(r['arm'] for r in fresh) == {'no_advice_same_ppo': 240, 'observable_conflict_audit': 240}
    for r in fresh:
        assert not any(int(r[k]) for k in ('unavailable_entries', 'release_violations', 'direction_violations'))
    with (FIRST / 'episodes.csv').open(encoding='utf-8-sig') as f:
        old = list(csv.DictReader(f))
    index = {(r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']), r['arm']): r for r in old}
    # Verify recovered fault arm gives the same outcome as the original exact
    # algorithm. Recovery changes serialization, not simulated decisions.
    metrics = ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate', 'episode_end_seconds')
    for r in fresh:
        if r['arm'] == 'observable_conflict_audit':
            b = index[(r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']), 'wrong_advice_ppo')]
            assert all(abs(float(r[m])-float(b[m])) < 1e-12 for m in metrics)
    out = unique_output('timestamped_pipeline_followup_recovered')
    sources = [failed / 'episodes.csv', FIRST / 'execution_trace.jsonl', FIRST / 'episodes.csv', Path(__file__),
               ROOT / 'scripts/run_qwen35_timestamped_live_v3.py',
               ROOT / 'tests/test_timestamped_live_v3.py', FREEZE / 'preparation_manifest.json']
    save(out / 'recovery_manifest.json', {'purpose': 'postprocessing_recovery_only', 'new_rollouts': 0, 'new_inference': 0,
         'completed_rollouts_recovered': 480, 'original_failure': 'JSON serializer rejected NumPy int64 counters after episodes.csv completed.',
         'fault_arm_outcomes_equal_original_240_pairs': True,
         'error_audit_basis': 'Original per-step trace joined to contemporaneous no-noise synthetic sensor values; hidden scenario used only after decisions, never supplied to LLM/PPO.',
         'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}})
    counters = Counter()
    bank = {r['scenario_id']: r for r in read(PREP / 'manifest.json')['scenarios']}
    from fire_evac.hazard import SyntheticHazardProvider
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    provider = SyntheticHazardProvider()
    detailed = []
    with (FIRST / 'execution_trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['arm'] != 'wrong_advice_ppo' or r['source'] != 'injected_observable_contradiction':
                continue
            scene = DynamicConflictScenario(**bank[r['scenario_id']])
            smoke = provider.values(scene, r['step']).astype(np.float32)[3:5]
            worse = int(smoke[1] > smoke[0])
            capacities = [scene.north_exit_capacity, scene.south_exit_capacity]
            if scene.blockage_start_step <= r['step'] < scene.blockage_end_step and scene.blocked_branch in (0, 1):
                capacities[scene.blocked_branch] = scene.blockage_capacity
            usable_other = bool(capacities[1-worse] > 0 and smoke[1-worse] < .15)
            contradiction = 'live_smoke_contradiction' in r['gate_reasons']
            final_same = r['actual_joint_action'] != 8 and r['actual_sign'] == worse
            counters['total_injections'] += 1
            counters['alternative_admissible'] += int(usable_other)
            counters['alternative_unavailable_ambiguous'] += int(not usable_other)
            counters['rejected_by_smoke_contradiction'] += int(contradiction)
            counters['any_gate_rejection'] += int(not r['gate_accepted'])
            counters['accepted_while_alternative_unavailable'] += int(r['gate_accepted'] and not usable_other)
            counters['higher_smoke_direction_executed'] += int(final_same)
            counters['rejected_contradiction_but_same_direction_executed'] += int(contradiction and final_same)
            counters['ambiguous_only_available_direction_executed'] += int(not usable_other and final_same)
            detailed.append(dict(scenario_id=r['scenario_id'], behavior=r['behavior'], population_seed=r['population_seed'],
                                 training_seed=r['training_seed'], step=r['step'], contemporaneous_smoke=list(map(float, smoke)),
                                 capacities=capacities, proposed_branch=worse, gate_reasons=r['gate_reasons'],
                                 alternative_admissible=usable_other, final_joint_action=r['actual_joint_action'],
                                 rejected_contradiction_but_same_direction=contradiction and final_same))
    assert counters['total_injections'] == read(FIRST / 'summary.json')['arms']['wrong_advice_ppo']['wrong_injections']
    assert counters['higher_smoke_direction_executed'] == read(FIRST / 'summary.json')['arms']['wrong_advice_ppo']['wrong_executed_direction']
    comparisons = []
    base = [r for r in fresh if r['arm'] == 'no_advice_same_ppo']
    for arm in ('cached_llm_ppo', 'delayed_advice_ppo', 'wrong_advice_ppo'):
        for metric in metrics:
            differences = []
            cluster = defaultdict(list)
            for b in base:
                a = index[(b['scenario_id'], b['behavior'], int(b['population_seed']), int(b['training_seed']), arm)]
                d = float(a[metric]) - float(b[metric])
                differences.append(d)
                cluster[b['scenario_id']].append(d)
            scene_means = np.array([np.mean(cluster[k]) for k in sorted(cluster)])
            rng = np.random.default_rng(2026100711)
            boots = scene_means[rng.integers(0, len(scene_means), (10000, len(scene_means)))].mean(axis=1)
            comparisons.append({'candidate': arm, 'baseline': 'no_advice_same_ppo', 'metric': metric,
                'mean_difference': float(np.mean(differences)), 'nonzero_pairs': int(np.count_nonzero(np.abs(differences) > 1e-12)),
                'paired_episodes': 240, 'scenario_clusters': 12,
                'descriptive_scene_bootstrap_95_interval': list(map(float, np.quantile(boots, [.025, .975])))})
    summary = {'completed_rollouts_recovered': 480, 'new_rollouts_during_recovery': 0, 'new_27b_calls': 0,
               'error_audit': dict(counters), 'same_checkpoint_comparisons': comparisons,
               'no_advice_means': {m: float(np.mean([float(r[m]) for r in base])) for m in metrics},
               'postprocessing_bug_recovered_without_repeating_simulations': True}
    save(out / 'summary.json', summary)
    with (out / 'error_trace.jsonl').open('w', encoding='utf-8') as f:
        for r in detailed:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    lines = ['# 同一PPO权重对照与错误方向审计', '', '另行冻结新增480回合已完成。CSV保存后JSON计数序列化失败；本报告从CSV与原始轨迹恢复，不重复仿真。', '',
             '使用相同raw_llm_ppo权重、约束、场景与人口种子，比较有缓存建议和无建议；这才能区分建议本身与模型权重的作用。', '',
             '|候选−相同权重无建议|指标|均差|场景bootstrap描述性95%区间|非零配对|', '|---|---|---:|---|---:|']
    for r in comparisons:
        lo, hi = r['descriptive_scene_bootstrap_95_interval']
        lines.append(f"|{r['candidate']}|{r['metric']}|{r['mean_difference']:.6f}|{lo:.6f}, {hi:.6f}|{r['nonzero_pairs']}/240|")
    lines += ['', '区间按12个复用开发场景计算，条件于既有五组权重，不是最终资格结论或完整训练不确定性。', '',
              '更高烟气方向不必然错误：另一出口不可用时，仅看烟气不能判定应覆盖。以下把此类情况列为歧义，另统计“实时矛盾已触发拒绝，但最终又执行同一方向”。', '',
              '```json', json.dumps(dict(counters), ensure_ascii=False, indent=2), '```', '',
              '门控拒绝与PPO纠错不同。此处“执行同一方向”也不等同于反事实最终伤害更大，需后续专门比较可行替代方案的收益。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    for p, digest in read(out / 'recovery_manifest.json')['sources'].items():
        assert sha(ROOT / p) == digest, p
    save(out / 'completion_manifest.json', {'state': 'complete_recovered', 'protected_sources_unchanged': True,
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'error_audit': dict(counters), 'same_checkpoint_comparisons': comparisons[:5]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
