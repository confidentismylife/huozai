"""Additional frozen within-checkpoint comparison and error-count audit."""
from dataclasses import asdict
from collections import Counter, defaultdict
import argparse
import itertools
import json
import sys
import time

import numpy as np
from run_timestamped_development_campaign import (ROOT, PREP, OLD, read, save, sha, csv_write,
                                                 unique_output, verify_preparation, payload_for)
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, Proposal, ppo_action

FIRST = ROOT / 'outputs/timestamped_pipeline_development_20261006T172554364147Z'
FREEZE = ROOT / 'outputs/timestamped_pipeline_followup_preparation_20261007'


def prepare():
    assert not FREEZE.exists()
    verify_preparation()
    sources = [__file__, ROOT / 'src/fire_evac/timestamped_orchestrator.py', ROOT / 'tests/test_timestamped_orchestrator.py',
               ROOT / 'scripts/run_qwen35_timestamped_live_v2.py',
               FIRST / 'episodes.csv', FIRST / 'summary.json', FIRST / 'completion_manifest.json',
               PREP / 'manifest.json', PREP / 'preparation_manifest.json']
    FREEZE.mkdir()
    save(FREEZE / 'manifest.json', {'purpose': 'same_ppo_checkpoint_no_advice_ablation_and_direction_count_audit',
         'trigger': 'Cached/advised and pure PPO use different weights; rejected proposals did not imply different executed direction.',
         'rollouts_cap': 480, 'seconds_cap': 240, 'new_27b_calls': 0, 'new_ppo_training': 0,
         'arms': ['no_advice_same_ppo', 'observable_conflict_audit'],
         'population_seeds_match_original': True, 'scenarios': read(PREP / 'manifest.json')['scenarios'],
         'rule': 'wrong-direction rejection only identifiable when the lower-smoke alternative is currently admissible; other cases ambiguous',
         'sources': {str(Path(p).relative_to(ROOT).as_posix()): sha(p) for p in sources},
         'extension': 'live v2 keeps the same PPO checkpoint through pending, rejected, invalid, accepted, and expired advice'})
    save(FREEZE / 'preparation_manifest.json', {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in FREEZE.iterdir() if p.is_file()}})
    verify()


from pathlib import Path
def verify():
    verify_preparation()
    for p, digest in read(FREEZE / 'preparation_manifest.json')['files'].items():
        assert sha(ROOT / p) == digest, p
    for p, digest in read(FREEZE / 'manifest.json')['sources'].items():
        assert sha(ROOT / p) == digest, p


def run():
    import torch
    from stable_baselines3 import PPO
    import csv
    verify()
    torch.set_num_threads(1)
    plan = read(PREP / 'manifest.json')
    models = {s: PPO.load(plan['ppo_models'][str(s)]['raw_llm_ppo'], device='cpu') for s in plan['training_seeds']}
    out = unique_output('timestamped_pipeline_followup')
    rows, counters = [], Counter()
    started = time.monotonic()
    detailed = []
    for si, rec in enumerate(plan['scenarios']):
        for behavior, rep, seed, arm in itertools.product(plan['behavior_models'], range(2), plan['training_seeds'], ('no_advice_same_ppo', 'observable_conflict_audit')):
            if len(rows) >= 480 or time.monotonic() - started > 240:
                raise RuntimeError('frozen additional budget exhausted')
            scene = DynamicConflictScenario(**rec)
            env = TimestampedControlEnv(RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json'), (scene,))
            popseed = 1407100000 + si * 10 + rep
            _, info = env.reset(seed=popseed, options={'scenario': scene})
            ended = False
            physical_violations = Counter()
            while not ended:
                s = env.snapshot
                p, gate = None, None
                if arm == 'observable_conflict_audit' and abs(s.smoke[3]-s.smoke[4]) > .01:
                    worse = int(s.smoke[4] > s.smoke[3])
                    p = Proposal.parse(payload_for(s, worse, .99), s, s.observation_time_s, f'audit-{s.step}', 'fault_audit')
                    gate = env.constraints.validate_proposal(p, s, s.observation_time_s)
                action, raw = ppo_action(models[seed], env.policy_observation(p, gate), env.constraints.action_mask(s), p, gate)
                _, _, term, trunc, info = env.step(action)
                physical_violations['unavailable_entries'] += info['unavailable_corridor_entries']
                physical_violations['release_violations'] += not info['actual_release_within_limit']
                physical_violations['direction_violations'] += info['actual_direction_violation']
                ended = term or trunc
                if p is not None:
                    contradiction = 'live_smoke_contradiction' in gate.reasons
                    final_same = info['actual_joint_action'] != 8 and info['actual_sign_action'] == p.preferred_branch
                    usable_other = env.constraints.branch_allowed(s, 1-p.preferred_branch)
                    counters['total_injections'] += 1
                    counters['alternative_admissible'] += usable_other
                    counters['alternative_unavailable_ambiguous'] += not usable_other
                    counters['rejected_by_smoke_contradiction'] += contradiction
                    counters['any_gate_rejection'] += not gate.accepted
                    counters['accepted_while_alternative_unavailable'] += gate.accepted and not usable_other
                    counters['higher_smoke_direction_executed'] += final_same
                    counters['rejected_contradiction_but_same_direction_executed'] += contradiction and final_same
                    counters['ambiguous_only_available_direction_executed'] += not usable_other and final_same
                    detailed.append({'scenario_id': scene.scenario_id, 'behavior': behavior, 'population_seed': popseed,
                                     'training_seed': seed, 'step': s.step, 'smoke': s.smoke[3:], 'closed': s.closed,
                                     'proposed_branch': p.preferred_branch, 'gate_reasons': gate.reasons,
                                     'other_branch_admissible': usable_other, 'raw_ppo_action': raw,
                                     'final_action': info['actual_joint_action'], 'final_sign': info['actual_sign_action'],
                                     'rejected_contradiction_but_same_direction': contradiction and final_same})
            rows.append({'scenario_id': scene.scenario_id, 'behavior': behavior, 'population_seed': popseed,
                         'training_seed': seed, 'arm': arm, 'mean_cumulative_fed': info['mean_cumulative_fed'],
                         'cumulative_congestion': info['cumulative_congestion'], 'evacuation_rate': info['evacuation_rate'],
                         'incapacitation_rate': info['incapacitation_rate'], 'episode_end_seconds': info['step'] * 5.,
                         **{k: int(physical_violations[k]) for k in ('unavailable_entries', 'release_violations', 'direction_violations')}})
            env.close()
    csv_write(out / 'episodes.csv', rows)
    save(out / 'error_direction_audit.json', dict(counters))
    with (out / 'error_trace.jsonl').open('w', encoding='utf-8') as f:
        for r in detailed:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (FIRST / 'episodes.csv').open(encoding='utf-8-sig') as f:
        old = list(csv.DictReader(f))
    index = {(r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']), r['arm']): r for r in old}
    comparisons = []
    for arm in ('cached_llm_ppo', 'delayed_advice_ppo', 'wrong_advice_ppo'):
        base = [r for r in rows if r['arm'] == 'no_advice_same_ppo']
        for metric in ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate', 'episode_end_seconds'):
            differences = []
            cluster = defaultdict(list)
            for b in base:
                a = index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], arm)]
                d = float(a[metric]) - b[metric]
                differences.append(d)
                cluster[b['scenario_id']].append(d)
            scene_means = np.array([np.mean(cluster[k]) for k in sorted(cluster)])
            rng = np.random.default_rng(2026100711)
            boots = scene_means[rng.integers(0, len(scene_means), (10000, len(scene_means)))].mean(axis=1)
            comparisons.append({'candidate': arm, 'baseline': 'no_advice_same_ppo', 'metric': metric,
                                'mean_difference': float(np.mean(differences)), 'nonzero_pairs': int(np.count_nonzero(np.abs(differences) > 1e-12)),
                                'paired_episodes': len(base), 'scenario_clusters': 12,
                                'descriptive_scene_bootstrap_95_interval': list(map(float, np.quantile(boots, [.025, .975]))),
                                'not_final_inference_or_training_seed_uncertainty': True})
    summary = {'completed': len(rows), 'new_27b_calls': 0, 'new_ppo_training': 0, 'elapsed_seconds': time.monotonic()-started,
               'error_audit': dict(counters), 'same_checkpoint_comparisons': comparisons,
               'no_advice_means': {m: float(np.mean([r[m] for r in rows if r['arm'] == 'no_advice_same_ppo'])) for m in ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate', 'episode_end_seconds')}}
    save(out / 'summary.json', summary)
    lines = ['# 同一PPO权重对照与错误方向审计', '', '另行冻结后新增480回合；原1488回合保持封存。新27B调用0，新训练0。', '',
             '新对照使用与缓存LLM组完全相同的raw_llm_ppo权重、动作约束和人群种子，只去掉建议。', '',
             '|候选−相同权重无建议|指标|均差|描述性场景bootstrap 95%区间|非零配对|', '|---|---|---:|---|---:|']
    for r in comparisons:
        interval = r['descriptive_scene_bootstrap_95_interval']
        lines.append(f"|{r['candidate']}|{r['metric']}|{r['mean_difference']:.6f}|{interval[0]:.6f}, {interval[1]:.6f}|{r['nonzero_pairs']}/240|")
    lines += ['', 'bootstrap以12个复用开发场景为单位，条件于已有五组权重，不是最终资格结论或完整训练不确定性。', '',
              '更高烟气方向不一定是错误：若另一出口不可用，仅比较烟气不能认定应覆盖。本审计将这类情况标为歧义。', '',
              json.dumps(dict(counters), ensure_ascii=False, indent=2), '',
              '“rejected_contradiction_but_same_direction_executed”才表示实时矛盾已触发拒绝、但PPO仍执行同一方向。它说明门控拒绝和动态纠错是不同环节，仍不等同于反事实收益最差。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'previous_sources_unchanged': True,
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'completed': len(rows), 'output': str(out), 'error_audit': dict(counters), 'comparisons': comparisons[:5]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'run', 'verify'])
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify}[args.stage]()

