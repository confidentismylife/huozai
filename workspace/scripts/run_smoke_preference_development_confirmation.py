"""Full-episode development check on a separate, frozen AULL subset."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import sys
import time
import random
import numpy as np

from run_timestamped_development_campaign import (ROOT, PREP, OLD, read, save, sha,
    csv_write, unique_output, payload_for)
from run_direction_counterfactual_audit import verify as verify_direction
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, CommonActionConstraints, Proposal, ppo_action
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, terminal_metrics
from fire_evac.smoke_preference_probe import SmokePreferenceProbeConstraints

FREEZE = ROOT / 'outputs/smoke_preference_development_confirmation_preparation_20261007'
CONTROLLERS = ('pure_rule', 'same_weight_ppo', 'cached_llm_same_weight_ppo')
METRICS = ('mean_cumulative_fed', 'incapacitation_rate', 'evacuation_rate', 'cumulative_congestion')


def verify():
    verify_direction()
    if FREEZE.exists():
        for p, digest in read(FREEZE / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == digest, p


def prepare():
    assert not FREEZE.exists()
    verify_direction()
    excluded = {r['scenario_id'] for r in read(PREP / 'manifest.json')['scenarios']}
    groups = defaultdict(list)
    for s in read(OLD / 'manifest.json')['banks']['llm']:
        if s['scenario_id'] not in excluded:
            groups[s['conflict_type']].append(s)
    scenes = [s for name in sorted(groups) for s in sorted(groups[name], key=lambda r: r['scenario_id'])[:2]]
    assert len(scenes) == 12 and len(groups) == 6
    files = [Path(__file__), ROOT / 'src/fire_evac/smoke_preference_probe.py', ROOT / 'tests/test_smoke_preference_probe.py',
        ROOT / 'src/fire_evac/counterfactual_direction_audit.py', PREP / 'manifest.json', OLD / 'manifest.json',
        ROOT / 'outputs/behavior_pipeline/behavior_model.json',
        ROOT / 'outputs/direction_counterfactual_final_audit_20261007T032154325639Z/completion_manifest.json',
        ROOT / 'outputs/local_llm_selection.json', ROOT / 'outputs/third_round_development_status.json']
    files += [OLD / 'llm_cache' / (s['scenario_id'] + '.json') for s in scenes]
    models = read(PREP / 'manifest.json')['ppo_models']
    files += [Path(models[str(seed)]['raw_llm_ppo']) for seed in range(601, 606)]
    FREEZE.mkdir()
    save(FREEZE / 'manifest.json', dict(scope='separate_development_confirmation_subset', scenes=scenes,
        excluded_direction_audit_scenes=sorted(excluded),
        warning='These AULL scenarios already existed in older development work. They are separate from the 596-state analysis, not pristine final-test scenes.',
        behaviors=['rule', 'calibrated_logit'], population_seeds=[1467100000 + si * 10 + rep for si in range(12) for rep in range(2)],
        training_seeds=list(range(601, 606)), controllers=CONTROLLERS,
        constraint_arms=['current_operational_constraints', 'experimental_smoke_preference'],
        completed_episode_cap=1056, wall_seconds_cap=600,
        ppo_checkpoints={str(seed): models[str(seed)]['raw_llm_ppo'] for seed in range(601, 606)},
        cached_llm_origin='Oct5 Qwen3 4B records; no fresh inference; one initial timestamped proposal, expires after 15s.',
        paired_randomness='Same people and per-person/per-absolute-step uniform table in both constraints and both same-weight PPO arms.',
        stopping_rule='Smoke-only rule is not adopted. If FED or incapacitation worsen on scene averages, reject it as an assumed correction method; no qualification implied even if it improves.',
        new_training=0, llm_requests=0, reserved_final_scenarios_opened=0,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print('Frozen 12 separate development scenarios and 1056 full episodes.', flush=True)


def run():
    import torch
    from stable_baselines3 import PPO
    verify()
    plan = read(FREEZE / 'manifest.json')
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['ppo_checkpoints'][str(s)], device='cpu') for s in plan['training_seeds']}
    out = unique_output('smoke_preference_development_confirmation')
    save(out / 'run_manifest.json', {'preparation_sha256': sha(FREEZE / 'manifest.json'),
                                   'planned_full_episodes': 1056, 'new_training': 0, 'llm_requests': 0})
    schedule = [(si, behavior, rep, seed, ctrl, variant)
        for si in range(12) for behavior in plan['behaviors'] for rep in range(2)
        for ctrl in CONTROLLERS for seed in ([0] if ctrl == 'pure_rule' else plan['training_seeds'])
        for variant in plan['constraint_arms']]
    random.Random(2026100734).shuffle(schedule)
    assert len(schedule) == 1056
    started = time.monotonic()
    rows = []
    for si, behavior, rep, seed, ctrl, variant in schedule:
        if time.monotonic() - started > plan['wall_seconds_cap']:
            csv_write(out / 'partial_episodes.csv', rows)
            raise RuntimeError('Frozen time budget exhausted; retain completed episodes.')
        scene = DynamicConflictScenario(**plan['scenes'][si])
        actor = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
        constraints = CommonActionConstraints() if variant == 'current_operational_constraints' else SmokePreferenceProbeConstraints()
        original_env = TimestampedControlEnv(actor, (scene,), constraints=constraints)
        popseed = 1467100000 + si * 10 + rep
        original_env.reset(seed=popseed, options={'scenario': scene})
        env = clone_for_future(original_env, popseed + 770000)
        original_env.close()
        initial = env.snapshot
        p = None
        if ctrl == 'cached_llm_same_weight_ppo':
            cache = read(OLD / 'llm_cache' / (scene.scenario_id + '.json'))
            p = Proposal.parse(payload_for(initial, cache['advice']['target_branch'], cache['advice']['confidence']),
                               initial, 0., 'confirmation-cache', 'cached_qwen3_4b')
        counters = Counter()
        while True:
            s = env.snapshot
            gate = None if p is None else env.constraints.validate_proposal(p, s, s.observation_time_s)
            mask = env.constraints.action_mask(s)
            original_mask = CommonActionConstraints().action_mask(s)
            counters['smoke_preference_extra_masked_actions'] += int(np.sum(original_mask & ~mask))
            counters['advice_accepted_steps'] += int(gate is not None and gate.accepted)
            if ctrl == 'pure_rule':
                action = env.constraints.rule_action(s)
            else:
                action, raw = ppo_action(models[seed], env.policy_observation(p, gate), mask, p, gate,
                    elapsed_since_receipt=0 if p is None else s.observation_time_s - p.received_time_s)
                counters['raw_ppo_masked_steps'] += int(not mask[raw])
            _, _, term, trunc, info = paired_step(env, action)
            counters['unavailable_entries'] += int(info['unavailable_corridor_entries'])
            counters['release_violations'] += int(not info['actual_release_within_limit'])
            counters['direction_violations'] += int(info['actual_direction_violation'])
            counters['steps'] += 1
            if term or trunc:
                break
        rows.append(dict(scenario_id=scene.scenario_id, conflict_type=scene.conflict_type,
                         behavior=behavior, population_seed=popseed, training_seed=seed,
                         controller=ctrl, constraints=variant,
                         **terminal_metrics(info, env.unwrapped.domain_env.config),
                         **{k: int(counters[k]) for k in ('smoke_preference_extra_masked_actions', 'advice_accepted_steps',
                             'raw_ppo_masked_steps', 'unavailable_entries', 'release_violations', 'direction_violations', 'steps')}))
        env.close()
        if len(rows) % 200 == 0:
            print(json.dumps({'episodes': len(rows), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    csv_write(out / 'episodes.csv', rows)
    assert len(rows) == 1056
    index = {(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['controller'], r['constraints']): r for r in rows}
    effects = []
    for ctrl in CONTROLLERS:
        for behavior in ('all', 'rule', 'calibrated_logit'):
            base = [r for r in rows if r['controller'] == ctrl and r['constraints'] == 'current_operational_constraints'
                    and (behavior == 'all' or r['behavior'] == behavior)]
            for metric in METRICS:
                cluster = defaultdict(list)
                for b in base:
                    a = index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], ctrl, 'experimental_smoke_preference')]
                    cluster[b['scenario_id']].append(a[metric] - b[metric])
                values = np.array([np.mean(v) for v in cluster.values()])
                rng = np.random.default_rng(2026100735)
                boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                effects.append(dict(controller=ctrl, behavior=behavior, metric=metric, paired_episodes=len(base),
                    scenario_clusters=len(cluster), mean_difference=float(values.mean()),
                    descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975]))),
                    scene_means={k: float(np.mean(v)) for k, v in cluster.items()}))
    advice_effects = []
    for variant in plan['constraint_arms']:
        base = [r for r in rows if r['controller'] == 'same_weight_ppo' and r['constraints'] == variant]
        for metric in METRICS:
            cluster = defaultdict(list)
            for b in base:
                a = index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], 'cached_llm_same_weight_ppo', variant)]
                cluster[b['scenario_id']].append(a[metric] - b[metric])
            values = np.array([np.mean(v) for v in cluster.values()])
            rng = np.random.default_rng(2026100736)
            boots = values[rng.integers(0, 12, (10000, 12))].mean(axis=1)
            advice_effects.append(dict(constraints=variant, metric=metric, paired_episodes=len(base),
                scenario_clusters=12, mean_difference=float(values.mean()),
                descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boots, [.025, .975])))))
    means = [dict(controller=ctrl, constraints=variant, episodes=len(selected),
                  **{m: float(np.mean([r[m] for r in selected])) for m in METRICS})
             for ctrl in CONTROLLERS for variant in plan['constraint_arms']
             for selected in [[r for r in rows if r['controller'] == ctrl and r['constraints'] == variant]]]
    violations = {k: sum(r[k] for r in rows) for k in ('unavailable_entries', 'release_violations', 'direction_violations')}
    save(out / 'summary.json', dict(completed_full_episodes=len(rows), elapsed_seconds=time.monotonic() - started,
        means=means, constraint_effects=effects, same_checkpoint_advice_effects=advice_effects,
        violations=violations, development_only=True, fresh_llm_requests=0, new_training=0,
        reserved_final_scenarios_opened=0, original_frozen_sources_unchanged=True))
    lines = ['# 低烟气优先规则：另行冻结开发子集的完整回合验证', '',
        '共1056个完整回合；12个场景未参与此前596状态分析，但都来自既有AULL开发库，不是首次使用的最终测试库。原始最终库继续封存。', '',
        '两个约束版本都应用于纯规则、同权重无建议PPO、缓存LLM＋相同权重PPO。烟气优先版本仅在低烟气方向有合法动作时排除高烟气方向，保留所有物理限制和指示牌保持规则；目前仅是实验探针，没有采用为新安全盾。', '',
        '相同人群种子、5组既有权重、两种行为模型；未来随机数按同一人同一时刻配对。缓存LLM来自Qwen3 4B，初始提案只发一次，15秒后过期；不代表新27B效果。', '',
        '|控制器|指标|完整配对|烟气优先−原约束均差|场景描述性95%区间|', '|---|---|---:|---:|---|']
    for r in effects:
        if r['behavior'] == 'all':
            ci = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['controller']}|{r['metric']}|{r['paired_episodes']}|{r['mean_difference']:.7f}|{ci[0]:.7f}, {ci[1]:.7f}|")
    lines += ['', '## 同权重的建议收益', '',
              '|约束|指标|缓存建议−无建议均差|描述性95%区间|', '|---|---|---:|---|']
    for r in advice_effects:
        ci = r['descriptive_scene_bootstrap_95_interval']
        lines.append(f"|{r['constraints']}|{r['metric']}|{r['mean_difference']:.7f}|{ci[0]:.7f}, {ci[1]:.7f}|")
    lines += ['', '原始回合数据与分行为结果见episodes.csv和summary.json。所有结论仍属于复用开发数据的验证；不改变正式第三轮训练或资格状态。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'completed': len(rows), 'violations': violations,
                      'effects': [r for r in effects if r['behavior'] == 'all' and r['metric'] in METRICS[:2]]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'verify'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify}[args.stage]()
