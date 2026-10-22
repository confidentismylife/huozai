"""Fresh population development check of a fixed direction-only adapter."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time
import numpy as np
from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from run_qwen14b_evacuations import FREEZE as MODEL_PREP, CACHE, cached_proposal, METRICS
from run_qwen14b_action_attribution import verify as verify_attribution, MODES
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, ppo_action
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, terminal_metrics
from fire_evac.advice_causal_audit import active_advice
from fire_evac.direction_hint_adapter import direction_hint_action

PREP = ROOT / 'outputs/direction_hint_fresh_population_preparation_20261007'
VARIANTS = ('no_advice', 'original_advice_adapter', 'direction_hint_only')


def verify():
    verify_attribution()
    if PREP.exists():
        for p, h in read(PREP / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p


def prepare():
    assert not PREP.exists()
    verify_attribution()
    old = read(MODEL_PREP / 'manifest.json')
    sources = [Path(__file__), ROOT / 'src/fire_evac/direction_hint_adapter.py', ROOT / 'tests/test_direction_hint_adapter.py',
        MODEL_PREP / 'manifest.json', CACHE / 'completion_manifest.json',
        ROOT / 'outputs/qwen14b_action_attribution_final_audit_20261007T052306160460Z/completion_manifest.json']
    PREP.mkdir()
    save(PREP / 'manifest.json', dict(scope='fresh_population_development_confirmation_not_new_scene_test',
        trigger='Most advice-induced changes adjust release volume; inspect a direction-only hint without advice input features or partial release-level bonuses.',
        scenarios=old['scenarios'], primary_scenario_ids=old['primary_scenario_ids'],
        population_seeds=[1477100000 + si * 10 + rep for si in range(24) for rep in range(2)],
        previous_population_seed_prefix=1467100000, new_people=True,
        behavior_models=old['behavior_models'], checkpoints=old['ppo_models'], training_seeds=old['training_seeds'],
        modes=MODES, variants=VARIANTS, full_episode_budget=2880, wall_seconds_cap=600,
        new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        adapter='Clear advice inputs to PPO; if accepted, add .25*confidence equally to every currently legal action on preferred direction. Preserve conditional release ranking. Same gate/TTL/mask.',
        no_hyperparameter_sweep=True,
        criteria='Report paired terminal incap/FED/congestion; do not adopt on lower FED with higher incap. No qualification or stable LLM benefit claim merely from beating the original adapter.',
        pairing='Same people and per-person/per-absolute-step uniforms across all variants; two new population seeds per scene.',
        limits='Scenarios and 14B caches reused after mechanism analysis; fresh people do not create independent new building scenarios.',
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    print('Frozen direction-only hint, 48 fresh populations and 2880 full episodes.', flush=True)


def run():
    import torch
    from stable_baselines3 import PPO
    verify()
    plan = read(PREP / 'manifest.json')
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['checkpoints'][str(s)], device='cpu') for s in plan['training_seeds']}
    out = unique_output('direction_hint_fresh_population_confirmation')
    save(out / 'run_manifest.json', dict(preparation_sha256=sha(PREP / 'manifest.json'), planned_full_episodes=2880,
                                      new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0))
    cases = [(si, b, rep, seed, mode, variant) for si in range(24) for b in plan['behavior_models']
             for rep in range(2) for seed in plan['training_seeds'] for mode in MODES for variant in VARIANTS]
    random.Random(2026100764).shuffle(cases)
    assert len(cases) == 2880
    rows = []
    started = time.monotonic()
    for si, behavior, rep, seed, mode, variant in cases:
        if time.monotonic() - started > 600:
            csv_write(out / 'partial_episodes.csv', rows)
            raise RuntimeError('Frozen confirmation time budget exhausted')
        scene = DynamicConflictScenario(**plan['scenarios'][si])
        actor = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
        initial = TimestampedControlEnv(actor, (scene,))
        popseed = 1477100000 + si * 10 + rep
        initial.reset(seed=popseed, options={'scenario': scene})
        env = clone_for_future(initial, popseed + 770000)
        initial.close()
        p = cached_proposal(read(CACHE / 'responses' / (scene.scenario_id + '.json')), env.snapshot, mode == MODES[1])
        counts = Counter()
        while True:
            active, gate, _ = active_advice(env, p)
            mask = env.constraints.action_mask(env.snapshot)
            base_action, _ = ppo_action(models[seed], env.policy_observation(), mask)
            if variant == 'no_advice':
                action = base_action
            elif variant == 'original_advice_adapter':
                action, _ = ppo_action(models[seed], env.policy_observation(active, gate), mask, active, gate,
                    elapsed_since_receipt=0 if active is None else env.snapshot.observation_time_s - active.received_time_s)
            else:
                action, _ = direction_hint_action(models[seed], env.policy_observation(), mask, active, gate)
            counts['action_changed_steps'] += int(action != base_action)
            same_dir_release_change = action != 8 and base_action != 8 and action // 4 == base_action // 4 and action % 4 != base_action % 4
            counts['same_direction_release_changed_steps'] += int(same_dir_release_change)
            if variant == 'direction_hint_only':
                assert not same_dir_release_change
            _, _, term, trunc, info = paired_step(env, action)
            counts['unavailable_entries'] += int(info['unavailable_corridor_entries'])
            counts['release_violations'] += int(not info['actual_release_within_limit'])
            counts['direction_violations'] += int(info['actual_direction_violation'])
            if term or trunc:
                break
        rows.append(dict(scenario_id=scene.scenario_id,
            cohort='primary_separate_development' if scene.scenario_id in plan['primary_scenario_ids'] else 'exploratory_mechanism_development',
            behavior=behavior, population_seed=popseed, training_seed=seed, source_mode=mode, variant=variant,
            **terminal_metrics(info, env.unwrapped.domain_env.config),
            **{k: int(counts[k]) for k in ('action_changed_steps', 'same_direction_release_changed_steps',
                                        'unavailable_entries', 'release_violations', 'direction_violations')}))
        env.close()
        if len(rows) % 400 == 0:
            print(json.dumps({'full_episodes': len(rows), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    csv_write(out / 'episodes.csv', rows)
    index = {(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['source_mode'], r['variant']): r for r in rows}
    effects = []
    for cohort in ('primary_separate_development', 'exploratory_mechanism_development', 'all_development'):
        for mode in MODES:
            for candidate, baseline in [('original_advice_adapter', 'no_advice'), ('direction_hint_only', 'no_advice'), ('direction_hint_only', 'original_advice_adapter')]:
                base = [r for r in rows if r['source_mode'] == mode and r['variant'] == baseline
                        and (cohort == 'all_development' or r['cohort'] == cohort)]
                for metric in METRICS:
                    scenes = defaultdict(list)
                    nonzero = 0
                    for b in base:
                        a = index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], mode, candidate)]
                        d = a[metric] - b[metric]
                        scenes[b['scenario_id']].append(d)
                        nonzero += abs(d) > 1e-12
                    values = np.array([np.mean(v) for v in scenes.values()])
                    rng = np.random.default_rng(2026100765)
                    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                    effects.append(dict(cohort=cohort, mode=mode, candidate=candidate, baseline=baseline, metric=metric,
                        paired_episodes=len(base), scenario_clusters=len(scenes), nonzero_pairs=int(nonzero),
                        mean_difference=float(values.mean()),
                        descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975])))))
    summary = dict(completed_full_episodes=len(rows), elapsed_seconds=time.monotonic() - started,
        paired_effects=effects, violations={k: sum(r[k] for r in rows) for k in ('unavailable_entries', 'release_violations', 'direction_violations')},
        direction_hint_same_direction_release_changes=sum(r['same_direction_release_changed_steps'] for r in rows if r['variant'] == 'direction_hint_only'),
        action_change_counts=[dict(cohort=cohort, mode=mode, variant=variant,
             action_changed_steps=sum(r['action_changed_steps'] for r in rows if r['cohort'] == cohort and r['source_mode'] == mode and r['variant'] == variant))
             for cohort in ('primary_separate_development', 'exploratory_mechanism_development') for mode in MODES for variant in VARIANTS],
        new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        fresh_populations_but_reused_scenes=True, no_qualification_claim=True)
    save(out / 'summary.json', summary)
    lines = ['# 仅出口方向提示适配器：新人口种子开发验证', '',
        '固定修正：不把建议字段输入PPO，不只给25%/50%候选加分；有效建议对目标方向全部合法动作等量加分，方向不变时放行档位不变。没有修改物理约束、TTL、门控或PPO权重。', '',
        '使用24个复用开发场景、48批新人口、两种行为模型、5套已有权重；即时和实测延迟模式；无建议、原适配器、新方向提示三个版本。共2880个完整回合。新人口不是新建筑场景，结果仍是开发验证。', '',
        '|主要12场景|候选−对照|指标|均差|描述性场景95%区间|非零配对|', '|---|---|---|---:|---|---:|']
    for r in effects:
        if r['cohort'] == 'primary_separate_development' and r['metric'] in ('mean_cumulative_fed', 'incapacitation_rate'):
            lo, hi = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['mode']}|{r['candidate']}−{r['baseline']}|{r['metric']}|{r['mean_difference']:+.7f}|{lo:.7f}, {hi:.7f}|{r['nonzero_pairs']}/{r['paired_episodes']}|")
    lines += ['', '率差乘100为百分点。比原适配器好并不自动意味着比无建议PPO好；两种对照都报告。没有提高建议强度或搜索超参数；没有拟合反事实分类器。', '',
        '全部结果、操作检查和动作变化记录在summary.json与episodes.csv。新LLM请求0、新训练0，保留最终场景未启用；没有修改全局资格状态。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'completed_full_episodes': len(rows),
        'primary_direction_hint_effects': [r for r in effects if r['cohort'] == 'primary_separate_development' and r['candidate'] == 'direction_hint_only' and r['metric'] in ('mean_cumulative_fed', 'incapacitation_rate')]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'verify'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify}[args.stage]()
