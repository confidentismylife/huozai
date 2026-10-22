"""Frozen finite mechanisms on development cases, with exact trajectory traces."""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import argparse
import copy
import csv
import json
import pickle
import random
import sys
import time

import numpy as np

from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from audit_direction_hint_full_cached_trace import verify_sources
from run_direction_hint_fresh_population_confirmation import PREP as OLD_PREP, MODES
from run_qwen14b_evacuations import CACHE, cached_proposal, METRICS
from fire_evac.config import EnvConfig
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, ppo_action
from fire_evac.direction_hint_adapter import direction_hint_action
from fire_evac.direction_release_mechanism import preserve_base_release, set_diagnostic_hold_zero, step_zone_accounting
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, terminal_metrics
from fire_evac.advice_causal_audit import active_advice, reseed_continuation, outcome_class
from fire_evac.cached_direction_diagnosis import joint_effect_cells, factor_effects

PREP = ROOT / 'outputs/direction_release_mechanism_preparation_20261007'
PREVIOUS = ROOT / 'outputs/direction_hint_fresh_population_confirmation_20261007T052903182504Z'
REVIEW = ROOT / 'outputs/direction_harm_cache_final_review_20261007T060152567789Z'
SCENES = ('AULL0010', 'AULL0013', 'AULL0019', 'AULL0020', 'AULL0024')
POLICIES = ('no_advice', 'direction_hint_only', 'preserve_release_hint')
ZONE_NAMES = ('room_a', 'room_b', 'junction', 'north_corridor', 'south_corridor')
FRESH_FUTURES = (2026100782, 2026100783, 2026100784)


def verify_completion(folder):
    for p, h in read(Path(folder) / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p


def verify():
    verify_sources()
    verify_completion(REVIEW)
    if PREP.exists():
        for p, h in read(PREP / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p


def prepare():
    assert not PREP.exists()
    verify()
    old = read(OLD_PREP / 'manifest.json')
    records = [dict(scene_index=i, scene=s) for i, s in enumerate(old['scenarios']) if s['scenario_id'] in SCENES]
    assert len(records) == 5
    files = [Path(__file__), ROOT / 'src/fire_evac/direction_release_mechanism.py',
        ROOT / 'tests/test_direction_release_mechanism.py', OLD_PREP / 'manifest.json',
        PREVIOUS / 'completion_manifest.json', REVIEW / 'completion_manifest.json']
    PREP.mkdir()
    save(PREP / 'manifest.json', dict(
        scope='Finite, outcome-informed development mechanisms; selected buildings reused, no independent final qualification.',
        scene_records=records, checkpoints=old['checkpoints'], training_seeds=old['training_seeds'],
        behaviors=old['behavior_models'], modes=MODES, policies=POLICIES,
        population_groups={'replay': 1477100000, 'fresh_frozen': 1487100000}, populations_per_scene=2,
        hold_settings={'replay': [2, 0], 'fresh_frozen': [2]},
        full_episodes=1920, exact_old_replay_episodes=400, full_wall_seconds_cap=900,
        counterfactual_terminal_cap=4000, counterfactual_wall_seconds_cap=1200, selected_state_cap=300,
        selection='Every accepted advice-induced action change on actual replay-population direction_hint_only paths at hold2; includes beneficial and harmful cases. All full policies get per-step traces.',
        snapshots='Offline copies of hidden people/history and paired random table; snapshots never become policy inputs. Preserve actual per-person/per-absolute-step random pairing.',
        full_pairing='Same population and population_seed+770000 future table for all policies and hold settings; rule episodes run once per setting, not once per checkpoint.',
        candidate='When direction hint changes a nonwait action, keep same-state no-advice PPO release level on hinted route; if impossible keep base. Wait transitions explicitly remain separate. No strength, TTL, gate or weight tuning.',
        rule_control='Shared current constraints.rule_action with exactly the same public sensors/forecast and action mask; no hidden source or terminal outcome.',
        hold='Hold0 changes both domain config and wrapper hold and is only a diagnostic. Compare each policy to no-advice at the same hold setting.',
        continuation='All current legal actions forced for one step then common no-advice PPO; two forced base/hint then resume original hint policy; forced base/hint under hold0 then common PPO.',
        futures='Rule: same realized future only. Calibrated: realized population future plus three frozen fresh future seeds; unwrap old random wrapper before reseeding.',
        fresh_future_seeds=FRESH_FUTURES,
        exact_checks='Replayed terminal metrics/counts match400 archived rows; realized forced-hint/resume-hint reproduces every suffix action, actuator, zone counts and terminal. Zone FED accounting conserves total FED.',
        analyses='Report paired full outcomes by population group/mode/hold against no-advice and original hint; target-specific values, capacity effects, direction effects and interactions. First changed state per episode is the descriptive primary CF subset; all changed states separately. Do not sum event effects.',
        criteria='Do not adopt merely for lower FED with increased incapacitation. Targeted fresh people are not fresh independent buildings; retain all outcome signs and behavioral variation. No controller qualification or final-bank opening.',
        new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        tests_passed_new=5, tests_passed_related_regression=7,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print('Frozen 1920 full episodes, exact replay400, targeted fresh people and bounded paired continuations.', flush=True)


def csv_read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def models_for(plan):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    return {s: PPO.load(plan['checkpoints'][str(s)], device='cpu') for s in plan['training_seeds']}


def decide(env, model, proposal, policy):
    active, gate, status = active_advice(env, proposal)
    mask = env.constraints.action_mask(env.snapshot)
    if policy == 'pure_rule':
        action = env.constraints.rule_action(env.snapshot)
        return action, action, action, 'rule', active, gate, status, mask
    obs = env.policy_observation()
    base, _ = ppo_action(model, obs, mask)
    hint, _ = direction_hint_action(model, obs, mask, active, gate)
    if policy == 'no_advice':
        action, matching = base, 'no_advice'
    elif policy == 'direction_hint_only':
        action, matching = hint, 'original_hint'
    else:
        action, matching = preserve_base_release(base, hint, mask)
    assert mask[action]
    return action, base, hint, matching, active, gate, status, mask


def new_env(scene, behavior, popseed, hold):
    actor = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
    initial = TimestampedControlEnv(actor, (scene,), config=replace(EnvConfig(), min_sign_hold_steps=hold))
    initial.reset(seed=popseed, options={'scenario': scene})
    env = clone_for_future(initial, popseed + 770000)
    initial.close()
    return env


def zone_fields(fed, person_steps, incap, config):
    return {**{f'fed_{name}': float(fed[i] / config.population) for i, name in enumerate(ZONE_NAMES)},
            **{f'person_seconds_{name}': float(person_steps[i] * config.dt_seconds / config.population) for i, name in enumerate(ZONE_NAMES)},
            **{f'incap_count_{name}': int(incap[i]) for i, name in enumerate(ZONE_NAMES)}}


def full_trials():
    verify()
    plan = read(PREP / 'manifest.json')
    models = models_for(plan)
    records = plan['scene_records']
    cases = []
    for group, prefix in plan['population_groups'].items():
        for record in records:
            for rep in range(2):
                pop = prefix + record['scene_index'] * 10 + rep
                for behavior in plan['behaviors']:
                    for mode in MODES:
                        for hold in plan['hold_settings'][group]:
                            for seed in plan['training_seeds']:
                                for policy in POLICIES:
                                    cases.append((group, record, pop, behavior, mode, hold, seed, policy))
                            cases.append((group, record, pop, behavior, mode, hold, 0, 'pure_rule'))
    assert len(cases) == 1920
    random.Random(2026100781).shuffle(cases)
    old = {(r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']), r['source_mode'], r['variant']): r
           for r in csv_read(PREVIOUS / 'episodes.csv')}
    out = unique_output('direction_release_mechanism_full')
    (out / 'offline_states').mkdir()
    rows, selected, checks = [], [], 0
    start = time.monotonic()
    with (out / 'actual_step_traces.jsonl').open('w', encoding='utf-8') as stream:
        for group, record, pop, behavior, mode, hold, seed, policy in cases:
            assert time.monotonic() - start < 900, 'Frozen full-episode budget exhausted'
            scene = DynamicConflictScenario(**record['scene'])
            env = new_env(scene, behavior, pop, hold)
            proposal = cached_proposal(read(CACHE / 'responses' / (scene.scenario_id + '.json')), env.snapshot, mode == MODES[1])
            case_id = '|'.join(map(str, (group, scene.scenario_id, behavior, pop, seed, mode, hold, policy)))
            counts = Counter()
            fed, person_steps, incapac = np.zeros(5), np.zeros(5), np.zeros(5, dtype=int)
            first_selected = True
            while True:
                action, base, hint, matching, active, gate, status, mask = decide(env, models.get(seed), proposal, policy)
                s = env.snapshot
                if group == 'replay' and hold == 2 and policy == 'direction_hint_only' and action != base:
                    assert gate is not None and gate.accepted
                    assert len(selected) < 300
                    sid = f'{case_id}|{s.step}'
                    path = out / 'offline_states' / f'{len(selected):04d}.pkl'
                    path.write_bytes(pickle.dumps({'env': copy.deepcopy(env), 'proposal': proposal}, protocol=pickle.HIGHEST_PROTOCOL))
                    selected.append(dict(state_id=sid, case_id=case_id, scenario_id=scene.scenario_id,
                        behavior=behavior, population_seed=pop, training_seed=seed, source_mode=mode,
                        source_step=s.step, first_changed_state=first_selected, base_action=base, hint_action=hint,
                        matched_release_action=preserve_base_release(base, hint, mask)[0],
                        legal_actions=list(map(int, np.flatnonzero(mask))), rule_action=env.constraints.rule_action(s),
                        observation=env.policy_observation().tolist(), public_snapshot=s.public_data(),
                        past_zone_fed=fed.tolist(), past_zone_person_steps=person_steps.tolist(),
                        past_zone_incapacitations=incapac.tolist(), snapshot_path=path.relative_to(ROOT).as_posix(), snapshot_sha256=sha(path)))
                    first_selected = False
                counts['action_changed_steps'] += action != base
                counts['same_direction_release_changed_steps'] += action != 8 and base != 8 and action // 4 == base // 4 and action % 4 != base % 4
                counts['release_preserve_wait_exceptions'] += matching == 'waiting_not_factorizable'
                counts['release_preserve_infeasible_fallbacks'] += matching == 'matched_release_infeasible_keep_base'
                obs = env.policy_observation().tolist()
                (result, account) = step_zone_accounting(env, action, paired_step)
                _, _, term, trunc, info = result
                fed += account['zone_fed_increments']
                person_steps += account['zone_person_steps']
                incapac += account['zone_new_incapacitations']
                counts['unavailable_entries'] += int(info['unavailable_corridor_entries'])
                counts['release_violations'] += int(not info['actual_release_within_limit'])
                counts['direction_violations'] += int(info['actual_direction_violation'])
                stream.write(json.dumps(dict(case_id=case_id, source_step=s.step, observation_time_s=s.observation_time_s,
                    observation=obs, public_snapshot=s.public_data(), gate_status=status,
                    gate_reasons=[] if gate is None else list(gate.reasons), base_action=base,
                    original_hint_action=hint, requested_action=action, release_matching=matching,
                    applied_action=int(info['constraint_applied_action']), actual_joint_action=int(info['actual_joint_action']),
                    actual_sign=int(info['actual_sign_action']), actual_room_capacity=int(info['room_release_capacity']),
                    actual_room_count=int(info['room_release_count']), branch_entries=env.unwrapped.domain_env.last_branch_flow.tolist(),
                    zone_counts_after=info['zone_counts'], **account), ensure_ascii=False) + '\n')
                if term or trunc:
                    break
            config = env.unwrapped.domain_env.config
            assert abs(fed.sum() / config.population - info['mean_cumulative_fed']) < 1e-9
            row = dict(case_id=case_id, population_group=group, scenario_id=scene.scenario_id,
                behavior=behavior, population_seed=pop, training_seed=seed, source_mode=mode, hold_steps=hold, policy=policy,
                **terminal_metrics(info, config), **zone_fields(fed, person_steps, incapac, config),
                **{k: int(counts[k]) for k in ('action_changed_steps', 'same_direction_release_changed_steps',
                    'release_preserve_wait_exceptions', 'release_preserve_infeasible_fallbacks',
                    'unavailable_entries', 'release_violations', 'direction_violations')})
            if group == 'replay' and hold == 2 and policy in ('no_advice', 'direction_hint_only'):
                ref = old[(scene.scenario_id, behavior, pop, seed, mode, policy)]
                for metric in (*METRICS, 'remaining', 'episode_end_seconds', 'action_changed_steps',
                               'same_direction_release_changed_steps', 'unavailable_entries', 'release_violations', 'direction_violations'):
                    assert abs(float(row[metric]) - float(ref[metric])) < 1e-12, (case_id, metric)
                expected_time = None if not ref['all_evacuated_seconds'] else float(ref['all_evacuated_seconds'])
                assert row['all_evacuated_seconds'] == expected_time
                checks += 1
            rows.append(row)
            env.close()
            if len(rows) % 128 == 0:
                print(json.dumps(dict(stage='full', completed=len(rows), target=1920,
                                     selected_states=len(selected), elapsed_seconds=round(time.monotonic()-start,1))), flush=True)
    assert checks == 400 and len(rows) == 1920
    csv_write(out / 'episodes.csv', rows)
    with (out / 'selected_states.jsonl').open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    save(out / 'summary.json', dict(preparation_sha256=sha(PREP / 'manifest.json'), full_episodes=1920,
        exact_old_replays=checks, selected_states=len(selected), elapsed_seconds=time.monotonic()-start,
        new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        violations={k: sum(r[k] for r in rows) for k in ('unavailable_entries','release_violations','direction_violations')}))
    verify()
    save(out / 'completion_manifest.json', dict(state='complete', files={
        p.relative_to(ROOT).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(stage='full_complete', output=str(out), episodes=1920, exact_old_replays=checks,
                         selected_states=len(selected))), flush=True)
    return out


def continuations(full):
    verify()
    full = Path(full).resolve()
    verify_completion(full)
    plan = read(PREP / 'manifest.json')
    assert read(full / 'summary.json')['preparation_sha256'] == sha(PREP / 'manifest.json')
    models = models_for(plan)
    with (full / 'selected_states.jsonl').open(encoding='utf-8') as f:
        selected = list(map(json.loads, f))
    parent = {r['case_id']: r for r in csv_read(full / 'episodes.csv')}
    needed_cases = {r['case_id'] for r in selected}
    expected_steps = defaultdict(dict)
    with (full / 'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads, f):
            if r['case_id'] in needed_cases:
                expected_steps[r['case_id']][r['source_step']] = r
    planned = sum((1 if r['behavior'] == 'rule' else 4) * (len(r['legal_actions']) + 4) for r in selected)
    assert planned <= 4000
    out = unique_output('direction_release_mechanism_continuations')
    rows, exact_suffixes, exact_steps = [], 0, 0
    start = time.monotonic()
    for state in selected:
        path = ROOT / state['snapshot_path']
        assert sha(path) == state['snapshot_sha256']
        source = pickle.loads(path.read_bytes())
        futures = [(state['population_seed'] + 770000, 'realized')]
        if state['behavior'] != 'rule':
            futures += [(seed, 'fresh_future') for seed in FRESH_FUTURES]
        for seed, future_kind in futures:
            template = reseed_continuation(source['env'], seed)
            variants = [(a, 'no_advice', 2) for a in state['legal_actions']]
            variants += [(a, 'direction_hint_only', 2) for a in (state['base_action'], state['hint_action'])]
            variants += [(a, 'no_advice', 0) for a in (state['base_action'], state['hint_action'])]
            for forced, future_policy, hold in variants:
                assert time.monotonic() - start < 1200 and len(rows) < 4000
                env = copy.deepcopy(template)
                if hold == 0:
                    set_diagnostic_hold_zero(env)
                fed = np.array(state['past_zone_fed'], dtype=float)
                people_time = np.array(state['past_zone_person_steps'], dtype=float)
                incap = np.array(state['past_zone_incapacitations'], dtype=int)
                counts = Counter()
                first = True
                check = future_kind == 'realized' and hold == 2 and future_policy == 'direction_hint_only' and forced == state['hint_action']
                while True:
                    s = env.snapshot
                    action = int(forced) if first else decide(env, models[state['training_seed']], source['proposal'], future_policy)[0]
                    assert env.constraints.action_mask(s)[action]
                    result, account = step_zone_accounting(env, action, paired_step)
                    _, _, term, trunc, info = result
                    fed += account['zone_fed_increments']
                    people_time += account['zone_person_steps']
                    incap += account['zone_new_incapacitations']
                    counts['unavailable_entries'] += int(info['unavailable_corridor_entries'])
                    counts['release_violations'] += int(not info['actual_release_within_limit'])
                    counts['direction_violations'] += int(info['actual_direction_violation'])
                    if check:
                        expected = expected_steps[state['case_id']][s.step]
                        assert action == expected['requested_action']
                        for k, field in [('actual_joint_action','actual_joint_action'), ('actual_sign_action','actual_sign'),
                                         ('room_release_capacity','actual_room_capacity'), ('room_release_count','actual_room_count')]:
                            assert int(info[k]) == expected[field], (state['state_id'], s.step, k)
                        assert list(info['zone_counts']) == expected['zone_counts_after']
                        assert np.allclose(account['zone_fed_increments'], expected['zone_fed_increments'], atol=1e-12, rtol=0)
                        exact_steps += 1
                    first = False
                    if term or trunc:
                        break
                config = env.unwrapped.domain_env.config
                assert abs(fed.sum() / config.population - info['mean_cumulative_fed']) < 1e-9
                metrics = terminal_metrics(info, config)
                if check:
                    for m in (*METRICS, 'episode_end_seconds', *[f'fed_{n}' for n in ZONE_NAMES]):
                        value = metrics[m] if m in metrics else fed[ZONE_NAMES.index(m[4:])] / config.population
                        assert abs(value - float(parent[state['case_id']][m])) < 1e-12
                    exact_suffixes += 1
                rows.append(dict(state_id=state['state_id'], case_id=state['case_id'], scenario_id=state['scenario_id'],
                    behavior=state['behavior'], population_seed=state['population_seed'], training_seed=state['training_seed'],
                    source_mode=state['source_mode'], source_step=state['source_step'], first_changed_state=state['first_changed_state'],
                    future_seed=seed, future_kind=future_kind, forced_action=forced, future_policy=future_policy, hold_steps=hold,
                    **metrics, **zone_fields(fed, people_time, incap, config),
                    **{k: int(counts[k]) for k in ('unavailable_entries','release_violations','direction_violations')}))
                env.close()
                if len(rows) % 200 == 0:
                    print(json.dumps(dict(stage='counterfactual', completed=len(rows), planned=planned,
                                         elapsed_seconds=round(time.monotonic()-start,1))), flush=True)
    assert len(rows) == planned and exact_suffixes == len(selected)
    csv_write(out / 'terminal_rollouts.csv', rows)
    save(out / 'summary.json', dict(full_source=str(full), full_source_manifest_sha256=sha(full / 'completion_manifest.json'),
        terminal_rollouts=len(rows), exact_realized_suffixes=exact_suffixes, exact_suffix_control_steps=exact_steps,
        elapsed_seconds=time.monotonic()-start, new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        violations={k: sum(r[k] for r in rows) for k in ('unavailable_entries','release_violations','direction_violations')}))
    verify()
    save(out / 'completion_manifest.json', dict(state='complete', files={p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(stage='continuations_complete', output=str(out), terminal_rollouts=len(rows),
                         exact_realized_suffixes=exact_suffixes)), flush=True)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','run','continuations','verify'))
    parser.add_argument('--full')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'verify':
        verify()
    elif args.stage == 'continuations':
        assert args.full
        continuations(args.full)
    else:
        full = full_trials()
        continuations(full)
