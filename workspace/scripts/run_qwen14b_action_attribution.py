"""Frozen replay of advice influence and paired legal-action continuations."""
from pathlib import Path
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import random
import sys
import time
import argparse
import numpy as np

from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from run_qwen14b_evacuations import FREEZE as OLD_PREP, CACHE, cached_proposal, verify_cache, METRICS
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, ppo_action
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, terminal_metrics
from fire_evac.advice_causal_audit import (active_advice, action_variants, reseed_continuation,
                                        outcome_class, ranked_outcomes, actuator_signature)

PREPARATION = ROOT / 'outputs/qwen14b_action_attribution_preparation_20261007'
PREVIOUS = ROOT / 'outputs/qwen14b_evacuation_comparison_20261007T033736587283Z'
MODES = ('qwen14b_instant_cache', 'qwen14b_measured_arrival_cache')
FUTURES = [2026100751, 2026100752, 2026100753, 2026100754]


def verify():
    verify_cache()
    for p, h in read(PREVIOUS / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p
    if PREPARATION.exists():
        for p, h in read(PREPARATION / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p
        for p, h in read(PREPARATION / 'preparation_manifest.json')['files'].items():
            assert sha(ROOT / p) == h, p


def key(rec):
    return (rec['scenario_id'], rec['behavior'], int(rec['population_seed']), int(rec['training_seed']), rec['arm'])


def identifier(rec):
    return '|'.join(map(str, (*key(rec), int(rec['step']))))


def fingerprint(env):
    d = env.unwrapped.domain_env
    payload = dict(observation=env.latest_observation.tolist(), step=d.step_index,
        people=[vars(p) for p in d.people], zones=d.zones, evacuated=d.evacuated,
        incapacitated=d.incapacitated, cumulative_fed=d.cumulative_fed,
        congestion=d.cumulative_congestion, sign=d.last_action, steps_since_switch=d.steps_since_switch,
        branch_flow=d.last_branch_flow.tolist(), wrapper_hold=env._hold_remaining)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare():
    assert not PREPARATION.exists()
    verify()
    sources = [Path(__file__), ROOT / 'src/fire_evac/advice_causal_audit.py', ROOT / 'tests/test_advice_causal_audit.py',
        OLD_PREP / 'manifest.json', CACHE / 'completion_manifest.json', PREVIOUS / 'completion_manifest.json',
        ROOT / 'outputs/qwen14b_evacuation_final_audit_20261007T033830514618Z/completion_manifest.json',
        ROOT / 'outputs/local_llm_selection.json', ROOT / 'outputs/third_round_development_status.json']
    PREPARATION.mkdir()
    save(PREPARATION / 'manifest.json', dict(scope='development_action_attribution_and_candidate_diagnostics',
        replay_full_episodes=960, modes=MODES, checkpoints_frozen=True,
        selection='At most two states per episode: first step when the response is available (including rejected responses), and first gate-accepted step where advised action differs from same-state no-advice PPO. Merge coincident selections.',
        primary_estimand='First available response states in the prior 12-scene primary development cohort. First-change states are deliberately selected exploratory diagnostics.',
        exact_replay='Match original terminal metrics and all recorded gate/count fields. Paired full-episode future seed population_seed+770000.',
        feature_decomposition='Same-state no-advice action, advice-feature-only action with prior strength0, and original full advice action with prior .25; descriptive decomposition, no fitted model.',
        actions='Enumerate every currently legal one-step action in the frozen nine-action mask. Forced action for one step, then identical no-advice PPO to termination for all candidates.',
        direction_comparison='Current 14B-guided PPO action; rule action; 14B target at no-advice release level when executable; best outcome among legal actions on the 14B target direction is an offline upper bound.',
        futures=FUTURES, deterministic_rule_future_count=1, calibrated_future_count=4,
        future_pairing='Fresh per-person/per-absolute-step uniforms after copying current people/history. Unwrap an existing paired behavior before replacing the table.',
        target_state_cap=1920, terminal_rollout_cap=45000, replay_wall_seconds_cap=600, continuation_wall_seconds_cap=1800,
        outcome_class='Expected incapacitation difference <=-.001 is beneficial, >=+.001 harmful; otherwise FED difference <=-.001 beneficial, >=+.001 harmful; rest small_or_neutral. Incapacitation priority, no real-world dose threshold claim.',
        ranking='Offline rank by expected incapacitation, then FED, congestion and action ID. Also report every candidate and future-label reversals; no oracle fed into control.',
        statistics='Average futures within state, states within episode, episodes within scenario; descriptive scenario bootstrap conditional on the existing five checkpoints.',
        no_new_llm_requests=True, new_training=0, reserved_final_scenarios_opened=0,
        do_not_adopt_any_counterfactual_best_action_online=True,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    save(PREPARATION / 'preparation_manifest.json', {'files': {(PREPARATION / 'manifest.json').relative_to(ROOT).as_posix(): sha(PREPARATION / 'manifest.json')}})
    print('Frozen 960 replays and legal-action counterfactual protocol.', flush=True)


def load_setup():
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    plan = read(OLD_PREP / 'manifest.json')
    models = {s: PPO.load(plan['ppo_models'][str(s)], device='cpu') for s in plan['training_seeds']}
    with (PREVIOUS / 'episodes.csv').open(encoding='utf-8-sig') as f:
        old = {key(r): r for r in csv.DictReader(f) if r['arm'] in MODES}
    assert len(old) == 960
    return plan, models, old


def make_env(plan, si, behavior, rep, mode):
    scene = DynamicConflictScenario(**plan['scenarios'][si])
    actor = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
    original = TimestampedControlEnv(actor, (scene,))
    popseed = 1467100000 + si * 10 + rep
    original.reset(seed=popseed, options={'scenario': scene})
    env = clone_for_future(original, popseed + 770000)
    original.close()
    cached = read(CACHE / 'responses' / (scene.scenario_id + '.json'))
    p = cached_proposal(cached, env.snapshot, mode == MODES[1])
    return env, p, popseed


def schedule(plan):
    return [(si, behavior, rep, seed, mode) for si in range(24) for behavior in plan['behavior_models']
            for rep in range(2) for seed in plan['training_seeds'] for mode in MODES]


def replay():
    verify()
    plan, models, old = load_setup()
    out = unique_output('qwen14b_action_replay')
    selected, counters, episode_rows = [], defaultdict(Counter), []
    started = time.monotonic()
    steps = one_step_checks = 0
    with (out / 'action_trace.jsonl').open('w', encoding='utf-8') as stream:
        for si, behavior, rep, seed, mode in schedule(plan):
            if time.monotonic() - started > 600:
                raise RuntimeError('Frozen replay time budget exhausted')
            env, p, popseed = make_env(plan, si, behavior, rep, mode)
            scene_id = plan['scenarios'][si]['scenario_id']
            cohort = 'primary_separate_development' if scene_id in plan['primary_scenario_ids'] else 'exploratory_mechanism_development'
            counts = Counter()
            first_available = first_changed = False
            while True:
                s = env.snapshot
                active, gate, status = active_advice(env, p)
                acts = action_variants(env, models[seed], active, gate)
                mask = env.constraints.action_mask(s)
                legal = list(map(int, np.flatnonzero(mask)))
                accepted = gate is not None and gate.accepted
                changed = acts['with_advice'] != acts['no_advice']
                reasons = []
                if active is not None and not first_available:
                    reasons.append('first_response_available')
                    first_available = True
                if accepted and changed and not first_changed:
                    reasons.append('first_action_change')
                    first_changed = True
                rec = dict(scenario_id=scene_id, cohort=cohort, behavior=behavior, population_seed=popseed,
                    training_seed=seed, arm=mode, step=s.step, observation_time_s=s.observation_time_s,
                    gate_status=status, gate_accepted=bool(accepted), gate_reasons=[] if gate is None else list(gate.reasons),
                    proposal_time_s=None if p is None else p.observation_time_s,
                    response_time_s=None if p is None else p.received_time_s,
                    llm_target=None if p is None else p.preferred_branch,
                    legal_actions=legal, smoke=list(s.smoke), occupancy=list(s.occupancy),
                    capacities=list(s.capacities), sign=s.sign, hold_remaining=s.sign_hold_remaining,
                    action_no_advice=acts['no_advice'], action_features_only=acts['features_only'],
                    action_with_advice=acts['with_advice'], rule_action=env.constraints.rule_action(s),
                    raw_no_advice=acts['raw_no_advice'], raw_with_advice=acts['raw_with_advice'],
                    action_changed=changed, feature_changes_action=acts['features_only'] != acts['no_advice'],
                    prior_changes_action=acts['with_advice'] != acts['features_only'])
                if reasons:
                    rec['state_id'] = identifier(rec)
                    rec['selection_reasons'] = reasons
                    rec['state_fingerprint'] = fingerprint(env)
                    rec['target_direction_actions'] = [a for a in legal if a != 8 and p is not None and a // 4 == p.preferred_branch]
                    a = None if p is None or acts['no_advice'] == 8 else p.preferred_branch * 4 + acts['no_advice'] % 4
                    rec['matched_target_action'] = a if a in legal else None
                    selected.append(dict(rec))
                shadow = None
                if changed:
                    shadow = copy.deepcopy(env)
                    _, _, _, _, shadow_info = paired_step(shadow, acts['no_advice'])
                    one_step_checks += 1
                _, _, term, trunc, info = paired_step(env, acts['with_advice'])
                actuator_changed = changed and actuator_signature(info) != actuator_signature(shadow_info)
                flow_changed = changed and (info['room_release_count'] != shadow_info['room_release_count'] or
                    not np.array_equal(env.unwrapped.domain_env.last_branch_flow, shadow.unwrapped.domain_env.last_branch_flow))
                assignment_changed = changed and env.unwrapped.domain_env.zones != shadow.unwrapped.domain_env.zones
                rec.update(actual_joint_action=int(info['actual_joint_action']), actual_sign=int(info['actual_sign_action']),
                           actual_room_count=int(info['room_release_count']), actual_room_capacity=int(info['room_release_capacity']),
                           branch_entries=list(map(int, env.unwrapped.domain_env.last_branch_flow)),
                           actuator_changed=bool(actuator_changed), flow_changed=bool(flow_changed),
                           person_assignment_changed=bool(assignment_changed))
                if shadow is not None:
                    rec.update(no_advice_actual_joint_action=int(shadow_info['actual_joint_action']),
                               no_advice_actual_sign=int(shadow_info['actual_sign_action']),
                               no_advice_room_count=int(shadow_info['room_release_count']))
                    shadow.close()
                stream.write(json.dumps(rec, ensure_ascii=False) + '\n')
                counts['steps'] += 1
                counts['pending_steps'] += status == 'pending'
                counts['advice_accepted_steps'] += accepted
                counts['advice_rejected_steps'] += gate is not None and not accepted
                if gate is not None:
                    for reason in gate.reasons:
                        counts['reason_' + reason] += 1
                counts['raw_masked_steps'] += not mask[acts['raw_with_advice']]
                for name, flag in [('action_changed_steps', changed), ('feature_changes_action_steps', rec['feature_changes_action']),
                    ('prior_changes_action_steps', rec['prior_changes_action']), ('actuator_changed_steps', actuator_changed),
                    ('flow_changed_steps', flow_changed), ('person_assignment_changed_steps', assignment_changed),
                    ('accepted_sign_agrees_with_target_steps', accepted and acts['with_advice'] != 8 and acts['with_advice'] // 4 == p.preferred_branch),
                    ('accepted_nonwait_steps', accepted and acts['with_advice'] != 8)]:
                    counts[name] += int(flag)
                counts['unavailable_entries'] += int(info['unavailable_corridor_entries'])
                counts['release_violations'] += int(not info['actual_release_within_limit'])
                counts['direction_violations'] += int(info['actual_direction_violation'])
                steps += 1
                if term or trunc:
                    break
            k = (scene_id, behavior, popseed, seed, mode)
            expected = old[k]
            assert all(abs(float(info[m]) - float(expected[m])) < 1e-12 for m in METRICS), k
            for name in ('steps', 'pending_steps', 'advice_accepted_steps', 'advice_rejected_steps', 'reason_stale',
                         'reason_timeout', 'reason_live_smoke_contradiction', 'raw_masked_steps',
                         'unavailable_entries', 'release_violations', 'direction_violations'):
                assert int(counts[name]) == int(expected[name]), (k, name)
            for group in ((cohort, mode), ('all_development', mode)):
                counters[group].update(counts)
                counters[group]['episodes'] += 1
                counters[group]['episodes_with_action_change'] += counts['action_changed_steps'] > 0
                counters[group]['episodes_with_accepted_advice'] += counts['advice_accepted_steps'] > 0
            episode_rows.append(dict(scenario_id=scene_id, cohort=cohort, behavior=behavior, population_seed=popseed,
                training_seed=seed, arm=mode, **dict(counts), **terminal_metrics(info, env.unwrapped.domain_env.config)))
            env.close()
    assert len(episode_rows) == 960 and len(selected) <= 1920
    assert len({r['state_id'] for r in selected}) == len(selected)
    with (out / 'selected_states.jsonl').open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # Normalize counters to keep episode CSV rectangular.
    names = sorted({name for r in episode_rows for name in r})
    normalized = [{name: r.get(name, 0) for name in names} for r in episode_rows]
    csv_write(out / 'episodes.csv', normalized)
    expected_rollouts = sum(len(r['legal_actions']) * (1 if r['behavior'] == 'rule' else 4) for r in selected)
    assert expected_rollouts <= 45000
    summary = dict(replayed_full_episodes=960, replayed_control_steps=steps, selected_states=len(selected),
        one_step_action_checks=one_step_checks, planned_terminal_rollouts=expected_rollouts,
        selected_first_response_states=sum('first_response_available' in r['selection_reasons'] for r in selected),
        selected_first_action_change_states=sum('first_action_change' in r['selection_reasons'] for r in selected),
        exact_terminal_and_counter_parity=True, elapsed_seconds=time.monotonic() - started,
        groups=[dict(cohort=k[0], arm=k[1], **dict(v)) for k, v in counters.items()],
        new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0)
    save(out / 'summary.json', summary)
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), **{k:v for k,v in summary.items() if k != 'groups'}}, ensure_ascii=False), flush=True)


def verify_replay(path):
    replay_path = Path(path).resolve()
    for p, h in read(replay_path / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p
    return replay_path


def continuations(replay_path):
    verify()
    replay_path = verify_replay(replay_path)
    with (replay_path / 'selected_states.jsonl').open(encoding='utf-8') as f:
        selected = {r['state_id']: r for r in map(json.loads, f)}
    expected = read(replay_path / 'summary.json')['planned_terminal_rollouts']
    plan, models, _ = load_setup()
    out = unique_output('qwen14b_legal_action_continuations')
    save(out / 'run_manifest.json', dict(preparation_sha256=sha(PREPARATION / 'manifest.json'),
        replay_completion_sha256=sha(replay_path / 'completion_manifest.json'), replay_path=str(replay_path),
        selected_states=len(selected), terminal_rollouts=expected, future_seeds=FUTURES,
        future_controller='same no-advice PPO after a single forced legal action',
        new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0))
    count = state_count = continuation_steps = 0
    started = time.monotonic()
    writer = None
    with (out / 'terminal_rollouts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        for si, behavior, rep, seed, mode in schedule(plan):
            env, p, popseed = make_env(plan, si, behavior, rep, mode)
            scene_id = plan['scenarios'][si]['scenario_id']
            while True:
                active, gate, status = active_advice(env, p)
                sid = '|'.join(map(str, (scene_id, behavior, popseed, seed, mode, env.snapshot.step)))
                if sid in selected:
                    rec = selected[sid]
                    assert fingerprint(env) == rec['state_fingerprint'], sid
                    assert list(map(int, np.flatnonzero(env.constraints.action_mask(env.snapshot)))) == rec['legal_actions']
                    futures = FUTURES[:1] if behavior == 'rule' else FUTURES
                    for future in futures:
                        for first_action in rec['legal_actions']:
                            if time.monotonic() - started > 1800:
                                raise RuntimeError('Frozen continuation time budget exhausted; keep partial CSV')
                            branch = reseed_continuation(env, future)
                            action = first_action
                            n = 0
                            violations = Counter()
                            while True:
                                assert branch.constraints.action_mask(branch.snapshot)[action]
                                _, _, term, trunc, info = paired_step(branch, action)
                                violations['unavailable_entries'] += int(info['unavailable_corridor_entries'])
                                violations['release_violations'] += int(not info['actual_release_within_limit'])
                                violations['direction_violations'] += int(info['actual_direction_violation'])
                                if n == 0:
                                    first_actual = int(info['actual_joint_action'])
                                    first_sign = int(info['actual_sign_action'])
                                    first_room = int(info['room_release_count'])
                                n += 1
                                if term or trunc:
                                    break
                                action = ppo_action(models[seed], branch.policy_observation(), branch.constraints.action_mask(branch.snapshot))[0]
                            row = dict(state_id=sid, scenario_id=scene_id, cohort=rec['cohort'], behavior=behavior,
                                population_seed=popseed, training_seed=seed, source_mode=mode, source_step=rec['step'],
                                future_seed=future, first_requested_action=first_action, first_actual_action=first_actual,
                                first_actual_sign=first_sign, first_room_release=first_room, continuation_steps=n,
                                **terminal_metrics(info, branch.unwrapped.domain_env.config),
                                **{name: int(violations[name]) for name in ('unavailable_entries', 'release_violations', 'direction_violations')})
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            count += 1
                            continuation_steps += n
                            branch.close()
                    state_count += 1
                    if state_count % 80 == 0:
                        f.flush()
                        print(json.dumps({'states': state_count, 'terminal_rollouts': count,
                                          'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
                action = ppo_action(models[seed], env.policy_observation(active, gate),
                    env.constraints.action_mask(env.snapshot), active, gate,
                    elapsed_since_receipt=0 if active is None else env.snapshot.observation_time_s - active.received_time_s)[0]
                _, _, term, trunc, _ = paired_step(env, action)
                if term or trunc:
                    break
            env.close()
    assert count == expected and state_count == len(selected)
    save(out / 'execution_summary.json', dict(terminal_rollouts=count, selected_states=state_count,
         continuation_steps=continuation_steps, elapsed_seconds=time.monotonic() - started))
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'terminal_rollouts': count, 'selected_states': state_count}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'replay', 'continuations', 'verify'])
    p.add_argument('--replay-path')
    args = p.parse_args()
    if args.stage == 'continuations':
        assert args.replay_path
        continuations(args.replay_path)
    else:
        {'prepare': prepare, 'replay': replay, 'verify': verify}[args.stage]()
