"""Small, outcome-selected exploratory traces of exposure and queue pathways."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import numpy as np

from run_timestamped_development_campaign import ROOT, PREP, read, save, sha, csv_write, unique_output
from run_direction_counterfactual_audit import (verify, verify_audit, load_trace, make_env,
    validate_transition, FUTURE_SEEDS, METRICS)
from fire_evac.counterfactual_direction_audit import (IMMEDIATE_ARMS, clone_for_future,
    paired_step, intervention_action, faulty_policy_action, terminal_metrics)

STUDY = ROOT / 'outputs/direction_counterfactual_20261007T031906039073Z'
STATES = ROOT / 'outputs/direction_state_feasibility_20261007T031854810149Z'
MATCHED = ROOT / 'outputs/direction_counterfactual_final_audit_20261007T032154325639Z'
FREEZE = ROOT / 'outputs/direction_mechanism_preparation_20261007'


def prepare():
    verify()
    assert not FREEZE.exists()
    for p in (STUDY, STATES, MATCHED):
        verify_audit(p)
    with (STATES / 'states.jsonl').open(encoding='utf-8') as f:
        states = {r['state_id']: r for r in map(json.loads, f)}
    with (MATCHED / 'matched_direction_effects.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['metric'] == 'mean_cumulative_fed']
    selected = []
    for behavior in ('rule', 'calibrated_logit'):
        for capacity in ('ge', 'lt'):
            subset = []
            for r in rows:
                s = states[r['state_id']]
                ge = s['capacities'][s['lower_branch']] >= s['capacities'][s['higher_branch']]
                if r['behavior'] == behavior and ge == (capacity == 'ge'):
                    subset.append(r)
            subset.sort(key=lambda r: (float(r['difference']), r['state_id']))
            assert len(subset) >= 2
            for extreme, rec in [('minimum_fed_difference', subset[0]), ('maximum_fed_difference', subset[-1])]:
                s = states[rec['state_id']]
                selected.append(dict(state_id=rec['state_id'], behavior=behavior, capacity_stratum=capacity,
                    extreme=extreme, observed_mean_fed_difference=float(rec['difference']),
                    arms=['original_ppo', rec['arm'], 'wait_one_then_ppo'],
                    future_seeds=FUTURE_SEEDS[:1] if behavior == 'rule' else FUTURE_SEEDS,
                    original_level=s['original_requested_action'] % 4))
    FREEZE.mkdir()
    save(FREEZE / 'manifest.json', dict(selected=selected, terminal_rollouts=60,
        purpose='Explore actual flows, person-time by zone and exact incremental exposure decomposition.',
        status='Outcome-selected extremes, descriptive mechanism cases; not independent confirmation or unbiased prevalence.',
        new_training=0, llm_requests=0, reserved_final_scenarios_opened=0,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__),
            STUDY / 'completion_manifest.json', STATES / 'completion_manifest.json', MATCHED / 'completion_manifest.json']}))
    print('Frozen eight exploratory mechanism cases, 60 instrumented continuations.', flush=True)


def run():
    import torch
    from stable_baselines3 import PPO
    verify()
    plan = read(FREEZE / 'manifest.json')
    for p, digest in plan['sources'].items():
        assert sha(ROOT / p) == digest, p
    selection = {r['state_id']: r for r in plan['selected']}
    with (STUDY / 'terminal_rollouts.csv').open(encoding='utf-8-sig') as f:
        reference = {(r['state_id'], r['arm'], int(r['future_seed'])): r for r in csv.DictReader(f)}
    traces, _ = load_trace()
    original = read(PREP / 'manifest.json')
    scenes = {s['scenario_id']: s for s in original['scenarios']}
    torch.set_num_threads(1)
    models = {s: PPO.load(original['ppo_models'][str(s)]['raw_llm_ppo'], device='cpu') for s in original['training_seeds']}
    out = unique_output('direction_mechanism_trace')
    save(out / 'run_manifest.json', dict(preparation_sha256=sha(FREEZE / 'manifest.json'), terminal_rollouts=60,
         no_policy_changes=True, exploration_only=True))
    rows = []
    start = time.monotonic()
    with (out / 'step_trace.jsonl').open('w', encoding='utf-8') as stream:
        for key in sorted(traces):
            if not any(s.startswith('|'.join(map(str, key)) + '|') for s in selection):
                continue
            env = make_env(key, scenes)
            for old in traces[key]:
                sid = '|'.join(map(str, (*key, old['step'])))
                if sid in selection:
                    rec = selection[sid]
                    for future in rec['future_seeds']:
                        for arm in rec['arms']:
                            clone = clone_for_future(env, future)
                            config = clone.unwrapped.domain_env.config
                            before_fed = clone.unwrapped.domain_env.cumulative_fed
                            zone_fed = np.zeros(5)
                            person_seconds = np.zeros(5)
                            flow = np.zeros(2, dtype=int)
                            release = mask_only_wait = n = 0
                            active_plan = True
                            while True:
                                domain = clone.unwrapped.domain_env
                                s = clone.snapshot
                                counts = np.array([len(z) for z in domain.zones])
                                # Offline decomposition only: never input to the selector or PPO.
                                increments = counts * domain._hazard()
                                zone_fed += increments
                                person_seconds += counts * config.dt_seconds
                                mask = clone.constraints.action_mask(s)
                                mask_only_wait += int(not mask[:8].any())
                                action = intervention_action(arm, clone, n, rec['original_level']) if active_plan else None
                                if action is None:
                                    active_plan = False
                                    action = faulty_policy_action(clone, models[key[3]])
                                elif action != 8:
                                    active_plan = False
                                _, _, term, trunc, info = paired_step(clone, action)
                                release += int(info['room_release_count'])
                                flow += domain.last_branch_flow.astype(int)
                                stream.write(json.dumps(dict(state_id=sid, future_seed=future, arm=arm,
                                    step=s.step, occupancy=counts.tolist(), measured_smoke=list(s.smoke),
                                    capacities=list(s.capacities), hold_remaining=s.sign_hold_remaining,
                                    mask=list(map(bool, mask)), requested_action=action,
                                    actual_joint_action=int(info['actual_joint_action']),
                                    actual_sign=int(info['actual_sign_action']),
                                    room_release_count=int(info['room_release_count']),
                                    branch_entries=domain.last_branch_flow.astype(int).tolist(),
                                    cumulative_evacuated=int(info['evacuated']),
                                    cumulative_incapacitated=int(info['incapacitated']),
                                    incremental_zone_fed_sum=increments.tolist()), ensure_ascii=False) + '\n')
                                n += 1
                                if term or trunc:
                                    break
                            assert abs(zone_fed.sum() - (domain.cumulative_fed - before_fed)) < 1e-7
                            endpoint = terminal_metrics(info, config)
                            saved = reference[(sid, arm, future)]
                            assert all(abs(endpoint[m] - float(saved[m])) < 1e-12 for m in METRICS)
                            row = dict(state_id=sid, scenario_id=key[0], behavior=key[1],
                                capacity_stratum=rec['capacity_stratum'], extreme=rec['extreme'],
                                future_seed=future, arm=arm, continuation_steps=n,
                                mask_only_wait_steps=mask_only_wait, rooms_released=release,
                                north_entries=int(flow[0]), south_entries=int(flow[1]), **endpoint)
                            for i, zone in enumerate(('room_a', 'room_b', 'junction', 'north', 'south')):
                                row['future_fed_' + zone] = float(zone_fed[i] / config.population)
                                row['person_seconds_' + zone] = float(person_seconds[i])
                            rows.append(row)
                            clone.close()
                validate_transition(old, env)
            env.close()
    assert len(rows) == 60
    csv_write(out / 'instrumented_rollouts.csv', rows)
    diffs = []
    for case in plan['selected']:
        sid = case['state_id']
        for arm in case['arms'][1:]:
            values = []
            for future in case['future_seeds']:
                a = next(r for r in rows if r['state_id'] == sid and r['arm'] == arm and r['future_seed'] == future)
                b = next(r for r in rows if r['state_id'] == sid and r['arm'] == 'original_ppo' and r['future_seed'] == future)
                values.append({k: a[k] - b[k] for k in a if k in METRICS or k.startswith(('future_fed_', 'person_seconds_')) or k in ('mask_only_wait_steps', 'rooms_released', 'north_entries', 'south_entries')})
            diffs.append(dict(state_id=sid, arm=arm, behavior=case['behavior'],
                              capacity_stratum=case['capacity_stratum'], extreme=case['extreme'],
                              **{k: float(np.mean([r[k] for r in values])) for k in values[0]}))
    csv_write(out / 'case_mean_differences.csv', diffs)
    save(out / 'summary.json', dict(terminal_rollouts=60, cases=8,
         elapsed_seconds=time.monotonic() - start, endpoint_matches_original_counterfactuals=True,
         exact_incremental_fed_decomposition=True, exploratory_outcome_selected=True,
         differences=diffs, new_training=0, llm_requests=0, reserved_final_scenarios_opened=0))
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'completed': 60}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.stage]()
