"""Read saved observations/weights and audit Q calibration; no optimizer/rollout."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from fire_evac.safestep_baseline import SmallMap
from fire_evac.safestep_shared_observation import build_network, rate_table, legal_matrix
from fire_evac.safestep_q_cache_audit import exact_action_values, closed_walk_members, returns_to_go

CONFIG = ROOT / 'schemas/safestep_q_cache_audit_20261015.json'
OUT = ROOT / 'outputs/safestep_q_cache_audit_20261015'
NAMES = ['left', 'right', 'up', 'down']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def table(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    cfg = read(CONFIG)
    source = ROOT / cfg['source']
    source_cfg = read(ROOT / cfg['source_config'])
    old_plan = read(source / 'frozen_manifest.json')
    previous = read(source / 'completion_manifest.json')
    protected = {**old_plan['protected_previous_files'], **previous['artifact_sha256'],
                 str((source/'completion_manifest.json').relative_to(ROOT)): sha(source/'completion_manifest.json')}
    for path, digest in protected.items():
        assert sha(ROOT / path) == digest, path
    code = ['schemas/safestep_q_cache_audit_20261015.json', 'src/fire_evac/safestep_q_cache_audit.py',
        'scripts/audit_safestep_q_cache.py', 'scripts/verify_safestep_q_cache.py', 'tests/test_safestep_q_cache_audit.py']
    manifest = {'code_sha256': {p: sha(ROOT/p) for p in code}, 'protected_previous_files': protected,
        'known_source_results': True, 'new_training': 0, 'new_rollouts': 0, 'LLM_requests': 0,
        'old_final_scenarios_accessed': False}
    OUT.mkdir(exist_ok=True)
    target = OUT / 'frozen_manifest.json'
    if target.exists():
        prior = read(target)
        assert {k: v for k, v in prior.items() if k != 'audit_frozen_at_utc'} == manifest
        return cfg, source_cfg, old_plan, prior
    manifest['audit_frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(target, manifest)
    print(json.dumps({'status': 'cache_audit_scope_frozen', 'cached_cases': 96,
        'training_transitions_to_read': 120000, 'new_training': 0}), flush=True)
    return cfg, source_cfg, old_plan, manifest


def net_values(model, values):
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(values), 2048):
            chunks.append(model(torch.from_numpy(np.asarray(values[begin:begin+2048], dtype=np.float32))).numpy())
    return np.vstack(chunks)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {'count': int(len(values)), 'mean': float(np.mean(values)),
        'median': float(np.median(values)), 'mean_abs': float(np.mean(np.abs(values))),
        'p90_abs': float(np.quantile(np.abs(values), .9))}


def coverage(data, observation, chosen, reference, cfg, source_cfg, successes):
    states = data['obs'][:, :24].argmax(axis=1)
    state = int(np.argmax(observation[:24]))
    at_cell = states == state
    k = int(round(float(observation[48])*source_cfg['max_steps']))
    params = cfg['context_coverage']
    fer_dist = np.max(np.abs(data['obs'][:, 24:48]-observation[24:48]), axis=1)
    context = at_cell & (np.abs(data['k'].astype(int)-k) <= params['time_window_steps']) & (
        np.abs(data['obs'][:, 49]-observation[49]) <= params['personal_FED_tolerance']) & (
        fer_dist <= params['normalized_FER_Linf_tolerance'])
    exact = np.all(data['obs'] == observation, axis=1)
    tail = np.arange(len(states)) >= len(states)-source_cfg['training']['replay_capacity']
    succeeded = successes[data['episode_id']]
    result = {'training_cell_transitions': int(at_cell.sum()),
        'exact_observation_matches': int(exact.sum()), 'near_context_transitions': int(context.sum()),
        'final_replay_near_context_transitions': int((context & tail).sum()),
        'min_FER_distance_at_cell': float(fer_dist[at_cell].min()) if at_cell.any() else None}
    for prefix, action in [('chosen', chosen), ('reference', reference)]:
        took = data['actions'] == action
        result.update({f'training_{prefix}_action_at_cell': int((at_cell & took).sum()),
            f'final_replay_{prefix}_action_at_cell': int((at_cell & took & tail).sum()),
            f'near_context_{prefix}_action': int((context & took).sum()),
            f'near_context_{prefix}_action_in_successful_behavior_episode': int((context & took & succeeded).sum())})
    return result


def run(cfg, source_cfg, old_plan, manifest):
    if (OUT / 'summary.json').exists():
        raise RuntimeError('Audit already complete. Preserve output; use verifier.')
    source = ROOT / cfg['source']
    traces = [json.loads(line) for line in (source/'evaluation_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = [r for r in traces if r['method'].startswith('hazard_full_seed')]
    assert len(cases) == 96
    grid = SmallMap.paper_case_map()
    legal = legal_matrix(grid)
    old_settings = read(ROOT / source_cfg['source_config'])['paper_settings']
    dt = old_settings['cell_size_m']/old_settings['walking_speed_m_per_s']
    tr, tolerance = source_cfg['training'], cfg['Q_tolerance']
    fields, exact = {}, {}
    scenarios = {s['scenario_id']: s for split in ['development', 'confirmation'] for s in old_plan['banks'][split]}
    for sid, scenario in scenarios.items():
        fields[sid] = rate_table(grid, scenario, source_cfg)
        exact[sid] = exact_action_values(grid, fields[sid], dt, source_cfg['time_cost_in_FED_per_step'],
                                       tr['reward_scale'], tr['timeout_penalty_in_FED_units'])
    torch.set_num_threads(1)
    all_rows, representatives, cycles, training_stats = [], [], [], {}
    for seed in cfg['seeds']:
        method = f'hazard_full_seed{seed}'
        folder = source / 'runs' / method
        model = build_network(tr['hidden_layers'])
        model.load_state_dict(torch.load(folder/'policy.pt', weights_only=True, map_location='cpu'))
        model.eval()
        with np.load(folder/'training_transitions.npz') as stored:
            data = {key: stored[key] for key in stored.files}
        train_q = net_values(model, data['obs'])
        next_q = net_values(model, data['next_obs'])
        future = np.where(data['next_legal'], next_q, -np.inf).max(axis=1)
        final_targets = data['rewards'].astype(float)+np.where(data['dones'], 0., tr['discount']*future)
        selected = train_q[np.arange(len(train_q)), data['actions']]
        residual = selected-final_targets
        dest = data['next_obs'][:, :24].argmax(axis=1)
        terminal_expected = np.isin(dest, list(grid.exits)) | (data['k']+1 == source_cfg['max_steps'])
        assert np.array_equal(data['dones'], terminal_expected)
        timed_out = data['dones'] & ~np.isin(dest, list(grid.exits))
        expected_reward = -tr['reward_scale']*(data['dose_increment']+source_cfg['time_cost_in_FED_per_step']+
                                             timed_out*tr['timeout_penalty_in_FED_units'])
        assert np.allclose(data['rewards'], expected_reward, rtol=2e-7, atol=4e-6)
        assert np.array_equal(data['next_legal'], legal[dest])
        assert np.array_equal(data['obs'][:, 48], data['k']/source_cfg['max_steps'])
        training_stats[method] = {'training_transitions': len(selected),
            'final_network_Bellman_residual_all': distribution(residual),
            'final_network_Bellman_residual_retained_replay': distribution(residual[-tr['replay_capacity']:]),
            'positive_selected_Q_all_training': int((selected > tolerance).sum()),
            'task_timeouts_in_saved_training': int(timed_out.sum()),
            'task_exits_in_saved_training': int((data['dones'] & ~timed_out).sum()),
            'last_training_budget_transition_task_done': bool(data['dones'][-1]),
            'clock_rewards_terminal_masks_consistent': True,
            'historical_minibatches_and_target_checkpoints_available': False,
            'last_global_step_is_target_sync': tr['transitions_per_model'] % tr['target_sync_steps'] == 0}
        logs = csv_rows(folder/'episodes.csv')
        successes = np.zeros(max(int(row['episode']) for row in logs)+1, dtype=bool)
        for row in logs:
            successes[int(row['episode'])] = row['episode_complete'] == row['exit_reached'] == 'True'
        for item in [r for r in cases if r['method'] == method]:
            sid = item['scenario_id']
            trace, observations = item['trace'], np.array(item['effective_inputs'], dtype=np.float32)
            member, witnesses = closed_walk_members(trace, item['onset'])
            predictions = net_values(model, observations)
            rewards = [-tr['reward_scale']*(row['dose_increment']+source_cfg['time_cost_in_FED_per_step']+
                (tr['timeout_penalty_in_FED_units'] if k+1 == source_cfg['max_steps'] and not row['exit_reached'] else 0))
                for k, row in enumerate(trace)]
            returns = returns_to_go(rewards)
            local = []
            for k, step in enumerate(trace):
                state, action = step['from_state'], step['action']
                choices = np.flatnonzero(legal[state]).tolist()
                greedy = max(choices, key=lambda a: (predictions[k, a], -a))
                if greedy != action:
                    with torch.inference_mode():
                        predictions[k] = model(torch.from_numpy(observations[k])).numpy()
                    assert max(choices, key=lambda a: (predictions[k, a], -a)) == action
                q_true = exact[sid][k, state]
                best = max(choices, key=lambda a: (q_true[a], -a))
                terminal = step['exit_reached'] or k+1 == source_cfg['max_steps']
                next_best = 0. if terminal else float(max(predictions[k+1, a] for a in np.flatnonzero(legal[step['to_state']])))
                one_step = rewards[k]+tr['discount']*next_best
                row = {'scenario_id': sid, 'split': item['split'], 'method': method, 'k': k,
                    'cell': step['from_cell'], 'state': state, 'next_state': step['to_state'],
                    'onset': item['onset'], 'after_reveal': k >= item['onset'], 'cycle_member': bool(member[k]),
                    'chosen_action': action, 'chosen_name': NAMES[action], 'reference_action': best,
                    'reference_name': NAMES[best], 'personal_FED_before': float(observations[k, 49]),
                    **{f'Q_{name}': float(predictions[k, a]) for a, name in enumerate(NAMES)},
                    **{f'exact_Q_{name}': float(q_true[a]) if a in choices else None for a, name in enumerate(NAMES)},
                    'chosen_Q': float(predictions[k, action]), 'exact_chosen_Q': float(q_true[action]),
                    'optimal_Q': float(q_true[best]), 'action_ranking_regret': float(q_true[best]-q_true[action]),
                    'chosen_Q_optimal_overestimate': float(predictions[k, action]-q_true[action]),
                    'saved_policy_return_to_go': float(returns[k]),
                    'chosen_Q_minus_saved_policy_return': float(predictions[k, action]-returns[k]),
                    'final_network_one_step_target': float(one_step),
                    'final_network_Bellman_residual': float(predictions[k, action]-one_step)}
                all_rows.append(row)
                local.append(row)
            candidates = [row for row in local if row['after_reveal'] and row['cycle_member'] and row['action_ranking_regret'] > tolerance]
            selected_row = candidates[0] if candidates else next((r for r in local if r['after_reveal'] and r['cycle_member']),
                                                               next(r for r in local if r['after_reveal']))
            k = selected_row['k']
            witness = next((w for w in witnesses if w['begin_k'] <= k < w['end_k_exclusive']), witnesses[0] if witnesses else None)
            representatives.append({**selected_row, 'case_exit_reached': item['outcome']['exit_reached'],
                'case_steps': item['outcome']['steps'], 'case_revisits': item['outcome']['revisits'],
                'cycle_witness': json.dumps(witness, ensure_ascii=False),
                **coverage(data, observations[k], selected_row['chosen_action'], selected_row['reference_action'], cfg, source_cfg, successes)})
            cycles.append({'scenario_id': sid, 'method': method, 'post_reveal_cycle_members': int(member.sum()),
                           'closed_walk_witnesses': witnesses})
    table(OUT/'cached_decision_Q_values.csv', all_rows)
    table(OUT/'cycle_Q_coverage_cases.csv', representatives)
    dump(OUT/'closed_walk_witnesses.json', cycles)
    dump(OUT/'training_coverage_and_final_TD.json', training_stats)
    aggregates = {}
    for seed in cfg['seeds']:
        method = f'hazard_full_seed{seed}'
        subset = [r for r in all_rows if r['method'] == method and r['after_reveal']]
        cyclic = [r for r in subset if r['cycle_member']]
        reps = [r for r in representatives if r['method'] == method]
        aggregates[method] = {'cached_cases': len(reps), 'post_reveal_steps': len(subset),
            'post_reveal_cycle_member_steps': len(cyclic),
            'positive_selected_Q_post_reveal': sum(r['chosen_Q'] > tolerance for r in subset),
            'offline_nonoptimal_actions_post_reveal': sum(r['action_ranking_regret'] > tolerance for r in subset),
            'optimistic_Q_vs_exact_optimum_post_reveal': sum(r['chosen_Q_optimal_overestimate'] > tolerance for r in subset),
            'Q_minus_exact_optimal_for_chosen_action': distribution([r['chosen_Q_optimal_overestimate'] for r in subset]),
            'Q_minus_saved_policy_suffix_return': distribution([r['chosen_Q_minus_saved_policy_return'] for r in subset]),
            'final_network_one_step_residual': distribution([r['final_network_Bellman_residual'] for r in subset]),
            'representatives_with_zero_broad_chosen_action_coverage': sum(r['training_chosen_action_at_cell'] == 0 for r in reps),
            'representatives_with_zero_broad_reference_action_coverage': sum(r['training_reference_action_at_cell'] == 0 for r in reps),
            'representatives_with_zero_near_context': sum(r['near_context_transitions'] == 0 for r in reps),
            'representatives_with_zero_exact_observation_matches': sum(r['exact_observation_matches'] == 0 for r in reps)}
    summary = {'status': 'completed_cache_only_Q_audit', 'cached_cases': len(cases), 'cached_steps': len(all_rows),
        'training_transitions_read': sum(r['training_transitions'] for r in training_stats.values()),
        'aggregates': aggregates, 'historical_TD_targets_recovered': False,
        'source_artifacts_unchanged': True, 'new_training': 0, 'new_rollouts': 0, 'LLM_requests': 0,
        'old_final_scenarios_accessed': False,
        'caution': 'Posthoc diagnostic, no repair or causal isolation. Exact forced-action optimum and saved-policy return are distinct references. Current-network TD errors do not reconstruct historical optimizer targets.',
        'manifest_sha256': sha(OUT/'frozen_manifest.json'), 'decision_rows_sha256': sha(OUT/'cached_decision_Q_values.csv'),
        'case_table_sha256': sha(OUT/'cycle_Q_coverage_cases.csv')}
    for path, digest in manifest['protected_previous_files'].items():
        assert sha(ROOT/path) == digest
    dump(OUT/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    cfg, source_cfg, old_plan, manifest = prepare()
    if args.mode == 'run':
        run(cfg, source_cfg, old_plan, manifest)
