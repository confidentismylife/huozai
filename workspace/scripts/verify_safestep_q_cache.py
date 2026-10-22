"""Separate forward-cost reference and cached Q/coverage consistency checks."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_q_cache_audit_20261015'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, tolerance=1e-6):
    assert math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=tolerance), (a, b)


def main():
    cfg = read(ROOT/'schemas/safestep_q_cache_audit_20261015.json')
    source = ROOT/cfg['source']
    sc = read(ROOT/cfg['source_config'])
    old_plan = read(source/'frozen_manifest.json')
    plan = read(OUT/'frozen_manifest.json')
    for group in ['code_sha256', 'protected_previous_files']:
        for path, digest in plan[group].items():
            assert sha(ROOT/path) == digest, path
    coords = [(i, 0) for i in range(20)]+[(0, 1), (19, 1), (17, 1), (13, -1)]
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    neighbors = {s: [(a, coords.index((x+dx, y+dy))) for a, (dx, dy) in enumerate(deltas)
                     if (x+dx, y+dy) in coords] for s, (x, y) in enumerate(coords)}
    legal = np.zeros((24, 4), dtype=bool)
    for s in range(24):
        legal[s, [a for a, _ in neighbors[s]]] = True
    old_settings = read(ROOT/sc['source_config'])['paper_settings']
    dt = old_settings['cell_size_m']/old_settings['walking_speed_m_per_s']
    cap, tr, exits = sc['max_steps'], sc['training'], {20, 21}
    scenarios = {s['scenario_id']: s for split in ['development', 'confirmation'] for s in old_plan['banks'][split]}

    def rate(scenario, k, state):
        if state in exits:
            return 0.
        value = sc['background_rate_per_second']
        if k >= scenario['onset']:
            value += scenario['peak']*math.exp(-.5*((coords[state][0]-scenario['center'])/sc['spread_cells'])**2)
        return value

    def forced_reference(scenario, state, k, action):
        new = dict(neighbors[state])[action]
        initial = dt*rate(scenario, k, state)+sc['time_cost_in_FED_per_step']
        if new in exits:
            return -tr['reward_scale']*initial
        costs = np.full(24, np.inf)
        costs[new] = initial
        best_exit = float('inf')
        for t in range(k+1, cap):
            following = np.full(24, np.inf)
            for old in range(24):
                if old in exits:
                    continue
                cost = dt*rate(scenario, t, old)+sc['time_cost_in_FED_per_step']
                for _, dest in neighbors[old]:
                    following[dest] = min(following[dest], costs[old]+cost)
            best_exit = min(best_exit, following[20], following[21])
            costs = following
        best_failure = min(costs[s] for s in range(24) if s not in exits)+tr['timeout_penalty_in_FED_units']
        return -tr['reward_scale']*min(best_exit, best_failure)

    torch.set_num_threads(1)
    models = {}
    for seed in cfg['seeds']:
        layers, width = [], 50
        for hidden in tr['hidden_layers']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        model = nn.Sequential(*layers, nn.Linear(width, 4))
        model.load_state_dict(torch.load(source/'runs'/f'hazard_full_seed{seed}/policy.pt', weights_only=True, map_location='cpu'))
        model.eval()
        models[f'hazard_full_seed{seed}'] = model
    traces = [json.loads(line) for line in (source/'evaluation_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = {(r['scenario_id'], r['method']): r for r in traces if r['method'] in models}
    rows = csv_rows(OUT/'cached_decision_Q_values.csv')
    indexed = {(r['scenario_id'], r['method'], int(r['k'])): r for r in rows}
    assert len(indexed) == len(rows) == sum(len(r['trace']) for r in cases.values())
    witnesses = read(OUT/'closed_walk_witnesses.json')
    members_by_case = {}
    for entry in witnesses:
        key = (entry['scenario_id'], entry['method'])
        case = cases[key]
        members = set()
        for witness in entry['closed_walk_witnesses']:
            begin, end = witness['begin_k'], witness['end_k_exclusive']
            assert begin >= case['onset'] and witness['length'] == end-begin
            segment = case['trace'][begin:end]
            assert segment[0]['from_state'] == segment[-1]['to_state']
            assert witness['cells'] == [segment[0]['from_cell']]+[r['to_cell'] for r in segment]
            members.update(range(begin, end))
        assert len(members) == entry['post_reveal_cycle_members']
        members_by_case[key] = members
    for key, item in cases.items():
        model = models[item['method']]
        observations = np.asarray(item['effective_inputs'], dtype=np.float32)
        with torch.inference_mode():
            q = model(torch.from_numpy(observations)).numpy()
        rewards = [-tr['reward_scale']*(r['dose_increment']+sc['time_cost_in_FED_per_step']+
            (tr['timeout_penalty_in_FED_units'] if k+1 == cap and not r['exit_reached'] else 0))
            for k, r in enumerate(item['trace'])]
        total = 0.
        for k in range(len(item['trace'])-1, -1, -1):
            total += rewards[k]
            step = item['trace'][k]
            row = indexed[(*key, k)]
            action, state = step['action'], step['from_state']
            close(row['saved_policy_return_to_go'], total)
            for a, name in enumerate(['left', 'right', 'up', 'down']):
                close(row['Q_'+name], q[k, a])
            close(row['chosen_Q'], q[k, action])
            assert row['cycle_member'] == str(k in members_by_case[key])
            terminal = step['exit_reached'] or k+1 == cap
            future = 0. if terminal else max(q[k+1, a] for a, _ in neighbors[step['to_state']])
            close(row['final_network_one_step_target'], rewards[k]+future)
            close(row['final_network_Bellman_residual'], q[k, action]-(rewards[k]+future))
            assert row['after_reveal'] == str(k >= item['onset'])
    reps = csv_rows(OUT/'cycle_Q_coverage_cases.csv')
    assert len(reps) == len(cases) == 96
    references_checked = 0
    archives = {method: dict(np.load(source/'runs'/method/'training_transitions.npz')) for method in models}
    for row in reps:
        key, k, state = (row['scenario_id'], row['method']), int(row['k']), int(row['state'])
        item = cases[key]
        values = {}
        for action, _ in neighbors[state]:
            value = forced_reference(scenarios[row['scenario_id']], state, k, action)
            values[action] = value
            close(row['exact_Q_'+['left', 'right', 'up', 'down'][action]], value, 1e-9)
            references_checked += 1
        action, reference = int(row['chosen_action']), int(row['reference_action'])
        close(row['exact_chosen_Q'], values[action])
        close(row['optimal_Q'], max(values.values()))
        assert reference == max(values, key=lambda a: (values[a], -a))
        close(row['action_ranking_regret'], max(values.values())-values[action])
        close(row['chosen_Q_optimal_overestimate'], float(row['chosen_Q'])-values[action])
        earlier = [indexed[(*key, t)] for t in range(len(item['trace'])) if t >= item['onset'] and
            indexed[(*key, t)]['cycle_member'] == 'True' and float(indexed[(*key, t)]['action_ranking_regret']) > cfg['Q_tolerance']]
        if earlier:
            assert k == int(earlier[0]['k'])
        data = archives[row['method']]
        obs = np.array(item['effective_inputs'][k], dtype=np.float32)
        at_cell = data['obs'][:, :24].argmax(axis=1) == state
        exact = np.all(data['obs'] == obs, axis=1)
        context = at_cell & (np.abs(data['k'].astype(int)-k) <= cfg['context_coverage']['time_window_steps']) & (
            np.abs(data['obs'][:, 49]-obs[49]) <= cfg['context_coverage']['personal_FED_tolerance']) & (
            np.abs(data['obs'][:, 24:48]-obs[24:48]).max(axis=1) <= cfg['context_coverage']['normalized_FER_Linf_tolerance'])
        tail = np.arange(len(at_cell)) >= len(at_cell)-tr['replay_capacity']
        assert int(row['training_cell_transitions']) == int(at_cell.sum())
        assert int(row['exact_observation_matches']) == int(exact.sum())
        assert int(row['near_context_transitions']) == int(context.sum())
        assert int(row['final_replay_near_context_transitions']) == int((context & tail).sum())
        for prefix, a in [('chosen', action), ('reference', reference)]:
            assert int(row[f'training_{prefix}_action_at_cell']) == int((at_cell & (data['actions'] == a)).sum())
            assert int(row[f'near_context_{prefix}_action']) == int((context & (data['actions'] == a)).sum())
    training_stats = read(OUT/'training_coverage_and_final_TD.json')
    for method, data in archives.items():
        model = models[method]
        def forward(inputs):
            with torch.inference_mode():
                return np.vstack([model(torch.from_numpy(inputs[i:i+2048])).numpy() for i in range(0, len(inputs), 2048)])
        q, following = forward(data['obs']), forward(data['next_obs'])
        future = np.where(data['next_legal'], following, -np.inf).max(axis=1)
        targets = data['rewards'].astype(float)+np.where(data['dones'], 0., future)
        residual = q[np.arange(len(q)), data['actions']]-targets
        close(training_stats[method]['final_network_Bellman_residual_all']['mean_abs'], np.abs(residual).mean())
        close(training_stats[method]['final_network_Bellman_residual_retained_replay']['mean_abs'], np.abs(residual[-tr['replay_capacity']:]).mean())
    summary = read(OUT/'summary.json')
    assert summary['manifest_sha256'] == sha(OUT/'frozen_manifest.json')
    assert summary['decision_rows_sha256'] == sha(OUT/'cached_decision_Q_values.csv')
    assert summary['case_table_sha256'] == sha(OUT/'cycle_Q_coverage_cases.csv')
    for path, digest in plan['protected_previous_files'].items():
        assert sha(ROOT/path) == digest
    value = {'status': 'cache_Q_audit_computational_checks_passed', 'cached_cases': len(cases),
        'cached_Q_decisions_checked': len(rows), 'representatives_checked': len(reps),
        'forced_action_references_checked_by_forward_cost_DP': references_checked,
        'training_transitions_in_final_network_TD_check': sum(len(a['actions']) for a in archives.values()),
        'coverage_and_saved_returns_checked': True, 'old_artifacts_unchanged': True,
        'new_training': 0, 'new_rollouts': 0, 'LLM_requests': 0,
        'historical_optimizer_targets_not_reconstructed': True,
        'summary_sha256': sha(OUT/'summary.json'), 'verifier_sha256': sha(__file__)}
    target = OUT/'verification.json'
    payload = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
