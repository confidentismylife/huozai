"""Separate geometry/FER/reward and planner checks, without simulator imports."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_dynamic_probe_20261014'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-11), (a, b)


def main():
    cfg = read(ROOT / 'schemas/safestep_dynamic_probe_20261014.json')
    source = read(ROOT / cfg['source_config'])
    plan = read(OUT / 'frozen_manifest.json')
    for group in ['code_sha256', 'protected_previous_files', 'field_sha256']:
        for path, digest in plan[group].items():
            assert sha(ROOT / path) == digest, path
    coords = [(i, 0) for i in range(20)] + [(0, 1), (19, 1), (17, 1), (13, -1)]
    labels = [f'R{i+1}' for i in range(20)] + ['Exit_A', 'Exit_B', 'Room_A', 'Room_B']
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    neighbors = {}
    for s, (x, y) in enumerate(coords):
        neighbors[s] = [(a, coords.index((x+dx, y+dy))) for a, (dx, dy) in enumerate(deltas)
                        if (x+dx, y+dy) in coords]
    exits, start = {20, 21}, labels.index(cfg['start'])
    settings = source['paper_settings']
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    penalty, cap = cfg['time_cost_in_FED_per_step'], cfg['max_steps']
    scenarios = {s['scenario_id']: s for s in plan['scenarios']}
    fields, oracle_values = {}, {}
    for sid, scenario in scenarios.items():
        path = OUT / 'fields' / f'{sid}.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        field = np.array([[float(row[name]) for name in labels] for row in rows])
        assert field.shape == (cap+1, 24)
        for k, row in enumerate(rows):
            assert int(row['k']) == k
            close(row['time_s'], scenario['tau'] + k*dt)
            for state, (x, _) in enumerate(coords):
                rate = cfg['background_rate_per_second']
                if k >= cfg['reveal_steps'][scenario['reveal']]:
                    z = (x - cfg['danger_centers'][scenario['danger_side']]) / cfg['spread_cells']
                    rate += cfg['peak_rates_per_second'][scenario['severity']] * math.exp(-.5*z*z)
                if state in exits:
                    rate = 0
                close(field[k, state], rate)
        fields[sid] = field
        # Forward propagation is independent of the runner's backward recurrence.
        costs = np.full(24, np.inf)
        costs[start] = 0
        best_exit = float('inf')
        for k in range(cap):
            following = np.full(24, np.inf)
            for state in range(24):
                if state in exits:
                    continue
                for _, new in neighbors[state]:
                    value = costs[state] + dt*field[k, state] + penalty
                    following[new] = min(following[new], value)
            best_exit = min(best_exit, min(following[20], following[21]))
            costs = following
        oracle_values[sid] = best_exit

    def simple_paths(state, seen=None):
        if state in exits:
            return [([], [state])]
        seen = {state} if seen is None else seen | {state}
        found = []
        for action, new in neighbors[state]:
            if new not in seen:
                found.extend(([action] + path, [state] + cells) for path, cells in simple_paths(new, seen))
        return found

    def snapshot_optimal(state, rates, target=None):
        candidates = []
        for path, cells in simple_paths(state):
            if target is None or cells[-1] == target:
                cost = sum(dt*rates[s] + penalty for s in cells[:-1])
                candidates.append((cost, path))
        best = min(value for value, _ in candidates)
        return [path for value, path in candidates if math.isclose(value, best, rel_tol=0, abs_tol=1e-12)]

    torch.set_num_threads(1)
    q_maps = {}
    for case, seed in itertools.product(cfg['case_blocks'], cfg['seeds']):
        layers, width = [], 24
        for hidden in source['explicit_implementation_assumptions']['network_hidden_layers']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        model = nn.Sequential(*layers, nn.Linear(width, 4))
        model.load_state_dict(torch.load(ROOT / 'outputs/safestep_reproduction_20261014/runs' /
            f'{case}_seed{seed}/policy.pt', map_location='cpu', weights_only=True))
        model.eval()
        with torch.inference_mode():
            q = model(torch.eye(24)).numpy()
        choices = []
        for state in range(24):
            legal = [a for a, _ in neighbors[state]]
            choices.append(max(legal, key=lambda a: (q[state, a], -a)))
        q_maps[(case, seed)] = choices
    traces = [json.loads(line) for line in (OUT / 'traces.jsonl').read_text(encoding='utf-8').splitlines()]
    with (OUT / 'results.csv').open(encoding='utf-8-sig', newline='') as handle:
        results = list(csv.DictReader(handle))
    assert len(traces) == len(results) == len(plan['schedule']) == cfg['planned_rollouts']
    assert [(row['scenario_id'], row['method']) for row in traces] == [(row['scenario_id'], row['method']) for row in plan['schedule']]
    indexed = {(row['scenario_id'], row['method']): row for row in traces}
    assert len(indexed) == len(traces)
    steps_checked = 0
    for item, record in zip(traces, results):
        sid, method = item['scenario_id'], item['method']
        assert (record['scenario_id'], record['method']) == (sid, method)
        scenario, rates = scenarios[sid], fields[sid]
        state, fed, reward, visited = start, 0., 0., {start}
        revisits = 0
        actions = [row['action'] for row in item['trace']]
        if method in ['fixed_left', 'fixed_right', 'nearest_exit', 'snapshot_once']:
            rate_vector = rates[0] if method == 'snapshot_once' else np.zeros(24)
            target = 20 if method == 'fixed_left' else 21 if method == 'fixed_right' else None
            assert actions in snapshot_optimal(start, rate_vector, target)
        for k, (step, observation) in enumerate(zip(item['trace'], item['observations'])):
            assert observation['k'] == k and observation['state'] == state
            assert np.array_equal(observation['current_FER_per_second'], rates[k])
            assert step['from_state'] == state and state not in exits
            close(step['absolute_time_s'], scenario['tau'] + k*dt)
            action = step['action']
            assert action in [a for a, _ in neighbors[state]]
            if method == 'snapshot_replan':
                assert action in [path[0] for path in snapshot_optimal(state, rates[k])]
            elif method.startswith('legacy_DQN_seed'):
                seed = int(method.removeprefix('legacy_DQN_seed'))
                assert action == q_maps[(scenario['case'], seed)][state]
            new = dict(neighbors[state])[action]
            increment = dt*rates[k, state]
            fed += increment
            revisit = new in visited
            kind = 'exit' if new in exits else 'visited' if revisit else 'regular'
            utility = 0 if fed == 0 else -100 if fed < .3 else -300 if fed < 1 else -800
            immediate_reward = settings['sigma_path']*settings['path_rewards'][kind] + settings['sigma_tox']*utility*increment
            assert step['to_state'] == new and not step['invalid']
            assert step['path_reward_kind'] == kind and step['tox_utility'] == utility
            close(step['dose_increment'], increment)
            close(step['cumulative_personal_FED'], fed)
            close(step['reward'], immediate_reward)
            reward += settings['discount']**k * immediate_reward
            revisits += revisit
            state = new
            visited.add(new)
            steps_checked += 1
        assert len(item['trace']) == len(item['observations'])
        result = item['outcome']
        success = state in exits
        assert result['exit_reached'] == success and record['exit_reached'] == str(success)
        assert success or len(actions) == cap
        assert result['steps'] == len(actions)
        close(result['time_seconds'], len(actions)*dt)
        close(result['personal_FED'], fed)
        close(result['paper_discounted_reward'], reward)
        assert result['revisits'] == revisits
        close(result['future_oracle_J'], oracle_values[sid])
        if success:
            close(result['J'], fed + penalty*len(actions))
            close(result['J_regret_vs_future_oracle'], result['J'] - oracle_values[sid])
            assert result['J_regret_vs_future_oracle'] >= -1e-10
            assert result['exit'] == labels[state]
        if method == 'future_J_oracle':
            close(result['J'], oracle_values[sid])
        for name in ['steps', 'time_seconds', 'personal_FED', 'paper_discounted_reward', 'invalid_actions', 'revisits']:
            close(record[name], result[name])
    blind_pairs = 0
    for case, severity in itertools.product(cfg['case_blocks'], cfg['peak_rates_per_second']):
        left = f'{case}__delayed__left__{severity}'
        right = f'{case}__delayed__right__{severity}'
        assert np.array_equal(fields[left][:2], fields[right][:2])
        for method in cfg['methods']:
            if method != 'future_J_oracle':
                a, b = indexed[(left, method)], indexed[(right, method)]
                assert [s['action'] for s in a['trace'][:2]] == [s['action'] for s in b['trace'][:2]]
                blind_pairs += 1
    summary = read(OUT / 'summary.json')
    assert summary['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
    assert summary['results_sha256'] == sha(OUT / 'results.csv')
    assert summary['traces_sha256'] == sha(OUT / 'traces.jsonl')
    for method, aggregate in summary['aggregate_descriptive_only'].items():
        rows = [r['outcome'] for r in traces if r['method'] == method]
        successful = [r for r in rows if r['exit_reached']]
        assert aggregate['successes'] == len(successful)
        for summary_name, row_name in [('mean_FED_success_only', 'personal_FED'), ('mean_time_success_only', 'time_seconds'),
                                      ('mean_J_success_only', 'J'), ('mean_J_regret_success_only', 'J_regret_vs_future_oracle')]:
            if successful:
                close(aggregate[summary_name], np.mean([r[row_name] for r in successful]))
    for row in summary['delayed_right_paired_J_effects']:
        sid = row['scenario_id']
        a, b = indexed[(sid, 'snapshot_once')]['outcome'], indexed[(sid, 'snapshot_replan')]['outcome']
        close(row['once_J'], a['J'])
        close(row['replan_J'], b['J'])
        close(row['J_improvement'], a['J'] - b['J'])
    for path, digest in plan['protected_previous_files'].items():
        assert sha(ROOT / path) == digest
    verified = {'status': 'dynamic_probe_independent_computational_checks_passed',
        'episodes_recomputed': len(traces), 'steps_recomputed': steps_checked,
        'fields_formula_checked': len(fields), 'forward_DP_oracles_checked': len(oracle_values),
        'snapshot_planner_actions_checked_by_simple_path_enumeration': True,
        'pre_reveal_side_pairs_with_identical_online_actions': blind_pairs,
        'legacy_models_and_masked_actions_rechecked': 6,
        'old_artifacts_unchanged': True, 'new_training': 0, 'LLM_requests': 0,
        'scope': 'Computational consistency only; no physical/human validation or external replication.',
        'summary_sha256': sha(OUT / 'summary.json'), 'verifier_sha256': sha(__file__)}
    target = OUT / 'verification.json'
    payload = json.dumps(verified, indent=2, ensure_ascii=False) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
