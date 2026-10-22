"""Independent archive/observation/route checks; no production simulator import."""
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
OUT = ROOT / 'outputs/safestep_shared_observation_20261014'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), (a, b)


def main():
    cfg = read(ROOT / 'schemas/safestep_shared_observation_20261014.json')
    source = read(ROOT / cfg['source_config'])
    plan = read(OUT / 'frozen_manifest.json')
    for group in ['code_sha256', 'protected_previous_files', 'episode_schedules_sha256']:
        for path, digest in plan[group].items():
            assert sha(ROOT / path) == digest, path
    coords = [(i, 0) for i in range(20)]+[(0, 1), (19, 1), (17, 1), (13, -1)]
    x = np.array([p[0] for p in coords])
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    neighbors, legal = {}, np.zeros((24, 4), dtype=bool)
    for s, (px, py) in enumerate(coords):
        neighbors[s] = [(a, coords.index((px+dx, py+dy))) for a, (dx, dy) in enumerate(deltas)
                        if (px+dx, py+dy) in coords]
        legal[s, [a for a, _ in neighbors[s]]] = True
    cap, exits = cfg['max_steps'], {20, 21}
    dt = source['paper_settings']['cell_size_m']/source['paper_settings']['walking_speed_m_per_s']
    penalty, tr = cfg['time_cost_in_FED_per_step'], cfg['training']

    def rates_at(scenarios, steps):
        centers = np.array([s['center'] for s in scenarios])
        peaks = np.array([s['peak'] for s in scenarios])
        onsets = np.array([s['onset'] for s in scenarios])
        field = cfg['background_rate_per_second'] + (np.asarray(steps) >= onsets)[:, None]*peaks[:, None]*np.exp(
            -.5*((x[None, :]-centers[:, None])/cfg['spread_cells'])**2)
        field[:, [20, 21]] = 0
        return field

    def build(seed):
        torch.manual_seed(seed)
        layers, width = [], 50
        for h in tr['hidden_layers']:
            layers.extend([nn.Linear(width, h), nn.ReLU()])
            width = h
        return nn.Sequential(*layers, nn.Linear(width, 4))

    torch.set_num_threads(1)
    models, initials = {}, {}
    training_checked = updates_checked = 0
    for arm, seed in itertools.product(cfg['arms'], cfg['seeds']):
        folder = OUT / 'runs' / f'{arm}_seed{seed}'
        summary = read(folder / 'training_summary.json')
        archive = np.load(folder / 'training_transitions.npz')
        assert summary['archive_sha256'] == sha(folder / 'training_transitions.npz')
        assert summary['model_sha256'] == sha(folder / 'policy.pt')
        assert summary['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
        n = len(archive['actions'])
        assert n == summary['transitions'] == tr['transitions_per_model']
        obs, nxt, actions = archive['obs'], archive['next_obs'], archive['actions'].astype(int)
        assert obs.shape == nxt.shape == (n, 50) and np.isfinite(obs).all() and np.isfinite(nxt).all()
        state, new = obs[:, :24].argmax(axis=1), nxt[:, :24].argmax(axis=1)
        assert np.array_equal(obs[:, :24], np.eye(24)[state])
        assert np.array_equal(nxt[:, :24], np.eye(24)[new])
        assert legal[state, actions].all() and np.array_equal(archive['next_legal'], legal[new])
        for s in range(24):
            for a, dest in neighbors[s]:
                mask = (state == s) & (actions == a)
                assert np.all(new[mask] == dest)
        k = archive['k'].astype(int)
        scenarios = [plan['banks']['training'][i] for i in archive['scenario_index']]
        field = rates_at(scenarios, k)
        next_field = rates_at(scenarios, k+1)
        dose = field[np.arange(n), state]*dt
        assert np.allclose(archive['dose_increment'], dose, rtol=1e-12, atol=1e-13)
        ends = np.isin(new, [20, 21]) | (k+1 == cap)
        assert np.array_equal(archive['dones'], ends)
        expected_reward = -tr['reward_scale']*(dose+penalty+np.where(ends & ~np.isin(new, [20, 21]), tr['timeout_penalty_in_FED_units'], 0))
        assert np.allclose(archive['rewards'], expected_reward, rtol=2e-7, atol=4e-6)
        starts = np.r_[0, np.flatnonzero(np.diff(archive['episode_id']))+1]
        lengths = np.diff(np.r_[starts, n])
        total = np.cumsum(dose)
        offsets = np.where(starts > 0, total[np.maximum(0, starts-1)], 0)
        fed_after = total-np.repeat(offsets, lengths)
        fed_before = fed_after-dose
        assert np.allclose(obs[:, 49], fed_before, rtol=2e-7, atol=2e-6)
        assert np.allclose(nxt[:, 49], fed_after, rtol=2e-7, atol=2e-6)
        assert np.array_equal(obs[:, 48], k/cap) and np.array_equal(nxt[:, 48], (k+1)/cap)
        assert np.all(k[starts] == 0) and np.all(state[starts] == 23)
        assert np.all(ends[starts[1:]-1])
        schedule = np.load(OUT / f'episode_schedule_seed{seed}.npy')
        assert np.array_equal(archive['scenario_index'], schedule[archive['episode_id']-1])
        if arm == 'hazard_blind':
            assert np.all(obs[:, 24:48] == 0) and np.all(nxt[:, 24:48] == 0)
        else:
            assert np.allclose(obs[:, 24:48], field.astype(np.float32)/cfg['observation']['FER_divisor'], rtol=0, atol=1e-7)
            assert np.allclose(nxt[:, 24:48], next_field.astype(np.float32)/cfg['observation']['FER_divisor'], rtol=0, atol=1e-7)
        visits = np.zeros((24, 4), dtype=int)
        np.add.at(visits, (state, actions), 1)
        assert np.array_equal(visits, read(folder / 'state_action_visits.json')['counts'])
        log = rows(folder / 'episodes.csv')
        assert sum(int(row['steps']) for row in log) == n
        for row, begin, length in zip(log, starts, lengths):
            end = begin+length-1
            assert int(row['end_global_step']) == end+1 and row['episode_complete'] == str(bool(ends[end]))
            close(row['personal_FED'], fed_after[end])
        updates = sum(t >= tr['warmup_transitions'] and t % tr['update_every_steps'] == 0 for t in range(1, n+1))
        assert updates == summary['gradient_updates']
        assert len(rows(folder / 'progress.csv')) == 4
        for row in rows(folder / 'progress.csv'):
            assert math.isfinite(float(row['mean_TD_MSE']))
        model = build(seed)
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
        assert digest.hexdigest() == summary['initial_parameter_sha256']
        initials[(arm, seed)] = digest.hexdigest()
        model.load_state_dict(torch.load(folder / 'policy.pt', weights_only=True, map_location='cpu'))
        model.eval()
        models[f'{arm}_seed{seed}'] = model
        training_checked += n
        updates_checked += updates
    assert all(initials[('hazard_full', seed)] == initials[('hazard_blind', seed)] for seed in cfg['seeds'])
    locked = read(OUT / 'models_locked.json')
    summary = read(OUT / 'summary.json')
    assert all(item['completed_at_utc'] < locked['locked_at_utc'] for item in locked['models'])
    assert locked['locked_at_utc'] < summary['evaluation_started_at_utc']
    assert summary['models_locked_sha256'] == sha(OUT / 'models_locked.json')
    scenarios = {s['scenario_id']: s for split in ['development', 'confirmation'] for s in plan['banks'][split]}

    def simple_paths(state, seen=None):
        if state in exits:
            return [([], [state])]
        seen = {state} if seen is None else seen | {state}
        result = []
        for action, new in neighbors[state]:
            if new not in seen:
                result.extend(([action]+path, [state]+cells) for path, cells in simple_paths(new, seen))
        return result

    oracle = {}
    for sid, scenario in scenarios.items():
        field = rates_at([scenario]*(cap+1), np.arange(cap+1))
        costs = np.full(24, np.inf)
        costs[23] = 0
        best = float('inf')
        for k in range(cap):
            next_cost = np.full(24, np.inf)
            for state in range(24):
                if state not in exits:
                    for _, dest in neighbors[state]:
                        next_cost[dest] = min(next_cost[dest], costs[state]+dt*field[k, state]+penalty)
            best = min(best, next_cost[20], next_cost[21])
            costs = next_cost
        oracle[sid] = best
    traces = [json.loads(line) for line in (OUT / 'evaluation_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    csv = rows(OUT / 'evaluation_results.csv')
    assert len(traces) == len(csv) == cfg['planned_evaluation_episodes'] == 288
    assert [(r['scenario_id'], r['method']) for r in traces] == [(r['scenario_id'], r['method']) for r in plan['evaluation_schedule']]
    indexed = {(r['scenario_id'], r['method']): r for r in traces}
    assert len(indexed) == 288
    evaluation_steps = 0
    for item, record in zip(traces, csv):
        sid, method = item['scenario_id'], item['method']
        scenario = scenarios[sid]
        field = rates_at([scenario]*(cap+1), np.arange(cap+1))
        state, fed, revisits, visited = 23, 0., 0, {23}
        assert len(item['trace']) == len(item['common_observations']) == len(item['effective_inputs'])
        for k, (step, common, effective) in enumerate(zip(item['trace'], item['common_observations'], item['effective_inputs'])):
            expected = np.zeros(50, dtype=np.float32)
            expected[state] = 1
            expected[24:48] = field[k].astype(np.float32)/cfg['observation']['FER_divisor']
            expected[48], expected[49] = k/cap, fed
            assert np.array_equal(common, expected)
            eff = expected.copy()
            if method.startswith('hazard_blind'):
                eff[24:48] = 0
            assert np.array_equal(effective, eff)
            action = step['action']
            assert legal[state, action] and step['from_state'] == state
            if method in models:
                with torch.inference_mode():
                    q = models[method](torch.from_numpy(eff)).numpy()
                best = max(np.flatnonzero(legal[state]).tolist(), key=lambda a: (q[a], -a))
                assert action == best
            elif method in ['snapshot_replan', 'nearest_exit']:
                rates = expected[24:48].astype(float)*cfg['observation']['FER_divisor'] if method == 'snapshot_replan' else np.zeros(24)
                candidates = [(sum(dt*rates[s]+penalty for s in cells[:-1]), path[0]) for path, cells in simple_paths(state)]
                minimum = min(cost for cost, _ in candidates)
                assert action in [a for cost, a in candidates if math.isclose(cost, minimum, rel_tol=0, abs_tol=1e-11)]
            new = dict(neighbors[state])[action]
            increment = dt*field[k, state]
            fed += increment
            close(step['dose_increment'], increment)
            close(step['cumulative_personal_FED'], fed)
            close(step['absolute_time_s'], cfg['tau']+k*dt)
            assert step['to_state'] == new and not step['invalid']
            revisits += new in visited
            visited.add(new)
            state = new
            evaluation_steps += 1
        result = item['outcome']
        success = state in exits
        assert success == result['exit_reached'] and success == (record['exit_reached'] == 'True')
        assert success or len(item['trace']) == cap
        close(result['personal_FED'], fed)
        assert result['steps'] == len(item['trace']) and result['revisits'] == revisits
        close(result['time_seconds'], len(item['trace'])*dt)
        close(result['oracle_J'], oracle[sid])
        if success:
            close(result['J'], fed+penalty*len(item['trace']))
            close(result['J_regret'], result['J']-oracle[sid])
            close(record['J'], result['J'])
        else:
            assert result['J'] is None and record['J'] == ''
        if method == 'future_J_oracle':
            close(result['J'], oracle[sid])
        for key in ['steps', 'time_seconds', 'personal_FED', 'invalid_actions', 'revisits']:
            close(record[key], result[key])
    for split in ['development', 'confirmation']:
        for method, aggregate in summary['aggregate'][split].items():
            successful = [r['outcome'] for r in traces if r['split'] == split and r['method'] == method and r['outcome']['exit_reached']]
            assert aggregate['successes'] == len(successful)
            for name in ['personal_FED', 'time_seconds', 'J', 'J_regret']:
                if successful:
                    close(aggregate[f'mean_{name}_success_only'], np.mean([r[name] for r in successful]))
    assert summary['results_sha256'] == sha(OUT / 'evaluation_results.csv')
    assert summary['traces_sha256'] == sha(OUT / 'evaluation_traces.jsonl')
    for path, digest in plan['protected_previous_files'].items():
        assert sha(ROOT / path) == digest
    value = {'status': 'shared_observation_archives_and_evaluation_verified',
        'training_transitions_reconstructed': training_checked, 'gradient_update_counts_checked': updates_checked,
        'training_state_action_visits_reconstructed': True, 'paired_initial_weights_identical': True,
        'observation_ablation_and_no_future_features_checked': True,
        'all_models_locked_before_any_evaluation': True,
        'evaluation_episodes_reconstructed': len(traces), 'evaluation_steps_reconstructed': evaluation_steps,
        'forward_DP_references_checked': len(oracle), 'old_artifacts_unchanged': True,
        'LLM_requests': 0, 'PPO_updates': 0,
        'scope': 'Stored training transitions/flags/rewards and final model behavior verified; optimizer numerical history not independently replayed. No physical/human validation.',
        'summary_sha256': sha(OUT / 'summary.json'), 'verifier_sha256': sha(__file__)}
    target = OUT / 'verification.json'
    payload = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
