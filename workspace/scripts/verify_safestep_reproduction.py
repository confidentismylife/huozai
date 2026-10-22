"""Independent dose/reward reconstruction and source/training artifact audit."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_reproduction_20261014'
CONFIG = ROOT / 'schemas/safestep_reproduction_20261014.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11), (a, b)


def independently_reconstruct_trace(trace, start, tau, settings, times, rates, coordinates, labels):
    state, visited, dose, total, discounted = start, {start}, 0.0, 0.0, 0.0
    invalid_count = revisit_count = 0
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    exits = {labels.index('Exit_A'), labels.index('Exit_B')}
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for step, row in enumerate(trace):
        assert row['step'] == step + 1 and row['from_state'] == state
        t = tau + dt * step
        dose_increment = float(np.interp(t, times, rates[:, state])) * dt
        dose += dose_increment
        x, y = coordinates[state]
        dx, dy = deltas[row['action']]
        destination = (x + dx, y + dy)
        new = coordinates.index(destination) if destination in coordinates else state
        invalid, revisit = new == state, new != state and new in visited
        kind = 'impossible' if invalid else 'exit' if new in exits else 'visited' if revisit else 'regular'
        utility = 0 if dose == 0 else -100 if dose < .3 else -300 if dose < 1 else -800
        reward = settings['sigma_path'] * settings['path_rewards'][kind] + settings['sigma_tox'] * utility * dose_increment
        assert new == row['to_state'] and row['path_reward_kind'] == kind
        assert row['from_cell'] == labels[state] and row['to_cell'] == labels[new]
        assert row['invalid'] == invalid and row['revisit'] == revisit
        assert row['exit_reached'] == (new in exits)
        close(row['absolute_time_s'], t)
        close(row['dose_increment'], dose_increment)
        close(row['cumulative_personal_FED'], dose)
        close(row['tox_utility'], utility)
        close(row['reward'], reward)
        total += reward
        discounted += settings['discount'] ** step * reward
        invalid_count += invalid
        revisit_count += revisit
        state = new
        visited.add(new)
        assert state not in exits or step == len(trace) - 1
    return {'exit_reached': state in exits, 'steps': len(trace), 'time_seconds': dt * len(trace),
        'personal_FED': dose, 'undiscounted_reward': total, 'discounted_reward': discounted,
        'invalid_actions': invalid_count, 'revisits': revisit_count,
        'exit': labels[state] if state in exits else None}


def match_outcome(recomputed, outcome):
    for name, value in recomputed.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool): close(value, outcome[name])
        else: assert value == outcome[name], (name, value, outcome[name])


def independent_policy_rollout(policy, start, tau, settings, times, rates, coordinates, labels, cap):
    """No simulator import: independently verify coverage diagnostics."""
    state, visited, dose = start, {start}, 0.0
    total = discounted = 0.0
    invalid_count = revisits = 0
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    exits = {20, 21}
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    steps = 0
    with torch.no_grad():
        for k in range(cap):
            if state in exits: break
            action = int(policy(torch.eye(24)[state]).argmax())
            dose_increment = float(np.interp(tau + k*dt, times, rates[:, state])) * dt
            dose += dose_increment
            x, y = coordinates[state]
            dx, dy = deltas[action]
            point = (x+dx, y+dy)
            new = coordinates.index(point) if point in coordinates else state
            invalid = new == state
            revisit = new != state and new in visited
            kind = 'impossible' if invalid else 'exit' if new in exits else 'visited' if revisit else 'regular'
            u = 0 if dose == 0 else -100 if dose < .3 else -300 if dose < 1 else -800
            reward = settings['sigma_path']*settings['path_rewards'][kind] + settings['sigma_tox']*u*dose_increment
            total += reward
            discounted += settings['discount']**k * reward
            invalid_count += invalid
            revisits += revisit
            state = new
            visited.add(new)
            steps += 1
    return {'exit_reached': state in exits, 'steps': steps, 'time_seconds': steps*dt,
        'personal_FED': dose, 'undiscounted_reward': total, 'discounted_reward': discounted,
        'invalid_actions': invalid_count, 'revisits': revisits, 'exit': labels[state] if state in exits else None}


def main():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    assert plan['config_sha256'] == sha(CONFIG)
    assert plan['runner_sha256'] == sha(ROOT / 'scripts/run_safestep_reproduction.py')
    assert plan['module_sha256'] == sha(ROOT / 'src/fire_evac/safestep_baseline.py')
    assert plan['paper_sha256'] == sha(cfg['paper']['local_pdf'])
    assert plan['hazard_csv_sha256'] == sha(OUT / 'controlled_synthetic_FER_per_second.csv')
    assert summary['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
    assert summary['RoomB_results_sha256'] == sha(OUT / 'RoomB_method_results.csv')
    data = csv_rows(OUT / 'controlled_synthetic_FER_per_second.csv')
    labels = plan['labels']
    coordinates = [tuple(c) for c in plan['coordinates']]
    times = np.array([float(row['time_s']) for row in data])
    rates = np.array([[float(row[label]) for label in labels] for row in data])
    # Independently evaluate the declared synthetic rate formula, all 36,024 cells.
    hazard_cfg = cfg['controlled_hazard']
    for i, (x, y) in enumerate(coordinates):
        expected = np.zeros(len(times))
        if labels[i] not in ['Exit_A', 'Exit_B']:
            spread = hazard_cfg['initial_spread_cells'] + hazard_cfg['spread_per_second'] * np.minimum(times, hazard_cfg['spread_saturation_seconds'])
            expected = np.minimum((times / hazard_cfg['growth_scale_seconds']) ** 2, 1) * (
                hazard_cfg['background_rate_per_s'] + hazard_cfg['source_rate_per_s'] * np.exp(-.5 * ((x - 17) / spread) ** 2))
            if labels[i] == 'Room_A': expected *= hazard_cfg['fire_room_rate_multiplier']
        assert np.allclose(rates[:, i], expected, rtol=0, atol=1e-15)
    settings = cfg['paper_settings']
    impl = cfg['explicit_implementation_assumptions']
    torch.set_num_threads(1)
    coverage = csv_rows(OUT / 'all_start_final_policy_diagnostic.csv')
    coverage_checked = 0
    trace_count, step_count, transitions = 0, 0, 0
    for case, tau in settings['case_start_times_s'].items():
        for seed in impl['seeds']:
            directory = OUT / 'runs' / f'{case}_seed{seed}'
            log = csv_rows(directory / 'training_trace.csv')
            train = json.loads((directory / 'training_summary.json').read_text(encoding='utf-8'))
            assert len(log) == train['episodes'] == 2000
            assert [int(r['episode']) for r in log] == list(range(1, 2001))
            assert sum(int(r['steps']) for r in log) == train['training_transitions']
            assert sum(int(r['gradient_updates']) for r in log) == train['gradient_updates'] == train['training_transitions'] - 31
            transitions += train['training_transitions']
            assert train['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
            assert train['model_sha256'] == sha(directory / 'policy.pt')
            for row in log:
                episode = int(row['episode'])
                expected = 1 + min((episode-1) / 1000, 1) * (.05 - 1)
                close(row['epsilon'], expected)
                assert math.isfinite(float(row['mean_TD_MSE'])) and 1 <= int(row['steps']) <= 128
            layers = []
            width = 24
            for size in impl['network_hidden_layers']:
                layers += [nn.Linear(width, size), nn.ReLU()]
                width = size
            policy = nn.Sequential(*layers, nn.Linear(width, 4))
            policy.load_state_dict(torch.load(directory / 'policy.pt', weights_only=True, map_location='cpu'))
            policy.eval()
            for start in range(24):
                if start in [20, 21]: continue
                record = next(r for r in coverage if r['case'] == case and int(r['seed']) == seed and r['start_cell'] == labels[start])
                expected = independent_policy_rollout(policy, start, tau, settings, times, rates, coordinates, labels, 128)
                for name, value in expected.items():
                    if isinstance(value, bool): assert record[name] == str(value)
                    elif value is None: assert record[name] == ''
                    elif isinstance(value, (int, float)): close(record[name], value)
                    else: assert record[name] == value
                coverage_checked += 1
            trace = json.loads((directory / 'RoomB_evaluation_trace.json').read_text(encoding='utf-8'))
            with torch.no_grad():
                for row in trace: assert int(policy(torch.eye(24)[row['from_state']]).argmax()) == row['action']
            recomputed = independently_reconstruct_trace(trace, 23, tau, settings, times, rates, coordinates, labels)
            match_outcome(recomputed, train['final_greedy_RoomB'])
            record = next(r for r in summary['RoomB_results'] if r['case'] == case and r['seed'] == seed)
            match_outcome(recomputed, record)
            trace_count += 1
            step_count += len(trace)
        for method in cfg['baselines']:
            detail = json.loads((OUT / f'{case}_{method}_trace.json').read_text(encoding='utf-8'))
            if detail['outcome']['no_path']:
                assert method == 'paper_stationary_FED_threshold_Dijkstra' and case == 'case2_late'
                assert detail['outcome']['personal_FED'] is None and not detail['trace']
            else:
                recomputed = independently_reconstruct_trace(detail['trace'], 23, tau, settings, times, rates, coordinates, labels)
                match_outcome(recomputed, detail['outcome'])
                if method == 'time_expanded_future_FED_oracle': close(recomputed['personal_FED'], detail['oracle_min_FED'])
                trace_count += 1
                step_count += len(detail['trace'])
            if method == 'paper_stationary_FED_threshold_Dijkstra':
                ix = times <= tau
                stationary = np.trapezoid(rates[ix], times[ix], axis=0)
                expected = [label for label, dose in zip(labels, stationary) if dose >= .3]
                assert set(expected) == set(detail['blocked_labels'])
                assert ('Room_B' in expected) == detail['outcome']['no_path']
        # No aggregation hides a timeout: check all seeds and individual values.
        rs = [r for r in summary['RoomB_results'] if r['case'] == case and r['method'] == 'paper_style_DQN_final']
        agg = summary['aggregate'][case]
        assert agg['successful_seeds'] == sum(r['exit_reached'] for r in rs)
        close(agg['mean_personal_FED_all_final_rollouts'], np.mean([r['personal_FED'] for r in rs]))
        assert agg['FED_each_seed'] == [r['personal_FED'] for r in rs]
    assert summary['training_runs'] == 6 and summary['episodes'] == 12000 and summary['training_transitions'] == transitions
    assert len(coverage) == summary['all_start_diagnostic_rows'] == 132
    assert len({(r['case'], r['seed'], r['start_cell']) for r in coverage}) == 132
    assert summary['all_start_successes'] == sum(r['exit_reached'] == 'True' for r in coverage)
    assert not summary['paper_numeric_targets_reproduced'] and not summary['original_FDS_data_available']
    result = {'status': 'completed_method_level_reimplementation_verified', 'training_runs': 6,
        'episodes': 12000, 'training_transitions': transitions,
        'RoomB_realized_traces_independently_integrated': trace_count, 'realized_trace_steps_checked': step_count,
        'map_and_paper_parameters_verified': True, 'synthetic_provenance_verified': True,
        'original_pdf_unchanged': True, 'models_match_greedy_traces': True,
        'personal_and_stationary_FED_kept_distinct': True, 'DA_no_path_not_recorded_as_zero_exposure': True,
        'all_start_diagnostics_independently_recomputed': coverage_checked,
        'paper_numeric_targets_reproduced': False, 'summary_sha256': sha(OUT / 'summary.json'),
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'verifier_sha256': sha(__file__)}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
