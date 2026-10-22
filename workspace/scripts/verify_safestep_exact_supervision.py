"""Independent min-plus forward path labels, state scores and control replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_exact_supervision_20261015'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tolerance=1e-6):
    assert math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=tolerance), (a, b)


def main():
    cfg = read(ROOT/'schemas/safestep_exact_supervision_20261015.json')
    sc = read(ROOT/cfg['source_config'])
    plan = read(OUT/'frozen_manifest.json')
    for group in ['code_sha256', 'protected_previous_files', 'dataset_sha256']:
        for path, digest in plan[group].items():
            assert sha(ROOT/path) == digest, path
    coords = [(i, 0) for i in range(20)]+[(0, 1), (19, 1), (17, 1), (13, -1)]
    labels = [f'R{i+1}' for i in range(20)]+['Exit_A', 'Exit_B', 'Room_A', 'Room_B']
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    exits, cap = {20, 21}, sc['max_steps']
    neighbors = {s: [(a, coords.index((x+dx, y+dy))) for a, (dx, dy) in enumerate(deltas)
                     if (x+dx, y+dy) in coords] for s, (x, y) in enumerate(coords)}
    legal = np.zeros((24, 4), dtype=bool)
    for state in range(24):
        legal[state, [a for a, _ in neighbors[state]]] = True
    old_settings = read(ROOT/sc['source_config'])['paper_settings']
    dt = old_settings['cell_size_m']/old_settings['walking_speed_m_per_s']
    scale, timeout = sc['training']['reward_scale'], sc['training']['timeout_penalty_in_FED_units']
    reference_cache = {}

    def forward_reference(normalized_snapshot):
        key = normalized_snapshot.tobytes()
        if key in reference_cache:
            return reference_cache[key]
        rates = normalized_snapshot.astype(float)*sc['observation']['FER_divisor']
        costs = rates*dt+sc['time_cost_in_FED_per_step']
        adjacency = np.full((24, 24), np.inf)
        for state in range(24):
            if state in exits:
                adjacency[state, state] = 0  # Padding a terminated route has no cost.
            else:
                for _, dest in neighbors[state]:
                    adjacency[state, dest] = costs[state]
        terminal = np.full(24, timeout)
        terminal[list(exits)] = 0
        distances = np.full((24, 24), np.inf)
        np.fill_diagonal(distances, 0)
        continuation = [terminal.copy()]
        # Forward all-source path costs, rather than the production backward Q recurrence.
        for horizon in range(1, cap+1):
            distances = np.min(distances[:, :, None]+adjacency[None, :, :], axis=1)
            continuation.append(np.min(distances+terminal[None, :], axis=1))
        q = np.full((cap, 24, 4), np.nan)
        for k in range(cap):
            for state in range(24):
                if state not in exits:
                    for action, dest in neighbors[state]:
                        q[k, state, action] = -scale*(costs[state]+continuation[cap-k-1][dest])
        reference_cache[key] = q
        return q

    datasets = {}
    label_values = 0
    for split, file in [('training', 'training_dataset.npz'), ('new_state_grid', 'validation_state_grid.npz')]:
        with np.load(OUT/file) as stored:
            data = {k: stored[k] for k in stored.files}
        datasets[split] = data
        obs = data['obs']
        assert obs.shape[1] == 50 and np.isfinite(obs).all()
        states = obs[:, :24].argmax(axis=1)
        assert np.array_equal(obs[:, :24], np.eye(24)[states])
        assert np.array_equal(data['legal'], legal[states])
        for snapshot in np.unique(obs[:, 24:48], axis=0):
            selected = np.all(obs[:, 24:48] == snapshot, axis=1)
            k = np.rint(obs[selected, 48]*cap).astype(int)
            truth = forward_reference(snapshot)[k, states[selected]]
            mask = data['legal'][selected]
            assert np.allclose(data['targets'][selected][mask], truth[mask], rtol=0, atol=5e-6)
            assert np.all(data['targets'][selected][mask] < 0)
            label_values += int(mask.sum())
        assert np.all(data['targets'][~data['legal']] == 0)
    source = ROOT/cfg['source_round']
    source_plan = read(source/'frozen_manifest.json')
    train = datasets['training']
    for seed in cfg['training']['seeds']:
        select = train['source_seed'] == seed
        with np.load(source/'runs'/f'hazard_full_seed{seed}/training_transitions.npz') as archived:
            indices = train['source_index'][select]
            expected = np.array([source_plan['banks']['training'][i]['onset'] for i in archived['scenario_index']])
            assert np.array_equal(indices, np.flatnonzero(archived['k'] >= expected))
            assert np.array_equal(train['obs'][select], archived['obs'][indices])
    old_parameters = {(s['peak'], s['center']) for bank in source_plan['banks'].values() for s in bank}
    assert all((p['peak'], p['center']) not in old_parameters for p in plan['validation_profiles'])
    assert len(plan['validation_profiles']) == 12 and len(plan['control_cases']) == 24
    summary = read(OUT/'summary.json')
    locked = read(OUT/'models_locked.json')
    assert all(r['completed_at_utc'] < locked['locked_at_utc'] for r in locked['models'])
    assert locked['locked_at_utc'] < summary['evaluation_started_at_utc']
    assert summary['models_locked_sha256'] == sha(OUT/'models_locked.json')
    torch.set_num_threads(1)
    models = {}
    for seed in cfg['training']['seeds']:
        for prefix, path in [('legacy_DQN', source/'runs'/f'hazard_full_seed{seed}/policy.pt'),
                             ('supervised_Q', OUT/'models'/f'seed{seed}/policy.pt')]:
            layers, width = [], 50
            for h in cfg['training']['hidden_layers']:
                layers.extend([nn.Linear(width, h), nn.ReLU()])
                width = h
            model = nn.Sequential(*layers, nn.Linear(width, 4))
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            model.eval()
            models[f'{prefix}_seed{seed}'] = model
    with np.load(OUT/'validation_Q_predictions.npz') as stored:
        saved = {key: stored[key] for key in stored.files}
    for split, data in datasets.items():
        for method, model in models.items():
            with torch.inference_mode():
                q = np.vstack([model(torch.from_numpy(data['obs'][i:i+2048])).numpy() for i in range(0, len(data['obs']), 2048)])
            if split == 'new_state_grid':
                assert np.array_equal(q, saved[method])
            def check_metric(mask, report):
                predictions, truth, allowed = q[mask], data['targets'][mask], data['legal'][mask]
                scores = np.where(allowed, predictions, -np.inf)
                actions = scores.argmax(axis=1)
                errors = predictions[allowed]-truth[allowed]
                regret = np.where(allowed, truth, -np.inf).max(axis=1)-truth[np.arange(len(actions)), actions]
                close(report['legal_Q_MAE'], np.abs(errors).mean())
                close(report['optimal_action_accuracy'], np.mean(regret <= cfg['ranking_tolerance']))
                close(report['positive_selected_Q_fraction'], np.mean(scores.max(axis=1) > cfg['ranking_tolerance']))
            report = summary['state_metrics'][split][method]
            check_metric(np.ones(len(q), dtype=bool), report)
            if split == 'new_state_grid':
                ordinary = data['obs'][:, 48] <= 96/cap
                check_metric(ordinary, report['ordinary_k_le_96'])
                check_metric(~ordinary, report['near_deadline_k_gt_96'])
    cases = {c['case_id']: c for c in plan['control_cases']}
    traces = [json.loads(line) for line in (OUT/'control_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    with (OUT/'control_results.csv').open(encoding='utf-8-sig', newline='') as handle:
        results = list(csv.DictReader(handle))
    assert len(traces) == len(results) == cfg['planned_control_episodes'] == 168
    assert [(r['case_id'], r['method']) for r in traces] == [(r['case_id'], r['method']) for r in plan['control_schedule']]
    step_count = 0
    for row, result in zip(traces, results):
        case = cases[row['case_id']]
        rates = np.array([sc['background_rate_per_second']+case['peak']*math.exp(-.5*((x-case['center'])/sc['spread_cells'])**2) for x, y in coords])
        rates[list(exits)] = 0
        state, dose, visited = labels.index(case['start']), 0., {labels.index(case['start'])}
        revisits = 0
        for k, (step, common) in enumerate(zip(row['trace'], row['observations'])):
            expected = np.zeros(50, dtype=np.float32)
            expected[state] = 1
            expected[24:48] = rates.astype(np.float32)/sc['observation']['FER_divisor']
            expected[48], expected[49] = k/cap, dose
            assert np.allclose(common, expected, rtol=0, atol=1e-7)
            a = step['action']
            assert legal[state, a] and step['from_state'] == state
            if row['method'] in models:
                with torch.inference_mode():
                    q = models[row['method']](torch.from_numpy(np.asarray(common, dtype=np.float32))).numpy()
                assert a == max([a for a, _ in neighbors[state]], key=lambda a: (q[a], -a))
            else:
                truth = forward_reference(np.asarray(common[24:48], dtype=np.float32))[k, state]
                assert max(truth[np.isfinite(truth)])-truth[a] < 1e-6
            new = dict(neighbors[state])[a]
            dose += dt*rates[state]
            close(step['dose_increment'], dt*rates[state], 1e-10)
            close(step['cumulative_personal_FED'], dose, 1e-9)
            assert step['to_state'] == new
            revisits += new in visited
            visited.add(new)
            state = new
            step_count += 1
        success = state in exits
        outcome = row['outcome']
        assert outcome['exit_reached'] == success and result['exit_reached'] == str(success)
        assert success or len(row['trace']) == cap
        assert outcome['steps'] == len(row['trace']) and outcome['revisits'] == revisits
        close(outcome['personal_FED'], dose)
        if success:
            close(outcome['J'], dose+sc['time_cost_in_FED_per_step']*len(row['trace']))
        members = set()
        for witness in row['closed_walk_witnesses']:
            begin, end = witness['begin_k'], witness['end_k_exclusive']
            assert row['trace'][begin]['from_state'] == row['trace'][end-1]['to_state']
            members.update(range(begin, end))
        assert len(members) == outcome['closed_walk_steps']
    for method, aggregate in summary['control'].items():
        rows = [r['outcome'] for r in traces if r['method'] == method]
        successful = [r for r in rows if r['exit_reached']]
        assert aggregate['successes'] == len(successful)
        assert aggregate['closed_walk_steps'] == sum(r['closed_walk_steps'] for r in rows)
        if successful:
            close(aggregate['mean_J_success_only'], np.mean([r['J'] for r in successful]))
    for name, key in [('control_results.csv', 'control_results_sha256'), ('control_traces.jsonl', 'control_traces_sha256'),
                      ('validation_Q_predictions.npz', 'Q_predictions_sha256')]:
        assert summary[key] == sha(OUT/name)
    for path, digest in plan['protected_previous_files'].items():
        assert sha(ROOT/path) == digest
    value = {'status': 'exact_supervision_labels_metrics_and_controls_verified',
        'training_states': len(datasets['training']['obs']), 'new_validation_states': len(datasets['new_state_grid']['obs']),
        'legal_Q_labels_checked_by_forward_min_plus_paths': label_values,
        'stationary_profile_references': len(reference_cache), 'models_scored': len(models),
        'control_episodes_recomputed': len(traces), 'control_steps_recomputed': step_count,
        'post_reveal_training_filter_and_original_inputs_verified': True,
        'all_supervised_models_locked_before_scores': True, 'old_artifacts_unchanged': True,
        'optimizer_history_not_independently_replayed': True,
        'new_RL_updates': 0, 'LLM_requests': 0,
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
