"""Paper-parameter DQN baseline reimplementation, with explicit synthetic data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn

from fire_evac.safestep_baseline import (SmallMap, HazardField, SafeStepEpisode,
    controlled_hazard, shortest_path, future_fed_oracle, evaluate_actions)

CONFIG = ROOT / 'schemas/safestep_reproduction_20261014.json'
OUT = ROOT / 'outputs/safestep_reproduction_20261014'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def table(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    grid = SmallMap.paper_case_map()
    OUT.mkdir(exist_ok=True)
    profile_path = OUT / 'controlled_synthetic_FER_per_second.csv'
    if not profile_path.exists(): controlled_hazard(grid, cfg['controlled_hazard']).to_csv(profile_path)
    hazard = HazardField.from_csv(profile_path, grid)
    assert np.array_equal(hazard.fer_per_s, controlled_hazard(grid, cfg['controlled_hazard']).fer_per_s)
    plan = {'config_sha256': sha(CONFIG), 'runner_sha256': sha(__file__),
        'module_sha256': sha(ROOT / 'src/fire_evac/safestep_baseline.py'),
        'paper_sha256': sha(cfg['paper']['local_pdf']), 'hazard_csv_sha256': sha(profile_path),
        'labels': list(grid.labels), 'coordinates': [list(c) for c in grid.coordinates],
        'start_label': grid.labels[grid.start], 'exit_labels': [grid.labels[s] for s in grid.exits],
        'case_times': cfg['paper_settings']['case_start_times_s'],
        'seeds': cfg['explicit_implementation_assumptions']['seeds'],
        'episodes_per_seed_and_case': cfg['paper_settings']['episodes_per_run'],
        'reproduction_level': cfg['reproduction_level'],
        'software': {'Python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
        'PPO_updates': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False}
    path = OUT / 'frozen_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        assert {k: v for k, v in previous.items() if k != 'frozen_at_utc'} == plan
        return cfg, previous, grid, hazard
    plan['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(path, plan)
    print(json.dumps({'status': 'frozen_method_reimplementation', 'cells': 24, 'cases': 2,
        'seeds': plan['seeds'], 'episodes_per_run': 2000, 'original_FDS': False}), flush=True)
    return cfg, plan, grid, hazard


def network(n, hidden):
    layers, width = [], n
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, 4))


def greedy_actions(policy, grid, hazard, cfg, tau, start=None):
    env = SafeStepEpisode(grid, hazard, cfg['paper_settings'], tau,
        cfg['explicit_implementation_assumptions']['max_episode_steps'], start)
    one_hot = torch.eye(len(grid.labels))
    actions = []
    with torch.no_grad():
        while not env.done:
            action = int(policy(one_hot[env.state]).argmax().item())
            actions.append(action)
            env.step(action)
    return actions


def train_one(cfg, plan, grid, hazard, case, tau, seed):
    destination = OUT / 'runs' / f'{case}_seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / 'training_summary.json'
    if complete.exists():
        prior = json.loads(complete.read_text(encoding='utf-8'))
        assert prior['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
        assert prior['model_sha256'] == sha(destination / 'policy.pt')
        return prior
    if (destination / 'training_trace.csv').exists():
        raise RuntimeError('Partial training exists; preserve it and resume only with an explicitly separate attempt directory')
    settings, impl = cfg['paper_settings'], cfg['explicit_implementation_assumptions']
    torch.set_num_threads(impl['torch_threads'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, capacity = len(grid.labels), settings['replay_capacity']
    policy = network(n, impl['network_hidden_layers'])
    target = network(n, impl['network_hidden_layers'])
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings['learning_rate'])
    loss_fn = nn.MSELoss()
    one_hot = torch.eye(n)
    states = np.zeros(capacity, dtype=np.int64)
    actions = np.zeros(capacity, dtype=np.int64)
    rewards = np.zeros(capacity, dtype=np.float32)
    next_states = np.zeros(capacity, dtype=np.int64)
    dones = np.zeros(capacity, dtype=np.float32)
    count, global_step, updates = 0, 0, 0
    log, started = [], time.monotonic()
    for episode in range(settings['episodes_per_run']):
        env = SafeStepEpisode(grid, hazard, settings, tau, impl['max_episode_steps'])
        fraction = min(episode / impl['epsilon_decay_episodes'], 1)
        epsilon = impl['epsilon_start'] + fraction * (impl['epsilon_end'] - impl['epsilon_start'])
        loss_total, episode_updates = 0.0, 0
        while not env.done:
            old = env.state
            if rng.random() < epsilon: action = int(rng.integers(4))
            else:
                with torch.no_grad(): action = int(policy(one_hot[old]).argmax().item())
            new, reward, done, _ = env.step(action)
            slot = count % capacity
            states[slot], actions[slot], rewards[slot], next_states[slot], dones[slot] = old, action, reward, new, done
            count += 1
            global_step += 1
            if count >= impl['replay_warmup_transitions']:
                ix = rng.choice(min(count, capacity), settings['batch_size'], replace=False)
                batch_states = torch.from_numpy(states[ix])
                batch_actions = torch.from_numpy(actions[ix])
                q = policy(one_hot[batch_states]).gather(1, batch_actions[:, None]).squeeze(1)
                with torch.no_grad():
                    future = target(one_hot[torch.from_numpy(next_states[ix])]).max(dim=1).values
                    expected = torch.from_numpy(rewards[ix]) + settings['discount'] * (1 - torch.from_numpy(dones[ix])) * future
                loss = loss_fn(q, expected)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item())
                episode_updates += 1
                updates += 1
            if global_step % settings['target_sync_steps'] == 0:
                target.load_state_dict(policy.state_dict())
        log.append({'episode': episode + 1, 'epsilon': epsilon, 'steps': env.steps,
            'exit_reached': env.state in grid.exits, 'personal_FED': env.fed,
            'total_reward': sum(row['reward'] for row in env.trace),
            'mean_TD_MSE': loss_total / episode_updates if episode_updates else 0,
            'gradient_updates': episode_updates, 'global_step': global_step})
        if (episode + 1) % 200 == 0:
            print(json.dumps({'case': case, 'seed': seed, 'episode': episode + 1,
                'training_steps': global_step, 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
    table(destination / 'training_trace.csv', log)
    torch.save(policy.state_dict(), destination / 'policy.pt')
    evaluated, trace = evaluate_actions(grid, hazard, settings, tau, impl['max_episode_steps'],
        greedy_actions(policy, grid, hazard, cfg, tau))
    summary = {'case': case, 'seed': seed, 'episodes': settings['episodes_per_run'],
        'training_transitions': global_step, 'gradient_updates': updates,
        'wall_seconds': time.monotonic() - started, 'final_greedy_RoomB': evaluated,
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'model_sha256': sha(destination / 'policy.pt')}
    dump(complete, summary)
    dump(destination / 'RoomB_evaluation_trace.json', trace)
    print(json.dumps({'case': case, 'seed': seed, 'status': 'trained', 'final_greedy': evaluated}), flush=True)
    return summary


def analyze(cfg, plan, grid, hazard):
    settings, impl = cfg['paper_settings'], cfg['explicit_implementation_assumptions']
    results, coverage = [], []
    expected_runs = len(impl['seeds']) * len(settings['case_start_times_s'])
    for case, tau in settings['case_start_times_s'].items():
        for method in cfg['baselines']:
            blocked, oracle_min = [], None
            if method == 'paper_stationary_FED_threshold_Dijkstra':
                path, blocked = shortest_path(grid, grid.start, hazard, tau, settings['FED_thresholds'][0])
            elif method == 'time_expanded_future_FED_oracle':
                path, oracle_min = future_fed_oracle(grid, hazard, settings, tau, impl['max_episode_steps'])
            else: path, blocked = shortest_path(grid, grid.start)
            outcome, trace = evaluate_actions(grid, hazard, settings, tau, impl['max_episode_steps'], path)
            results.append({'case': case, 'method': method, 'seed': '', **outcome})
            dump(OUT / f'{case}_{method}_trace.json', {'blocked_labels': [grid.labels[s] for s in blocked],
                'oracle_min_FED': oracle_min, 'outcome': outcome, 'trace': trace})
        for seed in impl['seeds']:
            directory = OUT / 'runs' / f'{case}_seed{seed}'
            if not (directory / 'training_summary.json').exists():
                dump(OUT / 'campaign_status.json', {'status': 'partial_training', 'expected_runs': expected_runs})
                return
            policy = network(len(grid.labels), impl['network_hidden_layers'])
            policy.load_state_dict(torch.load(directory / 'policy.pt', map_location='cpu', weights_only=True))
            policy.eval()
            for start in range(len(grid.labels)):
                if start in grid.exits: continue
                outcome, trace = evaluate_actions(grid, hazard, settings, tau, impl['max_episode_steps'],
                    greedy_actions(policy, grid, hazard, cfg, tau, start), start)
                coverage.append({'case': case, 'method': 'paper_style_DQN_final', 'seed': seed,
                    'start_cell': grid.labels[start], **outcome})
                if start == grid.start:
                    results.append({'case': case, 'method': 'paper_style_DQN_final', 'seed': seed, **outcome})
                    dump(directory / 'RoomB_evaluation_trace.json', trace)
    table(OUT / 'RoomB_method_results.csv', results)
    table(OUT / 'all_start_final_policy_diagnostic.csv', coverage)
    aggregate = {}
    for case in settings['case_start_times_s']:
        rs = [r for r in results if r['case'] == case and r['method'] == 'paper_style_DQN_final']
        success = [r for r in rs if r['exit_reached']]
        base = next(r for r in results if r['case'] == case and r['method'] == 'unblocked_shortest_Dijkstra')
        aggregate[case] = {'training_seeds': len(rs), 'successful_seeds': len(success),
            'mean_personal_FED_all_final_rollouts': float(np.mean([r['personal_FED'] for r in rs])),
            'FED_each_seed': [r['personal_FED'] for r in rs], 'steps_each_seed': [r['steps'] for r in rs],
            'exit_each_seed': [r['exit'] for r in rs],
            'unblocked_shortest_FED': base['personal_FED'],
            'mean_FED_relative_change_vs_unblocked_shortest': float(np.mean([r['personal_FED'] for r in rs])) / base['personal_FED'] - 1,
            'caution': 'Timeout trajectories included; exposure alone never establishes successful evacuation.'}
    summaries = [json.loads((p / 'training_summary.json').read_text()) for p in sorted((OUT / 'runs').iterdir())]
    summary = {'status': 'completed_method_level_reimplementation_on_synthetic_FER',
        'reproduction_level': cfg['reproduction_level'], 'training_runs': expected_runs,
        'episodes': sum(s['episodes'] for s in summaries), 'training_transitions': sum(s['training_transitions'] for s in summaries),
        'gradient_updates': sum(s['gradient_updates'] for s in summaries),
        'RoomB_results': results, 'aggregate': aggregate, 'all_start_diagnostic_rows': len(coverage),
        'all_start_successes': sum(r['exit_reached'] for r in coverage),
        'paper_numeric_targets_reproduced': False, 'original_FDS_data_available': False,
        'LLM_requests': 0, 'PPO_updates': 0, 'old_final_scenarios_accessed': False,
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'RoomB_results_sha256': sha(OUT / 'RoomB_method_results.csv')}
    dump(OUT / 'summary.json', summary)
    dump(OUT / 'campaign_status.json', {'status': summary['status'], 'training_runs_completed': len(summaries),
        'expected_runs': expected_runs, 'original_FDS': False})
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    args = parser.parse_args()
    cfg, plan, grid, hazard = prepare()
    torch.set_num_threads(cfg['explicit_implementation_assumptions']['torch_threads'])
    if args.mode == 'run':
        for case, tau in cfg['paper_settings']['case_start_times_s'].items():
            for seed in cfg['explicit_implementation_assumptions']['seeds']:
                train_one(cfg, plan, grid, hazard, case, tau, seed)
    if args.mode in ['run', 'analyze']: analyze(cfg, plan, grid, hazard)
