"""One frozen, fixed-budget observed-hazard DQN round; no adaptive search."""
import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from fire_evac.safestep_baseline import SmallMap, SafeStepEpisode
from fire_evac.safestep_dynamic_probe import ControlTickField, snapshot_path, future_j_plan
from fire_evac.safestep_shared_observation import (encode_observation, effective_observation,
    build_network, masked_td_targets, rate_table, legal_matrix, training_reward)

CONFIG = ROOT / 'schemas/safestep_shared_observation_20261014.json'
OUT = ROOT / 'outputs/safestep_shared_observation_20261014'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare():
    cfg = read(CONFIG)
    source = read(ROOT / cfg['source_config'])
    previous_path = ROOT / cfg['source_probe_completion']
    previous = read(previous_path)
    old_plan = read(previous_path.parent / 'frozen_manifest.json')
    protected = {**old_plan['protected_previous_files'], **previous['artifact_sha256'],
                 cfg['source_probe_completion']: sha(previous_path)}
    for path, digest in protected.items():
        assert sha(ROOT / path) == digest, path
    banks = {}
    for split in ['training', 'development', 'confirmation']:
        settings = cfg[f'{split}_bank']
        scenarios = []
        for side in settings['centers']:
            for onset, peak, center in itertools.product(settings['onsets'], settings['peaks'], settings['centers'][side]):
                scenarios.append({'scenario_id': f'{split}_{len(scenarios):02d}', 'split': split,
                    'side': side, 'onset': onset, 'peak': peak, 'center': center})
        banks[split] = scenarios
    tuples = lambda split: {(s['onset'], s['peak'], s['center']) for s in banks[split]}
    assert len(banks['training']) == 24 and len(banks['development']) == 8 and len(banks['confirmation']) == 24
    assert not tuples('confirmation') & (tuples('training') | tuples('development'))
    OUT.mkdir(exist_ok=True)
    schedules = {}
    for seed in cfg['seeds']:
        rng = np.random.default_rng(seed + 17000)
        values = np.concatenate([rng.permutation(24) for _ in range(cfg['training']['transitions_per_model']//24 + 1)]).astype(np.int16)
        path = OUT / f'episode_schedule_seed{seed}.npy'
        if path.exists():
            assert np.array_equal(np.load(path), values)
        else:
            np.save(path, values)
        schedules[str(path.relative_to(ROOT))] = sha(path)
    jobs = [{'scenario_id': s['scenario_id'], 'method': method}
        for split in ['development', 'confirmation'] for s in banks[split] for method in cfg['evaluation_methods']]
    random.Random(cfg['order_seed']).shuffle(jobs)
    code = ['schemas/safestep_shared_observation_20261014.json',
        'src/fire_evac/safestep_shared_observation.py', 'scripts/run_safestep_shared_observation.py',
        'scripts/verify_safestep_shared_observation.py', 'tests/test_safestep_shared_observation.py']
    manifest = {'config_sha256': sha(CONFIG), 'code_sha256': {p: sha(ROOT / p) for p in code},
        'protected_previous_files': protected, 'episode_schedules_sha256': schedules,
        'banks': banks, 'evaluation_schedule': jobs, 'planned_training_transitions': cfg['planned_training_transitions'],
        'planned_evaluation_episodes': len(jobs), 'LLM_requests': 0, 'old_final_scenarios_accessed': False}
    target = OUT / 'frozen_manifest.json'
    if target.exists():
        prior = read(target)
        assert {k: v for k, v in prior.items() if k != 'frozen_at_utc'} == manifest
        return cfg, source, prior
    manifest['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(target, manifest)
    print(json.dumps({'status': 'shared_observation_round_frozen',
        'models': 6, 'transitions': cfg['planned_training_transitions'], 'evaluation_episodes': len(jobs)}), flush=True)
    return cfg, source, manifest


def train_one(cfg, source, manifest, arm, seed):
    directory = OUT / 'runs' / f'{arm}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / 'training_summary.json'
    if summary_path.exists():
        previous = read(summary_path)
        assert previous['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
        assert previous['model_sha256'] == sha(directory / 'policy.pt')
        return previous
    guard = directory / 'started.json'
    if guard.exists():
        raise RuntimeError(f'Preserve incomplete run {directory.name}; do not silently restart.')
    dump(guard, {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(OUT / 'frozen_manifest.json')})
    tr = cfg['training']
    torch.manual_seed(seed)
    torch.set_num_threads(tr['torch_threads'])
    rng = np.random.default_rng(seed + 500)
    model = build_network(tr['hidden_layers'])
    initial_sha = parameter_sha(model)
    target = build_network(tr['hidden_layers'])
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'])
    grid = SmallMap.paper_case_map()
    legal = legal_matrix(grid)
    settings = source['paper_settings']
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    fields = [rate_table(grid, s, cfg) for s in manifest['banks']['training']]
    schedule = np.load(OUT / f'episode_schedule_seed{seed}.npy')
    budget = tr['transitions_per_model']
    obs = np.empty((budget, 50), dtype=np.float32)
    nxt = np.empty_like(obs)
    actions = np.empty(budget, dtype=np.int8)
    rewards = np.empty(budget, dtype=np.float32)
    dones = np.empty(budget, dtype=bool)
    next_legal = np.empty((budget, 4), dtype=bool)
    sid = np.empty(budget, dtype=np.int16)
    episode_ids = np.empty(budget, dtype=np.int32)
    ks = np.empty(budget, dtype=np.int16)
    doses = np.empty(budget, dtype=np.float64)
    episodes, progress, visit_counts = [], [], np.zeros((24, 4), dtype=np.int64)
    updates = episode_id = loss_count = 0
    loss_sum = 0.
    env = None
    started = time.monotonic()
    for t in range(budget):
        if env is None or env.done:
            scenario_index = int(schedule[episode_id])
            rates = fields[scenario_index]
            env = SafeStepEpisode(grid, ControlTickField(rates, cfg['tau'], dt), settings,
                                   cfg['tau'], cfg['max_steps'], grid.labels.index(cfg['start']))
            episode_id += 1
            episode_reward = 0.
        old, k = env.state, env.steps
        common = encode_observation(old, rates[k], k, env.fed, cfg['max_steps'], cfg['observation']['FER_divisor'])
        effective = effective_observation(common, arm)
        epsilon = tr['epsilon_start'] + min(t/tr['epsilon_decay_steps'], 1)*(tr['epsilon_end']-tr['epsilon_start'])
        available = np.flatnonzero(legal[old])
        if rng.random() < epsilon:
            action = int(rng.choice(available))
        else:
            with torch.inference_mode():
                q = model(torch.from_numpy(effective)).numpy()
            action = int(max(available.tolist(), key=lambda a: (q[a], -a)))
        _, _, done, info = env.step(action)
        reward = training_reward(info['dose_increment'], info['exit_reached'], done, cfg)
        common_next = encode_observation(env.state, rates[env.steps], env.steps, env.fed,
                                        cfg['max_steps'], cfg['observation']['FER_divisor'])
        obs[t], nxt[t] = effective, effective_observation(common_next, arm)
        actions[t], rewards[t], dones[t] = action, reward, done
        next_legal[t] = legal[env.state]
        sid[t], episode_ids[t], ks[t], doses[t] = scenario_index, episode_id, k, info['dose_increment']
        visit_counts[old, action] += 1
        episode_reward += reward
        if t+1 >= tr['warmup_transitions'] and (t+1) % tr['update_every_steps'] == 0:
            low = max(0, t+1-tr['replay_capacity'])
            ix = rng.choice(np.arange(low, t+1), tr['batch_size'], replace=False)
            q_selected = model(torch.from_numpy(obs[ix])).gather(1, torch.from_numpy(actions[ix].astype(np.int64))[:, None]).squeeze(1)
            with torch.no_grad():
                target_q = target(torch.from_numpy(nxt[ix]))
                expected = masked_td_targets(torch.from_numpy(rewards[ix]), torch.from_numpy(dones[ix]),
                    target_q, torch.from_numpy(next_legal[ix]), tr['discount'])
            loss = torch.nn.functional.mse_loss(q_selected, expected)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; preserve failed run, do not retune automatically.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            loss_sum += float(loss.item())
            loss_count += 1
        if (t+1) % tr['target_sync_steps'] == 0:
            target.load_state_dict(model.state_dict())
        if done or t+1 == budget:
            episodes.append({'episode': episode_id, 'scenario_index': scenario_index,
                'end_global_step': t+1, 'steps': env.steps, 'episode_complete': bool(done),
                'exit_reached': env.state in grid.exits, 'personal_FED': env.fed, 'return': episode_reward})
        if (t+1) % 10000 == 0:
            progress.append({'global_step': t+1, 'updates': updates, 'episodes_started': episode_id,
                'epsilon': epsilon, 'mean_TD_MSE': loss_sum/loss_count if loss_count else None,
                'wall_seconds': time.monotonic()-started})
            print(json.dumps({'arm': arm, 'seed': seed, **progress[-1]}), flush=True)
            loss_sum = 0.
            loss_count = 0
    torch.save(model.state_dict(), directory / 'policy.pt')
    np.savez_compressed(directory / 'training_transitions.npz', obs=obs, next_obs=nxt, actions=actions,
        rewards=rewards, dones=dones, next_legal=next_legal, scenario_index=sid,
        episode_id=episode_ids, k=ks, dose_increment=doses)
    table(directory / 'episodes.csv', episodes)
    table(directory / 'progress.csv', progress)
    dump(directory / 'state_action_visits.json', {'labels': list(grid.labels), 'counts': visit_counts.tolist()})
    result = {'arm': arm, 'seed': seed, 'transitions': budget, 'gradient_updates': updates,
        'episodes_started': episode_id, 'episodes_completed': sum(r['episode_complete'] for r in episodes),
        'training_successes': sum(r['episode_complete'] and r['exit_reached'] for r in episodes),
        'initial_parameter_sha256': initial_sha, 'final_parameter_sha256': parameter_sha(model),
        'model_sha256': sha(directory / 'policy.pt'), 'archive_sha256': sha(directory / 'training_transitions.npz'),
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'wall_seconds': time.monotonic()-started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_performed': False}
    dump(summary_path, result)
    print(json.dumps({'status': 'trained_fixed_budget', **result}), flush=True)
    return result


def lock_models(cfg):
    models = []
    for arm, seed in itertools.product(cfg['arms'], cfg['seeds']):
        path = OUT / 'runs' / f'{arm}_seed{seed}/training_summary.json'
        result = read(path)
        assert result['transitions'] == cfg['training']['transitions_per_model']
        assert result['model_sha256'] == sha(path.parent / 'policy.pt')
        models.append({'arm': arm, 'seed': seed, 'model_sha256': result['model_sha256'],
                       'training_summary_sha256': sha(path), 'completed_at_utc': result['completed_at_utc']})
    target = OUT / 'models_locked.json'
    value = {'models': models, 'manifest_sha256': sha(OUT / 'frozen_manifest.json'),
             'training_transitions': sum(read(OUT / 'runs' / f"{r['arm']}_seed{r['seed']}/training_summary.json")['transitions'] for r in models)}
    if target.exists():
        prior = read(target)
        assert {k: v for k, v in prior.items() if k != 'locked_at_utc'} == value
        return prior
    value['locked_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(target, value)
    return value


def evaluate(cfg, source, manifest):
    locked = lock_models(cfg)
    if (OUT / 'evaluation_results.csv').exists():
        raise RuntimeError('Preserve completed evaluation. No rescore or adaptive search.')
    grid = SmallMap.paper_case_map()
    legal = legal_matrix(grid)
    settings = source['paper_settings']
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    start = grid.labels.index(cfg['start'])
    penalty = cfg['time_cost_in_FED_per_step']
    models = {}
    torch.set_num_threads(1)
    for item in locked['models']:
        arm, seed = item['arm'], item['seed']
        model = build_network(cfg['training']['hidden_layers'])
        model.load_state_dict(torch.load(OUT / 'runs' / f'{arm}_seed{seed}/policy.pt', weights_only=True, map_location='cpu'))
        model.eval()
        models[f'{arm}_seed{seed}'] = model
    scenarios = {s['scenario_id']: s for split in ['development', 'confirmation'] for s in manifest['banks'][split]}
    fields = {sid: rate_table(grid, s, cfg) for sid, s in scenarios.items()}
    oracles = {sid: future_j_plan(grid, field, dt, penalty, start) for sid, field in fields.items()}
    results, traces = [], []
    eval_started = datetime.now(timezone.utc).isoformat()
    assert eval_started > locked['locked_at_utc']
    for job in manifest['evaluation_schedule']:
        sid, method = job['scenario_id'], job['method']
        scenario, field = scenarios[sid], fields[sid]
        env = SafeStepEpisode(grid, ControlTickField(field, cfg['tau'], dt), settings, cfg['tau'], cfg['max_steps'], start)
        observations, effective_inputs = [], []
        path = oracles[sid][0] if method == 'future_J_oracle' else snapshot_path(grid, start, np.zeros(24), dt, penalty) if method == 'nearest_exit' else None
        while not env.done:
            common = encode_observation(env.state, field[env.steps], env.steps, env.fed,
                                        cfg['max_steps'], cfg['observation']['FER_divisor'])
            if method in models:
                arm = method.rsplit('_seed', 1)[0]
                effective = effective_observation(common, arm)
                with torch.inference_mode():
                    q = models[method](torch.from_numpy(effective)).numpy()
                action = max(np.flatnonzero(legal[env.state]).tolist(), key=lambda a: (q[a], -a))
            elif method == 'snapshot_replan':
                effective = common.copy()
                # Consume exactly the normalized current rates available to DQN.
                decoded = common[24:48].astype(float)*cfg['observation']['FER_divisor']
                action = snapshot_path(grid, env.state, decoded, dt, penalty)[0]
            else:
                effective = common.copy()
                action = path[env.steps]
            observations.append(common.tolist())
            effective_inputs.append(effective.tolist())
            env.step(action)
        success = env.state in grid.exits
        J = env.fed+penalty*env.steps if success else None
        outcome = {'exit_reached': success, 'steps': env.steps, 'time_seconds': env.steps*dt,
            'personal_FED': env.fed, 'J': J, 'exit': grid.labels[env.state] if success else None,
            'invalid_actions': sum(r['invalid'] for r in env.trace), 'revisits': sum(r['revisit'] for r in env.trace),
            'J_regret': J-oracles[sid][1] if success else None, 'oracle_J': oracles[sid][1]}
        assert not success or outcome['J_regret'] >= -1e-10
        results.append({**scenario, 'method': method, **outcome})
        traces.append({**scenario, 'method': method, 'outcome': outcome,
            'common_observations': observations, 'effective_inputs': effective_inputs, 'trace': env.trace})
    table(OUT / 'evaluation_results.csv', results)
    with (OUT / 'evaluation_traces.jsonl').open('w', encoding='utf-8') as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    aggregate = {}
    for split in ['development', 'confirmation']:
        aggregate[split] = {}
        for method in cfg['evaluation_methods']:
            rows = [r for r in results if r['split'] == split and r['method'] == method]
            successes = [r for r in rows if r['exit_reached']]
            aggregate[split][method] = {'episodes': len(rows), 'successes': len(successes), 'failures': len(rows)-len(successes),
                **{f'mean_{key}_success_only': float(np.mean([r[key] for r in successes])) if successes else None
                   for key in ['personal_FED', 'time_seconds', 'J', 'J_regret']}}
    lookup = {(r['scenario_id'], r['method']): r for r in results}
    pairs = []
    for seed in cfg['seeds']:
        common = [s['scenario_id'] for s in manifest['banks']['confirmation'] if all(
            lookup[(s['scenario_id'], f'{arm}_seed{seed}')]['exit_reached'] for arm in cfg['arms'])]
        pairs.append({'seed': seed, 'common_successes': len(common),
            'blind_failures': aggregate['confirmation'][f'hazard_blind_seed{seed}']['failures'],
            'full_failures': aggregate['confirmation'][f'hazard_full_seed{seed}']['failures'],
            'mean_J_full_minus_blind_common_successes': float(np.mean([
                lookup[(sid, f'hazard_full_seed{seed}')]['J']-lookup[(sid, f'hazard_blind_seed{seed}')]['J'] for sid in common])) if common else None})
    conf = aggregate['confirmation']
    planner = conf['snapshot_replan']
    all_full_success = all(conf[f'hazard_full_seed{seed}']['failures'] == 0 for seed in cfg['seeds'])
    signal = all_full_success and all(row['mean_J_full_minus_blind_common_successes'] is not None and row['mean_J_full_minus_blind_common_successes'] < 0 for row in pairs)
    added = all_full_success and all(
        conf[f'hazard_full_seed{seed}']['mean_J_success_only'] <= .95*planner['mean_J_success_only'] and
        conf[f'hazard_full_seed{seed}']['mean_personal_FED_success_only'] <= planner['mean_personal_FED_success_only'] and
        conf[f'hazard_full_seed{seed}']['mean_time_seconds_success_only'] <= 1.05*planner['mean_time_seconds_success_only'] for seed in cfg['seeds'])
    near = all_full_success and all(conf[f'hazard_full_seed{seed}']['mean_J_success_only'] <= 1.05*planner['mean_J_success_only'] for seed in cfg['seeds'])
    summary = {'status': 'completed_single_fixed_budget_shared_observation_round',
        'training_models': len(locked['models']), 'training_transitions': locked['training_transitions'],
        'evaluation_episodes': len(results), 'aggregate': aggregate, 'paired_confirmation_effects': pairs,
        'frozen_decisions': {'observation_signal_supported': signal, 'practical_RL_added_value': added, 'near_planner': near},
        'evaluation_started_at_utc': eval_started, 'models_locked_sha256': sha(OUT / 'models_locked.json'),
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'results_sha256': sha(OUT / 'evaluation_results.csv'),
        'traces_sha256': sha(OUT / 'evaluation_traces.jsonl'), 'LLM_requests': 0, 'PPO_updates': 0,
        'old_final_scenarios_accessed': False, 'decision': 'Close after verification/report; no additional training or parameter search.'}
    dump(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    cfg, source, manifest = prepare()
    if args.mode == 'train':
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                train_one(cfg, source, manifest, arm, seed)
        locked = lock_models(cfg)
        print(json.dumps({'status': 'all_six_models_locked', 'transitions': locked['training_transitions']}), flush=True)
    elif args.mode == 'evaluate':
        evaluate(cfg, source, manifest)
