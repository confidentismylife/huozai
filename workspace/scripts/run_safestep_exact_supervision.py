"""One fixed-budget exact-Q supervised diagnosis, then locked evaluation."""
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
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from fire_evac.safestep_baseline import SmallMap, SafeStepEpisode
from fire_evac.safestep_dynamic_probe import ControlTickField, snapshot_path
from fire_evac.safestep_shared_observation import encode_observation, build_network, legal_matrix
from fire_evac.safestep_exact_supervision import snapshot_teacher, legal_mse, quality
from fire_evac.safestep_q_cache_audit import closed_walk_members

CONFIG = ROOT/'schemas/safestep_exact_supervision_20261015.json'
OUT = ROOT/'outputs/safestep_exact_supervision_20261015'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


def prepare():
    cfg = read(CONFIG)
    sc = read(ROOT/cfg['source_config'])
    source = ROOT/cfg['source_round']
    source_plan = read(source/'frozen_manifest.json')
    audit = ROOT/cfg['source_audit']
    previous = read(audit/'completion_manifest.json')
    protected = {**read(audit/'frozen_manifest.json')['protected_previous_files'],
        **previous['artifact_sha256'], str((audit/'completion_manifest.json').relative_to(ROOT)): sha(audit/'completion_manifest.json')}
    for path, digest in protected.items():
        assert sha(ROOT/path) == digest, path
    OUT.mkdir(exist_ok=True)
    grid = SmallMap.paper_case_map()
    legal = legal_matrix(grid)
    settings = read(ROOT/sc['source_config'])['paper_settings']
    dt = settings['cell_size_m']/settings['walking_speed_m_per_s']
    code = ['schemas/safestep_exact_supervision_20261015.json',
        'src/fire_evac/safestep_exact_supervision.py', 'scripts/run_safestep_exact_supervision.py',
        'scripts/verify_safestep_exact_supervision.py', 'tests/test_safestep_exact_supervision.py']
    profiles = {}
    def labels(observations):
        result = np.empty((len(observations), 4), dtype=np.float32)
        # Current FER snapshots are stationary within each post-onset profile.
        for rates in np.unique(observations[:, 24:48], axis=0):
            key = rates.tobytes()
            if key not in profiles:
                profiles[key] = snapshot_teacher(grid, rates, sc, dt)
            mask = np.all(observations[:, 24:48] == rates, axis=1)
            selected = observations[mask]
            states = selected[:, :24].argmax(axis=1)
            k = np.rint(selected[:, 48]*sc['max_steps']).astype(int)
            result[mask] = np.nan_to_num(profiles[key][k, states], nan=0.).astype(np.float32)
        return result
    dataset = OUT/'training_dataset.npz'
    provenance = []
    if not dataset.exists():
        pools, origins, indices = [], [], []
        for seed in cfg['training']['seeds']:
            with np.load(source/'runs'/f'hazard_full_seed{seed}/training_transitions.npz') as archive:
                onset = np.array([source_plan['banks']['training'][i]['onset'] for i in archive['scenario_index']])
                keep = archive['k'] >= onset
                pools.append(archive['obs'][keep].copy())
                origins.append(np.full(int(keep.sum()), seed, dtype=np.int16))
                indices.append(np.flatnonzero(keep).astype(np.int32))
                provenance.append({'seed': seed, 'source_transitions': len(keep), 'post_reveal_kept': int(keep.sum())})
        observations = np.concatenate(pools)
        states = observations[:, :24].argmax(axis=1)
        targets = labels(observations)
        np.savez_compressed(dataset, obs=observations, targets=targets, legal=legal[states],
            source_seed=np.concatenate(origins), source_index=np.concatenate(indices))
        dump(OUT/'training_provenance.json', provenance)
    else:
        provenance = read(OUT/'training_provenance.json')
    validation_profiles = []
    val = cfg['validation']
    old_parameters = {(s['peak'], s['center']) for bank in source_plan['banks'].values() for s in bank}
    for side in val['centers']:
        for peak, center in itertools.product(val['peaks'], val['centers'][side]):
            assert (peak, center) not in old_parameters
            validation_profiles.append({'profile_id': f'new_profile_{len(validation_profiles):02d}',
                                        'side': side, 'peak': peak, 'center': center, 'onset': 0})
    state_dataset = OUT/'validation_state_grid.npz'
    if not state_dataset.exists():
        observations, metadata = [], []
        for pi, profile in enumerate(validation_profiles):
            x = np.array([p[0] for p in grid.coordinates])
            rates = sc['background_rate_per_second']+profile['peak']*np.exp(-.5*((x-profile['center'])/sc['spread_cells'])**2)
            rates[list(grid.exits)] = 0
            for k, fed, state in itertools.product(val['state_grid_k'], val['state_grid_FED'], range(24)):
                if state in grid.exits:
                    continue
                observations.append(encode_observation(state, rates, k, fed, sc['max_steps'], sc['observation']['FER_divisor']))
                metadata.append((pi, state, k, fed))
        observations = np.asarray(observations)
        targets = labels(observations)
        np.savez_compressed(state_dataset, obs=observations, targets=targets,
            legal=legal[observations[:, :24].argmax(axis=1)], metadata=np.asarray(metadata))
    cases = [{'case_id': f"{p['profile_id']}__{start}", **p, 'start': start}
             for p in validation_profiles for start in val['starts']]
    schedule = [{'case_id': c['case_id'], 'method': method} for c in cases for method in cfg['evaluation_methods']]
    random.Random(2026101501).shuffle(schedule)
    plan = {'code_sha256': {p: sha(ROOT/p) for p in code}, 'protected_previous_files': protected,
        'dataset_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [dataset, OUT/'training_provenance.json', state_dataset]},
        'training_provenance': provenance, 'validation_profiles': validation_profiles, 'control_cases': cases,
        'control_schedule': schedule, 'planned_control_episodes': len(schedule),
        'planned_optimizer_updates': cfg['training']['updates_per_model']*3,
        'RL_updates': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False}
    assert len(schedule) == cfg['planned_control_episodes']
    target = OUT/'frozen_manifest.json'
    if target.exists():
        old = read(target)
        assert {k: v for k, v in old.items() if k != 'frozen_at_utc'} == plan
        return cfg, sc, old
    plan['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(target, plan)
    print(json.dumps({'status': 'exact_supervision_round_frozen', 'training_states': sum(p['post_reveal_kept'] for p in provenance),
        'validation_profiles': len(validation_profiles), 'control_cases': len(cases), 'control_episodes': len(schedule)}), flush=True)
    return cfg, sc, plan


def train_one(cfg, sc, seed):
    folder = OUT/'models'/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training_summary.json').exists():
        summary = read(folder/'training_summary.json')
        assert summary['model_sha256'] == sha(folder/'policy.pt')
        return summary
    if (folder/'started.json').exists():
        raise RuntimeError('Preserve incomplete training; no silent restart.')
    dump(folder/'started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+991000)
    model = build_network(cfg['training']['hidden_layers'])
    initial = fingerprint(model)
    old = read(ROOT/cfg['source_round']/'runs'/f'hazard_full_seed{seed}/training_summary.json')
    assert initial == old['initial_parameter_sha256']
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    with np.load(OUT/'training_dataset.npz') as data:
        observations = torch.from_numpy(data['obs'])
        labels = torch.from_numpy(data['targets'])
        legal = torch.from_numpy(data['legal'])
    log, loss_sum, nloss = [], 0., 0
    started = time.monotonic()
    for update in range(1, cfg['training']['updates_per_model']+1):
        indices = rng.integers(len(observations), size=cfg['training']['batch_size'])
        prediction = model(observations[indices])
        loss = legal_mse(prediction, labels[indices], legal[indices])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite supervised loss; preserve run, do not retune.')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())
        nloss += 1
        if update % 2000 == 0 or update == cfg['training']['updates_per_model']:
            row = {'update': update, 'mean_batch_legal_MSE': loss_sum/nloss, 'wall_seconds': time.monotonic()-started}
            log.append(row)
            print(json.dumps({'seed': seed, **row}), flush=True)
            loss_sum = 0.
            nloss = 0
    torch.save(model.state_dict(), folder/'policy.pt')
    table(folder/'training_curve.csv', log)
    result = {'seed': seed, 'updates': cfg['training']['updates_per_model'],
        'initial_parameter_sha256': initial, 'final_parameter_sha256': fingerprint(model),
        'model_sha256': sha(folder/'policy.pt'), 'training_dataset_sha256': sha(OUT/'training_dataset.npz'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'scored_evaluation_performed': False, 'RL_updates': 0}
    dump(folder/'training_summary.json', result)
    return result


def lock_models(cfg):
    models = []
    for seed in cfg['training']['seeds']:
        path = OUT/'models'/f'seed{seed}/training_summary.json'
        row = read(path)
        assert row['model_sha256'] == sha(path.parent/'policy.pt')
        assert row['updates'] == cfg['training']['updates_per_model']
        models.append({'seed': seed, 'model_sha256': row['model_sha256'],
                       'completed_at_utc': row['completed_at_utc'], 'summary_sha256': sha(path)})
    target = OUT/'models_locked.json'
    plan = {'models': models, 'manifest_sha256': sha(OUT/'frozen_manifest.json')}
    if target.exists():
        old = read(target)
        assert {k: v for k, v in old.items() if k != 'locked_at_utc'} == plan
        return old
    plan['locked_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(target, plan)
    return plan


def predict(model, observations):
    with torch.inference_mode():
        return np.vstack([model(torch.from_numpy(observations[i:i+2048])).numpy() for i in range(0, len(observations), 2048)])


def evaluate(cfg, sc, plan):
    locked = lock_models(cfg)
    if (OUT/'summary.json').exists():
        raise RuntimeError('Evaluation exists; preserve it and use verifier.')
    began = datetime.now(timezone.utc).isoformat()
    assert began > locked['locked_at_utc']
    torch.set_num_threads(1)
    models = {}
    for seed in cfg['training']['seeds']:
        for prefix, path in [('legacy_DQN', ROOT/cfg['source_round']/'runs'/f'hazard_full_seed{seed}/policy.pt'),
                             ('supervised_Q', OUT/'models'/f'seed{seed}/policy.pt')]:
            model = build_network(cfg['training']['hidden_layers'])
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            model.eval()
            models[f'{prefix}_seed{seed}'] = model
    metrics, predictions = {}, {}
    for split, filename in [('training', 'training_dataset.npz'), ('new_state_grid', 'validation_state_grid.npz')]:
        with np.load(OUT/filename) as data:
            observations, labels, legal = data['obs'], data['targets'], data['legal']
            metrics[split] = {}
            for method, model in models.items():
                q = predict(model, observations)
                metrics[split][method] = quality(q, labels, legal, cfg['ranking_tolerance'])
                if split == 'new_state_grid':
                    ordinary = observations[:, 48] <= 96/sc['max_steps']
                    metrics[split][method]['ordinary_k_le_96'] = quality(q[ordinary], labels[ordinary], legal[ordinary], cfg['ranking_tolerance'])
                    metrics[split][method]['near_deadline_k_gt_96'] = quality(q[~ordinary], labels[~ordinary], legal[~ordinary], cfg['ranking_tolerance'])
                    predictions[method] = q
    np.savez_compressed(OUT/'validation_Q_predictions.npz', **predictions)
    grid, settings = SmallMap.paper_case_map(), read(ROOT/sc['source_config'])['paper_settings']
    legal = legal_matrix(grid)
    dt = settings['cell_size_m']/settings['walking_speed_m_per_s']
    cases = {c['case_id']: c for c in plan['control_cases']}
    fields = {}
    x = np.array([p[0] for p in grid.coordinates])
    for cid, case in cases.items():
        rates = sc['background_rate_per_second']+case['peak']*np.exp(-.5*((x-case['center'])/sc['spread_cells'])**2)
        rates[list(grid.exits)] = 0
        fields[cid] = np.repeat(rates[None, :], sc['max_steps']+1, axis=0)
    results, traces = [], []
    for job in plan['control_schedule']:
        cid, method = job['case_id'], job['method']
        case, field = cases[cid], fields[cid]
        env = SafeStepEpisode(grid, ControlTickField(field, sc['tau'], dt), settings, sc['tau'], sc['max_steps'], grid.labels.index(case['start']))
        observations = []
        while not env.done:
            common = encode_observation(env.state, field[env.steps], env.steps, env.fed,
                                        sc['max_steps'], sc['observation']['FER_divisor'])
            if method == 'snapshot_planner':
                decoded = common[24:48].astype(float)*sc['observation']['FER_divisor']
                action = snapshot_path(grid, env.state, decoded, dt, sc['time_cost_in_FED_per_step'])[0]
            else:
                with torch.inference_mode():
                    q = models[method](torch.from_numpy(common)).numpy()
                action = max(np.flatnonzero(legal[env.state]).tolist(), key=lambda a: (q[a], -a))
            observations.append(common.tolist())
            env.step(action)
        success = env.state in grid.exits
        member, witnesses = closed_walk_members(env.trace, 0)
        outcome = {'exit_reached': success, 'steps': env.steps, 'time_seconds': env.steps*dt,
            'personal_FED': env.fed, 'J': env.fed+sc['time_cost_in_FED_per_step']*env.steps if success else None,
            'revisits': sum(row['revisit'] for row in env.trace), 'closed_walk_steps': int(member.sum()),
            'invalid_actions': sum(row['invalid'] for row in env.trace)}
        results.append({**case, 'method': method, **outcome})
        traces.append({**case, 'method': method, 'outcome': outcome, 'observations': observations,
                       'closed_walk_witnesses': witnesses, 'trace': env.trace})
    table(OUT/'control_results.csv', results)
    with (OUT/'control_traces.jsonl').open('w', encoding='utf-8') as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    control = {}
    for method in cfg['evaluation_methods']:
        rows = [r for r in results if r['method'] == method]
        successes = [r for r in rows if r['exit_reached']]
        control[method] = {'episodes': len(rows), 'successes': len(successes),
            'mean_FED_success_only': float(np.mean([r['personal_FED'] for r in successes])) if successes else None,
            'mean_time_success_only': float(np.mean([r['time_seconds'] for r in successes])) if successes else None,
            'mean_J_success_only': float(np.mean([r['J'] for r in successes])) if successes else None,
            'closed_walk_steps': sum(r['closed_walk_steps'] for r in rows), 'revisits': sum(r['revisits'] for r in rows)}
    names = [f'supervised_Q_seed{seed}' for seed in cfg['training']['seeds']]
    fit = all(metrics['training'][name]['legal_Q_MAE'] <= .15 and metrics['training'][name]['optimal_action_accuracy'] >= .95 and
              metrics['training'][name]['positive_selected_Q_fraction'] <= .01 for name in names)
    generalizes = all(metrics['new_state_grid'][name]['optimal_action_accuracy'] >= .95 and
                      metrics['new_state_grid'][name]['positive_selected_Q_fraction'] <= .01 for name in names)
    usable = all(control[name]['successes'] == 24 and control[name]['closed_walk_steps'] == 0 and
                 control[name]['mean_J_success_only'] <= 1.05*control['snapshot_planner']['mean_J_success_only'] for name in names)
    summary = {'status': 'completed_fixed_budget_exact_Q_supervision_diagnostic', 'state_metrics': metrics,
        'control': control, 'frozen_decisions': {'training_fit': fit, 'new_state_generalization': generalizes, 'control_usable': usable},
        'evaluation_started_at_utc': began, 'models_locked_sha256': sha(OUT/'models_locked.json'),
        'manifest_sha256': sha(OUT/'frozen_manifest.json'), 'control_results_sha256': sha(OUT/'control_results.csv'),
        'control_traces_sha256': sha(OUT/'control_traces.jsonl'), 'Q_predictions_sha256': sha(OUT/'validation_Q_predictions.npz'),
        'supervised_models': 3, 'supervised_optimizer_updates': cfg['training']['updates_per_model']*3,
        'new_RL_updates': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False,
        'decision': 'Close this round after verification and reporting, irrespective of threshold outcomes. No adaptive extension.'}
    dump(OUT/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    cfg, sc, plan = prepare()
    if args.mode == 'train':
        for seed in cfg['training']['seeds']:
            result = train_one(cfg, sc, seed)
            print(json.dumps({'status': 'supervised_model_completed', **result}), flush=True)
        lock_models(cfg)
    elif args.mode == 'evaluate':
        evaluate(cfg, sc, plan)
