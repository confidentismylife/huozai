"""Re-evaluate old models under one frozen execution intervention; no training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from run_safestep_reproduction import network
from fire_evac.safestep_baseline import SmallMap, HazardField, SafeStepEpisode, evaluate_actions
from fire_evac.safestep_execution_checks import select_action, policy_graph_witness

CONFIG = ROOT / 'schemas/safestep_legal_action_audit_20261014.json'
OUT = ROOT / 'outputs/safestep_legal_action_audit_20261014'
ACTION_NAMES = ['left', 'right', 'up', 'down']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    source = json.loads((ROOT / cfg['source_config']).read_text(encoding='utf-8'))
    old = json.loads((ROOT / cfg['source_completion']).read_text(encoding='utf-8'))
    protected = dict(old['artifact_sha256'])
    protected[cfg['source_completion']] = sha(ROOT / cfg['source_completion'])
    for path, digest in protected.items(): assert sha(ROOT / path) == digest, path
    old_rows = read_csv(ROOT / cfg['source_coverage'])
    assert len(old_rows) == len({(r['case'], r['seed'], r['start_cell']) for r in old_rows}) == 132
    tasks = [{'case': r['case'], 'seed': int(r['seed']), 'start_cell': r['start_cell']} for r in old_rows]
    plan = {'config_sha256': sha(CONFIG), 'runner_sha256': sha(__file__),
        'execution_module_sha256': sha(ROOT / 'src/fire_evac/safestep_execution_checks.py'),
        'protected_previous_files': protected, 'tasks': tasks, 'expected_episodes': 264,
        'new_training': 0, 'LLM_requests': 0, 'PPO_updates': 0}
    OUT.mkdir(exist_ok=True)
    path = OUT / 'frozen_manifest.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert {k: v for k, v in saved.items() if k != 'frozen_at_utc'} == plan
        return cfg, source, saved
    plan['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(path, plan)
    print(json.dumps({'status': 'legal_action_audit_frozen', 'pairs': 132, 'episodes': 264, 'new_training': 0}), flush=True)
    return cfg, source, plan


def outcome_stats(rows):
    return {'episodes': len(rows), 'exit_reached': sum(r['exit_reached'] for r in rows),
        'exit_rate': statistics.mean(r['exit_reached'] for r in rows),
        'timeouts': sum(not r['exit_reached'] for r in rows),
        'invalid_moves': sum(r['invalid_actions'] for r in rows),
        'mean_FED_all_rollouts': statistics.mean(r['personal_FED'] for r in rows),
        'mean_steps_all_rollouts': statistics.mean(r['steps'] for r in rows),
        'warning': 'Timeouts included; dose/steps averages are descriptive, not evacuation efficacy.'}


def run(cfg, source, plan):
    torch.set_num_threads(1)
    grid = SmallMap.paper_case_map()
    hazard = HazardField.from_csv(ROOT / cfg['source_hazard'], grid)
    settings, impl = source['paper_settings'], source['explicit_implementation_assumptions']
    previous = {(r['case'], int(r['seed']), r['start_cell']): r for r in read_csv(ROOT / cfg['source_coverage'])}
    maps, q_records, visits = {}, [], []
    for case, tau in settings['case_start_times_s'].items():
        for seed in impl['seeds']:
            directory = ROOT / 'outputs/safestep_reproduction_20261014/runs' / f'{case}_seed{seed}'
            model = network(24, impl['network_hidden_layers'])
            model.load_state_dict(torch.load(directory / 'policy.pt', map_location='cpu', weights_only=True))
            model.eval()
            with torch.inference_mode(): q = model(torch.eye(24)).numpy()
            maps[(case, seed)] = {}
            for variant in cfg['variants']:
                maps[(case, seed)][variant] = [select_action(q[s],
                    [a for a, _ in grid.valid_neighbors(s)] if variant == 'geometric_legal_greedy' else None)
                    for s in range(24)]
            for s in range(24):
                legal = [a for a, _ in grid.valid_neighbors(s)]
                a = maps[(case, seed)]['original_greedy'][s]
                b = maps[(case, seed)]['geometric_legal_greedy'][s]
                q_records.append({'case': case, 'seed': seed, 'cell': grid.labels[s],
                    **{f'Q_{name}': float(q[s, i]) for i, name in enumerate(ACTION_NAMES)},
                    'legal_actions': json.dumps([ACTION_NAMES[i] for i in legal]),
                    'original_action': ACTION_NAMES[a], 'masked_action': ACTION_NAMES[b],
                    'original_action_legal': a in legal, 'decision_changed': a != b,
                    'raw_max_minus_best_legal_Q': float(q[s, a] - q[s, b])})
            logs = read_csv(directory / 'training_trace.csv')
            assert len(logs) == 2000
            fields = list(logs[0])
            visits.append({'case': case, 'seed': seed, 'training_episodes': len(logs),
                'known_training_reset_cell': 'Room_B', 'recorded_fields': fields,
                'per_cell_visitation_available': False, 'reason': 'No per-transition states/actions or replay buffer saved',
                'last_200_training_exit_count': sum(r['exit_reached'] == 'True' for r in logs[-200:])})
    table('saved_model_Q_values.csv', q_records)
    dump(OUT / 'training_visitation_availability.json', visits)
    results, traces, pairs, cycles = [], [], [], []
    for task in plan['tasks']:
        case, seed, cell = task['case'], task['seed'], task['start_cell']
        start, tau = grid.labels.index(cell), settings['case_start_times_s'][case]
        task_key = f'{case}__{seed}__{cell}'
        local = {}
        for variant in cfg['variants']:
            action_map = maps[(case, seed)][variant]
            env = SafeStepEpisode(grid, hazard, settings, tau, impl['max_episode_steps'], start)
            actions = []
            while not env.done:
                action = action_map[env.state]
                actions.append(action)
                env.step(action)
            outcome, realized = evaluate_actions(grid, hazard, settings, tau, impl['max_episode_steps'], actions, start)
            witness = policy_graph_witness(grid, action_map, start)
            assert witness['graph_reaches_exit'] == outcome['exit_reached']
            row = {**task, 'task_key': task_key, 'variant': variant, **outcome,
                   'cycle_type': witness['cycle_type'], 'cycle_labels': json.dumps(witness['cycle_labels'])}
            results.append(row)
            traces.append({**task, 'task_key': task_key, 'variant': variant, 'outcome': outcome,
                           'graph_witness': witness, 'trace': realized})
            cycles.append({**task, 'variant': variant, **witness})
            local[variant] = (row, actions)
        raw, raw_actions = local['original_greedy']
        masked, masked_actions = local['geometric_legal_greedy']
        prior = previous[(case, seed, cell)]
        for name in ['steps', 'time_seconds', 'personal_FED', 'invalid_actions', 'revisits']:
            assert math.isclose(float(raw[name]), float(prior[name]), rel_tol=0, abs_tol=1e-12)
        assert raw['exit_reached'] == (prior['exit_reached'] == 'True')
        pairs.append({**task, 'task_key': task_key,
            'original_success': raw['exit_reached'], 'masked_success': masked['exit_reached'],
            'rescued': not raw['exit_reached'] and masked['exit_reached'],
            'success_lost': raw['exit_reached'] and not masked['exit_reached'],
            'action_chain_changed': raw_actions != masked_actions,
            'original_steps': raw['steps'], 'masked_steps': masked['steps'],
            'original_FED': raw['personal_FED'], 'masked_FED': masked['personal_FED'],
            'FED_delta_masked_minus_original': masked['personal_FED'] - raw['personal_FED'],
            'time_delta_seconds': masked['time_seconds'] - raw['time_seconds'],
            'original_cycle': raw['cycle_type'], 'masked_cycle': masked['cycle_type']})
    table('all_execution_results.csv', results)
    table('paired_execution_effects.csv', pairs)
    with (OUT / 'actual_step_traces.jsonl').open('w', encoding='utf-8') as f:
        for trace in traces: f.write(json.dumps(trace, ensure_ascii=False) + '\n')
    dump(OUT / 'functional_graph_witnesses.json', cycles)
    mutual = [p for p in pairs if p['original_success'] and p['masked_success']]
    recovered = [p for p in pairs if p['rescued']]
    failures = [p for p in pairs if not p['original_success']]
    table('eight_original_failure_comparisons.csv', failures)
    summary = {'status': 'completed_fixed_weight_execution_audit', 'pairs': 132, 'episodes': 264,
        'variants': {v: outcome_stats([r for r in results if r['variant'] == v]) for v in cfg['variants']},
        'paired': {'old_failures': len(failures), 'rescued': len(recovered),
            'successes_lost': sum(p['success_lost'] for p in pairs),
            'original_success_action_chains_changed': sum(p['original_success'] and p['action_chain_changed'] for p in pairs),
            'mutual_successes': len(mutual),
            'mutual_success_FED_mean_delta': statistics.mean(p['FED_delta_masked_minus_original'] for p in mutual),
            'mutual_success_time_mean_delta_seconds': statistics.mean(p['time_delta_seconds'] for p in mutual),
            'recovered_cases': recovered,
            'remaining_masked_timeouts': [p for p in pairs if not p['masked_success']]},
        'by_case': {case: {v: outcome_stats([r for r in results if r['case'] == case and r['variant'] == v])
                    for v in cfg['variants']} for case in settings['case_start_times_s']},
        'training_visitation_available': False,
        'new_DQN_training': 0, 'PPO_updates': 0, 'LLM_requests': 0, 'old_final_scenarios_accessed': False,
        'decision': 'Mask-only round closed. Preserve remaining cycles; no second intervention or training added.',
        'scope': cfg['scope'], 'visibility': cfg['visibility'],
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'),
        'paired_effects_sha256': sha(OUT / 'paired_execution_effects.csv')}
    for path, digest in plan['protected_previous_files'].items(): assert sha(ROOT / path) == digest, path
    dump(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    cfg, source, plan = prepare()
    if args.mode == 'run': run(cfg, source, plan)
