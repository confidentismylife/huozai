"""Freeze then execute one bounded synthetic dynamic-observation probe."""
import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from run_safestep_reproduction import network
from fire_evac.safestep_baseline import SmallMap, SafeStepEpisode
from fire_evac.safestep_execution_checks import select_action
from fire_evac.safestep_dynamic_probe import ControlTickField, scenario_rates, snapshot_path, future_j_plan

CONFIG = ROOT / 'schemas/safestep_dynamic_probe_20261014.json'
OUT = ROOT / 'outputs/safestep_dynamic_probe_20261014'


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


def prepare():
    cfg = read(CONFIG)
    source = read(ROOT / cfg['source_config'])
    previous = read(ROOT / cfg['source_review'])
    protected = dict(previous['source_hashes'])
    for relative in [cfg['source_review'], 'scripts/review_safestep_cached_evidence.py',
                     'reports/SafeStep复现再审查与结论边界_2026-10-14.md']:
        protected[relative] = sha(ROOT / relative)
    for relative, digest in protected.items():
        assert sha(ROOT / relative) == digest, relative
    code_files = ['schemas/safestep_dynamic_probe_20261014.json',
        'src/fire_evac/safestep_dynamic_probe.py', 'scripts/run_safestep_dynamic_probe.py',
        'scripts/verify_safestep_dynamic_probe.py', 'tests/test_safestep_dynamic_probe.py']
    grid = SmallMap.paper_case_map()
    dt = source['paper_settings']['cell_size_m'] / source['paper_settings']['walking_speed_m_per_s']
    OUT.mkdir(exist_ok=True)
    (OUT / 'fields').mkdir(exist_ok=True)
    scenarios, fields = [], {}
    for case, reveal, side, severity in itertools.product(cfg['case_blocks'], cfg['reveal_steps'],
                                                         cfg['danger_centers'], cfg['peak_rates_per_second']):
        sid = f'{case}__{reveal}__{side}__{severity}'
        scenario = {'scenario_id': sid, 'case': case, 'tau': cfg['case_blocks'][case],
                    'reveal': reveal, 'danger_side': side, 'severity': severity}
        scenarios.append(scenario)
        rates = scenario_rates(grid, scenario, cfg)
        path = OUT / 'fields' / f'{sid}.csv'
        if not path.exists():
            table(path, [{'k': k, 'time_s': scenario['tau'] + k*dt,
                **dict(zip(grid.labels, [float(x) for x in row]))} for k, row in enumerate(rates)])
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert np.array_equal(rates, np.array([[float(row[label]) for label in grid.labels] for row in rows]))
        fields[str(path.relative_to(ROOT))] = sha(path)
    schedule = [{'scenario_id': s['scenario_id'], 'method': method} for s in scenarios for method in cfg['methods']]
    random.Random(cfg['order_seed']).shuffle(schedule)
    assert len(scenarios) == cfg['planned_scenarios'] and len(schedule) == cfg['planned_rollouts']
    plan = {'config_sha256': sha(CONFIG), 'code_sha256': {p: sha(ROOT / p) for p in code_files},
        'protected_previous_files': protected, 'field_sha256': fields,
        'scenarios': scenarios, 'schedule': schedule, 'new_training': 0, 'LLM_requests': 0,
        'old_final_scenarios_accessed': False}
    path = OUT / 'frozen_manifest.json'
    if path.exists():
        prior = read(path)
        assert {k: v for k, v in prior.items() if k != 'frozen_at_utc'} == plan
        return cfg, source, prior
    plan['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(path, plan)
    print(json.dumps({'status': 'dynamic_probe_frozen', 'scenarios': len(scenarios),
        'relative_hazard_patterns': 8, 'planned_rollouts': len(schedule), 'training': 0}), flush=True)
    return cfg, source, plan


def run(cfg, source, plan):
    if (OUT / 'results.csv').exists():
        raise RuntimeError('Results already exist. Preserve them and use the verifier; no rerun/retuning.')
    grid = SmallMap.paper_case_map()
    settings = source['paper_settings']
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    penalty = cfg['time_cost_in_FED_per_step']
    start = grid.labels.index(cfg['start'])
    torch.set_num_threads(1)
    maps = {}
    for case in cfg['case_blocks']:
        for seed in cfg['seeds']:
            model = network(24, source['explicit_implementation_assumptions']['network_hidden_layers'])
            path = ROOT / 'outputs/safestep_reproduction_20261014/runs' / f'{case}_seed{seed}/policy.pt'
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()
            with torch.inference_mode():
                q = model(torch.eye(24)).numpy()
            maps[(case, seed)] = [select_action(q[state], [a for a, _ in grid.valid_neighbors(state)])
                                   for state in range(24)]
    scenario_by_id = {s['scenario_id']: s for s in plan['scenarios']}
    fields, oracles = {}, {}
    for sid, scenario in scenario_by_id.items():
        fields[sid] = scenario_rates(grid, scenario, cfg)
        oracles[sid] = future_j_plan(grid, fields[sid], dt, penalty, start)
    results, traces = [], []
    for job in plan['schedule']:
        sid, method = job['scenario_id'], job['method']
        scenario, rates = scenario_by_id[sid], fields[sid]
        env = SafeStepEpisode(grid, ControlTickField(rates, scenario['tau'], dt),
                              settings, scenario['tau'], cfg['max_steps'], start)
        calls, path = 0, None
        if method in ['fixed_left', 'fixed_right', 'nearest_exit']:
            target = 20 if method == 'fixed_left' else 21 if method == 'fixed_right' else None
            path = snapshot_path(grid, start, np.zeros(24), dt, penalty, target)
            calls = 1
        elif method == 'snapshot_once':
            path = snapshot_path(grid, start, rates[0].copy(), dt, penalty)
            calls = 1
        elif method == 'future_J_oracle':
            path = oracles[sid][0]
            calls = 1
        observations = []
        while not env.done:
            k = env.steps
            current = rates[k].copy()
            if method == 'snapshot_replan':
                action = snapshot_path(grid, env.state, current, dt, penalty)[0]
                calls += 1
            elif method.startswith('legacy_DQN_seed'):
                seed = int(method.removeprefix('legacy_DQN_seed'))
                action = maps[(scenario['case'], seed)][env.state]
            else:
                action = path[k]
            assert action in [a for a, _ in grid.valid_neighbors(env.state)]
            observations.append({'k': k, 'state': env.state, 'current_FER_per_second': current.tolist()})
            env.step(action)
        success = env.state in grid.exits
        outcome = {'exit_reached': success, 'steps': env.steps, 'time_seconds': env.steps*dt,
            'personal_FED': env.fed, 'J': env.fed + penalty*env.steps if success else None,
            'exit': grid.labels[env.state] if success else None,
            'paper_discounted_reward': sum(settings['discount']**k*row['reward'] for k, row in enumerate(env.trace)),
            'invalid_actions': sum(row['invalid'] for row in env.trace),
            'revisits': sum(row['revisit'] for row in env.trace),
            'FED_above_0_3': env.fed >= .3, 'planning_calls': calls,
            'future_oracle_J': oracles[sid][1],
            'J_regret_vs_future_oracle': env.fed + penalty*env.steps - oracles[sid][1] if success else None}
        if success:
            assert outcome['J_regret_vs_future_oracle'] >= -1e-10
        horizontal = [(k, row['action']) for k, row in enumerate(env.trace) if row['action'] in [0, 1]]
        outcome['horizontal_reversal_after_reveal'] = any(a != b and k >= cfg['reveal_steps'][scenario['reveal']]
            for (_, a), (k, b) in zip(horizontal, horizontal[1:]))
        row = {**scenario, 'method': method, **outcome}
        results.append(row)
        traces.append({**scenario, 'method': method, 'outcome': outcome,
                       'observations': observations, 'trace': env.trace})
    table(OUT / 'results.csv', results)
    with (OUT / 'traces.jsonl').open('w', encoding='utf-8') as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    lookup = {(r['scenario_id'], r['method']): r for r in results}
    aggregate = {}
    for method in cfg['methods']:
        rows = [r for r in results if r['method'] == method]
        successes = [r for r in rows if r['exit_reached']]
        aggregate[method] = {'successes': len(successes), 'episodes': len(rows),
            'mean_FED_success_only': float(np.mean([r['personal_FED'] for r in successes])) if successes else None,
            'mean_time_success_only': float(np.mean([r['time_seconds'] for r in successes])) if successes else None,
            'mean_J_success_only': float(np.mean([r['J'] for r in successes])) if successes else None,
            'mean_J_regret_success_only': float(np.mean([r['J_regret_vs_future_oracle'] for r in successes])) if successes else None,
            'invalid_actions': sum(r['invalid_actions'] for r in rows),
            'revisits': sum(r['revisits'] for r in rows)}
    delayed = []
    for scenario in plan['scenarios']:
        if scenario['reveal'] == 'delayed' and scenario['danger_side'] == 'right':
            sid = scenario['scenario_id']
            a, b = lookup[(sid, 'snapshot_once')], lookup[(sid, 'snapshot_replan')]
            delayed.append({'scenario_id': sid, 'once_J': a['J'], 'replan_J': b['J'],
                'J_improvement': a['J'] - b['J'] if a['J'] is not None and b['J'] is not None else None})
    switches = []
    for case, severity in itertools.product(cfg['case_blocks'], cfg['peak_rates_per_second']):
        exits = [lookup[(f'{case}__immediate__{side}__{severity}', 'future_J_oracle')]['exit']
                 for side in cfg['danger_centers']]
        switches.append({'case': case, 'severity': severity, 'oracle_exits': exits, 'changes_exit': len(set(exits)) == 2})
    no_regression = all(not lookup[(s['scenario_id'], 'snapshot_once')]['exit_reached'] or
        lookup[(s['scenario_id'], 'snapshot_replan')]['exit_reached'] for s in plan['scenarios'])
    summary = {'status': 'completed_one_frozen_synthetic_probe', 'scenarios': len(plan['scenarios']),
        'unique_relative_hazard_patterns': 8, 'episodes': len(results),
        'aggregate_descriptive_only': aggregate,
        'delayed_right_paired_J_effects': delayed, 'immediate_oracle_exit_switches': switches,
        'frozen_decisions': {'immediate_exit_discrimination': all(r['changes_exit'] for r in switches),
            'replanner_success_all_16': aggregate['snapshot_replan']['successes'] == 16,
            'no_snapshot_success_lost': no_regression,
            'strict_J_improvement_on_all_four_delayed_right': all(r['J_improvement'] is not None and r['J_improvement'] > 1e-10 for r in delayed)},
        'manifest_sha256': sha(OUT / 'frozen_manifest.json'), 'results_sha256': sha(OUT / 'results.csv'),
        'traces_sha256': sha(OUT / 'traces.jsonl'), 'new_training': 0, 'LLM_requests': 0,
        'old_final_scenarios_accessed': False,
        'decision': 'Close this diagnostic round after verification/reporting. Do not expand training from these results.',
        'caveats': ['Synthetic mechanism probe, no original FDS numeric reproduction.',
            'Legacy DQN receives less current hazard information; no algorithm superiority claim.',
            'Oracle has privileged future information.',
            'Two absolute-time blocks repeat eight relative patterns; no independent-fire inference.',
            'J is an explicitly new engineering objective; paper reward is separately recorded.']}
    dump(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    cfg, source, plan = prepare()
    if args.mode == 'run':
        run(cfg, source, plan)
