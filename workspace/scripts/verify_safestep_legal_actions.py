"""Independent geometric masking, graph, transition and dose verification."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_legal_action_audit_20261014'
CONFIG = ROOT / 'schemas/safestep_legal_action_audit_20261014.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle: return list(csv.DictReader(handle))


def close(a, b): assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11), (a, b)


def main():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    source = json.loads((ROOT / cfg['source_config']).read_text(encoding='utf-8'))
    old_plan = json.loads((ROOT / cfg['source_manifest']).read_text(encoding='utf-8'))
    assert plan['config_sha256'] == sha(CONFIG)
    assert plan['runner_sha256'] == sha(ROOT / 'scripts/audit_safestep_legal_actions.py')
    assert plan['execution_module_sha256'] == sha(ROOT / 'src/fire_evac/safestep_execution_checks.py')
    for path, digest in plan['protected_previous_files'].items(): assert sha(ROOT / path) == digest, path
    labels = old_plan['labels']
    coords = [tuple(c) for c in old_plan['coordinates']]
    assert coords == [(i, 0) for i in range(20)] + [(0, 1), (19, 1), (17, 1), (13, -1)]
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    exits = {20, 21}
    settings, impl = source['paper_settings'], source['explicit_implementation_assumptions']
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    hazard = read_csv(ROOT / cfg['source_hazard'])
    times = np.array([float(r['time_s']) for r in hazard])
    rates = np.array([[float(r[label]) for label in labels] for r in hazard])
    q_records = read_csv(OUT / 'saved_model_Q_values.csv')
    assert len(q_records) == 144
    maps = {}
    torch.set_num_threads(1)
    for case in settings['case_start_times_s']:
        for seed in impl['seeds']:
            layers, width = [], 24
            for size in impl['network_hidden_layers']:
                layers.extend([nn.Linear(width, size), nn.ReLU()])
                width = size
            model = nn.Sequential(*layers, nn.Linear(width, 4))
            model.load_state_dict(torch.load(ROOT / 'outputs/safestep_reproduction_20261014/runs' /
                f'{case}_seed{seed}/policy.pt', map_location='cpu', weights_only=True))
            model.eval()
            with torch.inference_mode(): q = model(torch.eye(24)).numpy()
            maps[(case, seed)] = {v: [] for v in cfg['variants']}
            for state, (x, y) in enumerate(coords):
                legal = [a for a, (dx, dy) in enumerate(deltas) if (x+dx, y+dy) in coords]
                original = int(np.argmax(q[state]))
                # Explicit score scan gives an independent first-index tie rule.
                best = legal[0]
                for a in legal[1:]:
                    if q[state, a] > q[state, best]: best = a
                maps[(case, seed)]['original_greedy'].append(original)
                maps[(case, seed)]['geometric_legal_greedy'].append(best)
                row = next(r for r in q_records if r['case'] == case and int(r['seed']) == seed and r['cell'] == labels[state])
                for a, name in enumerate(['left', 'right', 'up', 'down']): close(row['Q_' + name], q[state, a])
                assert row['original_action'] == ['left', 'right', 'up', 'down'][original]
                assert row['masked_action'] == ['left', 'right', 'up', 'down'][best]
                assert row['decision_changed'] == str(original != best)
    traces = [json.loads(line) for line in (OUT / 'actual_step_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(traces) == len({(t['task_key'], t['variant']) for t in traces}) == 264
    results = read_csv(OUT / 'all_execution_results.csv')
    assert len(results) == 264
    steps_checked = 0
    trace_by_key = {}
    for item in traces:
        state = labels.index(item['start_cell'])
        tau = settings['case_start_times_s'][item['case']]
        dose = total = discounted = 0.0
        invalids = revisits = 0
        visited = {state}
        action_map = maps[(item['case'], item['seed'])][item['variant']]
        for k, step in enumerate(item['trace']):
            assert step['from_state'] == state and step['action'] == action_map[state]
            x, y = coords[state]
            dx, dy = deltas[step['action']]
            destination = (x+dx, y+dy)
            new = coords.index(destination) if destination in coords else state
            invalid = new == state
            revisit = new != state and new in visited
            delta_dose = float(np.interp(tau + k*dt, times, rates[:, state])) * dt
            dose += delta_dose
            kind = 'impossible' if invalid else 'exit' if new in exits else 'visited' if revisit else 'regular'
            u = 0 if dose == 0 else -100 if dose < .3 else -300 if dose < 1 else -800
            reward = .1*settings['path_rewards'][kind] + .2*u*delta_dose
            assert step['to_state'] == new and step['path_reward_kind'] == kind
            close(step['absolute_time_s'], tau+k*dt)
            close(step['dose_increment'], delta_dose)
            close(step['cumulative_personal_FED'], dose)
            close(step['reward'], reward)
            assert step['invalid'] == invalid and step['revisit'] == revisit
            if item['variant'] == 'geometric_legal_greedy': assert not invalid
            state = new
            visited.add(new)
            invalids += invalid
            revisits += revisit
            total += reward
            discounted += settings['discount']**k * reward
            assert state not in exits or k == len(item['trace']) - 1
            steps_checked += 1
        outcome = item['outcome']
        assert outcome['exit_reached'] == (state in exits)
        for name, value in {'steps': len(item['trace']), 'time_seconds': len(item['trace'])*dt,
            'personal_FED': dose, 'invalid_actions': invalids, 'revisits': revisits,
            'undiscounted_reward': total, 'discounted_reward': discounted}.items(): close(outcome[name], value)
        # Reconstruct graph independently, without importing the graph helper.
        graph_seen, graph_states = {}, []
        s = labels.index(item['start_cell'])
        while s not in exits and s not in graph_seen:
            graph_seen[s] = len(graph_states)
            graph_states.append(s)
            x, y = coords[s]
            dx, dy = deltas[action_map[s]]
            p = (x+dx, y+dy)
            s = coords.index(p) if p in coords else s
        cycle = graph_states[graph_seen[s]:] if s not in exits else []
        assert item['graph_witness']['cycle_states'] == cycle
        assert item['graph_witness']['graph_reaches_exit'] == (s in exits)
        record = next(r for r in results if r['task_key'] == item['task_key'] and r['variant'] == item['variant'])
        assert record['exit_reached'] == str(outcome['exit_reached'])
        for name in ['steps', 'time_seconds', 'personal_FED', 'invalid_actions', 'revisits']: close(record[name], outcome[name])
        trace_by_key[(item['task_key'], item['variant'])] = item
    pairs = read_csv(OUT / 'paired_execution_effects.csv')
    assert len(pairs) == 132
    rescued = lost = changed_success = 0
    for pair in pairs:
        raw = trace_by_key[(pair['task_key'], 'original_greedy')]
        masked = trace_by_key[(pair['task_key'], 'geometric_legal_greedy')]
        a, b = raw['outcome'], masked['outcome']
        rescue = not a['exit_reached'] and b['exit_reached']
        loss = a['exit_reached'] and not b['exit_reached']
        change = [r['action'] for r in raw['trace']] != [r['action'] for r in masked['trace']]
        assert pair['rescued'] == str(rescue) and pair['success_lost'] == str(loss)
        assert pair['action_chain_changed'] == str(change)
        close(pair['FED_delta_masked_minus_original'], b['personal_FED'] - a['personal_FED'])
        close(pair['time_delta_seconds'], b['time_seconds'] - a['time_seconds'])
        rescued += rescue
        lost += loss
        changed_success += a['exit_reached'] and change
        if a['exit_reached']: assert raw['trace'] == masked['trace']
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    assert summary['manifest_sha256'] == sha(OUT / 'frozen_manifest.json')
    assert summary['paired_effects_sha256'] == sha(OUT / 'paired_execution_effects.csv')
    for variant in cfg['variants']:
        subset = [t['outcome'] for t in traces if t['variant'] == variant]
        stats = summary['variants'][variant]
        assert stats['exit_reached'] == sum(r['exit_reached'] for r in subset)
        assert stats['invalid_moves'] == sum(r['invalid_actions'] for r in subset)
        close(stats['mean_FED_all_rollouts'], sum(r['personal_FED'] for r in subset) / len(subset))
        close(stats['mean_steps_all_rollouts'], sum(r['steps'] for r in subset) / len(subset))
    assert summary['paired']['rescued'] == rescued == 8
    assert summary['paired']['successes_lost'] == lost == 0
    assert summary['paired']['original_success_action_chains_changed'] == changed_success == 0
    visits = json.loads((OUT / 'training_visitation_availability.json').read_text(encoding='utf-8'))
    assert len(visits) == 6 and all(not r['per_cell_visitation_available'] for r in visits)
    for entry in visits:
        log = read_csv(ROOT / 'outputs/safestep_reproduction_20261014/runs' /
                       f"{entry['case']}_seed{entry['seed']}/training_trace.csv")
        assert entry['recorded_fields'] == list(log[0])
        assert entry['last_200_training_exit_count'] == sum(r['exit_reached'] == 'True' for r in log[-200:])
    for path, digest in plan['protected_previous_files'].items(): assert sha(ROOT / path) == digest
    verified = {'status': 'completed_fixed_weight_mask_audit_verified', 'paired_cases': 132,
        'episodes_recomputed': 264, 'steps_recomputed': steps_checked,
        'saved_model_Q_values_verified': 576, 'all_transition_dose_reward_and_cycle_checks_passed': True,
        'eight_failures_rescued': True, '124_original_successful_traces_unchanged': True,
        'original_artifacts_and_weights_unchanged': True, 'training_visits_missing_not_imputed': True,
        'new_training': 0, 'LLM_requests': 0, 'PPO_updates': 0,
        'summary_sha256': sha(OUT / 'summary.json'), 'manifest_sha256': sha(OUT / 'frozen_manifest.json'),
        'verifier_sha256': sha(__file__)}
    (OUT / 'verification.json').write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__': main()
