"""Posthoc description of already saved routes; no policy calls or new rollouts."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_shared_observation_20261014'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [json.loads(line) for line in (OUT / 'evaluation_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    confirmation = [r for r in rows if r['split'] == 'confirmation']
    behavior = {}
    for method in sorted({r['method'] for r in confirmation}):
        subset = [r for r in confirmation if r['method'] == method]
        cycle_steps = []
        for row in subset:
            # The synthetic field stays constant after onset. Any legal closed
            # subwalk wholly after onset has positive J cost, with no hazard relief.
            seen, repeated = set(), 0
            for k, step in enumerate(row['trace']):
                if k >= row['onset']:
                    seen.add(step['from_state'])
                    repeated += step['to_state'] in seen
            cycle_steps.append(repeated)
        behavior[method] = {'episodes': len(subset),
            'revisits_total': sum(r['outcome']['revisits'] for r in subset),
            'mean_revisits': statistics.mean(r['outcome']['revisits'] for r in subset),
            'min_steps': min(r['outcome']['steps'] for r in subset),
            'max_steps': max(r['outcome']['steps'] for r in subset),
            'post_reveal_repeated_destinations_total': sum(cycle_steps),
            'exit_counts': dict(collections.Counter(r['outcome']['exit'] for r in subset)),
            'wrong_exit_after_hazard_visible': sum(r['outcome']['exit'] == ('Exit_A' if r['side'] == 'left' else 'Exit_B') for r in subset)}
    failures = []
    for row in confirmation:
        if not row['outcome']['exit_reached']:
            failures.append({k: row[k] for k in ['scenario_id', 'method', 'side', 'onset', 'peak', 'center']} | {
                'outcome': row['outcome'],
                'most_visited_departure_cells': collections.Counter(s['from_cell'] for s in row['trace']).most_common(5),
                'first_12_moves': [[s['from_cell'], s['to_cell']] for s in row['trace'][:12]],
                'last_6_moves': [[s['from_cell'], s['to_cell']] for s in row['trace'][-6:]]})
    value = {'status': 'completed_posthoc_cached_behavior_description', 'behavior': behavior,
        'failed_confirmation_episodes': failures, 'new_training': 0, 'new_rollouts': 0,
        'caution': 'Observed repeated movements and scenario cluster describe failure execution; no optimizer cause is identified, no repair is tested.',
        'traces_sha256': sha(OUT / 'evaluation_traces.jsonl'), 'analysis_script_sha256': sha(Path(__file__))}
    payload = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    target = OUT / 'cached_behavior_diagnosis.json'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps({'failed_confirmation_episodes': len(failures),
        'full_models_mean_revisits': {k: v['mean_revisits'] for k, v in behavior.items() if k.startswith('hazard_full')},
        'new_rollouts': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
