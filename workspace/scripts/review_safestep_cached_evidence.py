"""Read-only review of frozen SafeStep evidence; no rollout, training or API use."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/safestep_reproduction_20261014'
MASK = ROOT / 'outputs/safestep_legal_action_audit_20261014'
OUT = ROOT / 'outputs/safestep_critical_review_20261014'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = {}
    for folder in (BASE, MASK):
        manifest = folder / 'completion_manifest.json'
        for relative, digest in read(manifest)['artifact_sha256'].items():
            assert sha(ROOT / relative) == digest, relative
            protected[relative] = digest
        protected[str(manifest.relative_to(ROOT))] = sha(manifest)
    traces = [json.loads(line) for line in (MASK / 'actual_step_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    masked = [row for row in traces if row['variant'] == 'geometric_legal_greedy']
    raw = [row for row in traces if row['variant'] == 'original_greedy']
    assert len(masked) == len(raw) == 132
    success = [row for row in traces if row['outcome']['exit_reached']]
    tiers = sorted({step['tox_utility'] for row in success for step in row['trace']})
    assert tiers == [-100]
    assert all(row['outcome']['invalid_actions'] == row['outcome']['revisits'] == 0 for row in success)
    # In the realized successful paths, the undiscounted reward reduces to this
    # linear expression. This does NOT establish equivalence of trained policies.
    for row in success:
        result = row['outcome']
        linear = 1 - .025 * (result['steps'] - 1) - 20 * result['personal_FED']
        assert abs(result['undiscounted_reward'] - linear) < 1e-10
    main_paths = []
    for path in sorted((BASE / 'runs').glob('*/RoomB_evaluation_trace.json')):
        main_paths.append([step['action'] for step in read(path)])
    fixed_left = [2] + [0] * 13 + [2]
    assert len(main_paths) == 6 and all(path == fixed_left for path in main_paths)
    room_a = [row for row in masked if row['case'] == 'case2_late' and row['start_cell'] == 'Room_A']
    a = next(row['outcome'] for row in room_a if row['seed'] == 42)
    b = next(row['outcome'] for row in room_a if row['seed'] == 43)
    assert a['personal_FED'] > b['personal_FED'] and a['time_seconds'] > b['time_seconds']
    assert a['discounted_reward'] < b['discounted_reward']
    training = []
    for path in sorted((BASE / 'runs').glob('*/training_trace.csv')):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        training.append({'run': path.parent.name, 'episodes': len(rows),
            'end_FED_at_least_0_3': sum(float(row['personal_FED']) >= .3 for row in rows),
            'end_FED_at_least_1': sum(float(row['personal_FED']) >= 1 for row in rows)})
    result = {
        'scope': 'Posthoc cached-evidence review; not an independent performance evaluation.',
        'protected_file_count': len(protected),
        'all_protected_files_match': True,
        'raw_successes': sum(row['outcome']['exit_reached'] for row in raw),
        'masked_successes': sum(row['outcome']['exit_reached'] for row in masked),
        'successful_trace_count_across_variants': len(success),
        'successful_step_utility_values': tiers,
        'masked_max_final_FED': max(row['outcome']['personal_FED'] for row in masked),
        'successful_undiscounted_reward_formula': '1 - 0.025*(steps-1) - 20*FED',
        'all_six_main_paths_equal_fixed_left_plan': True,
        'fixed_left_actions': fixed_left,
        'fixed_left_caveat': 'Identified after observing these cases; this is a logical trajectory equivalence, not a preregistered competing rule or generalization result.',
        'same_start_dominated_masked_policy_example': {
            'case': 'case2_late', 'start': 'Room_A',
            'seed42': a, 'seed43': b,
            'seed42_extra_time_seconds': a['time_seconds'] - b['time_seconds'],
            'seed42_relative_extra_FED': a['personal_FED'] / b['personal_FED'] - 1,
            'seed42_has_lower_discounted_return': True,
            'interpretation': 'Reaching every exit is not optimality. This comparison does not prove seed43 globally optimal.'},
        'training_terminal_FED_counts': training,
        'threshold_caveat': 'Higher utility tiers can affect training. Their absence from successful evaluation traces does not prove they were irrelevant to learned weights.',
        'new_rollouts': 0, 'new_training': 0, 'LLM_requests': 0,
        'source_hashes': protected,
        'review_script_sha256': sha(Path(__file__)),
    }
    OUT.mkdir(exist_ok=True)
    target = OUT / 'cached_evidence_review.json'
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == payload, 'Preserve prior review; use a new revision.'
    else:
        target.write_text(payload, encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
