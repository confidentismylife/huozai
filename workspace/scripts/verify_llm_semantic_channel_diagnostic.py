"""Independent reconstruction of pairing, advice features, traces and effects."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/llm_semantic_channel_diagnostic_20261005'
PRIOR = ROOT / 'outputs/evacuation_auto_llm_independent_20261005'
ARMS_DIAG = ('north_c50', 'north_c95', 'south_c50', 'south_c95', 'none', 'repaired_native')
ARMS_FRESH = ('none', 'rule_native', 'original_native', 'repaired_native')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def advice_value(arm, family, record):
    if arm == 'none':
        return None
    if arm.startswith(('north_c', 'south_c')):
        return {'target_branch': int(arm.startswith('south')), 'confidence': .5 if arm.endswith('50') else .95}
    return record[{'repaired_native': 'repaired', 'original_native': 'original', 'rule_native': 'rule'}[arm]][family]


def trace_check(z):
    risks, keys, offsets, trace = z['risks'], z['trace_keys'], z['trace_offsets'], z['execution_trace']
    assert trace.ndim == 2 and trace.shape[1] == 40
    assert len(keys) == 12 and len(offsets) == 13 and offsets[0] == 0 and offsets[-1] == len(trace)
    assert len({tuple(k) for k in keys}) == 12
    sequences = {}
    for i, key in enumerate(keys):
        key = tuple(map(int, key))
        t = trace[offsets[i]:offsets[i + 1]]
        r = risks[key]
        assert len(t) == r[5] and np.isfinite(t).all()
        np.testing.assert_array_equal(t[:, 0], np.arange(len(t)))
        assert np.isin(t[:, 1], np.arange(9)).all()
        assert np.isin(t[:, 2], np.arange(-1, 9)).all()
        np.testing.assert_allclose(t[:, 16:21].sum(axis=1), t[:, 21:26].sum(axis=1) + t[:, 37] + t[:, 38], rtol=0, atol=1e-9)
        assert (t[:, 37:40] >= -1e-9).all()
        np.testing.assert_allclose(t[:, 21:26].sum(axis=1) + np.cumsum(t[:, 37] + t[:, 38]), 240, rtol=0, atol=1e-9)
        np.testing.assert_allclose([t[:, 37].sum() / 240, t[:, 38].sum() / 240, t[-1, 21:26].sum() / 240], r[[0, 2, 3]], rtol=0, atol=1e-9)
        np.testing.assert_allclose([t[:, 4].mean(), (t[:, 1] == 8).sum(), t[:, 5].mean(), t[0, 1], t[0, 2]], r[6:11], rtol=0, atol=1e-9)
        np.testing.assert_allclose(t[:, 39].sum() / 240, r[1], rtol=0, atol=1e-8)
        assert (t[t[:, 1] == 8, 36] == 0).all()
        np.testing.assert_allclose(t[1:, 16:21], t[:-1, 21:26], rtol=0, atol=0)
        sequences[key] = t[:, 1].copy()
    return sequences, len(trace)


def verify():
    manifest = read(OUT / 'manifest.json')
    budget = read(OUT / 'budget.json')
    for name, digest in manifest['frozen_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in manifest['parent_completion'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / 'outputs/third_round_development_status.json') == manifest['global_status_sha256']
    raw_caches = list((OUT / 'llm_cache').glob('*.json'))
    for path in raw_caches:
        cached = read(path)
        raw = read(ROOT / cached['raw_path'])
        value = json.loads(raw['response']['message']['content'])
        field = 'target' if raw['version'] == 'original' else 'recommended_exit'
        assert cached['advice'] == {'target_branch': ('north', 'south').index(value[field]), 'confidence': value['confidence']}
        assert raw['model_digest'] == cached['model_digest'] == manifest['model_digest']
        assert raw['prompt'] == cached['prompt']
        data = json.loads(raw['prompt'].split('Data: ')[1])
        assert set(data) == {'growth_rate', 'onset_step', 'background_hazard', 'north_exit_capacity', 'south_exit_capacity',
                            'blocked_branch', 'blockage_end_step', 'forecast_source' if raw['version'] == 'original' else 'predicted_fire_location'}
    assert budget['llm_requests'] == len(list((OUT / 'llm_attempts').glob('*.started.json'))) <= 252
    semantic = read(OUT / 'semantic_summary.json')
    for version in ('original', 'repaired'):
        values = {s['scenario_id']: read(OUT / 'llm_cache' / f"probe_{version}_{s['scenario_id']}.json")['advice']['target_branch'] for s in manifest['probes']}
        correct = sum(values[s['scenario_id']] != s['forecast_source_branch'] for s in manifest['probes'])
        symmetry = sum(values[a['scenario_id']] != values[b['scenario_id']] for a, b in zip(manifest['probes'][::2], manifest['probes'][1::2]))
        assert semantic['by_prompt'][version]['reference_correct'] == correct
        assert semantic['by_prompt'][version]['mirror_pairs_correct'] == symmetry
    expected_semantic_pass = semantic['by_prompt']['repaired']['reference_correct'] == 12 and semantic['by_prompt']['repaired']['mirror_pairs_correct'] == 6
    assert semantic['repaired_passed'] == expected_semantic_pass

    counts, steps, factor_sequences, family_seed_losses = {}, 0, {}, {}
    non_advice = np.r_[np.arange(28), np.arange(33, 49)]
    diagnostic = read(OUT / 'diagnostic_summary.json')
    for block in ('diagnostic', 'fresh'):
        if block == 'fresh' and not diagnostic['proceed_to_fresh']:
            assert not (OUT / 'fresh_advice_frozen.json').exists() and not (OUT / 'fresh_actual').exists()
            continue
        scenes = manifest['development' if block == 'diagnostic' else 'fresh']
        registry = manifest['development_registry' if block == 'diagnostic' else 'fresh_registry']
        advice_path = OUT / f'{block}_advice_frozen.json'
        advice = read(advice_path)
        advice_sha = sha(advice_path)
        arms = ARMS_DIAG if block == 'diagnostic' else ARMS_FRESH
        count = 0
        for scene in scenes:
            family = scene['scenario_id']
            base = None
            for seed in range(601, 606):
                for arm in arms:
                    path = OUT / f'{block}_actual' / f'{seed}_{arm}_{family}.npz'
                    z = load(path)
                    r, obs = z['risks'], z['observations']
                    assert r.shape == (2, 1, 1, 6, 11) and z['done'].all() and np.isfinite(r).all()
                    np.testing.assert_allclose(r[..., 0] + r[..., 2] + r[..., 3], 1, rtol=0, atol=1e-10)
                    assert ((r[..., [0, 2, 3]] >= 0) & (r[..., [0, 2, 3]] <= 1)).all()
                    np.testing.assert_array_equal(z['population_seeds'], registry[family]['population'])
                    np.testing.assert_array_equal(z['future_seeds'], registry[family]['actual'])
                    if base is None:
                        base = obs[:, non_advice].copy()
                    np.testing.assert_array_equal(obs[:, non_advice], base)
                    value = advice_value(arm, family, advice)
                    expected = np.zeros((2, 5)) if value is None else np.array([[value['target_branch'] == 0,
                                value['target_branch'] == 1, value['confidence'], 1,
                                x[8 + value['target_branch']] > x[8 + 1 - value['target_branch']] + .01] for x in obs])
                    np.testing.assert_allclose(obs[:, 28:33], expected, rtol=0, atol=1e-7)
                    marker = read(path.with_suffix('.started.json'))
                    assert marker['selection_sha256'] == advice_sha
                    assert advice['created_at_utc'] < marker['started_at_utc']
                    assert marker['scene'] == scene
                    if block == 'diagnostic':
                        sequence, nsteps = trace_check(z)
                        factor_sequences[(family, seed, arm)] = sequence
                        steps += nsteps
                        reference = load(PRIOR / 'llm_actual' / f'{seed}_llm_{family}.npz')
                        np.testing.assert_array_equal(reference['observations'][:, non_advice], base)
                    family_seed_losses[(block, family, seed, arm)] = float(r[..., 0].mean() * 100)
                    count += 12
        files = list((OUT / f'{block}_actual').glob('*.npz'))
        assert len(files) == len(scenes) * 5 * len(arms) and not any(p.name.endswith('.partial.npz') for p in files)
        counts[block] = count
    direction = confidence = 0
    for scene in manifest['development']:
        family = scene['scenario_id']
        for seed in range(601, 606):
            for name, a, b in (('direction', 'north_c50', 'south_c50'), ('direction', 'north_c95', 'south_c95'),
                               ('confidence', 'north_c50', 'north_c95'), ('confidence', 'south_c50', 'south_c95')):
                aa, bb = factor_sequences[(family, seed, a)], factor_sequences[(family, seed, b)]
                changed = sum(not np.array_equal(aa[key], bb[key]) for key in aa)
                if name == 'direction':
                    direction += changed
                else:
                    confidence += changed
    assert diagnostic['direction_changed_requested_sequences'] == direction
    assert diagnostic['confidence_changed_requested_sequences'] == confidence
    assert diagnostic['channel_behaviour_changed'] == (direction + confidence > 0)
    assert diagnostic['proceed_to_fresh'] == bool(expected_semantic_pass and direction + confidence > 0)
    assert counts['diagnostic'] == 2160
    parity = read(OUT / 'cached_outcome_parity.json')
    assert parity['all_11_outcomes_match_cached_original'] and parity['rollouts'] == 2
    for row in parity['rows']:
        reference = load(PRIOR / 'llm_actual' / f"601_llm_{row['family']}.npz")['risks'][row['population'], 0, 0, 0]
        np.testing.assert_allclose(row['outcome'], reference, rtol=0, atol=1e-10)
    primary_recomputed = False
    if diagnostic['proceed_to_fresh']:
        assert counts['fresh'] == 11520
        summary = read(OUT / 'fresh_summary.json')
        families, kinds, per_seed = [], [], {str(seed): [] for seed in range(601, 606)}
        for scene in manifest['fresh']:
            family = scene['scenario_id']
            effects = []
            for seed in range(601, 606):
                delta = family_seed_losses[('fresh', family, seed, 'repaired_native')] - family_seed_losses[('fresh', family, seed, 'rule_native')]
                effects.append(delta)
                per_seed[str(seed)].append(delta)
            families.append(np.mean(effects))
            kinds.append(scene['conflict_type'])
        rng = np.random.default_rng(2026100571)
        boot = np.zeros(2000)
        values, kinds = np.asarray(families), np.asarray(kinds)
        for kind in sorted(set(kinds)):
            v = values[kinds == kind]
            boot += v[rng.integers(len(v), size=(2000, len(v)))].sum(axis=1)
        interval = np.quantile(boot / 48, [.025, .975])
        np.testing.assert_allclose(summary['primary_mean_delta_pp'], np.mean(families), rtol=0, atol=1e-9)
        np.testing.assert_allclose(summary['primary_ci95_pp'], interval, rtol=0, atol=1e-9)
        for seed, value in per_seed.items():
            np.testing.assert_allclose(summary['per_training_seed_delta_pp'][seed], np.mean(value), rtol=0, atol=1e-9)
        assert summary['improvement_signal'] == bool(interval[1] < 0 and sum(np.mean(v) < 0 for v in per_seed.values()) >= 4)
        primary_recomputed = True
    total = sum(counts.values()) + 2
    assert budget['simulations_started'] == budget['simulations_completed'] == total <= 13682
    assert budget['matrix_fits'] == budget['ppo_updates'] == 0
    return {'all_checks_passed': True, 'rollouts_checked': total, 'counts': counts,
            'instrumentation_rollouts': 2, 'full_trace_steps_checked': steps,
            'initial_non_advice_input_equality_checked': True, 'seed_blocks_and_advice_freezing_checked': True,
            'raw_responses_reparsed': len(raw_caches), 'semantic_result_reconstructed': True,
            'factorial_action_sequence_responses_reconstructed': True,
            'fresh_primary_effect_and_ci_independently_reconstructed': primary_recomputed,
            'old_global_status_unchanged': True, 'frozen_sources_unchanged': True,
            'new_simulations_for_verification': 0, 'new_fits': 0, 'new_ppo_updates': 0,
            'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(verify(), ensure_ascii=False))
