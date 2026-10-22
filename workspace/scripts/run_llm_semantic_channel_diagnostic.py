"""Checkpointed semantics and frozen-controller input study. No PPO training."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse
import copy
import json
import os
import sys
import time
import traceback

import numpy as np
import auto_evacuation_campaign as c
from llm_semantic_channel_core import (prompt, schema, parse, probes, fixed_advice, feature_advice,
                                      FACTORIAL_ARMS, DIAGNOSTIC_ARMS, FRESH_ARMS, stratified_ci)

ROOT = c.ROOT
PRIOR = ROOT / 'outputs/evacuation_auto_llm_independent_20261005'
OUT = ROOT / 'outputs/llm_semantic_channel_diagnostic_20261005'
RUNTIME = ROOT / 'outputs/llm_semantic_channel_runtime_20261005'
PROTOCOL = ROOT / 'reports/LLM建议语义与控制通道自动诊断固定方案_2026-10-05.md'
REPORT = ROOT / 'reports/LLM建议语义与控制通道实验报告_2026-10-05.md'
MODEL_DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
CAP = 13682
REQUEST_CAP = 252


def atomic(path, data):
    """Operational retry of the same bytes; no scientific or budget change."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def write(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def progress(stage, **detail):
    status = c.read(OUT / 'campaign_status.json')
    status.update(stage=stage, detail=detail, updated_at_utc=c.stamp(), pid=os.getpid())
    write(OUT / 'campaign_status.json', status)
    print(json.dumps({'stage': stage, **detail}, ensure_ascii=False), flush=True)


def historical_seed_values(value, inside=False):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found |= historical_seed_values(child, inside or 'seed' in key.lower())
    elif isinstance(value, list):
        for child in value:
            found |= historical_seed_values(child, inside)
    elif inside and isinstance(value, int) and not isinstance(value, bool):
        found.add(value)
    return found


def prepare():
    assert not OUT.exists()
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, LLM, MANAGER, HISTORY
    parent_checks = {'main': ledger_check(MAIN / 'completion_manifest_v2.json', HISTORY),
                     'llm': ledger_check(LLM / 'completion_manifest.json'),
                     'manager': ledger_check(MANAGER / 'completion_manifest.json'),
                     'review': ledger_check(ROOT / 'outputs/evacuation_auto_final_review_20261005/completion_manifest.json')}
    old = c.read(PRIOR / 'manifest.json')
    kinds = sorted({s['conflict_type'] for s in old['banks']['llm']})
    development = [min((s for s in old['banks']['llm'] if s['conflict_type'] == kind), key=lambda s: s['scenario_id']) for kind in kinds]
    from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank
    from run_proposal_gain_factorial import signature, scenario_records
    records, oldseeds = [], set()
    jsons = set((ROOT / 'outputs').rglob('*manifest*.json')) | set((ROOT / 'outputs').rglob('*scenarios.json'))
    for path in sorted(jsons):
        value = c.read(path)
        records += scenario_records(value)
        oldseeds |= historical_seed_values(value)
    # Read ancestor seed fields lazily, never large outcome or trajectory arrays.
    fresh_parent = c.read(ROOT / 'outputs/five_ppo_fresh_family_student_development_v1/development_manifest.json')
    seed_npz = [Path(p) for p in fresh_parent['source_input_sha256'] if Path(p).suffix == '.npz']
    for path in seed_npz:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if 'seed' in key:
                    oldseeds.update(map(int, np.asarray(z[key]).reshape(-1)))
    fresh = [asdict(s) for s in make_domain_randomized_trust_bank(48, 2026100561, 'SCF')]
    assert len({signature(s) for s in fresh}) == 48
    assert not {signature(s) for s in fresh} & {signature(s) for s in records}
    assert len(set(Counter(s['conflict_type'] for s in fresh).values())) == 1
    registry = {s['scenario_id']: {'population': [1870000000 + i * 1000 + p for p in range(2)],
                                 'actual': [[1970000000 + i * 10000 + p * 1000 + f for f in range(6)] for p in range(2)]}
                for i, s in enumerate(fresh)}
    newseeds = {int(x) for r in registry.values() for v in r.values() for x in np.asarray(v).reshape(-1)}
    assert not (newseeds | {s + 9000001 for s in newseeds}) & (oldseeds | {s + 9000001 for s in oldseeds})
    rng = np.random.default_rng(2026100562)
    probe_records = probes()
    probe_order = [(s['scenario_id'], version) for s in probe_records for version in ('original', 'repaired')]
    rng.shuffle(probe_order)
    diag_order = [(s['scenario_id'], seed, arm) for s in development for seed in range(601, 606) for arm in DIAGNOSTIC_ARMS]
    rng.shuffle(diag_order)
    fresh_order = [(s['scenario_id'], seed, arm) for s in fresh for seed in range(601, 606) for arm in FRESH_ARMS]
    rng.shuffle(fresh_order)
    fresh_request_order = [(s['scenario_id'], version) for s in fresh for version in ('original', 'repaired')]
    rng.shuffle(fresh_request_order)
    models = {str(seed): f'outputs/guarded_confirmatory_v2/models/seed_{seed}_raw_llm_ppo.zip' for seed in range(601, 606)}
    paths = {Path(__file__), ROOT / 'scripts/llm_semantic_channel_core.py', ROOT / 'scripts/verify_llm_semantic_channel_diagnostic.py',
             ROOT / 'tests/test_llm_semantic_channel_core.py', PROTOCOL,
             ROOT / 'scripts/auto_evacuation_campaign.py', ROOT / 'scripts/auto_evacuation_core.py'}
    paths |= {ROOT / p for p in models.values()}
    # Everything used to derive physics, behaviour and the input wrapper is frozen.
    paths |= set((ROOT / 'src/fire_evac').glob('*.py'))
    paths |= {ROOT / 'outputs/behavior_pipeline/behavior_model.json', ROOT / 'outputs/guarded_confirmatory_v2/training_summary.csv'}
    OUT.mkdir()
    write(OUT / 'manifest.json', {'created_at_utc': c.stamp(), 'role': 'frozen_semantic_and_advice_channel_development_diagnostic',
        'parent_checks': parent_checks, 'parent_completion': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in
            [MAIN / 'completion_manifest_v2.json', LLM / 'completion_manifest.json', MANAGER / 'completion_manifest.json',
             ROOT / 'outputs/evacuation_auto_final_review_20261005/completion_manifest.json']},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(paths)},
        'probes': probe_records, 'probe_order': probe_order, 'development': development,
        'development_registry': {s['scenario_id']: old['registry'][s['scenario_id']] for s in development},
        'fresh': fresh, 'fresh_registry': registry, 'diagnostic_order': diag_order, 'fresh_order': fresh_order,
        'fresh_request_order': fresh_request_order, 'models': models, 'model_digest': MODEL_DIGEST,
        'inference_options': {'temperature': 0, 'seed': 2026100551, 'num_predict': 96}, 'think': False,
        'request_cap': REQUEST_CAP, 'simulation_cap': CAP, 'new_ppo_updates': 0, 'new_fits': 0,
        'original_prompt_reproduction_required': True, 'fresh_signatures_disjoint': True,
        'historical_scenario_records_checked': len(records), 'historical_seed_values_checked': len(oldseeds),
        'new_seed_streams_checked_including_sensor_offsets': True, 'ancestor_seed_npz_checked': len(seed_npz),
        'thresholds_changed': False, 'formal_final_bank': False,
        'global_status_sha256': c.sha(c.STATUS), 'global_status_will_remain_unchanged': True,
        'operational_atomic_replace_retry_seconds': .05, 'max_atomic_replace_attempts': 20})
    write(OUT / 'budget.json', {'simulations_started': 0, 'simulations_completed': 0, 'by_kind': {},
                              'llm_requests': 0, 'matrix_fits': 0, 'mean_reference_estimates': 0, 'ppo_updates': 0})
    write(OUT / 'campaign_status.json', {'state': 'ready', 'stage': 'prepared', 'pid': None,
                                       'run_attempts': 0, 'consecutive_failures': 0, 'created_at_utc': c.stamp()})
    print('Prepared and frozen; no new inference or simulation performed.', flush=True)


def frozen_check(m):
    for name, digest in m['frozen_sha256'].items():
        assert c.sha(ROOT / name) == digest, name
    for name, digest in m['parent_completion'].items():
        assert c.sha(ROOT / name) == digest, name
    assert c.sha(c.STATUS) == m['global_status_sha256']


def source_audit(m):
    path = OUT / 'source_audit.json'
    if path.exists():
        return c.read(path)
    from fire_evac.dynamic_llm_rl import NORTH, SOUTH
    from fire_evac.confirmatory import decode_absolute_action, encode_absolute_action
    assert (NORTH, SOUTH) == (0, 1)
    for action in range(8):
        branch, release = decode_absolute_action(action)
        assert branch == action // 4 and encode_absolute_action(branch, release) == action
    assert decode_absolute_action(8) == (None, 0)
    old = c.read(PRIOR / 'manifest.json')
    advice = c.read(PRIOR / 'llm_advice_frozen.json')['advice']
    toward = 0
    for scene in old['banks']['llm']:
        family = scene['scenario_id']
        cached = c.read(PRIOR / 'llm_cache' / f'{family}.json')
        raw = c.read(PRIOR / 'llm_raw_responses' / f"{family}_attempt{cached['attempts']}.json")
        assert prompt(scene, 'original') == raw['prompt'] == cached['prompt']
        assert parse(raw['response']['message']['content'], 'original') == cached['advice'] == advice[family]
        toward += advice[family]['target_branch'] == scene['forecast_source_branch']
        z = c.load(PRIOR / 'llm_actual' / f'601_llm_{family}.npz')
        for obs in z['observations']:
            np.testing.assert_allclose(obs, feature_advice(obs, advice[family]), rtol=0, atol=1e-7)
    historical = {name: c.read(ROOT / 'outputs' / name / 'summary.json') for name in
                  ('confirmatory_v2_prompt_development', 'confirmatory_v2_1_prompt_development', 'confirmatory_v2_2_prompt_development')}
    result = {'original_requests_checked': 48, 'raw_response_parse_encoding_matches': True,
              'original_prompt_exactly_reproduced': True, 'toward_forecast_fire': int(toward),
              'branch_constants_north_south': [NORTH, SOUTH], 'absolute_action_roundtrip_checked': 9,
              'initial_advice_features_checked': 96, 'parser_does_not_flip_direction': True,
              'historical_prompt_audits': historical, 'interpretation': 'No direction inversion found in raw-to-input mapping; intent/reasoning cause remains a hypothesis.'}
    write(path, result)
    return result


def infer(scene, version, group, m):
    request_id = f"{group}_{version}_{scene['scenario_id']}"
    text = prompt(scene, version)
    cache_path = OUT / 'llm_cache' / f'{request_id}.json'
    if cache_path.exists():
        cached = c.read(cache_path)
        assert cached['prompt'] == text and cached['model_digest'] == MODEL_DIGEST
        return cached['advice']
    for attempt in (1, 2):
        marker = OUT / 'llm_attempts' / f'{request_id}_attempt{attempt}.started.json'
        if marker.exists():
            continue
        budget = c.read(OUT / 'budget.json')
        assert budget['llm_requests'] < REQUEST_CAP
        budget['llm_requests'] += 1
        write(OUT / 'budget.json', budget)
        write(marker, {'request_id': request_id, 'attempt': attempt, 'started_at_utc': c.stamp(), 'prompt': text,
                       'version': version, 'model_digest': MODEL_DIGEST})
        start = time.monotonic()
        try:
            response = c.api_json('http://127.0.0.1:11434/api/chat', {'model': 'qwen3:4b', 'messages': [{'role': 'user', 'content': text}],
                'format': schema(version), 'options': m['inference_options'], 'think': False, 'stream': False}, 60)
            raw_path = OUT / 'llm_raw_responses' / f'{request_id}_attempt{attempt}.json'
            write(raw_path, {'response': response, 'prompt': text, 'version': version, 'model_digest': MODEL_DIGEST,
                             'latency_seconds': time.monotonic() - start})
            parsed = parse(response['message']['content'], version)
            write(cache_path, {'request_id': request_id, 'advice': parsed, 'prompt': text, 'model_digest': MODEL_DIGEST,
                               'created_at_utc': c.stamp(), 'successful_attempt': attempt, 'raw_path': raw_path.relative_to(ROOT).as_posix()})
            return parsed
        except Exception as exc:
            write(OUT / 'llm_failures' / f'{request_id}_attempt{attempt}.json', {'request_id': request_id, 'attempt': attempt,
                  'error': repr(exc), 'created_at_utc': c.stamp(), 'wrong_choice_is_not_retry_reason': True})
    raise RuntimeError('Two used attempts without a valid cached response: ' + request_id)


def semantics(m):
    path = OUT / 'semantic_summary.json'
    if path.exists():
        return c.read(path)
    records = {s['scenario_id']: s for s in m['probes']}
    answers = {}
    for i, (family, version) in enumerate(m['probe_order']):
        answers.setdefault(version, {})[family] = infer(records[family], version, 'probe', m)
        progress('semantic_probes', completed=i + 1, total=24)
    summary = {}
    for version, values in answers.items():
        correct = sum(values[s['scenario_id']]['target_branch'] != s['forecast_source_branch'] for s in m['probes'])
        symmetry = sum(values[a['scenario_id']]['target_branch'] != values[b['scenario_id']]['target_branch']
                       for a, b in zip(m['probes'][::2], m['probes'][1::2]))
        summary[version] = {'reference_correct': correct, 'total': 12, 'mirror_pairs_correct': symmetry, 'mirror_pairs': 6}
    result = {'by_prompt': summary, 'repaired_passed': summary['repaired']['reference_correct'] == 12 and summary['repaired']['mirror_pairs_correct'] == 6,
              'frozen_at_utc': c.stamp(), 'simple_comprehension_check_not_complex_route_optimality': True}
    write(path, result)
    return result


def advice_frozen(block, m):
    path = OUT / f'{block}_advice_frozen.json'
    if path.exists():
        return c.read(path)
    from fire_evac.confirmatory import frozen_rule_advice
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    scenes = m['development' if block == 'diagnostic' else 'fresh']
    original, repaired, rule = {}, {}, {}
    if block == 'diagnostic':
        old = c.read(PRIOR / 'llm_advice_frozen.json')['advice']
        for i, s in enumerate(scenes):
            family = s['scenario_id']
            original[family] = old[family]
            repaired[family] = infer(s, 'repaired', 'diagnostic', m)
            progress('diagnostic_advice', completed=i + 1, total=6)
    else:
        by_id = {s['scenario_id']: s for s in scenes}
        for i, (family, version) in enumerate(m['fresh_request_order']):
            (original if version == 'original' else repaired)[family] = infer(by_id[family], version, 'fresh', m)
            progress('fresh_advice', completed=i + 1, total=96)
    for s in scenes:
        value = frozen_rule_advice(DynamicConflictScenario(**s))
        rule[s['scenario_id']] = {'target_branch': value.target_branch, 'confidence': value.confidence}
    result = {'created_at_utc': c.stamp(), 'actual_seen': False, 'original': original, 'repaired': repaired,
              'rule': rule, 'prompt_sha256': c.sha(ROOT / 'scripts/llm_semantic_channel_core.py')}
    write(path, result)
    return result


def actors(m):
    import torch
    from stable_baselines3 import PPO
    from auto_evacuation_core import AdviceActor
    torch.set_num_threads(1)
    result = {}
    for seed, name in m['models'].items():
        model = PPO.load(ROOT / name, device='cpu')
        assert model.action_space.n == 9 and model.observation_space.shape == (34,)
        model.policy.set_training_mode(False)
        result[int(seed)] = AdviceActor(model)
    return result


def parity(m, actorlist):
    path = OUT / 'cached_outcome_parity.json'
    if path.exists():
        return c.read(path)
    from fire_evac.dynamic_llm_rl import DynamicLLMAdvice
    s = m['development'][0]
    family = s['scenario_id']
    old_advice = c.read(PRIOR / 'llm_advice_frozen.json')['advice'][family]
    env = c.make_env([s], mode='llm', advice={family: DynamicLLMAdvice(old_advice['target_branch'], old_advice['confidence'], ('cached_original',))})
    reference = c.load(PRIOR / 'llm_actual' / f'601_llm_{family}.npz')
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    rows = []
    try:
        for p in range(2):
            cached_path = OUT / 'parity' / f'population{p}.json'
            if cached_path.exists():
                row = c.read(cached_path)
            else:
                x, _ = env.reset(seed=m['development_registry'][family]['population'][p], options={'scenario': DynamicConflictScenario(**s)})
                c.reserve('cached_parity')
                out, _ = c.continuation(env, x, actorlist[601], m['development_registry'][family]['actual'][p][0], False)
                row = {'outcome': out.tolist(), 'population': p, 'family': family}
                budget = c.read(OUT / 'budget.json'); budget['simulations_completed'] += 1; write(OUT / 'budget.json', budget)
                write(cached_path, row)
            np.testing.assert_allclose(row['outcome'], reference['risks'][p, 0, 0, 0], atol=1e-10, rtol=0)
            rows.append(row)
    finally:
        env.close()
    result = {'all_11_outcomes_match_cached_original': True, 'rollouts': 2, 'rows': rows}
    write(path, result)
    return result


def arm_advice(arm, family, frozen):
    if arm in FACTORIAL_ARMS:
        return fixed_advice(arm)
    if arm == 'none':
        return None
    return frozen[{'original_native': 'original', 'repaired_native': 'repaired', 'rule_native': 'rule'}[arm]][family]


def input_responses(m, actorlist, frozen):
    path = OUT / 'initial_policy_responses.csv'
    if path.exists():
        return
    import torch
    rows = []
    for s in m['development']:
        family = s['scenario_id']
        obs = c.load(PRIOR / 'llm_actual' / f'601_llm_{family}.npz')['observations'][0]
        for seed, actor in actorlist.items():
            for arm in DIAGNOSTIC_ARMS:
                x = feature_advice(obs, arm_advice(arm, family, frozen))
                with torch.no_grad():
                    probs = actor.model.policy.get_distribution(torch.as_tensor(x[:34]).reshape(1, 34)).distribution.probs.numpy()[0]
                rows.append({'family': family, 'seed': seed, 'arm': arm, 'requested_action': actor.action(x),
                             **{f'p{j}': float(value) for j, value in enumerate(probs)}})
    c.csv_write(path, rows)


def collect(block, m, actorlist, frozen):
    from fire_evac.dynamic_llm_rl import DynamicLLMAdvice
    scenes = m['development' if block == 'diagnostic' else 'fresh']
    by_id = {s['scenario_id']: s for s in scenes}
    registry = m['development_registry' if block == 'diagnostic' else 'fresh_registry']
    order = m['diagnostic_order' if block == 'diagnostic' else 'fresh_order']
    envs = {}
    arms = DIAGNOSTIC_ARMS if block == 'diagnostic' else FRESH_ARMS
    for arm in arms:
        if arm == 'none':
            envs[arm] = c.make_env(scenes, mode='none')
        else:
            values = {s['scenario_id']: arm_advice(arm, s['scenario_id'], frozen) for s in scenes}
            envs[arm] = c.make_env(scenes, mode='llm', advice={k: DynamicLLMAdvice(v['target_branch'], v['confidence'], ('frozen_input',)) for k, v in values.items()})
    try:
        for i, (family, seed, arm) in enumerate(order):
            path = OUT / f'{block}_actual' / f'{seed}_{arm}_{family}.npz'
            if path.exists():
                continue
            s, r = by_id[family], registry[family]
            c.episode_unit(path, envs[arm], s, r['population'], r['actual'], [actorlist[seed]],
                           False, block == 'diagnostic', block, OUT / f'{block}_advice_frozen.json')
            progress(f'{block}_actual', completed=i + 1, total=len(order))
    finally:
        for env in envs.values():
            env.close()


def trajectories(z):
    return {tuple(map(int, key)): z['execution_trace'][z['trace_offsets'][i]:z['trace_offsets'][i + 1]]
            for i, key in enumerate(z['trace_keys'])}


def diagnostic_summary(m, semantics_result):
    path = OUT / 'diagnostic_summary.json'
    if path.exists():
        return c.read(path)
    rows, response_rows = [], []
    direction_changes = confidence_changes = 0
    for s in m['development']:
        family = s['scenario_id']
        for seed in range(601, 606):
            zs = {arm: c.load(OUT / 'diagnostic_actual' / f'{seed}_{arm}_{family}.npz') for arm in DIAGNOSTIC_ARMS}
            for arm, z in zs.items():
                rows.append({'family': family, 'kind': s['conflict_type'], 'seed': seed, 'arm': arm,
                             'loss_pp': float(z['risks'][..., 0].mean() * 100)})
            for arm, oldarm in (('original_native', 'llm'), ('rule_native', 'same_llm_actor_rule')):
                z = c.load(PRIOR / 'llm_actual' / f'{seed}_{oldarm}_{family}.npz')
                rows.append({'family': family, 'kind': s['conflict_type'], 'seed': seed, 'arm': arm,
                             'loss_pp': float(z['risks'][..., 0].mean() * 100)})
            for factor, a, b in (('direction_c50', 'north_c50', 'south_c50'), ('direction_c95', 'north_c95', 'south_c95'),
                                 ('confidence_north', 'north_c50', 'north_c95'), ('confidence_south', 'south_c50', 'south_c95')):
                ta, tb = trajectories(zs[a]), trajectories(zs[b])
                changed = sum(not np.array_equal(ta[k][:, 1], tb[k][:, 1]) for k in ta)
                if factor.startswith('direction'):
                    direction_changes += changed
                else:
                    confidence_changes += changed
                response_rows.append({'family': family, 'seed': seed, 'factor': factor,
                    'paired_episodes': 12, 'requested_sequence_changed': changed,
                    'loss_b_minus_a_pp': float((zs[b]['risks'][..., 0].mean() - zs[a]['risks'][..., 0].mean()) * 100)})
    c.csv_write(OUT / 'diagnostic_family_results.csv', rows)
    c.csv_write(OUT / 'factorial_channel_responses.csv', response_rows)
    primary = []
    for s in m['development']:
        family = s['scenario_id']
        losses = {arm: np.mean([r['loss_pp'] for r in rows if r['family'] == family and r['arm'] == arm])
                  for arm in DIAGNOSTIC_ARMS + ('original_native', 'rule_native')}
        primary.append({'family': family, 'repaired_minus_original_pp': float(losses['repaired_native'] - losses['original_native']),
                        'repaired_minus_rule_pp': float(losses['repaired_native'] - losses['rule_native'])})
    channel = direction_changes + confidence_changes > 0
    result = {'new_rollouts': 2160, 'historical_rollouts_reused': 720,
              'direction_changed_requested_sequences': direction_changes, 'confidence_changed_requested_sequences': confidence_changes,
              'paired_sequences_per_factor': 360, 'channel_behaviour_changed': channel,
              'simple_semantics_passed': semantics_result['repaired_passed'],
              'proceed_to_fresh': bool(semantics_result['repaired_passed'] and channel),
              'native_advice_family_effects': primary, 'created_at_utc': c.stamp(),
              'diagnostic_effects_not_independent_final_evidence': True}
    write(path, result)
    return result


def fresh_summary(m):
    path = OUT / 'fresh_summary.json'
    if path.exists():
        return c.read(path)
    rows, paired = [], []
    for s in m['fresh']:
        family = s['scenario_id']
        for seed in range(601, 606):
            losses = {}
            for arm in FRESH_ARMS:
                z = c.load(OUT / 'fresh_actual' / f'{seed}_{arm}_{family}.npz')
                losses[arm] = float(z['risks'][..., 0].mean() * 100)
                rows.append({'family': family, 'kind': s['conflict_type'], 'seed': seed, 'arm': arm, 'loss_pp': losses[arm]})
            paired.append({'family': family, 'kind': s['conflict_type'], 'seed': seed,
                           'repaired_minus_rule_pp': losses['repaired_native'] - losses['rule_native'],
                           'repaired_minus_original_pp': losses['repaired_native'] - losses['original_native'],
                           'repaired_minus_none_pp': losses['repaired_native'] - losses['none']})
    family_effects = [float(np.mean([r['repaired_minus_rule_pp'] for r in paired if r['family'] == s['scenario_id']])) for s in m['fresh']]
    seed_effects = {str(seed): float(np.mean([r['repaired_minus_rule_pp'] for r in paired if r['seed'] == seed])) for seed in range(601, 606)}
    interval = stratified_ci(family_effects, [s['conflict_type'] for s in m['fresh']])
    result = {'families': 48, 'primary_mean_delta_pp': float(np.mean(family_effects)), 'primary_ci95_pp': interval.tolist(),
              'per_training_seed_delta_pp': seed_effects, 'improvement_signal': bool(interval[1] < 0 and sum(v < 0 for v in seed_effects.values()) >= 4),
              'secondary_mean_repaired_minus_original_pp': float(np.mean([r['repaired_minus_original_pp'] for r in paired])),
              'secondary_mean_repaired_minus_none_pp': float(np.mean([r['repaired_minus_none_pp'] for r in paired])),
              'new_ppo_updates': 0, 'created_at_utc': c.stamp(), 'qualification_passed': False}
    c.csv_write(OUT / 'fresh_family_results.csv', rows)
    c.csv_write(OUT / 'fresh_paired_results.csv', paired)
    write(path, result)
    return result


def report(m, stop):
    semantic = c.read(OUT / 'semantic_summary.json')
    diagnostic = c.read(OUT / 'diagnostic_summary.json')
    budget = c.read(OUT / 'budget.json')
    text = f'''# LLM建议语义与控制通道实验报告

日期：2026-10-05。本轮完成原因：{stop}。所有模型与提示在新增数据前冻结；没有PPO更新或学生拟合。

旧48条输出在原始回答阶段就全部指向预测火源，解析与初始PPO建议列一致。北/南编号和绝对动作映射核对通过；未发现把正确出口在解析时反转的证据。历史三轮语义提示诊断也均未通过。因而本轮检查提示/输出语义与控制通道，而不把负结果简单归因于训练不够。

原提示的简单案例参考方向正确{semantic['by_prompt']['original']['reference_correct']}/12，镜像互换{semantic['by_prompt']['original']['mirror_pairs_correct']}/6；修订提示正确{semantic['by_prompt']['repaired']['reference_correct']}/12，镜像互换{semantic['by_prompt']['repaired']['mirror_pairs_correct']}/6。修订语义通过判定：{semantic['repaired_passed']}。参考只用于等容量、开放、立即起火的简单案例，不代表复杂疏散最优。修订没有提供计算好的出口或强制所有实际场景选相反方向。

6个已看开发家庭、五个历史raw_llm PPO、2人口×6未来。2×2方向/置信度因子、无建议和修订原生建议共新增2160条模拟；旧原生LLM与同控制器规则结局720条只读复用。另2条仪器检查与原缓存11项指标一致。

固定置信度、交换推荐方向，改变请求动作序列{diagnostic['direction_changed_requested_sequences']}/720组配对；固定方向、置信度0.50→0.95，改变{diagnostic['confidence_changed_requested_sequences']}/720组配对。每种固定背景的比较有360匹配回合，不能把重复未来当独立家庭。控制通道行为响应判定：{diagnostic['channel_behaviour_changed']}。

是否进入全新场景：{diagnostic['proceed_to_fresh']}。诊断只检查语义/接口是否可用，不要求已看数据阳性。

'''
    if (OUT / 'fresh_summary.json').exists():
        fresh = c.read(OUT / 'fresh_summary.json')
        text += f'''48个全新场景、五个历史PPO种子，所有臂均同控制器，比较无建议、规则、原提示LLM、修订LLM，共11520条模拟。两种真实LLM建议统一冻结后才采集该分支结局。

主差为修订LLM失能率减规则失能率，负值更好：{fresh['primary_mean_delta_pp']:+.6f}个百分点，按家庭分层bootstrap95%区间[{fresh['primary_ci95_pp'][0]:+.6f}, {fresh['primary_ci95_pp'][1]:+.6f}]。各历史种子：{fresh['per_training_seed_delta_pp']}。预先固定的改善信号判定：{fresh['improvement_signal']}。

次比较修订减原提示为{fresh['secondary_mean_repaired_minus_original_pp']:+.6f}个百分点，修订减无建议为{fresh['secondary_mean_repaired_minus_none_pp']:+.6f}个百分点。次比较不替代主比较，不把其中最有利一项挑为总体成功。

'''
    else:
        text += '前序未通过，因此未调用新场景LLM或采集新场景结局。48个已冻结新场景保留为未使用银行，不把跳过说成通过。\n\n'
    text += f'''本轮启动{budget['simulations_started']}条完整模拟、完成{budget['simulations_completed']}条，真实本地LLM请求{budget['llm_requests']}次，模型拟合0、PPO更新0。正常请求上限126、包含语法/网络重试的上限252；每请求错误选择不触发重试。原始响应和失败证据保留，无规则替代伪装成LLM。

提示/JSON字段组合是修订干预，不能单独归因于某个词。复杂场景允许权衡火势和容量，预测火源方向不等于真实火源。该研究是初始建议与冻结闭环PPO的组合，不证明在线动态信任门控、真实火灾、跨地图/FDS或原门控资格。旧全局状态、模型、报告、完成台账和阈值保持原字节；已看场景只作开发诊断。

[固定方案]({PROTOCOL.as_posix()})；[语义结果]({(OUT/'semantic_summary.json').as_posix()})；[接口审计]({(OUT/'source_audit.json').as_posix()})；[控制通道结果]({(OUT/'diagnostic_summary.json').as_posix()})；[2×2配对原始表]({(OUT/'factorial_channel_responses.csv').as_posix()})；[独立复核]({(OUT/'verification.json').as_posix()})。
'''
    REPORT.write_text(text, encoding='utf-8')


def run():
    m = c.read(OUT / 'manifest.json')
    status = c.read(OUT / 'campaign_status.json')
    if status['state'] == 'complete':
        return
    if c.pid_live(status.get('pid')) and status['pid'] != os.getpid():
        print('Existing study worker active; no duplicate start.', flush=True)
        return
    frozen_check(m)
    c.OUT = OUT
    c.CAP = CAP
    c.atomic_bytes = atomic
    status.update(state='running', pid=os.getpid(), run_attempts=status['run_attempts'] + 1)
    write(OUT / 'campaign_status.json', status)
    try:
        tags = c.api_json('http://127.0.0.1:11434/api/tags', timeout=10)
        assert next(x for x in tags['models'] if x['name'] == 'qwen3:4b')['digest'] == MODEL_DIGEST
        source_audit(m)
        semantic = semantics(m)
        frozen = advice_frozen('diagnostic', m)
        actorlist = actors(m)
        parity(m, actorlist)
        input_responses(m, actorlist, frozen)
        collect('diagnostic', m, actorlist, frozen)
        diagnostic = diagnostic_summary(m, semantic)
        stop = 'diagnostic_failed_fixed_progression'
        if diagnostic['proceed_to_fresh']:
            fresh_advice = advice_frozen('fresh', m)
            collect('fresh', m, actorlist, fresh_advice)
            fresh_summary(m)
            stop = 'all_applicable_stages_complete'
        frozen_check(m)
        from verify_llm_semantic_channel_diagnostic import verify
        result = verify()
        write(OUT / 'verification.json', result)
        report(m, stop)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in
                 ('campaign_status.json', 'events.jsonl', 'completion_manifest.json') and not p.name.endswith('.tmp')]
        paths += [PROTOCOL, REPORT, Path(__file__), ROOT / 'scripts/llm_semantic_channel_core.py',
                  ROOT / 'scripts/verify_llm_semantic_channel_diagnostic.py', ROOT / 'tests/test_llm_semantic_channel_core.py']
        write(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'role': 'semantic_channel_diagnostic_completion',
            'stop_reason': stop, 'budget': c.read(OUT / 'budget.json'), 'new_ppo_updates': 0, 'new_fits': 0,
            'created_at_utc': c.stamp(), 'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(set(paths))}})
        for name, digest in c.read(OUT / 'completion_manifest.json')['files'].items():
            assert c.sha(ROOT / name) == digest, name
        status = c.read(OUT / 'campaign_status.json')
        status.update(state='complete', stage='complete', stop_reason=stop, completed_at_utc=c.stamp(), consecutive_failures=0)
        write(OUT / 'campaign_status.json', status)
        print('Semantic and channel study sealed and complete.', flush=True)
    except Exception:
        status = c.read(OUT / 'campaign_status.json')
        status.update(state='failed', consecutive_failures=status.get('consecutive_failures', 0) + 1, failed_at_utc=c.stamp())
        write(OUT / 'campaign_status.json', status)
        write(OUT / 'failures' / f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc(),
              'budget': c.read(OUT / 'budget.json'), 'created_at_utc': c.stamp()})
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
