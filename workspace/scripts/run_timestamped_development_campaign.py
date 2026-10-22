"""Frozen bounded development evaluation; old weights/ledgers remain read-only."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, Proposal, ppo_action

PREP = ROOT / 'outputs/timestamped_pipeline_preparation_20261007'
PROTOCOL = ROOT / 'reports/27B提速与时间戳控制流程固定方案_2026-10-07.md'
OLD = ROOT / 'outputs/evacuation_auto_llm_independent_20261005'
MODEL_DIR = ROOT / 'outputs/guarded_confirmatory_v2/models'
SEEDS = (601, 602, 603, 604, 605)
ARMS = ('pure_rule', 'pure_ppo', 'rule_advice_ppo', 'cached_llm_ppo', 'wrong_advice_ppo', 'delayed_advice_ppo', 'timeout_advice_ppo')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def unique_output(prefix):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = ROOT / 'outputs' / (prefix + '_' + stamp)
    result.mkdir()
    return result


def protected_files():
    return [ROOT / 'outputs/third_round_development_status.json', ROOT / 'outputs/local_llm_selection.json',
            ROOT / 'outputs/qwen35_label_order_probe_20261006/preparation_manifest.json',
            ROOT / 'outputs/qwen35_27b_text_final_audit_20261006/completion_manifest.json']


def prepare():
    if PREP.exists():
        verify_preparation()
        print('Existing preparation verified; no overwrites.', flush=True)
        return
    old = read(OLD / 'manifest.json')
    groups = defaultdict(list)
    for s in old['banks']['llm']:
        groups[s['conflict_type']].append(s)
    selected = [s for typ in sorted(groups) for s in sorted(groups[typ], key=lambda r: r['scenario_id'])[:2]]
    assert len(selected) == 12 and len(groups) == 6
    files = [PROTOCOL, Path(__file__), ROOT / 'scripts/run_qwen35_timestamped_live.py',
             ROOT / 'src/fire_evac/timestamped_pipeline.py', ROOT / 'tests/test_timestamped_pipeline.py',
             ROOT / 'tests/test_qwen35_timestamped_live.py',
             OLD / 'manifest.json', OLD / 'llm_advice_frozen.json',
             ROOT / 'outputs/behavior_pipeline/behavior_model.json',
             ROOT / 'outputs/qwen35_label_order_probe_20261006/manifest.json',
             ROOT / 'outputs/qwen35_27b_text_creation_20261006/text_model_identity.json',
             ROOT / 'outputs/llm_goal_role_20261006/manifest.json',
             ROOT / 'outputs/qwen35_27b_text_low_memory_20261006/summary.json']
    files += protected_files()
    files += list((ROOT / 'src/fire_evac').glob('*.py'))
    files += [OLD / 'llm_cache' / (s['scenario_id'] + '.json') for s in selected]
    files += [MODEL_DIR / f'seed_{seed}_{arm}.zip' for seed in SEEDS for arm in ('pure_ppo', 'rule_advice_ppo', 'raw_llm_ppo')]
    expected = read(ROOT / 'outputs/qwen35_label_order_probe_20261006/preparation_manifest.json')['files']
    assert all(sha(ROOT / p) == digest for p, digest in expected.items())
    # Preserve the precise old 27B identity; never load/hash its 16GB weight into RAM.
    PREP.mkdir()
    confirmation = confirmation_requests()
    save(PREP / 'confirmation_manifest.json', {'purpose': 'separate_numerical_semantic_confirmation',
         'not_fire_scenarios': True, 'physical_families': 2, 'planned_requests': 16,
         'probabilities': [.60, .73], 'correction': 'explicit role-to-cost arithmetic instruction and independent observable arithmetic validator',
         'raw_and_rule_checked_scores_must_be_separate': True, 'request_order': confirmation})
    manifest = {'created_at_utc': utc(), 'scope': 'development_only', 'scenario_origin': str(OLD / 'manifest.json'),
                'scenarios': selected, 'training_seeds': SEEDS, 'population_seeds': [1407100000 + i * 10 + r for i in range(12) for r in range(2)],
                'repetitions': 2, 'behavior_models': ['calibrated_logit', 'rule'], 'arms': ARMS,
                'simulations_cap': 1488, 'offline_seconds_cap': 720,
                'live_request_cap': 58, 'live_seconds_cap': 3300,
                'speed_request_cap': 6, 'diagnostic_request_cap': 32, 'confirmation_request_cap': 16, 'live_proposal_request_cap': 4,
                'startup_memory_floor_bytes': 17 * 1024**3, 'sampled_memory_floor_bytes': 256 * 1024**2,
                'advice_ttl_seconds': 15, 'advice_timeout_seconds': 60,
                'cached_llm_model': 'qwen3:4b (Oct5 recorded cache; NOT a new 27B evaluation)',
                'ppo_models': {str(s): {a: str(MODEL_DIR / f'seed_{s}_{a}.zip') for a in ('pure_ppo', 'rule_advice_ppo', 'raw_llm_ppo')} for s in SEEDS},
                'new_ppo_updates': 0, 'new_fits': 0, 'reserved_final_scenarios_opened': 0,
                'mask_is_operational_not_qualified_predictive_shield': True,
                'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}}
    save(PREP / 'manifest.json', manifest)
    save(PREP / 'preparation_manifest.json', {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in PREP.iterdir() if p.is_file()},
                                            'model_requests': 0, 'new_training': 0})
    verify_preparation()
    print(json.dumps({'prepared': str(PREP), 'scenarios': 12, 'offline_rollouts': 1488, 'live_cap': 58}, ensure_ascii=False), flush=True)


def confirmation_requests():
    """New probabilities/mirror families; no private reference in messages."""
    from llm_goal_role_core import PREFIX
    result = []
    for q, physical, role, objective in itertools.product((.60, .73), (0, 1), ('hazard', 'clear'), ('minimize', 'maximize')):
        reported = physical if role == 'hazard' else 1 - physical
        data = {'both_routes_open': True, 'north_exit_capacity': 7, 'south_exit_capacity': 7,
                'north_route_distance': 31, 'south_route_distance': 31, 'initial_room_occupancy': [95, 95],
                'measured_smoke_north': 0, 'measured_smoke_south': 0, 'reported_location': ('north', 'south')[reported],
                'report_role': role, 'probability_report_correct': q, 'objective': objective}
        prompt = PREFIX.replace('Data: ', '') + (
            'Arithmetic check: let p=probability_report_correct. The expected cost AT reported_location is p if '
            'report_role=hazard, and 1-p if report_role=clear. The other cost is one minus that cost. '
            'Then compare the two costs under objective; changing labels must not change physical selection. Data: ')
        at = q if role == 'hazard' else 1 - q
        costs = {('north', 'south')[reported]: at, ('north', 'south')[1-reported]: 1-at}
        ref = (min if objective == 'minimize' else max)(costs, key=costs.get)
        result.append({'request_id': f'CF_{q}_{physical}_{role}_{objective}', 'family': str(q), 'mirror': physical,
                       'role': role, 'objective': objective, 'naming': 'compass', 'candidate_order': ['north', 'south'],
                       'payload': {'model': 'qwen35-27b-text-q4:latest', 'messages': [{'role': 'user', 'content': prompt + json.dumps(data, sort_keys=True)}],
                                   'format': {'type': 'object', 'additionalProperties': False, 'properties': {
                                       'recommended_exit': {'type': 'string', 'enum': ['north', 'south']},
                                       'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}, 'required': ['recommended_exit', 'confidence']},
                                   'options': {'temperature': 0, 'seed': 2026100701, 'num_predict': 1024, 'num_ctx': 2048, 'num_batch': 32},
                                   'think': False, 'stream': False, 'keep_alive': 0},
                       'private_reference': {'reference_exit': ref, 'reference_physical_exit': int(ref == 'south'),
                                             'label_to_physical': {'north': 0, 'south': 1}, 'expected_cost_by_label': costs}})
    random.Random(2026100703).shuffle(result)
    assert len(result) == 16
    return result


def verify_preparation():
    for p, digest in read(PREP / 'preparation_manifest.json')['files'].items():
        assert sha(ROOT / p) == digest, p
    for p, digest in read(PREP / 'manifest.json')['source_sha256'].items():
        assert sha(ROOT / p) == digest, p


def payload_for(s, branch, confidence, actions=None):
    return dict(observation_time_s=s.observation_time_s, preferred_exit=('north', 'south')[branch], confidence=confidence,
                candidate_actions=actions or [branch * 4, branch * 4 + 1], wait_steps=0, north_share=float(branch == 0))


def evaluate():
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    verify_preparation()
    plan = read(PREP / 'manifest.json')
    out = unique_output('timestamped_pipeline_development')
    save(out / 'run_manifest.json', {'plan_sha256': sha(PREP / 'manifest.json'), 'started': utc(),
         'fresh_llm_requests': 0, 'mode': 'cached_and_fault_injection_only', 'no_new_27b_efficacy_claim': True})
    models = {(seed, a): PPO.load(plan['ppo_models'][str(seed)][a], device='cpu') for seed in SEEDS for a in ('pure_ppo', 'rule_advice_ppo', 'raw_llm_ppo')}
    assert all(m.action_space.n == 9 and m.observation_space.shape == (34,) for m in models.values())
    rows, audit, start = [], [], time.monotonic()
    trace = (out / 'execution_trace.jsonl').open('w', encoding='utf-8')
    schedule = []
    for si, r in enumerate(plan['scenarios']):
        for behavior, rep, arm in itertools.product(plan['behavior_models'], range(2), ARMS):
            for seed in ((0,) if arm == 'pure_rule' else SEEDS):
                schedule.append((si, r, behavior, rep, arm, seed))
    random.Random(2026100704).shuffle(schedule)
    assert len(schedule) == plan['simulations_cap']
    status = 'complete'
    try:
        for si, r, behavior, rep, arm, seed in schedule:
            if time.monotonic() - start > plan['offline_seconds_cap']:
                status = 'stopped_time_budget'
                break
            scene = DynamicConflictScenario(**r)
            actor_behavior = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
            env = TimestampedControlEnv(actor_behavior, (scene,))
            popseed = 1407100000 + si * 10 + rep
            _, info = env.reset(seed=popseed, options={'scenario': scene})
            initial = env.snapshot
            cached = read(OLD / 'llm_cache' / (scene.scenario_id + '.json'))
            cached_p = Proposal.parse(payload_for(initial, cached['advice']['target_branch'], cached['advice']['confidence']),
                                      initial, initial.observation_time_s, 'cache-' + scene.scenario_id, 'cached_qwen3_4b')
            ended = False
            sums = Counter()
            total_reward = 0.
            first_all_evacuated = None
            while not ended:
                s = env.snapshot
                p, gate = None, None
                source = 'none'
                if arm in ('cached_llm_ppo', 'delayed_advice_ppo'):
                    p = cached_p
                    source = p.provenance
                    if arm == 'delayed_advice_ppo' and s.observation_time_s < 50:
                        p = None
                        sums['pending_steps'] += 1
                    elif arm == 'delayed_advice_ppo':
                        from dataclasses import replace
                        p = replace(p, received_time_s=50., provenance='cache_with_injected_50s_delay')
                elif arm == 'wrong_advice_ppo':
                    # Observable smoke-contradiction fixture; never peek at true fire source.
                    if abs(s.smoke[3] - s.smoke[4]) > .01:
                        worse = int(s.smoke[4] > s.smoke[3])
                        p = Proposal.parse(payload_for(s, worse, .99), s, s.observation_time_s, f'wrong-{s.step}', 'injected_observable_contradiction')
                        source = p.provenance
                        sums['wrong_injections'] += 1
                elif arm == 'rule_advice_ppo':
                    act = env.constraints.rule_action(s)
                    if act != 8:
                        b = act // 4
                        p = Proposal.parse(payload_for(s, b, .8, [act]), s, s.observation_time_s, f'rule-{s.step}', 'live_rule_proposal')
                        source = p.provenance
                elif arm == 'timeout_advice_ppo':
                    sums['pending_steps'] += int(s.observation_time_s <= 60)
                    sums['timeout_fallback_steps'] += int(s.observation_time_s > 60)
                if p is not None:
                    gate = env.constraints.validate_proposal(p, s, s.observation_time_s)
                    sums['advice_checked_steps'] += 1
                    sums['advice_accepted_steps'] += int(gate.accepted)
                    sums['advice_rejected_steps'] += int(not gate.accepted)
                    for reason in gate.reasons:
                        sums[reason] += 1
                    if arm == 'wrong_advice_ppo':
                        sums['wrong_rejected'] += int(not gate.accepted and 'live_smoke_contradiction' in gate.reasons)
                if arm == 'pure_rule':
                    requested = raw = env.constraints.rule_action(s)
                else:
                    mname = 'pure_ppo' if arm in ('pure_ppo', 'timeout_advice_ppo') else 'rule_advice_ppo' if arm == 'rule_advice_ppo' else 'raw_llm_ppo'
                    x = env.policy_observation(p, gate)
                    requested, raw = ppo_action(models[(seed, mname)], x, env.constraints.action_mask(s), p, gate,
                                                elapsed_since_receipt=0 if p is None else s.observation_time_s - p.received_time_s)
                    sums['ppo_raw_masked'] += int(not env.constraints.action_mask(s)[raw])
                sums['policy_final_differs_raw'] += requested != raw
                if p is not None and gate.accepted:
                    if requested != 8:
                        sums['accepted_advice_direction_choices'] += 1
                        sums['ppo_direction_overrides'] += int(requested // 4 != p.preferred_branch)
                    sums['ppo_outside_proposal_candidates'] += requested not in p.candidate_actions
                _, reward, term, trunc, info = env.step(requested)
                sums['steps'] += 1
                sums['constraint_interventions'] += info['constraint_intervened']
                sums['actual_direction_violations'] += info['actual_direction_violation']
                sums['release_violations'] += not info['actual_release_within_limit']
                sums['unavailable_entries'] += info['unavailable_corridor_entries']
                sums['requested_actual_mismatches'] += not info['requested_actual_match']
                sums['unavailable_sign_feedback_steps'] += info['actual_sign_unavailable']
                if arm == 'wrong_advice_ppo' and p is not None:
                    sums['wrong_executed_direction'] += int(info['actual_joint_action'] != 8 and info['actual_sign_action'] == p.preferred_branch)
                total_reward += reward
                if info['evacuated'] == env.unwrapped.domain_env.config.population and first_all_evacuated is None:
                    first_all_evacuated = info['step'] * 5.
                record = dict(scenario_id=scene.scenario_id, behavior=behavior, population_seed=popseed, training_seed=seed,
                              arm=arm, step=s.step, observation_time_s=s.observation_time_s, source=source,
                              proposal_observation_time_s=None if p is None else p.observation_time_s,
                              proposal_received_time_s=None if p is None else p.received_time_s,
                              gate_accepted=None if gate is None else gate.accepted,
                              gate_reasons=[] if gate is None else gate.reasons,
                              ppo_raw_action=raw, requested_action=requested,
                              applied_action=info['constraint_applied_action'], actual_joint_action=info['actual_joint_action'],
                              actual_sign=info['actual_sign_action'], actual_room_count=info['actual_room_release_count'],
                              room_limit=info['pretransition_room_limit'], unavailable_entries=info['unavailable_corridor_entries'])
                trace.write(json.dumps(record, ensure_ascii=False) + '\n')
                ended = term or trunc
            row = dict(scenario_id=scene.scenario_id, conflict_type=scene.conflict_type, behavior=behavior,
                       population_seed=popseed, training_seed=seed, arm=arm,
                       mean_cumulative_fed=info['mean_cumulative_fed'], cumulative_congestion=info['cumulative_congestion'],
                       evacuation_rate=info['evacuation_rate'], incapacitation_rate=info['incapacitation_rate'],
                       episode_end_seconds=info['step'] * 5., all_evacuated_seconds=first_all_evacuated,
                       all_evacuated=int(first_all_evacuated is not None), truncated=int(trunc), total_reward=total_reward)
            for key in ('steps', 'pending_steps', 'advice_checked_steps', 'advice_accepted_steps', 'advice_rejected_steps',
                        'stale', 'timeout_fallback_steps', 'live_smoke_contradiction', 'preferred_exit_unavailable',
                        'ppo_raw_masked', 'policy_final_differs_raw', 'ppo_direction_overrides', 'accepted_advice_direction_choices',
                        'ppo_outside_proposal_candidates', 'constraint_interventions', 'actual_direction_violations',
                        'release_violations', 'unavailable_entries', 'requested_actual_mismatches', 'unavailable_sign_feedback_steps',
                        'wrong_injections', 'wrong_rejected', 'wrong_executed_direction'):
                row[key] = int(sums[key])
            rows.append(row)
            env.close()
            if len(rows) % 60 == 0:
                csv_write(out / 'episodes.csv', rows)
                save(out / 'progress.json', {'completed': len(rows), 'planned': len(schedule), 'elapsed_s': time.monotonic() - start})
                print(json.dumps({'completed': len(rows), 'planned': len(schedule), 'elapsed_s': round(time.monotonic() - start, 1)}), flush=True)
    finally:
        trace.close()
    csv_write(out / 'episodes.csv', rows)
    summary = summarize(rows, status, time.monotonic() - start)
    save(out / 'summary.json', summary)
    write_report(out, summary)
    verify_preparation()
    save(out / 'completion_manifest.json', {'state': status, 'source_unchanged': True,
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()},
         'actual_rollouts': len(rows), 'fresh_llm_calls': 0, 'new_ppo_updates': 0, 'reserved_final_scenarios_opened': 0})
    print(json.dumps({'state': status, 'completed': len(rows), 'report': str(out / 'report.md'), 'output': str(out)}, ensure_ascii=False), flush=True)


def summarize(rows, status, elapsed):
    result = {'state': status, 'completed': len(rows), 'elapsed_seconds': elapsed, 'fresh_27b_calls': 0,
              'arms': {}, 'paired_effects': [], 'uncertainty': 'descriptive paired effects on 12 reused development scenarios; not final qualification'}
    metrics = ('mean_cumulative_fed', 'cumulative_congestion', 'episode_end_seconds', 'evacuation_rate', 'incapacitation_rate')
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        if not group:
            continue
        result['arms'][arm] = {'episodes': len(group), **{m: float(np.mean([r[m] for r in group])) for m in metrics},
            'fully_evacuated_episodes': sum(r['all_evacuated'] for r in group),
            'full_evacuation_seconds_mean_if_complete': float(np.mean([r['all_evacuated_seconds'] for r in group if r['all_evacuated']])) if any(r['all_evacuated'] for r in group) else None,
            **{k: sum(r[k] for r in group) for k in ('unavailable_entries', 'release_violations', 'actual_direction_violations',
                'requested_actual_mismatches', 'wrong_injections', 'wrong_rejected', 'wrong_executed_direction',
                'advice_accepted_steps', 'advice_rejected_steps', 'stale', 'ppo_direction_overrides',
                'accepted_advice_direction_choices', 'ppo_outside_proposal_candidates', 'ppo_raw_masked', 'unavailable_sign_feedback_steps')}}
    index = {(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['arm']): r for r in rows}
    for candidate in ('cached_llm_ppo', 'rule_advice_ppo', 'wrong_advice_ppo', 'delayed_advice_ppo', 'timeout_advice_ppo'):
        pairs = []
        for r in rows:
            if r['arm'] != candidate:
                continue
            b = index.get((r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], 'pure_ppo'))
            if b is not None:
                pairs.append((r, b))
        for metric in metrics:
            if pairs:
                d = [a[metric] - b[metric] for a, b in pairs]
                by_scene = defaultdict(list)
                for (a, _), delta in zip(pairs, d):
                    by_scene[a['scenario_id']].append(delta)
                result['paired_effects'].append({'candidate': candidate, 'baseline': 'pure_ppo', 'metric': metric,
                    'paired_episodes': len(pairs), 'scenario_clusters': len(by_scene), 'mean_difference': float(np.mean(d)),
                    'scene_mean_min': min(float(np.mean(v)) for v in by_scene.values()),
                    'scene_mean_max': max(float(np.mean(v)) for v in by_scene.values())})
    for behavior in ('calibrated_logit', 'rule'):
        result.setdefault('by_behavior', {})[behavior] = {a: {m: float(np.mean([r[m] for r in rows if r['arm'] == a and r['behavior'] == behavior])) for m in metrics}
                                                       for a in ARMS if any(r['arm'] == a and r['behavior'] == behavior for r in rows)}
    return result


def write_report(out, s):
    lines = ['# 时间戳提案与统一动作约束：开发仿真报告', '',
             f"完成 {s['completed']}/1488 回合，用时 {s['elapsed_seconds']:.1f} 秒；状态 {s['state']}。", '',
             '本轮真实27B请求为0。缓存LLM来自2026-10-05的Qwen3 4B；错误、50秒延迟与超时是明确的故障注入。', '',
             '使用12个已有开发场景、2个人群行为模型、2个人群种子、5组既有PPO训练权重。没有新训练，没有打开保留最终场景。', '',
             '所有组采用相同9动作及出口封闭、烟气阈值、指示牌保持时间、下游容量和路口队列余量约束；封闭走廊进入容量也设为0。', '',
             'LLM提案经检查后提供目标特征和强度0.25的有限动作偏好；PPO可选择其他允许动作。拒绝提案时清空建议特征。confidence仍是未校准自报值。', '',
             '缓存建议在初始时刻仅提供一次，超过15秒失效；不能刷新时间戳假装是新观测。错误组在实时烟气差大于0.01时提名烟气更高出口，只检验可观察矛盾的处理。', '',
             '|组|回合|人均烟气累积指标|累计拥堵|仿真结束秒|疏散率|失能率|全部疏散回合|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, a in s['arms'].items():
        lines.append(f"|{arm}|{a['episodes']}|{a['mean_cumulative_fed']:.4f}|{a['cumulative_congestion']:.2f}|{a['episode_end_seconds']:.1f}|{a['evacuation_rate']:.3f}|{a['incapacitation_rate']:.3f}|{a['fully_evacuated_episodes']}|")
    lines += ['', '仿真结束时间含失能后终止，不能当作全员成功疏散时间。all_evacuated_seconds只对确实全员疏散的回合记录，其余为空；缩短结束时间但失能增加不是改善。', '',
              '|组|封闭/超阈值走廊进入|放行超限|实际方向违规|PPO原始动作被屏蔽|错误提案被拒绝/注入|错误方向实际执行|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, a in s['arms'].items():
        lines.append(f"|{arm}|{a['unavailable_entries']}|{a['release_violations']}|{a['actual_direction_violations']}|{a['ppo_raw_masked']}|{a['wrong_rejected']}/{a['wrong_injections']}|{a['wrong_executed_direction']}|")
    lines += ['', '“门控拒绝提案”与“PPO覆盖提案”分别计数。建议被拒绝后，下游PPO仍可能按自身权重选择烟气较高但未超阈值的可用出口；不能把拒绝率当成PPO纠正成功率。', '',
              '|候选相对纯PPO|指标|配对回合|场景数|均差（候选−纯PPO）|各场景均差最小/最大|',
              '|---|---|---:|---:|---:|---:|']
    for r in s['paired_effects']:
        if r['candidate'] in ('cached_llm_ppo', 'rule_advice_ppo'):
            lines.append(f"|{r['candidate']}|{r['metric']}|{r['paired_episodes']}|{r['scenario_clusters']}|{r['mean_difference']:.5f}|{r['scene_mean_min']:.5f}/{r['scene_mean_max']:.5f}|")
    lines += ['', '证据边界：开发场景复用、既有策略在新增约束下有分布变化，故此为工程与开发对照。没有把重复种子当成独立场景，也没有宣称预测盾已通过资格审查。烟气仍是简化的每步FED增量，不能直接作为真实人体剂量。', '',
              '分流比例目前仅影响连续控制步中的方向偏好，原9动作不能同一步独立控制两个出口的放行人数；本轮不宣称实现精确同时分流。', '',
              '详细逐步记录在execution_trace.jsonl；回合数据在episodes.csv；分行为与配对结果在summary.json。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'offline', 'verify'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'offline':
        evaluate()
    else:
        verify_preparation()
        print('Preparation hashes verified.')
