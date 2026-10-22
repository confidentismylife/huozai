"""Generate real 14B caches and compare with fixed PPO without rewriting 4B."""
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import replace
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import threading
import time

import numpy as np
import psutil
import requests

from run_timestamped_development_campaign import ROOT, PREP, OLD, read, save, sha, csv_write, unique_output, payload_for
from run_smoke_preference_development_confirmation import verify as verify_previous, FREEZE as CONFIRM_PREP
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, Proposal, ppo_action
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, terminal_metrics

MODEL = 'qwen3:14b'
DIGEST = 'bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8'
BASE = 'http://127.0.0.1:11434'
FREEZE = ROOT / 'outputs/qwen14b_evacuation_preparation_20261007'
CACHE = ROOT / 'outputs/qwen14b_evacuation_cache_20261007'
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'properties': {'target': {'type': 'string', 'enum': ['north', 'south']},
                         'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
          'required': ['target', 'confidence']}
ARMS = ('pure_rule', 'same_weight_no_advice', 'legacy_4b_instant_cache',
        'qwen14b_instant_cache', 'qwen14b_measured_arrival_cache')
METRICS = ('mean_cumulative_fed', 'incapacitation_rate', 'evacuation_rate', 'cumulative_congestion')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def api(path, payload=None, timeout=10):
    response = requests.get(BASE + path, timeout=timeout) if payload is None else requests.post(BASE + path, json=payload, timeout=(5, timeout))
    response.raise_for_status()
    return response.json()


def identity():
    item = next(r for r in api('/api/tags')['models'] if r['name'] == MODEL)
    assert item['digest'] == DIGEST, '14B model digest changed'
    assert item['details']['quantization_level'] == 'Q4_K_M'
    return item


def verify():
    verify_previous()
    if FREEZE.exists():
        for p, h in read(FREEZE / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p
        for p, h in read(FREEZE / 'preparation_manifest.json')['files'].items():
            assert sha(ROOT / p) == h, p


def grade_response(response):
    if not response.get('done') or response.get('done_reason') == 'length':
        return None, 'truncated_or_incomplete'
    try:
        content = response['message']['content']
        value = json.loads(content)
        assert type(value) is dict and set(value) == {'target', 'confidence'}
        assert value['target'] in ('north', 'south')
        confidence = value['confidence']
        assert type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
    except (KeyError, TypeError, ValueError, AssertionError):
        return None, 'invalid_format'
    return {'target_branch': int(value['target'] == 'south'), 'confidence': float(confidence)}, 'valid'


def cached_proposal(record, initial, measured_arrival=False):
    if record['advice'] is None:
        return None
    advice = record['advice']
    p = Proposal.parse(payload_for(initial, advice['target_branch'], advice['confidence']), initial,
                       0., 'qwen14b-' + initial.episode_id, 'real_qwen3_14b_cache')
    if measured_arrival:
        p = replace(p, received_time_s=float(record['latency_seconds']),
                    provenance='real_qwen3_14b_cache_with_measured_wall_arrival')
    return p


def prepare():
    assert not FREEZE.exists() and not CACHE.exists()
    verify_previous()
    model = identity()
    assert not api('/api/ps')['models'], 'Do not replace another currently loaded model'
    first = read(PREP / 'manifest.json')
    second = read(CONFIRM_PREP / 'manifest.json')
    scenes = first['scenarios'] + second['scenes']
    assert len(scenes) == len({s['scenario_id'] for s in scenes}) == 24
    requests_plan = []
    for scene in sorted(scenes, key=lambda r: r['scenario_id']):
        previous = read(OLD / 'llm_cache' / (scene['scenario_id'] + '.json'))
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': previous['prompt']}],
                   'format': SCHEMA, 'think': False, 'stream': False, 'keep_alive': '5m',
                   'options': {'temperature': 0, 'seed': 2026100551, 'num_predict': 96,
                               'num_ctx': 2048, 'num_batch': 32, 'num_thread': 8, 'num_gpu': 0}}
        requests_plan.append({'scenario_id': scene['scenario_id'], 'payload': payload,
                              'payload_sha256': digest_json(payload), 'historical_cache_sha256': sha(OLD / 'llm_cache' / (scene['scenario_id'] + '.json'))})
    random.Random(2026100741).shuffle(requests_plan)
    files = [Path(__file__), ROOT / 'tests/test_qwen14b_evacuations.py', PREP / 'manifest.json', CONFIRM_PREP / 'manifest.json',
             ROOT / 'outputs/smoke_preference_development_confirmation_20261007T032614390080Z/completion_manifest.json',
             ROOT / 'outputs/local_llm_selection.json', ROOT / 'outputs/third_round_development_status.json']
    files += [OLD / 'llm_cache' / (s['scenario_id'] + '.json') for s in scenes]
    files += [Path(first['ppo_models'][str(seed)]['raw_llm_ppo']) for seed in first['training_seeds']]
    FREEZE.mkdir()
    save(FREEZE / 'manifest.json', dict(model=MODEL, model_digest=DIGEST, model_identity=model,
        request_order=requests_plan, scenarios=scenes, primary_scenario_ids=sorted(s['scenario_id'] for s in second['scenes']),
        exploratory_scenario_ids=sorted(s['scenario_id'] for s in first['scenarios']),
        request_attempt_cap=24, no_answer_retry=True, inference_wall_seconds_cap=1800,
        inference_request_timeout_seconds=180, startup_available_memory_floor_bytes=10 * 1024**3,
        sampled_available_memory_floor_bytes=256 * 1024**2,
        full_episode_cap=2016, evaluation_wall_seconds_cap=600, training_seeds=first['training_seeds'],
        ppo_models={str(seed): first['ppo_models'][str(seed)]['raw_llm_ppo'] for seed in first['training_seeds']},
        behavior_models=['rule', 'calibrated_logit'], arms=ARMS,
        guidance_mode='One cached direction/confidence at initial time. Same payload_for adapter, action constraints, prior strength .25 and TTL15s as historical 4B.',
        forecast_scope='Same recorded forecast-only prompt as the 4B cache; no true source_branch is added. Forecast growth, onset and reopen fields are supplied as simulator forecasts, not perfect real-world knowledge.',
        latency_mode='Instant cached availability isolates advice effects. Measured HTTP delay is replayed separately; no timestamp refresh. A 50s answer is stale under TTL15s. This is a delay replay, not a live rolling deployment.',
        model_comparison_caveat='Identical text/schema/temp/seed/output cap to the old 4B. Context/batch/CPU limits explicitly bounded for 14B; old 4B engine defaults are not all recorded, so no pure parameter-scale causal claim.',
        pairing='Population seed 1467100000+global_scene_index*10+rep. Same fixed PPO and person/time uniform table across all arms. No PPO training.',
        statistical_unit='Scenario cluster; 2 population replicates, 2 behaviors, 5 old checkpoints; 12-scene primary cohort and separate 12-scene mechanism cohort.',
        invalid_policy='Invalid or truncated advice remains recorded and falls back to no advice; no selective regeneration.',
        new_training=0, reserved_final_scenarios_opened=0, old_caches_and_reports_not_relabelled=True,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    save(FREEZE / 'preparation_manifest.json', {'files': {(FREEZE / 'manifest.json').relative_to(ROOT).as_posix(): sha(FREEZE / 'manifest.json')}})
    CACHE.mkdir()
    (CACHE / 'responses').mkdir()
    (CACHE / 'requests').mkdir()
    save(CACHE / 'model_selection.json', dict(model=MODEL, digest=DIGEST, quantization='Q4_K_M',
         user_instruction='Qwen3 4B缓存使用14B的', current_experiment=str(FREEZE),
         historical_selection_file_preserved=True, affects_only_new_14b_branch=True))
    print(json.dumps({'prepared': str(FREEZE), 'model': MODEL, 'new_requests': 24, 'full_episodes': 2016}, ensure_ascii=False), flush=True)


class MemorySamples:
    def __init__(self):
        self.samples = []
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.event.is_set():
            self.samples.append({'monotonic': time.monotonic(), 'available_bytes': psutil.virtual_memory().available})
            self.event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join(timeout=2)
        return self.samples


def infer():
    verify()
    identity()
    plan = read(FREEZE / 'manifest.json')
    start = time.monotonic()
    already = list((CACHE / 'responses').glob('*.json'))
    floor = plan['startup_available_memory_floor_bytes'] if not already else 512 * 1024**2
    assert psutil.virtual_memory().available >= floor, 'Insufficient available memory for frozen 14B request workload'
    counts = Counter()
    for i, rec in enumerate(plan['request_order']):
        result_file = CACHE / 'responses' / (rec['scenario_id'] + '.json')
        marker = CACHE / 'requests' / (rec['scenario_id'] + '.json')
        if result_file.exists():
            result = read(result_file)
            assert result['payload_sha256'] == rec['payload_sha256']
            counts[result['format_status']] += 1
            continue
        assert not marker.exists(), 'Interrupted attempt cannot be repeated silently'
        assert time.monotonic() - start <= plan['inference_wall_seconds_cap']
        assert psutil.virtual_memory().available >= 512 * 1024**2
        assert digest_json(rec['payload']) == rec['payload_sha256']
        save(marker, dict(scenario_id=rec['scenario_id'], payload=rec['payload'], payload_sha256=rec['payload_sha256'], attempt=1))
        monitor = MemorySamples()
        monitor.start()
        started = time.monotonic()
        try:
            response = api('/api/chat', rec['payload'], timeout=plan['inference_request_timeout_seconds'])
        except Exception as exc:
            samples = monitor.stop()
            save(CACHE / ('failed_' + rec['scenario_id'] + '.json'), dict(error=str(exc), elapsed_seconds=time.monotonic() - started, memory_samples=samples))
            raise
        latency = time.monotonic() - started
        samples = monitor.stop()
        assert response.get('model') == MODEL
        advice, status = grade_response(response)
        result = dict(scenario_id=rec['scenario_id'], model=MODEL, model_digest=DIGEST,
                      prompt=rec['payload']['messages'][0]['content'], payload_sha256=rec['payload_sha256'],
                      advice=advice, format_status=status, latency_seconds=latency,
                      input_observation_time_s=0., raw_response=response, memory_samples=samples,
                      minimum_sampled_available_memory_bytes=min(r['available_bytes'] for r in samples), attempts=1)
        save(result_file, result)
        counts[status] += 1
        if i == 0:
            save(CACHE / 'loaded_model_snapshot.json', api('/api/ps'))
        save(CACHE / 'progress.json', dict(completed=len(list((CACHE / 'responses').glob('*.json'))), target=24,
             last_scenario=rec['scenario_id'], elapsed_seconds=time.monotonic() - start,
             new_training=0, reserved_final_scenarios_opened=0))
        print(json.dumps({'completed': i + 1, 'total': 24, 'scenario': rec['scenario_id'], 'status': status,
                          'advice': advice, 'latency_seconds': round(latency, 2),
                          'minimum_available_GiB': round(result['minimum_sampled_available_memory_bytes'] / 1024**3, 2)}, ensure_ascii=False), flush=True)
        if result['minimum_sampled_available_memory_bytes'] < plan['sampled_available_memory_floor_bytes']:
            api('/api/generate', {'model': MODEL, 'keep_alive': 0}, timeout=30)
            raise RuntimeError('14B memory pressure; retain completed responses and stop further requests')
    assert len(list((CACHE / 'responses').glob('*.json'))) == 24
    identity()
    # Release the model loaded by this workload; do not change the running service.
    api('/api/generate', {'model': MODEL, 'keep_alive': 0}, timeout=30)
    save(CACHE / 'inference_summary.json', dict(completed=24, format_counts=dict(counts),
         wall_seconds_this_invocation=time.monotonic() - start, model=MODEL, model_digest=DIGEST,
         model_unload_maintenance_requests=1, no_new_training=True))
    verify()
    save(CACHE / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in CACHE.rglob('*') if p.is_file()}})
    print('Real 14B advice cache complete.', flush=True)


def verify_cache():
    verify()
    for p, h in read(CACHE / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p


def evaluate():
    import torch
    from stable_baselines3 import PPO
    verify_cache()
    plan = read(FREEZE / 'manifest.json')
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['ppo_models'][str(s)], device='cpu') for s in plan['training_seeds']}
    out = unique_output('qwen14b_evacuation_comparison')
    save(out / 'run_manifest.json', dict(preparation_sha256=sha(FREEZE / 'manifest.json'),
         cache_completion_sha256=sha(CACHE / 'completion_manifest.json'), full_episode_cap=2016,
         primary_cohort=plan['primary_scenario_ids'], new_training=0, new_inference_during_simulation=0))
    schedule = [(si, behavior, rep, seed, arm) for si in range(24) for behavior in plan['behavior_models']
                for rep in range(2) for arm in ARMS for seed in ([0] if arm == 'pure_rule' else plan['training_seeds'])]
    random.Random(2026100742).shuffle(schedule)
    assert len(schedule) == 2016
    rows = []
    started = time.monotonic()
    for si, behavior, rep, seed, arm in schedule:
        if time.monotonic() - started > plan['evaluation_wall_seconds_cap']:
            csv_write(out / 'partial_episodes.csv', rows)
            raise RuntimeError('Frozen evaluation time budget exhausted')
        scene = DynamicConflictScenario(**plan['scenarios'][si])
        actor = RuleBehavior() if behavior == 'rule' else CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
        original_env = TimestampedControlEnv(actor, (scene,))
        popseed = 1467100000 + si * 10 + rep
        original_env.reset(seed=popseed, options={'scenario': scene})
        env = clone_for_future(original_env, popseed + 770000)
        original_env.close()
        initial = env.snapshot
        cached = read(CACHE / 'responses' / (scene.scenario_id + '.json'))
        p = None
        if arm.startswith('qwen14b'):
            p = cached_proposal(cached, initial, arm == 'qwen14b_measured_arrival_cache')
        elif arm == 'legacy_4b_instant_cache':
            old = read(OLD / 'llm_cache' / (scene.scenario_id + '.json'))['advice']
            p = Proposal.parse(payload_for(initial, old['target_branch'], old['confidence']), initial,
                               0., 'legacy4b', 'historical_qwen3_4b_cache')
        counts = Counter()
        while True:
            s = env.snapshot
            active = p
            if p is not None and arm == 'qwen14b_measured_arrival_cache' and s.observation_time_s < p.received_time_s:
                active = None
                counts['pending_steps'] += 1
            gate = None if active is None else env.constraints.validate_proposal(active, s, s.observation_time_s)
            counts['advice_accepted_steps'] += int(gate is not None and gate.accepted)
            counts['advice_rejected_steps'] += int(gate is not None and not gate.accepted)
            if gate is not None:
                for reason in gate.reasons:
                    counts['reason_' + reason] += 1
            if arm == 'pure_rule':
                action = env.constraints.rule_action(s)
            else:
                action, raw = ppo_action(models[seed], env.policy_observation(active, gate), env.constraints.action_mask(s), active, gate,
                    elapsed_since_receipt=0 if active is None else s.observation_time_s - active.received_time_s)
                counts['raw_masked_steps'] += int(not env.constraints.action_mask(s)[raw])
            _, _, term, trunc, info = paired_step(env, action)
            counts['unavailable_entries'] += int(info['unavailable_corridor_entries'])
            counts['release_violations'] += int(not info['actual_release_within_limit'])
            counts['direction_violations'] += int(info['actual_direction_violation'])
            counts['steps'] += 1
            if term or trunc:
                break
        rows.append(dict(scenario_id=scene.scenario_id, cohort='primary_separate_development' if scene.scenario_id in plan['primary_scenario_ids'] else 'exploratory_mechanism_development',
            behavior=behavior, population_seed=popseed, training_seed=seed, arm=arm,
            request_latency_seconds=float(cached['latency_seconds']), **terminal_metrics(info, env.unwrapped.domain_env.config),
            **{k: int(counts[k]) for k in ('steps', 'pending_steps', 'advice_accepted_steps', 'advice_rejected_steps',
                'reason_stale', 'reason_timeout', 'reason_live_smoke_contradiction', 'raw_masked_steps',
                'unavailable_entries', 'release_violations', 'direction_violations')}))
        env.close()
        if len(rows) % 300 == 0:
            print(json.dumps({'full_episodes': len(rows), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    assert len(rows) == 2016
    csv_write(out / 'episodes.csv', rows)
    analyze(out, rows, time.monotonic() - started)
    verify_cache()
    save(out / 'completion_manifest.json', {'state': 'complete', 'sources_unchanged': True,
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'completed_full_episodes': len(rows)}, ensure_ascii=False), flush=True)


def analyze(out, rows, elapsed):
    index = {(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['arm']): r for r in rows}
    comparisons = [('legacy_4b_instant_cache', 'same_weight_no_advice'),
                   ('qwen14b_instant_cache', 'same_weight_no_advice'),
                   ('qwen14b_instant_cache', 'legacy_4b_instant_cache'),
                   ('qwen14b_measured_arrival_cache', 'same_weight_no_advice')]
    effects = []
    for cohort in ('primary_separate_development', 'exploratory_mechanism_development', 'all_development'):
        for candidate, baseline in comparisons:
            base = [r for r in rows if r['arm'] == baseline and (cohort == 'all_development' or r['cohort'] == cohort)]
            for metric in METRICS:
                clusters = defaultdict(list)
                nonzero = 0
                for b in base:
                    a = index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], candidate)]
                    d = a[metric] - b[metric]
                    clusters[b['scenario_id']].append(d)
                    nonzero += int(abs(d) > 1e-12)
                values = np.array([np.mean(v) for v in clusters.values()])
                rng = np.random.default_rng(2026100743)
                boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                effects.append(dict(cohort=cohort, candidate=candidate, baseline=baseline, metric=metric,
                    paired_episodes=len(base), scenario_clusters=len(clusters), nonzero_pairs=nonzero,
                    mean_difference=float(values.mean()), descriptive_scene_bootstrap_95_interval=list(map(float, np.quantile(boot, [.025, .975])))))
    requests_data = [read(p) for p in sorted((CACHE / 'responses').glob('*.json'))]
    latencies = [r['latency_seconds'] for r in requests_data]
    means = [dict(cohort=cohort, arm=arm, episodes=len(selected),
             **{m: float(np.mean([r[m] for r in selected])) for m in METRICS},
             advice_accepted_steps=sum(r['advice_accepted_steps'] for r in selected))
             for cohort in ('primary_separate_development', 'exploratory_mechanism_development') for arm in ARMS
             for selected in [[r for r in rows if r['arm'] == arm and r['cohort'] == cohort]]]
    summary = dict(completed_full_episodes=len(rows), elapsed_seconds=elapsed, model=MODEL, model_digest=DIGEST,
        fresh_14b_requests=24, request_format_counts=dict(Counter(r['format_status'] for r in requests_data)),
        mean_request_latency_seconds=float(np.mean(latencies)), min_request_latency_seconds=min(latencies),
        max_request_latency_seconds=max(latencies), latency_above_15s=sum(t > 15 for t in latencies),
        minimum_sampled_available_memory_bytes=min(r['minimum_sampled_available_memory_bytes'] for r in requests_data),
        effects=effects, arm_means=means,
        violations={k: sum(r[k] for r in rows) for k in ('unavailable_entries', 'release_violations', 'direction_violations')},
        new_training=0, reserved_final_scenarios_opened=0, no_live_deployment_claim=True,
        no_parameter_scale_causal_claim=True, old_caches_reports_status_and_weights_unchanged=True)
    save(out / 'summary.json', summary)
    lines = ['# Qwen3 14B真实缓存替换与疏散开发对照', '',
        '本轮按用户要求使用qwen3:14b、Q4_K_M，实际生成24个疏散场景的响应缓存。旧4B缓存保留原样；没有把旧结果改名为14B。', '',
        f"模型digest：`{DIGEST}`。响应状态：{summary['request_format_counts']}。实际HTTP平均耗时{np.mean(latencies):.2f}秒，范围{min(latencies):.2f}—{max(latencies):.2f}秒；{summary['latency_above_15s']}/24次超过15秒TTL。", '',
        '文本提示与原4B逐场景一致，只含原来已经给出的初始占用、预测与地图容量。JSON schema、temperature0、seed2026100551、96输出token预算一致；14B上下文2048、batch32、CPU8、GPU0显式限制。旧4B引擎默认选项没有完整记录，不能作纯参数规模因果归因。', '',
        '24场景分成两批：此前机制审计12场景和另行冻结开发验证12场景，后者作为主要结果。两批都属于复用开发数据，最终保留场景仍封存。', '',
        '完成2016个完整疏散回合：纯规则、同权重无建议PPO、旧4B即时缓存、真实14B即时缓存、14B实测返回延迟回放。两个行为模型、两个人口种子、PPO组五套既有权重；动作、物理约束、PPO权重和人/时刻随机数配对。', '',
        '即时缓存组在模拟初始时刻提供预生成建议，只衡量建议本身对控制的影响；延迟组等到实测HTTP耗时对应的控制步才提供原时刻建议，时间戳不刷新，每步重新校验。不把即时缓存当成真实在线响应速度。', '',
        '|主要开发子集比较|指标|均差（候选−对照）|描述性场景95%区间|非零配对|', '|---|---|---:|---|---:|']
    for r in effects:
        if r['cohort'] == 'primary_separate_development':
            lo, hi = r['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{r['candidate']} − {r['baseline']}|{r['metric']}|{r['mean_difference']:+.7f}|{lo:.7f}, {hi:.7f}|{r['nonzero_pairs']}/{r['paired_episodes']}|")
    lines += ['', '差值为率时乘100才是百分点；FED和失能率负差为改善，疏散率正差为改善。置信区间为复用开发场景的描述性bootstrap，条件于已有权重。', '',
        '原始响应、请求与模型身份在独立14B缓存目录；无效/截断回答记录并回退，未挑选重试。自报confidence仍未校准。没有新增训练，未修改预测盾资格或原始报告。', '',
        '全部回合的执行限制检查见summary.json；仿真结束时间包含失能终止，只有全部疏散成功才记录all_evacuated_seconds。', '',
        '完整分组、逐回合数据与两批场景的结果见summary.json与episodes.csv。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'infer', 'evaluate', 'verify'])
    args = p.parse_args()
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate, 'verify': verify}[args.stage]()
