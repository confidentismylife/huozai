"""Resume-safe finite live stages, with fresh run folders and resource guards.

Never starts/restarts a model service or closes applications to free memory.
"""
from pathlib import Path
from collections import defaultdict
import copy
import json
import math
import random
import sys
import threading
import time

import psutil
import requests

from run_timestamped_development_campaign import (ROOT, PREP, OLD, read, save, sha, utc,
                                                 unique_output, verify_preparation)

MODEL = 'qwen35-27b-text-q4:latest'
URL = 'http://127.0.0.1:11434'


def api(path, payload=None, timeout=5):
    r = requests.get(URL + path, timeout=(3, timeout)) if payload is None else requests.post(URL + path, json=payload, timeout=(3, timeout))
    r.raise_for_status()
    return r.json()


def preflight():
    result = {'at': utc(), 'available_memory_bytes': psutil.virtual_memory().available,
              'system_pagefile_used_bytes': psutil.swap_memory().used, 'state': 'ready'}
    try:
        result['version'] = api('/api/version')
        result['ps'] = api('/api/ps')
        if result['ps']['models']:
            result['state'] = 'blocked_existing_model_preserved'
        elif result['available_memory_bytes'] < 17 * 1024**3:
            result['state'] = 'blocked_insufficient_memory'
        else:
            tags = api('/api/tags')
            identity = read(ROOT / 'outputs/qwen35_27b_text_creation_20261006/text_model_identity.json')
            match = next((x for x in tags['models'] if x['name'] == MODEL), None)
            if match is None or match['digest'] != identity['manifest_digest']:
                result['state'] = 'blocked_model_identity_mismatch'
            result['tag'] = match
    except requests.RequestException as e:
        result['state'] = 'blocked_service_unavailable'
        result['error_type'] = type(e).__name__
    return result


def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    def reject(_):
        raise ValueError('nonfinite JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def score(response, request):
    row = {'format_valid': False, 'correct': False, 'selected_physical': None,
           'selected_label': None, 'selected_first': None, 'truncated': response.get('done_reason') == 'length'}
    try:
        p = strict_json(response['message']['content'])
        if set(p) != {'recommended_exit', 'confidence'} or p['recommended_exit'] not in request['candidate_order']:
            return row
        confidence = p['confidence']
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return row
        if row['truncated'] or not response.get('done'):
            return row
        row.update(format_valid=True, correct=p['recommended_exit'] == request['private_reference']['reference_exit'],
                   selected_physical=request['private_reference']['label_to_physical'][p['recommended_exit']],
                   selected_label=p['recommended_exit'], selected_first=p['recommended_exit'] == request['candidate_order'][0])
    except (ValueError, KeyError, TypeError):
        pass
    return row


def observable_arithmetic_check(response, request):
    """Explicit deterministic rule, reported separately from LLM accuracy."""
    content = request['payload']['messages'][0]['content']
    data = strict_json(content.split('Data: ', 1)[1])
    labels = request['candidate_order']
    at = data['probability_report_correct'] if data['report_role'] == 'hazard' else 1-data['probability_report_correct']
    other = next(v for v in labels if v != data['reported_location'])
    costs = {data['reported_location']: at, other: 1-at}
    if data['objective'] not in ('minimize', 'maximize') or data['report_role'] not in ('hazard', 'clear'):
        raise ValueError('invalid numerical task')
    expected = (min if data['objective'] == 'minimize' else max)(costs, key=costs.get)
    raw = score(response, request)
    return {'raw_score': raw, 'rule_replaced_or_rejected_raw': not raw['correct'], 'rule_checked_exit': expected,
            'rule_checked_correct': expected == request['private_reference']['reference_exit'],
            'mathematical_rule_not_a_fire_oracle': True}


def relation_summary(requests_list, rows):
    indexed = {r['request_id']: r for r in rows}
    result = {'completed': len(rows), 'planned': len(requests_list),
              'correct': sum(r.get('correct', False) for r in rows),
              'format_valid': sum(r.get('format_valid', False) for r in rows), 'relations': {}}
    factors = ('mirror', 'role', 'objective', 'naming', 'reversed_order')
    definitions = {'naming': True, 'reversed_order': True, 'role': True, 'mirror': False, 'objective': False}
    for varying, equal in definitions.items():
        groups = defaultdict(list)
        for req in requests_list:
            groups[tuple(req.get(f) for f in factors if f != varying)].append(req)
        complete = consistent = both_correct = 0
        for group in groups.values():
            if len(group) != 2:
                continue
            a, b = (indexed.get(r['request_id']) for r in group)
            if a is None or b is None or not a['format_valid'] or not b['format_valid']:
                continue
            complete += 1
            consistent += (a['selected_physical'] == b['selected_physical']) == equal
            both_correct += a['correct'] and b['correct']
        result['relations'][varying] = {'planned_pairs': sum(len(g) == 2 for g in groups.values()),
            'valid_complete_pairs': complete, 'relation_consistent_pairs': consistent, 'both_correct_pairs': both_correct,
            'missing_or_invalid_pairs': sum(len(g) == 2 for g in groups.values()) - complete}
    return result


class Run:
    def __init__(self):
        self.out = unique_output('qwen35_timestamped_live')
        self.start = time.monotonic()
        self.attempts = 0
        self.rows = defaultdict(list)
        self.stop = threading.Event()
        self.samples = []
        self.worker = None
        self.owned_model = False

    def record_status(self, state, **kwargs):
        save(self.out / 'status.json', {'state': state, 'at': utc(), 'attempts': self.attempts,
                                      'elapsed_seconds': time.monotonic() - self.start, **kwargs})
        print(json.dumps({'state': state, 'attempts': self.attempts, **kwargs}, ensure_ascii=False), flush=True)

    def collect(self):
        while not self.stop.is_set():
            self.samples.append({'at': utc(), 'available_memory_bytes': psutil.virtual_memory().available,
                                 'pagefile_used_bytes': psutil.swap_memory().used})
            self.stop.wait(.5)

    def unload(self):
        if self.owned_model:
            result = api('/api/generate', {'model': MODEL, 'keep_alive': 0}, timeout=30)
            ps = api('/api/ps')
            save(self.out / f'unload_{self.attempts}.json', {'response': result, 'ps': ps})
            if any(m['name'] == MODEL for m in ps['models']):
                raise RuntimeError('owned model did not unload')
            self.owned_model = False

    def request(self, stage, req, resident=False):
        if self.attempts >= 58 or time.monotonic() - self.start > 3300:
            raise RuntimeError('campaign budget exhausted')
        if psutil.virtual_memory().available < (1024**3 if resident else 17 * 1024**3):
            raise RuntimeError('available-memory guard')
        self.attempts += 1
        self.owned_model = True
        save(self.out / 'attempts' / f'{self.attempts:03d}.json', {'stage': stage, 'request_id': req['request_id'], 'payload': req['payload']})
        sample_start = len(self.samples)
        begin = time.monotonic()
        self.record_status('running', stage=stage, request_id=req['request_id'])
        try:
            r = requests.post(URL + '/api/chat', json=req['payload'], timeout=(5, 110))
            elapsed = time.monotonic() - begin
            try:
                response = r.json()
            except ValueError:
                response = {'error': r.text[:1000]}
            save(self.out / 'responses' / f'{self.attempts:03d}.json', {'http_status': r.status_code, 'response': response,
                                                                     'latency_seconds': elapsed, 'request_id': req['request_id'], 'stage': stage})
            r.raise_for_status()
            if response.get('model') != MODEL or not response.get('done'):
                raise RuntimeError('incomplete response or model identity mismatch')
            row = dict(request_id=req['request_id'], latency_seconds=elapsed,
                       load_seconds=response.get('load_duration', 0) / 1e9,
                       prompt_seconds=response.get('prompt_eval_duration', 0) / 1e9,
                       generate_seconds=response.get('eval_duration', 0) / 1e9,
                       prompt_tokens=response.get('prompt_eval_count'), output_tokens=response.get('eval_count'))
            row['generate_tps'] = row['output_tokens'] / row['generate_seconds'] if row['generate_seconds'] else None
            if 'private_reference' in req:
                row.update(score(response, req))
            self.rows[stage].append(row)
            save(self.out / f'{stage}_rows.json', self.rows[stage])
            if req['payload'].get('keep_alive') == 0:
                self.unload()
            else:
                ps = api('/api/ps')
                save(self.out / f'loaded_{self.attempts}.json', ps)
                match = next(m for m in ps['models'] if m['name'] == MODEL)
                if match['size_vram'] != 0:
                    raise RuntimeError('speed design requires CPU backend')
            if self.samples[sample_start:] and min(s['available_memory_bytes'] for s in self.samples[sample_start:]) < 256 * 1024**2:
                raise RuntimeError('sampled low-memory guard')
            return response
        except Exception:
            self.unload()
            raise

    def execute(self):
        verify_preparation()
        save(self.out / 'run_manifest.json', {'prepared_plan_sha256': sha(PREP / 'manifest.json'),
             'inference_model': MODEL, 'stages': ['speed_6', 'diagnostic_32', 'confirmation_16', 'live_proposals_4'],
             'no_automatic_retries': True, 'service_start_or_restart': False, 'global_seconds_cap': 3300})
        before = preflight()
        save(self.out / 'preflight.json', before)
        state = before['state']
        try:
            if state != 'ready':
                self.record_status(state)
                return
            self.worker = threading.Thread(target=self.collect, daemon=True)
            self.worker.start()
            self.speed()
            self.unload()
            diag = read(ROOT / 'outputs/qwen35_label_order_probe_20261006/manifest.json')['request_order']
            # Preserve original frozen payload settings, including ctx2048/batch32.
            for req in diag:
                req = copy.deepcopy(req)
                req['payload']['keep_alive'] = 0
                self.request('diagnostic', req)
            summary = relation_summary(diag, self.rows['diagnostic'])
            save(self.out / 'diagnostic_summary.json', summary)
            checks = []
            for req in read(PREP / 'confirmation_manifest.json')['request_order']:
                response = self.request('confirmation', req)
                checks.append({'request_id': req['request_id'], **observable_arithmetic_check(response, req)})
                save(self.out / 'confirmation_checks.json', checks)
            self.proposals()
            state = 'complete'
            self.record_status(state)
        except Exception as exc:
            state = 'stopped_runtime_error'
            self.record_status(state, error=repr(exc))
        finally:
            try:
                self.unload()
            except Exception as exc:
                state = 'stopped_unload_error'
                self.record_status(state, error=repr(exc))
            self.stop.set()
            if self.worker:
                self.worker.join(timeout=2)
            save(self.out / 'resources.json', {'samples': self.samples, 'sampling_not_exact_peak': True})
            save(self.out / 'summary.json', {'state': state, 'attempts': self.attempts,
                 'completed_by_stage': {s: len(v) for s, v in self.rows.items()},
                 'speed_measured': len(self.rows['speed']) == 6, 'reserved_final_scenarios_opened': 0, 'new_ppo_updates': 0})
            (self.out / 'report.md').write_text(
                '# 27B实时提速与语义诊断执行记录\n\n'
                f'状态：{state}。真实请求尝试：{self.attempts}。\n\n'
                '若服务不可用或内存不足，后续32条诊断、16条确认与4条现场提案均未执行，不能报告新速度或新正确率。'
                '速度阶段区分加载、输入处理与生成，并保留冷/驻留标签；线程参数和前缀复用需要同轮对照。\n\n'
                '数学确认的raw_score与rule_checked_correct分开记录；规则替换不能视作模型理解改善。\n', encoding='utf-8')
            verify_preparation()
            save(self.out / 'completion_manifest.json', {'state': state, 'protected_sources_unchanged': True,
                 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in self.out.rglob('*') if p.is_file()}})
            print(json.dumps({'state': state, 'attempts': self.attempts, 'output': str(self.out)}, ensure_ascii=False), flush=True)

    def speed(self):
        base = read(ROOT / 'outputs/llm_goal_role_20261006/manifest.json')['request_order']
        chosen = {r['request_id']: r for r in base}
        ids = ('GR00_north_minimize_hazard', 'GR00_south_minimize_hazard', 'GR00_north_minimize_clear')
        threads_order = [4, 8]
        random.Random(2026100702).shuffle(threads_order)
        for threads in threads_order:
            for i, rid in enumerate(ids):
                req = copy.deepcopy(chosen[rid])
                req['request_id'] = f'SPEED_t{threads}_{rid}'
                req['candidate_order'] = ['north', 'south']
                req['private_reference']['label_to_physical'] = {'north': 0, 'south': 1}
                req['payload']['model'] = MODEL
                req['payload']['options'] = {'temperature': 0, 'seed': 2026100601, 'num_ctx': 1024,
                                             'num_predict': 128, 'num_batch': 32, 'num_thread': threads, 'num_gpu': 0}
                req['payload']['keep_alive'] = '60s'
                self.request('speed', req, resident=i > 0)
                self.rows['speed'][-1].update(cpu_threads=threads, load_kind='cold' if i == 0 else 'resident_shared_prefix')
            self.unload()
        save(self.out / 'speed_rows.json', self.rows['speed'])

    def proposals(self):
        from concurrent.futures import ThreadPoolExecutor
        from stable_baselines3 import PPO
        import torch
        from fire_evac.behavior import RuleBehavior
        from fire_evac.dynamic_conflicts import DynamicConflictScenario
        from fire_evac.timestamped_pipeline import TimestampedControlEnv, Proposal, ADVICE_PREFIX, CANDIDATE_SCHEMA, ppo_action
        torch.set_num_threads(1)
        plan = read(PREP / 'manifest.json')
        pure = PPO.load(plan['ppo_models']['601']['pure_ppo'], device='cpu')
        advised = PPO.load(plan['ppo_models']['601']['raw_llm_ppo'], device='cpu')
        scenes = read(PREP / 'manifest.json')['scenarios'][:4]
        for i, record in enumerate(scenes):
            scene = DynamicConflictScenario(**record)
            env = TimestampedControlEnv(RuleBehavior(), (scene,))
            env.reset(seed=1407200000 + i, options={'scenario': scene})
            observed = env.snapshot
            req = {'request_id': f'LIVE_{scene.scenario_id}', 'payload': {
                'model': MODEL, 'messages': [{'role': 'user', 'content': ADVICE_PREFIX + json.dumps(observed.public_data(), separators=(',', ':'))}],
                'format': CANDIDATE_SCHEMA, 'think': False, 'stream': False, 'keep_alive': 0,
                'options': {'temperature': 0, 'num_ctx': 1024, 'num_batch': 32, 'num_thread': 8, 'num_gpu': 0, 'num_predict': 192}}}
            begin = time.monotonic()
            control_trace = []
            ended = False
            next_control = 5.
            # HTTP runs in a worker while the simulation continues on wall time.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.request, 'live_proposals', req)
                while not future.done():
                    elapsed = time.monotonic() - begin
                    if elapsed >= next_control and not ended:
                        s = env.snapshot
                        act, raw = ppo_action(pure, env.policy_observation(), env.constraints.action_mask(s))
                        _, _, term, trunc, info = env.step(act)
                        ended = term or trunc
                        control_trace.append({'phase': 'pending_ppo', 'step': s.step, 'raw_action': raw,
                                              'action': act, 'actual_action': info['actual_joint_action'],
                                              'release_violation': not info['actual_release_within_limit'],
                                              'unavailable_entries': info['unavailable_corridor_entries']})
                        next_control += 5.
                    time.sleep(.05)
                response = future.result()
            latency = time.monotonic() - begin
            try:
                proposal = Proposal.parse(strict_json(response['message']['content']), observed, latency, req['request_id'], 'live_qwen35_27b')
                decision = env.constraints.validate_proposal(proposal, env.snapshot, max(latency, env.snapshot.observation_time_s))
                initial_decision = decision
                # Finish development episode, rechecking the same timestamped
                # advice after every fresh observation; no artificial refresh.
                while not ended:
                    s = env.snapshot
                    now = max(latency, s.observation_time_s)
                    decision = env.constraints.validate_proposal(proposal, s, now)
                    act, raw = ppo_action(advised, env.policy_observation(proposal, decision),
                                          env.constraints.action_mask(s), proposal, decision,
                                          elapsed_since_receipt=max(0, now-latency))
                    _, _, term, trunc, info = env.step(act)
                    ended = term or trunc
                    control_trace.append({'phase': 'response_ppo', 'step': s.step, 'gate_accepted': decision.accepted,
                                          'gate_reasons': decision.reasons, 'raw_action': raw,
                                          'action': act, 'actual_action': info['actual_joint_action'],
                                          'release_violation': not info['actual_release_within_limit'],
                                          'unavailable_entries': info['unavailable_corridor_entries']})
                save(self.out / 'live_checks' / (scene.scenario_id + '.json'), {'latency': latency,
                     'gate_accepted_on_return': initial_decision.accepted, 'gate_reasons_on_return': initial_decision.reasons,
                     'candidate_actions_remaining_on_return': initial_decision.valid_candidate_actions,
                     'asynchronous_wall_time_observation_updates': True, 'execution_trace': control_trace,
                     'final_metrics': {k: info[k] for k in ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate', 'step')}})
            except ValueError as exc:
                save(self.out / 'live_checks' / (scene.scenario_id + '.json'), {'invalid_response': repr(exc), 'latency': latency})
            env.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    Run().execute()
