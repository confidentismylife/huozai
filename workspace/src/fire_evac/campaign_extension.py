"""Small, audited helpers for the timed development experiment extension."""
from __future__ import annotations

import base64
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
import urllib.error
import urllib.request

import numpy as np
import gymnasium as gym

from .deepseek_advice import (
    AdviceResult, DeepSeekUsage, EvacuationPlan, MODEL_NAME, SYSTEM_PROMPT,
    parse_plan, rule_plan,
)
from .deepseek_plan_wrapper import DeepSeekPlanObservationWrapper


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.{time.monotonic_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers and antivirus scanners can briefly deny replace while
    # holding the destination open. Do not abort a long experiment for that.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.02 * (attempt+1), .1))


def admission_allowed(now: float, deadline: float, block_seconds: float,
                      report_seconds: float = 120) -> bool:
    return now + block_seconds + report_seconds <= deadline


def plan_phase(plan: EvacuationPlan, age: int) -> int:
    boundary = 0
    for index, phase in enumerate(plan.phases[:2]):
        boundary += phase.duration_steps
        if age < boundary:
            return index
    return 2


class ActivePhasePlanWrapper(DeepSeekPlanObservationWrapper):
    """Extension-only correction; leave the original experiment unchanged."""
    def _contradiction(self, observation: np.ndarray) -> bool:
        if self.current_plan is None:
            return False
        phase=self.current_plan.phases[plan_phase(self.current_plan,self.plan_age_steps)]
        if phase.route=='wait':
            return False
        target=0 if phase.route=='north' else 1
        return float(observation[8+target]) > float(observation[8+(1-target)]) + self.smoke_margin


class EpisodeSeedScheduleWrapper(gym.Wrapper):
    """Supply reproducible episode seeds to a domain that re-seeds on None.

    Explicit evaluation population seeds are passed through unchanged. Training
    reset(None) draws only from a local RNG initialized by the run's first seed.
    The legacy domain/reset implementation remains untouched for saved studies.
    """
    def __init__(self,env):
        super().__init__(env)
        self.schedule_rng=np.random.default_rng(0)
        self.episode_index=0
        self.last_reset_seed=None

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            selected=int(seed)
            self.schedule_rng=np.random.default_rng(selected)
            self.episode_index=0
        else:
            selected=int(self.schedule_rng.integers(0,2**32-1,dtype=np.uint32))
        self.last_reset_seed=selected
        self.episode_index+=1
        return self.env.reset(seed=selected,options=options)


def stale_plan_reason(plan: EvacuationPlan, age: int, state: dict,
                      smoke_margin: float = .01) -> str | None:
    if age >= sum(phase.duration_steps for phase in plan.phases):
        return 'expired'
    active=plan.phases[plan_phase(plan,age)]
    if active.route=='wait':
        return None
    target=0 if active.route=='north' else 1
    capacities=(int(state.get('north_exit_capacity',0)),
                int(state.get('south_exit_capacity',0)))
    if capacities[target]<=0:
        return 'blocked_exit'
    smoke=state.get('smoke',[0.0]*5)
    if float(smoke[3+target])>float(smoke[3+(1-target)])+smoke_margin:
        return 'observed_smoke_contradiction'
    return None


def execution_diagnostic(requested: int, info: dict, target_route: int,
                         target_release: float, full_capacity: int) -> dict:
    waiting = int(requested) == 8
    requested_route = 2 if waiting else int(requested) // 4
    actual = int(info['actual_joint_action'])
    executed_route = 2 if waiting else int(info['actual_sign_action'])
    actual_release = float(info['actual_release_fraction'])
    target_actual_release = round(full_capacity * target_release) / max(1, full_capacity)
    return {
        'requested_joint_action': int(requested), 'actual_joint_action': actual,
        'requested_route': requested_route, 'executed_route': executed_route,
        'actual_release_fraction': actual_release,
        'joint_action_compliance': int(actual == int(requested)),
        'requested_plan_route_match': int(requested_route == target_route),
        'executed_plan_route_match': int(executed_route == target_route),
        'executed_plan_flow_match': (
            float(abs(actual_release - target_actual_release) < 1e-6)
            if not waiting and target_route != 2 else math.nan
        ),
    }


def saved_credential() -> str:
    """Decrypt the authorized user's DPAPI credential without shell arguments."""
    path = Path(os.environ['LOCALAPPDATA']) / 'fire_evac' / 'deepseek_key.dpapi'
    protected = base64.b64decode(path.read_text(encoding='ascii'))
    class Blob(ctypes.Structure):
        _fields_ = [('size', ctypes.c_ulong), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    def blob(value):
        buffer = ctypes.create_string_buffer(value)
        return Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer
    source, source_buffer = blob(protected)
    entropy, entropy_buffer = blob(b'fire-evac-deepseek-v1')
    plain = Blob()
    success = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0,
        ctypes.byref(plain))
    if not success:
        raise RuntimeError('The saved Windows credential could not be decrypted')
    try:
        value = ctypes.string_at(plain.data, plain.size).decode('utf-8')
        if not value.startswith('sk-'):
            raise RuntimeError('Invalid saved credential format')
        return value
    finally:
        ctypes.memset(plain.data, 0, plain.size)
        ctypes.windll.kernel32.LocalFree(plain.data)


UNIT_CONTEXT = (
    ' Units: each control step is 5 seconds. occupancy entries are fractions of '
    'the initial 240 people, ordered roomA, roomB, junction, north, south. smoke '
    'entries follow the same order and are simulated per-step hazard increments; '
    'they are not calibrated toxic-gas concentrations. Capacities are people per '
    'control step. Release_fraction scales room-to-junction capacity. Current '
    'observations override a contradictory forecast. There is no access to future '
    'outcomes. Do not assume you control evacuees directly.'
)


class BudgetedPlanner:
    """Exact-query cache, no automatic retries, persisted pre-request reservation.

    The upper rate (2.5/9 CNY per million input/output tokens) exceeds the official
    peak Flash rate checked on 2026-10-22. A timeout conservatively retains the
    entire request reservation because server-side completion is unknown.
    """
    def __init__(self, output: Path, max_cny: float = 2.0, max_requests: int = 240,
                 transport: Callable | None = None, max_tokens: int = 512):
        self.output = Path(output)
        self.cache = self.output / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.path = self.output / 'budget.json'
        self.max_cny = float(max_cny)
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self.transport = transport
        self.records = self.output / 'requests.jsonl'
        self._key: str | None = None
        self.last_latency_seconds=0.0
        if not self.path.exists():
            self._save({'attempted_requests':0, 'successful_requests':0,
                        'input_tokens':0, 'output_tokens':0, 'cache_hit_tokens':0,
                        'actual_cost_upper_cny':0.0, 'reserved_cny':0.0})

    def ledger(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _save(self, ledger):
        write_json(self.path, {**ledger, 'maximum_cny':self.max_cny,
            'maximum_requests':self.max_requests, 'model':MODEL_NAME,
            'input_cny_per_million_upper':2.5,
            'output_cny_per_million_upper':9.0,
            'credential_logged':False})

    def messages(self, state):
        return [{'role':'system','content':SYSTEM_PROMPT + UNIT_CONTEXT},
                {'role':'user','content':'observable_state=' + json.dumps(
                    state, ensure_ascii=False, sort_keys=True, separators=(',',':'))}]

    def _call(self, messages, maximum):
        if self._key is None:
            self._key = saved_credential()
        body = json.dumps({'model':MODEL_NAME, 'messages':messages,
            'temperature':0, 'max_tokens':maximum,
            'thinking':{'type':'disabled'}, 'response_format':{'type':'json_object'},
            'stream':False}, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request('https://api.deepseek.com/chat/completions',
            data=body, headers={'Authorization':'Bearer '+self._key,
                                'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not include a response body, request headers, or the key in errors.
            raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None
        return data['choices'][0]['message']['content'], data['usage']

    def __call__(self, state):
        messages = self.messages(state)
        canonical = json.dumps({'model':MODEL_NAME,'prompt':'online-units-v1',
                                'messages':messages}, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        cached = self.cache / (key+'.json')
        if cached.exists():
            payload = json.loads(cached.read_text(encoding='utf-8'))
            self.last_latency_seconds=float(payload.get('latency_seconds',0.0))
            return AdviceResult(parse_plan(payload['plan']), 'cache', DeepSeekUsage())
        ledger = self.ledger()
        # UTF-8 bytes give a conservative input-token bound for this compact JSON.
        max_input = sum(len(m['content'].encode('utf-8')) for m in messages) + 256
        reservation = (max_input*2.5 + self.max_tokens*9.0) / 1e6
        if (ledger['attempted_requests'] >= self.max_requests or
            ledger['actual_cost_upper_cny']+ledger['reserved_cny']+reservation > self.max_cny):
            return AdviceResult(rule_plan(state), 'fallback_budget', DeepSeekUsage())
        ledger['attempted_requests'] += 1
        ledger['reserved_cny'] += reservation
        self._save(ledger)
        started = time.perf_counter()
        raw = None
        usage = DeepSeekUsage()
        status = 'fallback_transport'
        error_type = None
        plan = rule_plan(state)
        try:
            raw, reported = (self.transport or self._call)(messages, self.max_tokens)
            usage = DeepSeekUsage(int(reported.get('prompt_tokens',0)),
                int(reported.get('completion_tokens',0)),
                int(reported.get('prompt_cache_hit_tokens',0)))
            cost = (usage.input_tokens*2.5 + usage.output_tokens*9.0) / 1e6
            ledger['reserved_cny'] = max(0.0, ledger['reserved_cny']-reservation)
            ledger['actual_cost_upper_cny'] += cost
            ledger['input_tokens'] += usage.input_tokens
            ledger['output_tokens'] += usage.output_tokens
            ledger['cache_hit_tokens'] += usage.cache_hit_tokens
            try:
                plan = parse_plan(raw)
                status = 'api'
                ledger['successful_requests'] += 1
                write_json(cached, {'plan':asdict(plan),'usage':asdict(usage),
                    'state':state,'model':MODEL_NAME,'latency_seconds':time.perf_counter()-started})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                status = 'fallback_schema'
                error_type = type(exc).__name__
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            self.last_latency_seconds=time.perf_counter()-started
            self._save(ledger)
            record = {'utc':datetime.now(timezone.utc).isoformat(),'cache_key':key,
                'scenario_id':state.get('scenario_id'), 'event_type':state.get('event_type'),
                'source':status,'error_type':error_type,'state':state,
                'plan':asdict(plan),'usage':asdict(usage),
                'latency_seconds':self.last_latency_seconds}
            if raw is not None:
                record['raw_response'] = raw
            with self.records.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        return AdviceResult(plan, status, usage)
