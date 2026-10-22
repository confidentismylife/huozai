"""Development-only asynchronous advice and common actuator constraints.

No oracle future or individual latent attributes enter the advice interface.
These operational constraints do not certify low exposure or safe evacuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from typing import Callable

import gymnasium as gym
import numpy as np

from .confirmatory import ConfirmatoryJointControlWrapper, decode_absolute_action, encode_absolute_action
from .dynamic_conflicts import DynamicConflictFireEvacuationEnv, GymDynamicConflictEnv
from .dynamic_llm_rl import DynamicLLMAdvice


@dataclass(frozen=True)
class Snapshot:
    episode_id: str
    step: int
    observation_time_s: float
    captured_at_utc: str
    occupancy: tuple[float, ...]
    smoke: tuple[float, ...]
    capacities: tuple[int, int]
    closed: tuple[bool, bool]
    forecast_source: int
    forecast_growth: float
    forecast_onset_step: int
    forecast_reopen_steps: float
    sign: int
    sign_hold_remaining: int
    full_release_capacity: int
    population: int = 240

    @classmethod
    def from_sensor(cls, observation, info, config, episode_id, hold_remaining=0):
        """Read only public sensor/forecast fields; ignore info['scenario']."""
        x = np.asarray(observation, dtype=float)
        if x.shape != (34,) or not np.isfinite(x).all():
            raise ValueError("finite 34-input observation required")
        capacities = tuple(max(0, int(round(v * config.branch_to_exit_capacity))) for v in x[14:16])
        return cls(str(episode_id), int(info['step']), float(info['step']) * config.dt_seconds,
                   datetime.now(timezone.utc).isoformat(), tuple(x[:5]), tuple(x[5:10]),
                   capacities, tuple(bool(x[16 + b] >= .5 or capacities[b] == 0) for b in range(2)),
                   int(x[20] > x[19]), float(x[21]) * .05, int(round(float(x[22]) * 20)),
                   float(x[18]) * 20, int(info['actual_sign_action']), int(hold_remaining),
                   config.start_to_junction_capacity, config.population)

    def public_data(self):
        return {"episode_id": self.episode_id, "observation_time_s": self.observation_time_s,
                "step": self.step, "captured_at_utc": self.captured_at_utc,
                "occupancy_fraction": self.occupancy, "smoke_increment_per_step": self.smoke,
                "exit_capacity_persons_per_step": self.capacities, "exit_closed": self.closed,
                "forecast_source": ('north', 'south')[self.forecast_source],
                "forecast_growth": self.forecast_growth, "forecast_onset_step": self.forecast_onset_step,
                "forecast_reopen_steps": self.forecast_reopen_steps,
                "actual_sign": self.sign, "sign_hold_remaining_steps": self.sign_hold_remaining}


CANDIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "observation_time_s": {"type": "number", "minimum": 0},
        "preferred_exit": {"type": "string", "enum": ["north", "south"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "candidate_actions": {"type": "array", "minItems": 1, "maxItems": 3,
                              "items": {"type": "integer", "minimum": 0, "maximum": 8}},
        "wait_steps": {"type": "integer", "minimum": 0, "maximum": 2},
        "north_share": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["observation_time_s", "preferred_exit", "confidence", "candidate_actions", "wait_steps", "north_share"],
}
ADVICE_PREFIX = (
    "You advise a research evacuation simulator. Use only timestamped current sensors and uncertain forecasts. "
    "Minimize smoke exposure and congestion. A forecast source names potential smoke, not a destination. "
    "Never assume a lower-level controller will fix a bad proposal. The response will be checked against newer sensors. "
    "Propose up to three candidate actions: 0..3=north with room release 25/50/75/100%; "
    "4..7=south with the same levels; 8=wait (zero room release). wait_steps at most 2. "
    "north_share is a desired split over multiple control steps, not an instantaneous two-way action. "
    "Copy observation_time_s. Confidence is uncalibrated. Return only the requested JSON. Data: "
)


@dataclass(frozen=True)
class Proposal:
    request_id: str
    episode_id: str
    observation_time_s: float
    received_time_s: float
    preferred_branch: int
    confidence: float
    candidate_actions: tuple[int, ...]
    wait_steps: int
    north_share: float
    provenance: str

    @classmethod
    def parse(cls, payload, snapshot, received_time_s, request_id, provenance):
        if set(payload) != set(CANDIDATE_SCHEMA['required']):
            raise ValueError("wrong proposal fields")
        def numeric(v):
            return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
        for name in ('observation_time_s', 'confidence', 'north_share'):
            if not numeric(payload[name]):
                raise ValueError("non-finite or non-numeric proposal")
        if payload['observation_time_s'] != snapshot.observation_time_s:
            raise ValueError("response timestamp does not match request")
        if payload['preferred_exit'] not in ('north', 'south'):
            raise ValueError("unknown exit")
        if not 0 <= payload['confidence'] <= 1 or not 0 <= payload['north_share'] <= 1:
            raise ValueError("invalid fraction")
        actions = payload['candidate_actions']
        if not isinstance(actions, list) or not 1 <= len(actions) <= 3 or len(set(actions)) != len(actions):
            raise ValueError("one to three unique candidate actions required")
        if any(type(a) is not int or a not in range(9) for a in actions):
            raise ValueError("invalid candidate action")
        wait = payload['wait_steps']
        if type(wait) is not int or wait not in range(3):
            raise ValueError("invalid wait duration")
        if (wait > 0) != (8 in actions):
            raise ValueError("wait duration and candidate disagree")
        return cls(str(request_id), snapshot.episode_id, snapshot.observation_time_s, float(received_time_s),
                   int(payload['preferred_exit'] == 'south'), float(payload['confidence']),
                   tuple(actions), wait, float(payload['north_share']), provenance)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    valid_candidate_actions: tuple[int, ...]
    age_seconds: float


class CommonActionConstraints:
    """Observable action mask, followed by checks of actual actuator effects."""
    def __init__(self, smoke_ceiling=.15, queue_limit=45, smoke_margin=.01):
        self.smoke_ceiling, self.queue_limit, self.smoke_margin = smoke_ceiling, queue_limit, smoke_margin

    def branch_allowed(self, s, branch):
        return not s.closed[branch] and s.smoke[3 + branch] < self.smoke_ceiling

    def room_limit(self, s):
        eligible = [b for b in range(2) if self.branch_allowed(s, b)]
        if not eligible:
            return 0
        # Rate budget sums usable downstream capacities. Queue headroom is a
        # conservative observed junction bound, without anticipating departures.
        junction_count = int(round(s.occupancy[2] * s.population))
        return max(0, min(s.full_release_capacity, sum(s.capacities[b] for b in eligible),
                          self.queue_limit - junction_count))

    def action_mask(self, s):
        mask = np.zeros(9, dtype=bool)
        mask[8] = True
        limit = self.room_limit(s)
        for a in range(8):
            branch, fraction = decode_absolute_action(a)
            sign_ok = not s.sign_hold_remaining or branch == s.sign
            mask[a] = self.branch_allowed(s, branch) and sign_ok and round(s.full_release_capacity * fraction) <= limit
        return mask

    def filter_action(self, requested, s):
        if type(requested) not in (int, np.int64, np.int32) or requested not in range(9):
            raise ValueError("joint action must be an integer in [0, 8]")
        mask = self.action_mask(s)
        if mask[requested]:
            return int(requested)
        if requested != 8:
            branch, fraction = decode_absolute_action(requested)
            same = [a for a in range(8) if a // 4 == branch and mask[a] and decode_absolute_action(a)[1] <= fraction]
            if same:
                return max(same)
        candidates = np.flatnonzero(mask[:8])
        if len(candidates):
            costs = [3 * s.smoke[3 + a // 4] + s.occupancy[3 + a // 4] - .001 * (a % 4) for a in candidates]
            return int(candidates[int(np.argmin(costs))])
        return 8

    def validate_proposal(self, p, current, now_s, ttl_s=15., timeout_s=60.):
        reasons = []
        age = now_s - p.observation_time_s
        if p.episode_id != current.episode_id:
            reasons.append('episode_mismatch')
        if p.observation_time_s > now_s or p.received_time_s > now_s:
            reasons.append('future_timestamp')
        if p.received_time_s - p.observation_time_s > timeout_s:
            reasons.append('timeout')
        if age > ttl_s:
            reasons.append('stale')
        if p.confidence < .55:
            reasons.append('low_uncalibrated_confidence')
        if not self.branch_allowed(current, p.preferred_branch):
            reasons.append('preferred_exit_unavailable')
        other = 1 - p.preferred_branch
        if self.branch_allowed(current, other) and current.smoke[3 + p.preferred_branch] > current.smoke[3 + other] + self.smoke_margin:
            reasons.append('live_smoke_contradiction')
        mask = self.action_mask(current)
        valid = tuple(a for a in p.candidate_actions if mask[a])
        if not valid:
            reasons.append('no_admissible_candidate')
        return GateDecision(not reasons, tuple(reasons), valid, age)

    def rule_action(self, s):
        mask = self.action_mask(s)
        candidates = np.flatnonzero(mask[:8])
        if not len(candidates):
            return 8
        # Same raw forecast is available to PPO and the advice service.
        costs = []
        for a in candidates:
            b = a // 4
            forecast_penalty = .05 if b == s.forecast_source and s.step < s.forecast_onset_step else 0.
            costs.append(3 * s.smoke[3+b] + s.occupancy[3+b] + forecast_penalty - .001 * (a % 4))
        return int(candidates[int(np.argmin(costs))])


class OperationalDomainEnv(DynamicConflictFireEvacuationEnv):
    """Prevent entry to unavailable corridors; clip actual upstream release."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operational_branches = (True, True)
        self.operational_room_limit = self.config.start_to_junction_capacity

    def _junction_to_branch_capacities(self):
        values = super()._junction_to_branch_capacities()
        current = self._branch_to_exit_capacities()
        for b in range(2):
            values[b] = min(values[b], current[b])
            if not self.operational_branches[b] or current[b] <= 0:
                values[b] = 0
        return values

    def _start_to_junction_capacity(self):
        return min(super()._start_to_junction_capacity(), self.operational_room_limit)


class OperationalGymEnv(GymDynamicConflictEnv):
    def __init__(self, behavior_model, scenarios, config=None):
        super().__init__(behavior_model, scenarios, config)
        self.domain_env = OperationalDomainEnv(behavior_model, scenarios, config=config)


class TimestampedControlEnv(ConfirmatoryJointControlWrapper):
    """34-input checkpoint-compatible adapter; same constraints on every arm."""
    def __init__(self, behavior_model, scenarios, config=None, constraints=None):
        super().__init__(OperationalGymEnv(behavior_model, scenarios, config), 'none')
        self.constraints = constraints or CommonActionConstraints()
        self.snapshot = None
        self._hold_remaining = 0
        self._episode_id = ''
        self.latest_observation = None

    def reset(self, *, seed=None, options=None):
        x, info = super().reset(seed=seed, options=options)
        self._hold_remaining = 0
        self._episode_id = f"{self.current_scenario.scenario_id}:{seed}"
        self.latest_observation = x.copy()
        self.snapshot = Snapshot.from_sensor(x, info, self.unwrapped.domain_env.config, self._episode_id)
        return x, info

    def policy_observation(self, proposal=None, decision=None):
        x = self.latest_observation.copy()
        x[28:33] = 0
        if proposal is not None and decision is not None and decision.accepted:
            target = proposal.preferred_branch
            x[28 + target] = 1
            x[30], x[31] = proposal.confidence, 1
            x[32] = float(self.snapshot.smoke[3+target] > self.snapshot.smoke[4-target] + .01)
        return x

    def step(self, action):
        old = self.snapshot
        requested = int(action)
        applied = self.constraints.filter_action(requested, old)
        domain = self.unwrapped.domain_env
        domain.operational_branches = tuple(self.constraints.branch_allowed(old, b) for b in range(2))
        domain.operational_room_limit = self.constraints.room_limit(old)
        x, reward, term, trunc, info = super().step(applied)
        actual = int(info['actual_sign_action'])
        self._hold_remaining = domain.config.min_sign_hold_steps if actual != old.sign else max(0, self._hold_remaining - 1)
        actual_room_count = int(info['room_release_count'])
        actual_room_capacity = int(info['room_release_capacity'])
        illegal_entry = sum(int(domain.last_branch_flow[b]) for b in range(2) if not domain.operational_branches[b])
        # Check the pretransition conditions: info capacities describe the next step.
        actual_direction_bad = applied != 8 and (actual not in (0, 1) or not self.constraints.branch_allowed(old, actual))
        enriched = dict(info, pipeline_requested_action=requested, constraint_applied_action=applied,
                        constraint_intervened=applied != requested,
                        pretransition_room_limit=domain.operational_room_limit,
                        actual_room_release_count=actual_room_count,
                        actual_release_within_limit=actual_room_count <= domain.operational_room_limit and actual_room_capacity <= domain.operational_room_limit,
                        unavailable_corridor_entries=illegal_entry,
                        actual_sign_unavailable=actual in (0, 1) and not domain.operational_branches[actual],
                        actual_direction_violation=bool(actual_direction_bad),
                        requested_actual_match=info['actual_joint_action'] == applied)
        self.latest_observation = x.copy()
        self.snapshot = Snapshot.from_sensor(x, enriched, domain.config, self._episode_id, self._hold_remaining)
        return x, reward, term, trunc, enriched


def ppo_action(model, observation, mask, proposal=None, decision=None, prior_strength=.25, elapsed_since_receipt=0.):
    """All actions constrained; a bounded soft candidate prior permits override.

    Candidate ratios mean preferences across successive signs. This adapter
    cannot command simultaneous flows in the existing nine-action simulator.
    """
    import torch
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observation)
        probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
    scores = np.log(np.maximum(probs, 1e-30))
    raw = int(np.argmax(scores))
    if proposal is not None and decision is not None and decision.accepted:
        for a in decision.valid_candidate_actions:
            if a == 8:
                if elapsed_since_receipt < proposal.wait_steps * 5.:
                    scores[a] += prior_strength * proposal.confidence
            else:
                share = proposal.north_share if a < 4 else 1 - proposal.north_share
                scores[a] += prior_strength * proposal.confidence * share
    scores = np.where(mask, scores, -np.inf)
    return int(np.argmax(scores)), raw


class AsyncAdviceClient:
    """One in-flight request, clock-aware polling, no simulator blocking.

    A late HTTP response is collected for audit and cannot replace current
    advice. An HTTP timeout is a read deadline, not server cancellation.
    """
    def __init__(self, transport: Callable, timeout_seconds=60., ttl_seconds=15.):
        self.transport, self.timeout_seconds, self.ttl_seconds = transport, timeout_seconds, ttl_seconds
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.counter = 0

    def submit(self, snapshot, now_monotonic=None):
        if self.pending is not None:
            return False
        self.counter += 1
        started = time.monotonic() if now_monotonic is None else now_monotonic
        future = self.executor.submit(self.transport, snapshot)
        self.pending = (future, snapshot, started, f"live-{self.counter}")
        return True

    def poll(self, current, constraints, now_monotonic=None):
        if self.pending is None:
            return None, None, 'idle'
        f, snapshot, started, rid = self.pending
        elapsed = (time.monotonic() if now_monotonic is None else now_monotonic) - started
        # Wall time continues even if caller has not advanced simulation time.
        now = max(current.observation_time_s, snapshot.observation_time_s + elapsed)
        if not f.done():
            return None, None, 'timeout_pending_audit' if elapsed > self.timeout_seconds else 'pending'
        self.pending = None
        try:
            p = Proposal.parse(f.result(), snapshot, now, rid, 'live_ollama')
        except Exception as exc:
            return None, None, f'invalid_or_transport_error:{type(exc).__name__}'
        decision = constraints.validate_proposal(p, current, now, self.ttl_seconds, self.timeout_seconds)
        return p, decision, 'accepted' if decision.accepted else 'rejected'

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def ollama_transport(model='qwen35-27b-text-q4:latest', options=None, base_url='http://127.0.0.1:11434'):
    import requests
    def call(snapshot):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': ADVICE_PREFIX + json.dumps(snapshot.public_data(), separators=(',', ':'))}],
                   'format': CANDIDATE_SCHEMA, 'think': False, 'stream': False, 'keep_alive': 0,
                   'options': options or {'num_ctx': 1024, 'num_batch': 32, 'num_thread': 8, 'num_gpu': 0, 'num_predict': 192, 'temperature': 0}}
        r = requests.post(base_url + '/api/chat', json=payload, timeout=(5, 90))
        r.raise_for_status()
        response = r.json()
        if response.get('done_reason') == 'length' or not response.get('done'):
            raise ValueError('incomplete model response')
        return json.loads(response['message']['content'])
    return call
