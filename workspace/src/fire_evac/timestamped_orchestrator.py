"""One-checkpoint controller, including invalid/late advice fallback.

The same PPO weights run before and after the response: only the explicitly
accepted advice features and bounded soft prior change. No trained model swap
can masquerade as an advice gain.
"""
from __future__ import annotations
import time
from .timestamped_pipeline import AsyncAdviceClient, ppo_action


def execute_episode(env, model, transport, *, control_interval_seconds=5., action_function=ppo_action,
                    clock=time.monotonic, idle=lambda: time.sleep(.02)):
    """HTTP in background; update sensors on wall time until first response.

    After response the development simulation finishes in accelerated time.
    Its simulation clock continues to age advice, including after HTTP failure.
    All failures/expiry clear advice; PPO remains subject to the common mask.
    """
    initial = env.snapshot
    client = AsyncAdviceClient(transport)
    started = clock()
    client.submit(initial, now_monotonic=started)
    proposal, decision = None, None
    state = 'pending'
    trace = []
    next_control = 0.
    ended = False
    info = None
    received = False
    try:
        while not ended:
            now = clock()
            pending = client.pending is not None
            if pending:
                p, gate, state = client.poll(env.snapshot, env.constraints, now_monotonic=now)
                if p is not None:
                    proposal, decision = p, gate
                if client.pending is None:
                    received = True
            elapsed = now - started
            if client.pending is not None and elapsed < next_control:
                idle()
                continue
            s = env.snapshot
            decision_time = max(s.observation_time_s, initial.observation_time_s + elapsed)
            if proposal is not None:
                decision = env.constraints.validate_proposal(proposal, s, decision_time)
            observation = env.policy_observation(proposal, decision)
            action, raw = action_function(model, observation, env.constraints.action_mask(s), proposal, decision,
                                          elapsed_since_receipt=0 if proposal is None else max(0., decision_time-proposal.received_time_s))
            _, _, term, trunc, info = env.step(action)
            trace.append({'step': s.step, 'observation_time_s': s.observation_time_s,
                          'wall_elapsed_seconds': elapsed, 'phase': 'wall_time_pending' if client.pending is not None else 'accelerated_after_response',
                          'advice_state': state, 'gate_accepted': bool(decision and decision.accepted),
                          'gate_reasons': () if decision is None else decision.reasons,
                          'raw_action': raw, 'requested_action': action, 'actual_action': info['actual_joint_action'],
                          'room_release_count': info['actual_room_release_count'], 'room_limit': info['pretransition_room_limit'],
                          'unavailable_entries': info['unavailable_corridor_entries'],
                          'release_violation': not info['actual_release_within_limit']})
            next_control += control_interval_seconds
            ended = term or trunc
    finally:
        client.close()
    return {'response_collected_before_episode_end': received, 'advice_final_state': state, 'trace': trace,
            'final_metrics': {k: info[k] for k in ('step', 'mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate')},
            'one_checkpoint_throughout': True, 'simulation_after_response_accelerated': True}

