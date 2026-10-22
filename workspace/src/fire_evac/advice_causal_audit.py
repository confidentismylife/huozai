"""Offline advice attribution; selectors use only public observations.

Hidden simulator state is copied for paired terminal evaluation only. No
oracle label is passed back to the controller.
"""
import copy
import numpy as np
from .timestamped_pipeline import ppo_action
from .counterfactual_direction_audit import PairedFutureBehavior


def active_advice(env, proposal):
    if proposal is None:
        return None, None, 'invalid_or_missing'
    s = env.snapshot
    if s.observation_time_s < proposal.received_time_s:
        return None, None, 'pending'
    decision = env.constraints.validate_proposal(proposal, s, s.observation_time_s)
    return proposal, decision, 'accepted' if decision.accepted else 'rejected'


def action_variants(env, model, proposal, decision):
    s = env.snapshot
    mask = env.constraints.action_mask(s)
    plain, raw_plain = ppo_action(model, env.policy_observation(), mask)
    x = env.policy_observation(proposal, decision)
    elapsed = 0. if proposal is None else s.observation_time_s - proposal.received_time_s
    features, raw_features = ppo_action(model, x, mask, proposal, decision,
                                       prior_strength=0., elapsed_since_receipt=elapsed)
    actual, raw_actual = ppo_action(model, x, mask, proposal, decision,
                                   elapsed_since_receipt=elapsed)
    if decision is None or not decision.accepted:
        assert actual == features == plain
    return dict(no_advice=int(plain), features_only=int(features), with_advice=int(actual),
                raw_no_advice=int(raw_plain), raw_features=int(raw_features), raw_with_advice=int(raw_actual))


def reseed_continuation(env, seed):
    """Unwrap a previously paired behavior before assigning a fresh future.

    Nesting PairedFutureBehavior would cause the inner wrapper to ignore the
    new uniforms. Prior frozen studies wrapped raw behaviors only; this
    helper handles the already-wrapped replay states used in this audit.
    """
    clone = copy.deepcopy(env)
    domain = clone.unwrapped.domain_env
    base = domain.behavior_model
    while isinstance(base, PairedFutureBehavior):
        base = base.base
    table = np.random.default_rng(seed).random((domain.config.max_steps + 1, domain.config.population))
    domain.behavior_model = PairedFutureBehavior(base, table)
    return clone


def outcome_class(incapacitation_difference, fed_difference, rate_tolerance=.001, fed_tolerance=.001):
    """Prioritize expected incapacitation; thresholds frozen for development."""
    if incapacitation_difference <= -rate_tolerance:
        return 'beneficial'
    if incapacitation_difference >= rate_tolerance:
        return 'harmful'
    if fed_difference <= -fed_tolerance:
        return 'beneficial'
    if fed_difference >= fed_tolerance:
        return 'harmful'
    return 'small_or_neutral'


def ranked_outcomes(candidate_means):
    """Offline diagnostic ordering, not an online action selector."""
    return sorted(candidate_means, key=lambda a: (round(candidate_means[a]['incapacitation_rate'], 12),
                 candidate_means[a]['mean_cumulative_fed'], candidate_means[a]['cumulative_congestion'], int(a)))


def actuator_signature(info):
    return tuple(int(info[k]) for k in ('actual_joint_action', 'actual_sign_action',
                                      'room_release_capacity', 'room_release_count'))
