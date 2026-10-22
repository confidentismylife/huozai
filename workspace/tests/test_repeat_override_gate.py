import copy
from fire_evac.repeat_override_gate import public_signature, fit_consensus, consensus_decision


def snapshot():
    return dict(occupancy_fraction=[.5,.5,0,0,0],smoke_increment_per_step=[.002]*5,
                exit_capacity_persons_per_step=[3,9],step=3,actual_sign=0,sign_hold_remaining_steps=0,
                forecast_source='south',forecast_onset_step=5,episode_id='ignored')


def test_signature_cannot_use_identity_hidden_fields_or_terminal_outcome():
    a=snapshot(); b=copy.deepcopy(a)
    b.update(episode_id='different',population_seed=999,training_seed=888,behavior='different',
             hidden_people=[1,2],terminal_fed=123,captured_at_utc='different')
    assert public_signature(a,None,1,5,0,0)==public_signature(b,None,1,5,0,0)


def test_previous_public_counts_and_attempt_history_really_change_signature():
    a=snapshot(); b=copy.deepcopy(a); b['occupancy_fraction']=[.45,.45,.1,0,0]
    assert public_signature(a,None,1,5,0,0)!=public_signature(a,b,1,5,0,0)
    assert public_signature(a,None,1,5,0,0)!=public_signature(a,None,1,5,1,1)


def test_consensus_abstains_on_identical_public_state_with_conflicting_outcomes():
    model=fit_consensus([dict(offline_identity='a',signature='x',paired_outcomes=[dict(incap_delta=0,fed_delta=-.01)]),
                         dict(offline_identity='b',signature='x',paired_outcomes=[dict(incap_delta=0,fed_delta=.01)])])
    assert not consensus_decision(model,'x')[0]
    assert not consensus_decision(model,'unseen')[0]


def test_positive_mean_cannot_hide_harmful_future_or_small_incapacitation_increase():
    model=fit_consensus([dict(offline_identity='a',signature='x',paired_outcomes=[dict(incap_delta=0,fed_delta=-.1),dict(incap_delta=0,fed_delta=.01)]),
                         dict(offline_identity='b',signature='y',paired_outcomes=[dict(incap_delta=.0001,fed_delta=-.01)])])
    assert not consensus_decision(model,'x')[0] and not consensus_decision(model,'y')[0]


def test_deduplicate_same_state_and_accept_consistently_beneficial_support():
    a=dict(offline_identity='same',signature='x',paired_outcomes=[dict(incap_delta=-.01,fed_delta=.02)])
    model=fit_consensus([a,copy.deepcopy(a)])
    assert model['unique_archived_states']==1 and model['cells']['x']['states']==1
    assert consensus_decision(model,'x')[0]
