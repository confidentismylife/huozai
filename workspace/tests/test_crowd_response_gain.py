import copy
import numpy as np
import pytest
from fire_evac.crowd_response_gain import context, observe_completed, features, fit_ridge, predict, calibrate, accepts


def public(step=0):
    return dict(step=step, occupancy_fraction=[.5,.5,0,0,0], smoke_increment_per_step=[.002]*5,
                exit_capacity_persons_per_step=[6,6], exit_closed=[False,False], forecast_source='south',
                forecast_growth=.03, forecast_onset_step=5, forecast_reopen_steps=0.,actual_sign=2,sign_hold_remaining_steps=0)


def test_features_ignore_identity_hidden_attributes_and_outcomes():
    s=public(); t=copy.deepcopy(s); t.update(episode_id='fake',behavior='rule',people=[1],fed=999)
    obs=[0.]*34; other=obs.copy();other[28:33]=[999.]*5
    for v in ('current','public_history','response'):
        assert features(obs,s,1,5,context(),v)==features(other,t,1,5,context(),v)


def test_history_must_precede_current_decision_and_is_allowlisted():
    c=context();s=public();sensors=dict(actual_sign=0,actual_room_capacity=10,actual_room_count=10,
        zone_counts_after=[115,115,10,0,0],branch_entries=[0,0],hidden_people=[999],zone_new_incapacitations=[1]*5)
    observe_completed(c,s,1,dict(eligible=True,base_action=5),sensors)
    assert 'hidden_people' not in c['history'][0] and 'zone_new_incapacitations' not in c['history'][0]
    with pytest.raises(AssertionError): features([0.]*34,s,1,5,c,'response')
    s=public(1);s['occupancy_fraction']=[115/240,115/240,10/240,0,0]
    assert np.isfinite(features([0.]*34,s,1,5,c,'response')).all()


def test_exit_inference_is_disabled_when_public_smoke_cannot_exclude_incapacitation():
    from fire_evac.crowd_response_gain import feature_names
    c=context();s=public();s['smoke_increment_per_step']=[1.1]*5
    observe_completed(c,s,1,dict(eligible=False,base_action=1),dict(actual_sign=0,actual_room_capacity=10,actual_room_count=10,zone_counts_after=[115,115,10,0,0],branch_entries=[0,0]))
    out=dict(zip(feature_names('response'),features([0.]*34,public(1),1,5,c,'response')))
    assert out['last3_exit_available']==0 and out['last3_north_exits']==0


def test_gain_gate_cannot_hide_incapacitation_increase_or_adverse_margin():
    assert accepts([0,-.01]) and accepts([-.01,.5])
    assert not accepts([.0001,-.01]) and not accepts([0,-.01],[.001,.001])
    assert not accepts([0,0])


def test_ridge_handles_constant_columns_and_family_calibration():
    x=features([0.]*34,public(),1,5,context(),'current')
    rows=[dict(family=str(i),features={'current':x},incap_delta=0,fed_delta=-.01) for i in range(4)]
    model=fit_ridge(rows,'current');assert np.allclose(predict(model,x),[0,-.01])
    band=calibrate(model,rows);assert band['calibration_families']==4
    assert np.allclose(band['upper_adverse_residual'],[0,0])
