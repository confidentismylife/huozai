import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from auto_evacuation_core import labels,physics_features,fit_kernel,predict_kernel,choose,calibrated_margin,TRACE_FIELDS


def test_world_pooling_and_two_policy_support():
    risks=np.zeros((2,2,2,6,11));risks[:,:,0,:,0]=.5
    risks[0,:,1,:,0]=np.array([.1,.9])[:,None];risks[1,:,1,:,0]=np.array([.9,.1])[:,None]
    l,y=labels(risks);np.testing.assert_allclose(l,.5);np.testing.assert_array_equal(y,np.zeros((2,2)))
    with pytest.raises(ValueError):labels(np.zeros((2,1,5,6,11)))


def test_source_and_advice_do_not_enter_physics_features():
    x=np.zeros(49);x[23:25]=[.625,1.25];x[21]=.2;x[22]=.3;x[5]=.006
    z=physics_features(x);assert len(z)==22
    y=x.copy();y[28:33]=99;y[34:]=88;np.testing.assert_array_equal(z,physics_features(y))
    y=x.copy();y[23]=.25;assert not np.array_equal(z,physics_features(y))


def test_kernel_normal_equation_and_reproducible_bootstrap():
    rng=np.random.default_rng(15);x=rng.normal(size=(12,3));y=np.c_[x[:,0],x[:,1]**2]
    a=fit_kernel(x,y,21);b=fit_kernel(x,y,21)
    assert float(a['equation_residual'])<1e-12
    np.testing.assert_array_equal(a['weights'],b['weights']);np.testing.assert_array_equal(predict_kernel(x,a),predict_kernel(x,b))
    assert np.isfinite(predict_kernel(x,a)).all()


def test_calibration_keeps_prior_and_candidate_maximum():
    p=np.zeros((24,5));y=p.copy();p[0,2]=.10
    margin,res=calibrated_margin(p,y,.03395401078737238)
    assert margin==pytest.approx(.03395401078737238) and res[0]==.10
    p[1,3]=.20;margin,res=calibrated_margin(p,y,.03)
    assert margin==.10
    assert choose([0,-.1,.03,.02,0],margin)==0 and choose([0,0,.15,0,0],margin)==2


def test_instrumentation_field_positions():
    assert len(TRACE_FIELDS)==40
    assert TRACE_FIELDS[16]=='occupancy_pre_room_a' and TRACE_FIELDS[19]=='occupancy_pre_north'
    assert TRACE_FIELDS[36:] == ('released_people','new_incapacitated','new_evacuated','fed_increment_sum')
