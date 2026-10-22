from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from proposal_gain_models import inputs,targets,fit,PairGainModel,calibrated_margin,accepted,family_weights


def test_inputs_exclude_advice_and_encode_candidate():
    x=np.zeros(49);x[28:33]=999;z=inputs(x,14)
    assert z.shape==(66,) and np.all(z[28:33]==0) and z[49+14]==1


def test_world_paired_target_can_disagree_with_worst_loss_difference():
    r=np.array([[[[.10]],[[.02]]],[[[.03]],[[.08]]]])
    np.testing.assert_allclose(targets(r,np.array([True,True])),[.02,-.05])


def test_baseline_duplicate_and_single_world_targets():
    r=np.ones((2,2,6,1))*.1;np.testing.assert_array_equal(targets(r,np.array([True,True])),[0,0])
    r[1,1]=.05;np.testing.assert_allclose(targets(r,np.array([False,True])),[.05,.05])


@pytest.mark.parametrize("kind",["linear","kernel"])
def test_closed_form_models_normal_equations_and_roundtrip(tmp_path,kind):
    x=np.zeros((8,66));x[:,0]=np.arange(8);y=np.array([0,.1,0,.1,0,.1,0,.1]);f=np.array([str(i//2) for i in range(8)])
    model,info=fit(x,y,f,kind);assert info["max_abs_normal_equation_residual"]<1e-10
    model.save(tmp_path/"model.npz");np.testing.assert_array_equal(PairGainModel.load(tmp_path/"model.npz").predict(x),model.predict(x))


def test_forbidden_family_leakage_rejected():
    with pytest.raises(ValueError):fit(np.zeros((2,66)),np.zeros(2),np.array(["train","test"]),"linear",("test",))


def test_family_weights_not_state_counts():
    np.testing.assert_allclose(family_weights(["a","a","b"]),[.25,.25,.5])


def test_margin_cannot_lower_old_floor_and_is_family_max():
    margin,v,q=calibrated_margin([.04,.01,.08],[0,0,0],["a","a","b"],.05)
    assert v=={"a":.04,"b":.08} and margin==q==.08
    assert calibrated_margin([0],[0],["a"],.05)[0]==.05


def test_empty_calibration_retains_floor():
    assert calibrated_margin([],[],[],.05)==(.05,{},None)


def test_full_filter_only_restricts_gain_gate_and_unseen_plan_abstains():
    assert accepted(.04,.03,[.05,.05,.90])
    assert not accepted(.04,.03,[.2,0,.8])
    assert accepted(.04,.03,[.2,0,.8],full=False)
    assert not accepted(.04,.03,[.05,.05,.90],seen=False)


def test_strict_threshold_and_invalid_inputs():
    assert not accepted(.001,0,[0,0,1])
    with pytest.raises(ValueError):inputs(np.zeros(49),17)
    with pytest.raises(ValueError):targets(np.zeros((2,3,6,1)),np.array([True,True]))
