from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from five_ppo_student import pooled_labels, fit, MultiGain, calibrate, choices


def test_population_pooling_precedes_nonlinear_world_targets():
    r = np.zeros((2,2,5,6,1)); r[0,0,0]=.1; r[1,1,0]=.1
    targets, means = pooled_labels(r,np.array([True,True]))
    assert targets[0,1] == pytest.approx(.05) and targets[1,1] == pytest.approx(.05)
    assert means.shape == (2,5) and not targets[:,0].any()


def test_R_and_M_distinguish_supported_world_conflict():
    r = np.zeros((2,2,5,6,1)); r[:,0,0]=.1; r[:,1,0]=.03; r[:,0,1]=.02; r[:,1,1]=.08
    target,_=pooled_labels(r,np.array([True,True]));assert target[0,1]==pytest.approx(.02) and target[1,1]==pytest.approx(-.05)


@pytest.mark.parametrize("kind",["linear","kernel"])
def test_candidate_heads_change_ranking_and_baseline_stays_zero(kind,tmp_path):
    x=np.zeros((10,49));x[:,0]=np.linspace(-1,1,10);y=np.zeros((10,5));y[:,1]=.01*x[:,0];y[:,2]=-.01*x[:,0]
    f=[f"A{i}" for i in range(10)];model,info=fit(x,y,f,kind,f,["B"])
    p=model.predict(x);assert np.argmax(p[0])==2 and np.argmax(p[-1])==1 and not p[:,0].any()
    assert info["normal_equation_max_residual"]<1e-10
    path=tmp_path/'m.npz';model.save(path);np.testing.assert_allclose(MultiGain.load(path).predict(x),p)


def test_training_cannot_take_calibration_family():
    x=np.zeros((2,49));y=np.zeros((2,5))
    with pytest.raises(ValueError):fit(x,y,["A","B"],"linear",["A"],["B"])


def test_pooling_rejects_negative_loss_or_missing_population():
    r=np.zeros((2,2,5,6,1));r[0,0,0,0,0]=-.1
    with pytest.raises(ValueError):pooled_labels(r,np.array([True,True]))
    with pytest.raises(ValueError):pooled_labels(np.zeros((1,2,5,6,1)),np.array([True,True]))


def test_calibration_uses_all_candidates_and_never_reduces_prior():
    p=np.zeros((3,5));e=np.zeros_like(p);p[1,4]=.2
    margin,residual,q=calibrate(p,e,["A","B","C"],.03395401078737238)
    assert margin==q==.2 and residual['B']==.2
    margin,_,_=calibrate(np.zeros_like(p),e,["A","B","C"],.03395401078737238)
    assert margin==.03395401078737238


def test_same_score_for_nomination_and_margin_and_strict_floor():
    g=np.zeros(5);g[1]=.001;assert choices(g,0)=={"point":0,"margin":0}
    g[2]=.02;assert choices(g,.03)=={"point":2,"margin":0}
