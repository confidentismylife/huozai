"""Tests for diagnostic estimands, threshold semantics and algebra only."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from observation_reversal_audit import scores, reversal, kernel_decomposition


def test_pool_before_world_minimum():
    a=np.zeros((2,2,5,2));a[:,:,0,:]=.5
    a[0,0,1,:]=.1;a[0,1,1,:]=.9;a[1,0,1,:]=.9;a[1,1,1,:]=.1
    losses,y=scores(a,np.array([True,True]))
    assert y[1,1] == 0 and y[0,1] == 0
    assert np.mean([scores(a[i:i+1],np.array([True,True]))[1][1,1] for i in range(2)]) == -.4
    np.testing.assert_array_equal(y[:,0],np.zeros(2))


def test_R_max_world_can_change_and_M_is_paired():
    a=np.zeros((1,2,5,1));a[0,:,0,0]=[.9,.2];a[0,:,1,0]=[.3,.5]
    _,y=scores(a,np.array([True,True]))
    assert y[0,1] == pytest.approx(.4)
    assert y[1,1] == pytest.approx(-.3)
    _,supported=scores(a,np.array([True,False]))
    assert supported[1,1] == pytest.approx(.6)


def test_sign_reversal_is_strict_and_symmetric():
    assert reversal(.01,-.01) and reversal(-.01,.01)
    assert not reversal(.001,-.01) and not reversal(-.001,.01)
    assert not reversal(0,.01) and not reversal(.01,.02)


def test_centered_kernel_exact_terms_without_convex_weights():
    m={'mean':np.zeros(2),'std':np.ones(2),'train':np.array([[0.,0.],[1.,1.]]),
       'gamma':np.array(.5),'weights':np.array([.5,.5]),'kernel_row_mean':np.array([.7,.7]),
       'kernel_grand_mean':np.array(.7),'coefficient':np.array([[2.,-1.,.5,0.],[-2.,1.,-.5,1.]]),'intercept':np.array([.1,.2,.3,.4])}
    _,k,c,terms,y=kernel_decomposition(np.zeros(2),m)
    assert c[1] < 0
    expected=(k-k.mean()-m['kernel_row_mean']+m['kernel_grand_mean'])@m['coefficient']+m['intercept']
    np.testing.assert_allclose(y[1:],expected)
    np.testing.assert_allclose(terms.sum(axis=0)+m['intercept'],y[1:]);assert y[0]==0


def test_invalid_support_and_loss_rejected():
    a=np.zeros((2,2,5,6))
    with pytest.raises(ValueError):scores(a,np.array([False,False]))
    with pytest.raises(ValueError):scores(a,np.array([1,1]))
    a[0,0,1,0]=1.1
    with pytest.raises(ValueError):scores(a,np.array([True,True]))
