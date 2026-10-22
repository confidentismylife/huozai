from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from candidate_library_scores import scores,diagnostic_source_scores,choose,actual_gains


def test_R_and_M_are_distinct_and_baseline_exact_zero():
    r=np.zeros((2,17,6,1));r[0,12]=.10;r[1,12]=.03;r[0,0]=.02;r[1,0]=.08
    R,M=scores(r,np.array([True,True]));assert R[0]==pytest.approx(.02) and M[0]==pytest.approx(-.05)
    assert R[12]==M[12]==0


def test_only_supported_world_enters_observable_scores():
    r=np.zeros((2,17,6,1));r[0,12]=.80;r[1,12]=.04;r[1,0]=.02
    R,M=scores(r,np.array([False,True]));assert R[0]==M[0]==pytest.approx(.02)


def test_ties_return_601_and_threshold_is_strict():
    assert choose(np.zeros(17))==12
    g=np.zeros(17);g[0]=.001;assert choose(g)==12
    g[0]=.00101;assert choose(g)==0


def test_positive_plan_ties_use_index_and_margin_can_only_restrict():
    g=np.zeros(17);g[1]=g[4]=.04;assert choose(g)==1
    assert choose(g,.0395)==12 and choose(g,.03)==1


def test_diagnostic_oracle_uses_explicit_source_not_observable_selection():
    r=np.zeros((2,17,6,1));r[0,12]=.10;r[1,12]=.10;r[1,1]=.10
    oracle=diagnostic_source_scores(r,1);assert oracle[0]==pytest.approx(.10) and oracle[1]==0


def test_actual_scores_are_paired_with_identical_base():
    o=np.zeros((17,6,1));o[12]=.10;o[0]=.02
    g=actual_gains(o);assert g[0]==pytest.approx(.08) and g[12]==0


def test_zero_baseline_world_makes_strict_M_benefit_impossible():
    r=np.zeros((2,17,6,1));r[1,12]=.10;r[1,0]=.02
    R,M=scores(r,np.array([True,True]));assert R[0]>0 and M[0]==0 and M.max()==0
    assert choose(M)==12


def test_invalid_support_scores_or_shapes_rejected():
    with pytest.raises(ValueError):scores(np.zeros((2,17,6,1)),np.array([False,False]))
    with pytest.raises(ValueError):choose(np.ones(17))
    with pytest.raises(ValueError):actual_gains(np.zeros((16,6,1)))
