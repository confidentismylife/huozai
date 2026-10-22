import itertools
import math

import numpy as np
import pytest

from fire_evac.safestep_public_benchmark import (
    Graph,RateField,nearest_exit_policy,snapshot_policy,future_min_fed_policy,
    forward_min_fed_reference,rollout,
)


def test_linear_rate_has_exact_integral_and_rejects_extrapolation():
    field=RateField(np.array([0.,5.,10.]),np.array([[0.,2.],[5.,2.],[10.,2.]]))
    assert field.integrate(2.,8.)==pytest.approx([30.,12.])
    assert field.sample(7.)==pytest.approx([7.,2.])
    with pytest.raises(ValueError):field.integrate(-1.,3.)
    with pytest.raises(ValueError):field.integrate(9.,11.)


def test_start_residence_charged_once_exit_stops_on_arrival():
    g=Graph([(0,0),(1,0),(2,0)],((2,0),))
    costs=np.array([[.1,.2,99.]]*4)
    p,_=nearest_exit_policy(g)
    r=rollout(g,np.tile(p,(4,1)),costs,0)
    assert r['reached'] and r['steps']==2 and r['FED']==pytest.approx(.3)
    assert [s['action'] for s in r['trace']]==[1,1]
    assert [s['source'] for s in r['trace']]==[(0,0),(1,0)]
    with pytest.raises(ValueError):g.destination(0,0)


def test_minimum_dose_can_choose_longer_route_than_nearest():
    g=Graph([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)],((2,0),))
    costs=np.array([[.01,1.,0.,.01,.01,.01]]*5)
    nearest,_=nearest_exit_policy(g)
    actions,values,_=future_min_fed_policy(g,costs)
    a=rollout(g,np.tile(nearest,(5,1)),costs,0)
    b=rollout(g,actions,costs,0)
    assert a['steps']==2 and b['steps']==4
    assert b['FED']==pytest.approx(.04) and b['FED']==pytest.approx(values[0,0])


def test_backward_solution_matches_exhaustive_paths_and_forward_search():
    g=Graph([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)],((2,0),))
    rng=np.random.default_rng(617)
    costs=rng.uniform(0,.1,(5,6));costs[:,2]=0
    _,values,_=future_min_fed_policy(g,costs)
    for start in (0,1,3,4,5):
        complete=[]
        def walk(k,i,dose):
            if i in g.exit_ids:complete.append(dose);return
            if k==len(costs):return
            for _,j in g.edges[i]:walk(k+1,j,dose+costs[k,i])
        walk(0,start,0.)
        assert values[0,start]==pytest.approx(min(complete))
        assert forward_min_fed_reference(g,costs,start)[0]==pytest.approx(values[0,start])


def test_snapshot_planner_has_no_future_input_and_replanning_can_change_action():
    g=Graph([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)],((2,0),))
    flat=snapshot_policy(g,np.zeros(6))
    smoke=snapshot_policy(g,np.array([.01,1.,0.,.01,.01,.01]))
    assert flat[0]==1  # right, nearest when the snapshot has no smoke
    assert smoke[0]==3  # down, detour when current route is smoky


def test_horizon_and_dose_objective_do_not_count_failure_as_zero_success_cost():
    g=Graph([(0,0),(1,0),(2,0)],((2,0),))
    costs=np.ones((1,3))
    _,v,_=future_min_fed_policy(g,costs)
    assert math.isinf(v[0,0])
    assert forward_min_fed_reference(g,costs,0)==(math.inf,None)
    p,_=nearest_exit_policy(g)
    r=rollout(g,np.tile(p,(1,1)),costs,0)
    assert not r['reached'] and r['steps']==1 and r['FED']==1.
