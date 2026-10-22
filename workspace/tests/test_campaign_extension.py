from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fire_evac.campaign_extension import (
    BudgetedPlanner, admission_allowed, plan_phase, execution_diagnostic, write_json,
)
from fire_evac.deepseek_advice import rule_plan


STATE = dict(scenario_id='unit', event_type='initial', north_exit_capacity=8,
             south_exit_capacity=4, forecast_source_branch=0,
             occupancy=[.4,.4,.1,.05,.05], smoke=[0,0,0,0,0])


def test_budget_is_reserved_before_transport_and_failure_retains_reserve(tmp_path):
    seen = []
    def failing(messages, limit):
        ledger = json.loads((tmp_path / 'budget.json').read_text())
        seen.append(ledger['reserved_cny'])
        raise TimeoutError('synthetic outage')
    planner = BudgetedPlanner(tmp_path, transport=failing, max_requests=1)
    first = planner(STATE)
    second = planner({**STATE, 'scenario_id':'second'})
    assert seen and seen[0] > 0
    ledger = planner.ledger()
    assert ledger['attempted_requests'] == 1
    assert ledger['reserved_cny'] > 0
    assert first.source == 'fallback_transport'
    assert second.source == 'fallback_budget'


def test_valid_response_is_cached_without_more_spending(tmp_path):
    plan = rule_plan(STATE)
    from dataclasses import asdict
    calls = []
    def transport(messages, limit):
        calls.append(1)
        return json.dumps(asdict(plan)), {'prompt_tokens':400,
            'completion_tokens':150, 'prompt_cache_hit_tokens':100}
    planner = BudgetedPlanner(tmp_path, transport=transport)
    a = planner(STATE)
    b = planner(STATE)
    assert a.source == 'api' and b.source == 'cache'
    assert len(calls) == 1
    assert planner.ledger()['reserved_cny'] == 0
    assert planner.ledger()['actual_cost_upper_cny'] > 0
    assert planner.ledger()['successful_requests'] == 1


def test_no_network_when_money_is_exhausted(tmp_path):
    planner = BudgetedPlanner(tmp_path, max_cny=0,
        transport=lambda *_: pytest.fail('network must remain unused'))
    assert planner(STATE).source == 'fallback_budget'
    assert planner.ledger()['attempted_requests'] == 0


def test_deadline_admission_reserves_time_for_a_whole_paired_block():
    assert admission_allowed(100, 1000, 600, 120)
    assert not admission_allowed(300, 1000, 600, 120)


def test_phase_boundaries_use_full_duration_without_twelve_step_clipping():
    plan = rule_plan(STATE)
    assert plan_phase(plan,0) == 0
    assert plan_phase(plan,2) == 1
    assert plan_phase(plan,5) == 2
    assert plan_phase(plan,50) == 2


def test_execution_diagnostic_does_not_confuse_requested_and_applied_route():
    info = {'requested_joint_action':3, 'actual_joint_action':7,
            'actual_sign_action':1, 'actual_release_fraction':1.0}
    result = execution_diagnostic(3, info, 0, 1.0, 12)
    assert result['requested_plan_route_match'] == 1
    assert result['executed_plan_route_match'] == 0
    assert result['joint_action_compliance'] == 0
    assert result['executed_plan_flow_match'] == 1


def test_wait_and_neutral_execution_have_explicit_semantics():
    neutral = execution_diagnostic(0, {'actual_joint_action':-1,
        'actual_sign_action':2,'actual_release_fraction':.25},0,.25,12)
    assert neutral['executed_plan_route_match'] == 0
    assert neutral['actual_joint_action'] == -1
    wait = execution_diagnostic(8, {'actual_joint_action':8,
        'actual_sign_action':0,'actual_release_fraction':0},2,.25,12)
    assert wait['executed_plan_route_match'] == 1
    assert np.isnan(wait['executed_plan_flow_match'])


def test_observed_state_collection_reaches_frozen_target_without_fake_events(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_mappo_six_hour_extension as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    scenarios = runner.base.make_domain_randomized_trust_bank(24,20261021,'DSG')[:12]
    states = runner.collect_states(scenarios)
    assert len(states) == 48
    assert len({state['scenario_id'] for state in states}) == 48
    assert all(state['original_scenario_id'] in {s.scenario_id for s in scenarios} for state in states)
    assert all(state['timestamp_step'] >= 0 for state in states)
    assert {state['event_type'] for state in states} <= {
        'initial','blockage_change','smoke_reversal','congestion_threshold'}
    assert (tmp_path/'plan_bank'/'freeze_manifest.json').exists()


def test_atomic_status_write_survives_transient_windows_reader_lock(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []
    def briefly_locked(self, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(5,'synthetic Windows sharing lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',briefly_locked)
    write_json(tmp_path/'status.json',{'status':'running'})
    assert json.loads((tmp_path/'status.json').read_text()) == {'status':'running'}
    assert len(attempts) == 3


def test_crossed_bootstrap_keeps_constant_paired_difference():
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_mappo_six_hour_extension import paired_effect
    import pandas as pd
    rows=[]
    for seed in (1,2,3):
        for scenario in ('a','b','c','d'):
            for population in (4,5):
                for method,delta in [('baseline',0),('candidate',-.02)]:
                    rows.append(dict(training_seed=seed,scenario_id=scenario,
                        population_seed=population,arm=method,loss=seed*.1+delta))
    result=paired_effect(pd.DataFrame(rows),'arm','candidate','baseline','loss')
    assert result['mean'] == pytest.approx(-.02)
    assert result['crossed_ci95'] == pytest.approx([-.02,-.02])
    assert result['n_training_seeds'] == 3
    assert result['n_scenarios'] == 4
    lo,hi=result['independent_training_crossed_ci95']
    assert lo < -.02 < hi


def test_pairing_rejects_missing_counterfactual_rows():
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_mappo_six_hour_extension import paired_effect
    import pandas as pd
    data=pd.DataFrame([
        dict(training_seed=1,scenario_id='a',population_seed=2,arm='baseline',loss=.1),
        dict(training_seed=1,scenario_id='b',population_seed=2,arm='candidate',loss=.1)])
    with pytest.raises(ValueError,match='pair'):
        paired_effect(data,'arm','candidate','baseline','loss')


def test_returned_plan_checks_current_active_phase_and_expiration():
    from fire_evac.campaign_extension import stale_plan_reason
    from fire_evac.deepseek_advice import EvacuationPlan,PlanPhase
    plan=EvacuationPlan((PlanPhase('north',.5,2),PlanPhase('south',.5,3),
        PlanPhase('south',.75,4)),.7,('respect_blockage',),('blockage_change',))
    state={**STATE,'north_exit_capacity':0,'south_exit_capacity':4}
    assert stale_plan_reason(plan,0,state)=='blocked_exit'
    assert stale_plan_reason(plan,2,state) is None
    assert stale_plan_reason(plan,9,state)=='expired'
    smoky={**state,'smoke':[0,0,0,0,.05]}
    assert stale_plan_reason(plan,2,smoky)=='observed_smoke_contradiction'


def test_smoke_contradiction_follows_active_plan_phase():
    from fire_evac.campaign_extension import ActivePhasePlanWrapper
    from fire_evac.deepseek_advice import EvacuationPlan,PlanPhase
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_mappo_capacity_stabilization_campaign as campaign
    scenario=campaign.make_domain_randomized_trust_bank(6,20261021,'UNIT')[0]
    env=campaign.make_env('mappo_34_raw',[scenario])
    wrapper=ActivePhasePlanWrapper(env,rule_plan)
    wrapper.current_plan=EvacuationPlan((PlanPhase('north',.5,2),PlanPhase('south',.5,3),
        PlanPhase('south',.75,4)),.7,('respect_blockage',),('blockage_change',))
    observation=np.zeros(34,dtype=np.float32);observation[8]=.03
    wrapper.plan_age_steps=0
    assert wrapper._contradiction(observation)
    wrapper.plan_age_steps=2
    assert not wrapper._contradiction(observation)
    assert wrapper._encode(observation)[48] == 0
    wrapper.close()


def test_episode_seed_schedule_reproduces_real_training_across_resets():
    from fire_evac.campaign_extension import EpisodeSeedScheduleWrapper
    from fire_evac.mappo import FactorizedMAPPO,MAPPOConfig
    import torch
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_mappo_capacity_stabilization_campaign as campaign
    scenes=campaign.make_domain_randomized_trust_bank(24,20261021,'DSG')[:12]
    networks=[]
    for repeat in range(2):
        env=EpisodeSeedScheduleWrapper(campaign.make_env('mappo_34_raw',scenes))
        model=FactorizedMAPPO(34,MAPPOConfig(hidden_size=16,update_epochs=1,batch_size=32),seed=913)
        record=model.train(env,total_timesteps=1024,rollout_steps=128)
        assert record['episodes']>2
        networks.append({k:v.detach().clone() for k,v in model.network.state_dict().items()})
        env.close()
    assert all(torch.equal(networks[0][k],networks[1][k]) for k in networks[0])


def test_episode_schedule_preserves_explicit_evaluation_population_seed():
    from fire_evac.campaign_extension import EpisodeSeedScheduleWrapper
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_mappo_capacity_stabilization_campaign as campaign
    scenes=campaign.make_domain_randomized_trust_bank(24,20261021,'DSG')[12:]
    ordinary=campaign.make_env('mappo_34_raw',scenes)
    scheduled=EpisodeSeedScheduleWrapper(campaign.make_env('mappo_34_raw',scenes))
    a,_=ordinary.reset(seed=42001,options={'scenario':scenes[0]})
    b,_=scheduled.reset(seed=42001,options={'scenario':scenes[0]})
    np.testing.assert_array_equal(a,b)
    for step in range(12):
        a,ra,ta,ua,ia=ordinary.step(3)
        b,rb,tb,ub,ib=scheduled.step(3)
        np.testing.assert_array_equal(a,b)
        assert (ra,ta,ua,ia['incapacitation_rate'])==(rb,tb,ub,ib['incapacitation_rate'])
        if ta or ua:break
    ordinary.close();scheduled.close()


def test_derived_execution_statistics_distinguish_neutral_release_from_wait():
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_mappo_six_hour_extension import normalize_execution_routes
    import pandas as pd
    source=pd.DataFrame({'requested_joint_action':[0,8],
        'actual_joint_action':[-1,8],'target_route':[2,2],
        'executed_route':[2,2],'executed_plan_route_match':[1,1]})
    corrected=normalize_execution_routes(source)
    assert corrected.executed_route.tolist()==[-1,2]
    assert corrected.executed_plan_route_match.tolist()==[0,1]
    assert source.executed_route.tolist()==[2,2]
