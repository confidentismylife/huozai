"""Matched 50-feature architecture control; plan fields stay zero throughout."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_mappo_capacity_stabilization_campaign as base
import run_mappo_six_hour_extension as core
from fire_evac.campaign_extension import write_json
from fire_evac.deepseek_advice import rule_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.mappo import MAPPOConfig

OUT=ROOT/'outputs'/'mappo_six_hour_extension_20261022'
MASKED='mappo_50_masked_control'
original_factory=base.make_env


def make_env(arm,scenarios,provider_variant='native'):
    if arm==MASKED:
        physical=original_factory('mappo_34_raw',scenarios)
        return core.training_seed_wrapper(base.MaskPlanFeatures(DeepSeekPlanObservationWrapper(physical,rule_plan)),scenarios)
    return original_factory(arm,scenarios,provider_variant)


def main():
    root=OUT/'mechanism'/'shared_masked'
    write_json(root/'manifest.json',{
        'purpose':'same 50-feature actor/critic architecture with plan inputs always zero',
        'training_seeds':list(range(20261301,20261306)),
        'timesteps':150000,'flow_entropy':.03,'route_entropy':.01,
        'plan_imitation':0.0,'shared_between_weak_and_strong_supervision':True,
        'API_requests':0,'final_test_scenarios_loaded':False})
    base.make_env=make_env
    scenarios=base.make_domain_randomized_trust_bank(24,20261021,'DSG')
    core.TRAIN_SCENARIO_IDS=tuple(s.scenario_id for s in scenarios[:12])
    config=MAPPOConfig(hidden_size=64,update_epochs=2,batch_size=64,
        route_entropy_coefficient=.01,flow_entropy_coefficient=.03,plan_imitation_coefficient=0)
    for seed in range(20261301,20261306):
        core.recorded_seeded_train_run(root/f'seed_{seed}'/MASKED,MASKED,seed,config,
            scenarios[:12],scenarios[12:],150000,30000)
        write_json(root/'status.json',{'status':'running','last_completed_seed':seed})
    write_json(root/'status.json',{'status':'complete','completed_seeds':5})
    print(json.dumps({'status':'complete','completed_seeds':5}))


if __name__=='__main__':main()
