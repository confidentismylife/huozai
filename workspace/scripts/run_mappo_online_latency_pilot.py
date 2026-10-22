"""Small 2x2 timing/return-check pilot with one fixed, already-trained policy."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_mappo_six_hour_extension as runner
from fire_evac.campaign_extension import BudgetedPlanner,stale_plan_reason,write_json
from fire_evac.deepseek_advice import rule_plan
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.mappo import FactorizedMAPPO

OUT=runner.OUT/'online_latency_pilot'


class TimedOnlineWrapper(DeepSeekPlanObservationWrapper):
    def __init__(self,env,planner,delayed,checked):
        super().__init__(env,planner,max_requests_per_episode=3)
        self.delayed=bool(delayed);self.checked=bool(checked)
        self.pending=None;self.receipts=[];self.rejections=Counter()

    def reset(self,**kwargs):
        self.pending=None;self.receipts=[];self.rejections=Counter()
        return super().reset(**kwargs)

    def _accept(self,plan,source,generated_step,observation,info):
        age=max(0,int(info.get('step',0))-generated_step)
        state=self._snapshot(observation,info,'return_check')
        reason=stale_plan_reason(plan,age,state) if self.checked else None
        self.receipts.append({'generated_step':generated_step,'returned_step':int(info.get('step',0)),
            'age_steps':age,'rejection_reason':reason,'source':source})
        if reason:
            self.rejections[reason]+=1
            self.current_plan=rule_plan(state);self.plan_source='return_check_'+reason
            self.plan_age_steps=0
        else:
            self.current_plan=plan;self.plan_source=source;self.plan_age_steps=age

    def _request_plan(self,observation,info,event_type):
        if self.pending is not None or self.requests_this_episode>=self.max_requests_per_episode:
            return
        state=self._snapshot(observation,info,event_type)
        result=self.provider(state)
        self.requests_this_episode+=1
        generated=int(info.get('step',0))
        latency=float(getattr(self.provider,'last_latency_seconds',0.0))
        if self.delayed:
            delay=max(1,int(math.ceil(latency/5.0)))
            self.pending=(generated+delay,result.plan,result.source,generated)
            if self.current_plan is None:
                self.current_plan=rule_plan(state);self.plan_source='awaiting_llm'
                self.plan_age_steps=0
        else:
            self._accept(result.plan,result.source,generated,observation,info)

    def step(self,action):
        observation,reward,terminated,truncated,info=self.env.step(action)
        self.plan_age_steps+=1
        if self.pending is not None and int(info.get('step',0))>=self.pending[0]:
            _,plan,source,generated=self.pending;self.pending=None
            self._accept(plan,source,generated,observation,info)
        event=self._event(observation,info)
        if event is not None and not (terminated or truncated):
            self._request_plan(observation,info,event)
        return self._encode(observation),reward,terminated,truncated,self._enrich_info(info)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():return 0
    write_json(OUT/'manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'policy':'original primary DeepSeek model seed 20261201, best checkpoint',
        'scenarios':'first six validation scenarios, never final test',
        'population_seed':42004,
        'variants':['native','rule','shuffled','masked','instant_unchecked','instant_checked',
                    'delayed_unchecked','delayed_checked'],
        'delay_semantics':'ceil(previously measured request latency / 5s), minimum one control step',
        'cache_semantics':'cached exact-state response replays its originally measured cold latency',
        'while_waiting':'MAPPO acts using prior plan; initial prior is deterministic rule',
        'return_check':'active phase blocked, observed smoke contradiction, or plan expired',
        'note':'return check screens advice; physical environment constraints identical in all variants',
        'API_budget_shared_with_extension':True,'final_test_scenarios_loaded':False})
    planner=BudgetedPlanner(runner.OUT/'api',max_cny=2,max_requests=240)
    checkpoint=runner.SOURCE/'confirmation'/'seed_20261201'/runner.CACHED/'best.pt'
    model,_=FactorizedMAPPO.load(checkpoint)
    scenarios=runner.base.make_domain_randomized_trust_bank(24,20261021,'DSG')[12:18]
    episodes=[];steps=[];receipts=[]
    for variant in ('native','rule','shuffled','masked','instant_unchecked','instant_checked',
                    'delayed_unchecked','delayed_checked'):
        if variant in ('native','rule','shuffled','masked'):
            env=runner.factory(runner.CACHED,scenarios,variant)
        else:
            env=TimedOnlineWrapper(runner.ORIGINAL_FACTORY(runner.RAW,scenarios),planner,
                delayed=variant.startswith('delayed'),checked=variant.endswith('_checked'))
        for scenario in scenarios:
            er,sr=runner.detailed_evaluation(model,env,[scenario],(42004,),runner.CACHED,variant,
                {'training_seed':20261201,'model_label':'primary_seed_20261201'})
            er[0]['return_rejections']=sum(getattr(env,'rejections',{}).values())
            er[0]['rejected_blocked']=getattr(env,'rejections',{}).get('blocked_exit',0)
            er[0]['rejected_smoke']=getattr(env,'rejections',{}).get('observed_smoke_contradiction',0)
            er[0]['rejected_expired']=getattr(env,'rejections',{}).get('expired',0)
            episodes.extend(er);steps.extend(sr)
            for receipt in getattr(env,'receipts',[]):
                receipts.append({'scenario_id':scenario.scenario_id,'variant':variant,**receipt})
            runner.base.write_csv(OUT/'episode_results.csv',episodes)
            runner.base.write_csv(OUT/'step_results.csv',steps)
            write_json(OUT/'receipts.json',receipts)
            write_json(OUT/'status.json',{'status':'running','last_completed':f'{variant}/{scenario.scenario_id}',
                'episodes':len(episodes),'api':planner.ledger()})
        env.close()
    write_json(OUT/'complete.json',{'episodes':len(episodes),'steps':len(steps),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'budget':planner.ledger(),'final_test_scenarios_loaded':False})
    write_json(OUT/'status.json',{'status':'complete','episodes':len(episodes)})
    return 0


if __name__=='__main__':raise SystemExit(main())
