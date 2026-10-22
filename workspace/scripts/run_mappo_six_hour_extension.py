"""Six-hour continuation; preserve the original experiment and its selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_mappo_capacity_stabilization_campaign as base
from fire_evac.campaign_extension import (
    ActivePhasePlanWrapper, EpisodeSeedScheduleWrapper, BudgetedPlanner,
    admission_allowed, execution_diagnostic, plan_phase, write_json,
)
from fire_evac.deepseek_advice import encode_plan_features, rule_plan
from fire_evac.deepseek_format_gate import plan_safety_violations
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from fire_evac.mappo import FactorizedMAPPO, MAPPOConfig, FLOW_LEVELS, encode_factorized_action

SOURCE = ROOT / 'outputs' / 'mappo_capacity_stabilization_20261022'
OUT = ROOT / 'outputs' / 'mappo_six_hour_extension_20261022'
RAW, RULE, CACHED = base.ARMS
EXPANDED = 'mappo_50_deepseek_expanded'
CONFIGS = (('e003_i002',.03,.02), ('e003_i005',.03,.05))
SEEDS = tuple(range(20261301,20261306))
ORIGINAL_FACTORY = base.make_env
TRAIN_SCENARIO_IDS=()


def training_seed_wrapper(env,scenarios):
    ids=tuple(s.scenario_id for s in scenarios)
    return EpisodeSeedScheduleWrapper(env) if TRAIN_SCENARIO_IDS and ids==TRAIN_SCENARIO_IDS else env


def recorded_seeded_train_run(output,arm,seed,config,train,validation,timesteps,checkpoint_interval):
    protocol=output/'training_protocol.json'
    if (output/'summary.json').exists() and not protocol.exists():
        raise RuntimeError('Refusing to reuse a run without deterministic episode-seed provenance')
    probe=base.make_env(arm,train)
    if not isinstance(probe,EpisodeSeedScheduleWrapper):
        raise RuntimeError('Training environment lacks the required episode seed schedule')
    probe.close()
    if not protocol.exists():
        write_json(protocol,{'training_seed':seed,'episode_seed_schedule':'derived-explicit-uint32-v1',
            'explicit_evaluation_seeds_unchanged':True,'train_scenario_ids':[s.scenario_id for s in train]})
    result=base.train_run(output,arm,seed,config,train,validation,timesteps,checkpoint_interval)
    result['episode_seed_schedule']='derived-explicit-uint32-v1'
    write_json(output/'summary.json',result)
    return result


def now():
    return datetime.now(timezone.utc)


def status(**changes):
    path = OUT / 'extension_status.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data.update(changes)
    data.update(updated_at_utc=now().isoformat(), pid=__import__('os').getpid(),
                final_test_scenarios_loaded=False)
    write_json(path,data)


def expanded_library():
    records = json.loads((OUT/'plan_bank'/'response_audit.json').read_text(encoding='utf-8'))
    from fire_evac.deepseek_advice import parse_plan
    return FrozenPlanLibrary([(r['state'],parse_plan(r['plan'])) for r in records
                               if r['valid_model_response']])


def factory(arm, scenarios, provider_variant='native'):
    if arm == RAW:
        return training_seed_wrapper(ORIGINAL_FACTORY(arm,scenarios,provider_variant),scenarios)
    physical = ORIGINAL_FACTORY(RAW,scenarios)
    library=expanded_library if arm==EXPANDED else base.new_library
    if provider_variant == 'rule' or arm==RULE:
        provider = rule_plan
    elif provider_variant == 'shuffled':
        provider = base.ShuffledPlanProvider(library())
    else:
        provider = library()
    wrapped = ActivePhasePlanWrapper(physical,provider,max_requests_per_episode=3)
    finished=base.MaskPlanFeatures(wrapped) if provider_variant == 'masked' else wrapped
    return training_seed_wrapper(finished,scenarios)


def prepare_manifest(deadline):
    manifest = {
        'protocol':'six-hour-development-extension-v1',
        'frozen_at_utc':now().isoformat(),
        'deadline_utc':deadline.isoformat(), 'total_hours_from_original_start':6,
        'source':str(SOURCE),'original_results_preserved':True,
        'development_only':True,'final_test_scenarios_loaded':False,
        'mechanism_design_is_exploratory_after_observing_screen':True,
        'configs':[{'config_id':c,'flow_entropy':e,'plan_imitation':i} for c,e,i in CONFIGS],
        'training_seeds':list(SEEDS),'arms':[RAW,RULE,CACHED,EXPANDED],
        'timesteps_per_arm':150000,'checkpoint_interval':30000,
        'shared_raw_baseline':'one entropy=.03, imitation=0 raw model per seed shared between settings',
        'new_plan_bank_states':48,'new_api_hard_cap_cny':2.0,'new_api_request_cap':240,
        'raw_actor_information':'same numeric physical/forecast information; no plan features',
        'direct_online_pilot':{'primary_seed':20261201,'mechanism_seed':20261301,
            'models':['primary_cached','weak_cached','strong_cached','weak_expanded'],
            'population_seed':42003,'scenarios':'first six validation scenarios in generator order',
            'variants':['native','rule','shuffled','masked','online'],
            'timing':'simulator paused during API; ideal zero-simulation-latency diagnostic only'},
        'extension_policy':'complete all mandatory matched blocks, then add whole primary seed triples; reserve report time',
    }
    path = OUT/'extension_manifest.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing['deadline_utc'] != manifest['deadline_utc']:
            raise RuntimeError('Extension deadline changed after freeze')
    else:
        write_json(path,manifest)


def collect_states(train_scenarios):
    path=OUT/'plan_bank'/'frozen_states.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    candidates={}
    # A run often has only one or two genuine trigger events. Collect more
    # population realizations rather than manufacturing events to hit a quota.
    for population in range(43001,43009):
        for scenario in train_scenarios:
            captured=[]
            def recorder(state):
                captured.append(dict(state))
                return rule_plan(state)
            env=DeepSeekPlanObservationWrapper(ORIGINAL_FACTORY(RAW,[scenario]),recorder,
                max_requests_per_episode=3)
            observation,info=env.reset(seed=population,options={'scenario':scenario})
            while True:
                active=env.current_plan.phases[plan_phase(env.current_plan,env.plan_age_steps)]
                route=('north','south','wait').index(active.route)
                flow=FLOW_LEVELS.index(active.release_fraction)
                observation,_,terminated,truncated,info=env.step(encode_factorized_action(route,flow))
                if terminated or truncated:
                    break
            env.close()
            for state in captured:
                identity=json.dumps(state,sort_keys=True)
                digest=hashlib.sha256(identity.encode()).hexdigest()
                state={**state,'original_scenario_id':state['scenario_id'],
                    'scenario_id':'LIVEBANK_'+digest[:16],
                    'semantic_context':{'control_period_seconds':5,
                        'observation_source':'training-only actual simulator decision snapshot'}}
                candidates.setdefault(digest,state)
    # Stratify by trigger label before a deterministic hash order round robin.
    grouped={}
    for digest,state in sorted(candidates.items()):
        grouped.setdefault(state['event_type'],[]).append((digest,state))
    states=[]
    while len(states)<48 and any(grouped.values()):
        for event in sorted(grouped):
            if grouped[event] and len(states)<48:
                states.append(grouped[event].pop(0)[1])
    if len(states)!=48:
        raise RuntimeError(f'Only {len(states)} unique observed event snapshots; require 48')
    write_json(path,states)
    write_json(OUT/'plan_bank'/'freeze_manifest.json',{
        'frozen_at_utc':now().isoformat(),'state_count':len(states),
        'states_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'final_test_scenarios_loaded':False,'source':'actual training-only rollouts',
        'event_counts':{event:sum(s['event_type']==event for s in states)
                         for event in sorted({s['event_type'] for s in states})}})
    return states


def generate_bank(planner,train_scenarios):
    path=OUT/'plan_bank'/'response_audit.json'
    if path.exists() and (OUT/'plan_bank'/'gate.json').exists():
        return json.loads((OUT/'plan_bank'/'gate.json').read_text(encoding='utf-8'))['passed']
    states=collect_states(train_scenarios)
    rows=[]
    for index,state in enumerate(states):
        result=planner(state)
        rows.append({'state':state,'source':result.source,
            'valid_model_response':result.source in ('api','cache'),
            'plan':asdict(result.plan),'current_state_safety_violations':plan_safety_violations(result.plan,state),
            'usage':asdict(result.usage)})
        write_json(path,rows)
        status(phase='generating_observed_state_plans',bank_completed=index+1,
               api=planner.ledger())
    valid=sum(r['valid_model_response'] for r in rows)
    gate={'passed':valid/len(rows)>=.95,'valid':valid,'total':len(rows),
          'valid_rate':valid/len(rows),
          'current_state_safety_violations':sum(len(r['current_state_safety_violations']) for r in rows),
          'criterion':'schema validity >=95%; unsafe first routes recorded, not erased',
          'new_api_ledger':planner.ledger()}
    write_json(OUT/'plan_bank'/'gate.json',gate)
    return gate['passed']


def wait_original():
    # Status alone does not prove a process is live. Validate the exact PID/command.
    launch=json.loads((SOURCE/'launch.json').read_text(encoding='utf-8'))
    import psutil
    process_id=int(launch['process_id'])
    while True:
        original=json.loads((SOURCE/'campaign_status.json').read_text(encoding='utf-8'))
        live=False
        try:
            process=psutil.Process(process_id)
            live=process.is_running() and 'run_mappo_capacity_stabilization_campaign.py' in ' '.join(process.cmdline())
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
        status(phase='waiting_original_confirmation',original_live=live,
               original_status=original.get('status'), original_last_completed=original.get('last_completed'))
        if original.get('status')=='complete' and (SOURCE/'channel_audit.csv').exists():
            return
        if not live:
            raise RuntimeError('Original process ended before its complete four-channel audit')
        time.sleep(30)


def detailed_evaluation(model,env,scenarios,population_seeds,arm,variant,context):
    rows=[]; steps=[]
    for scenario in scenarios:
        for population in population_seeds:
            observation,info=env.reset(seed=population,options={'scenario':scenario})
            lengths=[]; waits=0; requested_matches=[]; executed_matches=[]
            flows=[]; executed_flows=[]; compliance=[]; sources={}
            while True:
                routes,flow_actions,_,_=model.act(observation,deterministic=True)
                route,flow=int(routes[0]),int(flow_actions[0])
                action=encode_factorized_action(route,flow)
                # Find plan wrapper through the masking wrapper when necessary.
                wrapped=env.env if isinstance(env,base.MaskPlanFeatures) else env
                plan=getattr(wrapped,'current_plan',None)
                age=int(getattr(wrapped,'plan_age_steps',0))
                phase=plan_phase(plan,age) if plan is not None else 0
                target_route=('north','south','wait').index(plan.phases[phase].route) if plan else -1
                target_release=plan.phases[phase].release_fraction if plan else 0.0
                source=str(info.get('deepseek_plan_source','none'))
                sources[source]=sources.get(source,0)+1
                full_capacity=int(env.unwrapped.domain_env.config.start_to_junction_capacity)
                previous_step=int(info.get('step',0))
                observation,_,terminated,truncated,info=env.step(action)
                diag=execution_diagnostic(action,info,target_route,target_release,full_capacity)
                requested_matches.append(diag['requested_plan_route_match'])
                executed_matches.append(diag['executed_plan_route_match'])
                executed_flows.append(diag['executed_plan_flow_match'])
                compliance.append(diag['joint_action_compliance'])
                if route==2: waits+=1
                else: flows.append(FLOW_LEVELS[flow])
                lengths.append(float(info['actual_release_fraction']))
                steps.append({**context,'arm':arm,'provider_variant':variant,
                    'scenario_id':scenario.scenario_id,'population_seed':population,
                    'decision_step':previous_step,'plan_age_steps':age,'plan_phase':phase,
                    'plan_source':source,'target_route':target_route,'target_release':target_release,
                    **diag})
                if terminated or truncated:
                    break
            finite=np.asarray(executed_flows);finite=finite[np.isfinite(finite)]
            rows.append({**context,'arm':arm,'provider_variant':variant,
                'scenario_id':scenario.scenario_id,'conflict_type':scenario.conflict_type,
                'population_seed':population,'incapacitation_rate':float(info['incapacitation_rate']),
                'mean_cumulative_fed':float(info['mean_cumulative_fed']),
                'cumulative_congestion':float(info['cumulative_congestion']),
                'evacuation_time_seconds':5.0*int(info['step']),
                'wait_fraction':waits/len(lengths),
                'mean_release_fraction_nonwait':float(np.mean(flows)) if flows else 0.0,
                'mean_actual_release_fraction':float(np.mean(lengths)),
                'joint_action_compliance':float(np.mean(compliance)),
                'requested_plan_route_match':float(np.mean(requested_matches)),
                'executed_plan_route_match':float(np.mean(executed_matches)),
                'executed_plan_flow_match':float(finite.mean()) if len(finite) else np.nan,
                'plan_requests':int(info.get('deepseek_requests_this_episode',0)),
                'api_source_control_steps':sources.get('api',0),
                'cache_source_control_steps':sources.get('cache',0),
                'fallback_control_steps':sum(v for k,v in sources.items() if k.startswith('fallback'))})
    return rows,steps


def audit_checkpoint(checkpoint,scenarios,arm,directory,context,variants=('native','rule','shuffled','masked'),env_factory=None):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'audit_complete.json').exists():
        return
    model,_=FactorizedMAPPO.load(checkpoint)
    episodes=[];steps=[]
    for variant in variants:
        env=(env_factory or factory)(arm,scenarios,variant)
        episode_rows,step_rows=detailed_evaluation(model,env,scenarios,(42001,42002),arm,variant,context)
        env.close();episodes.extend(episode_rows);steps.extend(step_rows)
    base.write_csv(directory/'episode_results.csv',episodes)
    base.write_csv(directory/'step_results.csv',steps)
    write_json(directory/'audit_complete.json',{'checkpoint':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'episodes':len(episodes),'steps':len(steps),'variants':list(variants),
        'completed_at_utc':now().isoformat()})


def run_mechanism(train,validation,bank_passed):
    arms=(RULE,CACHED,EXPANDED) if bank_passed else (RULE,CACHED)
    for seed in SEEDS:
        raw_config=MAPPOConfig(hidden_size=64,update_epochs=2,batch_size=64,
            route_entropy_coefficient=.01,flow_entropy_coefficient=.03,plan_imitation_coefficient=0)
        raw_directory=OUT/'mechanism'/'shared_raw'/f'seed_{seed}'/RAW
        recorded_seeded_train_run(raw_directory,RAW,seed,raw_config,train,validation,150000,30000)
        for cid,entropy,imitation in CONFIGS:
            config=MAPPOConfig(hidden_size=64,update_epochs=2,batch_size=64,
                route_entropy_coefficient=.01,flow_entropy_coefficient=entropy,
                plan_imitation_coefficient=imitation)
            for arm in arms:
                directory=OUT/'mechanism'/cid/f'seed_{seed}'/arm
                recorded_seeded_train_run(directory,arm,seed,config,train,validation,150000,30000)
                status(phase='mechanism_training',last_completed=f'{cid}/{seed}/{arm}')
    for cid,_,_ in CONFIGS:
        for seed in SEEDS:
            for arm in (CACHED,EXPANDED) if bank_passed else (CACHED,):
                checkpoint=OUT/'mechanism'/cid/f'seed_{seed}'/arm/'best.pt'
                directory=OUT/'mechanism_audit'/cid/f'seed_{seed}'/arm
                audit_checkpoint(checkpoint,validation,arm,directory,
                    {'config_id':cid,'training_seed':seed})
                status(phase='mechanism_channel_audit',last_completed=f'{cid}/{seed}/{arm}')


def run_online_pilot(planner,validation,bank_passed):
    models=[('primary_cached',SOURCE/'confirmation'/'seed_20261201'/CACHED/'best.pt',CACHED),
        ('weak_cached',OUT/'mechanism'/'e003_i002'/'seed_20261301'/CACHED/'best.pt',CACHED),
        ('strong_cached',OUT/'mechanism'/'e003_i005'/'seed_20261301'/CACHED/'best.pt',CACHED)]
    if bank_passed:
        models.append(('weak_expanded',OUT/'mechanism'/'e003_i002'/'seed_20261301'/EXPANDED/'best.pt',EXPANDED))
    for label,checkpoint,arm in models:
        directory=OUT/'online_pilot'/label
        if (directory/'complete.json').exists():continue
        directory.mkdir(parents=True,exist_ok=True)
        model,_=FactorizedMAPPO.load(checkpoint)
        episodes=[];steps=[]
        for variant in ('native','rule','shuffled','masked','online'):
            if variant=='online':
                wrapper=DeepSeekPlanObservationWrapper if label=='primary_cached' else ActivePhasePlanWrapper
                env=wrapper(ORIGINAL_FACTORY(RAW,validation[:6]),planner,max_requests_per_episode=3)
            else:
                env=(ORIGINAL_FACTORY if label=='primary_cached' else factory)(arm,validation[:6],variant)
            er,sr=detailed_evaluation(model,env,validation[:6],(42003,),arm,variant,
                {'model_label':label,'training_seed':20261201 if label=='primary_cached' else 20261301})
            env.close();episodes.extend(er);steps.extend(sr)
            base.write_csv(directory/'episode_results.csv',episodes)
            base.write_csv(directory/'step_results.csv',steps)
            status(phase='direct_online_pilot',last_completed=f'{label}/{variant}',api=planner.ledger())
        write_json(directory/'complete.json',{'episodes':len(episodes),'steps':len(steps),
            'ideal_zero_simulation_latency':True,'checkpoint':str(checkpoint),
            'completed_at_utc':now().isoformat()})


def extend_primary(train,validation,deadline):
    if (OUT/'deterministic_training_correction.json').exists():
        status(phase='finalizing',additional_primary_seeds=0,
            additional_legacy_training_skipped_reason='Do not expand OS-reseeded legacy training; remaining time used for deterministic checkpoint replays')
        return
    # These new seeds must retain the original protocol, including its encoder.
    base.make_env=ORIGINAL_FACTORY
    selection=json.loads((SOURCE/'selection.json').read_text(encoding='utf-8'))['selected']['config_id']
    _,entropy,imitation=next(c for c in base.SCREEN_CONFIGS if c[0]==selection)
    config=MAPPOConfig(hidden_size=64,update_epochs=2,batch_size=64,
        route_entropy_coefficient=.01,flow_entropy_coefficient=entropy,
        plan_imitation_coefficient=imitation)
    seed=20261401
    observed_blocks=[]
    while True:
        estimate=max(observed_blocks[-3:] or [600.0])*1.2
        if not admission_allowed(time.time(),deadline.timestamp(),estimate,600):break
        started=time.monotonic()
        for arm in base.ARMS:
            directory=OUT/'additional_primary'/f'seed_{seed}'/arm
            base.train_run(directory,arm,seed,config,train,validation,150000,30000)
            status(phase='additional_primary_confirmation',last_completed=f'{seed}/{arm}')
        audit_checkpoint(OUT/'additional_primary'/f'seed_{seed}'/CACHED/'best.pt',validation,
            CACHED,OUT/'additional_primary_audit'/f'seed_{seed}',
            {'config_id':selection,'training_seed':seed},env_factory=ORIGINAL_FACTORY)
        observed_blocks.append(time.monotonic()-started)
        seed+=1
    status(phase='finalizing',additional_primary_seeds=seed-20261401)


def main():
    global TRAIN_SCENARIO_IDS
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started=datetime.fromisoformat(json.loads((SOURCE/'campaign_status.json').read_text(encoding='utf-8'))['started_at_utc'])
    deadline=started+timedelta(hours=6)
    prepare_manifest(deadline)
    if args.prepare_only:
        print(json.dumps({'prepared':True,'deadline_utc':deadline.isoformat()}));return 0
    status(status='running',phase='prepare_bank',deadline_utc=deadline.isoformat(),
        original_started_at_utc=started.isoformat(),requested_total_hours=6,
        error_type=None,error=None)
    scenarios=base.make_domain_randomized_trust_bank(24,20261021,'DSG')
    train,validation=scenarios[:12],scenarios[12:]
    TRAIN_SCENARIO_IDS=tuple(s.scenario_id for s in train)
    planner=BudgetedPlanner(OUT/'api',max_cny=2.0,max_requests=240)
    bank_passed=generate_bank(planner,train)
    wait_original()
    subprocess.run([sys.executable,str(ROOT/'scripts'/'analyze_mappo_capacity_stabilization_campaign.py')],
        cwd=ROOT,stdout=(OUT/'original_analysis_stdout.log').open('w',encoding='utf-8'),check=True)
    base.make_env=factory
    run_mechanism(train,validation,bank_passed)
    run_online_pilot(planner,validation,bank_passed)
    extend_primary(train,validation,deadline)
    status(status='experiments_complete',phase='awaiting_analysis',api=planner.ledger(),
           experiments_completed_at_utc=now().isoformat())
    print(json.dumps({'status':'experiments_complete','api':planner.ledger()},indent=2))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        status(status='failed',phase='failed',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
