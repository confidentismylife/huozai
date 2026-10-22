"""Frozen public-gate development confirmation, without PPO or LLM training."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from run_direction_release_mechanism_trial import (
    verify as verify_previous, verify_completion, read, save, sha, csv_read, csv_write, unique_output,
    PREP as OLD_PREP, MODES, METRICS, ZONE_NAMES, models_for, new_env, decide as old_decide,
    zone_fields, CACHE, cached_proposal, DynamicConflictScenario, paired_step, terminal_metrics,
    step_zone_accounting, reseed_continuation, preserve_base_release,
)
from fire_evac.timestamped_pipeline import ppo_action
from fire_evac.repeat_override_gate import public_signature, fit_consensus, consensus_decision, evidence_class

PREP = ROOT/'outputs/repeat_override_confirmation_preparation_20261007'
OLD_FULL = ROOT/'outputs/direction_release_mechanism_full_20261007T142136292884Z'
OLD_CF = ROOT/'outputs/direction_release_mechanism_continuations_20261007T142443133563Z'
POLICIES = ('no_advice','original_hint','preserve_always','first_only','public_consensus')
LABEL_POLICIES = ('preserve_always','first_only','public_consensus')
FUTURES = (2026100882,2026100883,2026100884)
PROTECTED = {
    'outputs/local_llm_selection.json':'302c6db2c857811dd41d1ed36248d1d9a71f3e32108cd870ecfe02aa520ff721',
    'outputs/third_round_development_status.json':'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0',
}


def verify():
    verify_previous()
    for p,h in PROTECTED.items():
        assert sha(ROOT/p)==h,p
    if PREP.exists():
        for p,h in read(PREP/'manifest.json')['sources'].items():
            assert sha(ROOT/p)==h,p


def context():
    return dict(previous=None,offers=0,overrides=0)


def update_context(ctx, public, detail, action):
    ctx['previous']=public
    ctx['offers']+=int(detail['eligible'])
    ctx['overrides']+=int(action!=detail['base_action'])


def choose(env, model, proposal, policy, ctx, fitted):
    _,base,hint,_,_,gate,status,mask=old_decide(env,model,proposal,'direction_hint_only')
    candidate,matching=preserve_base_release(base,hint,mask)
    eligible=candidate!=base
    public=env.snapshot.public_data()
    signature=public_signature(public,ctx['previous'],base,candidate,ctx['offers'],ctx['overrides'])
    supported,reason=consensus_decision(fitted,signature)
    first=ctx['offers']==0
    if policy=='no_advice': action=base
    elif policy=='original_hint': action=hint
    elif policy=='preserve_always': action=candidate
    elif policy=='first_only': action=candidate if first else base
    elif policy=='public_consensus': action=candidate if supported else base
    else: raise ValueError(policy)
    assert mask[action]
    if eligible: assert gate is not None and gate.accepted
    return int(action),dict(base_action=int(base),hint_action=int(hint),candidate_action=int(candidate),
        eligible=bool(eligible),signature=signature,offers_before=ctx['offers'],overrides_before=ctx['overrides'],
        first_only_accept=bool(first),consensus_accept=bool(supported),consensus_reason=reason,
        matching=matching,gate_status=status,legal_actions=list(map(int,np.flatnonzero(mask))))


def fit_archived():
    verify_previous()
    verify_completion(OLD_FULL);verify_completion(OLD_CF)
    selected=[json.loads(x) for x in (OLD_FULL/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    states={r['state_id']:r for r in selected}
    cfs={}
    for r in csv_read(OLD_CF/'terminal_rollouts.csv'):
        if r['future_policy']=='no_advice' and int(r['hold_steps'])==2:
            cfs[(r['state_id'],int(r['forced_action']),int(r['future_seed']))]=r
    samples=[];memory=defaultdict(context);prefixes=defaultdict(list)
    cases={r['case_id'] for r in selected}
    with (OLD_FULL/'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for t in map(json.loads,f):
            cid=t['case_id']
            if cid not in cases:continue
            ctx=memory[cid];public=t['public_snapshot'];sid=f"{cid}|{t['source_step']}"
            if sid in states:
                s=states[sid]
                sig=public_signature(public,ctx['previous'],s['base_action'],s['matched_release_action'],ctx['offers'],ctx['overrides'])
                futures=[s['population_seed']+770000]+([] if s['behavior']=='rule' else list((2026100782,2026100783,2026100784)))
                deltas=[]
                for seed in futures:
                    b=cfs[(sid,s['base_action'],seed)];a=cfs[(sid,s['matched_release_action'],seed)]
                    deltas.append(dict(incap_delta=float(a['incapacitation_rate'])-float(b['incapacitation_rate']),
                                       fed_delta=float(a['mean_cumulative_fed'])-float(b['mean_cumulative_fed'])))
                # Identity is offline deduplication only, never an online feature.
                identity=[s['scenario_id'],s['behavior'],s['population_seed'],s['training_seed'],s['source_step'],
                          prefixes[cid],t['observation'],s['past_zone_fed']]
                offline_identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                samples.append(dict(state_id=sid,signature=sig,offline_identity=offline_identity,
                                    offers_before=ctx['offers'],paired_outcomes=deltas))
            prefixes[cid].append([t['observation'],t['requested_action'],t['zone_fed_increments']])
            ctx['previous']=public
            if t['requested_action']!=t['base_action']:
                ctx['offers']+=1;ctx['overrides']+=1
    assert len(samples)==44
    fitted=fit_consensus(samples)
    return fitted,samples


def prepare():
    assert not PREP.exists()
    fitted,samples=fit_archived()
    old=read(OLD_PREP/'manifest.json')
    sources={p:sha(ROOT/p) for p in (
        'scripts/run_repeat_override_confirmation.py','src/fire_evac/repeat_override_gate.py','tests/test_repeat_override_gate.py',
        'outputs/direction_release_mechanism_preparation_20261007/manifest.json',
        OLD_FULL.relative_to(ROOT).as_posix()+'/completion_manifest.json',
        OLD_CF.relative_to(ROOT).as_posix()+'/completion_manifest.json',
        'outputs/behavior_pipeline/behavior_model.json',*PROTECTED)}
    for p in old['checkpoints'].values():sources[Path(p).relative_to(ROOT).as_posix()]=sha(p)
    for rec in old['scene_records']:
        p=CACHE/'responses'/(rec['scene']['scenario_id']+'.json')
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    PREP.mkdir()
    save(PREP/'frozen_public_gate.json',fitted)
    save(PREP/'archived_fit_evidence.json',samples)
    for p in ('frozen_public_gate.json','archived_fit_evidence.json'):
        sources[(PREP/p).relative_to(ROOT).as_posix()]=sha(PREP/p)
    manifest=dict(scope='Development confirmation on fresh people in five selected parameter scenarios sharing one graph; no independent building or final qualification.',
        scene_records=old['scene_records'],checkpoints=old['checkpoints'],training_seeds=old['training_seeds'],
        behaviors=old['behaviors'],modes=list(MODES),policies=list(POLICIES),
        population_prefix=1497100000,population_formula='prefix + original_scene_index*100 + repetition',populations_per_scene=4,
        population_batches=20,hold_steps=2,full_episodes=2080,rule_episodes=80,case_order_seed=2026100791,
        full_wall_seconds_cap=900,selected_state_cap=500,counterfactual_rollout_cap=5000,counterfactual_wall_seconds_cap=1200,
        frozen_future_seeds=list(FUTURES),actual_future='population_seed+770000; paired by individual and absolute step',
        gate_fit_states=44,unique_archived_states=fitted['unique_archived_states'],
        gate='Finite public signature consensus; no confidence calibration claim; unsupported, mixed, neutral or any known incapacitation increase rejects. No confirmation labels used in fitting.',
        first_only='At most first eligible proposal-induced override attempt per episode; rejects later attempts, regardless of whether earlier attempts were applied.',
        public_features='Base and matched candidate actions; past offer/actual override counts; sensor step, sign/hold, current exit capacities, forecast source/onset; five public occupancies in bins5 people, smoke bins.005, previous-step occupancy changes bins5 people. Excludes scenario/population/checkpoint/behavior ids, hidden individuals and outcomes.',
        state_selection='Every eligible preserved-release proposal on preserve_always, first_only and public_consensus actual paths; includes accepted and rejected opportunities. Shadow gates on preserve_always paths are primary common-state comparison; own-path metrics separate.',
        repeated_definition='offers_before>=1; also report previously executed overrides_before>=1 separately',
        counterfactual='Force base versus matched candidate once, then common no-advice PPO; also restore parent policy with separately updated public memory. Expected differences averaged over paired futures; realized restored actual action must exactly replay all parent suffix steps.',
        outcomes='Incapacitation prioritized at tolerance.001 then FED at.001. Confusion counts for beneficial/harmful/neutral, harm blocked and benefit rejected; neutral rejection not credited as harm prevented. Report per-future reversals.',
        pairing='Same fresh people and random table for every policy; rule run once per setting, reused only in paired descriptive comparisons.',
        adoption='No automatic adoption. Compare both gates against no_advice and preserve_always; keep all signs by scene/behavior/checkpoint/mode. No significance treating events/futures/weights as independent buildings.',
        tests_passed_new=5,tests_passed_related=12,new_ppo_training=0,new_llm_requests=0,reserved_final_scenarios_opened=0,sources=sources)
    save(PREP/'manifest.json',manifest)
    print(json.dumps(dict(stage='prepared',path=str(PREP),archived_cells=len(fitted['cells']),supported_cells=sum(c['all_supported'] for c in fitted['cells'].values()),unique_archived_states=fitted['unique_archived_states'],full_episodes=2080),ensure_ascii=False),flush=True)


def full_trials():
    verify();plan=read(PREP/'manifest.json');fitted=read(PREP/'frozen_public_gate.json');models=models_for(plan)
    cases=[]
    for rec in plan['scene_records']:
        for rep in range(4):
            pop=plan['population_prefix']+rec['scene_index']*100+rep
            for behavior in plan['behaviors']:
                for mode in MODES:
                    for seed in plan['training_seeds']:
                        for policy in POLICIES:cases.append((rec,pop,behavior,mode,seed,policy))
                    cases.append((rec,pop,behavior,mode,0,'pure_rule'))
    assert len(cases)==2080
    random.Random(plan['case_order_seed']).shuffle(cases)
    out=unique_output('repeat_override_confirmation_full');(out/'offline_states').mkdir()
    rows,selected=[],[];start=time.monotonic()
    with (out/'actual_step_traces.jsonl').open('w',encoding='utf-8') as stream:
        for rec,pop,behavior,mode,seed,policy in cases:
            assert time.monotonic()-start<900
            scene=DynamicConflictScenario(**rec['scene']);env=new_env(scene,behavior,pop,2)
            proposal=cached_proposal(read(CACHE/'responses'/(scene.scenario_id+'.json')),env.snapshot,mode==MODES[1])
            cid='|'.join(map(str,(scene.scenario_id,behavior,pop,seed,mode,policy)))
            ctx=context();counts=Counter();fed=np.zeros(5);people=np.zeros(5);incap=np.zeros(5,dtype=int)
            while True:
                s=env.snapshot;public=s.public_data();observation=env.policy_observation().tolist()
                if policy=='pure_rule':
                    action=env.constraints.rule_action(s)
                    detail=dict(base_action=action,hint_action=action,candidate_action=action,eligible=False,signature='',
                                offers_before=0,overrides_before=0,first_only_accept=False,consensus_accept=False,consensus_reason='rule',matching='rule',gate_status='rule',legal_actions=[])
                else:action,detail=choose(env,models[seed],proposal,policy,ctx,fitted)
                if policy in LABEL_POLICIES and detail['eligible']:
                    assert len(selected)<500
                    sid=f'{cid}|{s.step}';path=out/'offline_states'/f'{len(selected):04d}.pkl'
                    path.write_bytes(pickle.dumps(dict(env=copy.deepcopy(env),proposal=proposal,context=copy.deepcopy(ctx)),protocol=pickle.HIGHEST_PROTOCOL))
                    selected.append(dict(state_id=sid,case_id=cid,scenario_id=scene.scenario_id,behavior=behavior,population_seed=pop,training_seed=seed,source_mode=mode,policy=policy,
                        source_step=s.step,actual_action=action,**detail,past_zone_fed=fed.tolist(),past_zone_person_steps=people.tolist(),past_zone_incapacitations=incap.tolist(),
                        snapshot_path=path.relative_to(ROOT).as_posix(),snapshot_sha256=sha(path)))
                counts['proposed_override_steps']+=detail['eligible'];counts['advice_override_steps']+=action!=detail['base_action']
                counts['blocked_override_steps']+=detail['eligible'] and action==detail['base_action']
                counts['repeat_offer_steps']+=detail['eligible'] and ctx['offers']>=1
                counts['repeat_executed_steps']+=detail['eligible'] and ctx['offers']>=1 and action!=detail['base_action']
                result,account=step_zone_accounting(env,action,paired_step);_,_,term,trunc,info=result
                fed+=account['zone_fed_increments'];people+=account['zone_person_steps'];incap+=account['zone_new_incapacitations']
                counts['unavailable_entries']+=int(info['unavailable_corridor_entries']);counts['release_violations']+=not info['actual_release_within_limit'];counts['direction_violations']+=info['actual_direction_violation']
                stream.write(json.dumps(dict(case_id=cid,source_step=s.step,public_snapshot=public,observation=observation,
                    requested_action=action,**detail,applied_action=int(info['constraint_applied_action']),actual_joint_action=int(info['actual_joint_action']),
                    actual_sign=int(info['actual_sign_action']),actual_room_capacity=int(info['room_release_capacity']),actual_room_count=int(info['room_release_count']),zone_counts_after=info['zone_counts'],
                    branch_entries=env.unwrapped.domain_env.last_branch_flow.tolist(),unavailable_entries=int(info['unavailable_corridor_entries']),
                    release_valid=bool(info['actual_release_within_limit']),direction_violation=bool(info['actual_direction_violation']),**account),ensure_ascii=False)+'\n')
                update_context(ctx,public,detail,action)
                if term or trunc:break
            config=env.unwrapped.domain_env.config
            assert abs(fed.sum()/240-info['mean_cumulative_fed'])<1e-9
            rows.append(dict(case_id=cid,scenario_id=scene.scenario_id,behavior=behavior,population_seed=pop,training_seed=seed,source_mode=mode,policy=policy,
                **terminal_metrics(info,config),**zone_fields(fed,people,incap,config),**{k:int(counts[k]) for k in ('proposed_override_steps','advice_override_steps','blocked_override_steps','repeat_offer_steps','repeat_executed_steps','unavailable_entries','release_violations','direction_violations')}))
            env.close()
            if len(rows)%160==0:print(json.dumps(dict(stage='full',completed=len(rows),target=2080,selected_states=len(selected),elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
    csv_write(out/'episodes.csv',rows)
    with (out/'selected_states.jsonl').open('w',encoding='utf-8') as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save(out/'summary.json',dict(full_episodes=len(rows),selected_states=len(selected),preparation_sha256=sha(PREP/'manifest.json'),elapsed_seconds=time.monotonic()-start,
        new_ppo_training=0,new_llm_requests=0,reserved_final_scenarios_opened=0,violations={k:sum(r[k] for r in rows) for k in ('unavailable_entries','release_violations','direction_violations')}))
    verify();save(out/'completion_manifest.json',dict(state='complete',files={p.relative_to(ROOT).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(stage='full_complete',output=str(out),episodes=len(rows),selected_states=len(selected)),ensure_ascii=False),flush=True)
    return out


def continuations(full):
    verify();full=Path(full).resolve();verify_completion(full)
    plan=read(PREP/'manifest.json');fitted=read(PREP/'frozen_public_gate.json');models=models_for(plan)
    states=[json.loads(x) for x in (full/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    parents={r['case_id']:r for r in csv_read(full/'episodes.csv')}
    cases={r['case_id'] for r in states};expected=defaultdict(dict)
    with (full/'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['case_id'] in cases:expected[r['case_id']][r['source_step']]=r
    planned=sum((1 if s['behavior']=='rule' else 4)*4 for s in states)
    assert planned<=5000
    out=unique_output('repeat_override_confirmation_continuations');rows=[];exact=0;steps=0;start=time.monotonic()
    for state in states:
        path=ROOT/state['snapshot_path'];assert sha(path)==state['snapshot_sha256'];source=pickle.loads(path.read_bytes())
        futures=[(state['population_seed']+770000,'realized')]+([] if state['behavior']=='rule' else [(s,'fresh_future') for s in FUTURES])
        for seed,kind in futures:
            template=reseed_continuation(source['env'],seed)
            for future_policy in ('common_ppo','restore_parent'):
                for forced in (state['base_action'],state['candidate_action']):
                    assert time.monotonic()-start<1200 and len(rows)<5000
                    env=copy.deepcopy(template);ctx=copy.deepcopy(source['context']);fed=np.array(state['past_zone_fed']);people=np.array(state['past_zone_person_steps']);incap=np.array(state['past_zone_incapacitations']);counts=Counter();first=True
                    check=kind=='realized' and future_policy=='restore_parent' and forced==state['actual_action']
                    while True:
                        s=env.snapshot;public=s.public_data()
                        if first:
                            action=int(forced);detail=state
                        elif future_policy=='common_ppo':
                            action=ppo_action(models[state['training_seed']],env.policy_observation(),env.constraints.action_mask(s))[0]
                            detail=dict(eligible=False,base_action=action)
                        else:action,detail=choose(env,models[state['training_seed']],source['proposal'],state['policy'],ctx,fitted)
                        assert env.constraints.action_mask(s)[action]
                        result,account=step_zone_accounting(env,action,paired_step);_,_,term,trunc,info=result
                        fed+=account['zone_fed_increments'];people+=account['zone_person_steps'];incap+=account['zone_new_incapacitations']
                        counts['unavailable_entries']+=int(info['unavailable_corridor_entries']);counts['release_violations']+=not info['actual_release_within_limit'];counts['direction_violations']+=info['actual_direction_violation']
                        if check:
                            e=expected[state['case_id']][s.step]
                            assert action==e['requested_action']
                            for k,v in [('actual_joint_action','actual_joint_action'),('actual_sign_action','actual_sign'),('room_release_capacity','actual_room_capacity'),('room_release_count','actual_room_count')]:assert int(info[k])==e[v]
                            assert list(info['zone_counts'])==e['zone_counts_after']
                            assert np.allclose(account['zone_fed_increments'],e['zone_fed_increments'],atol=1e-12,rtol=0)
                            if not first:assert detail['signature']==e['signature'] and detail['offers_before']==e['offers_before'] and detail['overrides_before']==e['overrides_before']
                            steps+=1
                        update_context(ctx,public,detail,action);first=False
                        if term or trunc:break
                    config=env.unwrapped.domain_env.config;metrics=terminal_metrics(info,config)
                    assert abs(fed.sum()/240-info['mean_cumulative_fed'])<1e-9
                    fields=zone_fields(fed,people,incap,config)
                    if check:
                        parent=parents[state['case_id']]
                        for m in (*METRICS,'remaining','episode_end_seconds',*fields):
                            value=metrics[m] if m in metrics else fields[m]
                            assert abs(float(value)-float(parent[m]))<1e-12,(state['state_id'],m)
                        exact+=1
                    rows.append(dict(state_id=state['state_id'],future_seed=seed,future_kind=kind,forced_action=forced,future_policy=future_policy,**metrics,**fields,
                                     **{k:int(counts[k]) for k in ('unavailable_entries','release_violations','direction_violations')}))
                    env.close()
                    if len(rows)%200==0:print(json.dumps(dict(stage='counterfactual',completed=len(rows),planned=planned,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
    assert len(rows)==planned and exact==len(states)
    csv_write(out/'terminal_rollouts.csv',rows)
    save(out/'summary.json',dict(terminal_rollouts=len(rows),exact_realized_suffixes=exact,exact_suffix_control_steps=steps,full_manifest_sha256=sha(full/'completion_manifest.json'),
        preparation_sha256=sha(PREP/'manifest.json'),elapsed_seconds=time.monotonic()-start,new_ppo_training=0,new_llm_requests=0,reserved_final_scenarios_opened=0,
        violations={k:sum(r[k] for r in rows) for k in ('unavailable_entries','release_violations','direction_violations')}))
    verify();save(out/'completion_manifest.json',dict(state='complete',files={p.relative_to(ROOT).as_posix():sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(stage='continuations_complete',output=str(out),terminal_rollouts=len(rows),exact_suffixes=exact),ensure_ascii=False),flush=True)
    return out


def smoke():
    fitted,_=fit_archived();plan=read(OLD_PREP/'manifest.json');models=models_for(plan)
    # Old development population only; no confirmation people are opened here.
    scene=DynamicConflictScenario(**next(r['scene'] for r in plan['scene_records'] if r['scene']['scenario_id']=='AULL0024'))
    for policy in LABEL_POLICIES:
        env=new_env(scene,'rule',1477100150,2);ctx=context()
        proposal=cached_proposal(read(CACHE/'responses'/(scene.scenario_id+'.json')),env.snapshot,False)
        saved=None;expected=[]
        while True:
            s=env.snapshot;a,d=choose(env,models[602],proposal,policy,ctx,fitted);p=s.public_data()
            if d['eligible'] and saved is None:saved=(copy.deepcopy(env),copy.deepcopy(ctx),d,a)
            result,_=step_zone_accounting(env,a,paired_step)
            if saved is not None:expected.append((s.step,a))
            update_context(ctx,p,d,a)
            if result[2] or result[3]:break
        if saved:
            e,c,detail,forced=saved;found=[];first=True
            while True:
                s=e.snapshot;p=s.public_data()
                if first:a,d=forced,detail
                else:a,d=choose(e,models[602],proposal,policy,c,fitted)
                found.append((s.step,a));r=paired_step(e,a);update_context(c,p,d,a);first=False
                if r[2] or r[3]:break
            assert found==expected
            e.close()
        env.close()
    print(json.dumps(dict(stage='smoke_passed',old_population_only=True,gate_cells=len(fitted['cells']),supported_cells=sum(x['all_supported'] for x in fitted['cells'].values()))),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('smoke','prepare','run','continuations','verify'));parser.add_argument('--full');args=parser.parse_args()
    if args.stage=='smoke':smoke()
    elif args.stage=='prepare':prepare()
    elif args.stage=='verify':verify()
    elif args.stage=='continuations':
        assert args.full;continuations(args.full)
    else:continuations(full_trials())
