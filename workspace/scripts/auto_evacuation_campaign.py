"""Bounded, checkpointed development campaign with fixed conditional stages."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict,replace
from collections import Counter
import argparse,copy,csv,ctypes,hashlib,json,os,sys,time,traceback,urllib.request
import numpy as np
from auto_evacuation_core import OUTCOME_FIELDS,TRACE_FIELDS,labels,physics_features,fit_kernel,predict_kernel,gains,choose,calibrated_margin,continuation,AdviceActor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'outputs/evacuation_auto_campaign_20261005'
PROTOCOL=ROOT/'reports/疏散自动实验总方案_2026-10-05.md'
REPORT=ROOT/'reports/疏散自动实验阶段总报告_2026-10-05.md'
STATUS=ROOT/'outputs/third_round_development_status.json'
HISTORY=ROOT/'outputs/third_round_status_history/before_automatic_campaign_20261005.json'
PRIOR=ROOT/'outputs/observation_reversal_cache_audit_20261005'
FRESH=ROOT/'outputs/five_ppo_fresh_family_student_development_v1'
BOOT_SEEDS=(2026100531,2026100532,2026100533)
CAP=44738


def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_bytes(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(data);os.replace(tmp,p)
def write(p,v):atomic_bytes(p,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def np_save(p,**v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.npz');np.savez_compressed(tmp,**v);os.replace(tmp,p)
def csv_write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def event(kind,**v):
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'at':stamp(),'kind':kind,**v},ensure_ascii=False,allow_nan=False)+'\n')
def progress(stage,detail):
    s=read(OUT/'campaign_status.json');s.update(stage=stage,detail=detail,updated_at_utc=stamp(),pid=os.getpid());write(OUT/'campaign_status.json',s)
    print(json.dumps({'stage':stage,'detail':detail,'budget':read(OUT/'budget.json')},ensure_ascii=False),flush=True)
def reserve(kind):
    b=read(OUT/'budget.json')
    if b['simulations_started']>=CAP:raise RuntimeError('fixed campaign rollout budget exhausted')
    b['simulations_started']+=1;b['by_kind'][kind]=b['by_kind'].get(kind,0)+1;write(OUT/'budget.json',b)
def cached_input_check():
    m=read(OUT/'manifest.json')
    for p,h in m['frozen_sha256'].items():assert sha(ROOT/p)==h,p


def prepare():
    assert not OUT.exists() and not HISTORY.exists()
    from finalize_set_aware_distillation import check_completion
    from observation_reversal_audit import preserve
    assert check_completion(PRIOR.name,STATUS)==36
    checks=preserve(ROOT/'outputs/third_round_status_history/before_observation_reversal_audit_20261005.json')
    from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank
    from run_proposal_gain_factorial import signature,scenario_records
    oldrecords=[]
    for p in sorted(set((ROOT/'outputs').rglob('development_manifest.json'))|set((ROOT/'outputs').rglob('*scenarios.json'))):oldrecords+=scenario_records(read(p))
    oldsignatures={signature(r) for r in oldrecords}
    banks={b:[asdict(s) for s in make_domain_randomized_trust_bank(n,seed,prefix)] for b,n,seed,prefix in (
        ('train',60,2026100511,'AUTR'),('calibration',24,2026100512,'AUCA'),('test',48,2026100513,'AUTE'),('llm',48,2026100514,'AULL'))}
    records=sum(banks.values(),[]);assert len({signature(r) for r in records})==180 and not {signature(r) for r in records}&oldsignatures
    for b,bank in banks.items():assert len(set(Counter(r['conflict_type'] for r in bank).values()))==1
    registry={}
    for i,r in enumerate(records):registry[r['scenario_id']]={'population':[1100000000+i*1000+p for p in range(2)],
        'planning':[[1200000000+i*10000+p*1000+f for f in range(6)] for p in range(2)],
        'actual':[[1300000000+i*10000+p*1000+f for f in range(6)] for p in range(2)]}
    oldseeds=set()
    for p in read(FRESH/'development_manifest.json')['source_input_sha256']:
        if Path(p).suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for key in ('population_seed','population_seeds','planning_seeds','evaluation_seeds','future_seeds'):
                    if key in z:oldseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    for e in read(FRESH/'development_manifest.json')['seed_registry']:
        for k in ('population_seeds','planning_seeds','evaluation_seeds'):oldseeds.update(map(int,np.asarray(e[k]).reshape(-1)))
    newseeds={int(x) for e in registry.values() for v in e.values() for x in np.asarray(v).reshape(-1)}
    newseeds|={1050000000+p for p in range(2)}|{1150000000+p*1000+f for p in range(2) for f in range(6)}
    assert not (newseeds|{s+9000001 for s in newseeds})&(oldseeds|{s+9000001 for s in oldseeds})
    base=read(FRESH/'development_manifest.json');scene={r['scenario_id']:r for bank in base['banks'].values() for r in bank}
    factors=['north_exit_capacity','south_exit_capacity','growth_rate','onset_step','background_hazard']
    cells=[]
    for i in range(32):
        bits=[(i>>(4-k))&1 for k in range(5)];s=scene['F5TR0006'].copy();s.update({key:scene['F5TE0018'][key] if bit else scene['F5TR0006'][key] for key,bit in zip(factors,bits)})
        s.update(scenario_id='AUM'+''.join(map(str,bits)));cells.append({'bits':bits,'scene':s})
    group=[0,24,7,31];rng=np.random.default_rng(2026100515);order=list(rng.permutation(group))+list(rng.permutation([i for i in range(32) if i not in group]))
    frozen={ROOT/PROTOCOL.relative_to(ROOT),Path(__file__),ROOT/'scripts/auto_evacuation_core.py',ROOT/'tests/test_auto_evacuation_core.py',
        ROOT/'scripts/verify_auto_evacuation_campaign.py',ROOT/'scripts/write_auto_evacuation_campaign_report.py'}
    frozen.update(ROOT/'src'/p.relative_to(ROOT/'src') for p in (ROOT/'src/fire_evac').glob('*.py'))
    frozen.update(Path(r['path']) for r in base['frozen_ppo_models'])
    frozen.update((ROOT/'outputs/guarded_confirmatory_v2/models').glob('*.zip'))
    frozen.update([ROOT/'outputs/guarded_confirmatory_v2/metadata.json',ROOT/'outputs/guarded_confirmatory_v2/training_summary.csv',ROOT/'outputs/behavior_pipeline/behavior_model.json'])
    OUT.mkdir();HISTORY.write_bytes(STATUS.read_bytes());(OUT/'previous_status_exact_bytes.json').write_bytes(STATUS.read_bytes())
    write(OUT/'manifest.json',{'role':'bounded_automatic_development_research_campaign','created_at_utc':stamp(),'date':'2026-10-05',
        'banks':banks,'registry':registry,'factors':factors,'mechanism_cells':cells,'mechanism_order':list(map(int,order)),
        'group_cells':group,'mechanism_population_seeds':[1050000000,1050000001],
        'mechanism_future_seeds':[[1150000000+p*1000+f for f in range(6)] for p in range(2)],
        'pure_models':base['frozen_ppo_models'],'prior_margin':base['prior_margin'],'bootstrap_seeds':list(BOOT_SEEDS),
        'new_matrix_fit_cap':4,'simulation_cap':CAP,'llm_request_cap':96,'new_ppo_updates':0,
        'frozen_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(frozen)},'historical_scenarios_checked':len(oldrecords),
        'new_scene_signatures_disjoint':True,'new_seed_streams_disjoint':True,'formal_final_bank':False,'qualification_thresholds_changed':False,
        'previous_status_sha256':sha(HISTORY),'prior_preservation':checks,'prior_completion_sha256':sha(PRIOR/'completion_manifest.json'),
        'outcome_fields':list(OUTCOME_FIELDS),'trace_fields':list(TRACE_FIELDS),'automation_id':'automation',
        'promotion_is_not_original_shield_qualification':True})
    write(OUT/'budget.json',{'simulations_started':0,'simulations_completed':0,'by_kind':{},'matrix_fits':0,'mean_reference_estimates':0,'llm_requests':0,'ppo_updates':0})
    write(OUT/'campaign_status.json',{'state':'ready','stage':'prepared','detail':'frozen plan and fresh development banks','updated_at_utc':stamp(),'pid':None,'run_attempts':0,'consecutive_failures':0})
    write(OUT/'setup_events.json',{'heartbeat_interval_minutes':30,'heartbeat_created':True,'first_automation_create_missing_destination_repaired':True,
        'optional_sklearn_unavailable_numpy_models_used':True,'no_dependencies_installed':True})
    old=read(STATUS);write(STATUS,{'stage':'automatic_campaign_prepared','updated_date':'2026-10-05','updated_at_utc':stamp(),
        'formal_round3_training_started':False,'formal_round3_final_bank_generated':False,'hard_shield_qualification_passed':False,
        'qualification_thresholds_changed':False,'new_simulations':0,'new_fits':0,'new_ppo_training_seeds':0,'new_llm_calls':0,
        'campaign_status':(OUT/'campaign_status.json').relative_to(ROOT).as_posix(),'manifest':(OUT/'manifest.json').relative_to(ROOT).as_posix(),
        'report':REPORT.relative_to(ROOT).as_posix(),'previous_stage_status':HISTORY.relative_to(ROOT).as_posix(),'previous_status_sha256':sha(HISTORY),
        'previous_cached_reversal_stage':old,'automation_id':'automation','scope':'bounded automatic development chain, no formal qualification'})
    event('prepared',simulation_cap=CAP);print('Prepared automatic campaign',flush=True)


def make_env(scenes,temperature=1.,noise='none',mode='none',advice=None):
    from fire_evac.behavior import CalibratedLogitBehavior
    from fire_evac.dynamic_conflicts import DynamicConflictScenario,GymDynamicConflictEnv
    from fire_evac.confirmatory import ConfirmatoryJointControlWrapper,NOISE_LEVELS
    from fire_evac.terminal_shield import ObservableHistoryWrapper
    records=[s if isinstance(s,DynamicConflictScenario) else DynamicConflictScenario(**s) for s in scenes]
    return ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(
        CalibratedLogitBehavior(ROOT/'outputs/behavior_pipeline/behavior_model.json',temperature),records),mode,advice,NOISE_LEVELS[noise]))


def actors(manifest):
    import torch
    from stable_baselines3 import PPO
    from fire_evac.ppo_reference_planning import FrozenPPOContinuation,policy_parameter_sha256
    torch.set_num_threads(1);result=[]
    for r in manifest['pure_models']:
        model=PPO.load(r['path'],device='cpu');assert policy_parameter_sha256(model)==r['parameter_sha256'];result.append(FrozenPPOContinuation(r['training_seed'],model))
    return result


def episode_unit(path,env,scene,population_seeds,future_seeds,actorlist,worlds=False,full_trace=False,kind='planning',selection=None):
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    path=Path(path)
    if path.exists():return load(path)
    part=path.with_suffix('.partial.npz');shape=(2,2 if worlds else 1,len(actorlist),6)
    if part.exists():z=load(part);risk=z['risks'];done=z['done'];observations=z['observations'];traces=[];keys=[]
    else:risk=np.full(shape+(11,),np.nan);done=np.zeros(shape,bool);observations=np.full((2,49),np.nan);traces=[];keys=[]
    if full_trace and part.exists():
        for i,key in enumerate(z['trace_keys']):keys.append(tuple(map(int,key)));traces.append(z['execution_trace'][z['trace_offsets'][i]:z['trace_offsets'][i+1]])
    marker=path.with_suffix('.started.json')
    if not marker.exists():write(marker,{'started_at_utc':stamp(),'planned_rollouts':int(np.prod(shape)),'scene':scene,'kind':kind,'selection_sha256':None if selection is None else sha(selection)})
    for p in range(2):
        for w in range(shape[1]):
            template=scene.copy()
            if worlds:template['source_branch']=w
            x,_=env.reset(seed=int(population_seeds[p]),options={'scenario':DynamicConflictScenario(**template)})
            if np.isnan(observations[p]).all():observations[p]=x
            elif worlds:np.testing.assert_array_equal(x,observations[p])
            for j,actor in enumerate(actorlist):
                for f in range(6):
                    if done[p,w,j,f]:continue
                    reserve(kind);out,tr=continuation(env,x,actor,int(future_seeds[p][f]),full_trace)
                    risk[p,w,j,f]=out;done[p,w,j,f]=True
                    if full_trace:keys.append((p,w,j,f));traces.append(tr)
                    b=read(OUT/'budget.json');b['simulations_completed']+=1;write(OUT/'budget.json',b)
                    payload={'risks':risk,'done':done,'observations':observations,'population_seeds':np.asarray(population_seeds),'future_seeds':np.asarray(future_seeds)}
                    if full_trace:payload.update(trace_keys=np.array(keys),trace_offsets=np.cumsum([0]+[len(t) for t in traces]),execution_trace=np.concatenate(traces))
                    np_save(part,**payload)
    assert done.all() and np.isfinite(risk).all()
    if worlds:np.testing.assert_array_equal(observations[0],observations[1])
    os.replace(part,path);event('unit_complete',unit=path.relative_to(OUT).as_posix(),rollouts=int(done.sum()),sha256=sha(path));return load(path)


def parity(manifest,actorlist):
    marker=OUT/'instrumentation_parity.json'
    if marker.exists():return
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    from fire_evac.ppo_reference_planning import evaluate_continuation
    from run_source_world_planning import vector
    scene=manifest['mechanism_cells'][0]['scene'];env=make_env([scene]);x,_=env.reset(seed=1050090000,options={'scenario':DynamicConflictScenario(**scene)})
    reserve('instrumentation_check');old=vector(evaluate_continuation(env,x,actorlist[0],1150090000));b=read(OUT/'budget.json');b['simulations_completed']+=1;write(OUT/'budget.json',b)
    reserve('instrumentation_check');new,tr=continuation(env,x,actorlist[0],1150090000,True);b=read(OUT/'budget.json');b['simulations_completed']+=1;write(OUT/'budget.json',b)
    np.testing.assert_array_equal(old,new);assert len(tr)==new[5] and tr[:,37].sum()==round(new[0]*240)
    write(marker,{'passed':True,'outcome':new.tolist(),'trace_steps':len(tr),'new_simulations':2,'comparison':'original helper versus instrumented same episode, exact11 metrics','created_at_utc':stamp()});env.close()


def mechanism(manifest,actorlist):
    if (OUT/'mechanism_summary.json').exists():return read(OUT/'mechanism_summary.json')
    parity(manifest,actorlist);env=make_env([c['scene'] for c in manifest['mechanism_cells']]);selected=[actorlist[0],actorlist[2]]
    try:
        for n,index in enumerate(manifest['mechanism_order']):
            cell=manifest['mechanism_cells'][index];progress('mechanism_group' if n<4 else 'mechanism_full_factorial',{'completed_cells':n,'total_cells':32,'next':cell['scene']['scenario_id']})
            episode_unit(OUT/'mechanism'/f"{cell['scene']['scenario_id']}.npz",env,cell['scene'],manifest['mechanism_population_seeds'],manifest['mechanism_future_seeds'],selected,True,True,'mechanism')
            if n==3:event('group192_complete',scientific_rollouts=192)
    finally:env.close()
    rows=[];cellvalues={};worldrows=[];episodes=[]
    for cell in manifest['mechanism_cells']:
        z=load(OUT/'mechanism'/f"{cell['scene']['scenario_id']}.npz");l,y=labels(z['risks']);cellvalues[tuple(cell['bits'])]=y[1,1]
        popM=[labels(z['risks'][p:p+1])[1][1,1] for p in range(2)]
        rows.append({'cell':cell['scene']['scenario_id'],**dict(zip(manifest['factors'],cell['bits'])),'R_gain_pp':float(y[0,1]*100),'M_gain_pp':float(y[1,1]*100),
            'population0_M_pp':float(popM[0]*100),'population1_M_pp':float(popM[1]*100),
            'wait601':float(z['risks'][:,:,0,:,7].mean()),'wait603':float(z['risks'][:,:,1,:,7].mean())})
        for w in range(2):worldrows.append({'cell':cell['scene']['scenario_id'],'source_world':w,'loss601_pp':float(l[w,0]*100),'loss603_pp':float(l[w,1]*100),'gain_pp':float((l[w,0]-l[w,1])*100)})
        for ix,key in enumerate(z['trace_keys']):
            p,w,j,f=map(int,key);tr=z['execution_trace'][z['trace_offsets'][ix]:z['trace_offsets'][ix+1]]
            assert tr.shape[1]==len(TRACE_FIELDS);assert tr[:,37].sum()==round(z['risks'][p,w,j,f,0]*240)
            initialwait=np.flatnonzero(tr[:,1]==8);incsteps=np.flatnonzero(tr[:,37]>0)
            episodes.append({'cell':cell['scene']['scenario_id'],'population':p,'source_world':w,'ppo':601 if j==0 else 603,'future':f,
                'loss_pp':float(z['risks'][p,w,j,f,0]*100),'wait_requests':int((tr[:,1]==8).sum()),
                'first_wait':int(initialwait[0]) if len(initialwait) else -1,'first_incapacitation':int(incsteps[0]) if len(incsteps) else -1,
                'actual_released_people':int(tr[:,36].sum()),'north_occupancy_person_steps':int(tr[:,19].sum()),'south_occupancy_person_steps':int(tr[:,20].sum()),
                'room_occupancy_person_steps':int(tr[:,16:18].sum()),'junction_occupancy_person_steps':int(tr[:,18].sum()),
                'fed_increment_sum':float(tr[:,39].sum()),'steps':len(tr)})
    effects=[]
    for k,factor in enumerate(manifest['factors']):
        changes=[];flips=0
        for bits,v in cellvalues.items():
            if bits[k]:continue
            other=list(bits);other[k]=1;delta=cellvalues[tuple(other)]-v;changes.append(delta)
            flips+=bool((v<-.001 and cellvalues[tuple(other)]>.001) or (v>.001 and cellvalues[tuple(other)]<-.001))
        effects.append({'factor':factor,'mean_M_change_pp':float(np.mean(changes)*100),'min_M_change_pp':float(np.min(changes)*100),'max_M_change_pp':float(np.max(changes)*100),'strict_sign_flips_of16':flips})
    robustpos=[r for r in rows if min(r['M_gain_pp'],r['population0_M_pp'],r['population1_M_pp'])>.1]
    robustneg=[r for r in rows if max(r['M_gain_pp'],r['population0_M_pp'],r['population1_M_pp'])<-.1]
    interaction=[]
    for mask in range(1,32):
        included=[k for k in range(5) if mask&(1<<k)]
        val=np.mean([v*np.prod([1 if bits[k] else -1 for k in included]) for bits,v in cellvalues.items()])*2*100
        interaction.append({'factor_mask':mask,'factors':'+'.join(manifest['factors'][k] for k in included),'order':len(included),'factorial_contrast_pp':float(val)})
    csv_write(OUT/'mechanism_cells.csv',rows);csv_write(OUT/'mechanism_worlds.csv',worldrows);csv_write(OUT/'mechanism_trace_episodes.csv',episodes)
    csv_write(OUT/'mechanism_factor_effects.csv',effects);csv_write(OUT/'mechanism_all_interactions.csv',interaction)
    s={'cells':32,'scientific_rollouts':1536,'instrumentation_rollouts':2,'matched_populations':2,'worlds':2,'policies':[601,603],
        'positive_cells':len(robustpos),'negative_cells':len(robustneg),'proceed_to_students':bool(robustpos and robustneg),
        'group192':{r['cell']:r for r in rows if r['cell'] in [manifest['mechanism_cells'][i]['scene']['scenario_id'] for i in manifest['group_cells']]},
        'all_cells':rows,'factor_effects':effects,'all_interactions':interaction,'trace_steps':sum(r['steps'] for r in episodes),'completed_at_utc':stamp(),
        'causal_scope':'controlled simulator parameter changes within this fixed range, two populations, path-dependent random consumption; not real-fire causal validation'}
    write(OUT/'mechanism_summary.json',s);event('mechanism_complete',proceed=s['proceed_to_students']);return s


def collect_bank(block,manifest,actorlist):
    folder=OUT/f'{block}_planning';env=make_env(manifest['banks'][block])
    try:
        for i,scene in enumerate(manifest['banks'][block]):
            e=manifest['registry'][scene['scenario_id']];progress(block+'_planning',{'completed_families':i,'total':len(manifest['banks'][block])})
            episode_unit(folder/f"{scene['scenario_id']}.npz",env,scene,e['population'],e['planning'],actorlist,True,False,block+'_planning')
    finally:env.close()


def trained_models(manifest):
    folder=OUT/'models';folder.mkdir(exist_ok=True);record=folder/'frozen_models.json'
    if record.exists():
        z=read(record)
        for p,h in z['sha256'].items():assert sha(folder/p)==h
        return {'direct':load(folder/'direct.npz'),'absolute':[load(folder/f'absolute_{seed}.npz') for seed in BOOT_SEEDS],'mean':load(folder/'mean.npz')['gains']}
    xx=[];yy=[];ll=[]
    for scene in manifest['banks']['train']:
        z=load(OUT/'train_planning'/f"{scene['scenario_id']}.npz");l,y=labels(z['risks']);f=z['observations'][0].copy();f[28:33]=0;xx.append(f);yy.append(y[1]);ll.append(l.reshape(-1))
    xx=np.array(xx);yy=np.array(yy);ll=np.array(ll);pf=np.array([physics_features(x) for x in xx]);b=read(OUT/'budget.json')
    if b['matrix_fits']!=0:raise RuntimeError('interrupted model fit requires preservation and explicit new attempt, fixed cap4')
    direct=fit_kernel(xx,yy[:,1:]);np_save(folder/'direct.npz',**direct);b['matrix_fits']+=1;write(OUT/'budget.json',b)
    absolute=[]
    for seed in BOOT_SEEDS:
        model=fit_kernel(pf,ll,seed);np_save(folder/f'absolute_{seed}.npz',**model);absolute.append(model);b['matrix_fits']+=1;write(OUT/'budget.json',b)
    mean=yy.mean(axis=0);mean[0]=0.;np_save(folder/'mean.npz',gains=mean);b['mean_reference_estimates']=1;write(OUT/'budget.json',b)
    write(record,{'created_at_utc':stamp(),'fits':4,'mean_reference_estimates':1,'training_families':[s['scenario_id'] for s in manifest['banks']['train']],
        'sha256':{p.name:sha(p) for p in folder.glob('*.npz')},'bootstrap_seeds':list(BOOT_SEEDS),'physics_feature_dimension':len(pf[0]),'no_calibration_or_test_used':True})
    return dict(direct=direct,absolute=absolute,mean=mean)


def freeze_gates(manifest,models):
    path=OUT/'frozen_gates.json'
    if path.exists():return read(path)
    predictions={k:[] for k in ('direct','absolute','mean')};targets=[]
    for scene in manifest['banks']['calibration']:
        z=load(OUT/'calibration_planning'/f"{scene['scenario_id']}.npz");targets.append(labels(z['risks'])[1][1]);pred=gains(z['observations'][0],models)
        for k in predictions:predictions[k].append(pred[k])
    margins={};residuals={}
    for k,pred in predictions.items():margin,res=calibrated_margin(pred,targets,manifest['prior_margin']);margins[k]=margin;residuals[k]=res.tolist()
    record={'created_at_utc':stamp(),'margins':margins,'residuals':residuals,'model_record_sha256':sha(OUT/'models/frozen_models.json'),'prior_floor':manifest['prior_margin'],'test_seen':False}
    write(path,record);event('gates_frozen',margins=margins);return record


def freeze_choices(block,manifest,models,gates,env=None):
    for scene in manifest['banks'][block]:
        path=OUT/(block+'_choices')/f"{scene['scenario_id']}.json"
        if path.exists():continue
        z=load(OUT/(block+'_planning')/f"{scene['scenario_id']}.npz");obs=z['observations'][0];pred=gains(obs,models)
        choices={'baseline':0}
        for name in ('direct','absolute','mean'):
            choices[name+'_point']=choose(pred[name]);choices[name+'_margin']=choose(pred[name],gates['margins'][name])
        for name in ('absolute_seed1','absolute_seed2','absolute_seed3'):
            choices[name+'_point']=choose(pred[name]);choices[name+'_margin']=choose(pred[name],gates['margins']['absolute'])
        write(path,{'created_at_utc':stamp(),'family':scene['scenario_id'],'predictions':{k:v.tolist() for k,v in pred.items()},'choices':choices,
            'planning_sha256':sha(OUT/(block+'_planning')/f"{scene['scenario_id']}.npz"),'gate_sha256':sha(OUT/'frozen_gates.json'),'actual_seen':False})


def actual_bank(block,manifest,actorlist):
    env=make_env(manifest['banks'][block])
    try:
        for i,scene in enumerate(manifest['banks'][block]):
            e=manifest['registry'][scene['scenario_id']];progress(block+'_actual',{'completed_families':i,'total':len(manifest['banks'][block])})
            episode_unit(OUT/(block+'_actual')/f"{scene['scenario_id']}.npz",env,scene,e['population'],e['actual'],actorlist,False,False,block+'_actual',OUT/(block+'_choices')/f"{scene['scenario_id']}.json")
    finally:env.close()


def family_ci(delta,kinds,seed=2026100541):
    rng=np.random.default_rng(seed);values=np.asarray(delta);kinds=np.asarray(kinds);draw=np.zeros(2000)
    for k in sorted(set(kinds)):
        v=values[kinds==k];draw+=v[rng.integers(len(v),size=(2000,len(v)))].sum(axis=1)
    return np.quantile(draw/len(values),[.025,.975])


def summarize_student(manifest):
    rows=[];predrows=[]
    for scene in manifest['banks']['test']:
        family=scene['scenario_id'];z=load(OUT/'test_actual'/f'{family}.npz');loss=z['risks'][:,0,:,:,0].mean(axis=(0,2));actual=loss[0]-loss
        ch=read(OUT/'test_choices'/f'{family}.json')
        for arm,index in ch['choices'].items():rows.append({'family':family,'kind':scene['conflict_type'],'arm':arm,'choice':index,'loss_pp':float(loss[index]*100),'delta_pp':float((loss[index]-loss[0])*100),'override':index!=0})
        for model,pred in ch['predictions'].items():
            for j in range(1,5):predrows.append({'family':family,'model':model,'candidate':601+j,'prediction_pp':pred[j]*100,'actual_gain_pp':float(actual[j]*100),'absolute_error_pp':float(abs(pred[j]-actual[j])*100)})
    summary=[]
    for arm in sorted({r['arm'] for r in rows}):
        rr=[r for r in rows if r['arm']==arm];delta=[r['delta_pp'] for r in rr];ci=family_ci(delta,[r['kind'] for r in rr]);over=[r for r in rr if r['override']]
        summary.append({'arm':arm,'mean_delta_pp':float(np.mean(delta)),'ci95_pp':ci.tolist(),'overrides':len(over),
            'improved_families':sum(d<-.1 for d in delta),'harmed_families':sum(d>1e-10 for d in delta),
            'override_benefit_precision':sum(r['delta_pp']<-.1 for r in over)/len(over) if over else None})
    mae={k:float(np.mean([r['absolute_error_pp'] for r in predrows if r['model']==k])) for k in ('absolute','direct','mean','zero','absolute_seed1','absolute_seed2','absolute_seed3')}
    primary=next(r for r in summary if r['arm']=='absolute_margin');proceed=primary['overrides']>0 and primary['ci95_pp'][1]<0 and mae['absolute']<mae['zero'] and primary['harmed_families']/48<=.05
    csv_write(OUT/'student_family_results.csv',rows);csv_write(OUT/'student_prediction_results.csv',predrows)
    opportunity_count=sum(r['model']=='zero' and r['actual_gain_pp']>.1 for r in predrows)
    recall={}
    for arm in sorted({r['arm'] for r in rows}):
        selected={(r['family'],601+r['choice']) for r in rows if r['arm']==arm and r['choice']!=0}
        tp=sum((r['family'],r['candidate']) in selected and r['actual_gain_pp']>.1 for r in predrows if r['model']=='zero')
        recall[arm]=tp/opportunity_count if opportunity_count else None
    record={'primary':primary,'all_arms':summary,'actual_gain_MAE_pp':mae,'proceed_to_robustness':bool(proceed),'heldout_families':48,
        'actual_beneficial_candidate_count':opportunity_count,'candidate_opportunity_recall':recall,
        'new_fits':4,'bootstrap_seeds':list(BOOT_SEEDS),'new_ppo_updates':0,'completed_at_utc':stamp(),'original_qualification_passed':False}
    write(OUT/'student_summary.json',record);event('students_complete',proceed=proceed);return record


def students(manifest,actorlist):
    if (OUT/'student_summary.json').exists():return read(OUT/'student_summary.json')
    collect_bank('train',manifest,actorlist);models=trained_models(manifest)
    collect_bank('calibration',manifest,actorlist);gates=freeze_gates(manifest,models)
    freeze_choices('calibration',manifest,models,gates);actual_bank('calibration',manifest,actorlist)
    collect_bank('test',manifest,actorlist);freeze_choices('test',manifest,models,gates);actual_bank('test',manifest,actorlist)
    return summarize_student(manifest)


def robustness(manifest,actorlist,models,gates):
    if (OUT/'robustness_summary.json').exists():return read(OUT/'robustness_summary.json')
    rows=[]
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    for ci,(name,temp,noise) in enumerate([('temperature08',.8,'none'),('temperature12',1.2,'none'),('light',1.,'light'),('medium',1.,'medium')]):
        env=make_env(manifest['banks']['test'],temp,noise)
        try:
            for i,scene in enumerate(manifest['banks']['test']):
                family=scene['scenario_id'];e=manifest['registry'][family];cp=OUT/'robustness_choices'/name/f'{family}.json'
                if not cp.exists():
                    # Two sensor seeds can give different initial observations under noise.
                    choices=[];obs=[]
                    for p in range(2):
                        x,_=env.reset(seed=e['population'][p],options={'scenario':DynamicConflictScenario(**scene)});obs.append(x.tolist());choices.append(choose(gains(x,models)['absolute'],gates['margins']['absolute']))
                    write(cp,{'created_at_utc':stamp(),'choices':choices,'observations':obs,'gate_sha256':sha(OUT/'frozen_gates.json'),'actual_seen':False})
                choice=read(cp);future=[[1400000000+ci*10000000+i*10000+p*1000+f for f in range(6)] for p in range(2)]
                progress('robustness_'+name,{'completed_families':i,'total':48})
                z=episode_unit(OUT/'robustness_actual'/name/f'{family}.npz',env,scene,e['population'],future,actorlist,False,False,'robustness_'+name,cp)
                for p,j in enumerate(choice['choices']):np.testing.assert_array_equal(z['observations'][p],choice['observations'][p])
                base=np.mean(z['risks'][:,0,0,:,0]);selected=np.mean([z['risks'][p,0,j,:,0].mean() for p,j in enumerate(choice['choices'])])
                rows.append({'condition':name,'family':family,'kind':scene['conflict_type'],'delta_pp':float((selected-base)*100),'overrides':sum(j!=0 for j in choice['choices'])})
        finally:env.close()
    summary=[]
    for condition in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==condition];d=[r['delta_pp'] for r in rr];summary.append({'condition':condition,'mean_delta_pp':float(np.mean(d)),
            'ci95_pp':family_ci(d,[r['kind'] for r in rr]).tolist(),'harmed_families':sum(x>1e-10 for x in d),'overrides':sum(r['overrides'] for r in rr)})
    proceed=all(r['overrides']>0 and r['mean_delta_pp']<0 and r['harmed_families']/48<=.05 for r in summary)
    csv_write(OUT/'robustness_family_results.csv',rows);record={'conditions':summary,'proceed_to_llm':bool(proceed),'new_fits':0,'completed_at_utc':stamp()};write(OUT/'robustness_summary.json',record);return record


def api_json(url,payload=None,timeout=60):
    body=None if payload is None else json.dumps(payload).encode('utf-8');req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)


def llm_advice(manifest):
    from fire_evac.dynamic_llm_rl import DynamicLLMAdvice
    path=OUT/'llm_advice_frozen.json'
    if path.exists():records=read(path)['advice'];return {k:DynamicLLMAdvice(v['target_branch'],v['confidence'],('local_llm',)) for k,v in records.items()}
    tags=api_json('http://127.0.0.1:11434/api/tags',timeout=10);match=next(x for x in tags['models'] if x['name']=='qwen3:4b');records={}
    schema={'type':'object','properties':{'target':{'type':'string','enum':['north','south']},'confidence':{'type':'number','minimum':0,'maximum':1}},'required':['target','confidence'],'additionalProperties':False}
    for i,scene in enumerate(manifest['banks']['llm']):
        family=scene['scenario_id'];cache=OUT/'llm_cache'/f'{family}.json'
        if cache.exists():records[family]=read(cache)['advice'];continue
        data={k:scene[k] for k in ('forecast_source_branch','growth_rate','onset_step','background_hazard','north_exit_capacity','south_exit_capacity','blocked_branch','blockage_end_step')}
        prompt='You advise a research evacuation simulator. Choose a direction using only the forecast, initial equal room occupancy (120 each), and map capacities below. Forecast source can be wrong. No smoke difference is visible initially. Consider smoke risk, throughput and temporary blockage. Return JSON target north/south and confidence 0..1. Do not invent observations. Data: '+json.dumps(data,sort_keys=True)
        last=None
        for attempt in range(2):
            b=read(OUT/'budget.json');assert b['llm_requests']<96;b['llm_requests']+=1;write(OUT/'budget.json',b);start=time.monotonic()
            try:
                response=api_json('http://127.0.0.1:11434/api/chat',{'model':'qwen3:4b','messages':[{'role':'user','content':prompt}],
                    'format':schema,'options':{'temperature':0,'seed':2026100551,'num_predict':96},'think':False,'stream':False},60)
                write(OUT/'llm_raw_responses'/f'{family}_attempt{attempt+1}.json',{'response':response,'prompt':prompt,'latency_seconds':time.monotonic()-start,'model_digest':match['digest']})
                parsed=json.loads(response['message']['content']);assert parsed['target'] in ('north','south') and 0<=float(parsed['confidence'])<=1
                advice={'target_branch':0 if parsed['target']=='north' else 1,'confidence':float(parsed['confidence'])}
                write(cache,{'advice':advice,'prompt':prompt,'model_digest':match['digest'],'created_at_utc':stamp(),'attempts':attempt+1});records[family]=advice;break
            except Exception as exc:
                last=exc;write(OUT/'llm_failures'/f'{family}_attempt{attempt+1}.json',{'error':repr(exc),'prompt':prompt,'created_at_utc':stamp()})
        if family not in records:raise RuntimeError(f'LLM advice failed twice for {family}; preserve failure, no rule substitution') from last
        progress('llm_advice',{'completed':i+1,'total':48})
    write(path,{'created_at_utc':stamp(),'advice':records,'model_digest':match['digest'],'actual_seen':False,'prompt_fields_exclude_true_source_and_type':True})
    return {k:DynamicLLMAdvice(v['target_branch'],v['confidence'],('local_llm',)) for k,v in records.items()}


def llm_comparison(manifest):
    if (OUT/'llm_summary.json').exists():return read(OUT/'llm_summary.json')
    from stable_baselines3 import PPO
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    advice=llm_advice(manifest);rows=[];arms=[('pure','pure_ppo','none'),('rule','rule_advice_ppo','rule'),('llm','raw_llm_ppo','llm'),('same_llm_actor_rule','raw_llm_ppo','rule')]
    for seed in range(601,606):
        for arm,checkpoint,mode in arms:
            actor=AdviceActor(PPO.load(ROOT/f'outputs/guarded_confirmatory_v2/models/seed_{seed}_{checkpoint}.zip',device='cpu'));actor.model.policy.set_training_mode(False)
            env=make_env(manifest['banks']['llm'],mode=mode,advice=advice)
            try:
                for i,scene in enumerate(manifest['banks']['llm']):
                    family=scene['scenario_id'];e=manifest['registry'][family];path=OUT/'llm_actual'/f'{seed}_{arm}_{family}.npz'
                    progress('llm_control_comparison',{'seed':seed,'arm':arm,'family_index':i,'total':48})
                    # Unit has two populations, six futures and one actor =12, using same streams across arms/seeds.
                    z=episode_unit(path,env,scene,e['population'],e['actual'],[actor],False,False,'llm_comparison',OUT/'llm_advice_frozen.json')
                    rows.append({'seed':seed,'arm':arm,'family':family,'kind':scene['conflict_type'],'loss_pp':float(z['risks'][...,0].mean()*100)})
            finally:env.close()
    paired=[]
    for seed in range(601,606):
        for scene in manifest['banks']['llm']:
            family=scene['scenario_id'];losses={r['arm']:r['loss_pp'] for r in rows if r['seed']==seed and r['family']==family}
            paired.append({'seed':seed,'family':family,'kind':scene['conflict_type'],'llm_minus_same_actor_rule_pp':losses['llm']-losses['same_llm_actor_rule'],
                'llm_minus_pure_pp':losses['llm']-losses['pure'],'llm_minus_rule_actor_pp':losses['llm']-losses['rule']})
    seedmeans={str(seed):float(np.mean([r['llm_minus_same_actor_rule_pp'] for r in paired if r['seed']==seed])) for seed in range(601,606)}
    familymeans=[float(np.mean([r['llm_minus_same_actor_rule_pp'] for r in paired if r['family']==scene['scenario_id']])) for scene in manifest['banks']['llm']]
    ci=family_ci(familymeans,[s['conflict_type'] for s in manifest['banks']['llm']],2026100552)
    record={'primary_mean_delta_pp':float(np.mean(familymeans)),'primary_ci95_pp':ci.tolist(),'per_training_seed_delta_pp':seedmeans,
        'improvement_signal':bool(ci[1]<0 and sum(v<0 for v in seedmeans.values())>=4),'new_ppo_updates':0,
        'same_controller_advice_ablation':True,'local_model':'qwen3:4b','completed_at_utc':stamp(),'real_fire_efficacy_established':False}
    csv_write(OUT/'llm_family_results.csv',rows);csv_write(OUT/'llm_paired_results.csv',paired);write(OUT/'llm_summary.json',record);return record


def pid_live(pid):
    if not pid:return False
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
    return bool(ok and code.value==259)


def run():
    manifest=read(OUT/'manifest.json');status=read(OUT/'campaign_status.json');lock=OUT/'run_lock.json'
    if status['state']=='complete':print('Campaign already complete',flush=True);return
    if lock.exists() and pid_live(read(lock).get('pid')):print('Existing campaign worker is running',flush=True);return
    if status['consecutive_failures']>=3:raise RuntimeError('three consecutive failures; preserve data and diagnose before continuation')
    write(lock,{'pid':os.getpid(),'created_at_utc':stamp()});status.update(state='running',run_attempts=status['run_attempts']+1,pid=os.getpid());write(OUT/'campaign_status.json',status)
    event('worker_started',pid=os.getpid(),attempt=status['run_attempts'])
    try:
        cached_input_check()
        act=actors(manifest);mech=mechanism(manifest,act);stop='mechanism_no_robust_reversal'
        if mech['proceed_to_students']:
            student=students(manifest,act);stop='student_failed_fixed_development_progression'
            if student['proceed_to_robustness']:
                models=trained_models(manifest);gates=read(OUT/'frozen_gates.json');rob=robustness(manifest,act,models,gates);stop='robustness_failed_fixed_progression'
                if rob['proceed_to_llm']:llm_comparison(manifest);stop='all_applicable_stages_complete'
        cached_input_check();progress('verifying_and_reporting',{'reason':stop})
        from verify_auto_evacuation_campaign import verify
        verification=verify(manifest);write(OUT/'verification.json',verification)
        from write_auto_evacuation_campaign_report import write_report
        write_report(stop)
        state=read(OUT/'campaign_status.json');state.update(state='complete',stage='complete',stop_reason=stop,completed_at_utc=stamp(),consecutive_failures=0);write(OUT/'campaign_status.json',state)
        finalize(manifest,stop);event('campaign_complete',reason=stop,budget=read(OUT/'budget.json'));print('Campaign complete: '+stop,flush=True)
    except Exception:
        status=read(OUT/'campaign_status.json');status.update(state='failed',consecutive_failures=status['consecutive_failures']+1,last_failure_at_utc=stamp());write(OUT/'campaign_status.json',status)
        failure=OUT/'failures'/f"attempt_{status['run_attempts']:03d}.json";write(failure,{'created_at_utc':stamp(),'traceback':traceback.format_exc(),'stage':status['stage'],'budget':read(OUT/'budget.json')});event('worker_failed',failure=failure.relative_to(OUT).as_posix());raise
    finally:
        if lock.exists():
            saved=read(lock);saved['released_at_utc']=stamp();saved['pid']=None;write(lock,saved)


def finalize(manifest,reason):
    from finalize_set_aware_distillation import check_completion
    from observation_reversal_audit import preserve
    assert check_completion(PRIOR.name,HISTORY)==36
    checks=preserve(ROOT/'outputs/third_round_status_history/before_observation_reversal_audit_20261005.json')
    running_history=ROOT/'outputs/third_round_status_history/before_automatic_campaign_complete_20261005.json'
    if not running_history.exists():running_history.write_bytes(STATUS.read_bytes())
    b=read(OUT/'budget.json');old=read(HISTORY)
    write(STATUS,{'stage':'automatic_campaign_complete','updated_date':'2026-10-05','updated_at_utc':stamp(),'stop_reason':reason,
        'report':REPORT.relative_to(ROOT).as_posix(),'summary':(OUT/'campaign_status.json').relative_to(ROOT).as_posix(),
        'manifest':(OUT/'manifest.json').relative_to(ROOT).as_posix(),'completion_manifest':(OUT/'completion_manifest.json').relative_to(ROOT).as_posix(),
        'previous_stage_status':HISTORY.relative_to(ROOT).as_posix(),'previous_stage_status_sha256':sha(HISTORY),'previous_cached_reversal_stage':old,
        'budget':b,'new_simulations':b['simulations_started'],'new_fits':b['matrix_fits'],'new_ppo_training_seeds':0,'new_llm_calls':b['llm_requests'],
        'hard_shield_qualification_passed':False,'qualification_thresholds_changed':False,'formal_round3_training_started':False,'formal_round3_final_bank_generated':False,
        'development_only':True,'automation_id':'automation','next_step':'review the fixed campaign conclusion; do not repeat viewed holdout as independent final qualification'})
    # Exclude mutable worker lock, event log and campaign status from sealed scientific ledger.
    excluded={'run_lock.json','events.jsonl','campaign_status.json','completion_manifest.json'}
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded and not p.name.endswith('.tmp')]
    paths+=[STATUS,HISTORY,running_history,PROTOCOL,REPORT]
    paths+=[ROOT/'scripts'/f for f in ('auto_evacuation_core.py','auto_evacuation_campaign.py','verify_auto_evacuation_campaign.py','write_auto_evacuation_campaign_report.py')]
    paths+=[ROOT/'tests/test_auto_evacuation_core.py']
    write(OUT/'completion_manifest.json',{'all_checks_passed':True,'role':'bounded_automatic_development_campaign_completion','budget':b,'stop_reason':reason,
        'prior_preservation':checks,'qualification_thresholds_unchanged':True,'new_ppo_updates':0,'created_at_utc':stamp(),
        'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}})
    for p,h in read(OUT/'completion_manifest.json')['files'].items():assert sha(ROOT/p)==h,p


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    else:parser.error('choose --prepare or --run')
