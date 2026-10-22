"""Independently recalculate diagnostic tables using cached tensors only."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import numpy as np
from observation_reversal_audit import ROOT, SOURCE, OUT, PAIR, FLOOR, sha, write, read, load, check_inputs


def table(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
def cl(v):return 'positive' if v>.001 else 'negative' if v<-.001 else 'neutral_or_small'


def main():
    assert not (OUT/'verification.json').exists();check_inputs()
    sm=read(SOURCE/'development_manifest.json');s=read(OUT/'summary.json');am=read(OUT/'analysis_manifest.json')
    assert sha(OUT/'summary.json')==am['summary_sha256'] and am['script_sha256']==sha(ROOT/'scripts/observation_reversal_audit.py')
    for p,h in am['files'].items():assert sha(OUT/p)==h
    train=load(SOURCE/'training_dataset.npz');model=load(SOURCE/'models/M_kernel.npz')
    entries={e['family']:e for e in sm['seed_registry']};cache={};scenes={r['scenario_id']:r for bank in sm['banks'].values() for r in bank}
    for e in sm['seed_registry']:
        z=load(SOURCE/f"{e['block']}_planning/{e['family']}.npz");cache[e['family']]=z
        means=np.zeros((2,5))
        for w in range(2):
            for j in range(5):means[w,j]=np.mean([z['planning_risks'][p,w,j,f,0] for p in range(2) for f in range(6)])
        supported=means[z['supported']]
        expected=np.array([[supported[:,0].max()-supported[:,j].max() for j in range(5)],
                           [np.min(supported[:,0]-supported[:,j]) for j in range(5)]])
        close(expected,z['pooled_targets']);close(means,z['pooled_world_losses'])
        assert z['supported'].all() and z['world_errors'].max()<=1e-6
        np.testing.assert_array_equal(z['observations'][0],z['observations'][1]);np.testing.assert_array_equal(z['observations'][0],z['observation'])
        masked=z['observation'].copy();masked[28:33]=0;np.testing.assert_array_equal(z['features'],masked)
        scene=scenes[e['family']]
        close(z['features'][21],np.float32(scene['growth_rate']/.05));close(z['features'][22],np.float32(scene['onset_step']/20))
        close(z['features'][5:10],np.full(5,np.float32(scene['background_hazard'])))
        close(z['features'][23:25],[scene['north_exit_capacity']/8,scene['south_exit_capacity']/8])
        np.testing.assert_array_equal(z['population_seeds'],e['population_seeds']);np.testing.assert_array_equal(z['planning_seeds'],e['planning_seeds'])
    feats=table('feature_comparison.csv');assert len(feats)==49
    ta,tb=cache[PAIR[0]],cache[PAIR[1]];expected=((tb['features'].astype(float)-ta['features'].astype(float))/model['std'])**2
    for i,r in enumerate(feats):
        assert int(r['index'])==i;close(float(r['train_raw']),ta['observation'][i]);close(float(r['test_raw']),tb['observation'][i]);close(float(r['squared_standardized_distance']),expected[i])
    changed=[i for i in range(49) if ta['observation'][i]!=tb['observation'][i]];assert s['changed_feature_indices']==changed
    world=table('pair_world_outcomes.csv');assert len(world)==20
    for r in world:
        z=cache[r['family']];w,j=int(r['world']),int(r['candidate_index']);loss=z['planning_risks'][...,0]
        baseline=loss[:,w,0,:];cand=loss[:,w,j,:];g=baseline-cand
        close(float(r['baseline_loss_pp']),baseline.mean()*100);close(float(r['candidate_loss_pp']),cand.mean()*100);close(float(r['gain_pp']),g.mean()*100)
        assert int(r['gain_positive_count'])==int((g>.001).sum()) and int(r['gain_negative_count'])==int((g<-.001).sum())
        for k,name in enumerate(sm['outcome_fields']):close(float(r['mean_'+name]),z['planning_risks'][:,w,j,:,k].mean())
    scopes={'pooled':(range(2),range(6)),'population0':([0],range(6)),'population1':([1],range(6)),
            'first3_futures':(range(2),range(3)),'last3_futures':(range(2),range(3,6)),
            'population0_first3':([0],range(3)),'population0_last3':([0],range(3,6)),
            'population1_first3':([1],range(3)),'population1_last3':([1],range(3,6))}
    split=table('pair_fixed_splits.csv');assert len(split)==90
    for r in split:
        z=cache[r['family']];pp,ff=scopes[r['scope']];j=int(r['candidate_index'])
        means=np.array([[np.mean([z['planning_risks'][p,w,k,f,0] for p in pp for f in ff]) for k in range(5)] for w in range(2)])
        R=means[:,0].max()-means[:,j].max();M=np.min(means[:,0]-means[:,j])
        close(float(r['R_pp']),R*100);close(float(r['M_pp']),M*100);assert r['R_class']==cl(R) and r['M_class']==cl(M)
    seedrows=table('pair_seed_gains.csv');assert len(seedrows)==240
    for r in seedrows:
        z=cache[r['family']];p,w,j,f=(int(r[k]) for k in ('population','world','candidate_index','future_index'))
        close(float(r['gain_pp']),(z['planning_risks'][p,w,0,f,0]-z['planning_risks'][p,w,j,f,0])*100)
        assert int(r['planning_seed'])==z['planning_seeds'][p,f] and int(r['population_seed'])==z['population_seeds'][p]
    terms=table('kernel_contributions.csv');assert len(terms)==192
    for target in ('R','M'):
        mm=load(SOURCE/f'models/{target}_kernel.npz');x=(tb['features'].astype(float)-mm['mean'])/mm['std']
        distance=np.linalg.norm(mm['train']-x,axis=1)**2;k=np.exp(-float(mm['gamma'])*distance)
        centered=k-np.dot(k,mm['weights'])-mm['kernel_row_mean']+float(mm['kernel_grand_mean'])
        pred=mm['intercept'].copy()
        for r in terms:
            if r['target']!=target:continue
            ix=list(train['families']).index(r['family']);j=int(r['candidate_index'])-1;contribution=centered[ix]*mm['coefficient'][ix,j]
            close(float(r['contribution_pp']),contribution*100);close(float(r['kernel_similarity']),k[ix]);pred[j]+=contribution
        expected=np.r_[0,pred];close(expected,read(SOURCE/f'test_choices/{PAIR[1]}.json')['predictions'][target+'_kernel'])
        close(expected*100,s['kernel'][target]['prediction_pp'])
        close(distance.min(),np.sum([float(r['squared_standardized_distance']) for r in feats]))
        assert str(train['families'][distance.argmin()])==PAIR[0]
    actual=load(SOURCE/f'test_actual/{PAIR[1]}.npz');ac=actual['evaluation_outcomes'];arows=table('test18_actual_outcomes.csv');assert len(arows)==5
    for r in arows:
        j=int(r['candidate_index']);loss=ac[:,j,:,0];gain=ac[:,0,:,0]-loss
        close(float(r['actual_loss_pp']),loss.mean()*100);close(float(r['actual_gain_pp']),gain.mean()*100)
        assert int(r['actual_gain_positive_count'])==int((gain>.001).sum()) and int(r['actual_gain_negative_count'])==int((gain<-.001).sum())
    erows=table('test18_actual_trace_episodes.csv');assert len(erows)==60
    for i,key in enumerate(actual['trace_keys']):
        p,j,f=map(int,key);tr=actual['execution_trace'][actual['trace_offsets'][i]:actual['trace_offsets'][i+1]];r=erows[i]
        assert (int(r['population']),int(r['candidate_index']),int(r['future_index']))==(p,j,f)
        assert len(tr)==int(r['steps'])==ac[p,j,f,5] and tr[0,1]==ac[p,j,f,9] and tr[0,2]==ac[p,j,f,10]
        close(tr[:,4].mean(),ac[p,j,f,6]);close(tr[:,5].mean(),ac[p,j,f,8]);assert (tr[:,1]==8).sum()==ac[p,j,f,7]
        assert int(r['unrepresented_joint_steps'])==((tr[:,3]==2)&(tr[:,4]>0)).sum()==(tr[:,2]==-1).sum()
    bystep=table('test18_actual_trace_by_step.csv');phases=table('test18_actual_trace_phases.csv');assert len(phases)==15
    for r in bystep:
        j=int(r['candidate_index']);t=int(r['elapsed']);matches=[]
        for i,key in enumerate(actual['trace_keys']):
            if key[1]==j:
                tr=actual['execution_trace'][actual['trace_offsets'][i]:actual['trace_offsets'][i+1]];matches.extend(tr[tr[:,0]==t])
        tr=np.array(matches);assert len(tr)==int(r['episodes_still_running']);close(tr[:,4].mean(),float(r['mean_capacity_fraction']))
        for sign,name in ((0,'north_sign_count'),(1,'south_sign_count'),(2,'neutral_sign_count')):assert (tr[:,3]==sign).sum()==int(r[name])
    for r in phases:
        j=int(r['candidate_index']);trs=[]
        for i,key in enumerate(actual['trace_keys']):
            if key[1]==j:trs.append(actual['execution_trace'][actual['trace_offsets'][i]:actual['trace_offsets'][i+1]])
        tr=np.concatenate(trs)
        if r['phase']=='before_growth':tr=tr[tr[:,0]<=scenes[PAIR[1]]['onset_step']]
        elif r['phase']=='growth_started':tr=tr[tr[:,0]>scenes[PAIR[1]]['onset_step']]
        assert len(tr)==int(r['step_count']);close(tr[:,4].mean(),float(r['mean_capacity_fraction_step_weighted']))
        for sign,name in ((0,'north_sign_steps'),(1,'south_sign_steps'),(2,'neutral_sign_steps')):assert (tr[:,3]==sign).sum()==int(r[name])
    broad=table('all_test_nearest_candidate_pairs.csv');assert len(broad)==192
    counts=Counter();famsets={t:set() for t in ('R','M')}
    for r in broad:
        z=cache[r['family']];x=(z['features'].astype(float)-model['mean'])/model['std'];d=((model['train']-x)**2).sum(axis=1);ix=int(d.argmin())
        assert r['nearest_training_family']==str(train['families'][ix]);close(float(r['similarity']),np.exp(-float(model['gamma'])*d[ix]))
        ti=('R','M').index(r['target']);j=int(r['candidate_index']);a,b=train['targets'][ix,ti,j],z['pooled_targets'][ti,j]
        flip=(a>.001 and b<-.001) or (a<-.001 and b>.001);assert (r['strict_sign_reversal']=='True')==flip
        close(float(r['nearest_label_pp']),a*100);close(float(r['test_label_pp']),b*100)
        if flip:counts[r['target']]+=1;famsets[r['target']].add(r['family'])
    assert dict(counts)==s['nearest_pair_reversal_counts']
    for t in famsets:assert sorted(famsets[t])==s['nearest_pair_reversal_families'][t]
    pairs=table('all_bank_pair_comparisons.csv');assert len(pairs)==1770 and len({(r['family_a'],r['family_b']) for r in pairs})==1770
    exact=near=0
    for r in pairs:
        a,b=(cache[r[k]]['features'].astype(float) for k in ('family_a','family_b'))
        eq=np.array_equal(a,b);ne=np.abs(a-b).max()<=1e-6;exact+=eq;near+=ne
        assert (r['exact_equal']=='True')==eq and (r['within_1e_minus6']=='True')==ne
        close(float(r['squared_standardized_distance']),np.sum(((a-b)/model['std'])**2))
    assert exact==s['all_bank_collision_audit']['exact_equal_count'] and near==s['all_bank_collision_audit']['within_1e_minus6_count']
    assert not (SOURCE/'train_actual').exists()
    assert s['new_simulations']==s['new_model_fits']==s['new_ppo_updates']==s['new_llm_calls']==0
    assert not s['causal_mechanism_identified'] and not s['qualification_thresholds_changed']
    write(OUT/'verification.json',{'all_checks_passed':True,'independent_reaggregation':True,'planning_families_checked':60,
        'pair_world_rows_checked':20,'fixed_split_rows_checked':90,'paired_seed_rows_checked':240,'feature_rows_checked':49,
        'kernel_contributions_checked':192,'actual_episodes_checked':60,'trace_steps_checked':len(actual['execution_trace']),
        'nearest_candidate_rows_checked':192,'all_bank_pairs_checked':1770,'analysis_manifest_sha256':sha(OUT/'analysis_manifest.json'),
        'verifier_sha256':sha(__file__),'new_simulations':0,'new_model_fits':0})
    print(json.dumps(read(OUT/'verification.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
