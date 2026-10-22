"""Independent cache-only verification of the automated campaign."""
from pathlib import Path
from collections import Counter
import csv,json
import numpy as np
from auto_evacuation_campaign import ROOT,OUT,PROTOCOL,read,sha,load


def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def independent_labels(risks):
    l=risks[...,0].mean(axis=(0,3));y=np.array([l[:,0].max()-l.max(axis=0),(l[:,0,None]-l).min(axis=0)]);return l,y
def independent_choice(a,margin):
    j=int(np.argmax(a));return j if j and a[j]>max(.001,margin) else 0


def verify(m):
    for p,h in m['frozen_sha256'].items():assert sha(ROOT/p)==h,p
    counts=Counter();trace_steps=0;units=0
    for p in OUT.rglob('*.npz'):
        if p.parent.name=='models' or p.name.endswith(('.partial.npz','.tmp.npz')):continue
        z=load(p)
        if 'risks' not in z:continue
        r=z['risks'];assert z['done'].all() and np.isfinite(r).all() and ((r[...,0]>=0)&(r[...,0]<=1)).all()
        assert r.shape[-1]==11 and r.shape[0]==2 and r.shape[-2]==6
        close(r[...,0]+r[...,2]+r[...,3],np.ones_like(r[...,0]))
        assert (r[...,5]>0).all() and (r[...,5]<=120).all()
        rel=p.relative_to(OUT).parts
        kind='mechanism' if rel[0]=='mechanism' else rel[0] if rel[0] in ('train_planning','calibration_planning','test_planning','calibration_actual','test_actual') else 'robustness_'+rel[1] if rel[0]=='robustness_actual' else 'llm_comparison'
        counts[kind]+=int(z['done'].sum());units+=1
        if 'execution_trace' in z:
            tr=z['execution_trace'];assert tr.shape[1]==40 and len(z['trace_offsets'])==len(z['trace_keys'])+1;trace_steps+=len(tr)
            for i,key in enumerate(z['trace_keys']):
                pop,w,j,f=map(int,key);t=tr[z['trace_offsets'][i]:z['trace_offsets'][i+1]];out=r[pop,w,j,f]
                np.testing.assert_array_equal(t[:,0],np.arange(len(t)));assert len(t)==out[5]
                close(t[:,4].mean(),out[6]);assert (t[:,1]==8).sum()==out[7];close(t[:,5].mean(),out[8])
                close(t[:,37].sum()/240,out[0]);close(t[:,38].sum()/240,out[2]);close(t[:,39].sum()/240,out[1])
                close(t[:,16:21].sum(axis=1)-t[:,21:26].sum(axis=1),t[:,37]+t[:,38])
                assert t[:,36].sum()<=240 and (t[:,36]>=0).all()
    budget=read(OUT/'budget.json');expected_completed=sum(counts.values())+2
    assert budget['simulations_completed']==expected_completed and budget['simulations_started']>=expected_completed
    assert budget['simulations_started']<=m['simulation_cap'] and budget['matrix_fits']<=4 and budget['llm_requests']<=96 and budget['ppo_updates']==0
    assert read(OUT/'instrumentation_parity.json')['passed']
    summary=read(OUT/'mechanism_summary.json');assert counts['mechanism']==1536 and summary['trace_steps']==trace_steps
    rows=table(OUT/'mechanism_cells.csv');worlds=table(OUT/'mechanism_worlds.csv');values={}
    for r in rows:
        z=load(OUT/'mechanism'/f"{r['cell']}.npz");l,y=independent_labels(z['risks']);close(float(r['R_gain_pp']),y[0,1]*100);close(float(r['M_gain_pp']),y[1,1]*100)
        close(float(r['wait603']),z['risks'][:,:,1,:,7].mean());close(float(r['wait601']),z['risks'][:,:,0,:,7].mean())
        bits=tuple(int(r[k]) for k in m['factors']);values[bits]=y[1,1]
        for p in range(2):close(float(r[f'population{p}_M_pp']),independent_labels(z['risks'][p:p+1])[1][1,1]*100)
        for w in range(2):
            rr=next(x for x in worlds if x['cell']==r['cell'] and int(x['source_world'])==w);close(float(rr['gain_pp']),(l[w,0]-l[w,1])*100)
    assert len(values)==32
    for row in table(OUT/'mechanism_factor_effects.csv'):
        k=m['factors'].index(row['factor']);change=[]
        for bits,v in values.items():
            if not bits[k]:other=list(bits);other[k]=1;change.append(values[tuple(other)]-v)
        close(float(row['mean_M_change_pp']),np.mean(change)*100);close(float(row['min_M_change_pp']),np.min(change)*100);close(float(row['max_M_change_pp']),np.max(change)*100)
    positive=sum(min(float(r['M_gain_pp']),float(r['population0_M_pp']),float(r['population1_M_pp']))>.1 for r in rows)
    negative=sum(max(float(r['M_gain_pp']),float(r['population0_M_pp']),float(r['population1_M_pp']))<-.1 for r in rows)
    assert summary['proceed_to_students']==bool(positive and negative)
    student_checked=False;robust_checked=False;llm_checked=False
    if (OUT/'student_summary.json').exists():
        student_checked=True;student=read(OUT/'student_summary.json');gates=read(OUT/'frozen_gates.json');models=read(OUT/'models/frozen_models.json')
        assert budget['matrix_fits']==4 and budget['mean_reference_estimates']==1
        for k,h in models['sha256'].items():assert sha(OUT/'models'/k)==h
        assert models['created_at_utc']<gates['created_at_utc']
        for block,n in (('train',60),('calibration',24),('test',48)):
            assert counts[block+'_planning']==n*120
            for scene in m['banks'][block]:
                family=scene['scenario_id'];path=OUT/(block+'_planning')/f'{family}.npz';z=load(path);np.testing.assert_array_equal(z['observations'][0],z['observations'][1])
                if block=='test':assert gates['created_at_utc']<read(path.with_suffix('.started.json'))['started_at_utc']
                if block!='train':
                    ch=read(OUT/(block+'_choices')/f'{family}.json');act=OUT/(block+'_actual')/f'{family}.npz'
                    assert ch['created_at_utc']<read(act.with_suffix('.started.json'))['started_at_utc'];assert ch['planning_sha256']==sha(path)
                    for model in ('direct','absolute','mean'):
                        pred=ch['predictions'][model];assert pred[0]==0 and ch['choices'][model+'_point']==independent_choice(pred,0)
                        assert ch['choices'][model+'_margin']==independent_choice(pred,gates['margins'][model])
        for key,margin in gates['margins'].items():
            residual=[]
            for scene in m['banks']['calibration']:
                family=scene['scenario_id'];y=independent_labels(load(OUT/'calibration_planning'/f'{family}.npz')['risks'])[1][1]
                pred=read(OUT/'calibration_choices'/f'{family}.json')['predictions'][key];residual.append(max(np.array(pred)[1:]-y[1:]))
            close(margin,max(.001,m['prior_margin'],float(np.quantile(residual,.95,method='higher'))))
        familyrows=table(OUT/'student_family_results.csv');predrows=table(OUT/'student_prediction_results.csv');assert len(familyrows)==48*13 and len(predrows)==48*4*7
        for row in familyrows:
            z=load(OUT/'test_actual'/f"{row['family']}.npz");l=z['risks'][:,0,:,:,0].mean(axis=(0,2));j=int(row['choice']);close(float(row['delta_pp']),(l[j]-l[0])*100)
        for row in predrows:
            z=load(OUT/'test_actual'/f"{row['family']}.npz");l=z['risks'][:,0,:,:,0].mean(axis=(0,2));j=int(row['candidate'])-601;close(float(row['actual_gain_pp']),(l[0]-l[j])*100)
            close(float(row['absolute_error_pp']),abs(float(row['prediction_pp'])-float(row['actual_gain_pp'])))
        primary=student['primary'];expected=primary['overrides']>0 and primary['ci95_pp'][1]<0 and student['actual_gain_MAE_pp']['absolute']<student['actual_gain_MAE_pp']['zero'] and primary['harmed_families']/48<=.05
        assert student['proceed_to_robustness']==bool(expected)
    if (OUT/'robustness_summary.json').exists():
        robust_checked=True;record=read(OUT/'robustness_summary.json')
        assert student_checked and read(OUT/'student_summary.json')['proceed_to_robustness']
        for r in record['conditions']:assert counts['robustness_'+r['condition']]==2880
        expected=all(r['overrides']>0 and r['mean_delta_pp']<0 and r['harmed_families']/48<=.05 for r in record['conditions']);assert record['proceed_to_llm']==expected
    if (OUT/'llm_summary.json').exists():
        llm_checked=True;assert robust_checked and read(OUT/'robustness_summary.json')['proceed_to_llm'];assert counts['llm_comparison']==11520
        advice=read(OUT/'llm_advice_frozen.json');assert len(advice['advice'])==48
        for p in (OUT/'llm_actual').glob('*.started.json'):assert advice['created_at_utc']<read(p)['started_at_utc']
        record=read(OUT/'llm_summary.json');assert record['improvement_signal']==bool(record['primary_ci95_pp'][1]<0 and sum(v<0 for v in record['per_training_seed_delta_pp'].values())>=4)
    return {'all_checks_passed':True,'verifier_sha256':sha(__file__),'units_checked':units,'rollouts_from_saved_units':sum(counts.values()),
        'instrumentation_check_rollouts':2,'simulations_completed':budget['simulations_completed'],'counts':dict(counts),'trace_steps_checked':trace_steps,
        'student_stage_checked':student_checked,'robustness_stage_checked':robust_checked,'llm_stage_checked':llm_checked,'new_ppo_updates':0,
        'qualification_thresholds_unchanged':True,'future_blocks_not_counted_as_independent_families':True}
