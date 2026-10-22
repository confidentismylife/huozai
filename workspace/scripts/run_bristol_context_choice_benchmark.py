"""Fixed mirror-family validation and 34 local LLM scene probabilities."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

from audit_liao_dynamic_data import ROOT,sha
from run_liao_motion_pilot import fit_logit
from run_liao_llm_motion_diagnostic import api

CONFIG=ROOT/'schemas/bristol_context_choice_benchmark_20261013.json'
DATA=ROOT/'outputs/dynamic_information_search_20261013/audit'
OUTPUT=ROOT/'outputs/bristol_context_choice_benchmark_20261013'
SYSTEM=("Predict which room exit a human participant will choose in a first-person virtual evacuation task. "
    "Return a calibrated probability of choosing the LEFT room exit, not a recommendation for the best route. "
    "Each participant starts equidistant from two available, equal-width (1.5m) room doors in a 20m square room. "
    "They are told a fire alarm has sounded and to go to the exit as quickly as possible. "
    "Assigned simulated people are moving through the left and right doors. Small designated crowds are free-flowing; "
    "large designated crowds cause congestion. These counts are design conditions, not measured instantaneous queues. "
    "Some participants receive a map before moving: it may show equal onward distance or one door nearer to the final exit. "
    "With no map, the onward location of the final exit is unknown. No smoke or staff-guidance observation is provided. "
    "Use only the current scene and the baseline left-choice probability estimated from other conditions. "
    "Other people's exit use may provide both social information and congestion information. Do not invent missing observations. "
    "Return only JSON {\"p_left\":number}, between 0 and 1.")
SCHEMA={'type':'object','properties':{'p_left':{'type':'number','minimum':0,'maximum':1}},'required':['p_left'],'additionalProperties':False}


def family(r):return (min(int(r['assigned_left_crowd']),int(r['assigned_right_crowd'])),max(int(r['assigned_left_crowd']),int(r['assigned_right_crowd'])),int(r['knowledge_code']))


def condition(r):return (int(r['assigned_left_crowd']),int(r['assigned_right_crowd']),int(r['knowledge_code']))


def features(r):
    near=r['nearer_to_final_exit'];map_adv=1 if near=='left' else -1 if near=='right' else 0
    available=int(r['map_information']!='no_map');left,right=int(r['assigned_left_crowd']),int(r['assigned_right_crowd'])
    adv=(right-left)/(left+right);busy=int(r['crowd_state_design']=='busy')
    return np.array([map_adv,available,adv,int(left==0),int(right==0),busy,adv*busy,adv*available])


def weights(records):
    counts=Counter(condition(r) for r in records);n=len(counts)
    w=np.array([1/(n*counts[condition(r)]) for r in records]);assert np.isclose(w.sum(),1);return w


def preference(r):
    if r['nearer_to_final_exit']=='left':return 1
    if r['nearer_to_final_exit']=='right':return -1
    l,t=int(r['assigned_left_crowd']),int(r['assigned_right_crowd'])
    return 1 if l<t else -1 if l>t else 0


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    records=[r for r in csv.DictReader((DATA/'bristol_context_cases.csv').open(encoding='utf-8-sig')) if r['eligible_context_benchmark']=='True']
    assert len(records)==489 and len({condition(r) for r in records})==34
    x=np.array([features(r) for r in records]);y=np.array([int(r['observed_exit'])==1 for r in records],dtype=float)
    names=cfg['methods'][:3];predictions={name:np.full(len(records),np.nan) for name in names};folds=[];tasks=[]
    for fam in sorted({family(r) for r in records}):
        test=np.array([family(r)==fam for r in records]);train=~test
        training=[r for r,a in zip(records,train) if a];testing=[r for r,a in zip(records,test) if a]
        w=weights(training);p0=float((w@y[train]*len(training)+.5)/(len(training)+1))
        predictions[names[0]][test]=p0
        rec=np.array([preference(r) for r in training]);nonzero=rec!=0
        # Single confidence shared by mirrored recommendation directions.
        agree=((y[train]==1)&(rec==1))|((y[train]==0)&(rec==-1))
        if nonzero.any():
            q=float((np.sum(w[nonzero]*agree[nonzero])*len(training)+1)/(np.sum(w[nonzero])*len(training)+2))
        else:q=.5
        predictions[names[1]][test]=[q if preference(r)==1 else 1-q if preference(r)==-1 else .5 for r in testing]
        center,scale,beta,opt=fit_logit(x[train],y[train],w,p0,cfg['logit']['l2'],cfg['logit']['maxiter'])
        predictions[names[2]][test]=expit(np.column_stack([np.ones(test.sum()),(x[test]-center)/scale])@beta)
        folds.append(dict(family=fam,train_records=len(training),test_records=len(testing),train_base_rate=p0,
            rule_agreement_probability=q,train_conditions=[list(c) for c in sorted({condition(r) for r in training})],
            held_conditions=[list(c) for c in sorted({condition(r) for r in testing})],optimizer=opt))
        for cond in sorted({condition(r) for r in testing}):
            row=next(r for r in testing if condition(r)==cond)
            obs={k:row[k] for k in ['assigned_left_crowd','assigned_right_crowd','map_information','nearer_to_final_exit','crowd_state_design']}
            obs['assigned_left_crowd']=int(obs['assigned_left_crowd']);obs['assigned_right_crowd']=int(obs['assigned_right_crowd'])
            obs['training_base_probability_left']=p0
            content=json.dumps(obs,sort_keys=True,separators=(',',':'))
            options={k:cfg['LLM'][k] for k in ['temperature','seed','num_ctx','num_predict','num_thread','num_batch','num_gpu']}
            payload={'model':cfg['LLM']['model'],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],
                     'format':SCHEMA,'think':False,'stream':False,'keep_alive':'2m','options':options}
            tasks.append(dict(condition=list(cond),key='_'.join(map(str,cond)),payload=payload,
                prompt_sha256=hashlib.sha256((SYSTEM+'\n'+content).encode()).hexdigest()))
    assert all(np.isfinite(p).all() for p in predictions.values())
    rows=[{**r,**{name:float(predictions[name][i]) for name in names}} for i,r in enumerate(records)]
    with (OUTPUT/'baseline_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path=OUTPUT/'frozen_LLM_manifest.json'
    manifest=dict(config_sha256=sha(CONFIG),source_cases_sha256=sha(DATA/'bristol_context_cases.csv'),model_digest=cfg['LLM']['digest'],
                  expected_requests=34,tasks=tasks,folds=folds)
    # Compare serialized forms: JSON stores Python tuple family keys as lists.
    manifest=json.loads(json.dumps(manifest))
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old!=manifest:raise ValueError('Frozen manifest differs; do not replace or mix')
    else:path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return cfg,manifest,rows


def evaluate(records,name):
    y=np.array([int(r['observed_exit'])==1 for r in records],dtype=float)
    p=np.clip([float(r[name]) for r in records],1e-9,1-1e-9);w=weights(records)
    loss=-(y*np.log(p)+(1-y)*np.log1p(-p));brier=(p-y)**2
    accuracy=float(np.mean((p>=.5)==y))
    cells=[]
    for c in sorted({condition(r) for r in records}):
        ix=np.array([condition(r)==c for r in records]);cells.append(abs(float(p[ix][0])-float(y[ix].mean())))
    return dict(records=len(records),conditions=len(cells),condition_macro_NLL=float(w@loss),condition_macro_Brier=float(w@brier),
                participant_NLL=float(loss.mean()),participant_accuracy=accuracy,mean_absolute_condition_probability_error=float(np.mean(cells)))


def run():
    cfg,plan,records=prepare()
    tags=api('/api/tags',timeout=5);m=next(m for m in tags['models'] if m['name']==cfg['LLM']['model'])
    if m['digest']!=plan['model_digest']:raise ValueError('LLM digest mismatch')
    path=OUTPUT/'responses.jsonl';saved={}
    if path.exists():saved={r['key']:r for r in (json.loads(line) for line in path.read_text().splitlines())}
    errors=0;started=time.monotonic()
    for task in plan['tasks']:
        if task['key'] in saved:
            assert saved[task['key']]['prompt_sha256']==task['prompt_sha256']
            assert saved[task['key']]['model_digest']==plan['model_digest'];continue
        row=dict(key=task['key'],prompt_sha256=task['prompt_sha256'],model_digest=plan['model_digest'])
        before=time.monotonic()
        try:
            response=api('/api/chat',task['payload'],90);row['response']=response;obj=json.loads(response['message']['content']);p=obj.get('p_left')
            if set(obj)!={'p_left'} or isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
            row.update(valid=True,p_left=float(p));errors=0
        except Exception as e:
            row.update(valid=False,error=f'{type(e).__name__}: {e}',p_left=json.loads(task['payload']['messages'][1]['content'])['training_base_probability_left']);errors+=1
        row['wall_seconds']=time.monotonic()-before
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        saved[row['key']]=row
        if len(saved)==1 or len(saved)%5==0 or not row['valid']:print(json.dumps(dict(completed=len(saved),planned=34,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
        if errors>=3:break
    if len(saved)!=34:
        print('Partial LLM ledger saved; full benchmark not evaluated.',flush=True);return
    for row in records:
        key='_'.join(map(str,condition(row)));response=saved[key]
        row['qwen3_4b']=response['p_left'];row['LLM_valid']=response['valid']
    with (OUTPUT/'all_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    per_family=[]
    for f in sorted({family(r) for r in records}):
        rs=[r for r in records if family(r)==f]
        for name in cfg['methods']:per_family.append(dict(family=f,method=name,**evaluate(rs,name)))
    result=dict(status='completed_exploratory_mirror_family_validation',records=len(records),conditions=34,mirror_families=len(plan['folds']),
        methods={name:evaluate(records,name) for name in cfg['methods']},per_family=per_family,
        LLM_requests=len(saved),valid_LLM_responses=sum(r['valid'] for r in saved.values()),
        mean_request_seconds=float(np.mean([r['wall_seconds'] for r in saved.values()])),
        total_request_seconds=sum(r['wall_seconds'] for r in saved.values()),
        config_sha256=sha(CONFIG),manifest_sha256=sha(OUTPUT/'frozen_LLM_manifest.json'),
        limitations=['36 treatment cells audited before evaluation; exploratory.', '58 biased-map balanced-crowd records excluded without outcome-based decoding.',
                    'One LLM probability per condition is not 489 independent LLM samples.', 'Static contextual exit choices, not dynamic replanning or real fire control.'],
        PPO_updates=0,old_final_scenarios_accessed=False,Liao_confirmation_scores_computed=False)
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','records','conditions','mirror_families','methods','LLM_requests','valid_LLM_responses','total_request_seconds']},indent=2),flush=True)


if __name__=='__main__':run()
