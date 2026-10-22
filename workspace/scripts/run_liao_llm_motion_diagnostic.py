"""Frozen 120-case local Qwen3 4B diagnostic on development data only.

Case-control sampling is deliberate: census of positive windows, 9 negative
windows randomly selected per development run. Outcomes are never sent to LLM.
Fresh cache only; no reuse of historical 14B or other-model responses.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_liao_dynamic_data import ROOT, sha
from run_liao_motion_pilot import hierarchy_weights

OUTPUT=ROOT/'outputs/liao_llm_motion_diagnostic_20261013'
BASE=ROOT/'outputs/liao_motion_baseline_20261013'
MODEL='qwen3:4b'
EXPECTED_DIGEST='359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
SYSTEM=("Estimate the probability that a person changes their intended exit target within the next 2 seconds, "
    "during low-pressure autonomous room egress. This is a behavior forecast, not a route recommendation. "
    "A bend in movement or collision avoidance does not necessarily mean changing exit target. "
    "You receive six summaries of this person's past 1 second of motion and a baseline probability estimated "
    "from other experimental runs. No map, current target label, head direction, queue state, or other people's "
    "motion is observed. Do not invent those missing observations. Predict natural-population probability, "
    "not the fraction of positives in any evaluation sample. Output only JSON {\"p_change\":number}, between 0 and 1. "
    "The motion summaries are: mean path speed over 1s; net speed in recent and earlier 0.5s; recent minus early "
    "speed; absolute angle between those displacements (0 if either speed <0.10m/s); and path length minus net displacement.")
SCHEMA={'type':'object','properties':{'p_change':{'type':'number','minimum':0,'maximum':1}},
        'required':['p_change'],'additionalProperties':False}
OPTIONS={'temperature':0,'seed':2026101307,'num_ctx':1024,'num_predict':32,'num_thread':8,'num_batch':32,'num_gpu':0}


def api(path,payload=None,timeout=90):
    data=json.dumps(payload).encode('utf-8') if payload is not None else None
    req=urllib.request.Request('http://127.0.0.1:11434'+path,data=data,
                               headers={'Content-Type':'application/json'} if data else {})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    path=OUTPUT/'frozen_manifest.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    tags=api('/api/tags',timeout=5)
    match=next(m for m in tags['models'] if m['name']==MODEL)
    if match['digest']!=EXPECTED_DIGEST:raise ValueError('Model digest changed')
    original=list(csv.DictReader((BASE/'development_out_of_fold_predictions.csv').open(encoding='utf-8-sig')))
    features={r['window_key']:{k:float(v) for k,v in r.items() if k!='window_key'}
              for r in csv.DictReader((BASE/'past_only_motion_features.csv').open(encoding='utf-8-sig'))}
    rng=np.random.default_rng(2026101307);selected=[]
    for run in sorted({r['run'] for r in original}):
        positive=[r for r in original if r['run']==run and int(r['switch_within_horizon'])]
        negative=[r for r in original if r['run']==run and not int(r['switch_within_horizon'])]
        for row in positive:selected.append({**row,'inclusion_probability':1.0})
        for ix in rng.choice(len(negative),size=9,replace=False):
            selected.append({**negative[int(ix)],'inclusion_probability':9/len(negative)})
    rng.shuffle(selected)
    tasks=[]
    for row in selected:
        observation={'history_seconds':1,'forecast_seconds':2,
            'training_baseline_probability':float(row['training_frequency']),
            'motion':features[row['window_key']]}
        content=json.dumps(observation,sort_keys=True,separators=(',',':'))
        payload={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],
                 'think':False,'stream':False,'format':SCHEMA,'options':OPTIONS,'keep_alive':'2m'}
        tasks.append({'window_key':row['window_key'],'payload':payload,
                      'prompt_sha256':hashlib.sha256((SYSTEM+'\n'+content).encode()).hexdigest(),
                      'evaluation_only':row})
    assert len(tasks)==120 and sum(int(t['evaluation_only']['switch_within_horizon']) for t in tasks)==30
    result=dict(status='frozen_before_first_inference',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        model=MODEL,model_digest=EXPECTED_DIGEST,system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        seed=2026101307,requests=120,positive_windows=30,negative_windows=90,
        scope='Exploratory case-control diagnostic, no confirmation evaluation.',
        probability_threshold=.02,timeout_seconds=90,maximum_consecutive_errors=3,retries=0,
        invalid_output_policy='Use same-window training frequency; preserve failure and report separately.',
        source_sha256={name:sha(BASE/name) for name in ['development_out_of_fold_predictions.csv','past_only_motion_features.csv']},
        no_old_model_cache=True,tasks=tasks)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def run():
    plan=prepare()
    tags=api('/api/tags',timeout=5);match=next(m for m in tags['models'] if m['name']==MODEL)
    if match['digest']!=plan['model_digest']:raise ValueError('Model identity no longer matches freeze')
    cache=OUTPUT/'responses.jsonl';saved={}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);saved[row['window_key']]=row
    started=time.monotonic();errors=0
    for ix,task in enumerate(plan['tasks'],1):
        key=task['window_key']
        if key in saved:
            if saved[key]['prompt_sha256']!=task['prompt_sha256'] or saved[key]['model_digest']!=plan['model_digest']:
                raise ValueError('Cache provenance mismatch')
            continue
        row=dict(window_key=key,prompt_sha256=task['prompt_sha256'],model_digest=plan['model_digest'],
                 started_at_utc=datetime.now(timezone.utc).isoformat())
        request_start=time.monotonic()
        try:
            response=api('/api/chat',task['payload'],plan['timeout_seconds']);row['response']=response
            parsed=json.loads(response['message']['content']);p=parsed.get('p_change')
            if set(parsed)!={'p_change'} or isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
                raise ValueError('Invalid probability output')
            row.update(valid=True,p_change=float(p));errors=0
        except Exception as e:
            row.update(valid=False,error=f'{type(e).__name__}: {e}',
                       p_change=float(task['evaluation_only']['training_frequency']))
            errors+=1
        row['wall_seconds']=time.monotonic()-request_start
        with cache.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        saved[key]=row
        if ix==1 or ix%10==0 or not row['valid']:
            print(json.dumps({'completed':len(saved),'planned':120,'valid':sum(r['valid'] for r in saved.values()),
                               'elapsed_seconds':round(time.monotonic()-started,1),'last_request_seconds':round(row['wall_seconds'],2)}),flush=True)
        if errors>=plan['maximum_consecutive_errors']:
            print('Stopped after 3 consecutive errors; no retry or prompt change.',flush=True);break
    summarize(plan,saved)


def summarize(plan,saved):
    complete=len(saved)==len(plan['tasks'])
    population=list(csv.DictReader((BASE/'development_out_of_fold_predictions.csv').open(encoding='utf-8-sig')))
    original_weights=hierarchy_weights(population)
    weights={r['window_key']:float(w) for r,w in zip(population,original_weights)}
    cases=[]
    for task in plan['tasks']:
        if task['window_key'] not in saved:continue
        response=saved[task['window_key']];r=task['evaluation_only']
        cases.append({**r,'LLM_probability':response['p_change'],'LLM_valid':response['valid'],
                      'wall_seconds':response['wall_seconds'],'hierarchical_population_weight':weights[task['window_key']]})
    with (OUTPUT/'diagnostic_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    method_results={}
    for name in ['training_frequency','fixed_motion_rule_with_train_only_bin_risk','L2_motion_logit','LLM_probability']:
        y=np.array([int(r['switch_within_horizon']) for r in cases]);p=np.clip([float(r[name]) for r in cases],1e-9,1-1e-9)
        losses=-(y*np.log(p)+(1-y)*np.log1p(-p));alarms=p>=.02
        detected={(r['run'],r['local_id'],r['first_different_frame']) for r,a in zip(cases,alarms) if int(r['switch_within_horizon']) and a}
        method_results[name]=dict(sample_NLL=float(losses.mean()),
            positive_windows_alerted=int(np.sum(alarms & (y==1))),negative_sample_windows_alerted=int(np.sum(alarms & (y==0))),
            detected_events=len(detected),event_denominator=15 if complete else None)
        if complete:
            factors=np.array([r['hierarchical_population_weight']/float(r['inclusion_probability']) for r in cases])
            method_results[name]['HT_hierarchical_population_NLL_estimate']=float(factors@losses)
    paired={}
    if complete:
        for name in ['training_frequency','L2_motion_logit']:
            deltas={}
            for r in cases:
                y=int(r['switch_within_horizon'])
                def loss(p):
                    p=min(1-1e-9,max(1e-9,float(p)));return -y*math.log(p)-(1-y)*math.log1p(-p)
                deltas[r['window_key']]=r['hierarchical_population_weight']*(loss(r['LLM_probability'])-loss(r[name]))
            estimate=sum(deltas[r['window_key']]/float(r['inclusion_probability']) for r in cases)
            variance=0.0
            for run in sorted({r['run'] for r in population}):
                negatives=[r for r in cases if r['run']==run and not int(r['switch_within_horizon'])]
                N=sum(r['run']==run and not int(r['switch_within_horizon']) for r in population);n=len(negatives)
                vals=np.array([deltas[r['window_key']] for r in negatives])
                variance+=N*N*(1-n/N)*float(np.var(vals,ddof=1))/n
            se=math.sqrt(variance)
            paired[name]=dict(LLM_minus_baseline_NLL_estimate=estimate,negative_sampling_design_SE=se,
                conditional_normal_95_interval=[estimate-1.96*se,estimate+1.96*se],
                note='Conditional on fixed development dataset/prediction function; interval covers negative-window sampling only, not independent-person or cross-scene generalization.')
    valid=sum(r['valid'] for r in saved.values())
    timings=[r['wall_seconds'] for r in saved.values()]
    summary=dict(status='completed' if complete else 'partial_stopped',model=MODEL,model_digest=plan['model_digest'],
        requests_completed=len(saved),planned=120,valid_responses=valid,invalid_responses=len(saved)-valid,
        mean_request_seconds=float(np.mean(timings)),median_request_seconds=float(np.median(timings)),
        total_request_wall_seconds=float(sum(timings)),methods=method_results,paired_sampling_estimates=paired,
        manifest_sha256=sha(OUTPUT/'frozen_manifest.json'),
        sample_note='30 positive windows/15 events + 90 negatives; raw sample precision/prevalence cannot be interpreted as population values.',
        confirmation_performance_computed=False,PPO_updates=0,old_14B_cache_reused=False)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
