"""24 bounded paired presentation-order checks, using frozen original prompts."""
from datetime import datetime,timezone
from pathlib import Path
import copy
import json
import hashlib
import sys
import time
import numpy as np
import exit_choice_initial_benchmark as old

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/exit_choice_initial_benchmark_20261012'
OUT=ROOT/'outputs/exit_choice_order_stability_20261012'
DEADLINE=datetime.fromisoformat('2026-10-12T12:16:40+00:00')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen_plan.json').exists():return
    cases=read(SOURCE/'strict_initial_cases.json');selected=[]
    for d in sorted({r['date'] for r in cases}):
        rows=[r for r in cases if r['date']==d]
        rows.sort(key=lambda r:hashlib.sha256(('order20261012:'+r['record_id']).encode()).hexdigest())
        selected+=rows[:3]
    tasks=[]
    for c in selected:
        for channel in ['primary','guide_conditional']:
            name=c['record_id'].replace(':','_')+'_'+channel+'.json'
            src=SOURCE/'responses'/name;original=read(src)
            req=copy.deepcopy(original['payload'])
            observation=json.loads(req['messages'][1]['content'])
            observation['exits'].reverse()
            req['messages'][1]['content']=json.dumps(observation,ensure_ascii=False)
            assert req['options']==original['payload']['options']
            tasks.append({'case_id':c['record_id'],'date':c['date'],'channel':channel,
                          'source_path':str(src),'source_sha256':sha(src),'original_probabilities':original['probabilities'],
                          'payload':req})
    save('requests.json',tasks)
    save('frozen_plan.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'case_selection':'3 per date by fixed case-ID hash; no labels or model errors used',
         'cases':12,'requests':24,'operation':'reverse exit presentation order only; physical IDs and attributes unchanged',
         'original_prompt_schema_options_and_seed_retained':True,
         'requests_sha256':sha(OUT/'requests.json'),'source_script_sha256':sha(__file__),
         'paired_repetitions_not_new_human_samples':True,'no_adaptive_prompt_revision':True,
         'deadline_utc':DEADLINE.isoformat(),'retry_count':0,'probability_sum_tolerance':.01})
    print('Frozen 12 cases, 24 reverse-order requests',flush=True)

def run():
    plan=read(OUT/'frozen_plan.json');tasks=read(OUT/'requests.json')
    assert sha(OUT/'requests.json')==plan['requests_sha256']
    tag=next(r for r in old.api('/api/tags')['models'] if r['name']=='qwen3:8b')
    assert tag['digest']==read(ROOT/'outputs/local_llm_selection.json')['digest']
    save('runtime.json',{'model':tag,'version':old.api('/api/version')})
    failures=0
    for index,r in enumerate(tasks,1):
        name='responses/'+r['case_id'].replace(':','_')+'_'+r['channel']+'.json'
        if (OUT/name).exists():continue
        if datetime.now(timezone.utc)>=DEADLINE:break
        assert sha(r['source_path'])==r['source_sha256']
        start=time.monotonic();result={'case_id':r['case_id'],'channel':r['channel'],'payload':r['payload']}
        try:
            response=old.api('/api/chat',r['payload'],180);result['response']=response
            data=json.loads(response['message']['content'])
            assert set(data)=={'p1','p2','p3'} and all(type(v) in (int,float) for v in data.values())
            pp=np.array([data[f'p{i}'] for i in [1,2,3]],dtype=float)
            assert np.isfinite(pp).all() and (pp>=0).all() and (pp<=1).all() and abs(pp.sum()-1)<=.01
            result.update({'valid':True,'probabilities':(pp/pp.sum()).tolist()});failures=0
        except Exception as exc:
            result.update({'valid':False,'error':str(exc)});failures+=1
        result['seconds']=time.monotonic()-start;save(name,result)
        save('progress.json',{'saved':len(list((OUT/'responses').glob('*.json'))),'planned':24,'last_valid':result['valid']})
        print(json.dumps({'index':index,'total':24,'valid':result['valid'],'seconds':round(result['seconds'],2)}),flush=True)
        if failures>=3:break

def report():
    tasks=read(OUT/'requests.json');cases={r['record_id']:r for r in read(SOURCE/'strict_initial_cases.json')};rows=[]
    for r in tasks:
        p=OUT/('responses/'+r['case_id'].replace(':','_')+'_'+r['channel']+'.json')
        if not p.exists():continue
        result=read(p)
        if not result['valid']:continue
        a=np.array(r['original_probabilities']);b=np.array(result['probabilities']);label=cases[r['case_id']]['initial_label']-1
        rows.append({'case_id':r['case_id'],'date':r['date'],'channel':r['channel'],
                     'original':a.tolist(),'reversed':b.tolist(),'total_variation':float(abs(a-b).sum()/2),
                     'top_exit_changed':bool(a.argmax()!=b.argmax()),'original_correct':bool(a.argmax()==label),
                     'reversed_correct':bool(b.argmax()==label),'nll_change':float(-np.log(max(b[label],1e-6))+np.log(max(a[label],1e-6)))})
    save('paired_results.json',rows);summary=[]
    for c in ['primary','guide_conditional']:
        rr=[r for r in rows if r['channel']==c]
        if rr:summary.append({'channel':c,'pairs':len(rr),'mean_probability_TV':float(np.mean([r['total_variation'] for r in rr])),
                              'top_exit_changes':sum(r['top_exit_changed'] for r in rr),
                              'original_correct':sum(r['original_correct'] for r in rr),'reversed_correct':sum(r['reversed_correct'] for r in rr),
                              'mean_nll_change':float(np.mean([r['nll_change'] for r in rr]))})
    save('summary.json',{'expected':24,'valid':len(rows),'complete':len(rows)==24,'channels':summary})
    lines=['# 出口展示顺序的配对检查','', '2026年10月12日。每个日期按固定编号哈希选3例，12个既有真人案例，两种信息条件，共24次新本地调用。','',
           '只把E1,E2,E3的展示顺序改成E3,E2,E1；物理身份、属性、原始提示、模型、采样参数、schema和种子保持一致。选择案例时没有使用标签或模型错误。','',
           '| 条件 | 有效配对 | 平均概率总变差 | 最大概率出口改变 | 原顺序正确 | 逆序正确 |','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['channel']} | {r['pairs']} | {r['mean_probability_TV']:.2%} | {r['top_exit_changes']} | {r['original_correct']} | {r['reversed_correct']} |")
    lines+=['','这是12个案例的表达一致性检查，不是新增独立人类样本，也不根据结果修改提示或选取更有利的顺序。输出合法不能证明出口语义或概率稳定。','',
            '[相关方法依据：Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions，NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.130/)。本轮仅报告本模型与本样本的结果。','']
    report=ROOT/'reports/Exit_choice出口展示顺序检查_2026-10-12.md';report.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'report':str(report),'channels':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    prepare();run();report()
