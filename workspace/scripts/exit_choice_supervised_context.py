"""Bounded same-setting Qwen comparison: no data, train prior, all train rows.

Raw outer-date predictions only. Never calibrate these leave-date-out outputs
with a naive second cross-validation: their prompts could leak outer labels
into the meta-model. No model weights or PPO are trained here.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import psutil
import exit_choice_initial_benchmark as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/exit_choice_supervised_context_20261012"
SOURCE=ROOT/"outputs/exit_choice_initial_benchmark_20261012"
DEADLINE=datetime.fromisoformat("2026-10-12T12:16:40+00:00")
MODES=["zero_same_settings","training_prior","all_training_records"]
OPTIONS={"temperature":0,"seed":2026101202,"num_ctx":4096,"num_predict":128,
         "num_batch":32,"num_thread":8,"num_gpu":0}
SYSTEM=old.SYSTEM+"\n可提供来自其他疏散事件的训练资料。训练资料仅帮助预测当前人的实际选择，当前事件可能存在分布变化。训练表每行依次为D1,D2,D3,C1,C2,C3,Y；D为距离，C为此前选择人数，Y为实际选择出口编号。所有训练记录Fam固定为[1,0,0]。只输出当前个体的p1,p2,p3。"


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"frozen_plan.json").exists():print("Existing supervised-context plan retained");return
    cases=read(SOURCE/"strict_initial_cases.json")
    files=[p for p in SOURCE.rglob("*") if p.is_file() and p.suffix!=".log"]
    files +=[ROOT/"outputs/local_llm_selection.json",Path("C:/Users/摩天轮大盗/Downloads/Exit choice.xlsx"),Path("C:/Users/摩天轮大盗/Downloads/Exit choice ppt.pptx")]
    save("preservation_before.json",[{"path":str(p),"sha256":sha(p)} for p in files])
    tasks=[]; contexts=[]
    for date in sorted({r["date"] for r in cases}):
        tr=[r for r in cases if r["date"]!=date]; te=[r for r in cases if r["date"]==date]
        rng=np.random.default_rng(np.random.SeedSequence([2026101202,int(date.replace('-',''))]))
        order=rng.permutation(len(tr)); tr=[tr[i] for i in order]
        counts=[sum(r["initial_label"]==k for r in tr) for k in (1,2,3)]
        table=[[*[e["dist"] for e in r["exits"]],*[e["people"] for e in r["exits"]],r["initial_label"]] for r in tr]
        assert len(table)==len(tr) and all(r["date"]!=date for r in tr)
        for mode in MODES:
            info={"training_records":None,"training_exit_counts":None}
            if mode!="zero_same_settings":info["training_exit_counts"]=counts
            if mode=="all_training_records":info["training_records"]=table
            context=json.dumps(info,ensure_ascii=False,separators=(',',':'))
            contexts.append({"test_date":date,"mode":mode,"train_ids":[r["record_id"] for r in tr],
                             "training_n":len(tr),"test_n":len(te),"context":context,
                             "context_sha256":hashlib.sha256(context.encode()).hexdigest()})
            for case in sorted(te,key=lambda r:r["record_id"]):
                target={"exits":[{"exit":f"E{e['exit']}","distance":e["dist"],"previous_choosers":e["people"],"commonly_used":e["fam"]} for e in case["exits"]]}
                payload={"model":"qwen3:8b","messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":"其他事件训练资料："+context},
                        {"role":"user","content":"当前个体："+json.dumps(target,ensure_ascii=False,separators=(',',':'))}],
                        "format":old.SCHEMA,"think":False,"stream":False,"keep_alive":"2m","options":OPTIONS}
                tasks.append({"case_id":case["record_id"],"test_date":date,"mode":mode,"payload":payload,
                              "training_n":len(tr),"training_ids":[r["record_id"] for r in tr]})
    save("contexts.json",contexts);save("requests.json",tasks)
    plan={"frozen_at_utc":now(),"deadline_utc":DEADLINE.isoformat(),"modes":MODES,
          "primary_channel_only":True,"case_count":len(cases),"maximum_new_requests":len(tasks),
          "task_order":"by held-out date then mode; identical training prefix supports cache reuse",
          "model":"qwen3:8b","options":OPTIONS,"system_prompt":SYSTEM,
          "same_settings_baseline_included":True,"all_examples_from_other_three_dates":True,
          "no_target_date_or_person_identifiers_in_prompt":True,"no_observation_noise_or_scene_changes":True,
          "all_train_records_usage":"All 59-87 available training records per date; all observed attribute columns passed; only label sources differ by mode",
          "maximum_context":4096,"timeout_seconds":180,"retries":0,"stop_after_consecutive_failures":3,
          "minimum_available_GiB_before_cold_model_load":6.5,
          "schema":old.SCHEMA,"sum_tolerance":.01,"invalid_output_fallback":"training frequency; report separately",
          "cases_sha256":sha(SOURCE/"strict_initial_cases.json"),"requests_sha256":sha(OUT/"requests.json"),
          "source_script_sha256":sha(__file__),"prior_results_used_to_define_followup":True,
          "not_new_confirmatory_human_data":True,"no_naive_meta_calibration_of_contextual_outputs":True,
          "no_ppo_or_model_weight_training":True}
    save("frozen_plan.json",plan)
    (OUT/"source_snapshot").mkdir(exist_ok=True)
    (OUT/"source_snapshot/exit_choice_supervised_context.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"phase":"supervised_context_frozen","requests":len(tasks),"case_count":len(cases)}),flush=True)


def preservation():
    rows=read(OUT/"preservation_before.json")
    result=[{**r,"after":sha(r["path"])} for r in rows]
    assert all(r["after"]==r["sha256"] for r in result)
    save("preservation_after.json",result)


def run():
    plan=read(OUT/"frozen_plan.json");tasks=read(OUT/"requests.json")
    assert sha(OUT/"requests.json")==plan["requests_sha256"]
    tag=next(r for r in old.api('/api/tags')["models"] if r["name"]==plan["model"])
    selection=read(ROOT/"outputs/local_llm_selection.json")
    assert tag["digest"]==selection["digest"]
    active=old.api('/api/ps')["models"]
    available=psutil.virtual_memory().available/2**30
    if not any(r.get('name')==plan['model'] for r in active) and available<plan['minimum_available_GiB_before_cold_model_load']:
        save('run_status.json',{'state':'cold_load_memory_insufficient','available_GiB':available,'at_utc':now()})
        print('Memory limit prevents cold load; continue other authorized analyses',flush=True);return
    save('runtime.json',{'at_utc':now(),'tag':tag,'version':old.api('/api/version'),'available_GiB_before':available})
    failures=0
    for index,task in enumerate(tasks,1):
        key=task['case_id'].replace(':','_')+'_'+task['mode'];name=f'responses/{key}.json'
        if (OUT/name).exists():continue
        if datetime.now(timezone.utc)>=DEADLINE:
            save('run_status.json',{'state':'deadline_reached','at_utc':now(),'last_index':index-1});break
        rec={**task,'started_at_utc':now()};start=time.monotonic()
        try:
            response=old.api('/api/chat',task['payload'],plan['timeout_seconds']);rec['response']=response
            parsed=json.loads(response['message']['content'])
            assert set(parsed)=={'p1','p2','p3'}
            assert all(type(v) in (int,float) for v in parsed.values())
            pp=np.array([parsed[f'p{i}'] for i in (1,2,3)],dtype=float)
            assert np.isfinite(pp).all() and (pp>=0).all() and (pp<=1).all()
            assert abs(pp.sum()-1)<=plan['sum_tolerance']
            assert response.get('done',False),'Incomplete generation'
            assert response.get('prompt_eval_count',0)+response.get('eval_count',0)<plan['maximum_context'], 'Context capacity approached; potential prompt truncation'
            rec.update({'valid':True,'probabilities':(pp/pp.sum()).tolist(),'raw_sum':float(pp.sum())});failures=0
        except Exception as exc:
            rec.update({'valid':False,'error':f'{type(exc).__name__}: {exc}'});failures+=1
        rec['wall_seconds']=time.monotonic()-start;save(name,rec)
        count=len(list((OUT/'responses').glob('*.json')))
        save('run_status.json',{'state':'running','at_utc':now(),'saved_requests':count,'planned_requests':len(tasks),
             'last_index':index,'last_mode':task['mode'],'last_valid':rec['valid'],'consecutive_failures':failures})
        print(json.dumps({'index':index,'total':len(tasks),'mode':task['mode'],'case':task['case_id'],
                          'valid':rec['valid'],'seconds':round(rec['wall_seconds'],2)},ensure_ascii=False),flush=True)
        if failures>=3:
            save('run_status.json',{'state':'three_consecutive_failures','at_utc':now(),'saved_requests':count,'planned_requests':len(tasks)});break
    else:
        save('run_status.json',{'state':'completed','at_utc':now(),'saved_requests':len(tasks),'planned_requests':len(tasks)})
    preservation()


def report():
    plan=read(OUT/'frozen_plan.json');cases=read(SOURCE/'strict_initial_cases.json');lookup={r['record_id']:r for r in cases}
    records=[read(p) for p in (OUT/'responses').glob('*.json')]
    predictions=[]
    for r in records:
        case=lookup[r['case_id']];tr=[c for c in cases if c['date']!=case['date']]
        assert case['record_id'] not in r['training_ids'] and all(lookup[i]['date']!=case['date'] for i in r['training_ids'])
        predictions.append({'case_id':r['case_id'],'mode':r['mode'],'date':case['date'],
                            'probabilities':r['probabilities'] if r['valid'] else old.frequency(tr).tolist(),
                            'fallback':not r['valid'],'true_exit':case['initial_label']})
    rows=[]
    for mode in MODES:
        selected=[r for r in predictions if r['mode']==mode]
        if not selected:continue
        for d in sorted({r['date'] for r in selected})+['pooled']:
            rr=[r for r in selected if d=='pooled' or r['date']==d]
            rows.append({'mode':mode,'date':d,**old.metrics([lookup[r['case_id']] for r in rr],np.array([r['probabilities'] for r in rr])),
                         'fallbacks':sum(r['fallback'] for r in rr)})
        dd=[r for r in rows if r['mode']==mode and r['date']!='pooled']
        if len(dd)==4:
            rows.append({'mode':mode,'date':'date_macro','n':sum(r['n'] for r in dd),
                         **{k:float(np.mean([r[k] for r in dd])) for k in ['accuracy','log_loss','brier','exit_share_tv']},
                         'true_shares':[],'predicted_shares':[],'fallbacks':sum(r['fallbacks'] for r in dd)})
    save('predictions.json',predictions);save('metrics.json',rows)
    stats={'expected_requests':plan['maximum_new_requests'],'saved':len(records),'valid':sum(r['valid'] for r in records),
           'complete':len(records)==plan['maximum_new_requests'],
           'median_wall_seconds':float(np.median([r['wall_seconds'] for r in records])) if records else None,
           'max_prompt_tokens':max([r.get('response',{}).get('prompt_eval_count',0) for r in records],default=0),
           'model_weights_trained':False,'ppo_runs':0}
    save('summary.json',stats);preservation()
    labels={'zero_same_settings':'相同提示和参数的零样本对照','training_prior':'仅提供训练出口频率','all_training_records':'提供全部训练记录'}
    lines=['# Qwen真人学习资料对照','', '2026年10月12日。两小时自主任务的第二阶段。','',
           '比较相同模型、提示和推理参数下：不提供真人标签、只提供其他日期出口频率、提供其他日期全部训练记录。每个人所属日期的记录和标签始终排除于其提示。','',
           '训练记录为紧凑D1,D2,D3,C1,C2,C3,Y表。日期、编号、外貌、改道备注和PPT箭头不进入提示；Fam固定为[1,0,0]。没有训练模型权重。','',
           f"保存{stats['saved']}/{stats['expected_requests']}次回答，有效{stats['valid']}次。最大记录的提示token数为{stats['max_prompt_tokens']}。",'',
           '| 方法 | 日期等权准确率 | 日期等权对数损失 | 合并准确率 | 合并对数损失 | 回退 |','|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        macro=next((r for r in rows if r['mode']==mode and r['date']=='date_macro'),None)
        pooled=next((r for r in rows if r['mode']==mode and r['date']=='pooled'),None)
        if macro and pooled:lines.append(f"| {labels[mode]} | {macro['accuracy']:.2%} | {macro['log_loss']:.4f} | {pooled['accuracy']:.2%} | {pooled['log_loss']:.4f} | {pooled['fallbacks']} |")
    lines+=['','只有完整队列可作全样本比较；未完成时不得把局部结果与其他完整队列直接比较。', '',
            '本轮是基于既有日期的探索跟进，不是新最终确认数据。同一人的三个条件属于配对比较，不是三个独立人类样本。', '',
            '不能对本轮按日期交叉生成的监督上下文分数直接做另一层普通交叉验证校准：训练个体的原始LLM提示可能含当前外层评价日期的标签，造成堆叠泄漏。额外校准需要重新做完整嵌套生成，本轮不执行。', '',
            '与旧2048上下文零样本结果相比，参数和提示布局改变可能影响输出，因此本轮加入4096上下文的相同设置零样本对照。', '',
            '原数据缺乏动态时间对应和出口3选择者。结果属于出口选择预测，不能解释成真实火灾疏散效率或安全收益。', '']
    path=ROOT/'reports/Exit_choice_Qwen真人学习资料对照_2026-10-12.md';path.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(path),**stats},ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report'])
    globals()[p.parse_args().command]()
