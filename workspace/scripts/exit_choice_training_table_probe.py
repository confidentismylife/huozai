"""Eight frozen factual checks. Diagnostic only; no behavior-prompt repair."""
from datetime import datetime,timezone
from pathlib import Path
import json,hashlib,sys,time
import exit_choice_initial_benchmark as old
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/exit_choice_training_table_probe_20261012'
SOURCE=ROOT/'outputs/exit_choice_supervised_context_20261012'
SYSTEM='这是数据读取诊断。输入提供出口计数，或者紧凑训练表。training_exit_counts按E1,E2,E3排列。表格每行是[D1,D2,D3,C1,C2,C3,Y]；最后的Y是实际选择出口1、2或3。统计每种Y的出现次数和总行数，输出整数n1,n2,n3,total。不要把距离或人群数量当作出口选择标签。只输出JSON。'
SCHEMA={'type':'object','properties':{k:{'type':'integer','minimum':0} for k in ['n1','n2','n3','total']},'required':['n1','n2','n3','total'],'additionalProperties':False}
OPTIONS={'temperature':0,'seed':2026101203,'num_ctx':4096,'num_predict':128,'num_batch':32,'num_thread':8,'num_gpu':0}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'frozen_plan.json').exists():
        contexts=read(SOURCE/'contexts.json');tasks=[]
        for c in contexts:
            if c['mode']!='all_training_records':continue
            data=json.loads(c['context']);counts=data['training_exit_counts'];expected=dict(zip(['n1','n2','n3'],counts));expected['total']=sum(counts)
            for mode in ['copy_given_counts','count_labels_in_table']:
                inp={'training_exit_counts':counts} if mode=='copy_given_counts' else {'training_records':data['training_records']}
                tasks.append({'date':c['test_date'],'mode':mode,'expected':expected,
                              'payload':{'model':'qwen3:8b','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(inp,separators=(',',':'))}],
                                         'format':SCHEMA,'options':OPTIONS,'think':False,'stream':False,'keep_alive':'2m'}})
        save('requests.json',tasks);save('frozen_plan.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'requests':8,
             'task':'copy given counts vs tally labels from the exact same training tables',
             'expected_answers_computed_before_inference':True,'not_human_behavior_accuracy':True,
             'diagnostic_followup_after_supervised_context_failure':True,'no_adaptive_prompt_revision':True,
             'requests_sha256':hashlib.sha256((OUT/'requests.json').read_bytes()).hexdigest()})
    plan=read(OUT/'frozen_plan.json');assert hashlib.sha256((OUT/'requests.json').read_bytes()).hexdigest()==plan['requests_sha256']
    tasks=read(OUT/'requests.json');tag=next(r for r in old.api('/api/tags')['models'] if r['name']=='qwen3:8b')
    assert tag['digest']==read(ROOT/'outputs/local_llm_selection.json')['digest']
    for index,r in enumerate(tasks,1):
        p=OUT/(r['date']+'_'+r['mode']+'.json')
        if p.exists():continue
        start=time.monotonic();result={**r}
        try:
            response=old.api('/api/chat',r['payload'],180);result['response']=response
            answer=json.loads(response['message']['content']);assert set(answer)==set(r['expected'])
            assert all(type(v)==int and v>=0 for v in answer.values())
            result.update({'valid':True,'answer':answer,'exact_match':answer==r['expected']})
        except Exception as exc:result.update({'valid':False,'exact_match':False,'error':str(exc)})
        result['seconds']=time.monotonic()-start;save(p.name,result)
        print(json.dumps({'index':index,'total':8,'mode':r['mode'],'exact_match':result['exact_match']}),flush=True)
    records=[read(OUT/(r['date']+'_'+r['mode']+'.json')) for r in tasks]
    summary=[{'mode':m,'questions':4,'valid':sum(r['valid'] for r in records if r['mode']==m),
              'exact_matches':sum(r['exact_match'] for r in records if r['mode']==m)} for m in ['copy_given_counts','count_labels_in_table']]
    save('summary.json',summary)
    lines=['# 训练资料的基本读取诊断','', '2026年10月12日。8道预先写定正确答案的诊断题，复用四个外层训练表，不属于新的真人预测样本。','',
           '| 任务 | 问题数 | 合法输出 | 完全正确 |','|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['mode']} | {r['questions']} | {r['valid']} | {r['exact_matches']} |")
    lines+=['','“复述已给计数”检验明确统计任务下的数字读取；“统计表格Y”检验是否能准确汇总相同紧凑训练表。两者都不能单独证明模型理解了人群决策逻辑，也不能把错误唯一归因为行为预测失败的原因。','',
            '没有改动前面的预测提示、标签或得分。所有正确答案在模型调用前固定，完整请求与回答保存。','']
    report=ROOT/'reports/Exit_choice训练资料读取诊断_2026-10-12.md';report.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'report':str(report),'summary':summary},ensure_ascii=False),flush=True)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf8');main()
