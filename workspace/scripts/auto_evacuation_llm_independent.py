"""Independent LLM advice ablation, reusing frozen collection code without edits."""
from pathlib import Path
from collections import Counter
import sys,json,traceback,time
import numpy as np
import auto_evacuation_campaign as campaign

ROOT=campaign.ROOT;BASE=campaign.OUT;OUT=ROOT/'outputs/evacuation_auto_llm_independent_20261005'
PROTOCOL=ROOT/'reports/疏散自动实验LLM独立分支固定补充方案_2026-10-05.md'
REPORT=ROOT/'reports/LLM独立建议对照自动实验报告_2026-10-05.md'
HISTORY=ROOT/'outputs/third_round_status_history/before_automatic_llm_independent_20261005.json'
STATUS=campaign.STATUS


def prepare():
    assert not OUT.exists() and not HISTORY.exists()
    base_status=campaign.read(BASE/'campaign_status.json');assert base_status['state']=='complete' and not (BASE/'llm_summary.json').exists()
    completion_path=BASE/('completion_manifest_v2.json' if (BASE/'completion_manifest_v2.json').exists() else 'completion_manifest.json')
    completion=campaign.read(completion_path)
    for p,h in completion['files'].items():assert campaign.sha(ROOT/p)==h,p
    m=campaign.read(BASE/'manifest.json');OUT.mkdir();HISTORY.write_bytes(STATUS.read_bytes());(OUT/'previous_status_exact_bytes.json').write_bytes(STATUS.read_bytes())
    campaign.write(OUT/'manifest.json',{'role':'independent_advice_input_ablation_not_student_gate_validation','base_manifest':str(BASE/'manifest.json'),
        'base_manifest_sha256':campaign.sha(BASE/'manifest.json'),'base_completion_sha256':campaign.sha(completion_path),'base_completion_path':str(completion_path),
        'protocol_sha256':campaign.sha(PROTOCOL),'runner_sha256':campaign.sha(__file__),
        'collector_sha256':campaign.sha(ROOT/'scripts/auto_evacuation_campaign.py'),'banks':{'llm':m['banks']['llm']},
        'registry':{s['scenario_id']:m['registry'][s['scenario_id']] for s in m['banks']['llm']},'simulation_cap':11520,
        'llm_request_cap':96,'new_ppo_updates':0,'previous_status_sha256':campaign.sha(HISTORY),'created_at_utc':campaign.stamp(),
        'base_source_inputs_unchanged':True,'primary_same_controller_different_advice':True,'formal_final_bank':False})
    campaign.write(OUT/'budget.json',{'simulations_started':0,'simulations_completed':0,'by_kind':{},'matrix_fits':0,'mean_reference_estimates':0,'llm_requests':0,'ppo_updates':0})
    campaign.write(OUT/'campaign_status.json',{'state':'ready','stage':'prepared','detail':'LLM independent input ablation','updated_at_utc':campaign.stamp(),'pid':None,'run_attempts':0,'consecutive_failures':0})
    campaign.write(STATUS,{'stage':'automatic_llm_independent_branch_prepared','updated_date':'2026-10-05','manifest':(OUT/'manifest.json').relative_to(ROOT).as_posix(),
        'campaign_status':(OUT/'campaign_status.json').relative_to(ROOT).as_posix(),'report':REPORT.relative_to(ROOT).as_posix(),
        'previous_stage_status':HISTORY.relative_to(ROOT).as_posix(),'previous_stage_status_sha256':campaign.sha(HISTORY),
        'formal_round3_training_started':False,'formal_round3_final_bank_generated':False,'qualification_thresholds_changed':False,
        'new_simulations':0,'new_fits':0,'new_ppo_training_seeds':0,'new_llm_calls':0})


def clear_advice(manifest):
    from fire_evac.dynamic_llm_rl import DynamicLLMAdvice
    path=OUT/'llm_advice_frozen.json'
    if path.exists():records=campaign.read(path)['advice'];return {k:DynamicLLMAdvice(v['target_branch'],v['confidence'],('local_llm',)) for k,v in records.items()}
    tags=campaign.api_json('http://127.0.0.1:11434/api/tags',timeout=10);match=next(x for x in tags['models'] if x['name']=='qwen3:4b');records={}
    schema={'type':'object','properties':{'target':{'type':'string','enum':['north','south']},'confidence':{'type':'number','minimum':0,'maximum':1}},'required':['target','confidence'],'additionalProperties':False}
    for i,scene in enumerate(manifest['banks']['llm']):
        family=scene['scenario_id'];cache=OUT/'llm_cache'/f'{family}.json'
        if cache.exists():records[family]=campaign.read(cache)['advice'];continue
        data={k:scene[k] for k in ('growth_rate','onset_step','background_hazard','north_exit_capacity','south_exit_capacity','blocked_branch','blockage_end_step')}
        data['forecast_source']=('north','south')[scene['forecast_source_branch']]
        prompt='You advise a research evacuation simulator, not a real emergency. Branch codes: 0=north, 1=south, -1=none. Choose north or south using only the forecast, initial equal room occupancy120 each, equal initial smoke at background_hazard, and map capacities. Forecast source can be wrong. Capacity is persons per control step. Fire growth begins after onset_step. Consider smoke risk, throughput and blockage. Return JSON target north/south and confidence0..1. Do not invent observations. Data: '+json.dumps(data,sort_keys=True)
        last=None
        for attempt in range(2):
            budget=campaign.read(OUT/'budget.json');assert budget['llm_requests']<96;budget['llm_requests']+=1;campaign.write(OUT/'budget.json',budget);start=time.monotonic()
            try:
                response=campaign.api_json('http://127.0.0.1:11434/api/chat',{'model':'qwen3:4b','messages':[{'role':'user','content':prompt}],
                    'format':schema,'options':{'temperature':0,'seed':2026100551,'num_predict':96},'think':False,'stream':False},60)
                campaign.write(OUT/'llm_raw_responses'/f'{family}_attempt{attempt+1}.json',{'response':response,'prompt':prompt,'latency_seconds':time.monotonic()-start,'model_digest':match['digest']})
                parsed=json.loads(response['message']['content']);assert parsed['target'] in ('north','south') and 0<=float(parsed['confidence'])<=1
                advice={'target_branch':0 if parsed['target']=='north' else 1,'confidence':float(parsed['confidence'])}
                campaign.write(cache,{'advice':advice,'prompt':prompt,'model_digest':match['digest'],'created_at_utc':campaign.stamp(),'attempts':attempt+1});records[family]=advice;break
            except Exception as exc:
                last=exc;campaign.write(OUT/'llm_failures'/f'{family}_attempt{attempt+1}.json',{'error':repr(exc),'prompt':prompt,'created_at_utc':campaign.stamp()})
        if family not in records:raise RuntimeError('LLM advice failed twice without rule substitution for '+family) from last
        campaign.progress('llm_advice',{'completed':i+1,'total':48})
    campaign.write(path,{'created_at_utc':campaign.stamp(),'advice':records,'model_digest':match['digest'],'actual_seen':False,'prompt_fields_exclude_true_source_and_type':True,'branch_codes_explicitly_defined':True})
    return {k:DynamicLLMAdvice(v['target_branch'],v['confidence'],('local_llm',)) for k,v in records.items()}


def verify(m):
    counts=0;advice=campaign.read(OUT/'llm_advice_frozen.json');assert len(advice['advice'])==48
    for p in sorted((OUT/'llm_actual').glob('*.npz')):
        if p.name.endswith('.partial.npz'):continue
        z=campaign.load(p);r=z['risks'];assert r.shape==(2,1,1,6,11) and z['done'].all()
        np.testing.assert_allclose(r[...,0]+r[...,2]+r[...,3],1,atol=1e-10,rtol=0);assert np.isfinite(r).all()
        assert advice['created_at_utc']<campaign.read(p.with_suffix('.started.json'))['started_at_utc'];counts+=12
    budget=campaign.read(OUT/'budget.json');assert counts==11520==budget['simulations_completed'] and budget['simulations_started']==11520
    assert budget['matrix_fits']==budget['ppo_updates']==0 and budget['llm_requests']<=96
    rows=[]
    for seed in range(601,606):
        for scene in m['banks']['llm']:
            family=scene['scenario_id'];a=campaign.load(OUT/'llm_actual'/f'{seed}_llm_{family}.npz')['risks'][...,0].mean()
            b=campaign.load(OUT/'llm_actual'/f'{seed}_same_llm_actor_rule_{family}.npz')['risks'][...,0].mean()
            rows.append({'seed':seed,'family':family,'kind':scene['conflict_type'],'delta_pp':float((a-b)*100)})
    summary=campaign.read(OUT/'llm_summary.json');np.testing.assert_allclose(summary['primary_mean_delta_pp'],np.mean([r['delta_pp'] for r in rows]),atol=1e-10,rtol=0)
    for seed in range(601,606):np.testing.assert_allclose(summary['per_training_seed_delta_pp'][str(seed)],np.mean([r['delta_pp'] for r in rows if r['seed']==seed]),atol=1e-10,rtol=0)
    assert summary['improvement_signal']==bool(summary['primary_ci95_pp'][1]<0 and sum(x<0 for x in summary['per_training_seed_delta_pp'].values())>=4)
    raw=campaign.read(OUT/'manifest.json');assert campaign.sha(PROTOCOL)==raw['protocol_sha256'] and campaign.sha(__file__)==raw['runner_sha256']
    assert campaign.sha(ROOT/'scripts/auto_evacuation_campaign.py')==raw['collector_sha256']
    result={'all_checks_passed':True,'rollouts_independently_checked':counts,'family_seed_pairs_checked':len(rows),'advice_frozen_before_all_actuals':True,
        'same_controller_advice_contrast_recalculated':True,'new_fits':0,'new_ppo_updates':0,'verifier_sha256':campaign.sha(__file__)}
    campaign.write(OUT/'verification.json',result);return result


def report():
    from fire_evac.confirmatory import frozen_rule_advice
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    m=campaign.read(OUT/'manifest.json');summary=campaign.read(OUT/'llm_summary.json');budget=campaign.read(OUT/'budget.json');advice=campaign.read(OUT/'llm_advice_frozen.json')
    agreements=[]
    for scene in m['banks']['llm']:
        rule=frozen_rule_advice(DynamicConflictScenario(**scene));llm=advice['advice'][scene['scenario_id']]
        agreements.append({'family':scene['scenario_id'],'rule_target':rule.target_branch,'llm_target':llm['target_branch'],'rule_confidence':rule.confidence,
            'llm_confidence':llm['confidence'],'same_direction':rule.target_branch==llm['target_branch']})
    campaign.csv_write(OUT/'advice_agreement.csv',agreements)
    text=f'''# LLM独立建议对照自动实验报告

日期：2026-10-05。该分支独立于选择器是否通过，保留原学生负/正结果。

## 结果

48个全新家庭，5个冻结历史PPO训练种子，4臂，2人口×6未来，共完成{budget['simulations_completed']}条模拟；本地qwen3:4b请求{budget['llm_requests']}次，PPO更新0、学生拟合0。

主对照固定同一个raw_llm控制器，只将建议输入从规则改为真实本地LLM。失能差为LLM损失减规则损失，负值更好。平均差{summary['primary_mean_delta_pp']:+.6f}个百分点，按家庭95%区间[{summary['primary_ci95_pp'][0]:+.6f}, {summary['primary_ci95_pp'][1]:+.6f}]。预先固定的改善信号判定：{summary['improvement_signal']}。

各PPO训练种子差：{summary['per_training_seed_delta_pp']}。五种子复用同一家庭的人口/未来块，区间按家庭聚合，不能把种子×未来当成独立场景样本。

## 建议归因

LLM与规则方向一致的家庭{sum(r['same_direction'] for r in agreements)}/48。主对照允许方向和置信度同时改变，因此结果归因于建议输入组合，不能单独声称方向推理、置信度或自然语言解释有效。文字解释不进入PPO输入。

其余纯PPO、规则PPO及LLM PPO的跨训练臂比较在原始表中完整保留，属于次要比较。历史训练元数据标记每臂/种子请求50000步，实际步数应查看历史training_summary；raw_llm历史训练臂为探索性，不能追认为此前正式确认。四臂均9动作、34基础输入，LLM推理延迟/tokens单独记录，不把规则与LLM计算成本说成相同。

## 验证与局限

48条真实LLM建议在全部本分支实际结局之前冻结，提示不含真实火源、冲突标签或未来结果。模型digest：{advice['model_digest']}。原始响应、失败/重试、提示与延迟保留；无语法错误时不会自动作规则覆盖。

独立验证重算11520条结局及240个家庭—训练种子主对照。无论改善信号是否成立，都不修改原95%/10%/5%/95%门控资格标准；本分支不证明现实火灾效能、FDS/地图泛化或运行中动态纠正LLM。

[固定补充方案]({PROTOCOL.as_posix()})；[主结果]({(OUT/'llm_summary.json').as_posix()})；[每家庭原始结果]({(OUT/'llm_family_results.csv').as_posix()})；[独立验证]({(OUT/'verification.json').as_posix()})；[建议一致性]({(OUT/'advice_agreement.csv').as_posix()})。
'''
    REPORT.write_text(text,encoding='utf-8')


def run():
    if not OUT.exists():prepare()
    status=campaign.read(OUT/'campaign_status.json')
    if status['state']=='complete':return
    if status['consecutive_failures']>=3:raise RuntimeError('three independent branch failures require diagnosis')
    campaign.OUT=OUT;campaign.CAP=11520;campaign.llm_advice=clear_advice
    status.update(state='running',pid=__import__('os').getpid(),run_attempts=status['run_attempts']+1);campaign.write(OUT/'campaign_status.json',status)
    try:
        m=campaign.read(OUT/'manifest.json');base=campaign.read(BASE/'manifest.json');campaign.llm_comparison(m);verify(m);report()
        b=campaign.read(OUT/'budget.json');assert campaign.read(BASE/'budget.json')['simulations_started']+b['simulations_started']<=44738
        status=campaign.read(OUT/'campaign_status.json');status.update(state='complete',stage='complete',completed_at_utc=campaign.stamp(),consecutive_failures=0);campaign.write(OUT/'campaign_status.json',status)
        campaign.write(STATUS,{'stage':'automatic_llm_independent_branch_complete','updated_date':'2026-10-05','report':REPORT.relative_to(ROOT).as_posix(),
            'summary':(OUT/'llm_summary.json').relative_to(ROOT).as_posix(),'manifest':(OUT/'manifest.json').relative_to(ROOT).as_posix(),
            'completion_manifest':(OUT/'completion_manifest.json').relative_to(ROOT).as_posix(),'previous_stage_status':HISTORY.relative_to(ROOT).as_posix(),
            'previous_stage_status_sha256':campaign.sha(HISTORY),'new_simulations':b['simulations_started'],'new_fits':0,'new_ppo_training_seeds':0,'new_llm_calls':b['llm_requests'],
            'formal_round3_training_started':False,'formal_round3_final_bank_generated':False,'qualification_thresholds_changed':False,'hard_shield_qualification_passed':False})
        parent_path=Path(m['base_completion_path']);parent=campaign.read(parent_path)
        for p,h in parent['files'].items():assert campaign.sha(HISTORY if (ROOT/p).resolve()==STATUS else ROOT/p)==h,p
        paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('events.jsonl','campaign_status.json','completion_manifest.json')]+[REPORT,PROTOCOL,Path(__file__),HISTORY,STATUS]
        campaign.write(OUT/'completion_manifest.json',{'all_checks_passed':True,'role':'independent_local_LLM_advice_ablation_completion','budget':b,
            'parent_completion_sha256':campaign.sha(parent_path),'parent_completion_path':str(parent_path),'previous_status_exact_bytes_saved':True,
            'files':{p.relative_to(ROOT).as_posix():campaign.sha(p) for p in sorted(set(paths))}})
        for p,h in campaign.read(OUT/'completion_manifest.json')['files'].items():assert campaign.sha(ROOT/p)==h,p
        print('Independent LLM branch complete',flush=True)
    except Exception:
        status=campaign.read(OUT/'campaign_status.json');status.update(state='failed',consecutive_failures=status['consecutive_failures']+1);campaign.write(OUT/'campaign_status.json',status)
        campaign.write(OUT/'failures'/f"attempt_{status['run_attempts']:03d}.json",{'traceback':traceback.format_exc(),'budget':campaign.read(OUT/'budget.json'),'created_at_utc':campaign.stamp()});raise


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
