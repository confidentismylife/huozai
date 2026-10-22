"""Analyze separate primary and exploratory experiments without selecting winners."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fire_evac.campaign_extension import write_json
from fire_evac.deepseek_advice import encode_plan_features,parse_plan
from fire_evac.frozen_plan_library import FrozenPlanLibrary
from mappo_research_judgement import brief_judgement, render as render_judgement

OUT=ROOT/'outputs'/'mappo_six_hour_extension_20261022'
SOURCE=ROOT/'outputs'/'mappo_capacity_stabilization_20261022'
RAW='mappo_34_raw'
RULE='mappo_50_rule_plan'
CACHED='mappo_50_deepseek_retrieval'
EXPANDED='mappo_50_deepseek_expanded'
MASKED='mappo_50_masked_control'
METRICS=('incapacitation_rate','mean_cumulative_fed','evacuation_time_seconds',
         'cumulative_congestion','mean_release_fraction_nonwait')
LABELS={RAW:'34维原始MAPPO',RULE:'规则计划',CACHED:'旧DeepSeek计划库',
        EXPANDED:'新DeepSeek快照计划库',MASKED:'50维全零计划通道'}


def paired_effect(frame,group,candidate,baseline,metric,replicates=5000):
    subset=frame[frame[group].isin([candidate,baseline])]
    index=['training_seed','scenario_id','population_seed']
    if subset.duplicated(index+[group]).any():
        raise ValueError('duplicate paired observations')
    pivot=subset.pivot(index=index,columns=group,values=metric)
    if candidate not in pivot or baseline not in pivot or pivot[[candidate,baseline]].isna().any().any():
        raise ValueError('incomplete paired observations')
    differences=pivot[candidate]-pivot[baseline]
    matrix=differences.groupby(['training_seed','scenario_id']).mean().unstack('scenario_id')
    if matrix.isna().any().any():
        raise ValueError('incomplete crossed seed/scenario pairing')
    a=matrix.to_numpy(dtype=float)
    rng=np.random.default_rng(20261022)
    si=rng.integers(0,a.shape[0],size=(replicates,a.shape[0]))
    ci=rng.integers(0,a.shape[1],size=(replicates,a.shape[1]))
    crossed=a[si[:,:,None],ci[:,None,:]].mean(axis=(1,2))
    means=differences.groupby('training_seed').mean()
    seed_boot=means.to_numpy()[si].mean(axis=1)
    independent_ci=None
    if group=='arm':
        # Sensitivity for independently trained policies: preserve evaluation
        # scenario pairing but resample the two training-run indices separately.
        left=pivot[candidate].groupby(['training_seed','scenario_id']).mean().unstack('scenario_id').to_numpy()
        right=pivot[baseline].groupby(['training_seed','scenario_id']).mean().unstack('scenario_id').to_numpy()
        sj=rng.integers(0,a.shape[0],size=(replicates,a.shape[0]))
        independent=(left[si[:,:,None],ci[:,None,:]]-right[sj[:,:,None],ci[:,None,:]]).mean(axis=(1,2))
        independent_ci=np.quantile(independent,[.025,.975]).tolist()
    return {'mean':float(differences.mean()),
        'seed_ci95':np.quantile(seed_boot,[.025,.975]).tolist(),
        'crossed_ci95':np.quantile(crossed,[.025,.975]).tolist(),
        'independent_training_crossed_ci95':independent_ci,
        'training_seed_sd':float(means.std(ddof=1)) if len(means)>1 else None,
        'n_training_seeds':len(means),'n_scenarios':a.shape[1],
        'n_paired_episodes':len(differences),
        'by_training_seed':{str(s):float(v) for s,v in means.items()},
        'scope':'exploratory crossed bootstrap within this development scenario bank'}


def effects(frame,group,candidate,baseline):
    return {metric:paired_effect(frame,group,candidate,baseline,metric)
            for metric in METRICS if metric in frame}


def read_episodes(paths):
    frames=[]
    for path in paths:
        df=pd.read_csv(path)
        if 'training_seed' not in df:
            seed=next(int(p.name.split('_')[-1]) for p in path.parents if p.name.startswith('seed_'))
            df.insert(0,'training_seed',seed)
        frames.append(df)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def method_table(data,group='arm'):
    if data.empty:return '尚未完成。'
    byseed=data.groupby(['training_seed',group]).agg(
        incapacitation_rate=('incapacitation_rate','mean'),
        time=('evacuation_time_seconds','mean'),
        fed=('mean_cumulative_fed','mean'),
        congestion=('cumulative_congestion','mean'),
        wait=('wait_fraction','mean'),
        release=('mean_release_fraction_nonwait','mean'))
    rows=['| 方法 | 已完成种子 | 失能率 | 疏散时间s | 累积暴露 | 拥堵 | 等待步占比 | 非等待请求放行均值 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,grouped in byseed.groupby(level=1):
        rows.append(f'| {LABELS.get(name,name)} | {len(grouped)} | '
            f'{100*grouped.incapacitation_rate.mean():.3f}% | {grouped.time.mean():.2f} | '
            f'{grouped.fed.mean():.4f} | {grouped.congestion.mean():.2f} | '
            f'{100*grouped.wait.mean():.2f}% | {grouped.release.mean():.3f} |')
    return '\n'.join(rows)


def formatted_effect(value):
    mean=100*value['mean'];lo,hi=[100*v for v in value['crossed_ci95']]
    return f'{mean:+.3f}个百分点，开发种子×场景重采样95%区间[{lo:+.3f}, {hi:+.3f}]'


def complete_seed_subset(frame,arms):
    if frame.empty:return frame
    counts=frame.groupby('training_seed').arm.nunique()
    seeds=counts[counts==len(arms)].index
    return frame[frame.training_seed.isin(seeds)].copy()


def normalize_execution_routes(frame):
    result=frame.copy()
    joint=result.actual_joint_action.to_numpy(dtype=int)
    # A neutral sign with positive release is -1 in the physical joint protocol.
    # Its sign code 2 must never be mistaken for the route actor's WAIT code 2.
    route=np.where(joint==8,2,np.where(joint<0,-1,joint//4))
    result['executed_route']=route
    eligible=result.target_route.to_numpy(dtype=int)>=0
    result['executed_plan_route_match']=np.where(eligible,(route==result.target_route).astype(float),np.nan)
    return result


def actor_diagnostics(analysis):
    checkpoint_rows=[];selected_rows=[]
    inputs=[('original_screen',p) for p in sorted((SOURCE/'screen').glob('*/seed_*/summary.json'))]
    inputs += [('new_mechanism',p) for p in sorted((OUT/'mechanism').glob('*/seed_*/*/summary.json'))]
    for stage,path in inputs:
        item=json.loads(path.read_text(encoding='utf-8'))
        config=path.parents[1].name if stage=='original_screen' else path.parents[2].name
        checkpoints=json.loads((path.parent/'checkpoints.json').read_text(encoding='utf-8'))
        chosen=int(item['selected_checkpoint']['timesteps'])
        for checkpoint in checkpoints:
            training=checkpoint['training'];validation=checkpoint['validation']
            route=np.asarray(training['route_action_fractions'],dtype=float)
            flow=np.asarray(training['flow_action_fractions_nonwait'],dtype=float)
            hr=float(training['route_entropy']);hf=float(training['flow_entropy'])
            if not (np.isfinite(hr) and np.isfinite(hf) and -1e-6<=hr<=np.log(3)+1e-6 and -1e-6<=hf<=np.log(4)+1e-6):
                raise ValueError('Invalid independent actor entropy in '+str(path))
            if not np.isclose(route.sum(),1) or not (np.isclose(flow.sum(),1) or np.isclose(flow.sum(),0)):
                raise ValueError('Invalid actor action-frequency normalization in '+str(path))
            checkpoint_rows.append({'stage':stage,'config_id':config,'arm':item['arm'],
                'training_seed':item['seed'],'timesteps':checkpoint['timesteps'],
                'selected':int(checkpoint['timesteps']==chosen),
                'route_entropy':hr,'flow_entropy':hf,
                'train_route_north':route[0],'train_route_south':route[1],'train_route_wait':route[2],
                **{f'train_flow_{int(level*100)}':float(flow[i]) for i,level in enumerate((.25,.5,.75,1))},
                'validation_incapacitation_12':validation['mean_incapacitation_rate'],
                'validation_mean_release_12':validation['mean_release_fraction_nonwait']})
        selected=item['selected_checkpoint'];last=checkpoints[-1]
        selected_rows.append({'stage':stage,'config_id':config,'arm':item['arm'],'training_seed':item['seed'],
            'selected_timesteps':chosen,'final_timesteps':int(item['timesteps']),
            'selected_route_entropy':selected['training']['route_entropy'],
            'selected_flow_entropy':selected['training']['flow_entropy'],
            'selected_validation_rate_12':selected['validation']['mean_incapacitation_rate'],
            'last_validation_rate_12':last['validation']['mean_incapacitation_rate'],
            'last_minus_selected_rate_12':last['validation']['mean_incapacitation_rate']-selected['validation']['mean_incapacitation_rate'],
            'selected_model_re_evaluation_rate_24':item['validation']['mean_incapacitation_rate']})
    history=pd.DataFrame(checkpoint_rows);selections=pd.DataFrame(selected_rows)
    history.to_csv(analysis/'actor_checkpoint_diagnostics.csv',index=False,encoding='utf-8-sig')
    selections.to_csv(analysis/'checkpoint_selection_diagnostics.csv',index=False,encoding='utf-8-sig')
    return {stage:{'runs':len(grouped),'selected_before_final':int((grouped.selected_timesteps<grouped.final_timesteps).sum()),
        'mean_last_minus_selected_incapacitation_pp':float(100*grouped.last_minus_selected_rate_12.mean())}
        for stage,grouped in selections.groupby('stage')}


def permutation_strength():
    from run_mappo_capacity_stabilization_campaign import new_library,ShuffledPlanProvider
    audits=json.loads((OUT/'plan_bank'/'response_audit.json').read_text(encoding='utf-8'))
    expanded=FrozenPlanLibrary([(r['state'],parse_plan(r['plan'])) for r in audits if r['valid_model_response']])
    result={}
    for label,library in [('old',new_library()),('new',expanded)]:
        shifted=ShuffledPlanProvider(library);rows=[]
        for state,_ in library.records:
            native=library(state);alternative=shifted(state)
            delta=np.abs(encode_plan_features(native)-encode_plan_features(alternative))
            rows.append({'scenario_id':state['scenario_id'],'event_type':state['event_type'],
                'encoded_plan_changes':bool(np.any(delta>1e-6)),
                'first_route_changes':native.phases[0].route!=alternative.phases[0].route,
                'first_release_changes':native.phases[0].release_fraction!=alternative.phases[0].release_fraction,
                'total_encoded_absolute_change':float(delta.sum())})
        result[label]={'library_cases':len(rows),
            'encoded_changed_fraction':float(np.mean([r['encoded_plan_changes'] for r in rows])),
            'first_route_changed_fraction':float(np.mean([r['first_route_changes'] for r in rows])),
            'first_release_changed_fraction':float(np.mean([r['first_release_changes'] for r in rows])),
            'scope':'unweighted library-case audit; not actual runtime retrieval frequencies','rows':rows}
    write_json(OUT/'permutation_control_strength.json',result)
    return result


def audit_input_scope():
    from run_mappo_capacity_stabilization_campaign import make_domain_randomized_trust_bank
    scenes=make_domain_randomized_trust_bank(24,20261021,'DSG')
    train={s.scenario_id for s in scenes[:12]};by_id={s.scenario_id:s for s in scenes}
    states_path=OUT/'plan_bank'/'frozen_states.json'
    new=json.loads(states_path.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'outputs'/'deepseek_format_gate_20261021'/'frozen_development_states.json').read_text(encoding='utf-8'))
    if len(new)!=48 or not all(s['original_scenario_id'] in train for s in new):
        raise ValueError('New plan-bank snapshots are not limited to the current training split')
    if any('source_branch' in s or 'incapacitation_rate' in s for s in new):
        raise ValueError('Unexpected hidden source or endpoint label in plan snapshots')
    expected=json.loads((OUT/'plan_bank'/'freeze_manifest.json').read_text(encoding='utf-8'))['states_sha256']
    if hashlib.sha256(states_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('New plan-bank input hash changed')
    shared=[]
    for state in old:
        scene=by_id.get(state['scenario_id'])
        if scene and abs(state['growth_rate']-scene.growth_rate)<1e-12 and state['onset_step']==scene.onset_step and state['forecast_source_branch']==scene.forecast_source_branch:
            shared.append({'scenario_id':scene.scenario_id,'current_split':'train' if scene.scenario_id in train else 'validation'})
    result={'new_observed_snapshots':len(new),'all_new_snapshots_from_current_training_split':True,
        'new_bank_input_hash_unchanged':True,'new_snapshots_have_no_true_source_or_endpoint_fields':True,
        'old_library_states':len(old),'old_library_shared_growth_onset_forecast':shared,
        'old_prototypes_overlapping_validation_parameters':sum(s['current_split']=='validation' for s in shared),
        'scope':'48 training decision snapshots are distinct from the separate sealed 48 final scenarios'}
    write_json(OUT/'input_scope_audit.json',result)
    return result


def write_meeting_brief(payload,analysis):
    """Numbers come directly from the saved paired effects, never hand-edited."""
    partial=bool(payload['partial'])
    title='五分钟组会汇报：运行中预览' if partial else '五分钟组会汇报：MAPPO与DeepSeek开发实验'
    header='本文件是运行中预览，部分方法的种子数不同，不能用作最终结论。' if partial else (
        '本文件概括已完成开发实验；它不包含最终封存场景结果。')
    primary=[]
    for baseline in (RAW,RULE):
        result=payload['primary_effects'].get(CACHED+'_minus_'+baseline)
        if result:
            primary.append(f'- 相对{LABELS[baseline]}：{formatted_effect(result["incapacitation_rate"])}。')
    mechanism=[]
    for cid,comparisons in payload['mechanism_effects'].items():
        for baseline in (RAW,RULE):
            result=comparisons.get(EXPANDED+'_minus_'+baseline)
            if result:
                n=result['incapacitation_rate']['n_training_seeds']
                mechanism.append(f'- {cid}，新库相对{LABELS[baseline]}（{n}种子）：'
                    f'{formatted_effect(result["incapacitation_rate"])}。')
    body=f'''# {title}

{header}

## 0:00—0:45：这轮解决什么问题

这轮检查生成的计划是否进入了实际控制，以及进入后有没有改善结果。环境是五分区、两出口的容量流模拟。DeepSeek给三阶段方向、全局放行比例和时长；MAPPO的两个actor分别决定引导方向与放行档位，每5秒执行一次，critic评估团队收益。这里的智能体是控制角色。

## 0:45—2:15：先讲两组结果

第一组保留原筛选规则，做独立训练三臂比较。已完成{len(payload['primary_training_seeds'])}个完整配对训练种子：

{chr(10).join(primary) if primary else '尚未产生完整配对结果。'}

第二组修正了计划阶段与烟气矛盾标志不一致的问题，比较弱、强计划监督。每臂目标是五个新种子、150000步；34维和50维全零计划基线独立训练，并在两个监督设置之间共享。新库根据48份真实训练决策快照生成，全部响应格式有效，但没有拥堵触发案例。后续独立审查已复现触发器缺陷：当较安全侧明确且计划同时启用烟气与拥堵触发时，即使烟气没有反转，判断分支也可能跳过拥堵检查。本轮保留冻结实现并披露，不能称为已完整验证三类事件。

{chr(10).join(mechanism) if mechanism else '新机制实验仍在训练，结果待齐全。'}

这些比较按各自协议解释。新库还增加了提示的单位说明，不能把新旧库差异全部归因于库覆盖。

另一个关键发现是旧环境在后续回合重新取系统随机数；旧训练无法仅凭初始化种子精确重建。原数据和权重完整保留，新矩阵统一派生每个回合的显式种子后重跑。旧协议比较与新矩阵分别报告，不选择性丢弃修正前的好结果或坏结果。

## 2:15—3:30：怎样判断LLM是否真的起作用

首先，逐步区分请求动作和实际动作；信号保持可能让两者不同。其次，同一模型分别接收真实、规则、置换和屏蔽计划。屏蔽后变差只说明它依赖这个通道，还要看真实计划是否优于规则与置换。最后，用独立训练的50维全零通道控制网络宽度差异。

首个原模型的即时/延迟×返回检查小实验中，屏蔽计划会明显恶化，但置换或规则并不必然比真实计划差；返回检查也没有自动产生长期收益。因此“拦住几条建议”不能代替失能、时间与拥堵的终点比较。单个模型、六个场景只能作诊断。

## 3:30—5:00：结论应该怎么说

{brief_judgement(payload)}

更听从计划可能改善，也可能损害疏散。只有独立基线和内容审计方向一致，才更支持计划内容有用；仍不能据此说真实建筑中的LLM控制已经有效。当前只有一个布局，聚合观测还隐藏了个体行为和暴露状态。在线API另含时间戳，不能算严格等信息比较。12步后计划年龄编码会饱和，计划到期也不会自动重问。

本轮API尝试{payload['budget']['attempted_requests']}次，保守计费上界累计{payload['budget']['actual_cost_upper_cny']:.4f}元，新调用硬上限2元。最终48个场景继续封存。下一步依据完整结果决定是否值得冻结一次最终评估，而不是看到某个好种子就继续调到它赢。

完整报告：[六小时实验报告]({(analysis/'六小时MAPPO与DeepSeek开发实验完整报告.md').as_posix()})。
原始数值：[analysis.json]({(analysis/'analysis.json').as_posix()})。
'''
    name='五分钟组会汇报_运行中预览.md' if partial else '五分钟组会汇报_MAPPO与DeepSeek.md'
    (analysis/name).write_text(body,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    status=json.loads((OUT/'extension_status.json').read_text(encoding='utf-8'))
    if not args.partial and status['status'] not in ('experiments_complete','complete'):
        raise RuntimeError('extension experiments are not complete')
    analysis=OUT/('progress_analysis' if args.partial else 'analysis')
    analysis.mkdir(parents=True,exist_ok=True)
    diagnostics=actor_diagnostics(analysis)
    manifest=json.loads((OUT/'extension_manifest.json').read_text(encoding='utf-8'))
    gate=json.loads((OUT/'plan_bank'/'gate.json').read_text(encoding='utf-8'))
    ledger=json.loads((OUT/'api'/'budget.json').read_text(encoding='utf-8'))
    states_path=OUT/'plan_bank'/'frozen_states.json'
    bank_freeze=json.loads((OUT/'plan_bank'/'freeze_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(states_path.read_bytes()).hexdigest()!=bank_freeze['states_sha256']:
        raise RuntimeError('frozen API input states changed')
    bank=json.loads((OUT/'plan_bank'/'response_audit.json').read_text(encoding='utf-8'))
    permutation=permutation_strength()
    input_scope=audit_input_scope()
    legacy_rows=[]
    for path in sorted((OUT/'legacy_uncontrolled_rng'/'mechanism').glob('*/seed_*/*/summary.json')):
        item=json.loads(path.read_text(encoding='utf-8'))
        legacy_rows.append({'source_path':str(path),'config_id':path.parents[2].name,
            'training_seed':int(item['seed']),'arm':item['arm'],**item['validation']})
    if legacy_rows:
        pd.DataFrame(legacy_rows).to_csv(analysis/'legacy_pre_seed_correction_results.csv',index=False,encoding='utf-8-sig')
    bank_description={'states':len(bank),'valid':sum(r['valid_model_response'] for r in bank),
        'event_counts':dict(Counter(r['state']['event_type'] for r in bank)),
        'unique_phase_sequences':len({json.dumps(r['plan']['phases'],sort_keys=True) for r in bank}),
        'confidence_distribution':dict(Counter(str(r['plan']['confidence']) for r in bank)),
        'state_safety_violations':gate['current_state_safety_violations']}
    primary=read_episodes(sorted((SOURCE/'confirmation').glob('seed_*/*/episode_results.csv'))+
        sorted((OUT/'additional_primary').glob('seed_*/*/episode_results.csv')))
    primary=complete_seed_subset(primary,(RAW,RULE,CACHED))
    primary.to_csv(analysis/'primary_episode_results.csv',index=False,encoding='utf-8-sig')
    primary_effects={}
    if not primary.empty:
        for baseline in (RAW,RULE):
            primary_effects[f'{CACHED}_minus_{baseline}']=effects(primary,'arm',CACHED,baseline)
    original_audit_frames=[]
    original_audit_path=SOURCE/'channel_audit.csv'
    if original_audit_path.exists():original_audit_frames.append(pd.read_csv(original_audit_path))
    for path in sorted((OUT/'additional_primary_audit').glob('seed_*/episode_results.csv')):
        original_audit_frames.append(pd.read_csv(path))
    original_audit=pd.concat(original_audit_frames,ignore_index=True) if original_audit_frames else pd.DataFrame()
    original_channel_effects={}
    if not original_audit.empty:
        original_audit.to_csv(analysis/'original_protocol_channel_audit.csv',index=False,encoding='utf-8-sig')
        for variant in ('rule','shuffled','masked'):
            original_channel_effects['native_minus_'+variant]=effects(
                original_audit,'provider_variant','native',variant)
    precision=read_episodes(sorted((OUT/'precision').glob('population_*.csv')))
    precision_effects={}
    if not precision.empty:
        precision.to_csv(analysis/'precision_episode_results.csv',index=False,encoding='utf-8-sig')
        for baseline in (RAW,RULE):
            precision_effects[f'{CACHED}_minus_{baseline}']=effects(precision,'arm',CACHED,baseline)
    mechanism_frames=[]
    configs=['e003_i002','e003_i005']
    for cid in configs:
        own=read_episodes(sorted((OUT/'mechanism'/cid).glob('seed_*/*/episode_results.csv')))
        for folder in ('shared_raw','shared_masked'):
            shared=read_episodes(sorted((OUT/'mechanism'/folder).glob('seed_*/*/episode_results.csv')))
            if not shared.empty:
                shared['shared_baseline']=True
                own=pd.concat([own,shared],ignore_index=True)
        if not own.empty:
            own['config_id']=cid
            if 'shared_baseline' not in own:own['shared_baseline']=False
            own['shared_baseline']=own.shared_baseline.fillna(False).astype(bool)
            mechanism_frames.append(own)
    mechanism=pd.concat(mechanism_frames,ignore_index=True) if mechanism_frames else pd.DataFrame()
    mechanism_effects={}
    if not mechanism.empty:
        mechanism.to_csv(analysis/'mechanism_episode_results.csv',index=False,encoding='utf-8-sig')
        for cid,grouped in mechanism.groupby('config_id'):
            mechanism_effects[cid]={}
            for candidate in (CACHED,EXPANDED):
                for baseline in (RAW,MASKED,RULE,CACHED):
                    if candidate==baseline:continue
                    a=grouped[grouped.arm==candidate]
                    b=grouped[grouped.arm==baseline]
                    common=sorted(set(a.training_seed)&set(b.training_seed))
                    if not common:continue
                    paired=grouped[grouped.training_seed.isin(common)&grouped.arm.isin([candidate,baseline])]
                    mechanism_effects[cid][candidate+'_minus_'+baseline]=effects(paired,'arm',candidate,baseline)
    audit_frames=[]
    for path in sorted((OUT/'mechanism_audit').glob('*/seed_*/*/episode_results.csv')):
        audit_frames.append(pd.read_csv(path))
    audit=pd.concat(audit_frames,ignore_index=True) if audit_frames else pd.DataFrame()
    audit_effects={}
    if not audit.empty:
        audit.to_csv(analysis/'mechanism_channel_audit.csv',index=False,encoding='utf-8-sig')
        for (cid,arm),grouped in audit.groupby(['config_id','arm']):
            audit_effects[cid+'/'+arm]={}
            for variant in ('rule','shuffled','masked'):
                audit_effects[cid+'/'+arm]['native_minus_'+variant]=effects(
                    grouped,'provider_variant','native',variant)
    online_paths=sorted((OUT/'online_pilot').glob('*/episode_results.csv'))
    online=pd.concat([pd.read_csv(p) for p in online_paths],ignore_index=True) if online_paths else pd.DataFrame()
    online_effects={}
    if not online.empty:
        online.to_csv(analysis/'online_episode_results.csv',index=False,encoding='utf-8-sig')
        for label,grouped in online.groupby('model_label'):
            if 'online' not in set(grouped.provider_variant):continue
            online_effects[label]={}
            for variant in ('native','rule','shuffled','masked'):
                if variant in set(grouped.provider_variant):
                    online_effects[label]['online_minus_'+variant]=effects(
                        grouped,'provider_variant','online',variant)
    latency_path=OUT/'online_latency_pilot'/'episode_results.csv'
    latency=pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()
    latency_effects={}
    if not latency.empty:
        latency.to_csv(analysis/'timing_return_check_results.csv',index=False,encoding='utf-8-sig')
        for candidate,baseline in [('instant_unchecked','native'),
            ('instant_checked','instant_unchecked'),('delayed_unchecked','instant_unchecked'),
            ('delayed_checked','delayed_unchecked'),('delayed_checked','instant_checked')]:
            if candidate in set(latency.provider_variant) and baseline in set(latency.provider_variant):
                latency_effects[candidate+'_minus_'+baseline]=effects(
                    latency,'provider_variant',candidate,baseline)
    traces=[]
    for path in sorted((OUT/'mechanism_audit').glob('*/seed_*/*/step_results.csv')):
        traces.append(pd.read_csv(path))
    execution_summary=[]
    if traces:
        trace=normalize_execution_routes(pd.concat(traces,ignore_index=True))
        execution=trace.groupby(['config_id','arm','provider_variant'],as_index=False).agg(
            control_steps=('decision_step','count'),
            requested_route_match=('requested_plan_route_match','mean'),
            executed_route_match=('executed_plan_route_match','mean'),
            executed_flow_match=('executed_plan_flow_match','mean'),
            compliance=('joint_action_compliance','mean'),
            actual_release=('actual_release_fraction','mean'),
            mean_plan_age_steps=('plan_age_steps','mean'),
            plan_age_encoding_saturated_fraction=('plan_age_steps',lambda values:float((values>=12).mean())))
        execution.to_csv(analysis/'execution_summary.csv',index=False,encoding='utf-8-sig')
        execution_summary=execution.to_dict(orient='records')
        action_rows=[]
        for identity,grouped in trace.groupby(['config_id','arm','training_seed','provider_variant']):
            context=dict(zip(['config_id','arm','training_seed','provider_variant'],identity))
            for source,codes in [('requested_joint_action',range(9)),('actual_joint_action',range(-1,9))]:
                counts=grouped[source].value_counts()
                for code in codes:
                    action_rows.append({**context,'action_source':source,'action_code':code,
                        'count':int(counts.get(code,0)),'control_step_fraction':float(counts.get(code,0)/len(grouped))})
        pd.DataFrame(action_rows).to_csv(analysis/'action_execution_distributions.csv',index=False,encoding='utf-8-sig')
    request_path=OUT/'api'/'requests.jsonl'
    requests=[json.loads(line) for line in request_path.read_text(encoding='utf-8').splitlines()] if request_path.exists() else []
    latencies=[r['latency_seconds'] for r in requests if r['source']=='api']
    payload={'partial':args.partial,'analyzed_at_utc':datetime.now(timezone.utc).isoformat(),
        'development_only':True,'final_test_scenarios_loaded':False,
        'deadline_utc':manifest['deadline_utc'],'bank':bank_description,'budget':ledger,
        'successful_API_latency_seconds':{'median':float(np.median(latencies)) if latencies else None,
            'p90':float(np.quantile(latencies,.9)) if latencies else None},
        'primary_training_seeds':sorted(primary.training_seed.unique().astype(int).tolist()) if len(primary) else [],
        'primary_effects':primary_effects,'mechanism_effects':mechanism_effects,
        'original_protocol_channel_effects':original_channel_effects,
        'population_precision_effects':precision_effects,
        'precision_population_replicates':int(precision.population_seed.nunique()) if len(precision) else 0,
        'channel_effects':audit_effects,'online_effects':online_effects,
        'timing_return_check_effects':latency_effects,
        'execution_summary':execution_summary,
        'actor_and_checkpoint_diagnostics':diagnostics,
        'input_scope_audit':input_scope,
        'execution_route_statistics':'derived from actual_joint_action: neutral+release=-1, wait=8 maps to route2; raw traces preserved',
        'library_permutation_strength':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in permutation.items()},
        'legacy_pre_seed_correction_complete_jobs':len(legacy_rows),
        'new_mechanism_episode_seed_schedule':'derived-explicit-uint32-v1',
        'full_training_replay':json.loads((OUT/'full_training_reproducibility_result.json').read_text(encoding='utf-8')) if (OUT/'full_training_reproducibility_result.json').exists() else None,
        'unique_mechanism_training_jobs':len(list((OUT/'mechanism').glob('*/seed_*/*/summary.json'))),
        'analysis_warning':'seed/scenario bootstrap is conditional on this small development bank; exploratory comparisons are not multiplicity-adjusted'}
    write_json(analysis/'analysis.json',payload)
    table_parts=[]
    if not mechanism.empty:
        for cid,grouped in mechanism.groupby('config_id'):
            table_parts.extend([f'### {cid}',method_table(grouped)])
    effect_lines=[]
    for comparison,values in primary_effects.items():
        effect_lines.append(f'- {comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    mechanism_lines=[]
    for cid,comparisons in mechanism_effects.items():
        for comparison,values in comparisons.items():
            mechanism_lines.append(f'- {cid}，{comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    precision_lines=[]
    for comparison,values in precision_effects.items():
        precision_lines.append(f'- {comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    independent_lines=[]
    for comparison,values in primary_effects.items():
        interval=values['incapacitation_rate']['independent_training_crossed_ci95']
        if interval:
            independent_lines.append(f'- {comparison}：独立重采样训练批次并共同重采样评估场景的95%敏感性区间'
                f'[{100*interval[0]:+.3f}, {100*interval[1]:+.3f}]个百分点。')
    channel_lines=[]
    for setting,comparisons in audit_effects.items():
        for comparison,values in comparisons.items():
            channel_lines.append(f'- {setting}，{comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    original_channel_lines=[]
    for comparison,values in original_channel_effects.items():
        original_channel_lines.append(f'- 原协议{comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    online_lines=[]
    for setting,comparisons in online_effects.items():
        for comparison,values in comparisons.items():
            online_lines.append(f'- {setting}，{comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    latency_lines=[]
    for comparison,values in latency_effects.items():
        latency_lines.append(f'- {comparison}：{formatted_effect(values["incapacitation_rate"])}。')
    report=f'''# MAPPO与DeepSeek六小时开发实验：{'运行中快照' if args.partial else '完整结果与解释'}

生成时间：{payload['analyzed_at_utc']}。{'实验仍在进行，以下不作为最终结论。' if args.partial else '所有结论均局限于开发环境。'}

## 1. 本轮要回答什么

先把容量MAPPO训练稳定，再区分三个问题：策略是否读了计划；读计划是否改变了实际动作；改变后是否比独立规则与纯强化学习更安全、更快。模型更听话不等于建议更好。

用户把总运行时间先延长到五小时，再增加到六小时，并允许增加DeepSeek调用。原启动时间为2026-10-22 10:17，六小时截止为16:17。原零API实验与新增API机制实验分别保存，不追溯改写原实验。

## 2. 场景和控制对象

沿用五分区、两出口的容量流开发模拟。24个生成开发场景中前12个训练，后12个用于检查点选择及开发评估。控制周期5秒，动作是北/南方向乘四档放行，加等待。两个actor分别控制方向与容量，共享团队critic。它们均读取同一控制器状态，这不是把每位行人训练成独立智能体的实验，也不代表已验证真实建筑。

方向动作改变引导标志，真人是否服从在这里由MixedLogit模拟；容量动作缩放“房间到交汇区”的全局放行能力，并非直接改变北、南出口的物理宽度或分别控制两个出口容量。聚合观测没有包含每个人的行为参数和个体累积暴露，不能因为只有34维就把它称为完全可观测的物理状态。

34维原始状态与50维计划状态保留同一组物理和预测信息；后者用三阶段方向、容量、时长替换旧的零建议位。新50维全零通道基线进一步控制网络输入宽度。训练步数与场景预算相同；不同输入宽度本身不等于完全相同的参数数量。

因此问题不在于34这个维数本身太小；应检查观测是否保留决策所需信息、计划是否提供有用的表示及时间结构。由相同观测产生的计划未必增加新的现场信息，但仍可能通过先验或表示改善有限训练预算下的学习；这种可能性需要对照检验，不能仅凭“是观测的函数”就排除。反过来，多加维度也不能凭空创造有效信息。本轮提示已取消“必须选择预测火源反侧”的硬映射，允许使用容量、烟气和人群观测制定三阶段计划；实际semantic_context仍为空，没有加入真实建筑语义、消防指令或RAG案例。因此本轮检验的是数值观测到计划的辅助作用，没有覆盖LLM处理额外现场语言信息的能力。

方向和容量actor分别计算自己的PPO概率比和裁剪损失，使用共享团队优势与critic；方向熵与容量熵分别加权。实际采样为等待时，PPO容量损失和容量熵项被屏蔽。辅助监督按当前建议阶段给方向、放行档位交叉熵标签，遇到烟气矛盾时权重归0；容量标签还在建议为等待时屏蔽。监督鼓励参考建议，不保证建议正确，也没有额外训练一个可靠的收益判断器。

此处“相同预算”指同样的环境步数、PPO更新轮数、隐藏层宽度和batch大小。辅助监督会增加计算，API生成和检索也有额外开销；本轮不是严格相同FLOPs或墙钟时间的比较。LLM计划与规则计划臂使用同样的监督设置。

还须区分单次训练预算与完整调参预算：原40个筛选作业只用于旧DeepSeek检索臂，随后三个确认臂沿用选中的共同设置；没有分别给原始MAPPO和规则臂各做同样的40作业搜索。因此本轮是共同设置下、单次训练步数一致的系统对照，不是总搜索计算量完全相同或各方法已各自充分调优的算法排名。

代码核查发现，原计划烟气矛盾标志始终参考第一阶段方向，旧库9/12、新库38/48计划存在阶段改道。新增机制实验在任何相关训练开始前改用当前阶段的矛盾标志，并对规则与LLM同时生效。原实验、原设置扩种子和原模型时序诊断保留原编码。因此不能把原筛选与新机制的差异全部解释为监督强弱；方法修正记录见method_correction_before_mechanism_training.json。

## 3. 原筛选和确认保持原样

原8个配置×5种子筛选结果保留，冻结规则选中e001_i000。规则先在可能时排除全回合满档放行比例超过0.85的配置，再按平均失能率排序，以容量和方向计划匹配率依次打破平局。单次训练的检查点也先区分满档比例是否超过0.85，再依失能率、暴露与时间排序。这是带行为约束的选择规则，不能简化成无条件的安全最优，也不能据此宣称已找出所有MAPPO策略中的最优基线。未因LLM采用率低而改选。检查点选择使用单人口种子的12回合；选择后CSV用两个人口种子的24回合重评估。两类数值不能混用。

原环境的训练reset(None)会在每个后续回合重新取系统随机数，因此原“训练种子”实为初始化种子/运行编号，不能据此完整重建训练权重，也不能声称三臂训练共享了完整随机数流。以下原协议结果只作保存权重的开发评估；评估时每个回合显式给定人口种子，可以重放。24回合中还包括用于检查点选择的12回合42001，另一半只是同场景的42002人口重复，不是独立场景测试。

{method_table(primary)}

当前完整配对训练种子数：{len(payload['primary_training_seeds'])}。

{chr(10).join(effect_lines) if effect_lines else '尚无完整配对结果。'}

旧训练的随机数未被完整控制，因此另给出不依赖两臂训练随机流匹配的敏感性分析；它保持评估场景配对，但分别重采样两组训练运行：

{chr(10).join(independent_lines) if independent_lines else '待完整配对。'}

原协议的同权重四通道审计与独立训练三臂分别解释：

{chr(10).join(original_channel_lines) if original_channel_lines else '原协议四通道审计尚未完成。'}

屏蔽会使模型遇到训练中少见的全零输入，因此即使屏蔽导致大幅恶化，也不能代替真实计划与规则/置换计划的内容比较。

剩余日历时间用于同一批固定权重的成组人口重复，已完成{payload['precision_population_replicates']}个人口种子。停止仅依赖预先固定的六小时截止时间，不根据效应或显著性决定是否继续。每个完整块包含全部完整训练种子、三臂、十二个开发验证场景。这些重复用于降低人口随机数噪声，不增加独立训练种子或场景数量，不与原两个人口种子的结果混成更大的独立n。

{chr(10).join(precision_lines) if precision_lines else '人口重复精度分析尚未完成。'}

## 4. 新DeepSeek计划库的接口验收

在实际训练开发场景的规则控制轨迹中采集状态，按观测去重后先冻结48份输入及散列，再调用API。48份中有效响应{bank_description['valid']}份，当前状态下首阶段封闭出口冲突{bank_description['state_safety_violations']}次；共有{bank_description['unique_phase_sequences']}种不同三阶段动作序列。

实际事件分布：{json.dumps(bank_description['event_counts'],ensure_ascii=False)}。没有真实触发拥堵事件，不能宣称新库覆盖了拥堵响应。置信度分布：{json.dumps(bank_description['confidence_distribution'],ensure_ascii=False)}。这些自报数值没有经过收益概率校准，不能直接当作可靠性。

原库与新库都属于“LLM生成计划＋数值近邻检索”；每步检索不等于每步调用LLM。

新库48份训练快照与封存的48个最终场景不是同一批数据。来源审计确认所有新快照属于当前训练场景，没有真实火源分支字段或终点结果标签。旧库则有{input_scope['old_prototypes_overlapping_validation_parameters']}个原型与当前验证场景的增长率、起火时间和预测源参数重合；旧库不能视为严格独立于这些验证参数。它没有使用这些回合的终点标签，但当前验证仍属于开发验证，不能包装成严格场景外推。

新库同时改变了决策快照覆盖和提示中的单位说明。新旧库差值不能仅归因于覆盖更广，也不能视作模型版本的独立比较。

## 5. 弱/强监督机制对照

容量熵固定0.03，计划辅助监督分别0.02与0.05；每设置五个新种子，每臂150000步。纯34维及50维全零基线在两个设置间复用同一组独立训练结果，不能把复用算作新增独立证据。设置的选择来自已见筛选结果，因此本部分是冻结后执行的机制探索，不能称为原预注册确认。

新增矩阵在发现随机种子缺陷后全部重新运行：训练reset(None)使用由该运行种子派生的确定性uint32回合种子，所有臂采用相同安排；显式评估种子原样传入。修正前已有{len(legacy_rows)}个完整新作业，全部保存在legacy_uncontrolled_rng，汇总在legacy_pre_seed_correction_results.csv，不参与修正后矩阵的合并；未按成绩选择保留。一个固定的150000步完整训练另行重放，核对五个检查点及评估CSV，不算额外独立种子。完整重放状态见analysis.json中的full_training_replay。

{chr(10).join(table_parts) if table_parts else '尚未完成。'}

{chr(10).join(mechanism_lines) if mechanism_lines else '配对差异待生成。'}

### 独立熵、动作分布与检查点诊断

{json.dumps(diagnostics,ensure_ascii=False)}

actor_checkpoint_diagnostics.csv逐检查点保存方向/容量独立熵、最近随机训练动作分布，以及同一12回合开发验证分数；checkpoint_selection_diagnostics.csv区分选中检查点、最后检查点和选中模型24回合重评估。不能把随机训练时的动作多样性当作部署时的确定性动作多样性。审计完成后，action_execution_distributions.csv另外报告请求与实际联合动作的控制步占比，包括实际中性放行代码-1。

检查点诊断图的曲线为训练种子均值、色带为种子最小至最大范围、散点为单个种子；这些描述性范围不是推断置信区间。推断区间仍来自单独的配对重采样分析。

## 6. 同模型四通道审计

在相同权重、场景与人口随机数下分别输入真实检索、规则、同事件置换、全部屏蔽计划。闭环结果体现改变计划后的连锁影响。置换与屏蔽会引入输入分布偏移，只能作为机制证据；独立训练基线仍不可省。

置换强度也经过核查：按库内案例等权计算，旧库置换后{100*permutation['old']['encoded_changed_fraction']:.1f}%的案例改变了实际21维计划输入，新库为{100*permutation['new']['encoded_changed_fraction']:.1f}%。部分计划的动作、时间和置信度编码相同，或只在未输入控制器的原因码上不同。因此置换并非每次都是有效输入干预；此处是库层面的等权比例，不能当作真实运行的检索频率。

{chr(10).join(channel_lines) if channel_lines else '尚未完成。'}

## 7. 请求动作与实际动作

新审计逐步记录requested_joint_action、actual_joint_action、actual_sign_action和actual_release_fraction。信号保持可能让实际方向与请求方向不同；中性方向与放行同时发生时实际动作记为-1，不硬套入九动作。容量匹配按实际整数取整容量计算。详见execution_summary.csv和各审计目录的step_results.csv。

actual_release_fraction表示实际应用的房间放行容量占基准容量的比例，不是实际通过人数占比。房间清空后仍可能记录容量设置；动作匹配本身不能证明改变了人流或提高了安全性，仍须看疏散终点与同模型干预结果。

派生执行统计以actual_joint_action重建实际路线：-1表示中性标志下放行，8才表示等待，避免把标志的中性代码2误当成路线actor的等待代码2。原始逐步日志保留；报告汇总使用修正后的物理语义。本轮两个冻结库都没有等待阶段，在线计划仍允许等待，因此该区分属于接口正确性要求。

当等待时不把遗留的方向信号当作放行方向，也不把等待作为容量actor的有效放行动作。请求计划匹配率与实际计划匹配率分别报告。

旧验证summary的route_plan_match和flow_plan_match只在置信权重为正、烟气矛盾标志为0等有效子集统计；不能直接解释为所有控制步的服从率。逐步审计的路线匹配按所有有计划的控制步计算，并另外区分请求与实际执行。

计划年龄另有边界：50维输入把年龄编码截断在12步，最后阶段在下次事件更新前持续有效；当前协议没有单独按三阶段时长耗尽触发新请求。实际年龄仍在逐步日志中保留，execution_summary.csv报告年龄编码饱和比例。不能把这种有限事件更新称为每次计划到期都会重新推理。

独立审查另外确认事件优先级存在实现缺陷：_event的elif链在计划启用smoke_reversal且当前较安全侧非空时进入烟气分支；即使没有发生反转，也不会继续检查congestion_threshold。最小复现保持北侧更安全，把交汇区人数从阈值下方升到46人（阈值45人）：同时启用烟气和拥堵触发时返回空事件，仅启用拥堵触发时正确返回拥堵事件。复现脚本为scripts/audit_mappo_event_priority.py，证据为event_trigger_diagnostic.json。原包装器和ActivePhase包装器都继承该逻辑。本轮训练已冻结，因此不在运行中更改；这些结果只能代表含此缺陷的事件策略，不能当作完整事件触发方案的性能。构造复现尚未量化此问题在既有轨迹中的频率及终点影响，48份快照没有拥堵事件也不能全部归因于这一缺陷。

## 8. 真正当前状态API的小规模验证

固定首个训练种子、前六个开发验证场景和新人口种子42003，用相同权重比较检索/规则/置换/屏蔽/当前状态API。此处生成的建议以当前观测为输入，不提供未来结果；接口失败与预算回退明确标记，不混称为LLM成功。

{chr(10).join(online_lines) if online_lines else '尚未完成。'}

即时API子实验在网络等待期间暂停仿真，因此是零仿真延迟上界，不能解释成现场实时收益。实际服务延迟单独保存。此小样本子实验不能替代多种子最终评估。

信息匹配还有一个明确边界：API快照包含时间戳和场景标识，而原34维控制器没有单独的绝对时间戳输入。当前规则计划与近邻检索的决策公式不使用这些标识；在线LLM可能使用它们。因此在线差异不能归为严格等信息条件下的语言推理增益。当前开发参数的增长率、起火时间与封闭结束时间未超过原数值特征的裁剪上限，但这不消除时间戳差异。

另以原模型固定种子进行即时/延迟×返回校验开/关的2×2诊断。延迟按原请求实测耗时向上取整到5秒周期，至少一周期；精确缓存重放原冷请求延迟。等待时MAPPO继续使用旧计划，初始先用规则。返回时按当前阶段重新检查封闭、烟气矛盾或过期。这个门控仅检查建议，所有分支的实际动作约束仍由相同物理环境执行，不能宣称它保证全程安全。

{method_table(latency,'provider_variant')}

{chr(10).join(latency_lines) if latency_lines else '时序诊断待完成。'}

以上时序诊断只有一个权重和六个开发场景，交叉区间不能表达训练种子不确定性。拒绝看似危险的建议后，长期疏散结果仍可能恶化；不能用拦截次数代替终点收益。

## 9. 费用与可追溯性

新增请求尝试{ledger['attempted_requests']}次，成功解析{ledger['successful_requests']}次。按高于官方峰时费率的计价上界累计{ledger['actual_cost_upper_cny']:.6f}元；尚未结清的超时预留{ledger['reserved_cny']:.6f}元。新增硬上限2元，未把15元余额全部当作应花完的预算。原账本费用另存，二者不覆盖。

密钥读取自Windows DPAPI当前用户凭据；请求缓存、状态、报告只保存状态、计划、用量、来源与延迟，不保存认证头。官方价格来源：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/ 。

## 10. 统计方法与结论边界

先按场景/人口种子做候选减基线的配对差，再按训练种子汇总；同时给出种子重采样和种子×场景交叉重采样区间。失能差为负表示候选更好。区间只反映这个小型开发场景库，不涵盖新建筑、火灾物理误差及真实人群差异。机制比较未进行多重性校正，不能按最显著的一项挑结论。

独立训练臂另报告分别重采样训练运行、共同重采样评估场景的敏感性区间；同权重的通道干预仍保留模型种子配对，不把其同一个权重当作两个独立模型。旧协议的配对仅指初始化编号与评估场景/人口条件，并不意味着相同完整训练轨迹。

原先提出的2个百分点可作为实践意义的参考尺度，不是对本轮观测反推的显著性阈值。小于该尺度的统计差异，也不能自动解释为工程优势。仿真中的“失能”不是已经验证的真实伤亡预测。

真实计划只优于屏蔽而不优于规则/置换时，只能支持计划通道作用。采用率增加而安全变差时，应报告模仿造成的代价。即使新计划库改善，也需说明收益来自当前覆盖、检索与控制器的整个组合，不能把全部改善归给语言模型本体。

## 11. 仍缺什么

最终48个封存场景继续不读取。本轮不提供真实建筑验证、真人改道验证、LLM置信度校准，或跨建筑泛化证据。新快照库缺少拥堵事件，已有证据表明部分拥堵触发被事件分支抑制；在线小样本暂停仿真，仍需对真实延迟与观测过时进行独立验证。若后续继续，首先修复触发器并另行冻结小规模功能验证，不直接扩大当前缺陷协议的训练量，也不将修正前后批次合并。

## 12. 复现入口

- 执行方案：docs/mappo_six_hour_extension_20261022.md。
- 总执行器：scripts/run_mappo_six_hour_extension.py。
- 同架构基线：scripts/run_mappo_masked_architecture_control.py。
- 分析：scripts/analyze_mappo_six_hour_extension.py。
- 预算、DPAPI读取、执行统计：src/fire_evac/campaign_extension.py。
- 输入冻结：plan_bank/frozen_states.json和freeze_manifest.json。
- 主/机制/在线逐回合数据及完整配对差异：本目录CSV与analysis.json。

运行完整分析：python scripts/analyze_mappo_six_hour_extension.py。实验进行期间仅允许加--partial输出运行中快照。

## 13. 图表与数据入口

![原设置逐训练种子配对差异]({(analysis/'figures'/'primary_paired_effects.png').as_posix()})

![新机制实验的每个训练种子结果]({(analysis/'figures'/'mechanism_seed_results.png').as_posix()})

![检查点验证曲线与容量actor独立熵]({(analysis/'figures'/'actor_checkpoint_diagnostics.png').as_posix()})

![时序与返回校验的小样本诊断]({(analysis/'figures'/'timing_return_check.png').as_posix()})

完整数值：[analysis.json]({(analysis/'analysis.json').as_posix()})。主实验逐回合数据：[primary_episode_results.csv]({(analysis/'primary_episode_results.csv').as_posix()})。所有图同时导出PNG、PDF及数据来源散列，图片只是现有数据的呈现。
'''
    judgement=render_judgement(payload,OUT)
    (analysis/'研究判断与下一步.md').write_text(judgement,encoding='utf-8')
    (analysis/'六小时MAPPO与DeepSeek开发实验完整报告.md').write_text(report+'\n'+judgement,encoding='utf-8')
    write_meeting_brief(payload,analysis)
    print(json.dumps({'partial':args.partial,'primary_seeds':len(payload['primary_training_seeds']),
        'mechanism_jobs':payload['unique_mechanism_training_jobs'],'report':str(analysis)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
