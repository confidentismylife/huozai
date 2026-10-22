"""Audit and describe frozen repeated-override confirmation without refitting."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import fmean
import argparse
import json
import math
import sys

from run_repeat_override_confirmation import (
    ROOT,PREP,MODES,METRICS,ZONE_NAMES,PROTECTED,verify,verify_completion,read,sha,save,csv_read,csv_write,unique_output,
    public_signature,consensus_decision,evidence_class,context,update_context,
)

REPORT=ROOT/'reports/再次覆盖PPO门控确认实验报告_2026-10-07.md'
MEASURES=(*METRICS,'episode_end_seconds',*(f'fed_{z}' for z in ZONE_NAMES),*(f'person_seconds_{z}' for z in ZONE_NAMES))


def close(a,b,label):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a)-float(b))<1e-9,(label,a,b)


def gate_counts(states,decision_field,label_field):
    counts=Counter()
    for s in states:
        action='accept' if s[decision_field] else 'reject'
        counts[f'{action}_{s[label_field]}']+=1
    n=len(states);harm=counts['accept_harmful']+counts['reject_harmful'];benefit=counts['accept_beneficial']+counts['reject_beneficial']
    accepted=sum(counts['accept_'+k] for k in ('harmful','beneficial','small_or_neutral'))
    return dict(states=n,counts=dict(counts),harmful_opportunities=harm,beneficial_opportunities=benefit,
        harmful_block_rate=None if not harm else counts['reject_harmful']/harm,
        beneficial_false_reject_rate=None if not benefit else counts['reject_beneficial']/benefit,
        accepted_harm_fraction=None if not accepted else counts['accept_harmful']/accepted)


def run(full,continuations):
    verify();full,continuations=Path(full).resolve(),Path(continuations).resolve()
    for p in (full,continuations):assert p.parent==ROOT/'outputs';verify_completion(p)
    assert not REPORT.exists()
    plan=read(PREP/'manifest.json');fitted=read(PREP/'frozen_public_gate.json');fs,cs=read(full/'summary.json'),read(continuations/'summary.json')
    assert fs['preparation_sha256']==cs['preparation_sha256']==sha(PREP/'manifest.json')
    assert cs['full_manifest_sha256']==sha(full/'completion_manifest.json')
    assert not any(fs['violations'].values()) and not any(cs['violations'].values())
    rows=csv_read(full/'episodes.csv');index={r['case_id']:r for r in rows}
    assert len(rows)==len(index)==2080
    assert Counter(r['policy'] for r in rows)==Counter(no_advice=400,original_hint=400,preserve_always=400,first_only=400,public_consensus=400,pure_rule=80)
    expected_cases=set()
    for rec in plan['scene_records']:
        for rep in range(4):
            pop=plan['population_prefix']+rec['scene_index']*100+rep
            for behavior in plan['behaviors']:
                for mode in MODES:
                    for seed in plan['training_seeds']:
                        for policy in plan['policies']:
                            expected_cases.add('|'.join(map(str,(rec['scene']['scenario_id'],behavior,pop,seed,mode,policy))))
                    expected_cases.add('|'.join(map(str,(rec['scene']['scenario_id'],behavior,pop,0,mode,'pure_rule'))))
    assert set(index)==expected_cases
    assert len({r['population_seed'] for r in rows})==20
    with (full/'selected_states.jsonl').open(encoding='utf-8') as f:states={r['state_id']:r for r in map(json.loads,f)}
    assert len(states)==fs['selected_states']==cs['exact_realized_suffixes']
    aggregates=defaultdict(lambda:dict(ctx=context(),steps=0,fed=[0.]*5,people=[0]*5,incap=[0]*5,counts=Counter(),congestion=0.))
    actual_steps=0;selected_ids=set();violations=Counter();signature_checks=0
    with (full/'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for t in map(json.loads,f):
            actual_steps+=1;cid=t['case_id'];a=aggregates[cid];r=index[cid];s=t['public_snapshot'];ctx=a['ctx']
            assert t['source_step']==a['steps'] and len(t['observation'])==34 and not any(t['observation'][28:33])
            assert ctx['offers']==t['offers_before'] and ctx['overrides']==t['overrides_before'] if r['policy']!='pure_rule' else True
            if r['policy']!='pure_rule':
                sig=public_signature(s,ctx['previous'],t['base_action'],t['candidate_action'],ctx['offers'],ctx['overrides'])
                supported,reason=consensus_decision(fitted,sig)
                assert sig==t['signature'] and supported==t['consensus_accept'] and reason==t['consensus_reason']
                assert (ctx['offers']==0)==t['first_only_accept'];signature_checks+=1
                assert t['eligible']==(t['candidate_action']!=t['base_action'])
                if t['eligible'] and 8 not in (t['candidate_action'],t['base_action']):
                    assert t['candidate_action']%4==t['base_action']%4
                    assert t['candidate_action']//4==t['hint_action']//4
            eligible=[not s['exit_closed'][b] and s['smoke_increment_per_step'][3+b]<.15 for b in range(2)]
            limit=max(0,min(20,sum(s['exit_capacity_persons_per_step'][b] for b in range(2) if eligible[b]),45-round(s['occupancy_fraction'][2]*240)))
            action=t['requested_action'];mask=[eligible[i//4] and (not s['sign_hold_remaining_steps'] or i//4==s['actual_sign']) and (i%4+1)*5<=limit for i in range(8)]+[True]
            assert mask[action]
            violations['actuator_mismatch']+=not(action==t['applied_action']==t['actual_joint_action'])
            violations['release_violations']+=not(0<=t['actual_room_count']<=t['actual_room_capacity']<=limit)
            violations['direction_violations']+=action!=8 and (t['actual_sign']!=action//4 or not eligible[action//4])
            violations['unavailable_entries']+=sum(int(t['branch_entries'][b]) for b in range(2) if not eligible[b])
            if r['policy'] in ('preserve_always','first_only','public_consensus') and t['eligible']:
                sid=f"{cid}|{t['source_step']}";state=states[sid]
                assert state['actual_action']==action and state['signature']==t['signature']
                assert state['legal_actions']==[i for i,v in enumerate(mask) if v]
                for z in range(5):
                    close(state['past_zone_fed'][z],a['fed'][z],('past_fed',sid))
                    close(state['past_zone_person_steps'][z],a['people'][z],('past_people',sid))
                selected_ids.add(sid)
            a['counts']['proposed_override_steps']+=t['eligible'];a['counts']['advice_override_steps']+=action!=t['base_action']
            a['counts']['blocked_override_steps']+=t['eligible'] and action==t['base_action']
            a['counts']['repeat_offer_steps']+=t['eligible'] and ctx['offers']>=1
            a['counts']['repeat_executed_steps']+=t['eligible'] and ctx['offers']>=1 and action!=t['base_action']
            for z in range(5):
                assert t['zone_person_steps'][z]==round(s['occupancy_fraction'][z]*240)
                a['fed'][z]+=t['zone_fed_increments'][z];a['people'][z]+=t['zone_person_steps'][z];a['incap'][z]+=t['zone_new_incapacitations'][z]
            a['congestion']+=sum(max(0,n/45-1) for n in t['zone_counts_after'])
            a['steps']+=1;a['remaining']=sum(t['zone_counts_after']);update_context(ctx,s,t,action)
    assert not any(violations.values()) and selected_ids==set(states) and set(aggregates)==set(index)
    for cid,a in aggregates.items():
        r=index[cid]
        for z,name in enumerate(ZONE_NAMES):
            close(r[f'fed_{name}'],a['fed'][z]/240,(cid,'fed',name));close(r[f'person_seconds_{name}'],a['people'][z]*5/240,(cid,'time',name));close(r[f'incap_count_{name}'],a['incap'][z],(cid,'incap',name))
        close(r['mean_cumulative_fed'],sum(a['fed'])/240,(cid,'total_fed'));close(r['incapacitation_rate'],sum(a['incap'])/240,(cid,'incap'))
        close(r['evacuation_rate'],1-(sum(a['incap'])+a['remaining'])/240,(cid,'evacuation'));close(r['remaining'],a['remaining'],(cid,'remaining'))
        close(r['episode_end_seconds'],a['steps']*5,(cid,'end'));close(r['cumulative_congestion'],a['congestion'],(cid,'congestion'))
        for k,v in a['counts'].items():assert int(r[k])==v
        if r['all_evacuated_seconds']:
            assert not sum(a['incap']) and a['remaining']==0;close(r['all_evacuated_seconds'],r['episode_end_seconds'],(cid,'success_time'))
        else:assert sum(a['incap']) or a['remaining']
    # Independent paired complete-episode calculation from CSV rows.
    effects=[]
    comparisons=[(p,'no_advice') for p in ('original_hint','preserve_always','first_only','public_consensus','pure_rule')]
    comparisons += [(p,b) for p in ('first_only','public_consensus') for b in ('preserve_always','original_hint')]
    strata=[]
    for mode in MODES:
        for candidate,baseline in comparisons:
            pairs=[];scene=defaultdict(list);behavior=defaultdict(list);weight=defaultdict(list)
            for b in rows:
                if b['source_mode']!=mode or b['policy']!=baseline:continue
                parts=b['case_id'].split('|');parts[-1]=candidate
                if candidate=='pure_rule':parts[3]='0'
                r=index['|'.join(parts)];d={m:float(r[m])-float(b[m]) for m in MEASURES}
                pairs.append(d);scene[b['scenario_id']].append(d);behavior[b['behavior']].append(d);weight[b['training_seed']].append(d)
                strata.append(dict(mode=mode,candidate=candidate,baseline=baseline,scenario_id=b['scenario_id'],behavior=b['behavior'],training_seed=int(b['training_seed']),population_seed=int(b['population_seed']),differences=d))
            assert len(pairs)==200
            def means(ds):return {m:fmean(d[m] for d in ds) for m in MEASURES}
            effects.append(dict(mode=mode,candidate=candidate,baseline=baseline,paired_comparisons=200,mean_differences=means(pairs),
                scene_means={k:means(v) for k,v in scene.items()},behavior_means={k:means(v) for k,v in behavior.items()},checkpoint_means={k:means(v) for k,v in weight.items()},
                classes=dict(Counter(evidence_class(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in pairs))))
    cf=csv_read(continuations/'terminal_rollouts.csv');assert len(cf)==cs['terminal_rollouts']
    cf_index={};expected_checks=0
    for r in cf:
        key=(r['state_id'],int(r['future_seed']),r['future_policy'],int(r['forced_action']))
        assert key not in cf_index;cf_index[key]=r
        close(sum(float(r[f'fed_{z}']) for z in ZONE_NAMES),r['mean_cumulative_fed'],('cf_fed',key))
        close(sum(int(r[f'incap_count_{z}']) for z in ZONE_NAMES)/240,r['incapacitation_rate'],('cf_incap',key))
        assert not any(int(r[k]) for k in ('unavailable_entries','release_violations','direction_violations'))
        state=states[r['state_id']]
        if r['future_kind']=='realized' and r['future_policy']=='restore_parent' and int(r['forced_action'])==state['actual_action']:
            parent=index[state['case_id']]
            for m in (*MEASURES,*(f'incap_count_{z}' for z in ZONE_NAMES),'remaining'):close(r[m],parent[m],('realized_parent',key,m))
            expected_checks+=1
    assert expected_checks==len(states)
    event_rows=[]
    for sid,s in states.items():
        futures=[s['population_seed']+770000]+([] if s['behavior']=='rule' else plan['frozen_future_seeds'])
        diffs={}
        for followup in ('common_ppo','restore_parent'):
            ds=[]
            for seed in futures:
                a=cf_index[(sid,seed,followup,s['candidate_action'])];b=cf_index[(sid,seed,followup,s['base_action'])]
                ds.append({m:float(a[m])-float(b[m]) for m in MEASURES})
            diffs[followup]={m:fmean(d[m] for d in ds) for m in MEASURES}
            diffs[followup+'_future_classes']=dict(Counter(evidence_class(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in ds))
        d=diffs['common_ppo'];rest=diffs['restore_parent']
        record=dict(state_id=sid,scenario_id=s['scenario_id'],behavior=s['behavior'],population_seed=s['population_seed'],training_seed=s['training_seed'],mode=s['source_mode'],policy=s['policy'],source_step=s['source_step'],
            offers_before=s['offers_before'],overrides_before=s['overrides_before'],base_action=s['base_action'],candidate_action=s['candidate_action'],actual_action=s['actual_action'],
            first_only_accept=s['first_only_accept'],consensus_accept=s['consensus_accept'],consensus_reason=s['consensus_reason'],actual_accept=s['actual_action']!=s['base_action'],
            local_class=evidence_class(d['incapacitation_rate'],d['mean_cumulative_fed']),restored_class=evidence_class(rest['incapacitation_rate'],rest['mean_cumulative_fed']),
            common_future_harm_present=bool(diffs['common_ppo_future_classes'].get('harmful',0)),
            restored_future_harm_present=bool(diffs['restore_parent_future_classes'].get('harmful',0)),future_samples=len(futures),**diffs)
        event_rows.append(record)
    def subsets(rs):
        return {'all_offers':rs,'first_offers':[r for r in rs if not r['offers_before']],
                'repeat_offers':[r for r in rs if r['offers_before']>=1],
                'previously_executed_override':[r for r in rs if r['overrides_before']>=1]}
    shadow=[];own=[]
    for mode in MODES:
        reference=[r for r in event_rows if r['mode']==mode and r['policy']=='preserve_always']
        for gate,field in [('first_only','first_only_accept'),('public_consensus','consensus_accept')]:
            for subset,rs in subsets(reference).items():
                for label in ('local_class','restored_class'):
                    shadow.append(dict(mode=mode,gate=gate,subset=subset,label=label,**gate_counts(rs,field,label),
                        accepted_with_any_harmful_future=sum(r[field] and r['common_future_harm_present'] for r in rs),
                        unsupported_rejections=sum(not r[field] and r['consensus_reason']=='unsupported_public_state' for r in rs) if gate=='public_consensus' else None))
            actual=[r for r in event_rows if r['mode']==mode and r['policy']==gate]
            for subset,rs in subsets(actual).items():
                for label in ('local_class','restored_class'):
                    own.append(dict(mode=mode,gate=gate,subset=subset,label=label,**gate_counts(rs,'actual_accept',label)))
    out=unique_output('repeat_override_confirmation_analysis')
    save(out/'paired_full_effects.json',effects);save(out/'paired_individual_differences.json',strata)
    save(out/'event_effects.json',event_rows);save(out/'shadow_gate_confusion.json',shadow);save(out/'actual_path_gate_confusion.json',own)
    summary=dict(full_episodes=2080,terminal_rollouts=len(cf),selected_states=len(states),actual_control_steps=actual_steps,signature_decisions_recomputed=signature_checks,
        exact_realized_suffixes=expected_checks,exact_suffix_control_steps=cs['exact_suffix_control_steps'],physical_violations=dict(violations),
        unique_archived_states=fitted['unique_archived_states'],archived_public_cells=len(fitted['cells']),supported_public_cells=sum(c['all_supported'] for c in fitted['cells'].values()),
        shadow_confusion=shadow,actual_path_confusion=own,paired_effects=effects,
        no_new_llm_requests=True,new_ppo_training=0,reserved_final_scenarios_opened=0,candidate_adopted=False,qualification_claim=False,
        full_source=str(full),continuation_source=str(continuations),preparation_sha256=sha(PREP/'manifest.json'),
        protected_sha256={p:sha(ROOT/p) for p in PROTECTED},tests_passed=17,
        decision='Finite public-support and repeat-abstention development confirmation only; no adoption or PPO training expansion; final bank remains sealed.')
    save(out/'summary.json',summary);write_report(out,summary)
    verify();save(out/'completion_manifest.json',dict(state='complete',files={p.relative_to(ROOT).as_posix():sha(p) for p in [*out.iterdir(),REPORT,Path(__file__)] if p.is_file()}))
    print(json.dumps(dict(output=str(out),report=str(REPORT),full_episodes=2080,terminal_rollouts=len(cf),reference_repeat_local=[r for r in shadow if r['subset']=='repeat_offers' and r['label']=='local_class'],
        complete_gate_vs_ppo=[dict(mode=e['mode'],candidate=e['candidate'],fed=e['mean_differences']['mean_cumulative_fed'],incap_pp=100*e['mean_differences']['incapacitation_rate']) for e in effects if e['candidate'] in ('first_only','public_consensus') and e['baseline']=='no_advice']),ensure_ascii=False),flush=True)


def write_report(out,s):
    names={'no_advice':'同权重无建议PPO','original_hint':'原方向提示','preserve_always':'保留档位、持续提示','first_only':'仅首次覆盖','public_consensus':'公开状态证据门控','pure_rule':'同信息规则'}
    modes={MODES[0]:'即时缓存',MODES[1]:'实测延迟'}
    lines=['# 当前是否值得再次覆盖PPO：新人口开发确认','',
        '日期：2026年10月7日。使用真实Qwen3 14B缓存与既有五套PPO权重；门控只读取公开观测、前一步公开人数变化、建议历史和基线/候选动作。确认数据不用于门控拟合或参数选择。','',
        '## 1. 冻结设计与完成量','',
        f"完成2,080个完整回合、{s['terminal_rollouts']}条终点反事实、{s['selected_states']}个真实建议机会及{s['actual_control_steps']}个实际控制步。逐步重算{s['signature_decisions_recomputed']}次门控特征和决策；{s['exact_realized_suffixes']}个原实际续跑、{s['exact_suffix_control_steps']}个后续控制步全部一致。17项测试通过，动作不一致和物理限制违规均为0。",'',
        '五个开发情景仍为AULL0010、0013、0019、0020、0024，共用同一五区域图结构。每个情景四批全新人口，共20批，使用1497100000前缀；两种人群行为、五套既有权重601—605、即时与实测延迟模式，指示牌保持2步。每种PPO控制400回合，规则单独80回合。','',
        '每行完整结果为200个配对比较。人口、模式、权重和随机未来不是独立建筑；只有这五个选过的参数情景，本轮不作独立场景显著性声明。旧1487100000人口没有冒充新确认人口。无建议PPO是既有同权重消融，不是新训练的纯PPO。','',
        '## 2. 两种门控分别检验什么','',
        '仅首次覆盖：每回合最多采纳第一次真正有机会改变PPO的建议，后续机会拒绝。它只是诊断基线，不判断当前净收益。','',
        f"公开状态证据门控：从旧44个状态的反事实构建固定查表，去除重复时序副本后有{s['unique_archived_states']}个归档状态，形成{s['archived_public_cells']}个公开特征格，其中{s['supported_public_cells']}格的所有已知配对未来都支持收益，且没有失能增加。其他格与新状态保留PPO。查表不是校准概率模型，也不证明某格未来永远有益。",'',
        '特征包括当前基线与保留档位候选动作、此前建议机会及实际覆盖次数、公开时间/方向/保持时间、出口容量、预报源与到预报起火的步数；五区域人数按5人分格、烟气按0.005分格、前一步人数变化按5人分格。特征不读取情景编号、人群模型编号、人口/权重种子、个体属性、未来真值或终点答案。全部分格及接受条件在新人口运行前冻结，无确认后调参。','',
        '两种门控的候选都保留同状态PPO的放行档位，继续执行同一个物理mask和时间戳门控。原方向提示作为完整回合对照，便于分开容量修复与再次覆盖的影响。','',
        '## 3. 同一批候选状态：拦住多少有害、错拒多少有益','',
        '主要对照在“保留档位、持续提示”的真实轨迹上，对同一批候选状态影子计算两种门控。每个状态强制候选或PPO一步，随后统一使用无建议PPO直到终点；Rule用实际未来，概率行为用实际未来加三个冻结未来。先按平均失能差±0.001，再按平均FED差±0.001分类。中性拒绝不计为拦住有害建议。','',
        '|时序|门控|机会类型|状态数|有害：拦住/总数|有益：误拒/总数|接受的有害|', '|---|---|---|---:|---:|---:|---:|']
    for r in s['shadow_confusion']:
        if r['label']!='local_class' or r['subset'] not in ('first_offers','repeat_offers'):continue
        c=r['counts'];lines.append(f"|{modes[r['mode']]}|{names[r['gate']]}|{'首次' if r['subset']=='first_offers' else '再次'}|{r['states']}|{c.get('reject_harmful',0)}/{r['harmful_opportunities']}|{c.get('reject_beneficial',0)}/{r['beneficial_opportunities']}|{c.get('accept_harmful',0)}|")
    lines+=['','“再次”定义为此前已经出现过一次可改变动作的建议机会，即offers_before≥1；此前已经实际覆盖过的子集另存。两种时序可能重复同一人口/状态，不合并计独立样本。','',
        '这些计数是同状态局部终点反事实；拒绝某次局部有害动作，不等于整个回合必然获救。另做强制一步后恢复父策略的反事实，检查建议重复和后续控制带来的交互。其分类及门控混淆表保存在JSON；实际门控自身轨迹也单独评价，不把不同轨迹的机会数直接当成同状态比较。','',
        '## 4. 完整疏散回合效果','',
        '以下均为候选减对照，FED和失能率正差较差，成功率正差较好。终止时间不是全员成功疏散时间；全部区域FED与区域人秒也保留。','',
        '|时序|候选减对照|FED差|失能率差（百分点）|成功率差（百分点）|累计拥堵差|回合终止时间差（秒）|',
        '|---|---|---:|---:|---:|---:|---:|']
    for e in s['paired_effects']:
        d=e['mean_differences'];lines.append(f"|{modes[e['mode']]}|{names[e['candidate']]}减{names[e['baseline']]}|{d['mean_cumulative_fed']:+.6f}|{100*d['incapacitation_rate']:+.4f}|{100*d['evacuation_rate']:+.4f}|{d['cumulative_congestion']:+.4f}|{d['episode_end_seconds']:+.3f}|")
    lines+=['','按行为拆开：','', '|时序|门控减无建议PPO|行为|FED差|失能差（百分点）|', '|---|---|---|---:|---:|']
    for e in s['paired_effects']:
        if e['candidate'] not in ('first_only','public_consensus') or e['baseline']!='no_advice':continue
        for behavior,d in e['behavior_means'].items():lines.append(f"|{modes[e['mode']]}|{names[e['candidate']]}|{behavior}|{d['mean_cumulative_fed']:+.6f}|{100*d['incapacitation_rate']:+.4f}|")
    lines+=['','完整效果还按情景及权重拆开，防止一套权重的少数改动主导均值。全部原始配对差异可进一步复核，不把未覆盖PPO的零差计作LLM解决冲突。','',
        '## 5. 局限与本轮决策','',
        '保守查表可能大量拒绝：未知状态、相似状态的收益反转，以及不同未来出现有害结果都会拒绝。即使接受的有害数为0，也必须同时看有益误拒率、支持覆盖率和完整回合效果；不能把几乎不动作的门控称为高精度智能体。','',
        '旧反事实标签只是离线监督材料；新反事实终点仅用于评估，不返回在线门控。特征排除了编号，但容量、烟气和预报仍可构成这五个参数情景的特征指纹，所以新人口验证不等于新情景或新建筑泛化。门控没有校准概率、风险上界或现实安全资格。','',
        '本轮不自动采用候选、不扩大PPO训练、不重新请求LLM，也不调整有效期/保持时间/分格参数。48个保留最终情景继续封存。新确认人口在结果解释后成为开发材料，后续修改需要另行冻结确认数据。','',
        '## 6. 可复现材料','',
        f'- [冻结方案]({(PREP/"manifest.json").as_posix()})',f'- [冻结公开状态门控]({(PREP/"frozen_public_gate.json").as_posix()})',
        f'- [2,080个完整回合]({(Path(s["full_source"])/"episodes.csv").as_posix()})',
        f'- [逐状态两类终点反事实效果]({(out/"event_effects.json").as_posix()})',
        f'- [同状态影子门控混淆表]({(out/"shadow_gate_confusion.json").as_posix()})',
        f'- [实际门控轨迹混淆表]({(out/"actual_path_gate_confusion.json").as_posix()})',
        f'- [完整回合逐配对差异]({(out/"paired_individual_differences.json").as_posix()})',
        f'- [总审计]({(out/"summary.json").as_posix()})','',
        '模型来源为真实Qwen3 14B缓存。旧模型选择文件、旧开发台账、冻结权重、旧脚本及旧报告保持原哈希。新PPO训练0、新LLM请求0、保留最终情景读取0。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser();parser.add_argument('--full',required=True);parser.add_argument('--continuations',required=True);args=parser.parse_args();run(args.full,args.continuations)
