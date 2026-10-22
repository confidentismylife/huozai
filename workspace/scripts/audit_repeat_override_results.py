"""Independent standard-library audit of repeated-override confirmation."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import fmean
import argparse
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def csv_read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def label(incap,fed):
    if incap<=-.001:return 'beneficial'
    if incap>=.001:return 'harmful'
    if fed<=-.001:return 'beneficial'
    if fed>=.001:return 'harmful'
    return 'small_or_neutral'


def close(a,b):assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<1e-10,(a,b)


def run(full,continuations,analysis):
    full,continuations,analysis=map(lambda p:Path(p).resolve(),(full,continuations,analysis))
    prep=ROOT/'outputs/repeat_override_confirmation_preparation_20261007';plan=read(prep/'manifest.json');summary=read(analysis/'summary.json')
    hashes={}
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p;hashes[p]=h
    hashes[(prep/'manifest.json').relative_to(ROOT).as_posix()]=sha(prep/'manifest.json')
    for folder in (full,continuations,analysis):
        assert folder.parent==ROOT/'outputs'
        m=read(folder/'completion_manifest.json');assert m['state']=='complete'
        for p,h in m['files'].items():assert sha(ROOT/p)==h,p;hashes[p]=h
        hashes[(folder/'completion_manifest.json').relative_to(ROOT).as_posix()]=sha(folder/'completion_manifest.json')
    assert sha(prep/'manifest.json')==summary['preparation_sha256']
    episodes=csv_read(full/'episodes.csv');index={r['case_id']:r for r in episodes};assert len(index)==2080
    effect_checks=0
    for e in summary['paired_effects']:
        differences=[]
        for b in episodes:
            if b['source_mode']!=e['mode'] or b['policy']!=e['baseline']:continue
            parts=b['case_id'].split('|');parts[-1]=e['candidate']
            if e['candidate']=='pure_rule':parts[3]='0'
            a=index['|'.join(parts)]
            differences.append({m:float(a[m])-float(b[m]) for m in e['mean_differences']})
        assert len(differences)==200
        for m,v in e['mean_differences'].items():close(fmean(d[m] for d in differences),v)
        assert Counter(label(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in differences)==e['classes']
        effect_checks+=1
    cf=csv_read(continuations/'terminal_rollouts.csv');cf_index={(r['state_id'],int(r['future_seed']),r['future_policy'],int(r['forced_action'])):r for r in cf}
    assert len(cf_index)==len(cf)==2480
    states={r['state_id']:r for r in map(json.loads,(full/'selected_states.jsonl').read_text(encoding='utf-8').splitlines())}
    events=read(analysis/'event_effects.json');assert len(events)==len(states)==248
    for e in events:
        s=states[e['state_id']]
        assert e['first_only_accept']==(s['offers_before']==0)
        assert e['actual_accept']==(s['actual_action']!=s['base_action'])
        fitted=read(prep/'frozen_public_gate.json')
        assert e['consensus_accept']==bool(fitted['cells'].get(s['signature'],{}).get('all_supported',False))
        futures=[s['population_seed']+770000]+([] if s['behavior']=='rule' else plan['frozen_future_seeds'])
        for followup,name in [('common_ppo','local_class'),('restore_parent','restored_class')]:
            differences=[]
            for seed in futures:
                a=cf_index[(e['state_id'],seed,followup,s['candidate_action'])];b=cf_index[(e['state_id'],seed,followup,s['base_action'])]
                differences.append({m:float(a[m])-float(b[m]) for m in e[followup]})
            for m,v in e[followup].items():close(fmean(d[m] for d in differences),v)
            assert label(fmean(d['incapacitation_rate'] for d in differences),fmean(d['mean_cumulative_fed'] for d in differences))==e[name]
            assert Counter(label(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in differences)==e[followup+'_future_classes']
    confusion_checks=0
    for name,group in [('shadow_gate_confusion.json',False),('actual_path_gate_confusion.json',True)]:
        for r in read(analysis/name):
            rows=[e for e in events if e['mode']==r['mode'] and e['policy']==(r['gate'] if group else 'preserve_always')]
            if r['subset']=='first_offers':rows=[e for e in rows if e['offers_before']==0]
            if r['subset']=='repeat_offers':rows=[e for e in rows if e['offers_before']>=1]
            if r['subset']=='previously_executed_override':rows=[e for e in rows if e['overrides_before']>=1]
            field='actual_accept' if group else {'first_only':'first_only_accept','public_consensus':'consensus_accept'}[r['gate']]
            counts=Counter(('accept_' if e[field] else 'reject_')+e[r['label']] for e in rows)
            assert len(rows)==r['states'] and dict(counts)==r['counts']
            harmful=counts['accept_harmful']+counts['reject_harmful'];beneficial=counts['accept_beneficial']+counts['reject_beneficial']
            assert harmful==r['harmful_opportunities'] and beneficial==r['beneficial_opportunities']
            if harmful:close(r['harmful_block_rate'],counts['reject_harmful']/harmful)
            else:assert r['harmful_block_rate'] is None
            if beneficial:close(r['beneficial_false_reject_rate'],counts['reject_beneficial']/beneficial)
            else:assert r['beneficial_false_reject_rate'] is None
            confusion_checks+=1

    # Read-only diagnostic: does the fixed grid conflate different raw public states?
    # These confirmation labels are not used to change or fit a new controller.
    reference={e['state_id']:e for e in events if e['policy']=='preserve_always'}
    feature_keys={};previous={};trajectory_digest={};full_chains=defaultdict(list)
    with (full/'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for t in map(json.loads,f):
            cid=t['case_id'];sid=f"{cid}|{t['source_step']}"
            if cid.endswith('|preserve_always'):
                if sid in reference:
                    common=[t['observation'],t['base_action'],t['candidate_action'],t['offers_before'],t['overrides_before']]
                    history=[common,previous.get(cid),trajectory_digest.get(cid,'initial')]
                    feature_keys[sid]=dict(coarse=t['signature'],current=json.dumps(common,separators=(',',':')),
                                          full_history=json.dumps(history,separators=(',',':')))
                previous[cid]=t['observation']
                old_digest=trajectory_digest.get(cid,'initial')
                trajectory_digest[cid]=hashlib.sha256(json.dumps([old_digest,t['observation'],t['requested_action'],t['actual_room_capacity'],t['actual_room_count']],separators=(',',':')).encode()).hexdigest()
            if '|qwen14b_measured_arrival_cache|' in cid and (cid.endswith('|preserve_always') or cid.endswith('|first_only')):
                full_chains[cid].append([t[k] for k in ('source_step','requested_action','actual_joint_action','actual_room_capacity','actual_room_count','zone_counts_after','zone_fed_increments')])
    same_delayed=0
    for cid,chain in full_chains.items():
        if cid.endswith('|first_only'):
            assert chain==full_chains[cid.rsplit('|',1)[0]+'|preserve_always'];same_delayed+=1
    assert same_delayed==200
    collisions=[]
    for subset in ('all','repeat'):
        selected=[e for e in reference.values() if subset=='all' or e['offers_before']>=1]
        for representation in ('coarse','current','full_history'):
            cells=defaultdict(list)
            for e in selected:cells[feature_keys[e['state_id']][representation]].append(e)
            mixed=[rs for rs in cells.values() if {'harmful','beneficial'}<=set(e['local_class'] for e in rs)]
            collisions.append(dict(subset=subset,representation=representation,public_cells=len(cells),opportunities=len(selected),
                cells_with_both_harm_and_benefit=len(mixed),opportunities_in_mixed_cells=sum(len(rs) for rs in mixed),
                meaning='Retrospective descriptive collision check only; feature sufficiency or generalization is not proven.'))
    result=dict(state='passed',full_episodes=2080,terminal_rollouts=2480,event_effects_recomputed=248,complete_pair_groups_recomputed=effect_checks,
        confusion_groups_recomputed=confusion_checks,hashes_checked=len(hashes),delayed_first_only_equals_always_complete_trajectories=same_delayed,
        public_feature_collisions=collisions,gate_refitted=False,new_llm_requests=0,new_ppo_training=0,reserved_final_scenarios_opened=0,candidate_adopted=False)
    out=ROOT/'outputs'/('repeat_override_independent_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    save=lambda p,obj:p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(out/'summary.json',result);save(out/'checked_hashes.json',hashes)
    save(out/'completion_manifest.json',dict(state='complete',files={p.relative_to(ROOT).as_posix():sha(p) for p in [out/'summary.json',out/'checked_hashes.json',Path(__file__)]}))
    print(json.dumps(dict(output=str(out),**result),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',required=True);parser.add_argument('--continuations',required=True);parser.add_argument('--analysis',required=True);args=parser.parse_args();run(args.full,args.continuations,args.analysis)
