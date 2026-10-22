"""Frozen cache-only source sensitivity review of 246 existing choices."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "schemas/clear_context_cache_review_20261014.json"
OUT = ROOT / "outputs/clear_context_cache_review_20261014"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_json(path, obj): path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def table(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def cond(row):
    return tuple(int(row[k]) for k in ["assigned_left_crowd","assigned_right_crowd","knowledge_code"])


def fam(row):
    a,b,k=cond(row); return min(a,b),max(a,b),k


def eligible_gate(row):
    k=int(row['knowledge_code']); a=int(row['assigned_left']); b=int(row['assigned_right'])
    return k==0 or (k==3 and a!=b)


def prepare():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    gates=read_csv(ROOT/cfg['selection_source'])
    assert len(gates)==len({r['case_key'] for r in gates})==547
    selected=[r for r in gates if r['source_gate']=='comparatively_clear_under_README']
    assert {r['case_key'] for r in selected}=={r['case_key'] for r in gates if eligible_gate(r)}
    assert len(selected)==cfg['expected_records']
    conditions=sorted({(int(r['assigned_left']),int(r['assigned_right']),int(r['knowledge_code'])) for r in selected})
    families=sorted({(min(a,b),max(a,b),k) for a,b,k in conditions})
    assert len(conditions)==17 and len(families)==9
    protected=[ROOT/cfg['selection_source'],ROOT/cfg['prediction_source']]
    for folder in ['bristol_context_choice_benchmark_20261013','bristol_prior_mechanism_probe_20261013',
                   'bristol_no_prior_choice_benchmark_20261013','bristol_source_semantics_audit_20261014']:
        protected.extend(p for p in (ROOT/'outputs'/folder).iterdir() if p.is_file() and p.suffix.lower() in {'.json','.jsonl','.csv'})
    for name in ['bristol_context_choice_benchmark_20261013','bristol_prior_mechanism_probe_20261013',
                 'bristol_no_prior_choice_benchmark_20261013','bristol_source_semantics_audit_20261014']:
        protected.append(ROOT/'schemas'/f'{name}.json')
    protected.extend(ROOT/'scripts'/name for name in ['run_bristol_context_choice_benchmark.py',
        'run_bristol_prior_mechanism_probe.py','run_bristol_no_prior_choice_benchmark.py',
        'analyze_bristol_no_prior_choice_benchmark.py','audit_bristol_source_semantics.py'])
    manifest={'config_sha256':sha(CONFIG),'runner_sha256':sha(Path(__file__)),
        'protected_sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(protected))},
        'selected_case_keys':[r['case_key'] for r in selected],'selected_conditions':conditions,'selected_families':families,
        'selection_uses_outcomes':False,'new_LLM_requests':0,'new_model_fits':0,'PPO_updates':0,'map_code_changes':0}
    # JSON normalizes tuples as lists; compare the same serialized types.
    manifest=json.loads(json.dumps(manifest))
    OUT.mkdir(exist_ok=True)
    path=OUT/'frozen_review_manifest.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        assert {k:v for k,v in previous.items() if k!='frozen_at_utc'}==manifest
        return cfg,previous
    manifest['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    write_json(path,manifest)
    print(json.dumps({'status':'subset_review_plan_frozen','records':246,'conditions':17,'families':9}),flush=True)
    return cfg,manifest


def metrics(rows,method):
    cells=defaultdict(list)
    losses=[]; correct=[]; fair_correct=[]; tied=[]
    for row in rows:
        raw=float(row[method]); assert math.isfinite(raw) and 0<=raw<=1
        p=min(1-1e-9,max(1e-9,raw)); y=int(row['observed_exit'])==1
        loss=-math.log(p if y else 1-p); brier=(p-y)**2
        losses.append(loss); correct.append((raw>=.5)==y); tied.append(raw==.5)
        fair_correct.append(.5 if raw==.5 else float((raw>.5)==y))
        cells[cond(row)].append((raw,p,y,loss,brier))
    cell_nll=[]; cell_brier=[]; errors=[]
    for values in cells.values():
        ps=[v[0] for v in values]; assert max(ps)-min(ps)<=1e-12
        q=statistics.mean(v[2] for v in values)
        errors.append(values[0][0]-q)
        cell_nll.append(statistics.mean(v[3] for v in values)); cell_brier.append(statistics.mean(v[4] for v in values))
    return {'records':len(rows),'conditions':len(cells),'condition_macro_NLL':statistics.mean(cell_nll),
        'condition_macro_Brier':statistics.mean(cell_brier),'participant_NLL':statistics.mean(losses),
        'participant_accuracy':statistics.mean(correct),'correct_records':sum(correct),
        'mean_absolute_condition_probability_error':statistics.mean(abs(e) for e in errors),
        'condition_probability_RMSE':math.sqrt(statistics.mean(e*e for e in errors)),
        'p_left_0_5_tied_records':sum(tied),'p_left_0_5_tie_fraction':statistics.mean(tied),
        'fair_coin_tie_expected_accuracy':statistics.mean(fair_correct)}


def analyze(cfg,manifest):
    all_rows=read_csv(ROOT/cfg['prediction_source'])
    keys=set(manifest['selected_case_keys']); rows=[r for r in all_rows if r['case_key'] in keys]
    assert len(rows)==246 and {r['case_key'] for r in rows}==keys
    assert len({r['case_key'] for r in all_rows})==489
    prior_manifest=json.loads((ROOT/'outputs/bristol_context_choice_benchmark_20261013/frozen_LLM_manifest.json').read_text())
    newer_manifest=json.loads((ROOT/'outputs/bristol_no_prior_choice_benchmark_20261013/frozen_manifest.json').read_text())
    old_responses={r['key']:r for r in [json.loads(line) for line in (ROOT/'outputs/bristol_context_choice_benchmark_20261013/responses.jsonl').read_text().splitlines()]}
    new_responses={r['key']:r for r in [json.loads(line) for line in (ROOT/'outputs/bristol_no_prior_choice_benchmark_20261013/responses.jsonl').read_text().splitlines()]}
    assert len(old_responses)==len(new_responses)==34
    new_tasks={t['key']:t for t in newer_manifest['tasks']}
    for row in rows:
        key='_'.join(map(str,cond(row)))
        for response,field in [(old_responses[key],'qwen3_4b'),(new_responses[key],'qwen3_4b_no_prior')]:
            assert response['valid'] and response['model_digest']==prior_manifest['model_digest']
            raw=json.loads(response['response']['message']['content'])
            assert abs(raw['p_left']-float(row[field]))<=1e-12
        task=new_tasks[key]
        obs=json.loads(task['payload']['messages'][1]['content'])
        assert obs['nearer_to_final_exit'] in ['equal','unknown'] and 'training_base_probability_left' not in obs
    table('selected_predictions.csv',rows)
    aggregate={name:metrics(rows,name) for name in cfg['methods']}
    condition_rows=[]; family_rows=[]; mirror_rows=[]
    for c in sorted({cond(r) for r in rows}):
        rs=[r for r in rows if cond(r)==c]
        for method in cfg['methods']:
            condition_rows.append({'condition':json.dumps(c),'family':json.dumps(fam(rs[0])),'method':method,
                'p_left':float(rs[0][method]),'observed_left_fraction':statistics.mean(int(r['observed_exit'])==1 for r in rs),**metrics(rs,method)})
    families=sorted({fam(r) for r in rows})
    for f in families:
        rs=[r for r in rows if fam(r)==f]; a,b,k=f
        for method in cfg['methods']:
            family_rows.append({'family':json.dumps(f),'method':method,**metrics(rs,method)})
            pa=float(next(r for r in rs if cond(r)==(a,b,k))[method]); pb=float(next(r for r in rs if cond(r)==(b,a,k))[method])
            mirror_rows.append({'family':json.dumps(f),'method':method,'p_left_original':pa,'p_left_mirror':pb,
                'self_mirror':a==b,'absolute_mirror_error':abs(pa+pb-1)})
    table('per_condition_metrics.csv',condition_rows); table('per_family_metrics.csv',family_rows); table('mirror_errors.csv',mirror_rows)
    new=aggregate['qwen3_4b_no_prior']; comparisons={}
    for method in cfg['methods'][:-1]:
        base=aggregate[method]
        deltas=[metrics([r for r in rows if fam(r)==f],'qwen3_4b_no_prior')['condition_macro_NLL']-
                metrics([r for r in rows if fam(r)==f],method)['condition_macro_NLL'] for f in families]
        comparisons[method]={'NLL_delta_new_minus_reference':new['condition_macro_NLL']-base['condition_macro_NLL'],
            'NLL_relative_change':new['condition_macro_NLL']/base['condition_macro_NLL']-1,
            'Brier_delta_new_minus_reference':new['condition_macro_Brier']-base['condition_macro_Brier'],
            'accuracy_delta_percentage_points':100*(new['participant_accuracy']-base['participant_accuracy']),
            'families_lower_NLL':sum(d< -1e-12 for d in deltas),'families_higher_NLL':sum(d>1e-12 for d in deltas),
            'families_tied_NLL':sum(abs(d)<=1e-12 for d in deltas)}
    groups={'no_map':[r for r in rows if int(r['knowledge_code'])==0],
        'equal_map_nonbalanced':[r for r in rows if int(r['knowledge_code'])==3],
        'busy':[r for r in rows if r['crowd_state_design']=='busy'],
        'free':[r for r in rows if r['crowd_state_design']=='free']}
    subgroups={label:{method:metrics(rs,method) for method in cfg['methods']} for label,rs in groups.items()}
    gate_by_key={r['case_key']:r['source_gate'] for r in read_csv(ROOT/cfg['selection_source'])}
    fold_counts=[]
    for fold in prior_manifest['folds']:
        if tuple(fold['family']) not in families: continue
        train={tuple(c) for c in fold['train_conditions']}; held={tuple(c) for c in fold['held_conditions']}
        assert not train&held and held=={cond(r) for r in rows if fam(r)==tuple(fold['family'])}
        counts=Counter(gate_by_key[r['case_key']] for r in all_rows if cond(r) in train)
        fold_counts.append({'family':fold['family'],'train_records':sum(counts.values()),'train_source_gate_counts':dict(counts)})
    assert len(fold_counts)==9
    mirror_summary={method:{'families':9,'mean_absolute_error':statistics.mean(r['absolute_mirror_error'] for r in mirror_rows if r['method']==method),
        'max_absolute_error':max(r['absolute_mirror_error'] for r in mirror_rows if r['method']==method)} for method in cfg['methods']}
    base=aggregate['training_frequency']
    signal=new['condition_macro_NLL']<base['condition_macro_NLL'] and new['condition_macro_Brier']<base['condition_macro_Brier']
    summary={'status':'completed_cached_source_sensitivity_review','records':246,'conditions':17,'mirror_families':9,
        'distinct_cached_conditions_used_per_LLM':17,'methods':aggregate,'comparisons':comparisons,'subgroups':subgroups,
        'mirror_summary':mirror_summary,'unchanged_training_fold_source_counts':fold_counts,
        'no_prior_better_than_frequency_on_both_probability_metrics':signal,
        'interpretation':'Exploratory improvement only; no external confirmation' if signal else 'Fixed scene-only predictor does not improve probability forecasting even on the relatively clear source subset',
        'frozen_review_manifest_sha256':sha(OUT/'frozen_review_manifest.json'),
        'selected_predictions_sha256':sha(OUT/'selected_predictions.csv'),
        'limitations':[cfg[k] for k in ['development_visibility','training_caveat','empirical_probability_errors']],
        'new_LLM_requests':0,'new_model_fits':0,'PPO_updates':0,'map_code_changes':0,
        'old_final_scenarios_accessed':False,'Liao_confirmation_scores_computed':False}
    write_json(OUT/'summary.json',summary)
    assert all(sha(ROOT/p)==value for p,value in manifest['protected_sources'].items())
    print(json.dumps({k:summary[k] for k in ['status','records','conditions','mirror_families','methods','comparisons','interpretation']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','analyze']); mode=parser.parse_args().mode
    cfg,manifest=prepare()
    if mode=='analyze': analyze(cfg,manifest)
