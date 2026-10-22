"""Independent cell-level verification of the cache-only source review."""
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/clear_context_cache_review_20261014'
CONFIG=ROOT/'schemas/clear_context_cache_review_20261014.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def cond(row):return tuple(int(row[k]) for k in ['assigned_left_crowd','assigned_right_crowd','knowledge_code'])


def fam(row):
    a,b,k=cond(row);return min(a,b),max(a,b),k


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)


def independent_metrics(rows,method):
    cells=defaultdict(list)
    for row in rows:cells[cond(row)].append(row)
    losses=[];briers=[];errors=[];total_loss=0;correct=ties=0;fair=0
    for cell in cells.values():
        ps=[float(r[method]) for r in cell]
        assert max(ps)-min(ps)<=1e-12
        raw=ps[0];p=min(1-1e-9,max(1e-9,raw));n=len(cell)
        left=sum(int(r['observed_exit'])==1 for r in cell);q=left/n
        loss=-q*math.log(p)-(1-q)*math.log1p(-p)
        brier=q*(1-p)**2+(1-q)*p**2
        losses.append(loss);briers.append(brier);errors.append(raw-q);total_loss+=loss*n
        correct+=left if raw>=.5 else n-left
        ties+=n if raw==.5 else 0
        fair+=n/2 if raw==.5 else left if raw>.5 else n-left
    return {'records':len(rows),'conditions':len(cells),'condition_macro_NLL':sum(losses)/len(cells),
        'condition_macro_Brier':sum(briers)/len(cells),'participant_NLL':total_loss/len(rows),
        'participant_accuracy':correct/len(rows),'correct_records':correct,
        'mean_absolute_condition_probability_error':sum(abs(e) for e in errors)/len(cells),
        'condition_probability_RMSE':math.sqrt(sum(e*e for e in errors)/len(cells)),
        'p_left_0_5_tied_records':ties,'p_left_0_5_tie_fraction':ties/len(rows),
        'fair_coin_tie_expected_accuracy':fair/len(rows)}


def match_metrics(record,expected):
    for key,value in expected.items():close(record[key],value)


def main():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'frozen_review_manifest.json').read_text(encoding='utf-8'))
    assert manifest['config_sha256']==sha(CONFIG)
    assert manifest['runner_sha256']==sha(ROOT/'scripts/review_clear_context_cache.py')
    assert all(sha(ROOT/path)==value for path,value in manifest['protected_sources'].items())
    gates=csv_rows(ROOT/cfg['selection_source'])
    selected_keys={r['case_key'] for r in gates if r['source_gate']=='comparatively_clear_under_README'}
    rule_keys={r['case_key'] for r in gates if int(r['knowledge_code'])==0 or (int(r['knowledge_code'])==3 and int(r['assigned_left'])!=int(r['assigned_right']))}
    assert selected_keys==rule_keys==set(manifest['selected_case_keys']) and len(selected_keys)==246
    all_rows=csv_rows(ROOT/cfg['prediction_source']);rows=csv_rows(OUT/'selected_predictions.csv')
    original={r['case_key']:r for r in all_rows}
    assert len(rows)==246 and len({r['case_key'] for r in rows})==246
    assert all(r==original[r['case_key']] for r in rows)
    assert {r['case_key'] for r in rows}==selected_keys
    assert sorted({cond(r) for r in rows})==[tuple(c) for c in manifest['selected_conditions']]
    assert sorted({fam(r) for r in rows})==[tuple(f) for f in manifest['selected_families']]
    families=sorted({fam(r) for r in rows})
    assert len({cond(r) for r in rows})==17 and len(families)==9
    # Labels/predictions can change without changing the independent source rule.
    for gate in gates:
        altered={**gate,'observed_exit':'999','qwen3_4b_no_prior':'999','context_logit':'999'}
        assert (int(altered['knowledge_code'])==0 or (int(altered['knowledge_code'])==3 and int(altered['assigned_left'])!=int(altered['assigned_right'])))==(gate['case_key'] in rule_keys)
    for folder,field,manifest_name in [('bristol_context_choice_benchmark_20261013','qwen3_4b','frozen_LLM_manifest.json'),
                                     ('bristol_no_prior_choice_benchmark_20261013','qwen3_4b_no_prior','frozen_manifest.json')]:
        parent=ROOT/'outputs'/folder
        cached=[json.loads(line) for line in (parent/'responses.jsonl').read_text().splitlines()]
        assert len(cached)==len({r['key'] for r in cached})==34
        cache={r['key']:r for r in cached};plan=json.loads((parent/manifest_name).read_text())
        tasks={t['key']:t for t in plan['tasks']}
        for row in rows:
            key='_'.join(map(str,cond(row)));response=cache[key];task=tasks[key]
            assert response['valid'] and response['model_digest']==plan['model_digest']
            assert response['prompt_sha256']==task['prompt_sha256']
            content=json.loads(response['response']['message']['content'])
            assert set(content)=={'p_left'};close(float(row[field]),content['p_left'])
            obs=json.loads(task['payload']['messages'][1]['content'])
            assert obs['nearer_to_final_exit']==('unknown' if int(row['knowledge_code'])==0 else 'equal')
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['frozen_review_manifest_sha256']==sha(OUT/'frozen_review_manifest.json')
    assert summary['selected_predictions_sha256']==sha(OUT/'selected_predictions.csv')
    for method in cfg['methods']:match_metrics(summary['methods'][method],independent_metrics(rows,method))
    for filename,key in [('per_condition_metrics.csv','condition'),('per_family_metrics.csv','family')]:
        detail=csv_rows(OUT/filename)
        assert len(detail)==(17 if key=='condition' else 9)*len(cfg['methods'])
        for record in detail:
            group=tuple(json.loads(record[key]));subset=[r for r in rows if (cond(r) if key=='condition' else fam(r))==group]
            match_metrics(record,independent_metrics(subset,record['method']))
    new=summary['methods']['qwen3_4b_no_prior']
    for method,comparison in summary['comparisons'].items():
        ref=summary['methods'][method]
        close(comparison['NLL_delta_new_minus_reference'],new['condition_macro_NLL']-ref['condition_macro_NLL'])
        close(comparison['NLL_relative_change'],new['condition_macro_NLL']/ref['condition_macro_NLL']-1)
        close(comparison['Brier_delta_new_minus_reference'],new['condition_macro_Brier']-ref['condition_macro_Brier'])
        close(comparison['accuracy_delta_percentage_points'],100*(new['participant_accuracy']-ref['participant_accuracy']))
        ds=[independent_metrics([r for r in rows if fam(r)==f],'qwen3_4b_no_prior')['condition_macro_NLL']-
            independent_metrics([r for r in rows if fam(r)==f],method)['condition_macro_NLL'] for f in families]
        assert comparison['families_lower_NLL']==sum(d< -1e-12 for d in ds)
        assert comparison['families_higher_NLL']==sum(d>1e-12 for d in ds)
        assert comparison['families_tied_NLL']==sum(abs(d)<=1e-12 for d in ds)
    subsets={'no_map':[r for r in rows if int(r['knowledge_code'])==0],
        'equal_map_nonbalanced':[r for r in rows if int(r['knowledge_code'])==3],
        'busy':[r for r in rows if r['crowd_state_design']=='busy'],'free':[r for r in rows if r['crowd_state_design']=='free']}
    for label,subset in subsets.items():
        for method in cfg['methods']:match_metrics(summary['subgroups'][label][method],independent_metrics(subset,method))
    mirrors=csv_rows(OUT/'mirror_errors.csv');assert len(mirrors)==45
    for record in mirrors:
        a,b,k=json.loads(record['family']);method=record['method']
        pa=float(next(r for r in rows if cond(r)==(a,b,k))[method]);pb=float(next(r for r in rows if cond(r)==(b,a,k))[method])
        close(record['absolute_mirror_error'],abs(pa+pb-1))
    for method,info in summary['mirror_summary'].items():
        values=[float(r['absolute_mirror_error']) for r in mirrors if r['method']==method]
        close(info['mean_absolute_error'],sum(values)/9);close(info['max_absolute_error'],max(values))
    gate_by_key={r['case_key']:r['source_gate'] for r in gates}
    old_plan=json.loads((ROOT/'outputs/bristol_context_choice_benchmark_20261013/frozen_LLM_manifest.json').read_text())
    folds={tuple(f['family']):f for f in old_plan['folds']}
    assert len(summary['unchanged_training_fold_source_counts'])==9
    for item in summary['unchanged_training_fold_source_counts']:
        fold=folds[tuple(item['family'])];train={tuple(c) for c in fold['train_conditions']}
        held={tuple(c) for c in fold['held_conditions']};assert not train&held
        counts=Counter(gate_by_key[r['case_key']] for r in all_rows if cond(r) in train)
        assert dict(counts)==item['train_source_gate_counts'] and sum(counts.values())==item['train_records']
        assert counts['map_direction_semantics_unresolved']==215
        assert counts['equal_map_balanced_count_discrepancy']==28
    signal=new['condition_macro_NLL']<summary['methods']['training_frequency']['condition_macro_NLL'] and new['condition_macro_Brier']<summary['methods']['training_frequency']['condition_macro_Brier']
    assert signal==summary['no_prior_better_than_frequency_on_both_probability_metrics']
    assert all(summary[k]==0 for k in ['new_LLM_requests','new_model_fits','PPO_updates','map_code_changes'])
    result={'status':'completed_cache_review_verified','source_selected_246_records_17_conditions_9_families_verified':True,
        'source_selection_independent_of_outcomes_and_scores':True,'selected_prediction_rows_unchanged':True,
        'both_LLM_columns_match_original_raw_responses':True,'all_metrics_independently_recomputed':True,
        'training_lineage_caveat_verified':True,'all_previous_files_unchanged':True,
        'summary_sha256':sha(OUT/'summary.json'),'manifest_sha256':sha(OUT/'frozen_review_manifest.json'),
        'verifier_sha256':sha(Path(__file__)),'new_LLM_requests':0,'new_model_fits':0,'PPO_updates':0,
        'old_final_scenarios_accessed':False,'Liao_confirmation_scores_computed':False}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
