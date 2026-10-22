"""Freeze then run one bounded grouped development comparison on cached rows."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fire_evac import crowd_response_comparison as m
from audit_crowd_response_cache import OUT, read, save, verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(rows):
    counts = Counter(r['family'] for r in rows)
    return np.asarray([1./counts[r['family']] for r in rows], dtype=float)


def mean(rows, values):
    return np.average(np.asarray(values), weights=weights(rows), axis=0).tolist()


def prepare():
    verify()
    rows = read(OUT/'primary_rows.json')
    audit = read(OUT/'audit_summary.json')
    assert len(rows)==72 and audit['families']==20
    scenes = sorted({r['scenario'] for r in rows})
    folds = []
    for scene in scenes:
        train = [r['state_id'] for r in rows if r['scenario']!=scene and r['population']%100!=3]
        calibration = [r['state_id'] for r in rows if r['scenario']!=scene and r['population']%100==3]
        test = [r['state_id'] for r in rows if r['scenario']==scene]
        assert train and calibration and test
        folds.append(dict(held_out_scene=scene,train=train,calibration=calibration,test=test))
    files = ['src/fire_evac/crowd_response_gain.py','src/fire_evac/crowd_response_comparison.py',
             'scripts/audit_crowd_response_cache.py','scripts/compare_crowd_response_representations.py',
             'tests/test_crowd_response_gain.py','tests/test_crowd_response_comparison.py',
             'outputs/research_convergence_20261011/audit_summary.json',
             'outputs/research_convergence_20261011/primary_rows.json']
    protocol = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Previously inspected development cache; descriptive internal leave-one-scenario-out diagnostics, not a fresh final or independent confirmation.',
        representations=list(m.VARIANTS),model='Weighted standardized two-target ridge; alpha=1, no hyperparameter search.',
        history='All completed steps of the early state, at most3; raw sensors identical in public_history and response; response appends deterministic17 summaries.',
        targets=['candidate_minus_base_incapacitation_rate','candidate_minus_base_mean_cumulative_FED'],
        target='Force one matched-release direction action, then common no-advice PPO to terminal; pair futures within state then average.',
        units='One unique decision state after path/mode deduplication; each scene/population family total weight1; behavior, checkpoint, time and future replicas stay grouped.',
        folds=folds,calibration='Other scenes repetition3 only; four families/fold. Quantile.9 of family-worst adverse residual, nonnegative. No coverage guarantee.',
        gate='Upper adverse prediction must not increase incapacitation (>1e-12); improve incapacitation or FED by at least.001.',
        baselines=['fit_only_family_weighted_constant','no_advice_all_reject','first_eligible_only'],
        reporting=['weighted MAE and signed bias per target','state and family upper residual coverage',
                   'harmful/beneficial/neutral accepts and rejects','any-incapacitation-increase accepts',
                   'by scene/behavior/checkpoint/first-repeat','per-future reversals and isolated-action shadow outcome deltas'],
        continue_criteria=dict(all_required=True,
            minimum_FED_MAE_improvement_vs_history_and_constant=.001,
            positive_FED_MAE_improvement_scene_count=3,
            no_incapacitation_MAE_worse_than_history_and_constant=True,
            no_harmful_accepts=True,no_any_incapacitation_increase_accepts=True,
            beneficial_accepts_span_at_least_scenes=2,beneficial_accepts_span_at_least_checkpoints=2,
            justification='.001 is the existing material-outcome tolerance, not a newly optimized threshold. Requiring multiple scenes/weights avoids repeating known single-subgroup success.'),
        failure='Stop expansion if criteria fail; no threshold/feature/model tuning on these held-out results. No conditional fresh-population confirmation unless all criteria pass.',
        bootstrap='2000 paired family resamples within fixed scenes, seed2026101131, conditional on fixed outer predictions; report descriptive interval, not five-building or refit uncertainty.',
        budget=dict(folds=5,ridge_fits=15,constant_fits=5,bootstraps=2000,
                    wall_seconds_cap=300,new_simulations=0,new_ppo_training=0,new_llm_requests=0,reserved_final_opened=0),
        sources={p:sha(ROOT/p) for p in files},tests_passed=9)
    save(OUT/'frozen_comparison_protocol.json',protocol)
    print(json.dumps({'prepared':True,'folds':len(folds),'states':len(rows),'ridge_fits':15}))


def constant_fit(rows):
    return dict(intercept=mean(rows,[[r['incap_delta'],r['fed_delta']] for r in rows]))


def constant_calibration(fitted, rows):
    residuals = defaultdict(lambda:[-np.inf,-np.inf])
    for r in rows:
        residual = np.asarray([r['incap_delta'],r['fed_delta']])-fitted['intercept']
        residuals[r['family']] = np.maximum(residuals[r['family']],residual).tolist()
    return dict(upper_adverse_residual=np.maximum(np.quantile(list(residuals.values()),.9,axis=0,method='higher'),0).tolist(),
                calibration_families=len(residuals))


def metrics(rows, records):
    index={r['state_id']:r for r in rows}
    data=[index[p['state_id']] for p in records]
    y=np.asarray([[r['incap_delta'],r['fed_delta']] for r in data])
    pred=np.asarray([r['prediction'] for r in records]);margin=np.asarray([r['margin'] for r in records])
    within=y<=pred+margin+1e-12
    family_bands=defaultdict(list)
    for r,w in zip(data,within):family_bands[r['family']].append(w)
    return dict(states=len(data),families=len(family_bands),
                MAE=mean(data,np.abs(pred-y)),prediction_minus_target_bias=mean(data,pred-y),
                upper_residual_state_coverage=mean(data,within.astype(float)),
                upper_residual_family_coverage=np.mean([np.all(v,axis=0) for v in family_bands.values()],axis=0).tolist())


def gate_counts(rows, accepted):
    counts=Counter(('accept_' if yes else 'reject_')+r['label'] for r,yes in zip(rows,accepted))
    future_harm=[]
    for r,yes in zip(rows,accepted):
        if yes:
            future_harm.append(any(p['incap_delta']>=.001 or (p['incap_delta']>-.001 and p['fed_delta']>=.001)
                                   for p in r['paired_futures']))
    return dict(states=len(rows),accepted=sum(accepted),confusion={
        a+c:counts[a+c] for a in ('accept_','reject_') for c in ('beneficial','harmful','small_or_neutral')},
        accepts_with_any_mean_incapacitation_increase=sum(yes and r['incap_delta']>1e-12 for r,yes in zip(rows,accepted)),
        accepted_states_with_at_least_one_harmful_paired_future=sum(future_harm),
        family_weighted_isolated_action_delta=mean(rows,[[r['incap_delta']*yes,r['fed_delta']*yes] for r,yes in zip(rows,accepted)]),
        delta_interpretation='Shadow single-action paired deltas on common cached paths; not a rollout of the combined gate policy.')


def run():
    import time
    start=time.monotonic()
    verify();protocol=read(OUT/'frozen_comparison_protocol.json')
    for p,h in protocol['sources'].items():assert sha(ROOT/p)==h,p
    rows=read(OUT/'primary_rows.json');index={r['state_id']:r for r in rows}
    predictions=[];models=[]
    for fold in protocol['folds']:
        train=[index[s] for s in fold['train']];cal=[index[s] for s in fold['calibration']];test=[index[s] for s in fold['test']]
        groups=[{r['family'] for r in rs} for rs in (train,cal,test)]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for variant in (*m.VARIANTS,'constant'):
            assert time.monotonic()-start<300
            model=constant_fit(train) if variant=='constant' else m.fit_ridge(train,variant)
            band=constant_calibration(model,cal) if variant=='constant' else m.calibrate(model,cal)
            models.append(dict(scene=fold['held_out_scene'],variant=variant,model=model,calibration=band,
                               training_states=len(train),training_families=len(groups[0]),calibration_states=len(cal),
                               calibration_families=len(groups[1]),test_states=len(test),test_families=len(groups[2])))
            for r in test:
                p=model['intercept'] if variant=='constant' else m.predict(model,r['features'][variant])
                predictions.append(dict(state_id=r['state_id'],scene=r['scenario'],family=r['family'],
                    variant=variant,prediction=p,margin=band['upper_adverse_residual'],
                    accepted=m.accepts(p,band['upper_adverse_residual']),raw_accepted=m.accepts(p)))
    methods={v:[p for p in predictions if p['variant']==v] for v in (*m.VARIANTS,'constant')}
    result={}
    for v,ps in methods.items():
        ordered=[index[p['state_id']] for p in ps]
        accepted=[p['accepted'] for p in ps]
        result[v]=dict(prediction_metrics=metrics(rows,ps),gate=gate_counts(ordered,accepted),
                       raw_gate_diagnostic=gate_counts(ordered,[p['raw_accepted'] for p in ps]),
                       scene_metrics={s:metrics(rows,[p for p in ps if p['scene']==s]) for s in sorted({r['scenario'] for r in rows})},
                       scene_gate={s:gate_counts([r for r in ordered if r['scenario']==s],
                           [a for r,a in zip(ordered,accepted) if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},
                       subgroup_gate={})
        for key in ('behavior','checkpoint','offers'):
            for value in sorted({r[key] for r in ordered}):
                mask=[r[key]==value for r in ordered]
                result[v]['subgroup_gate'][f'{key}={value}']=gate_counts([r for r,k in zip(ordered,mask) if k],
                                                                       [a for a,k in zip(accepted,mask) if k])
    for name,accepted in [('no_advice_all_reject',[False]*len(rows)),('first_eligible_only',[r['offers']==0 for r in rows])]:
        result[name]=dict(gate=gate_counts(rows,accepted))
    response=result['response'];history=result['public_history'];constant=result['constant']
    improvements={v:np.asarray(result[v]['prediction_metrics']['MAE'])-response['prediction_metrics']['MAE']
                  for v in ('public_history','constant','current')}
    passed_scenes={v:[s for s in response['scene_metrics']
                     if result[v]['scene_metrics'][s]['MAE'][1]-response['scene_metrics'][s]['MAE'][1]>1e-12]
                  for v in ('public_history','constant')}
    accepted_rows=[index[p['state_id']] for p in methods['response'] if p['accepted'] and index[p['state_id']]['label']=='beneficial']
    conditions=dict(
        material_FED_MAE_improvement=all(improvements[v][1]>=.001 for v in ('public_history','constant')),
        at_least_three_improved_scenes=all(len(passed_scenes[v])>=3 for v in passed_scenes),
        no_worse_incapacitation_MAE=all(improvements[v][0]>=-1e-12 for v in ('public_history','constant')),
        no_harmful_accepts=response['gate']['confusion']['accept_harmful']==0,
        no_incapacitation_increase_accepts=response['gate']['accepts_with_any_mean_incapacitation_increase']==0,
        beneficial_accepts_in_multiple_scenes=len({r['scenario'] for r in accepted_rows})>=2,
        beneficial_accepts_in_multiple_checkpoints=len({r['checkpoint'] for r in accepted_rows})>=2)
    # Conditional intervals over the already fixed held-out predictions only.
    mae_by_family={}
    for v,ps in methods.items():
        cells=defaultdict(list)
        for p in ps:
            r=index[p['state_id']]
            cells[r['family']].append(np.abs(np.asarray(p['prediction'])-[r['incap_delta'],r['fed_delta']]))
        mae_by_family[v]={f:np.mean(vals,axis=0) for f,vals in cells.items()}
    families=defaultdict(list)
    for r in rows:
        if r['family'] not in families[r['scenario']]:families[r['scenario']].append(r['family'])
    rng=np.random.default_rng(2026101131);boots=defaultdict(list)
    for _ in range(2000):
        draw=[f for fs in families.values() for f in rng.choice(fs,size=len(fs),replace=True)]
        for v in ('public_history','constant','current'):
            boots[v].append(np.mean([mae_by_family[v][f]-mae_by_family['response'][f] for f in draw],axis=0))
    intervals={v:np.quantile(vals,[.025,.975],axis=0).tolist() for v,vals in boots.items()}
    decision='Proceed to a separately frozen fresh-population confirmation.' if all(conditions.values()) else 'Stop expansion of this response-gain branch; write the limited/negative result. No new population confirmation in this cycle.'
    summary=dict(protocol_sha256=sha(OUT/'frozen_comparison_protocol.json'),states=len(rows),families=len({r['family'] for r in rows}),
                 scene_count=5,methods=result,response_MAE_improvement={v:a.tolist() for v,a in improvements.items()},
                 response_MAE_improvement_conditional_bootstrap_95=intervals,improved_scenes=passed_scenes,
                 continue_conditions=conditions,passed=all(conditions.values()),decision=decision,
                 uncertainty='Only five same-graph scenes and four calibration families/fold; bootstrap conditional on fixed predictions, no independence of shared fit folds or buildings implied.',
                 isolated_action_warning='Gate statistics use paired single-action outcomes on archived common paths. They are not complete closed-loop gate-policy effects.',
                 elapsed_seconds=time.monotonic()-start,new_simulations=0,new_training=0,new_llm_requests=0,reserved_final_opened=0,
                 preserved_files=verify())
    save(OUT/'comparison_summary.json',summary);save(OUT/'outer_fold_models.json',models);save(OUT/'held_out_predictions.json',predictions)
    with (OUT/'held_out_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['state_id','scene','family','variant','predicted_incap_delta','predicted_fed_delta','incap_margin','fed_margin','accepted','raw_accepted'])
        writer.writeheader()
        for p in predictions:
            writer.writerow({**{k:p[k] for k in ('state_id','scene','family','variant','accepted','raw_accepted')},
                'predicted_incap_delta':p['prediction'][0],'predicted_fed_delta':p['prediction'][1],
                'incap_margin':p['margin'][0],'fed_margin':p['margin'][1]})
    print(json.dumps({k:summary[k] for k in ('states','families','response_MAE_improvement','continue_conditions','passed','decision','elapsed_seconds')},ensure_ascii=False))
    for v in result:
        print(json.dumps({'method':v,**result[v]['gate'],**result[v].get('prediction_metrics',{})},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
