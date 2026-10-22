"""One fixed paired history ablation plus event-aligned descriptive diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from audit_liao_dynamic_data import ROOT, SOURCE, DEFAULT_OUT, parse, sha
from run_liao_motion_pilot import (hierarchy_weights, smooth_frequency, fit_logit, metrics)
from analyze_liao_motion_pilot import ranking

CONFIG=ROOT/'schemas/liao_history_ablation_20261013.json'
OUTPUT=ROOT/'outputs/liao_history_ablation_20261013'


def summarize_past(lookup,cutoff,history):
    if history%2:raise ValueError('History must have even frame count')
    positions=np.array([[lookup[f]['x'],lookup[f]['y']] for f in range(cutoff-history,cutoff+1)])
    duration=history/16;middle=history//2
    early_v=(positions[middle]-positions[0])/(duration/2)
    recent_v=(positions[-1]-positions[middle])/(duration/2)
    early,recent=np.linalg.norm(early_v),np.linalg.norm(recent_v)
    known=early>=.1 and recent>=.1
    turn=np.arccos(np.clip(early_v@recent_v/(early*recent),-1,1)) if known else 0.0
    path=np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()
    net=np.linalg.norm(positions[-1]-positions[0])
    return np.array([path/duration,recent,early,recent-early,turn,max(0,path-net)])


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    config=json.loads(CONFIG.read_text(encoding='utf-8'));OUTPUT.mkdir(exist_ok=True)
    tracks=parse(SOURCE)
    lookup={key:{r['frame']:r for r in rows} for key,rows in tracks.items() if key[0] in config['development_runs']}
    interface=DEFAULT_OUT/'draft_prediction_interface'
    old=[r for r in csv.DictReader((interface/'outcomes_and_provenance.csv').open(encoding='utf-8-sig'))
         if r['partition']=='proposed_development']
    records=[];removed=[]
    for row in old:
        key=(row['run'],int(row['local_id']));frame=int(row['current_frame'])
        if all(f in lookup[key] for f in range(frame-48,frame+1)):records.append(row)
        else:removed.append({**row,'reason':'missing_3s_history'})
    x={h:np.array([summarize_past(lookup[(r['run'],int(r['local_id']))],int(r['current_frame']),h)
                   for r in records]) for h in config['history_frames']}
    y=np.array([int(r['switch_within_horizon']) for r in records])
    names=config['models'];scores={name:np.full(len(records),np.nan) for name in names}
    fold_models=[];per_run=[]
    for held in config['development_runs']:
        test=np.array([r['run']==held for r in records]);train=~test
        if not test.any():raise ValueError(f'No common-cohort anchors in {held}')
        training=[r for r,a in zip(records,train) if a];testing=[r for r,a in zip(records,test) if a]
        w=hierarchy_weights(training)
        npeople=len({(r['run'],r['local_id']) for r in training})
        p0=smooth_frequency(y[train],w,npeople,config['smoothing_probability'],config['smoothing_prior_person_runs'])
        scores[names[0]][test]=p0
        for h,name in zip(config['history_frames'],names[1:]):
            center,scale,beta,opt=fit_logit(x[h][train],y[train],w,p0,config['logit_l2'],config['logit_maxiter'])
            scores[name][test]=expit(np.column_stack([np.ones(test.sum()),(x[h][test]-center)/scale])@beta)
            fold_models.append(dict(held_out_run=held,method=name,train_windows=int(train.sum()),
                                    training_window_keys=[r['window_key'] for r in training],
                                    center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),optimizer=opt))
        for name in names:per_run.append(dict(run=held,method=name,**metrics(y[test],scores[name][test],testing,config['probability_threshold'])))
    assert all(np.isfinite(p).all() for p in scores.values())
    predictions=[{**r,**{name:float(scores[name][i]) for name in names}} for i,r in enumerate(records)]
    csv_write(OUTPUT/'common_cohort_predictions.csv',predictions)
    csv_write(OUTPUT/'excluded_early_anchors.csv',removed)
    csv_write(OUTPUT/'per_run_metrics.csv',per_run)
    csv_write(OUTPUT/'past_only_features.csv',[dict(window_key=r['window_key'],history_frames=h,
        mean_step_speed_m_s=float(x[h][i,0]),recent_half_speed_m_s=float(x[h][i,1]),early_half_speed_m_s=float(x[h][i,2]),
        speed_change_m_s=float(x[h][i,3]),absolute_turn_radians=float(x[h][i,4]),path_inefficiency_m=float(x[h][i,5]))
        for i,r in enumerate(records) for h in config['history_frames']])
    # Descriptive label-aligned features, never injected into model inputs.
    events=[r for r in csv.DictReader((DEFAULT_OUT/'label_events.csv').open(encoding='utf-8-sig'))
            if r['run'] in config['development_runs'] and r['event_kind']=='nonzero_switch']
    aligned=[];missing=[]
    for event in events:
        key=(event['run'],int(event['local_id']));ev_frame=int(event['frame'])
        for offset in config['lead_lag_offsets_frames']:
            cutoff=ev_frame+offset
            trace=dict(run=key[0],local_id=key[1],event_frame=ev_frame,offset_s=offset/16,feature_cutoff_frame=cutoff)
            if not all(f in lookup[key] for f in range(cutoff-16,cutoff+1)):
                missing.append({**trace,'reason':'past_window_outside_observed_track'});continue
            feature=summarize_past(lookup[key],cutoff,16)
            aligned.append({**trace,'recent_speed_m_s':float(feature[1]),'speed_change_m_s':float(feature[3]),
                            'turn_degrees':float(np.degrees(feature[4])),
                            'slow_or_turn_rule':bool(feature[1]<.25 or (feature[1]>=.1 and feature[2]>=.1 and feature[4]>np.pi/6))})
    csv_write(OUTPUT/'event_aligned_motion.csv',aligned);csv_write(OUTPUT/'event_aligned_missing.csv',missing)
    aligned_summary=[]
    for offset in config['lead_lag_offsets_frames']:
        a=[r for r in aligned if r['offset_s']==offset/16]
        aligned_summary.append(dict(offset_s=offset/16,observed_events=len(a),total_events=len(events),
            median_speed_m_s=float(np.median([r['recent_speed_m_s'] for r in a])),
            median_turn_degrees=float(np.median([r['turn_degrees'] for r in a])),
            slow_or_turn_count=sum(r['slow_or_turn_rule'] for r in a)))
    old_events={(r['run'],r['local_id'],r['first_different_frame']) for r in old if int(r['switch_within_horizon'])}
    retained_events={(r['run'],r['local_id'],r['first_different_frame']) for r in records if int(r['switch_within_horizon'])}
    result=dict(status='paired_development_history_ablation_completed',old_windows=len(old),
        common_windows=len(records),removed_windows=len(removed),old_positive_windows=sum(int(r['switch_within_horizon']) for r in old),
        common_positive_windows=int(y.sum()),old_distinct_events=len(old_events),common_distinct_events=len(retained_events),
        events_lost_to_history_requirement=sorted(old_events-retained_events),
        methods={name:metrics(y,scores[name],records,config['probability_threshold']) for name in names},
        ranking=[ranking(predictions,name,q) for name in names for q in config['ranking_budgets']],
        event_aligned_summary=aligned_summary,
        config_sha256=sha(CONFIG),code_sha256=sha(Path(__file__)),
        all_20_logit_fits_converged=all(r['optimizer']['success'] for r in fold_models),
        same_windows_folds_and_fitting_budget=True,confirmation_performance_computed=False)
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'fold_models.json').write_text(json.dumps(fold_models,indent=2)+'\n',encoding='utf-8')
    verify(result,lookup,records,fold_models)
    plot(aligned_summary)
    return result


def verify(result,lookup,records,models):
    synthetic={f:dict(x=f/16,y=0) for f in range(-48,1)}
    for h in [16,48]:np.testing.assert_allclose(summarize_past(synthetic,0,h),[1,1,1,0,0,0],atol=1e-10)
    r=records[0];key=(r['run'],int(r['local_id']));cutoff=int(r['current_frame'])
    perturbed={f:{**row,'x':row['x']+1000 if f>cutoff else row['x'],'y':row['y']-1000 if f>cutoff else row['y'],
                  'label':0} for f,row in lookup[key].items()}
    for h in [16,48]:np.testing.assert_array_equal(summarize_past(lookup[key],cutoff,h),summarize_past(perturbed,cutoff,h))
    for model in models:
        actual={r['window_key'] for r in records if r['run']!=model['held_out_run']}
        assert set(model['training_window_keys'])==actual
    audit=json.loads((DEFAULT_OUT/'audit_summary.json').read_text(encoding='utf-8'))
    assert all(sha(SOURCE/name)==digest for name,digest in audit['source_sha256_before'].items())
    verification=dict(straight_motion_1s_3s_check=True,future_positions_and_all_labels_perturbation_invariance=True,
        exact_training_key_split_verified=True,same_cohort_verified=True,source_hashes_unchanged=True)
    (OUTPUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=[r['offset_s'] for r in rows]
    fig,ax=plt.subplots(1,2,figsize=(9,3.7))
    ax[0].plot(t,[r['median_speed_m_s'] for r in rows],marker='o')
    ax[1].plot(t,[r['median_turn_degrees'] for r in rows],marker='o')
    for a in ax:a.axvline(0,color='gray',linestyle='--');a.set_xlabel('Seconds relative to annotated switch')
    ax[0].set_ylabel('Median recent 0.5s speed (m/s)');ax[1].set_ylabel('Median past-1s turn (degrees)')
    for a in ax:
        for i,r in enumerate(rows):a.annotate(f"n={r['observed_events']}",(t[i],a.lines[0].get_ydata()[i]),xytext=(0,8),textcoords='offset points',fontsize=7,ha='center')
    fig.suptitle('Descriptive event alignment: changing sample size, no matched controls')
    fig.tight_layout();fig.savefig(OUTPUT/'event_aligned_motion.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    s=run()
    print(json.dumps({k:s[k] for k in ['status','old_windows','common_windows','removed_windows','old_positive_windows',
        'common_positive_windows','old_distinct_events','common_distinct_events','events_lost_to_history_requirement',
        'methods','event_aligned_summary','all_20_logit_fits_converged']},indent=2))
