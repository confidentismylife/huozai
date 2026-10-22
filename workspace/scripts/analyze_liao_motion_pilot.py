"""Cached-score ranking diagnostics and numerical checks; no model refitting."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import numpy as np

from run_liao_motion_pilot import (CONFIG,OUTPUT,average_precision,logit_objective,
                                  motion_features,hierarchy_weights)
from audit_liao_dynamic_data import ROOT, SOURCE, DEFAULT_OUT, sha


def ranking(records,name,fraction):
    tp=fp=selected=detected=events=0.0
    rows=[]
    for run in sorted({r['run'] for r in records}):
        group=[r for r in records if r['run']==run]
        scores=np.array([float(r[name]) for r in group]);y=np.array([int(r['switch_within_horizon']) for r in group])
        k=max(1,math.ceil(fraction*len(group)))
        boundary=np.sort(scores)[-k]
        strict=scores>boundary;tied=scores==boundary
        need=k-int(strict.sum());m=int(tied.sum())
        weights=strict.astype(float)+tied*(need/m)
        positive_events={(r['run'],r['local_id'],r['first_different_frame']) for r in group if int(r['switch_within_horizon'])}
        detected_here=0.0
        for event in positive_events:
            ix=np.array([(r['run'],r['local_id'],r['first_different_frame'])==event and int(r['switch_within_horizon'])==1 for r in group])
            if (ix & strict).any():detected_here+=1
            else:
                d=int(np.sum(ix & tied))
                if need>m-d:detected_here+=1
                elif d:detected_here+=1-math.comb(m-d,need)/math.comb(m,need)
        tp_here=float(weights@y);fp_here=float(weights@(1-y))
        rows.append(dict(run=run,windows=len(group),selected=k,expected_tp_windows=tp_here,
                         expected_fp_windows=fp_here,expected_detected_events=detected_here,
                         distinct_events=len(positive_events),score_boundary=float(boundary),ties_at_boundary=m))
        tp+=tp_here;fp+=fp_here;selected+=k;detected+=detected_here;events+=len(positive_events)
    return dict(method=name,fraction=fraction,selected_windows=int(selected),expected_tp_windows=tp,
                expected_fp_windows=fp,expected_window_recall=tp/sum(int(r['switch_within_horizon']) for r in records),
                expected_precision=tp/selected,expected_detected_events=detected,
                expected_event_recall=detected/events,per_run=rows,
                note='Offline ranking only. Equal scores use expected uniform selection, never outcome-based ties. Does not define an online alert threshold.')


def verify():
    rng=np.random.default_rng(913)
    design=np.column_stack([np.ones(40),rng.normal(size=(40,6))])
    y=rng.integers(0,2,40);w=rng.uniform(.1,1,size=40);w/=w.sum();beta=rng.normal(size=7)
    value,grad=logit_objective(beta,design,y,w,.01)
    numeric=[]
    for j in range(len(beta)):
        delta=np.zeros(len(beta));delta[j]=1e-6
        numeric.append((logit_objective(beta+delta,design,y,w,.01)[0]-logit_objective(beta-delta,design,y,w,.01)[0])/2e-6)
    error=float(np.max(np.abs(grad-numeric)))
    assert error<1e-7
    assert np.isclose(average_precision(np.array([0,1,0,1]),np.ones(4)*.01),.5)
    assert np.isclose(average_precision(np.array([0,1,0,1]),np.array([.1,.9,.2,.8])),1)
    feature=dict(positions=[dict(relative_time_s=i/16,x_m=i/16,y_m=0) for i in range(-16,1)])
    vector,diagnostic=motion_features(feature)
    np.testing.assert_allclose(vector,[1,1,1,0,0,0],atol=1e-10)
    translated=dict(positions=[dict(relative_time_s=r['relative_time_s'],x_m=r['x_m']+100,y_m=r['y_m']-50) for r in feature['positions']])
    rotated=dict(positions=[dict(relative_time_s=r['relative_time_s'],x_m=-r['y_m'],y_m=r['x_m']) for r in feature['positions']])
    np.testing.assert_allclose(motion_features(translated)[0],vector,atol=1e-10)
    np.testing.assert_allclose(motion_features(rotated)[0],vector,atol=1e-10)
    audit=json.loads((DEFAULT_OUT/'audit_summary.json').read_text(encoding='utf-8'))
    assert all(sha(SOURCE/name)==digest for name,digest in audit['source_sha256_before'].items())
    return dict(logit_gradient_max_abs_error=error,AP_equal_score_and_perfect_ranking_checks=True,
                straight_motion_geometry_check=True,translation_rotation_invariance_check=True,
                source_hashes_unchanged=True)


def main():
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    records=list(csv.DictReader((OUTPUT/'development_out_of_fold_predictions.csv').open(encoding='utf-8-sig')))
    diagnostics=[ranking(records,name,q) for name in config['models'] for q in [.01,.05,.10,.20]]
    result=dict(status='cached_scores_only_no_refitting',ranking=diagnostics,verification=verify())
    (OUTPUT/'ranking_and_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({**result,'ranking':[{k:r[k] for k in ['method','fraction','selected_windows','expected_precision','expected_event_recall','expected_tp_windows']} for r in diagnostics]},indent=2))


if __name__=='__main__':main()
