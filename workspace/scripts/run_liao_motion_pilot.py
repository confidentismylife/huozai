"""Fixed, development-only, leave-one-run-out rare-event baselines.

Requires existing NumPy/SciPy/Matplotlib. No LLM, map assumptions, PPO, or
confirmation scoring. Probability features are all derived from past positions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from audit_liao_dynamic_data import ROOT, sha

CONFIG = ROOT / "schemas/liao_motion_pilot_20261013.json"
OUTPUT = ROOT / "outputs/liao_motion_baseline_20261013"


def motion_features(feature):
    positions = feature['positions']
    times = np.array([r['relative_time_s'] for r in positions])
    xy = np.array([[r['x_m'],r['y_m']] for r in positions])
    if len(xy) != 17 or not np.allclose(times, np.arange(-16,1)/16):
        raise ValueError('Expected exactly the 17 past samples from -1 to 0 seconds')
    early_v, recent_v = (xy[8]-xy[0])*2, (xy[16]-xy[8])*2
    early, recent = np.linalg.norm(early_v), np.linalg.norm(recent_v)
    heading_known = early >= .1 and recent >= .1
    turn = np.arccos(np.clip(early_v @ recent_v/(early*recent),-1,1)) if heading_known else 0.0
    path = np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()
    net = np.linalg.norm(xy[-1]-xy[0])
    vector = np.array([path, recent, early, recent-early, turn, max(0,path-net)])
    diagnostic = dict(recent_heading_degrees=float(np.degrees(np.arctan2(recent_v[1],recent_v[0]))) if recent >= .1 else None,
                      recent_heading_known=bool(recent >= .1), both_headings_known=bool(heading_known))
    return vector, diagnostic


def hierarchy_weights(records):
    runs = {r['run'] for r in records}
    counts = Counter((r['run'],r['local_id']) for r in records)
    people = Counter(run for run,_ in counts)
    w = np.array([1/(len(runs)*people[r['run']]*counts[(r['run'],r['local_id'])]) for r in records])
    assert np.isclose(w.sum(),1)
    return w


def smooth_frequency(y, w, n_person_runs, prior, prior_n):
    return float((float(w @ y)*n_person_runs+prior*prior_n)/(n_person_runs+prior_n))


def logit_objective(beta, design, y, weights, l2):
    z = design @ beta
    loss = weights @ (np.logaddexp(0,z)-y*z) + l2*.5*(beta[1:] @ beta[1:])
    grad = design.T @ (weights*(expit(z)-y))
    grad[1:] += l2*beta[1:]
    return float(loss), grad


def fit_logit(x,y,w,p0,l2,maxiter):
    center = w @ x
    scale = np.sqrt(w @ ((x-center)**2))
    scale = np.where(scale > 1e-8,scale,1)
    design = np.column_stack([np.ones(len(x)),(x-center)/scale])
    beta = np.zeros(design.shape[1]);beta[0]=np.log(p0/(1-p0))
    result = minimize(logit_objective,beta,args=(design,y,w,l2),jac=True,method='L-BFGS-B',
                      options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-8})
    if not result.success:
        raise RuntimeError(f'Logit optimizer did not converge: {result.message}')
    return center,scale,result.x,dict(success=bool(result.success),iterations=int(result.nit),
                                   objective=float(result.fun),max_abs_gradient=float(np.max(np.abs(result.jac))))


def average_precision(y,p):
    # Evaluate all equal-score observations together; no optimistic tie sorting.
    if y.sum()==0:return None
    order=np.argsort(-p,kind='stable');yy=y[order];pp=p[order]
    ends=np.r_[np.flatnonzero(pp[:-1]!=pp[1:]),len(pp)-1]
    tp=np.cumsum(yy)[ends];recall=tp/y.sum();precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0,recall])*precision))


def metrics(y,p,records,threshold):
    p=np.clip(p,1e-9,1-1e-9)
    alarm=p>=threshold
    tp=int(np.sum(alarm & (y==1)));fp=int(np.sum(alarm & (y==0)))
    losses=-(y*np.log(p)+(1-y)*np.log1p(-p));brier=(p-y)**2
    w=hierarchy_weights(records)
    events={(r['run'],r['local_id'],r['first_different_frame']) for r in records if int(r['switch_within_horizon'])}
    detected={(r['run'],r['local_id'],r['first_different_frame']) for r,a in zip(records,alarm) if int(r['switch_within_horizon']) and a}
    return dict(windows=len(y),positive_windows=int(y.sum()),distinct_events=len(events),
                hierarchical_macro_NLL=float(w@losses),hierarchical_macro_Brier=float(w@brier),
                window_NLL=float(losses.mean()),window_Brier=float(brier.mean()),
                average_precision=average_precision(y,p),
                threshold=threshold,true_positive_windows=tp,false_positive_windows=fp,
                window_recall=tp/int(y.sum()) if y.sum() else None,
                window_precision=tp/(tp+fp) if tp+fp else None,
                detected_events=len(detected),event_recall=len(detected)/len(events) if events else None,
                false_alarms_per_eligible_person_minute=60*fp/len(y),
                denominator_note='Each eligible anchor represents one second of prediction opportunities; not full trajectory exposure time.',
                accuracy=float(np.mean(alarm==y)))


def run():
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    source=ROOT/config['data_interface']
    OUTPUT.mkdir(exist_ok=True)
    records=[r for r in csv.DictReader((source/'outcomes_and_provenance.csv').open(encoding='utf-8-sig'))
             if r['partition']=='proposed_development']
    if set(r['run'] for r in records)!=set(config['development_runs']):
        raise ValueError('Development run mismatch')
    keep={r['window_key'] for r in records}
    features={}
    for line in (source/'past_only_inputs.jsonl').open(encoding='utf-8'):
        row=json.loads(line)
        if row['window_key'] in keep:features[row['window_key']]=row['features']
    pairs=[motion_features(features[r['window_key']]) for r in records]
    x=np.array([f for f,_ in pairs]);y=np.array([int(r['switch_within_horizon']) for r in records])
    rule=config['fixed_rule']
    trigger=(x[:,1]<rule['slow_speed_m_s']) | ((x[:,1]>=rule['heading_min_speed_m_s']) &
        (x[:,2]>=rule['heading_min_speed_m_s']) & (x[:,4]>rule['turn_radians']))
    names=config['models'];predictions={name:np.full(len(records),np.nan) for name in names}
    fold_stats=[];models=[]
    for held_run in config['development_runs']:
        test=np.array([r['run']==held_run for r in records]);train=~test
        training=[r for r,a in zip(records,train) if a]
        testing=[r for r,a in zip(records,test) if a]
        assert not {r['run'] for r in training} & {r['run'] for r in testing}
        assert not {r['window_key'] for r in training} & {r['window_key'] for r in testing}
        w=hierarchy_weights(training)
        n_people=len({(r['run'],r['local_id']) for r in training})
        smooth=config['frequency_smoothing']
        p0=smooth_frequency(y[train],w,n_people,smooth['prior_probability'],smooth['prior_person_run_equivalents'])
        predictions[names[0]][test]=p0
        bin_risk=[]
        for value in [False,True]:
            mask=trigger[train]==value
            if mask.any():
                bin_people=len({(r['run'],r['local_id']) for r,a in zip(training,mask) if a})
                # Preserve the chosen hierarchy weights, then condition on rule bin.
                risk=smooth_frequency(y[train][mask],w[mask]/w[mask].sum(),bin_people,
                                      smooth['prior_probability'],smooth['prior_person_run_equivalents'])
            else:risk=p0
            bin_risk.append(risk)
        predictions[names[1]][test]=np.array(bin_risk)[trigger[test].astype(int)]
        c,s,b,opt=fit_logit(x[train],y[train],w,p0,config['logit']['l2'],config['logit']['maxiter'])
        predictions[names[2]][test]=expit(np.column_stack([np.ones(test.sum()),(x[test]-c)/s])@b)
        models.append(dict(held_out_run=held_run,train_windows=int(train.sum()),train_positive_windows=int(y[train].sum()),
                           frequency=p0,rule_bin_risks=bin_risk,center=c.tolist(),scale=s.tolist(),beta=b.tolist(),optimizer=opt))
        for name in names:
            fold_stats.append(dict(held_out_run=held_run,method=name,**metrics(y[test],predictions[name][test],testing,config['decision_probability_threshold'])))
    assert all(np.isfinite(p).all() for p in predictions.values())
    rows=[]
    for ix,r in enumerate(records):
        rows.append({**r,**{name:float(predictions[name][ix]) for name in names},
                     'fixed_rule_trigger':bool(trigger[ix]),**pairs[ix][1]})
    def write_csv(name,data):
        with (OUTPUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_csv('development_out_of_fold_predictions.csv',rows)
    write_csv('per_run_metrics.csv',fold_stats)
    feature_rows=[dict(window_key=r['window_key'],**{n:float(v) for n,v in zip(config['feature_names'],values)})
                  for r,values in zip(records,x)]
    write_csv('past_only_motion_features.csv',feature_rows)
    summary=dict(status='development_pilot_completed_no_confirmation_scores',
        windows=len(records),positive_windows=int(y.sum()),
        methods={name:metrics(y,predictions[name],records,config['decision_probability_threshold']) for name in names},
        config_sha256=sha(CONFIG),code_sha256=sha(Path(__file__)),
        source_sha256={name:sha(source/name) for name in ['past_only_inputs.jsonl','outcomes_and_provenance.csv']},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        all_folds_converged=all(m['optimizer']['success'] for m in models),
        training_leakage_checks_passed=True,
        proposed_confirmation_performance_computed=False,
        cautions=['15 rare events, repeated people with unknown cross-run identity; exploratory only.',
                  'Motion heading is not exit intention. Head/body/video annotations are not online input.',
                  'Fixed 2% threshold is a pilot operating point, not optimized or deployment-validated.'])
    (OUTPUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'fold_models.json').write_text(json.dumps(models,indent=2)+'\n',encoding='utf-8')
    plot_scores(y,predictions,names)
    return summary


def plot_scores(y,predictions,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    colors=['tab:gray','tab:orange','tab:blue']
    for name,color in zip(names,colors):
        p=predictions[name]
        order=np.argsort(-p,kind='stable');pp=p[order];yy=y[order]
        ends=np.r_[np.flatnonzero(pp[:-1]!=pp[1:]),len(pp)-1]
        tp=np.cumsum(yy)[ends];recall=tp/y.sum();precision=tp/(ends+1)
        ax[0].plot(recall,precision,label=name,color=color)
        ax[1].hist(p[y==1],bins=np.linspace(0,max(.05,max(q.max() for q in predictions.values())),20),
                   histtype='step',color=color,label=name)
    ax[0].axhline(y.mean(),linestyle='--',color='black',label='Positive window prevalence')
    ax[0].set(xlabel='Recall',ylabel='Precision',title='Development leave-one-run-out predictions')
    ax[1].axvline(.02,linestyle='--',color='black')
    ax[1].set(xlabel='Predicted change probability',ylabel='Positive windows',title='30 positive windows / 15 events')
    fig.legend(loc='lower center',fontsize=7,ncol=2)
    fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(OUTPUT/'motion_pilot_diagnostic.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    result=run()
    print(json.dumps({k:result[k] for k in ['status','windows','positive_windows','all_folds_converged','methods']},indent=2))
