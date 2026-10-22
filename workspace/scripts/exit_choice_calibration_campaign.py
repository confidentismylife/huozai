"""Frozen cached-score calibration study. No LLM, PPO, or sealed-scene calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import exit_choice_initial_benchmark as old

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/exit_choice_initial_benchmark_20261012"
OUT=ROOT/"outputs/exit_choice_calibration_campaign_20261012"
DEADLINE="2026-10-12T12:16:40+00:00"
METHODS=["raw_qwen","frequency","temperature","bias_calibration","score_adapter",
         "observables_logit","observables_nonlinear","hybrid_qwen_logit"]
CHANNELS=["primary","guide_conditional"]
FIT_CACHE={}


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def save(name,obj):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def csvsave(name,rows):
    import csv
    if not rows: return
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def preserve():
    rows=read(OUT/"preservation_before.json")
    result=[{**r,"sha256_after":sha(r["path"])} for r in rows]
    for r in result: r["unchanged"]=r["sha256"]==r["sha256_after"]
    assert all(r["unchanged"] for r in result)
    save("preservation_after.json",result)
    return len(result)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"frozen_plan.json").exists():
        preserve(); print("Existing frozen plan retained"); return
    files=[p for p in SOURCE.rglob("*") if p.is_file() and p.suffix!=".log"]
    files += [ROOT/"scripts/exit_choice_initial_benchmark.py",ROOT/"outputs/local_llm_selection.json",
              ROOT/"outputs/third_round_development_status.json",
              Path("C:/Users/摩天轮大盗/Downloads/Exit choice.xlsx"),
              Path("C:/Users/摩天轮大盗/Downloads/Exit choice ppt.pptx")]
    files += list((ROOT/"outputs/exit_choice_transition_audit_20261012").glob("*.json"))
    files += list((ROOT/"outputs/exit_choice_transition_audit_20261012").glob("*.csv"))
    save("preservation_before.json",[{"path":str(p),"sha256":sha(p)} for p in files])
    cases=read(SOURCE/"strict_initial_cases.json")
    plan={"frozen_at_utc":now(),"autonomous_session_deadline_utc":DEADLINE,
          "question":"Does training-date calibration improve Qwen, and does its score add information beyond observable-attribute Logit?",
          "data":"103 strict single-record cases across four previously inspected dates; no transition labels",
          "cases_sha256":sha(SOURCE/"strict_initial_cases.json"),
          "predictions_sha256":sha(SOURCE/"all_predictions.json"),
          "channels":CHANNELS,"methods":METHODS,
          "outer_split":"leave one entire date out",
          "inner_split":"leave one of the three training dates out; select by equal-date mean log loss",
          "same_supervision":"All fitted models receive the identical outer-training-date human labels",
          "ridge_sum_nll_grid":[0.01,0.1,1.0,10.0],
          "temperature_grid":[0.5,1.0,2.0,4.0],
          "smoothing_frequency":1.0,"score_log_floor":1e-6,
          "feature_scaling":"Dist/10, People/10, log(LLM probability), log1p(People); fixed before evaluation",
          "Fam":"constant [1,0,0], absorbed into E1 intercept; no independent familiarity effect",
          "calibration_models":{
              "temperature":"softmax(log(p)/T); positive T, cannot change predicted exit",
              "bias_calibration":"softmax(log(p)+two learned exit intercepts)",
              "score_adapter":"two exit intercepts plus learned common coefficient on log(p); coefficient unconstrained, treated as statistical adapter",
              "observables_logit":"two intercepts + Dist + People (+ Guide in conditional channel)",
              "observables_nonlinear":"observable Logit plus log1p(People), one additional feature",
              "hybrid_qwen_logit":"observable Logit plus log(p), one additional feature"},
          "primary_contrasts":["bias_calibration vs raw_qwen and frequency",
                               "score_adapter vs observables_logit",
                               "hybrid_qwen_logit vs observables_logit and observables_nonlinear"],
          "negative_control":"20 fixed seeds shuffle complete Qwen probability vectors within each date, without using labels; repeat hybrid with identical inner tuning",
          "negative_control_seeds":list(range(2026101201,2026101221)),
          "negative_control_inference":"descriptive case-association check, not a formal exchangeability-based p-value; repetitions are not human samples",
          "metrics":["per-date log loss","equal-date mean log loss","pooled log loss","accuracy","Brier","exit-share TV"],
          "uncertainty":"Four dates and shared training sets; report per-date differences and range, no person-level significance claims",
          "probability_support":"No observed E3 choices, cannot validate E3 sensitivity",
          "guide_limit":"Staff timing unverified, conditional predictive analysis only",
          "later_stability_check":"12 label-blind selected strict cases, reverse exit presentation; at most 24 new local requests across two channels, separately frozen before inference",
          "no_sealed_scenarios_or_ppo_training":True,"source_script_sha256":sha(__file__),
          "status":"exploratory_followup; outer dates already inspected in earlier work"}
    save("frozen_plan.json",plan)
    (OUT/"source_snapshot").mkdir(exist_ok=True)
    (OUT/"source_snapshot/exit_choice_calibration_campaign.py").write_bytes(Path(__file__).read_bytes())
    save("session_status.json",{"at_utc":now(),"deadline_utc":DEADLINE,"phase":"plan_frozen","cases":len(cases),"state":"running"})
    preserve()
    print(json.dumps({"phase":"frozen","cases":len(cases),"deadline_utc":DEADLINE}))


def load_data():
    plan=read(OUT/"frozen_plan.json")
    assert sha(SOURCE/"strict_initial_cases.json")==plan["cases_sha256"]
    assert sha(SOURCE/"all_predictions.json")==plan["predictions_sha256"]
    cases=read(SOURCE/"strict_initial_cases.json")
    data=read(SOURCE/"all_predictions.json")
    maps={c:{r["case_id"]:r["probabilities"] for r in data if r["cohort"]=="strict" and r["channel"]==c and r["method"]=="qwen3_8b"} for c in CHANNELS}
    y=np.array([r["initial_label"]-1 for r in cases])
    dates=np.array([r["date"] for r in cases])
    scores={c:np.array([maps[c][r["record_id"]] for r in cases]) for c in CHANNELS}
    assert all(len(m)==len(cases) for m in maps.values())
    return plan,cases,y,dates,scores


def matrices(cases,probabilities,guide,method):
    n=len(cases)
    asc=np.zeros((n,3,2)); asc[:,0,0]=1; asc[:,1,1]=1
    lp=np.log(np.maximum(probabilities,1e-6))
    if method=="bias_calibration": return asc,lp,["ASC_E1","ASC_E2"]
    if method=="score_adapter": return np.concatenate([asc,lp[:,:,None]],axis=2),np.zeros((n,3)),["ASC_E1","ASC_E2","Qwen_log_score"]
    base=old.design(cases,guide)
    names=["ASC_E1","ASC_E2","Dist_per10","People_per10"]+(["Guide"] if guide else [])
    if method=="observables_nonlinear":
        cc=np.array([[np.log1p(e["people"]) for e in r["exits"]] for r in cases])
        base=np.concatenate([base,cc[:,:,None]],axis=2); names.append("log1p_People")
    elif method=="hybrid_qwen_logit":
        base=np.concatenate([base,lp[:,:,None]],axis=2); names.append("Qwen_log_score")
    return base,np.zeros((n,3)),names


def optimize(x,offset,y,ridge):
    theta=np.zeros(x.shape[-1])
    def terms(t,hessian=True):
        z=np.einsum("njd,d->nj",x,t)+offset; pp=old.softmax(z)
        zm=z.max(1)
        value=float(np.sum(zm+np.log(np.exp(z-zm[:,None]).sum(1))-z[np.arange(len(y)),y])+ridge*np.dot(t,t)/2)
        if not hessian: return value
        ex=np.einsum("nj,njd->nd",pp,x)
        grad=(ex-x[np.arange(len(y)),y]).sum(0)+ridge*t
        h=np.einsum("nj,njd,nje->de",pp,x,x)-np.einsum("nd,ne->de",ex,ex)+ridge*np.eye(len(t))
        return value,grad,h
    for it in range(150):
        loss,g,h=terms(theta)
        if abs(g).max()<1e-7: break
        step=np.linalg.solve(h,g); a=1.
        while a>1e-10 and terms(theta-a*step,False)>loss-1e-4*a*np.dot(g,step): a/=2
        theta-=a*step
    loss,g,h=terms(theta)
    assert abs(g).max()<1e-6,(it,g)
    return theta,{"iterations":it+1,"max_abs_gradient":float(abs(g).max()),
                  "min_hessian_eigenvalue":float(np.linalg.eigvalsh(h).min()),
                  "penalized_sum_nll":loss}


def fitted_prediction(method,channel,cases,y,scores,train,test,param,seed=None):
    if method=="raw_qwen": return scores[test],{}
    if method=="frequency":
        pp=(np.bincount(y[train],minlength=3)+1)/(len(train)+3)
        return np.tile(pp,(len(test),1)),{"train_counts":np.bincount(y[train],minlength=3).tolist()}
    if method=="temperature": return old.softmax(np.log(np.maximum(scores[test],1e-6))/param),{"temperature":param}
    x,offset,names=matrices(cases,scores,channel=="guide_conditional",method)
    key=(channel,method,tuple(train.tolist()),param,seed)
    if key not in FIT_CACHE:
        FIT_CACHE[key]=optimize(x[train],offset[train],y[train],param)
    theta,stats=FIT_CACHE[key]
    pp=old.softmax(np.einsum("njd,d->nj",x[test],theta)+offset[test])
    return pp,{**stats,"coefficients":theta.tolist(),"coefficient_names":names,"ridge":param}


def nll(y,p): return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-6)).mean())


def tune(method,channel,cases,y,dates,scores,train,plan,seed=None):
    if method in ("raw_qwen","frequency"): return None,[]
    grid=plan["temperature_grid"] if method=="temperature" else plan["ridge_sum_nll_grid"]
    logs=[]
    for param in grid:
        folds=[]
        for date in sorted(set(dates[train])):
            tr=train[dates[train]!=date]; va=train[dates[train]==date]
            assert not set(tr)&set(va)
            pp,_=fitted_prediction(method,channel,cases,y,scores,tr,va,param,seed)
            folds.append({"held_out_training_date":date,"train_dates":sorted(set(dates[tr])),
                          "train_n":len(tr),"validation_n":len(va),"log_loss":nll(y[va],pp)})
        logs.append({"parameter":param,"mean_date_log_loss":float(np.mean([r["log_loss"] for r in folds])),"folds":folds})
    chosen=min(logs,key=lambda r:(r["mean_date_log_loss"],r["parameter"]))
    return chosen["parameter"],logs


def evaluate(method,channel,cases,y,dates,scores,plan,seed=None):
    outputs,details=[],[]
    for date in sorted(set(dates)):
        train=np.where(dates!=date)[0]; test=np.where(dates==date)[0]
        param,inner=tune(method,channel,cases,y,dates,scores,train,plan,seed)
        pp,fit=fitted_prediction(method,channel,cases,y,scores,train,test,param,seed)
        assert np.isfinite(pp).all() and np.allclose(pp.sum(1),1)
        details.append({"channel":channel,"method":method,"test_date":date,"seed":seed,
                        "training_ids":[cases[i]["record_id"] for i in train],
                        "test_ids":[cases[i]["record_id"] for i in test],
                        "chosen_parameter":param,"inner_selection":inner,**fit})
        for i,p in zip(test,pp):
            outputs.append({"channel":channel,"method":method,"test_date":date,"case_id":cases[i]["record_id"],
                            "true_exit":int(y[i]+1),"probabilities":p.tolist(),"seed":seed})
    return outputs,details


def aggregate(predictions,cases):
    lookup={r["record_id"]:r for r in cases}; groups=defaultdict(list)
    for r in predictions:
        for d in (r["test_date"],"pooled"):
            groups[(r["channel"],r["method"],r.get("seed"),d)].append(r)
    metrics=[]
    for (c,m,seed,d),rr in groups.items():
        pp=np.array([r["probabilities"] for r in rr])
        metrics.append({"channel":c,"method":m,"seed":seed,"date":d,**old.metrics([lookup[r["case_id"]] for r in rr],pp)})
    for c,m,seed in {(r["channel"],r["method"],r.get("seed")) for r in predictions}:
        rows=[r for r in metrics if r["channel"]==c and r["method"]==m and r["seed"]==seed and r["date"]!="pooled"]
        metrics.append({"channel":c,"method":m,"seed":seed,"date":"date_macro","n":sum(r["n"] for r in rows),
                        **{k:float(np.mean([r[k] for r in rows])) for k in ["log_loss","accuracy","brier","exit_share_tv"]},
                        "true_shares":[],"predicted_shares":[]})
    return metrics


def calibrate():
    plan,cases,y,dates,scores=load_data(); preds,details=[],[]
    for channel in CHANNELS:
        for method in METHODS:
            p,d=evaluate(method,channel,cases,y,dates,scores[channel],plan)
            preds+=p; details+=d
            save("calibration_progress.json",{"at_utc":now(),"channel":channel,"method":method,
                                              "completed_models":len(preds)//len(cases),"expected_models":len(CHANNELS)*len(METHODS)})
            print(json.dumps({"channel":channel,"method":method,"completed_models":len(preds)//len(cases)},ensure_ascii=False),flush=True)
    save("predictions.json",preds); csvsave("predictions.csv",preds)
    save("fit_details.json",details)
    mm=aggregate(preds,cases); save("metrics.json",mm); csvsave("metrics.csv",mm)
    save("calibration_status.json",{"completed_at_utc":now(),"predicted_case_method_rows":len(preds),
                                     "unique_optimizer_fits":len(FIT_CACHE),"extra_llm_calls":0,"ppo_training_runs":0})
    preserve()
    print(json.dumps([r for r in mm if r["date"]=="date_macro"],ensure_ascii=False),flush=True)


def null_controls():
    plan,cases,y,dates,scores=load_data(); preds,details=[],[]
    for channel in CHANNELS:
        for seed in plan["negative_control_seeds"]:
            perm=np.arange(len(cases))
            for date in sorted(set(dates)):
                ii=np.where(dates==date)[0]
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(date.replace('-','')),CHANNELS.index(channel)]))
                perm[ii]=rng.permutation(ii)
            shuffled=scores[channel][perm]
            p,d=evaluate("hybrid_qwen_logit",channel,cases,y,dates,shuffled,plan,seed)
            preds+=p; details+=d
            save("negative_control_progress.json",{"at_utc":now(),"channel":channel,"seed":seed,
                          "completed_runs":len(preds)//len(cases),"expected_runs":len(CHANNELS)*len(plan["negative_control_seeds"])})
            print(json.dumps({"channel":channel,"seed":seed,"completed_runs":len(preds)//len(cases)}),flush=True)
    save("negative_predictions.json",preds); csvsave("negative_predictions.csv",preds)
    save("negative_fit_details.json",details)
    mm=aggregate(preds,cases); save("negative_metrics.json",mm)
    main=read(OUT/"metrics.json"); summary=[]
    for c in CHANNELS:
        true=next(r for r in main if r["date"]=="date_macro" and r["channel"]==c and r["method"]=="hybrid_qwen_logit")
        losses=[r["log_loss"] for r in mm if r["date"]=="date_macro" and r["channel"]==c]
        summary.append({"channel":c,"real_hybrid_macro_log_loss":true["log_loss"],"randomized_runs":len(losses),
                        "shuffled_macro_log_loss_mean":float(np.mean(losses)),
                        "shuffled_macro_log_loss_min":min(losses),"shuffled_macro_log_loss_max":max(losses),
                        "shuffled_runs_not_worse_than_real":sum(v<=true["log_loss"] for v in losses),
                        "not_a_formal_p_value":True})
    save("negative_control_summary.json",summary)
    preserve(); print(json.dumps(summary,ensure_ascii=False),flush=True)


def audit():
    plan,cases,y,dates,scores=load_data()
    details=read(OUT/"fit_details.json")
    errors=[]
    lookup={r["record_id"]:r["date"] for r in cases}
    for d in details:
        assert set(d["training_ids"]).isdisjoint(d["test_ids"])
        assert all(lookup[cid]!=d["test_date"] for cid in d["training_ids"])
        for candidate in d["inner_selection"]:
            for f in candidate["folds"]:
                assert d["test_date"] not in f["train_dates"] and f["held_out_training_date"]!=d["test_date"]
    # An independently computed scalar penalized loss checks the fitted gradient.
    train=np.where(dates!="2019-10-31")[0]
    x,off,names=matrices(cases,scores["primary"],False,"hybrid_qwen_logit")
    t,info=optimize(x[train],off[train],y[train],.1)
    def scalar(v):
        p=old.softmax(np.einsum('njd,d->nj',x[train],v)+off[train])
        return -np.log(p[np.arange(len(train)),y[train]]).sum()+.05*np.dot(v,v)
    eps=1e-5
    numerical=[(scalar(t+np.eye(len(t))[j]*eps)-scalar(t-np.eye(len(t))[j]*eps))/(2*eps) for j in range(len(t))]
    assert max(abs(v) for v in numerical)<1e-4
    # Positive temperature must preserve all class ranks, including ties.
    for c in CHANNELS:
        for temp in plan["temperature_grid"]:
            assert np.array_equal(scores[c].argmax(1),old.softmax(np.log(np.maximum(scores[c],1e-6))/temp).argmax(1))
    # Reimplemented observable baseline must reproduce previous held-out outputs.
    prev=read(SOURCE/"all_predictions.json"); new=read(OUT/"predictions.json")
    diff=[]
    for c in CHANNELS:
        pa={r["case_id"]:r["probabilities"] for r in prev if r["cohort"]=="strict" and r["channel"]==c and r["method"]=="logit"}
        pb={r["case_id"]:r["probabilities"] for r in new if r["channel"]==c and r["method"]=="observables_logit"}
        diff += [float(np.max(abs(np.array(pa[k])-pb[k]))) for k in pa]
    assert max(diff)<1e-8,max(diff)
    result={"outer_and_inner_dates_disjoint":True,"positive_temperature_ranks_preserved":True,
            "finite_difference_max_gradient":max(abs(v) for v in numerical),
            "prior_observable_logit_max_probability_difference":max(diff),
            "protected_files_unchanged":preserve(),"extra_llm_calls":0,"ppo_training_runs":0}
    save("verification.json",result); print(json.dumps(result),flush=True)


def report():
    plan,cases,y,dates,scores=load_data(); metrics=read(OUT/"metrics.json")
    contrasts=[]
    targets=[("bias_calibration","raw_qwen"),("bias_calibration","frequency"),
             ("score_adapter","observables_logit"),("hybrid_qwen_logit","observables_logit"),
             ("hybrid_qwen_logit","observables_nonlinear")]
    for c in CHANNELS:
        for a,b in targets:
            for date in list(sorted(set(dates)))+["date_macro","pooled"]:
                aa=next(r for r in metrics if r["channel"]==c and r["method"]==a and r["date"]==date)
                bb=next(r for r in metrics if r["channel"]==c and r["method"]==b and r["date"]==date)
                contrasts.append({"channel":c,"candidate":a,"reference":b,"date":date,
                                  "log_loss_difference":aa["log_loss"]-bb["log_loss"],
                                  "accuracy_difference_pp":(aa["accuracy"]-bb["accuracy"])*100,
                                  "brier_difference":aa["brier"]-bb["brier"]})
    save("contrasts.json",contrasts); csvsave("contrasts.csv",contrasts)
    names={"raw_qwen":"原始Qwen","frequency":"训练出口频率","temperature":"温度校准",
           "bias_calibration":"出口偏差校准","score_adapter":"Qwen分数统计适配",
           "observables_logit":"观测属性Logit","observables_nonlinear":"属性Logit加人群非线性",
           "hybrid_qwen_logit":"属性Logit加Qwen分数"}
    lines=["# 真人出口选择的缓存校准与额外信息实验", "", "2026年10月12日。两小时自主任务的第一阶段。", "",
           "## 问题与范围", "", "检验两个问题：用其他日期真人数据校准Qwen能否改善预测；Qwen分数能否在相同观测属性Logit之外提供信息。",
           "103条单次记录，四个既有日期。数据已被前轮查看，本轮是冻结方案的探索跟进。仅拟合小型统计参数，没有重新训练LLM或PPO，没有访问48个最终场景。", "",
           "## 公平对照与验证", "", "所有拟合模型使用相同外层训练日期的人类标签；每轮留出一个完整日期，正则或温度在另三个日期内部按日期留出选择。每个模型四个候选设置，按日期等权对数损失选择。",
           "温度只改变概率尖锐程度，正温度保持出口排名；偏差校准学习出口整体偏好；分数适配可改变Qwen分数权重；联合模型给属性Logit增加一个Qwen分数特征。",
           "增加log1p(People)的统计对照同样多一个特征，用于区分LLM信息和增加模型灵活性。Fam固定为[1,0,0]，不能估计独立熟悉效应。Staff仅作时点未验证的条件性分析。", ""]
    for c in CHANNELS:
        lines += ["## "+("基础信息" if c=="primary" else "附加Staff编码（条件性）"),"",
                  "| 方法 | 日期等权对数损失 | 日期等权准确率 | 合并准确率 | 合并Brier |",
                  "|---|---:|---:|---:|---:|"]
        for m in METHODS:
            macro=next(r for r in metrics if r["channel"]==c and r["method"]==m and r["date"]=="date_macro")
            pooled=next(r for r in metrics if r["channel"]==c and r["method"]==m and r["date"]=="pooled")
            lines.append(f"| {names[m]} | {macro['log_loss']:.4f} | {macro['accuracy']:.2%} | {pooled['accuracy']:.2%} | {pooled['brier']:.4f} |")
        lines += ["", "| 留出日期 | 出口偏差校准损失 | 观测Logit损失 | 加Qwen损失 | 加非线性损失 |", "|---|---:|---:|---:|---:|"]
        for date in sorted(set(dates)):
            row=[next(r for r in metrics if r["channel"]==c and r["method"]==m and r["date"]==date)["log_loss"] for m in ["bias_calibration","observables_logit","hybrid_qwen_logit","observables_nonlinear"]]
            lines.append("| "+date+" | "+" | ".join(f"{v:.4f}" for v in row)+" |")
        lines.append("")
    lines += ["## 预先确定的对比", "", "差值为候选减参照：对数损失为负表示改善，准确率百分点为正表示改善。", "",
              "| 信息条件 | 候选 | 参照 | 日期等权损失差 | 日期等权准确率差 |", "|---|---|---|---:|---:|"]
    for r in contrasts:
        if r["date"]=="date_macro":
            lines.append(f"| {r['channel']} | {names[r['candidate']]} | {names[r['reference']]} | {r['log_loss_difference']:+.4f} | {r['accuracy_difference_pp']:+.2f} |")
    if (OUT/"negative_control_summary.json").exists():
        lines += ["", "## 打乱分数与个体对应的对照", "",
                  "20个固定种子分别在每个日期内打乱完整Qwen概率向量，保留该日期分数分布，重复相同联合模型及内部调参。重复不是新人类样本，也不作正式置换显著性检验。", "",
                  "| 条件 | 真实对应联合损失 | 打乱均值 | 打乱范围 | 打乱结果不差于真实对应的次数 |", "|---|---:|---:|---|---:|"]
        for r in read(OUT/"negative_control_summary.json"):
            lines.append(f"| {r['channel']} | {r['real_hybrid_macro_log_loss']:.4f} | {r['shuffled_macro_log_loss_mean']:.4f} | {r['shuffled_macro_log_loss_min']:.4f}至{r['shuffled_macro_log_loss_max']:.4f} | {r['shuffled_runs_not_worse_than_real']}/20 |")
    lines += ["", "## 解释限制", "", "校准改善原始Qwen不等于LLM具有额外价值。偏差模型可主要学到本建筑出口频率，需要与频率和观测Logit同时比较。",
              "四个事件、交叉验证共享训练数据，不报告把103人或20次打乱当作独立事件的显著性结论。逐日期差异比单一合并准确率更重要。",
              "出口3无人选择，不能验证该类识别能力，也不能把它标为非法。当前资料不足以验证动态改道、提醒反应时间和引导因果效果。",
              "本轮指标为人类行为预测，不是安全出口优化、疏散时间改善或烟气暴露减少。", "",
              "## 复现文件与参考", "", "frozen_plan.json记录方案；predictions.csv保存所有人的外层预测；fit_details.json保存训练编号、内部调参与系数；verification.json保存数值与日期隔离审核。",
              "[Guo等，On Calibration of Modern Neural Networks，ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)支持温度校准定义；本轮应用于LLM输出概率，效果由本数据检验，不沿用原文性能结论。",
              "[Kull等，Beyond temperature scaling，NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)讨论多分类校准。本轮使用更小的偏差/分数模型，不声称完整复现Dirichlet校准。", ""]
    path=ROOT/"reports/Exit_choice缓存校准与额外信息实验_2026-10-12.md"
    path.write_text("\n".join(lines),encoding="utf-8")
    preserve(); save("session_status.json",{"at_utc":now(),"deadline_utc":DEADLINE,"state":"running",
            "phase":"cached_calibration_report_complete","report":str(path),"continue_with":"stability audit and evidence/report consolidation"})
    print(json.dumps({"report":str(path),"state":"running","deadline_utc":DEADLINE},ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["prepare","calibrate","null_controls","audit","report"])
    command=parser.parse_args().command
    globals()[command]()
