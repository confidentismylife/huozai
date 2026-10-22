"""Frozen 2x2 planning-budget/world-aggregation development pilot; no training."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.stdout.reconfigure(encoding="utf-8")
from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank
from fire_evac.source_world_planning import PLANS,OUTCOME_FIELDS,source_world_scenarios,compatible_source_worlds,select_reference_plan
from fire_evac.ppo_reference_planning import FrozenPPOContinuation,evaluate_continuation,policy_parameter_sha256,TRACE_FIELDS
from run_source_world_planning import make_env,vector
from diagnose_plan_teacher import digest,json_write,write_csv,interval,family_average

RULES = (("worst3","robust",3),("worst12","robust",12),("mean3","mean",3),("mean12","mean",12),
         ("point12","point",12),("oracle12","oracle",12))
FIXED = tuple(f"pure_ppo_{seed}" for seed in range(601,606))
ARMS = tuple(item[0] for item in RULES)+FIXED
BASE = "pure_ppo_601"
EPS = 1e-12


def signature(record):
    return json.dumps({k:v for k,v in record.items() if k!="scenario_id"},sort_keys=True)


def choice(risks,supported,scenario,mode,budget):
    return select_reference_plan(risks[:,:,:budget,:],supported,scenario.forecast_source_branch,mode,
                                 true_source=scenario.source_branch if mode=="oracle" else None)


def freeze(args,scenarios,old):
    files = sorted((ROOT/"src/fire_evac").glob("*.py"))+[
        Path(__file__),ROOT/"scripts/run_source_world_planning.py",ROOT/"scripts/diagnose_plan_teacher.py",
        ROOT/"outputs/behavior_pipeline/behavior_model.json",args.source/"development_manifest.json",
        ROOT/"outputs/plan_teacher_diagnosis_development_v1/summary.json",
        ROOT/"reports/教师世界聚合与规划预算配对验证方案.md"]
    files += [Path(model["path"]) for model in old["frozen_ppo_models"]]
    prior_banks = old["old_scenario_records"]+old["training_scenario_records"]+old["audit_scenario_records"]+old["excluded_previous_audit_records"]
    for directory in ("confirmatory_joint_control","guarded_confirmatory_v2"):
        for filename in ("train_scenarios.json","final_scenarios.json"):
            path=ROOT/"outputs"/directory/filename
            assert path.is_file()
            files.append(path)
            prior_banks.extend(json.loads(path.read_text(encoding="utf-8")))
    old_signatures={signature(s) for s in prior_banks}
    records=[asdict(s) for s in scenarios]
    assert len({signature(s) for s in records}) == 24
    assert all(signature(s) not in old_signatures for s in records)
    assert set(Counter(s.conflict_type for s in scenarios).values()) == {4}
    previous_seeds=set()
    for folder in (args.source/"audit_states",args.source/"training_states"):
        for path in folder.glob("*.npz"):
            with np.load(path,allow_pickle=False) as z:
                for key in ("planning_seeds","evaluation_seeds"):
                    if key in z.files:
                        previous_seeds.update(map(int,z[key]))
                previous_seeds.add(int(z["population_seed"]))
    new_population={280_000_000+i for i in range(48)}
    new_planning={300_000_000+1000*i+f for i in range(48) for f in range(12)}
    new_evaluation={320_000_000+1000*i+f for i in range(48) for f in range(6)}
    new_sensor={seed+9_000_001 for seed in new_planning|new_evaluation}
    namespaces=(new_population,new_planning,new_evaluation,new_sensor)
    assert all(a.isdisjoint(b) for i,a in enumerate(namespaces) for b in namespaces[i+1:])
    assert all(previous_seeds.isdisjoint(s) for s in namespaces)
    body={
        "role":"new_development_initial_state_world_aggregation_by_planning_budget_pilot_not_final",
        "freeze_allowed":False,"scenario_seed":20261802,"bootstrap_seed":20261803,
        "families":24,"populations_per_family":2,"scenario_records":records,
        "source_scenario_signatures_checked":len(old_signatures),
        "candidate_names":[p.name for p in PLANS]+list(FIXED),"arms":list(ARMS),
        "rules":[{"arm":arm,"mode":mode,"planning_futures":n} for arm,mode,n in RULES],
        "planning_futures":12,"evaluation_futures":6,"three_futures_nested_in_twelve":True,
        "outcome_fields":list(OUTCOME_FIELDS),"trace_fields":list(TRACE_FIELDS),
        "seed_formula":{"population":"280000000+2*j+rep","planning":"300000000+1000*(2*j+rep)+f",
                        "evaluation":"320000000+1000*(2*j+rep)+f","cloned_sensor":"future+9000001"},
        "seed_disjointness_scope":"current source audit/training caches plus within-pilot population/planning/evaluation/sensor namespaces",
        "planning_rollout_upper_bound":19584,"evaluation_rollout_upper_bound":3168,
        "total_full_rollout_upper_bound":22752,"new_training_runs":0,"new_llm_calls":0,
        "frozen_ppo_models":old["frozen_ppo_models"],
        "source_input_sha256":{str(path.resolve()):digest(path) for path in sorted(set(files))},
    }
    destination=args.output/"development_manifest.json"
    if destination.exists():
        assert json.loads(destination.read_text(encoding="utf-8"))==body,"frozen pilot contract changed"
    else:
        if any((args.output/"states").glob("*.npz")):
            raise RuntimeError("partial states exist without a frozen contract")
        json_write(destination,body)
        snapshot=args.output/"source_snapshot"
        snapshot.mkdir(exist_ok=True)
        shutil.copy2(__file__,snapshot/Path(__file__).name)
    return body


def run(args):
    torch.set_num_threads(1)
    args.source=args.source.resolve()
    args.output=args.output.resolve()
    if args.source==args.output or args.source in args.output.parents:
        raise ValueError("pilot outputs must not overwrite the original experiment")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"states").mkdir(exist_ok=True)
    old=json.loads((args.source/"development_manifest.json").read_text(encoding="utf-8"))
    scenarios=make_domain_randomized_trust_bank(24,20261802,"PTDEV")
    manifest=freeze(args,scenarios,old)
    candidates=list(PLANS)
    for record in manifest["frozen_ppo_models"]:
        assert digest(record["path"])==record["checkpoint_sha256"]
        model=PPO.load(record["path"],device="cpu")
        assert policy_parameter_sha256(model)==record["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(record["training_seed"],model))
    assert [p.name for p in candidates]==manifest["candidate_names"]
    actual=make_env(scenarios)
    worlds=[make_env(scenarios),make_env(scenarios)]
    started=time.monotonic()
    new_planning=new_evaluation=0
    try:
        for j,scenario in enumerate(scenarios):
            hypotheses=source_world_scenarios(scenario)
            for rep in range(2):
                index=2*j+rep
                population_seed=280_000_000+index
                planning=tuple(300_000_000+1000*index+f for f in range(12))
                evaluation=tuple(320_000_000+1000*index+f for f in range(6))
                state_id=f"{scenario.scenario_id}_{rep:02d}_000"
                path=args.output/"states"/f"{state_id}.npz"
                observation,_=actual.reset(seed=population_seed,options={"scenario":scenario})
                support,errors,mismatch=compatible_source_worlds(hypotheses,observation,0)
                assert support[scenario.source_branch] and not mismatch
                if not path.exists():
                    risks=np.full((2,17,12,len(OUTCOME_FIELDS)),np.nan)
                    for source,hypothesis in enumerate(hypotheses):
                        if not support[source]:
                            continue
                        world_observation,_=worlds[source].reset(seed=population_seed,options={"scenario":hypothesis})
                        for p,candidate in enumerate(candidates):
                            for f,seed in enumerate(planning):
                                risks[source,p,f]=vector(evaluate_continuation(worlds[source],world_observation,candidate,seed))
                                new_planning+=1
                    chosen=[choice(risks,support,scenario,mode,budget) for _,mode,budget in RULES]+list(range(12,17))
                    results=np.empty((11,6,len(OUTCOME_FIELDS)))
                    evaluated,traces={},{}
                    for a,p in enumerate(chosen):
                        for f,seed in enumerate(evaluation):
                            if (p,f) not in evaluated:
                                trace=[]
                                evaluated[(p,f)]=vector(evaluate_continuation(actual,observation,candidates[p],seed,trace))
                                traces[(p,f)]=np.asarray(trace,np.float64)
                                new_evaluation+=1
                            results[a,f]=evaluated[(p,f)]
                    keys=sorted(evaluated)
                    temporary=path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary,observation=observation,supported=support,world_errors=errors,
                        population_seed=np.asarray(population_seed),planning_seeds=np.asarray(planning),
                        evaluation_seeds=np.asarray(evaluation),risks=risks,chosen_indices=np.asarray(chosen),
                        evaluation_outcomes=results,planning_rollouts=np.asarray(int(support.sum())*17*12),
                        evaluation_rollouts=np.asarray(len(evaluated)),trace_keys=np.asarray(keys),
                        trace_offsets=np.cumsum([0]+[len(traces[k]) for k in keys]),
                        execution_trace=np.concatenate([traces[k] for k in keys]))
                    temporary.replace(path)
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["observation"],observation)
                    np.testing.assert_array_equal(z["supported"],support)
                    np.testing.assert_array_equal(z["planning_seeds"],planning)
                    np.testing.assert_array_equal(z["evaluation_seeds"],evaluation)
                    assert int(z["population_seed"])==population_seed
            progress={"families_completed":j+1,"families_total":24,"elapsed_seconds":time.monotonic()-started,
                      "new_planning_rollouts_this_invocation":new_planning,"new_evaluation_rollouts_this_invocation":new_evaluation}
            json_write(args.output/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    finally:
        actual.close()
        for world in worlds:
            world.close()
    for candidate,record in zip(candidates[12:],manifest["frozen_ppo_models"]):
        assert policy_parameter_sha256(candidate.model)==record["parameter_sha256"]
    analyze(args,manifest,scenarios,new_planning,new_evaluation,time.monotonic()-started)


def analyze(args,manifest,scenarios,new_planning,new_evaluation,elapsed):
    episodes,selections=[],[]
    planning_count=evaluation_count=trace_count=neutral_count=blocked_count=duplicate_count=0
    supported_counts=Counter()
    for j,scenario in enumerate(scenarios):
        for rep in range(2):
            state_id=f"{scenario.scenario_id}_{rep:02d}_000"
            with np.load(args.output/"states"/f"{state_id}.npz",allow_pickle=False) as z:
                risk,support,chosen,outcomes=z["risks"],z["supported"],z["chosen_indices"],z["evaluation_outcomes"]
                assert support[scenario.source_branch]
                assert set(map(int,z["planning_seeds"])).isdisjoint(map(int,z["evaluation_seeds"]))
                supported_counts[int(support.sum())]+=1
                common={"state_id":state_id,"source_id":scenario.scenario_id,"conflict_type":scenario.conflict_type,
                        "repetition":rep,"population_seed":int(z["population_seed"]),"step":0,
                        "supported_world_count":int(support.sum()),
                        "forecast_wrong":scenario.forecast_source_branch!=scenario.source_branch}
                selection={**common}
                for a,arm in enumerate(ARMS):
                    p=int(chosen[a])
                    selection[arm]=manifest["candidate_names"][p]
                    if a<len(RULES):
                        _,mode,budget=RULES[a]
                        assert choice(risk,support,scenario,mode,budget)==p
                    else:
                        assert p==12+a-len(RULES)
                    for f,seed in enumerate(z["evaluation_seeds"]):
                        outcome=outcomes[a,f]
                        assert abs(outcome[0]+outcome[2]+outcome[3]-1)<=EPS
                        episodes.append({**common,"arm":arm,"selected_plan":manifest["candidate_names"][p],
                            "future_seed":int(seed),**dict(zip(OUTCOME_FIELDS,map(float,outcome)))})
                    for previous in range(a):
                        if chosen[previous]==p:
                            np.testing.assert_array_equal(outcomes[previous],outcomes[a])
                            duplicate_count+=1
                for k,(p,f) in enumerate(z["trace_keys"]):
                    trace=z["execution_trace"][z["trace_offsets"][k]:z["trace_offsets"][k+1]]
                    arm=int(np.flatnonzero(chosen==p)[0])
                    outcome=outcomes[arm,f]
                    assert len(trace)==int(outcome[5])
                    assert trace[0,1]==outcome[9] and trace[0,2]==outcome[10]
                    assert np.all((trace[:,2]==8)==(trace[:,4]==0))
                    assert np.all(trace[trace[:,2]==-1,4]>0)
                trace_count+=len(z["execution_trace"])
                neutral_count+=int((z["execution_trace"][:,2]==-1).sum())
                blocked_count+=int(z["execution_trace"][:,5].sum())
                planning_count+=int(z["planning_rollouts"])
                evaluation_count+=int(z["evaluation_rollouts"])
                selections.append(selection)
    assert len(episodes)==24*2*11*6
    assert planning_count<=19584 and evaluation_count<=3168
    write_csv(args.output/"evaluation_episode_results.csv",episodes)
    write_csv(args.output/"state_selection.csv",selections)
    by_arm={arm:[r for r in episodes if r["arm"]==arm] for arm in ARMS}
    summary_rows=[]
    for arm,rows in by_arm.items():
        summary_rows.append({"arm":arm,"families":24,"records":len(rows),
            **{field:float(np.mean(list(family_average(rows,field).values()))) for field in OUTCOME_FIELDS[:9]}})
    pairings=[(arm,BASE) for arm,_,_ in RULES]
    pairings += [("mean12","worst12"),("worst12","worst3"),("mean12","mean3"),("mean3","worst3")]
    pairs,family_rows=[],[]
    lookup={(r["state_id"],r["arm"],r["future_seed"]):r for r in episodes}
    for treatment,control in pairings:
        for metric in ("final_incapacitation","final_fed"):
            difference_rows=[{**r,"difference":float(r[metric]-lookup[(r["state_id"],control,r["future_seed"])][metric])}
                             for r in by_arm[treatment]]
            deltas=family_average(difference_rows,"difference")
            scale=100 if metric=="final_incapacitation" else 1
            values=np.asarray(list(deltas.values()))*scale
            low,high=interval(values,20261803)
            pairs.append({"treatment":treatment,"control":control,"metric":metric,"families":24,
                "delta":float(values.mean()),"exploratory_ci_low":low,"exploratory_ci_high":high,
                "unit":"percentage_points" if scale==100 else "FED",
                "improved_families":sum(v<-EPS for v in values),"worsened_families":sum(v>EPS for v in values),
                "tied_families":sum(abs(v)<=EPS for v in values)})
            family_rows.extend({"source_id":family,"treatment":treatment,"control":control,"metric":metric,
                                "delta":value*scale} for family,value in deltas.items())
    interaction_values=[]
    for family in sorted({r["source_id"] for r in episodes}):
        average=lambda arm:np.mean([r["final_incapacitation"] for r in by_arm[arm] if r["source_id"]==family])
        interaction_values.append((average("mean12")-average("worst12")-average("mean3")+average("worst3"))*100)
    low,high=interval(interaction_values,20261803)
    interaction={"contrast":"(mean12-worst12)-(mean3-worst3)","delta_pp":float(np.mean(interaction_values)),
                 "exploratory_ci_low_pp":low,"exploratory_ci_high_pp":high}
    conflict=[]
    for kind in sorted({s.conflict_type for s in scenarios}):
        for arm in ARMS:
            values=[r for r in by_arm[arm] if r["conflict_type"]==kind]
            conflict.append({"conflict_type":kind,"arm":arm,"families":4,
                "final_incapacitation":float(np.mean(list(family_average(values,"final_incapacitation").values()))),
                "final_fed":float(np.mean(list(family_average(values,"final_fed").values())))})
    write_csv(args.output/"arm_summary.csv",summary_rows)
    write_csv(args.output/"paired_effects.csv",pairs)
    write_csv(args.output/"family_paired_deltas.csv",family_rows)
    write_csv(args.output/"conflict_summary.csv",conflict)
    changed=[path for path,value in manifest["source_input_sha256"].items() if digest(path)!=value]
    assert not changed
    integrity={"all_checks_passed":True,"input_files_hashed_and_unchanged":len(manifest["source_input_sha256"]),
        "states_checked":48,"all_teacher_selections_rebuilt":True,"population_conservation_checked":True,
        "nested_planning_futures_verified":True,"planning_evaluation_population_sensor_streams_separated":True,
        "scenario_parameters_disjoint_from_source_and_confirmatory_banks":True,
        "duplicate_plan_equalities_verified":duplicate_count,"actual_trace_steps_verified":trace_count,
        "neutral_positive_release_steps":neutral_count,"hold_blocked_steps":blocked_count,
        "frozen_ppo_parameters_and_files_unchanged":True,"changed_original_inputs":[]}
    json_write(args.output/"integrity_audit.json",integrity)
    result={"role":manifest["role"],"freeze_allowed":False,"families":24,"initial_states":48,
        "planning_rollouts":planning_count,"unique_evaluation_rollouts":evaluation_count,
        "total_unique_full_rollouts":planning_count+evaluation_count,
        "logical_execution_records":len(episodes),"actual_execution_trace_steps":trace_count,
        "new_planning_rollouts_this_invocation":new_planning,"new_evaluation_rollouts_this_invocation":new_evaluation,
        "new_training_runs":0,"new_llm_calls":0,"elapsed_seconds_this_invocation":elapsed,
        "supported_world_counts":dict(supported_counts),"arm_summary":summary_rows,"paired_effects":pairs,
        "factorial_interaction":interaction,"initial_plan_counts":{arm:dict(Counter(row[arm] for row in selections)) for arm in ARMS},
        "limitations":["new same-generator development pilot, not final or realistic fire verification",
            "budget-limited 24 families not a power-backed confirmatory sample",
            "unadjusted exploratory bootstrap conditional on five frozen models",
            "equal-world weights are comparison weights, not calibrated source probabilities",
            "oracle privileged and limited to the seventeen-candidate library",
            "initial once-selected closed-loop continuation, not continuously reselected teacher",
            "synthetic hazards and synthetic calibrated behavior, no map/behavior/noise shift test",
            "larger planning budget uses more simulation; aggregation comparisons share candidate evidence"]}
    json_write(args.output/"summary.json",result)
    lines=["# 教师世界聚合×规划预算全新开发验证报告","",
        "日期：2026年10月3日。方案和输入在新成绩生成前冻结。本轮是24家庭的小型开发验证，不是第三轮最终评估。","",
        "## 1. 实验与成本","",
        "17候选、五纯PPO原权重，比较最坏世界/等权平均与3/12规划未来；前三未来嵌套。每家庭两人口，每起点六个独立检验未来。单世界和真实火源oracle为补充参照，五固定PPO全部保留。",
        f"实际规划{planning_count:,}回合、唯一检验{evaluation_count:,}回合，合计{planning_count+evaluation_count:,}个完整回合；逻辑执行记录{len(episodes):,}条。同计划成绩共享，逻辑记录不是独立模拟。没有学生/PPO训练或LLM调用。","",
        "## 2. 同银行结果","",
        "| 方法 | 失能率 | FED |","|---|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['arm']} | {row['final_incapacitation']:.4%} | {row['final_fed']:.6f} |")
    lines += ["","## 3. 失能配对差","",
        "| 比较 | 差（百分点） | 探索性95%家庭区间 | 改善/变差/持平家庭 |","|---|---:|---:|---:|"]
    for row in pairs:
        if row["metric"]=="final_incapacitation":
            lines.append(f"| {row['treatment']}−{row['control']} | {row['delta']:+.5f} | [{row['exploratory_ci_low']:+.5f},{row['exploratory_ci_high']:+.5f}] | {row['improved_families']}/{row['worsened_families']}/{row['tied_families']} |")
    lines += ["",f"交互项{interaction['contrast']}＝{interaction['delta_pp']:+.5f}个百分点，区间[{low:+.5f},{high:+.5f}]。",
        "各家庭先平均两人口与六检验未来，再家庭等权。区间按24家庭Bootstrap 2,000次，种子20261803；未校正多重比较。六类各4家庭仅作描述，小型试验不能把跨零解释成等效或永久无效。","",
        "## 4. 分冲突描述","",
        "| 冲突 | worst3 | worst12 | mean3 | mean12 | 固定601 |","|---|---:|---:|---:|---:|---:|"]
    for kind in sorted({s.conflict_type for s in scenarios}):
        values={row["arm"]:row["final_incapacitation"] for row in conflict if row["conflict_type"]==kind}
        lines.append("| "+kind+" | "+" | ".join(f"{values[arm]:.4%}" for arm in ("worst3","worst12","mean3","mean12",BASE))+" |")
    lines += ["","## 5. 解释边界与推进条件","",
        "规划预算和聚合方式的贡献由本轮四组配对差判断，不把旧PRAU数值拼成改进趋势。等权平均没有给真实世界概率认证；最坏世界也没有给现实风险界。教师仍复制隐藏人群状态，资源多于一个固定PPO，结果不能归因于LLM推理或同成本RL创新。",
        "保留全部四组和五PPO。某个组在新开发数据有优势也不直接冻结为最终主方法；若世界规则或预算未建立稳定差异，则记录不确定，不继续用同银行挑阈值。硬护栏门槛不变，正式第三轮和新最终银行保持未启动。","",
        "## 6. 完整性与文件","",
        f"重新构建48起点所有教师选择，核对{trace_count:,}实际执行步及人口守恒；原五PPO文件和参数、全部{integrity['input_files_hashed_and_unchanged']}份输入哈希未改变。中性正放行−1共{neutral_count}步，保持约束阻挡{blocked_count}步。",
        f"[完整汇总]({(args.output/'summary.json').as_posix()})；[配对效应]({(args.output/'paired_effects.csv').as_posix()})；"
        f"[分冲突结果]({(args.output/'conflict_summary.csv').as_posix()})；[冻结清单]({(args.output/'development_manifest.json').as_posix()})；"
        f"[完整性核查]({(args.output/'integrity_audit.json').as_posix()})。",
        "复现使用scripts/run_teacher_factorial_pilot.py；已有缓存续跑必须匹配冻结方案、代码和输入哈希，结果不会覆盖前轮实验。",
    ]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    json_write(args.output/"analysis_manifest.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "development_manifest_sha256":digest(args.output/"development_manifest.json"),
        "code_sha256":digest(__file__),"report_sha256":digest(args.report),"summary_sha256":digest(args.output/"summary.json"),
        "state_cache_sha256":{p.name:digest(p) for p in sorted((args.output/"states").glob("*.npz"))}})
    print(json.dumps({"completed":True,"families":24,"total_rollouts":planning_count+evaluation_count,
                      "paired_incap_effects":[r for r in pairs if r["metric"]=="final_incapacitation"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/baseline_relative_plan_distillation_development_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/teacher_factorial_pilot_development_v1")
    parser.add_argument("--report",type=Path,default=ROOT/"reports/教师世界聚合与规划预算全新开发验证报告.md")
    run(parser.parse_args())
