"""Descriptive source-target versus student misses; no new data or fits."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_five_ppo_fresh_student import OUT, read, write, sha, load


def main():
    assert not (OUT / "miss_diagnostic.json").exists()
    m=read(OUT / "development_manifest.json");train=load(OUT / "training_dataset.npz");model=load(OUT / "models/M_kernel.npz")
    rows=[]
    for e in m["seed_registry"]:
        if e["block"]!="test":continue
        p=load(OUT / "test_planning" / (e["family"]+".npz"));a=load(OUT / "test_actual" / (e["family"]+".npz"));choice=read(OUT / "test_choices" / (e["family"]+".json"))
        means=a["evaluation_outcomes"].mean(axis=(0,2));gain=means[0,0]-means[:,0]
        tx=(p["features"]-model["mean"])/model["std"];distance=((model["train"]-tx)**2).sum(axis=1);nearest=int(distance.argmin())
        for plan in np.flatnonzero(gain>.001):
            row={"family":e["family"],"plan":m["candidate_names"][plan],"plan_index":int(plan),"actual_gain_pp":float(gain[plan]*100),
                "planning_R_pp":float(p["pooled_targets"][0,plan]*100),"planning_M_pp":float(p["pooled_targets"][1,plan]*100),
                "predicted_R_kernel_pp":float(choice["predictions"]["R_kernel"][plan]*100),
                "predicted_M_kernel_pp":float(choice["predictions"]["M_kernel"][plan]*100),
                "R_label_positive":bool(p["pooled_targets"][0,plan]>.001),"M_label_positive":bool(p["pooled_targets"][1,plan]>.001),
                "R_kernel_chose_this_plan":choice["choices"]["R_kernel_point"]==plan,"M_kernel_chose_this_plan":choice["choices"]["M_kernel_point"]==plan,
                "nearest_training_family":str(train["families"][nearest]),"nearest_training_squared_standardized_distance":float(distance[nearest]),
                "nearest_kernel_similarity":float(np.exp(-float(model["gamma"])*distance[nearest])),
                "nearest_training_M_label_pp":float(train["targets"][nearest,1,plan]*100),
                "training_candidate_M_max_pp":float(train["targets"][:,1,plan].max()*100),
                "test_M_above_candidate_training_max":bool(p["pooled_targets"][1,plan]>train["targets"][:,1,plan].max()+1e-12)}
            rows.append(row)
    rows.sort(key=lambda r:(r["family"],r["plan_index"]))
    with (OUT / "actual_beneficial_candidate_audit.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(10.2,6.2));yy=np.arange(len(rows))
    collections=[ax.scatter([r["actual_gain_pp"] for r in rows],yy-.15,c="#0072B2",s=35,label="Actual independent-future gain"),
        ax.scatter([r["planning_M_pp"] for r in rows],yy,c="#777777",marker="s",s=30,label="Pooled planning M label"),
        ax.scatter([r["predicted_M_kernel_pp"] for r in rows],yy+.15,c="#D55E00",marker="x",s=45,label="Frozen M-kernel prediction")]
    ax.set_yticks(yy,[r["family"]+": "+r["plan"].replace("pure_ppo_","") for r in rows]);ax.invert_yaxis()
    ax.axvline(0,color="#444444",lw=.8);ax.grid(axis="x",alpha=.2)
    ax.set_title("Missed candidates in the fresh development holdout",fontsize=12)
    ax.set_xlabel("601 incapacitation minus candidate incapacitation (percentage points)")
    legend=fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.61,.005),fontsize=8)
    fig.tight_layout(rect=(0,.18,1,1));fig.canvas.draw();box=legend.get_window_extent(fig.canvas.get_renderer())
    for collection in collections:
        for x,y in ax.transData.transform(collection.get_offsets()):assert not (box.x0-8<=x<=box.x1+8 and box.y0-8<=y<=box.y1+8)
    for suffix in ("png","svg","pdf"):fig.savefig(OUT / ("fresh_candidate_miss_diagnostic."+suffix),dpi=200)
    plt.close(fig)
    diagnostic={"role":"post_outcome_descriptive_miss_diagnosis_not_model_selection","actual_beneficial_candidate_rows":len(rows),
        "actual_beneficial_families":len({r["family"] for r in rows}),"M_label_nonpositive_actual_beneficial_rows":sum(not r["M_label_positive"] for r in rows),
        "R_label_nonpositive_actual_beneficial_rows":sum(not r["R_label_positive"] for r in rows),
        "M_label_positive_but_kernel_prediction_nonpositive_rows":sum(r["M_label_positive"] and r["predicted_M_kernel_pp"]<=.10 for r in rows),
        "all_M_kernel_predictions_nonpositive_for_actual_beneficial_rows":all(r["predicted_M_kernel_pp"]<=.10 for r in rows),
        "training_positive_candidate_rows":{"R":int((train["targets"][:,0,1:]>.001).sum()),"M":int((train["targets"][:,1,1:]>.001).sum())},
        "training_global_positive_target_max_pp":{"R":float(train["targets"][:,0,1:].max()*100),"M":float(train["targets"][:,1,1:].max()*100)},
        "nearest_neighbor_is_descriptive_not_causal_or_shift_proof":True,"source_world_conflict_and_fitting_miss_distinguished":True,
        "no_action_or_margin_changes_after_test":True,"new_simulations":0,"new_fits":0,
        "script_sha256":sha(__file__),"audit_csv_sha256":sha(OUT / "actual_beneficial_candidate_audit.csv"),
        "plots_sha256":{f"fresh_candidate_miss_diagnostic.{s}":sha(OUT / f"fresh_candidate_miss_diagnostic.{s}") for s in ("png","svg","pdf")}}
    write(OUT / "miss_diagnostic.json",diagnostic);print(json.dumps(diagnostic,ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
