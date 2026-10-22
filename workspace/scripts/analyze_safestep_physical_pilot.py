"""Analyze a completed120s source-bounded reconstruction without route FED claims."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
from safestep_gas_to_fer import read_fds_gas_csv
from analyze_safestep_fds_smokecheck import parse_smv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_physics_time_contract_20261016"
CASE = OUT/"runs/hotel_literature_pilot120_resume"
PREFIX = "hotel_literature_pilot120_v2"
CHECKS = []


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def save(path, value):
    assert not path.exists(), path
    path.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2),"utf-8")


def csv_save(path, rows):
    assert not path.exists(), path
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def check(name, passed, detail=None):
    assert passed, (name,detail)
    CHECKS.append({"name":name,"passed":True,"detail":detail})


def numeric_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.reader(f))
    return [s.strip() for s in rows[0]], [s.strip() for s in rows[1]], np.array([[float(v) for v in r] for r in rows[2:]])


def render(data, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t=np.array(data["times"])
    with plt.rc_context({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10}):
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
        axes[0,0].plot(data["hrr_times"],data["hrr"],label="FDS输出HRR",color="#0072B2")
        axes[0,0].plot(data["hrr_times"],5000*(np.array(data["hrr_times"])/326)**2,"--",label="规定t²曲线",color="#D55E00")
        axes[0,0].set(ylabel="热释放率（kW）",title="实际燃烧输出与规定曲线")
        axes[0,0].legend()
        axes[0,1].plot(t,data["room_T"],color="#D55E00",label="火源房内测点")
        axes[0,1].axhline(150,color="#6C6C6C",ls=":",label="150℃（仅供参照）")
        axes[0,1].set(ylabel="气温（℃）",title="Room24测点气温；不是窗玻璃温度")
        axes[0,1].legend()
        for ax,gas in ((axes[1,0],"CO"),(axes[1,1],"HCN")):
            ax.plot(t,data["room_"+gas],label="Room24测点",color="#0072B2")
            ax.plot(t,data["corridor_max_"+gas],"--",label="58区域逐时最大值",color="#009E73")
            ax.set(ylabel=gas+"（ppm，湿基）",title=gas+"早期传播：同一探索设定")
            ax.legend()
        for ax in axes.flat:
            ax.axvline(100,color="#6D6D6D",ls=":",lw=.9)
            ax.axvline(115.49119,color="#8A4F78",ls="-.",lw=.9,label="原生续算点" if ax is axes[0,0] else None)
            ax.set(xlabel="火灾时间（秒）",xlim=(0,120))
            ax.grid(alpha=.15)
        axes[0,0].legend()
        fig.suptitle("120秒探索性火灾：HCN产率跨材料迁移，尚非原论文物理复现",fontsize=13)
        fig.savefig(OUT/"pilot120_native_observations_v2.png",dpi=180)
        fig.savefig(OUT/"pilot120_native_observations_v2.svg")
        plt.close(fig)


def main():
    assert not (OUT/"pilot_analysis.json").exists(),"Do not overwrite completed analysis"
    result=load(CASE/"run_result.json")
    check("solver_completed",result["passed"] and not result["timed_out"])
    check("native_time_analytic_tests_passed",load(OUT/"native_time_verification.json")["status"].startswith("PASS"))
    sources=load(OUT/"source_manifest.json")
    for name,expected in {**sources["protected_previous_sha256"],**sources["input_sha256"]}.items():
        check("source_unchanged:"+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected)
    units,headers,raw=numeric_csv(CASE/(PREFIX+"_devc.csv"))
    times,samples=read_fds_gas_csv(CASE/(PREFIX+"_devc.csv"))
    restart=load(OUT/"restart_continuation_provenance.json")
    parent=OUT/"runs"/restart["parent_case"]
    for name,expected in restart["copied_parent_files_sha256"].items():
        check("checkpoint_parent_preserved:"+name,hashlib.sha256((parent/name).read_bytes()).hexdigest()==expected)
    _,parent_headers,parent_raw=numeric_csv(parent/(PREFIX+"_devc.csv"))
    check("restart_headers_preserved",headers==parent_headers)
    resumed_by_time={float(row[0]):row for row in raw}
    matched=0
    for row in parent_raw:
        if float(row[0]) in resumed_by_time:
            check("pre_checkpoint_native_data_exactly_preserved",np.array_equal(row,resumed_by_time[float(row[0])]))
            matched+=1
    check("all_pre_checkpoint_regular_outputs_retained",matched>=len(parent_raw)-1,{"parent_rows":len(parent_raw),"exact_rows_retained":matched})
    save(OUT/"restart_continuity_audit.json",{"parent_hashes_unchanged":len(restart["copied_parent_files_sha256"]),
         "parent_native_rows":len(parent_raw),"exactly_matching_native_rows_retained":matched,
         "note":"FDSmayclipa forcedstopendpoint;allregularprecheckpointoutputs retained;no manual sorting/deduplication/stitching"})
    t=np.array(times);check("native_coverage0to120s",times[0]==0 and abs(times[-1]-120)<1e-6)
    max_gap=float(np.max(np.diff(t)))
    check("native_gaps_within_frozen1s_bound",max_gap<=1.0)
    check("all174_gas_output_columns",sum(h.startswith("SS_R") for h in headers)==174)
    rows=[]
    for time,sample in zip(times,samples):
        for r in range(1,59):
            g={gas:sample[(r,gas)] for gas in ("CO","CO2","HCN")}
            check("valid_gas_volume_fractions",all(math.isfinite(v) and 0<=v<=1 for v in g.values()) and sum(g.values())<=1+1e-12)
            rows.append({"region":f"R{r}","fire_time_s":time,"evacuation_elapsed_s":time-300,
                         "CO_ppm":g["CO"]*1e6,"CO2_percent":g["CO2"]*100,"HCN_ppm":g["HCN"]*1e6,
                         "evidence_label":"EXPLORATORY_RECONSTRUCTION_LITERATURE_TRANSFER_NOT_AUTHOR_ORIGINAL"})
    csv_save(OUT/"pilot120_native_gas_long.csv",rows)
    region_rows=[]
    for r in range(1,59):
        co=np.array([sample[(r,"CO")]*1e6 for sample in samples]);hcn=np.array([sample[(r,"HCN")]*1e6 for sample in samples])
        co_hits=np.flatnonzero(co>1);hcn_hits=np.flatnonzero(hcn>1)
        region_rows.append({"region":f"R{r}","CO_peak_ppm":float(co.max()),"CO_peak_time_s":times[int(co.argmax())],
                            "HCN_peak_ppm":float(hcn.max()),"HCN_peak_time_s":times[int(hcn.argmax())],
                            "first_CO_above1ppm_s":times[int(co_hits[0])] if len(co_hits) else None,
                            "first_HCN_above1ppm_s":times[int(hcn_hits[0])] if len(hcn_hits) else None,
                            "threshold_note":"1ppm is an arbitrary reporting threshold, not a safety or measurement-detection limit"})
    csv_save(OUT/"pilot120_region_gas_summary.csv",region_rows)
    room={}
    for gas in ("CO","CO2","HCN","T"):
        i=headers.index("ROOM24_"+gas)
        check("room_probe_unit:"+gas,units[i]==("C" if gas=="T" else "mol/mol"))
        room[gas]=raw[:,i]*(1 if gas=="T" else 100 if gas=="CO2" else 1e6)
    logical=[]
    with (CASE/(PREFIX+"_devc_ctrl_log.csv")).open() as f:
        for row in list(csv.reader(f))[1:]:
            logical.append({"time_s":float(row[0]),"ID":row[2].strip(),"state":row[3].strip(),"value":float(row[4]),"units":row[5].strip()})
    _,events=parse_smv(CASE/(PREFIX+".smv"))
    door=[e for e in logical if e["ID"]=="SS_DOOR_OPEN"]
    check("100s_door_control_was_reached",len(door)==1 and door[0]["state"]=="F" and 100<=door[0]["time_s"]<=101)
    door_geom=[e for e in events if e["ID"]=="SS_DOOR24"]
    check("door_geometry_removed_in_two_adjacent_meshes",len(door_geom)==2 and all(e["kind"]=="HIDE_OBST" for e in door_geom))
    window=[e for e in logical if e["ID"]=="SS_WINDOW_BREAK"]
    if window:
        check("window_trigger_uses150C_proxy",len(window)==1 and window[0]["state"]=="T" and window[0]["value"]>=150)
    save(OUT/"pilot120_control_events.json",{"logical":logical,"geometry":events})
    _,hheaders,hdata=numeric_csv(CASE/(PREFIX+"_hrr.csv"));hrr=hdata[:,hheaders.index("HRR")]
    check("actual_HRR_nonzero",np.max(hrr)>0)
    solver_out=(CASE/(PREFIX+".out")).read_text("utf-8",errors="replace")
    heat=re.search(r"REAC_FUEL\s+([\d.]+)\s*\n\s*EPUMO2:",solver_out)
    check("computed_heat_of_combustion_recorded",heat is not None)
    summary={"status":"COMPLETED_120S_EXPLORATORY_RECONSTRUCTION_NOT_EXACT_PAPER_PHYSICS",
             "created_at":datetime.now(timezone.utc).isoformat(),"solver":load(OUT/"frozen_physics_time_contract.json")["FDS_version"],
             "complete_case":CASE.name,"previous_interrupted_case_preserved":True,
             "elapsed_s":result["elapsed_s"]+load(OUT/"restart_continuation_provenance.json")["parent_checkpoint_wall_s"],
             "native_checkpoint_continuation_used":True,"resume_elapsed_s":result["elapsed_s"],
             "native_time_samples":len(times),"gas_columns":174,"native_region_rows":len(rows),"native_gap_min_s":float(np.min(np.diff(t))),"native_gap_max_s":max_gap,
             "HCN_yield_kg_per_kg":.007,"HCN_status":"Cross-material literature transfer, not recovered original",
             "fuel_heat_of_combustion_kJ_per_kg_from_FDS":float(heat.group(1)),"EPUMO2_kJ_per_kgO2":13100,
             "peak_actual_HRR_kW":float(hrr.max()),"requested_HRR_at120s_kW":5000*(120/326)**2,
             "door_logical_event_time_s":door[0]["time_s"],"door_geometry_event_times_s":[e["geometry_event_time_s"] for e in door_geom],
             "window_proxy_events":window,
             "room24_CO_peak_ppm":float(room["CO"].max()),"room24_HCN_peak_ppm":float(room["HCN"].max()),"room24_T_peak_C":float(room["T"].max()),
             "corridor_CO_peak_ppm":max(r["CO_peak_ppm"] for r in region_rows),"corridor_HCN_peak_ppm":max(r["HCN_peak_ppm"] for r in region_rows),
             "corridor_regions_CO_above1ppm":sum(r["CO_peak_ppm"]>1 for r in region_rows),"corridor_regions_HCN_above1ppm":sum(r["HCN_peak_ppm"]>1 for r in region_rows),
             "warnings":result["warnings"],"route_FED_computed":False,"before_evacuation_start300s_only":True,
             "training_updates":0,"LLM_requests":0,"full900s_runs":0,"old_final48_scenarios_accessed":False,
             "protected_previous_files_unchanged":len(sources["protected_previous_sha256"])}
    save(OUT/"pilot_analysis.json",summary)
    save(OUT/"pilot_verification.json",{"checks":CHECKS,"check_count":len(CHECKS),"status":"PASS_COMPLETED_PILOT_AND_PROVENANCE_NOT_PHYSICAL_ACCURACY_VALIDATION"})
    plot={"times":times,"hrr_times":list(hdata[:,0]),"hrr":list(hrr),"room_T":list(room["T"]),"room_CO":list(room["CO"]),"room_HCN":list(room["HCN"]),
          "corridor_max_CO":[max(sample[(r,"CO")]*1e6 for r in range(1,59)) for sample in samples],
          "corridor_max_HCN":[max(sample[(r,"HCN")]*1e6 for r in range(1,59)) for sample in samples]}
    render(plot,events)
    save(OUT/"figure_data.json",plot)
    print(json.dumps(summary,ensure_ascii=True))


if __name__=="__main__":
    main()
