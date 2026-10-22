"""Small diagnostic contrasts prompted by failed hotel gas/temperature screens."""
from datetime import datetime, timezone
import json
import shutil

import numpy as np
import pandas as pd

from prepare_safestep_clean_hotel120 import ROOT, OUT, sha, save
from run_safestep_restart_microfixture import run
from analyze_safestep_restart_hrr_audit import checkpoint_metadata

TEMPLATE = """! Diagnostic only; not a hotel calibration or author reproduction.
&HEAD CHID='{prefix}', TITLE='Dynamic door restart mechanism fixture' /
&TIME T_END=6.5, DT=0.02, LOCK_TIME_STEP=.TRUE. /
{meshes}
&REAC FUEL='PROPANE', SOOT_YIELD=0.01 /
&SURF ID='BURNER', HRRPUA=200.0 /
&VENT XB=0.8,1.2,0.6,1.0,0.0,0.0, SURF_ID='BURNER' /
&VENT XB=0.0,0.0,0.0,1.6,0.0,2.0, SURF_ID='OPEN' /
&VENT XB=4.0,4.0,0.0,1.6,0.0,2.0, SURF_ID='OPEN' /
&DEVC ID='OPEN', QUANTITY='TIME', XYZ=0.4,0.4,0.4, SETPOINT=3.0, INITIAL_STATE=.TRUE., LATCH=.TRUE. /
{barrier}
&DEVC ID='LEFT_T', QUANTITY='TEMPERATURE', XYZ=1.4,0.8,1.6 /
&DEVC ID='RIGHT_T', QUANTITY='TEMPERATURE', XYZ=2.4,0.8,1.6 /
&DEVC ID='LEFT_CO2', QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON DIOXIDE', XYZ=1.4,0.8,1.6 /
&DEVC ID='RIGHT_CO2', QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON DIOXIDE', XYZ=2.4,0.8,1.6 /
&DEVC ID='INTERFACE_U', QUANTITY='U-VELOCITY', XYZ=2.0,0.8,1.6 /
&DUMP DT_RESTART=4.0, DT_DEVC=0.02, DT_HRR=0.02 /
{restart}
&TAIL /
"""

TWO_MESHES = "&MESH IJK=10,8,10, XB=0.0,2.0,0.0,1.6,0.0,2.0 /\n&MESH IJK=10,8,10, XB=2.0,4.0,0.0,1.6,0.0,2.0 /"
ONE_MESH = "&MESH IJK=20,8,10, XB=0.0,4.0,0.0,1.6,0.0,2.0 /"
BARRIER = "&OBST ID='DOOR', XB=1.8,2.2,0.0,1.6,0.0,2.0, SURF_ID='INERT', DEVC_ID='OPEN' /"


def main():
    base = OUT/"dynamic_door_diagnostic"
    base.mkdir(exist_ok=False)
    save(base/"frozen_followup_plan.json", {"frozen_at":datetime.now(timezone.utc).isoformat(),
         "trigger":"Original hotel passive-restart screen failed for gases and temperature despite similar HRR",
         "primary_hotel_gates_unchanged":True,"alter_hotel_inputs_or_solver":False,
         "contrasts":["two_mesh_dynamic_door","two_mesh_static_open","one_mesh_dynamic_door"],
         "each_contrast":"Independent continuous reference and identical native 4s periodic checkpoint resumed to6.5s",
         "purpose":"Separate dynamic geometry from the mesh-interface effect; hypothesis diagnosis, not a repair",
         "planned_solver_runs":6,"stop_files_created":False,"interpretation":"Descriptive errors; do not transfer hotel screen pass status from fixtures"})
    results = {}; wall=0.0
    for kind, meshes, barrier, ijks in [
        ("two_mesh_dynamic_door",TWO_MESHES,BARRIER,[(10,8,10),(10,8,10)]),
        ("two_mesh_static_open",TWO_MESHES,"",[(10,8,10),(10,8,10)]),
        ("one_mesh_dynamic_door",ONE_MESH,BARRIER,[(20,8,10)]),
    ]:
        prefix=kind; parent=base/(kind+"_continuous");parent.mkdir()
        inp=TEMPLATE.format(prefix=prefix,meshes=meshes,barrier=barrier,restart="")
        (parent/(parent.name+".fds")).write_text(inp,"utf-8")
        run(parent,prefix)
        meta=[checkpoint_metadata(parent/f"{prefix}_{i}.restart",ijk,.02) for i,ijk in enumerate(ijks,1)]
        assert len({x["fire_time_s"] for x in meta})==1
        resumed=base/(kind+"_resumed");resumed.mkdir()
        copied={}
        for p in parent.iterdir():
            if p.is_file() and p.name.startswith(prefix) and p.suffix != ".stop":
                shutil.copyfile(p,resumed/p.name);copied[p.name]=sha(p)
        (resumed/(resumed.name+".fds")).write_text(TEMPLATE.format(prefix=prefix,meshes=meshes,barrier=barrier,restart="&MISC RESTART=.TRUE. /"),"utf-8")
        save(resumed/"derivation.json",{"parent":parent.relative_to(ROOT).as_posix(),"copied_sha256":copied,
            "same_CHID":prefix,"only_input_change":"RESTART=.TRUE.","checkpoint_bytes_modified":False})
        run(resumed,prefix)
        a=pd.read_csv(parent/f"{prefix}_devc.csv",skiprows=1);b=pd.read_csv(resumed/f"{prefix}_devc.csv",skiprows=1)
        cp=meta[0]["fire_time_s"];grid=np.arange(np.ceil((cp+1e-6)*50)/50,6.5+1e-8,.02)
        channels=[c for c in a.columns if c != "Time"]
        ref=np.column_stack([np.interp(grid,a.Time,a[c]) for c in channels])
        res=np.column_stack([np.interp(grid,b.Time,b[c]) for c in channels]);delta=res-ref
        metrics={c:{"max_abs_difference":float(np.max(np.abs(delta[:,i]))),"reference_max_abs":float(np.max(np.abs(ref[:,i]))),
                    "reference_final":float(ref[-1,i]),"resumed_final":float(res[-1,i])} for i,c in enumerate(channels)}
        for directory in (parent,resumed):wall+=json.loads((directory/"run_result.json").read_text("utf-8"))["elapsed_s"]
        results[kind]={"checkpoint_time_s":cp,"channels":metrics,"checkpoint_metadata":meta,
            "grid_start_s":float(grid[0]),"grid_end_s":float(grid[-1]),"grid_points":len(grid)}
        print(json.dumps({"contrast":kind,"metrics":metrics},ensure_ascii=True),flush=True)
    save(base/"diagnostic_results.json",{"status":"SIX_FOLLOWUP_FIXTURE_RUNS_COMPLETED","results":results,
         "total_solver_wall_s":wall,"solver_runs":6,"primary_hotel_failure_reclassified":False,
         "no_solver_or_checkpoint_byte_modification":True,"no_stop_file_used":True})


if __name__=="__main__":main()
