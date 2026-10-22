"""Check thin interface doors and exact event/checkpoint coincidence in fixtures."""
from datetime import datetime, timezone
import json
import shutil

import numpy as np
import pandas as pd

from diagnose_safestep_dynamic_door_restart import TEMPLATE, TWO_MESHES, BARRIER, run
from prepare_safestep_clean_hotel120 import ROOT, OUT, sha, save
from analyze_safestep_restart_hrr_audit import checkpoint_metadata


def main():
    base=OUT/"door_boundary_diagnostic";base.mkdir(exist_ok=False)
    save(base/"frozen_followup_plan.json",{"frozen_at":datetime.now(timezone.utc).isoformat(),
        "reason":"First six small runs did not reproduce hotel drift; hotel checkpoint coincides with door removal and its fire-side door is zero-thickness",
        "contrasts":["thin_interface_late_checkpoint","thick_interface_event_checkpoint","thin_interface_event_checkpoint"],
        "planned_runs":6,"hotel_screen_unchanged":True,"solver_and_checkpoint_bytes_unchanged":True,
        "purpose":"Mechanism diagnosis only; small successes never override hotel failures"})
    results={};wall=0.0
    for name,thin,event in [("thin_interface_late_checkpoint",True,False),("thick_interface_event_checkpoint",False,True),("thin_interface_event_checkpoint",True,True)]:
        barrier=BARRIER.replace("XB=1.8,2.2","XB=2.0,2.2") if thin else BARRIER
        text=TEMPLATE.format(prefix=name,meshes=TWO_MESHES,barrier=barrier,restart="")
        if event:text=text.replace("SETPOINT=3.0","SETPOINT=2.98").replace("DT_RESTART=4.0","DT_RESTART=3.0")
        # Keep just one checkpoint generation: event cases end before second 6s save.
        end=5.8 if event else 6.5
        text=text.replace("T_END=6.5",f"T_END={end}")
        parent=base/(name+"_continuous");parent.mkdir()
        (parent/(parent.name+".fds")).write_text(text,"utf-8");run(parent,name)
        meta=[checkpoint_metadata(parent/f"{name}_{i}.restart",(10,8,10),.02) for i in (1,2)]
        assert len({m["fire_time_s"] for m in meta})==1
        resumed=base/(name+"_resumed");resumed.mkdir();copied={}
        for p in parent.iterdir():
            if p.is_file() and p.name.startswith(name) and p.suffix!=".stop":
                shutil.copyfile(p,resumed/p.name);copied[p.name]=sha(p)
        (resumed/(resumed.name+".fds")).write_text(text.replace("&TAIL /","&MISC RESTART=.TRUE. /\n&TAIL /"),"utf-8")
        save(resumed/"derivation.json",{"parent":parent.relative_to(ROOT).as_posix(),"copied_sha256":copied,
            "only_input_change":"RESTART=.TRUE.","checkpoint_bytes_modified":False})
        run(resumed,name)
        a=pd.read_csv(parent/f"{name}_devc.csv",skiprows=1);b=pd.read_csv(resumed/f"{name}_devc.csv",skiprows=1)
        cp=meta[0]["fire_time_s"];grid=np.arange(np.ceil((cp+1e-6)*50)/50,end+1e-8,.02)
        cols=list(a.columns[1:]);ref=np.column_stack([np.interp(grid,a.Time,a[c]) for c in cols]);res=np.column_stack([np.interp(grid,b.Time,b[c]) for c in cols])
        metrics={c:{"max_abs_difference":float(np.max(np.abs(res[:,i]-ref[:,i]))),"reference_max_abs":float(np.max(np.abs(ref[:,i]))),
            "reference_final":float(ref[-1,i]),"resumed_final":float(res[-1,i])} for i,c in enumerate(cols)}
        for d in (parent,resumed):wall+=json.loads((d/"run_result.json").read_text("utf-8"))["elapsed_s"]
        results[name]={"checkpoint_time_s":cp,"metrics":metrics,"checkpoint_metadata":meta,"thin_fire_side_interface_door":thin,"requested_event_coincidence":event}
        print(json.dumps({"contrast":name,"checkpoint_s":cp,"metrics":metrics}),flush=True)
    save(base/"diagnostic_results.json",{"status":"SIX_BOUNDARY_DIAGNOSTIC_RUNS_COMPLETED","results":results,
        "solver_runs":6,"total_solver_wall_s":wall,"primary_hotel_failure_reclassified":False})


if __name__=="__main__":main()
