"""Final bounded mechanism check: sealed pressure zones versus a static opening."""
from datetime import datetime,timezone
import json
import shutil
import numpy as np
import pandas as pd
from diagnose_safestep_dynamic_door_restart import TEMPLATE,TWO_MESHES,BARRIER,run
from analyze_safestep_fds_smokecheck import parse_smv
from prepare_safestep_clean_hotel120 import ROOT,OUT,sha,save
from analyze_safestep_restart_hrr_audit import checkpoint_metadata


def main():
    base=OUT/"sealed_zone_diagnostic";base.mkdir(exist_ok=False)
    save(base/"frozen_followup_plan.json",{"frozen_at":datetime.now(timezone.utc).isoformat(),
        "reason":"Hotel has12 pressure zones; previous small fixtures had no pressure-zone output. Hotel zones1/12 diverge after restart",
        "contrasts":["sealed_dynamic_door","sealed_static_open"],"planned_runs":4,
        "purpose":"Test loss of dynamic pressure-zone coupling without changing solver or restart bytes",
        "keep_original_hotel_gate_failure":True,"no_further_simulation_expansion_in_this_stage":True})
    results={};wall=0.0
    for name,barrier in [("sealed_dynamic_door",BARRIER),("sealed_static_open","")]:
        text=TEMPLATE.format(prefix=name,meshes=TWO_MESHES,barrier=barrier,restart="")
        text="\n".join(line for line in text.splitlines() if not (line.startswith('&VENT XB=0.0,0.0') or line.startswith('&VENT XB=4.0,4.0')))+"\n"
        parent=base/(name+"_continuous");parent.mkdir()
        (parent/(parent.name+".fds")).write_text(text,"utf-8");run(parent,name)
        meta=[checkpoint_metadata(parent/f"{name}_{i}.restart",(10,8,10),.02) for i in (1,2)]
        assert len({m['fire_time_s'] for m in meta})==1
        resumed=base/(name+"_resumed");resumed.mkdir();copied={}
        for p in parent.iterdir():
            if p.is_file() and p.name.startswith(name) and p.suffix!='.stop':
                shutil.copyfile(p,resumed/p.name);copied[p.name]=sha(p)
        (resumed/(resumed.name+'.fds')).write_text(text.replace('&TAIL /','&MISC RESTART=.TRUE. /\n&TAIL /'),'utf-8')
        save(resumed/'derivation.json',{'parent':parent.relative_to(ROOT).as_posix(),'copied_sha256':copied,
            'only_input_change':'RESTART=.TRUE.','checkpoint_bytes_modified':False})
        run(resumed,name)
        a=pd.read_csv(parent/f'{name}_devc.csv',skiprows=1);b=pd.read_csv(resumed/f'{name}_devc.csv',skiprows=1)
        cp=meta[0]['fire_time_s'];grid=np.arange(np.ceil((cp+1e-6)*50)/50,6.5+1e-8,.02)
        cols=list(a.columns[1:]);ref=np.column_stack([np.interp(grid,a.Time,a[c]) for c in cols]);res=np.column_stack([np.interp(grid,b.Time,b[c]) for c in cols])
        metrics={c:{'max_abs_difference':float(np.max(np.abs(ref[:,i]-res[:,i]))),'reference_final':float(ref[-1,i]),'resumed_final':float(res[-1,i])} for i,c in enumerate(cols)}
        ha=pd.read_csv(parent/f'{name}_hrr.csv',skiprows=1);hb=pd.read_csv(resumed/f'{name}_hrr.csv',skiprows=1)
        zones=[c for c in ha.columns if c.startswith('ZONE_')]
        pressures={c:{'continuous_final_Pa':float(ha.iloc[-1][c]),'resumed_final_Pa':float(hb.iloc[-1][c])} for c in zones}
        for d in (parent,resumed):wall+=json.loads((d/'run_result.json').read_text('utf-8'))['elapsed_s']
        results[name]={'checkpoint_s':cp,'metrics':metrics,'pressure_zones':pressures,'checkpoint_metadata':meta}
        print(json.dumps({'contrast':name,'metrics':metrics,'pressures':pressures}),flush=True)
    save(base/'diagnostic_results.json',{'status':'FOUR_SEALED_ZONE_DIAGNOSTICS_COMPLETED','results':results,
        'solver_runs':4,'total_solver_wall_s':wall,'primary_hotel_failure_reclassified':False})


if __name__=='__main__':main()
