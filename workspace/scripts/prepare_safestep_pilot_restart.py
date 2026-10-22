"""Preserve a graceful checkpoint and resume in a copied directory, same physics."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/safestep_physics_time_contract_20261016"
PREFIX="hotel_literature_pilot120_v2"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parent=OUT/"runs"/PREFIX
    result=json.loads((parent/"run_result.json").read_text("utf-8"))
    assert result["returncode"]==0 and not result["timed_out"] and "STOP: FDS stopped by user" in (parent/"stderr.log").read_text("utf-8")
    restarts=list(parent.glob("*.restart"));assert len(restarts)==6
    dest=OUT/"runs/hotel_literature_pilot120_resume";dest.mkdir(exist_ok=False)
    copied={}
    for p in parent.iterdir():
        if p.is_file() and p.name.startswith(PREFIX) and not p.name.endswith(".stop"):
            shutil.copyfile(p,dest/p.name);copied[p.name]=sha(p)
    text=(parent/(PREFIX+".fds")).read_text("utf-8")
    resumed=text.replace("&MISC TMPA=20.0", "&MISC RESTART=.TRUE., TMPA=20.0")
    assert resumed!=text
    (dest/"hotel_literature_pilot120_resume.fds").write_text(resumed,"utf-8")
    record={"created_at":datetime.now(timezone.utc).isoformat(),"parent_case":PREFIX,
            "continuation_case":dest.name,"CHID_unchanged":PREFIX,"native_restart_not_independent_run":True,
            "input_change_only":"MISC RESTART=.TRUE.","physical_parameters_changed":False,
            "parent_checkpoint_wall_s":result["elapsed_s"],"restart_files":{p.name:sha(p) for p in restarts},
            "copied_parent_files_sha256":copied,"parent_output_files_overwritten":False,
            "reason":"Native gracefulcheckpoint115.49119s beforeperprocess1200swallclocklimit; continue remaining4.51s withoutchangingphysics",
            "guide_reference":"FDS6.11.1 UserGuide section5.6"}
    (dest/"derivation.json").write_text(json.dumps(record,indent=2),"utf-8")
    (OUT/"restart_continuation_provenance.json").write_text(json.dumps(record,indent=2),"utf-8")
    print(json.dumps({"restart_files":len(restarts),"copied_files":len(copied),"new_case":dest.name}))


if __name__=="__main__":main()
