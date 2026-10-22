"""Prepare explicit, separate follow-up cases after successful initialization."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"


def write_case(name, text, changes):
    path = OUT/"runs"/name
    path.mkdir(exist_ok=False)
    (path/(name+".fds")).write_text(text, "utf-8")
    (path/"derivation.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), "utf-8")


def micro(name, ambient):
    return f"""! CONTROL DIAGNOSTIC ONLY, not hotel geometry or fire physics.
&HEAD CHID='{name}', TITLE='150C gas trigger and door sign diagnostic only' /
&MESH IJK=20,10,12, XB=0,2,0,1,0,1.2 /
&TIME T_END=1.0, DT=0.01 /
&MISC TMPA={ambient:.1f} /
&DUMP DT_DEVC=0.05, DT_HRR=0.05 /
&OBST ID='WINDOW_WALL', XB=0.8,1.0,0,1,0,1.2, SURF_ID='INERT' /
&HOLE ID='WINDOW_HOLE', XB=0.79,1.01,0.2,0.8,0.2,1.0, DEVC_ID='WINDOW_BREAK' /
&DEVC ID='WINDOW_BREAK', XYZ=0.65,0.55,0.65, QUANTITY='TEMPERATURE', SETPOINT=150.0, INITIAL_STATE=.FALSE., LATCH=.TRUE., TIME_AVERAGED=.FALSE. /
&OBST ID='DOOR', XB=1.4,1.6,0.2,0.8,0,1.0, SURF_ID='INERT', DEVC_ID='DOOR_OPEN' /
&DEVC ID='DOOR_OPEN', XYZ=1.25,0.55,0.65, QUANTITY='TIME', SETPOINT=0.5, INITIAL_STATE=.TRUE., LATCH=.TRUE., TIME_AVERAGED=.FALSE. /
&TAIL /
"""


def main():
    parent = OUT/"runs/hotel_init_v2/hotel_init_v2.fds"
    log = (parent.parent/"stderr.log").read_text("utf-8", errors="replace")
    assert "STOP: Set-up only" in log and not re.search(r"\bERROR(?:\(|:)", log)
    base = parent.read_text("utf-8")
    base = base.replace("T_END=0.0", "T_END=2.0")
    base = base.replace("DT_DEVC=5.0, DT_HRR=5.0", "DT_DEVC=0.1, DT_HRR=0.1")
    name = "hotel_short"
    write_case(name, base.replace("CHID='hotel_init_v2'", f"CHID='{name}'"),
               {"parent": str(parent.relative_to(ROOT)), "T_END_s": 2, "DT_DEVC_s": .1,
                "HCN_yield": "arbitrary0.01 software fixture", "door_and_window_controls": "unchanged100s/150C",
                "exposure_metrics_allowed": False})
    name = "hotel_control_timers"
    controls = base.replace("CHID='hotel_init_v2'", f"CHID='{name}'")
    controls = controls.replace("SETPOINT=100.0", "SETPOINT=0.5")
    controls = controls.replace("ID='SS_WINDOW_BREAK', QUANTITY='TEMPERATURE'", "ID='SS_WINDOW_BREAK', QUANTITY='TIME'")
    controls = controls.replace("SETPOINT=150.0", "SETPOINT=0.75")
    write_case(name, controls, {"parent": str(parent.relative_to(ROOT)), "T_END_s": 2,
               "DT_DEVC_s": .1, "door_setpoint_s": .5, "window_quantity": "TIME", "window_setpoint_s": .75,
               "purpose": "Test geometry actuation direction only, not thermal breakage",
               "HCN_yield": "arbitrary0.01 software fixture", "exposure_metrics_allowed": False})
    for name, ambient in (("micro_temperature_cold", 20), ("micro_temperature_hot", 160)):
        write_case(name, micro(name, ambient), {"source": "self-contained diagnostic fixture, not hotel input",
                   "T_END_s": 1, "ambient_C": ambient, "window_temperature_threshold_C": 150,
                   "door_open_s": .5, "no_combustion": True, "exposure_metrics_allowed": False})
    print(json.dumps({"prepared": ["hotel_short", "hotel_control_timers", "micro_temperature_cold", "micro_temperature_hot"]}))


if __name__ == "__main__":
    main()
