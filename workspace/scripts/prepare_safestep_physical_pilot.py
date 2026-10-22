"""Freeze source-backed exploratory assumptions and prepare init/pilot inputs."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"schemas/safestep_physics_time_contract_20261016.json"


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def main():
    c = load(CONFIG)
    out = ROOT/c["output_directory"]
    assert not (out/"frozen_physics_time_contract.json").exists(), "Do not overwrite a frozen contract"
    previous = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"
    complete = load(previous/"completion_manifest.json")
    protected = {**load(previous/"source_manifest.json")["protected_previous_sha256"], **complete["artifact_sha256"],
                 (previous/"completion_manifest.json").relative_to(ROOT).as_posix(): sha(previous/"completion_manifest.json")}
    for name, expected in protected.items():
        assert sha(ROOT/name) == expected, name
    materials = out/"materials"
    for name in ("nistir4989.pdf", "lund_selecting_design_fires_7032.pdf", "railway_tenability_app12115667.pdf", "NIST.TN.1760.pdf"):
        assert (materials/name).is_file(), name
    save(out/"frozen_physics_time_contract.json", c)
    save(out/"source_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
         "protected_previous_sha256": protected,
         "input_sha256": {CONFIG.relative_to(ROOT).as_posix(): sha(CONFIG),
                          **{p.relative_to(ROOT).as_posix(): sha(p) for p in materials.iterdir() if p.is_file()}},
         "sources": {
             "NISTIR4989": "https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4989.pdf",
             "LUND7032": "https://lucris.lub.lu.se/ws/files/5957828/1736728.pdf",
             "RAIL2022": "https://doi.org/10.3390/app12115667",
             "NISTTN1760": "https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1760.pdf",
             "FDS6111_GUIDE": "C:/Program Files/firemodels/FDS6/Documentation/Guides_and_Release_Notes/FDS_User_Guide.pdf"}})
    save(out/"exploratory_human_reference_NOT_AUTHOR_ORIGINAL.json", c["exploratory_human_reference_parameters"])
    save(out/"timestamp_contract.json", c["timestamp_contract"])
    table = [{"parameter": key, "value_and_evidence_json": json.dumps(value, ensure_ascii=False)} for key,value in c["physical_parameters"].items()]
    with (out/"physical_parameter_register.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    # Keep composition from the pinned NIST template; expose the guide's cp model.
    # Atomic weights are the FDS output precision used by the previous case.
    mw = 12.0107 + 1.7*1.00794 + .3*15.9994 + .08*14.0067
    cp_kJ = (1.4/.4)*8.314472/mw
    text = (ROOT/c["parent_input"]).read_text("utf-8")
    text = re.sub(r"!.*(?:\n|$)", "", text)  # old diagnostic-only header replaced below
    text = text.replace("HCN_YIELD=0.01", "HCN_YIELD=0.007")
    for item in ("C=1.0,", "H=1.7,", "O=0.3,", "N=0.08,"):
        text = text.replace(item, "")
    text = text.replace("HCN_YIELD=0.007", "HCN_YIELD=0.007, EPUMO2=13100.0, FUEL_RADCAL_ID='METHANE'")
    spec = f"&SPEC ID='REAC_FUEL', C=1.0, H=1.7, O=0.3, N=0.08, SPECIFIC_HEAT={cp_kJ:.10f} /\n"
    text = text.replace("&REAC ", spec+"&REAC ", 1)
    text = text.replace("TMPA=20.0, HUMIDITY=40.0", "TMPA=20.0, HUMIDITY=40.0, GAMMA=1.4")
    text = text.replace("DT_DEVC=5.0, DT_HRR=5.0", "DT_DEVC=0.5, DT_HRR=0.5, DT_SLCF=10.0")
    text = text.replace("TITLE='SOFTWARE DIAGNOSTIC ONLY; arbitrary HCN yield; not original paper fire'", "TITLE='Exploratory reconstruction; literature-transferred HCN and documented solver assumptions'")
    probes = ""
    for gas, species in (("CO", "CARBON MONOXIDE"), ("CO2", "CARBON DIOXIDE"), ("HCN", "HYDROGEN CYANIDE")):
        probes += f"&DEVC ID='ROOM24_{gas}', XYZ=4.375,24.125,2.0, QUANTITY='VOLUME FRACTION', SPEC_ID='{species}', TIME_AVERAGED=.FALSE., DRY=.FALSE. /\n"
    probes += "&DEVC ID='ROOM24_T', XYZ=4.375,24.125,2.0, QUANTITY='TEMPERATURE', TIME_AVERAGED=.FALSE. /\n"
    text = text.replace("&TAIL", probes+"&TAIL")
    header = ("! EXPLORATORY RECONSTRUCTION 2026-10-16; NOT SAFESTEP ORIGINAL FIRE.\n"
              "! Derived from NIST FD-Gen; complete source notice in FD-Gen_LICENSE.md.\n"
              "! HCN=0.007 transferred from different NIST chair tests, not GM27 calibration.\n"
              "! Cp, transport and radiation use documented model assumptions, not measured mattress properties.\n")
    for name, duration in (("physics_init", 0), ("hotel_literature_pilot120", 120)):
        path = out/"runs"/name;path.mkdir(parents=True, exist_ok=False)
        case = text.replace("CHID='hotel_init_v2'", f"CHID='{name}'").replace("T_END=0.0", f"T_END={duration:.1f}")
        save(path/(name+".fds"), header+case)
        save(path/"derivation.json", {"parent": c["parent_input"], "physical_duration_s": duration,
             "changes": ["HCN0.01 to literature-transfer0.007", "Move elemental composition fromREAC toSPEC", "Explicit gamma-basedspecificheat", "ExplicitEPUMO2 andmethane radiation surrogate", "0.5s native device output", "FourRoom24 gas/temperatureprobes"],
             "calculated_fuel_MW_g_per_mol": mw, "model_specific_heat_kJ_per_kg_K": cp_kJ,
             "original_geometry_burner_door_window_unmodified": True,
             "original_paper_physics_claim_allowed": False})
    shutil.copyfile(previous/"FD-Gen_LICENSE.md", out/"FD-Gen_LICENSE.md")
    print(json.dumps({"prepared": ["physics_init", "hotel_literature_pilot120"], "HCN_yield": .007,
                      "specific_heat_model_kJ_kg_K": cp_kJ, "protected_previous": len(protected)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
