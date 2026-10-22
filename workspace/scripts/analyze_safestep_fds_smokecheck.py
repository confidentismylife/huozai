"""Audit solver-produced logs, native timestamps and initialized voxel geometry.

Concentrations remain diagnostic. This script does not compute FER/FED or fit
any fire parameter, and never reads archived hazard-rate values to compare.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from safestep_gas_to_fer import read_fds_gas_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"
CHECKS = []


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def csv_save(path, rows):
    assert not path.exists(), path
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(name, condition, detail=None):
    if not condition:
        raise AssertionError((name, detail))
    CHECKS.append({"name": name, "passed": True, "detail": detail})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def parse_smv(path):
    lines = path.read_text("utf-8", errors="replace").splitlines()
    meshes, events = [], []
    for i, line in enumerate(lines):
        if line.strip() == "OBST":
            count = int(lines[i+1])
            records = []
            for j in range(count):
                raw = lines[i+2+j]
                grid = lines[i+2+count+j]
                numbers = raw.split("!")[0].split()
                records.append({"local_OBST_index": j+1,
                                "requested_clipped_XB": [float(x) for x in numbers[:6]],
                                "source_OBST_index": int(numbers[6]),
                                "ID": raw.split("!", 1)[1].strip() if "!" in raw else "",
                                "grid_bounds": [int(x) for x in grid.split()[:6]]})
            meshes.append(records)
        match = re.match(r"^(HIDE_OBST|SHOW_OBST)\s+(\d+)", line)
        if match:
            j, t = lines[i+1].split()[:2]
            events.append({"kind": match.group(1), "mesh_index": int(match.group(2)),
                           "local_OBST_index": int(j), "geometry_event_time_s": float(t)})
    for event in events:
        item = meshes[event["mesh_index"]-1][event["local_OBST_index"]-1]
        event.update({"ID": item["ID"], "requested_clipped_XB": item["requested_clipped_XB"],
                      "grid_bounds": item["grid_bounds"]})
    return meshes, events


def controls(path):
    rows = read_csv(path)
    return [{"time_s": float(r[0]), "type": r[1].strip(), "ID": r[2].strip(),
             "state": r[3].strip(), "value": float(r[4]), "unit": r[5].strip()} for r in rows[1:]]


def hotel_grid_devices(init_dir):
    input_text = (init_dir/(init_dir.name+".fds")).read_text("utf-8")
    meshes = []
    for body in re.findall(r"&MESH\b([^/]*?)/", input_text, re.S):
        values = lambda key: [float(v.strip()) for v in re.search(rf"\b{key}=([+\-\d.eE,]+)", body).group(1).rstrip(",").split(",")]
        meshes.append({"ID": re.search(r"ID='([^']*)'", body).group(1), "IJK": values("IJK"), "XB": values("XB")})
    geometry, _ = parse_smv(init_dir/(init_dir.name+".smv"))
    check("initialized_mesh_geometry_count6", len(geometry) == len(meshes) == 6)
    with (ROOT/"outputs/safestep_hotel_fds_candidate_20261016/region_mapping_candidate.csv").open(encoding="utf-8-sig", newline="") as handle:
        regions = list(csv.DictReader(handle))
    rows = []
    for r in regions:
        xyz = [float(r[k]) for k in ("x_m", "y_m", "z_m")]
        matches = []
        for mesh_index, mesh in enumerate(meshes, 1):
            xb, dims = mesh["XB"], mesh["IJK"]
            if not all(xb[2*a] <= xyz[a] <= xb[2*a+1] for a in range(3)):
                continue
            incident = []
            for a in range(3):
                position = (xyz[a]-xb[2*a]) / ((xb[2*a+1]-xb[2*a])/dims[a])
                nearest = round(position)
                # On an interior grid plane, consider both neighboring cells.
                # This avoids claiming which exact cell the solver selected.
                candidates = [nearest, nearest+1] if abs(position-nearest) < 1e-7 else [math.floor(position)+1]
                incident.append([c for c in candidates if 1 <= c <= dims[a]])
            cells = list(itertools.product(*incident))
            solid = []
            for cell in cells:
                for obst in geometry[mesh_index-1]:
                    b = obst["grid_bounds"]
                    if all(b[2*a] < cell[a] <= b[2*a+1] for a in range(3)):
                        solid.append({"cell_IJK": list(cell), "OBST_local_index": obst["local_OBST_index"]})
            matches.append({"mesh_index": mesh_index, "mesh_ID": mesh["ID"], "incident_cells": [list(x) for x in cells],
                            "solid_cell_matches": solid})
        check("region_initialized_adjacent_cells_fluid:"+r["region"], bool(matches) and all(not m["solid_cell_matches"] for m in matches))
        rows.append({"region": r["region"], "requested_XYZ_m": xyz, "initialized_candidate_cells": matches,
                     "status": "ALL_INCIDENT_CELLS_CLEAR_OF_INITIALIZED_VOLUME_OBSTRUCTIONS",
                     "limit": "All neighboring cells checked; exact solver cell selection and author-original region coordinates not recovered"})
    return rows


def main():
    assert not (OUT/"verification.json").exists(), "Preserve completed analysis"
    manifest = load(OUT/"source_manifest.json")
    for name, sha in manifest["protected_previous_sha256"].items():
        check("protected_unchanged:"+name, digest(ROOT/name) == sha)
    check("installed_exe_unchanged", digest(Path(manifest["installed_executable"])) == manifest["installed_executable_sha256"])
    check("installed_guide_unchanged", digest(Path(manifest["installed_guide"])) == manifest["installed_guide_sha256"])
    runs = OUT/"runs"
    run_rows = []
    all_events = {}
    for case in sorted(p for p in runs.iterdir() if p.is_dir()):
        result = load(case/"run_result.json")
        stderr = (case/"stderr.log").read_text("utf-8", errors="replace")
        errors = [line.strip() for line in stderr.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
        initialization = "STOP: Set-up only" in stderr
        completed = "STOP: FDS completed successfully" in stderr
        classification = "FDS_ERROR" if errors else "SETUP_COMPLETED" if initialization else "DYNAMIC_COMPLETED" if completed else "UNCONFIRMED"
        row = {"case": case.name, "exit_code": result["returncode"], "classified_status": classification,
               "elapsed_s": result["elapsed_s"], "FDS_error_count": len(errors), "FDS_errors_json": json.dumps(errors),
               "warning_count": sum(line.strip().startswith("WARNING:") for line in stderr.splitlines()),
               "reached_fire_time_s": 0.0, "native_device_time_samples": 0, "gas_output_columns": 0}
        if case.name == "hotel_init_v1":
            check("exitcode0_not_sufficient_for_success", result["returncode"] == 0 and bool(errors))
        else:
            check("case_completed:"+case.name, not errors and result["returncode"] == 0 and (initialization or completed))
        if completed:
            table = read_csv(case/(case.name+"_devc.csv"))
            times = [float(r[0]) for r in table[2:]]
            row.update({"reached_fire_time_s": times[-1], "native_device_time_samples": len(times),
                        "gas_output_columns": sum(h.startswith("SS_R") for h in table[1])})
            check("times_finite_strictly_increasing:"+case.name, all(math.isfinite(t) for t in times) and all(b>a for a,b in zip(times,times[1:])))
            geometry, smv_events = parse_smv(case/(case.name+".smv"))
            all_events[case.name] = {"logical_events": controls(case/(case.name+"_devc_ctrl_log.csv")), "geometry_events": smv_events}
        run_rows.append(row)
    check("six_total_input_runs", len(run_rows) == 6)
    check("four_successful_dynamic_runs", sum(r["classified_status"] == "DYNAMIC_COMPLETED" for r in run_rows) == 4)
    check("hotel_short_no_early_control_activation", not all_events["hotel_short"]["logical_events"] and not all_events["hotel_short"]["geometry_events"])
    timer = all_events["hotel_control_timers"]
    check("timer_door_deactivates_present_OBST", any(e["ID"] == "SS_DOOR_OPEN" and e["state"] == "F" for e in timer["logical_events"]))
    check("timer_window_activates_HOLE", any(e["ID"] == "SS_WINDOW_BREAK" and e["state"] == "T" for e in timer["logical_events"]))
    door_geometry = [e for e in timer["geometry_events"] if e["ID"] == "SS_DOOR24"]
    window_geometry = [e for e in timer["geometry_events"] if e["requested_clipped_XB"] == [-.25, 0, 23.5, 25, 1.3, 2.8]]
    check("door_removed_in_both_touching_meshes", len(door_geometry) == 2 and all(e["kind"] == "HIDE_OBST" for e in door_geometry))
    check("window_panel_removed_in_room24_mesh", len(window_geometry) == 1 and window_geometry[0]["kind"] == "HIDE_OBST")
    cold, hot = all_events["micro_temperature_cold"], all_events["micro_temperature_hot"]
    check("20C_does_not_trigger150C_window", not any(e["ID"] == "WINDOW_BREAK" for e in cold["logical_events"]))
    check("160C_does_trigger150C_window", any(e["ID"] == "WINDOW_BREAK" and e["state"] == "T" and e["value"] >= 150 for e in hot["logical_events"]))
    check("cold_geometry_window_stays_closed", len(cold["geometry_events"]) == 1 and cold["geometry_events"][0]["ID"] == "DOOR")
    check("hot_geometry_window_and_door_open", len(hot["geometry_events"]) == 2)
    # This is the actual initialized/snapped obstacle occupancy from .smv.
    grid_audit = hotel_grid_devices(runs/"hotel_init_v2")
    save(OUT/"initialized_region_cell_screen.json", grid_audit)
    event_rows = []
    for case, details in all_events.items():
        for e in details["logical_events"]:
            if e["ID"] in ("SS_DOOR_OPEN", "DOOR_OPEN"):
                geometry = [g for g in details["geometry_events"] if g["ID"] in ("SS_DOOR24", "DOOR")]
                nominal = .5
            else:
                geometry = [g for g in details["geometry_events"] if g["ID"] not in ("SS_DOOR24", "DOOR")]
                nominal = .75 if case == "hotel_control_timers" else 0.0
            for g in geometry:
                event_rows.append({"case": case, "device_ID": e["ID"], "new_state": e["state"],
                                   "nominal_test_time_s": nominal, "logical_event_time_s": e["time_s"],
                                   "mesh_index": g["mesh_index"], "OBST_local_index": g["local_OBST_index"],
                                   "geometry_event_time_s": g["geometry_event_time_s"],
                                   "geometry_minus_logical_s": g["geometry_event_time_s"]-e["time_s"]})
    csv_save(OUT/"control_event_timing.csv", event_rows)
    save(OUT/"control_event_details.json", all_events)
    # Use the previous interface's non-resampling CSV reader on real FDS output.
    data_path = runs/"hotel_short/hotel_short_devc.csv"
    times, concentrations = read_fds_gas_csv(data_path)
    check("existing_native_reader_accepts_all174_actual_FDS_gas_columns", len(concentrations) == len(times) == 13)
    native_rows = []
    for t, sample in zip(times, concentrations):
        for r in range(1,59):
            values = {g: sample[(r,g)] for g in ("CO","CO2","HCN")}
            check("gas_volume_fraction_valid", all(0 <= v <= 1 for v in values.values()) and sum(values.values()) <= 1+1e-12)
            native_rows.append({"region": f"R{r}", "fire_time_s": t, "evacuation_elapsed_s": t-300,
                                "CO_mol_per_mol": values["CO"], "CO2_mol_per_mol": values["CO2"], "HCN_mol_per_mol": values["HCN"],
                                "evidence_label": "SOFTWARE_DIAGNOSTIC_ARBITRARY_HCN_YIELD_NOT_PAPER_FIRE"})
    csv_save(OUT/"native_diagnostic_gas_long.csv", native_rows)
    step_differences = [b-a for a,b in zip(times,times[1:])]
    time_errors = [min(abs(t-k*.1) for k in range(21)) for t in times]
    timestamp_summary = {"requested_device_interval_s": .1, "native_times_s": times,
                         "actual_min_interval_s": min(step_differences), "actual_max_interval_s": max(step_differences),
                         "maximum_distance_to_requested_grid_s": max(time_errors),
                         "exact_requested_grid": all(e < 1e-8 for e in time_errors),
                         "no_resampling_applied": True,
                         "implication": "FDSnative timestamps do not automatically satisfy the frozen0..900s/5s converter contract. Must define/audit actual-time integration or explicit resampling before paper-FER comparison."}
    check("native_times_are_not_exact_nominal_output_grid", not timestamp_summary["exact_requested_grid"])
    save(OUT/"native_timestamp_audit.json", timestamp_summary)
    hrr = read_csv(runs/"hotel_short/hotel_short_hrr.csv")
    hrr_i = hrr[1].index("HRR")
    hrr_peak = max(float(r[hrr_i]) for r in hrr[2:])
    summary = {"solver": "FDS6.11.1-0-gff928db-release", "input_run_attempts": len(run_rows),
               "initialization_failures": 1, "successful_initializations": 1, "successful_dynamic_runs": 4,
               "hotel_short_duration_s": 2.0, "hotel_control_diagnostic_duration_s": 2.0,
               "gas_devices": 174, "gas_time_samples_hotel_short": len(times),
               "native_long_gas_rows": len(native_rows), "initialized_region_screen_passed": len(grid_audit),
               "temporary_HCN_yield": .01, "HCN_yield_is_original_or_empirical": False,
               "CO_max_corridor_mol_per_mol": max(r["CO_mol_per_mol"] for r in native_rows),
               "HCN_max_corridor_mol_per_mol": max(r["HCN_mol_per_mol"] for r in native_rows),
               "actual_HRR_peak_kW_in2s_diagnostic": hrr_peak,
               "source_fuel_property_warnings_unresolved": True,
               "paper_effect_experiments": 0, "FER_FED_values_computed": 0,
               "full900s_fire_runs": 0, "training_updates": 0, "LLM_requests": 0,
               "protected_previous_files_unchanged": len(manifest["protected_previous_sha256"]),
               "old_final48_crowd_scenarios_accessed": False,
               "terminal_claim": "Successful initialization,4short solver diagnostics andnative export verified; no paper physical fire/FED reproduction"}
    save(OUT/"run_summary.json", summary)
    csv_save(OUT/"run_comparison.csv", run_rows)
    save(OUT/"readiness_after_diagnostics.json", {
        "software_diagnostic_gate": "PASS",
        "original_physics_reproduction_gate": "NOT_READY",
        "resolved": ["BundledMPI DLL path", "232legacy comma-containing device IDs", "Initialized hotel gas sampling adjacency", "Whole-door andwindow-HOLE actuation signs", "Actual174gas CSV output andwet volume-fraction units"],
        "new_findings": ["FDSmay return0even wheninputsetupfails", "Outputtimes followactual solver steps", "Geometrychanges canfollowlogical controls byanother timestep"],
        "remaining": ["OriginalHCN yield andgas time series", "Originalfuel thermophysical assumptions andsource fallback warnings", "Exactgeometry/mapping,burner/postpeak,glass model", "V,D,unitsandformula time basis", "Explicitnative-time integration/resampling rule", "Meaningful full-fire stability andresolution/convergence verification"],
        "next_bounded_step": "Define a source-backed physical parameter table andnative-time handling before planning any300s/900s run. Do notextenddiagnostic fixture intoeffect claims."})
    save(OUT/"verification.json", {"status": "PASS_BOUNDED_FDS_DIAGNOSTICS_NOT_PHYSICAL_REPRODUCTION",
         "created_at": datetime.now(timezone.utc).isoformat(), "checks": CHECKS, "check_count": len(CHECKS),
         "scope": summary})
    print(json.dumps({"status": "PASS", "checks": len(CHECKS), "runs": len(run_rows),
                      "dynamic_completed": 4, "gas_columns": 174, "native_rows": len(native_rows),
                      "peak_diagnostic_HRR_kW": hrr_peak}, ensure_ascii=True))


if __name__ == "__main__":
    main()
