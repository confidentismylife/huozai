"""Freeze a clean, normal-end hotel run and a passive-checkpoint comparison."""
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path

from prepare_safestep_restart_hrr_audit import ROOT, OLD, sha, save

OUT = ROOT / "outputs/safestep_clean_hotel120_20261017"
AUDIT = ROOT / "outputs/safestep_restart_hrr_audit_20261017"
PREFIX = "hotel_clean120"
PARENT_INPUT = OLD / "runs/hotel_literature_pilot120_v2/hotel_literature_pilot120_v2.fds"


def main():
    OUT.mkdir(exist_ok=False)
    source = json.loads((OLD / "source_manifest.json").read_text("utf-8"))
    protected = dict(source["protected_previous_sha256"])
    for directory in (OLD, AUDIT):
        manifest = directory / "completion_manifest.json"
        protected.update(json.loads(manifest.read_text("utf-8"))["artifact_sha256"])
        protected[manifest.relative_to(ROOT).as_posix()] = sha(manifest)
    bad = [name for name, expected in protected.items() if sha(ROOT / name) != expected]
    assert not bad, bad
    plan = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Clean exploratory hotel trajectory through 120s, then paired native restart fidelity check",
        "claim_boundary": "Not the recovered author fire input, material calibration, route FED or evacuation effectiveness",
        "FDS_executable": "C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe",
        "FDS_version": "FDS-6.11.1-0-gff928db-release",
        "parent_input": PARENT_INPUT.relative_to(ROOT).as_posix(), "parent_input_sha256": sha(PARENT_INPUT),
        "physical_parameters_and_geometry_changed": False,
        "input_changes": {"CHID": PREFIX, "DT_RESTART_s": 20.0},
        "normal_end_fire_time_s": 120.0, "hard_wall_limit_continuous_s": 2400,
        "OMP_NUM_THREADS": "2", "MPI_processes": 1,
        "stop_file_creation_permitted": False,
        "checkpoint_capture": {"first_native_checkpoint_at_or_after_s": 100.0,
            "upper_bound_s": 105.0, "require_six_matching_native_times": True,
            "require_complete_Fortran_records_and_stable_hashes": True,
            "capture_while_continuous_run_keeps_running": True},
        "paired_restart": {"T_END_s": 110.0, "hard_wall_limit_s": 600,
            "same_CHID": True, "same_checkpoint_bytes": True, "input_change": "MISC RESTART=.TRUE. plus shortened diagnostic T_END",
            "compare_interval": "After captured checkpoint through 110s, never extrapolate"},
        "comparison_policy": {
            "HRR_energy": "Integrate output-window means over common supported interval; account for saved partial accumulator",
            "HRR_shape": "Linear interpolation of output-window means at common 0.5s times, descriptive shape check only",
            "devices": "Linear interpolation at common 0.5s times for paired numerical comparison only; not FED authority",
            "zero_concentrations": "Use explicit absolute floors for peak-normalized errors",
            "native_data": "Preserve raw clocks and complete arrays; no silent repair or spike filtering"},
        "predeclared_engineering_screen": {
            "note": "Exploratory numerical consistency screens, not fire-safety or material-validity standards",
            "HRR_window_energy_relative_error_max": 0.01,
            "HRR_interpolated_point_relative_error_max": 0.05,
            "CO_peak_scaled_max_abs_error_max": 0.05, "CO_reference_peak_floor_ppm": 1.0,
            "HCN_peak_scaled_max_abs_error_max": 0.05, "HCN_reference_peak_floor_ppm": 0.2,
            "CO2_peak_scaled_max_abs_error_max": 0.05, "CO2_reference_peak_floor_ppm": 20.0,
            "temperature_max_abs_error_C": 2.0,
            "door_and_window_event_sequences_preserved": True,
            "all_gates_required_for_next_long_run_recommendation": True},
        "resource_scope": {"clean_continuous_runs": 1, "short_paired_restarts": 1,
            "training_updates": 0, "LLM_requests": 0, "300s_or900s_runs": 0,
            "sealed_final48_scenarios_accessed": False},
        "protected_previous_sha256": protected,
    }
    save(OUT / "frozen_plan.json", plan)
    original = PARENT_INPUT.read_text("utf-8")
    candidate = original.replace("CHID='hotel_literature_pilot120_v2'", f"CHID='{PREFIX}'")
    assert original.count("&DUMP DT_DEVC=0.5, DT_HRR=0.5, DT_SLCF=10.0 /") == 1
    candidate = candidate.replace("&DUMP DT_DEVC=0.5, DT_HRR=0.5, DT_SLCF=10.0 /",
                                  "&DUMP DT_DEVC=0.5, DT_HRR=0.5, DT_SLCF=10.0, DT_RESTART=20.0 /")
    case = OUT / "runs" / PREFIX
    case.mkdir(parents=True)
    target = case / f"{PREFIX}.fds"
    target.write_text(candidate, "utf-8")
    (OUT / "input_changes.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
         fromfile=PARENT_INPUT.name, tofile=target.name)), "utf-8")
    save(case / "derivation.json", {"source": plan["parent_input"], "source_sha256": sha(PARENT_INPUT),
         "new_input_sha256": sha(target), "only_changes": plan["input_changes"],
         "physical_parameters_changed": False, "normal_T_END_120s_unchanged": True})
    save(OUT / "preflight.json", {"previous_files_verified_unchanged": len(protected), "mismatches": bad,
         "new_input_exists": True, "native_restart_interval_s": 20.0, "no_stop_file": not (case/f"{PREFIX}.stop").exists(),
         "input_sha256": sha(target), "ready": True})
    print(json.dumps({"prepared": PREFIX, "protected_files": len(protected), "physical_changes": False}))


if __name__ == "__main__":
    main()
