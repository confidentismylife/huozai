"""Analyze native restart artifacts without editing any solver output."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_safestep_restart_hrr_audit import ROOT, OUT, OLD, sha, save
from run_safestep_restart_hrr_controls import PREFIX, PARENT, CASES


def read_csv(path):
    result = pd.read_csv(path, skiprows=1)
    assert result.Time.is_unique and np.all(np.diff(result.Time) > 0), path
    assert np.all(np.isfinite(result.to_numpy())), path
    return result


def checkpoint_metadata(path, ijk, hrr_interval):
    """Read only this pinned Cartesian FDS layout; assert record framing and size.

    dump.f90 writes 10 flow records, then allocated D_SOURCE and M_DOT_PPP,
    RHO, TMP, Q, QR, CHI_R. The timing record precedes 17 output intervals
    and the energy/mass accumulators. It ends in six double-precision scalars
    starting with T_LAST_DUMP_HRR. This is not a general-purpose FDS reader.
    """
    previous = None; snapshots = {}; state = None; state_index = None
    with path.open("rb") as stream:
        index = 0
        while header := stream.read(4):
            assert len(header) == 4
            size = struct.unpack("<i", header)[0]
            assert 0 <= size < 2**30
            data = stream.read(size); trailer = stream.read(4)
            assert len(data) == size and len(trailer) == 4 and struct.unpack("<i", trailer)[0] == size
            base_bytes = int(np.prod(np.array(ijk)+2))*8
            if index == 10:
                assert size == base_bytes, "This reader requires allocated D_SOURCE"
            if index == 11:
                assert size == base_bytes*3, "This audit requires three tracked species"
            if index in (14, 16):
                assert size == base_bytes
                values = np.frombuffer(data, dtype="<f8")
                snapshots["Q" if index == 14 else "CHI_R"] = {
                    "record_index_zero_based": index, "bytes": size,
                    "nonzero_count": int(np.count_nonzero(values)), "min": float(values.min()),
                    "max": float(values.max()), "raw_sum": float(values.sum()),
                }
            if size == 17*8 and abs(struct.unpack_from("<d", data, 6*8)[0]-hrr_interval) < 1e-12:
                assert state is None
                state = {"timing_record_index_zero_based": index-1,
                         "fire_time_s": struct.unpack_from("<d", previous)[0],
                         "iteration": struct.unpack_from("<i", previous, 8)[0],
                         "DT_s": struct.unpack_from("<d", previous, 24)[0],
                         "last_HRR_dump_time_s": struct.unpack_from("<d", previous, len(previous)-48)[0],
                         "trailing_six_scalar_values": list(struct.unpack("<6d", previous[-48:]))}
                state_index = index
            elif state_index is not None and index == state_index+1:
                assert size >= 12*8
                state["Q_DOT_SUM_J"] = list(struct.unpack_from("<9d", data))
                state["M_DOT_SUM_kg"] = list(struct.unpack_from("<3d", data, 9*8))
                break
            previous = data; index += 1
    assert state is not None and "Q_DOT_SUM_J" in state and "Q" in snapshots
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), **state, "snapshot_fields": snapshots}


def window_energy(data, left=None):
    # DUMP_HRR divides integrated Q_DOT_SUM by time since previous dump.
    # CSV timestamps are REAL(FB) rounded outputs: this is reconstructed output
    # window energy, not an independent conservation or true chemical-energy check.
    times = data.Time.to_numpy(); hrr = data.HRR.to_numpy()
    if left is None:
        return float(np.dot(np.diff(times), hrr[1:]))
    widths = np.diff(np.r_[left, times]); assert np.all(widths > 0)
    return float(np.dot(widths, hrr))


def source_evidence():
    patterns = {
        "stop_flag_set_before_combustion": ("main.f90", "STOP_STATUS = USER_STOP"),
        "wait_for_radiation_completion": ("main.f90", "IF (STOP_AT_ITER==0 .AND. .NOT.ALL(RADIATION_COMPLETED)) RETURN"),
        "zero_reaction_heat_before_chemistry": ("fire.f90", "Q     = 0._EB"),
        "skip_chemistry_when_stop_flag_present": ("fire.f90", "IF (STOP_STATUS/=NO_STOP) RETURN"),
        "checkpoint_keeps_partial_accumulator": ("dump.f90", "WRITE(LU_CORE(NM)) Q_DOT_SUM"),
        "native_HRR_time_average": ("dump.f90", "Q_DOT_SUM = Q_DOT_SUM/MAX(DT,T-T_LAST_DUMP_HRR)"),
        "user_frequency_override_after_restart": ("dump.f90", "IF (DT_HRR_SPECIFIED    > 0._EB) DT_HRR"),
    }
    result = {}
    commit = json.loads((OUT / "source_manifest.json").read_text("utf-8"))["commit"]
    for key, (filename, pattern) in patterns.items():
        lines = (OUT / "materials" / filename).read_text("utf-8").splitlines()
        hits = [i+1 for i, line in enumerate(lines) if pattern in line]
        assert hits
        result[key] = {"file": f"materials/{filename}", "line_numbers": hits,
                       "official_url": f"https://github.com/firemodels/fds/blob/{commit}/Source/{filename}#L{hits[0]}"}
    save(OUT / "source_code_evidence.json", result)


def main():
    assert not (OUT / "audit_results.json").exists()
    source_evidence()
    hotel_meta = []
    meshes = [(220,12,12), (7,67,12), (152,9,12), (17,40,12), (9,122,12), (53,21,17)]
    for mesh, ijk in enumerate(meshes, 1):
        hotel_meta.append(checkpoint_metadata(PARENT / f"{PREFIX}_{mesh}.restart", ijk, .5))
    assert len({m["fire_time_s"] for m in hotel_meta}) == 1
    assert len({m["last_HRR_dump_time_s"] for m in hotel_meta}) == 1
    checkpoint = hotel_meta[-1]["fire_time_s"]
    previous_dump = hotel_meta[-1]["last_HRR_dump_time_s"]
    stored_energy = hotel_meta[-1]["Q_DOT_SUM_J"][0]/1000
    assert all(m["snapshot_fields"]["Q"]["nonzero_count"] == 0 for m in hotel_meta)
    hotel_hrr = {}; hotel_results = {}; solver_runs = []
    for case in CASES:
        path = OUT / "runs" / case
        result = json.loads((path / "run_result.json").read_text("utf-8")); assert result["passed"]
        solver_runs.append({"case": case, **result})
        data = read_csv(path / f"{PREFIX}_hrr.csv"); data = data[data.Time > checkpoint].copy()
        hotel_hrr[case] = data
        hotel_results[case] = {"new_output_rows": len(data), "first_rows": data[["Time","HRR"]].head(5).to_dict("records"),
                "peak_output_mean_HRR_kW": float(data.HRR.max()), "peak_time_s": float(data.loc[data.HRR.idxmax(), "Time"]),
                "window_energy_from_previous_dump_kJ": window_energy(data, previous_dump),
                "energy_after_checkpoint_reconstructed_kJ": window_energy(data, previous_dump)-stored_energy}
    coarse = read_csv(OUT / "runs/hotel_repeat_hrr050" / f"{PREFIX}_devc.csv")
    dense = read_csv(OUT / "runs/hotel_dense_hrr0025" / f"{PREFIX}_devc.csv")
    assert coarse.shape == dense.shape and np.array_equal(coarse.to_numpy(), dense.to_numpy())
    old_resume = read_csv(OLD / "runs/hotel_literature_pilot120_resume" / f"{PREFIX}_hrr.csv")
    repeated = hotel_hrr["hotel_repeat_hrr050"]
    matched = repeated.merge(old_resume, on="Time", suffixes=("_repeat", "_old"))
    assert len(matched) == 5 and np.array_equal(matched.HRR_repeat.to_numpy(), matched.HRR_old.to_numpy())
    first = hotel_hrr["hotel_dense_hrr0025"].iloc[0]
    step_HRR = (float(first.HRR)*(float(first.Time)-previous_dump)-stored_energy)/(float(first.Time)-checkpoint)
    hotel_comparison = {
        "checkpoint_metadata": hotel_meta, "checkpoint_s": checkpoint, "last_HRR_dump_s": previous_dump,
        "saved_partial_HRR_energy_kJ": stored_energy, "saved_partial_interval_s": checkpoint-previous_dump,
        "saved_partial_mean_HRR_kW": stored_energy/(checkpoint-previous_dump),
        "first_resumed_step_HRR_inferred_kW": step_HRR,
        "inference_caveat": "Subtract saved Q_DOT_SUM from the first CSV interval energy; approximate due to printed REAL(FB) timestamp rounding",
        "cases": hotel_results, "coarse_dense_all_device_values_exact_match": True,
        "device_rows": len(coarse), "device_channels_excluding_time": coarse.shape[1]-1,
        "prior_resume_identical_HRR_rows": len(matched),
        "coarse_dense_energy_difference_kJ": hotel_results["hotel_dense_hrr0025"]["window_energy_from_previous_dump_kJ"]-hotel_results["hotel_repeat_hrr050"]["window_energy_from_previous_dump_kJ"],
        "uninterrupted_full_hotel_control_done": False,
    }
    micro_cases = [("micro_continuous", "micro_continuous"), ("micro_periodic_resume", "micro_continuous"),
                   ("micro_stop_parent", "micro_stop_parent"), ("micro_stop_resume", "micro_stop_parent")]
    micro = {}; micro_meta = {}
    for name, prefix in micro_cases:
        path = OUT / "microfixture" / name
        result = json.loads((path / "run_result.json").read_text("utf-8")); assert result["passed"]
        solver_runs.append({"case": name, **result})
        micro[name] = read_csv(path / f"{prefix}_hrr.csv")
        micro_meta[name] = checkpoint_metadata(path / f"{prefix}_1.restart", (12,12,16), .02)
    reference = micro["micro_continuous"]; ref_energy = window_energy(reference)
    comparisons = {}; device_diffs = {}
    ref_devices = read_csv(OUT / "microfixture/micro_continuous/micro_continuous_devc.csv")
    for name, prefix in micro_cases[1::2]:
        data = micro[name]; merged = reference.merge(data, on="Time", suffixes=("_continuous", "_resumed"))
        post = merged[merged.Time > 3.0]
        error = post.HRR_resumed-post.HRR_continuous
        devices = read_csv(OUT / "microfixture" / name / f"{prefix}_devc.csv")
        device_matched = ref_devices.merge(devices, on="Time", suffixes=("_continuous", "_resumed"))
        device_diffs[name] = pd.DataFrame({"Time": device_matched.Time,
                                         "temperature_difference_C": device_matched.T_PROBE_resumed-device_matched.T_PROBE_continuous})
        comparisons[name] = {
            "matching_native_HRR_times": len(merged), "reference_HRR_rows": len(reference), "resumed_HRR_rows": len(data),
            "reference_times_not_in_resumed_s": np.setdiff1d(reference.Time, data.Time).tolist(),
            "post_restart_matching_rows": len(post), "post_restart_max_abs_HRR_difference_kW": float(np.abs(error).max()),
            "post_restart_max_relative_HRR_difference": float(np.max(np.abs(error/post.HRR_continuous))),
            "post_restart_peak_HRR_kW": float(post.HRR_resumed.max()),
            "window_energy_0_to_5p5_kJ": window_energy(data),
            "relative_window_energy_difference": window_energy(data)/ref_energy-1,
            "max_matching_probe_temperature_difference_C": float(np.max(np.abs(device_diffs[name].temperature_difference_C))),
            "max_matching_probe_fuel_mole_fraction_difference": float(np.max(np.abs(device_matched.FUEL_PROBE_resumed-device_matched.FUEL_PROBE_continuous))),
            "max_matching_probe_CO2_mole_fraction_difference": float(np.max(np.abs(device_matched.CO2_PROBE_resumed-device_matched.CO2_PROBE_continuous))),
        }
    stopped = micro["micro_stop_parent"]
    zeros = stopped[(stopped.Time > 2.5) & (stopped.HRR == 0)]
    assert len(zeros) > 0
    assert micro_meta["micro_continuous"]["snapshot_fields"]["Q"]["nonzero_count"] > 0
    assert micro_meta["micro_stop_parent"]["snapshot_fields"]["Q"]["nonzero_count"] == 0
    micro_results = {"reference_window_energy_kJ": ref_energy, "checkpoint_metadata": micro_meta,
                     "stop_parent_zero_HRR_output_times_s": zeros.Time.tolist(), "comparisons": comparisons,
                     "continuous_trajectory_compared_from": 0.0, "end_s": 5.5,
                     "periodic_restart_bitwise_equivalence": False,
                     "interpretation": "Periodic restart is close in this fixture; native .stop visibly changes reaction heat and gas/temperature probes. Total window energy similarity does not imply trajectory fidelity."}
    save(OUT / "solver_run_summary.json", {"runs": solver_runs, "new_solver_processes": len(solver_runs),
         "passed": sum(r["passed"] for r in solver_runs), "total_solver_wall_s": sum(r["elapsed_s"] for r in solver_runs)})
    results = {"status": "STOP_RELATED_CHEMISTRY_TRANSIENT_SUPPORTED_BY_SOURCE_AND_CONTROL",
        "created_at": datetime.now(timezone.utc).isoformat(), "hotel": hotel_comparison, "microfixture": micro_results,
        "energy_reporting": "Sum output-window mean HRR times native interval duration; printed-time precision limited. Not trapezoidal interpolation and not independent conservation proof.",
        "decision": {"reuse_old_stop_checkpoint_for_longer_physical_runs": False,
             "filter_or_smooth_original_spikes": False, "modify_FDS_or_restart_bytes": False,
             "next_run": "Rebuild 0..120s normally with periodic restart saves, adequate hard wall limit; audit clean hotel restart before 300s",
             "extend_to300s_in_this_audit": False, "claim_original_SafeStep_physics_reproduced": False},
        "limitations": ["No clean uninterrupted full hotel comparison in this audit",
            "One small propane fixture is not a validation of hotel material parameters, mesh, radiation or HCN transfer",
            "Periodic restart is not bitwise identical; native output time grids can differ",
            "Old 120s outputs remain raw exploratory artifacts; stop-neighborhood and later states are not promoted to validated physical evidence"]}
    save(OUT / "audit_results.json", results)
    data = {"hotel": {name: value[["Time", "HRR"]].to_dict("records") for name, value in hotel_hrr.items()},
            "microfixture": {name: value[["Time", "HRR"]].to_dict("records") for name, value in micro.items()},
            "probe_temperature_differences": {name: value.to_dict("records") for name, value in device_diffs.items()}}
    save(OUT / "figure_data.json", data)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    ax = axes[0]
    for name, color, label in [("hotel_dense_hrr0025", blue, "Hotel: dense output"), ("hotel_repeat_hrr050", orange, "Hotel: 0.5 s output")]:
        d = hotel_hrr[name]; ax.plot(d.Time, d.HRR, ".-", color=color, lw=1.3, ms=4, label=label)
    t = np.linspace(115.0, 117.6, 200); ax.plot(t, 5000*(t/326)**2, "--", color="0.4", label="Prescribed burner curve")
    ax.axvline(checkpoint, color="0.3", ls=":"); ax.set(xlim=(115.35,117.65), ylabel="HRR output-window mean (kW)", title="A. Same hotel checkpoint: dense output resolves a real numerical transient")
    ax.legend(fontsize=8, ncols=2)
    ax = axes[1]
    for name, color, label, ls in [("micro_continuous", blue, "Continuous", "-"), ("micro_periodic_resume", green, "Periodic restart", "--"), ("micro_stop_resume", orange, "Stop-file restart", "-")]:
        d = micro[name]; ax.plot(d.Time, d.HRR, color=color, label=label, ls=ls, lw=1.5)
    ax.axvline(3.0, color="0.3", ls=":"); ax.set(xlim=(2.6,4.0), ylabel="HRR output-window mean (kW)", title="B. Small fixture: stop-file path suppresses combustion, then overshoots")
    ax.legend(fontsize=8)
    ax = axes[2]
    for name, color, label in [("micro_periodic_resume", green, "Periodic restart - continuous"), ("micro_stop_resume", orange, "Stop-file restart - continuous")]:
        d = device_diffs[name]; ax.plot(d.Time, d.temperature_difference_C, color=color, label=label, lw=1.5)
    ax.axvline(3.0, color="0.3", ls=":"); ax.axhline(0, color="0.6", lw=.8)
    ax.set(xlim=(2.6,5.5), xlabel="Native fire time (s); each panel is a different local experiment", ylabel="Probe temperature difference (C)", title="C. Independent temperature observation confirms a state disturbance")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "restart_mechanism_controls.png", dpi=180)
    fig.savefig(OUT / "restart_mechanism_controls.svg")
    plt.close(fig)
    print(json.dumps({"status": results["status"], "solver_processes": len(solver_runs),
        "hotel_inferred_first_step_kW": step_HRR, "micro_comparisons": comparisons}, ensure_ascii=True))


if __name__ == "__main__":
    main()
