"""Analyze the clean run and paired checkpoint without altering native outputs."""
import csv
from datetime import datetime, timezone
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_safestep_clean_hotel120 import ROOT, OUT, PREFIX, OLD, sha, save
from analyze_safestep_restart_hrr_audit import checkpoint_metadata
from analyze_safestep_fds_smokecheck import parse_smv, controls


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        units = [x.strip() for x in next(csv.reader(stream))]
    data = pd.read_csv(path, skiprows=1)
    data.columns = [x.strip() for x in data.columns]
    assert data.Time.is_unique and np.all(np.diff(data.Time) > 0), path
    assert np.all(np.isfinite(data.to_numpy())), path
    assert len(units) == len(data.columns)
    return dict(zip(data.columns, units)), data


def interpolate(data, grid, columns):
    assert grid[0] >= data.Time.iloc[0] and grid[-1] <= data.Time.iloc[-1], "No extrapolation"
    return np.column_stack([np.interp(grid, data.Time, data[c]) for c in columns])


def average_window_integral(times, means, left, right):
    """Reconstruct energy with constant output-window means, no trapezoids.

    Whole windows use their recorded average. A boundary inside a window uses
    its average over the fractional width, which is explicitly an approximation.
    """
    times = np.asarray(times, dtype=float); means = np.asarray(means, dtype=float)
    assert len(times) == len(means) and np.all(np.diff(times) > 0)
    assert times[0] <= left < right <= times[-1]
    starts = times[:-1]; ends = times[1:]
    widths = np.maximum(0, np.minimum(ends, right)-np.maximum(starts, left))
    partial = bool(np.any((widths > 0) & (np.abs(widths-(ends-starts)) > 1e-9)))
    return float(np.dot(widths, means[1:])), partial


def verify_math():
    checks = []
    for name, times, means, a, b, expected, partial in [
        ("whole_native_average_windows", [0,2,3], [0,10,20], 0,3,40,False),
        ("fractional_boundary_explicitly_flagged", [0,2,3], [0,10,20], 1,2.5,20,True),
        ("irregular_interval_means_not_trapezoids", [0,.3,1.2], [0,7,4], 0,1.2,5.7,False),
    ]:
        value, flag = average_window_integral(times, means, a, b)
        assert abs(value-expected) < 1e-10 and flag == partial
        checks.append({"name": name, "passed": True, "energy_kJ": value, "partial_boundary": flag})
    try:
        average_window_integral([0,1], [0,1], 0,2)
    except AssertionError:
        checks.append({"name": "reject_energy_extrapolation", "passed": True})
    else:
        raise AssertionError("Extrapolation must be rejected")
    save(OUT / "energy_comparison_math_verification.json", {"status": "PASS", "checks": checks})


def main():
    assert not (OUT / "analysis_results.json").exists()
    assert (OUT / "execution_complete.json").exists()
    verify_math()
    plan = json.loads((OUT / "frozen_plan.json").read_text("utf-8"))
    snapshot = json.loads((OUT / "captured_checkpoint.json").read_text("utf-8"))
    continuous_dir = OUT / "runs" / PREFIX
    resumed_dir = OUT / "runs/hotel_clean_checkpoint_resume110"
    runs = [json.loads((p / "run_result.json").read_text("utf-8")) for p in (continuous_dir, resumed_dir)]
    assert all(r["passed"] and not r["stop_file_created"] and not r["timed_out"] for r in runs)
    cu, cd = read_table(continuous_dir / f"{PREFIX}_devc.csv")
    ru, rd = read_table(resumed_dir / f"{PREFIX}_devc.csv")
    _, ch = read_table(continuous_dir / f"{PREFIX}_hrr.csv")
    _, rh = read_table(resumed_dir / f"{PREFIX}_hrr.csv")
    assert cu == ru and list(cd.columns) == list(rd.columns)
    assert cd.Time.iloc[0] == rd.Time.iloc[0] == 0 and cd.Time.iloc[-1] == 120.0 and rd.Time.iloc[-1] == 110.0
    assert np.max(np.diff(cd.Time)) <= 1.0 and np.max(np.diff(rd.Time)) <= 1.0
    checkpoint = snapshot["fire_time_s"]
    before = cd[cd.Time <= checkpoint]
    retained = before.merge(rd, on="Time", suffixes=("_parent", "_resume"))
    columns = list(cd.columns[1:])
    lhs = retained[[c+"_parent" for c in columns]].to_numpy()
    rhs = retained[[c+"_resume" for c in columns]].to_numpy()
    exact_prefix = len(retained) == len(before) and np.array_equal(lhs, rhs)
    assert exact_prefix, "Restart must preserve the already completed device prefix"
    gas_columns = [c for c in cd.columns if re.fullmatch(r"(?:SS_R\d{2}|ROOM24)_(?:CO|CO2|HCN)", c)]
    assert len(gas_columns) == 177 and all(cu[c] == "mol/mol" for c in gas_columns)
    assert np.min(cd[gas_columns].to_numpy()) >= 0 and np.max(cd[gas_columns].to_numpy()) <= 1
    temperature_columns = [c for c in cd.columns if cu[c] == "C"]
    grid = np.arange(math.ceil((checkpoint+1e-8)*2)/2, 110.0+1e-8, .5)
    assert len(grid) > 2
    gate = plan["predeclared_engineering_screen"]; screens = []
    def screen(name, metric, limit, passed):
        screens.append({"name": name, "metric": metric, "limit": limit, "passed": bool(passed)})
    screen("pre_checkpoint_device_prefix_exact", len(retained), len(before), exact_prefix)
    hrr_ref = interpolate(ch, grid, ["HRR"])[:,0]
    hrr_res = interpolate(rh, grid, ["HRR"])[:,0]
    hrr_relative = np.abs(hrr_res-hrr_ref)/np.maximum(np.abs(hrr_ref), 1.0)
    screen("HRR_shape_relative_max", float(hrr_relative.max()), gate["HRR_interpolated_point_relative_error_max"],
           hrr_relative.max() <= gate["HRR_interpolated_point_relative_error_max"])
    common_native = np.intersect1d(ch.Time, rh.Time)
    common_native = common_native[(common_native > checkpoint+1.0) & (common_native <= 110.0)]
    energy_end = float(common_native[-1]) if len(common_native) else 110.0
    # Start at the preceding printed native HRR row. Correct its finite-precision
    # time to the exact saved value, then subtract saved partial Q_DOT_SUM.
    last_dump = snapshot["snapshot_metadata"][-1]["last_HRR_dump_time_s"]
    saved_energy = snapshot["snapshot_metadata"][-1]["Q_DOT_SUM_J"][0]/1000
    energies = {}; energy_partial = {}
    for label, data in [("continuous", ch), ("resumed", rh)]:
        start_index = int(np.argmin(np.abs(data.Time.to_numpy()-last_dump)))
        assert abs(float(data.Time.iloc[start_index])-last_dump) < 1e-4
        native_times = data.Time.to_numpy()[start_index:].copy(); native_times[0] = last_dump
        value, partial = average_window_integral(native_times, data.HRR.to_numpy()[start_index:], last_dump, energy_end)
        energies[label] = value-saved_energy; energy_partial[label] = partial
    relative_energy_error = abs(energies["resumed"]-energies["continuous"])/abs(energies["continuous"])
    screen("HRR_window_energy_relative", relative_energy_error, gate["HRR_window_energy_relative_error_max"],
           relative_energy_error <= gate["HRR_window_energy_relative_error_max"])
    gas_results = {}; max_gas_diffs = {}; gas_comparison_rows = []
    for gas in ("CO", "CO2", "HCN"):
        selected = [c for c in gas_columns if c.endswith("_"+gas)]
        reference = interpolate(cd, grid, selected)*1e6
        resumed = interpolate(rd, grid, selected)*1e6
        delta = resumed-reference
        peak = float(np.abs(reference).max()); floor = gate[f"{gas}_reference_peak_floor_ppm"]
        scale = max(peak, floor); maximum = float(np.abs(delta).max()); scaled = maximum/scale
        worst = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        subgroup = {}
        for group, selected_indices in [("corridor58", [i for i,c in enumerate(selected) if c.startswith("SS_R")]),
                                         ("room24", [i for i,c in enumerate(selected) if c.startswith("ROOM24")])]:
            group_peak = float(np.max(np.abs(reference[:,selected_indices])))
            group_error = float(np.max(np.abs(delta[:,selected_indices])))
            subgroup[group] = {"reference_peak_ppm": group_peak, "max_abs_difference_ppm": group_error,
                "peak_scaled_max_abs_difference": group_error/max(group_peak,floor),
                "diagnostic_only_not_added_to_frozen_gates": True}
        gas_results[gas] = {"channels": len(selected), "reference_peak_ppm": peak, "scale_floor_ppm": floor,
             "max_abs_difference_ppm": maximum, "peak_scaled_max_abs_difference": scaled,
             "RMSE_ppm": float(np.sqrt(np.mean(delta**2))), "worst_time_s": float(grid[worst[0]]),
             "worst_channel": selected[worst[1]], "subgroups": subgroup}
        max_gas_diffs[gas] = np.max(np.abs(delta), axis=1)/scale
        limit = gate[f"{gas}_peak_scaled_max_abs_error_max"]
        screen(f"{gas}_peak_scaled_max", scaled, limit, scaled <= limit)
        for j, channel in enumerate(selected):
            for i, native_time in enumerate(grid):
                gas_comparison_rows.append({"fire_time_s": native_time, "channel": channel,
                    "gas": gas, "continuous_ppm": reference[i,j], "resumed_ppm": resumed[i,j],
                    "difference_ppm": delta[i,j], "comparison_only_not_FED_authority": True})
    pd.DataFrame(gas_comparison_rows).to_csv(OUT / "paired_gas_comparison_grid.csv", index=False, encoding="utf-8-sig")
    t_ref = interpolate(cd, grid, temperature_columns)
    t_res = interpolate(rd, grid, temperature_columns)
    temperature_delta = t_res-t_ref
    maximum_T = float(np.abs(temperature_delta).max())
    worst_T = np.unravel_index(np.argmax(np.abs(temperature_delta)), temperature_delta.shape)
    screen("temperature_max_abs_C", maximum_T, gate["temperature_max_abs_error_C"], maximum_T <= gate["temperature_max_abs_error_C"])
    logical_ref = controls(continuous_dir / f"{PREFIX}_devc_ctrl_log.csv")
    logical_res = controls(resumed_dir / f"{PREFIX}_devc_ctrl_log.csv")
    _, geom_ref = parse_smv(continuous_dir / f"{PREFIX}.smv")
    _, geom_res = parse_smv(resumed_dir / f"{PREFIX}.smv")
    event_ids = {"SS_DOOR_OPEN", "SS_WINDOW_BREAK"}
    logical = lambda values: [(x["time_s"],x["ID"],x["state"]) for x in values if x["ID"] in event_ids and x["time_s"] <= 110]
    geometry = lambda values: [(x["kind"],x["mesh_index"],x["ID"],x["geometry_event_time_s"]) for x in values
                               if x["ID"] in ("SS_DOOR24", "SS_WINDOW24") and x["geometry_event_time_s"] <= 110]
    events_match = logical(logical_ref) == logical(logical_res) and geometry(geom_ref) == geometry(geom_res)
    screen("door_window_event_sequence_preserved", events_match, True, events_match)
    final_meta = [checkpoint_metadata(continuous_dir / f"{PREFIX}_{i}.restart", ijk, .5)
         for i, ijk in enumerate([(220,12,12),(7,67,12),(152,9,12),(17,40,12),(9,122,12),(53,21,17)], 1)]
    assert all(abs(m["fire_time_s"]-120) < 1e-6 for m in final_meta)
    assert final_meta[-1]["snapshot_fields"]["Q"]["nonzero_count"] > 0
    save(OUT / "normal_end120_checkpoint_metadata.json", {"meshes": final_meta, "six_checkpoint_times120s": True,
        "ROOM24_Q_field_nonzero": True, "no_stop_file_used": True})
    # Earlier stop/restart pilot is a separate diagnostic comparison, not the
    # primary reference for judging the passive checkpoint.
    old_dir = OLD / "runs/hotel_literature_pilot120_resume"; old_prefix = "hotel_literature_pilot120_v2"
    _, old_hrr = read_table(old_dir / f"{old_prefix}_hrr.csv")
    _, old_devices = read_table(old_dir / f"{old_prefix}_devc.csv")
    matched_old = ch.merge(old_hrr, on="Time", suffixes=("_clean", "_old"))
    old_prefix_rows = matched_old[matched_old.Time < 115.34]
    old_transition = matched_old[(matched_old.Time >= 115.0) & (matched_old.Time <= 117.0)]
    summary = {
        "status": "PASS_NUMERICAL_SCREEN_EXPLORATORY_PHYSICS" if all(x["passed"] for x in screens) else "NUMERICAL_SCREEN_HAS_FAILURES",
        "created_at": datetime.now(timezone.utc).isoformat(), "continuous_run": runs[0], "paired_run": runs[1],
        "normal_end120_without_stop": True, "captured_passive_checkpoint_s": checkpoint,
        "engineering_screens": screens, "all_screens_passed": all(x["passed"] for x in screens),
        "devices": {"continuous_rows": len(cd), "resumed_rows": len(rd), "channels_excluding_time": len(columns),
             "gas_channels": len(gas_columns), "temperature_channels": len(temperature_columns),
             "maximum_native_gap_continuous_s": float(np.max(np.diff(cd.Time))),
             "maximum_native_gap_resumed_s": float(np.max(np.diff(rd.Time))),
             "exact_retained_prefix_rows": len(retained)},
        "paired_grid": {"start_s": float(grid[0]), "end_s": float(grid[-1]), "step_s": .5, "points": len(grid),
             "linear_interpolation_only_for_paired_comparison": True, "extrapolation": False},
        "HRR": {"max_interpolated_relative_error": float(hrr_relative.max()),
             "max_interpolated_abs_error_kW": float(np.max(np.abs(hrr_res-hrr_ref))),
             "energy_start_checkpoint_s": checkpoint, "energy_end_s": energy_end,
             "energy_end_is_shared_native_output_time": bool(len(common_native)),
             "saved_pre_checkpoint_partial_energy_kJ": saved_energy, "reconstructed_energy_kJ": energies,
             "fractional_output_window_used": energy_partial, "relative_energy_error": relative_energy_error,
             "not_independent_energy_conservation_validation": True},
        "gases": gas_results, "temperature": {"max_abs_difference_C": maximum_T,
             "worst_time_s": float(grid[worst_T[0]]), "worst_channel": temperature_columns[worst_T[1]],
             "RMSE_C": float(np.sqrt(np.mean(temperature_delta**2)))},
        "controls": {"logical_continuous": logical_ref, "logical_resumed": logical_res,
             "geometry_continuous": geom_ref, "geometry_resumed": geom_res, "event_sequences_match": events_match,
             "checkpoint_after_door_event": True,
             "scope_caveat": "100s checkpoint is after the door transition; retained prefix checks event preservation, not a fresh crossing of the door event"},
        "earlier_stop_pilot_comparison": {"common_native_prefix_HRR_rows": len(old_prefix_rows),
             "max_abs_common_prefix_HRR_difference_kW": float(np.max(np.abs(old_prefix_rows.HRR_clean-old_prefix_rows.HRR_old))),
             "transition_common_rows": old_transition[["Time","HRR_clean","HRR_old"]].to_dict("records"),
             "old_stop_checkpoint_reused": False},
        "limits": ["Engineering thresholds were frozen before this run; they are not safety or calibration standards",
             "Only one hotel checkpoint and a roughly 10s continuation are compared",
             "Finite precision and interpolation limit pairwise conclusions; no bitwise physical equivalence is asserted",
             "Geometry, fuel, HCN transfer and window proxy assumptions remain unresolved",
             "All trajectories end before 300s evacuation start; no route FED or evacuation improvement claim"],
        "300s_runs": 0, "900s_runs": 0, "route_FED_computed": False, "training_updates": 0,
        "LLM_requests": 0, "sealed_final48_scenarios_accessed": False,
    }
    room_summary = {}; corridor_summary = {}
    for gas in ("CO", "CO2", "HCN"):
        factor = 100 if gas == "CO2" else 1e6
        room = cd["ROOM24_"+gas].to_numpy()*factor
        corridor = cd[[f"SS_R{i:02d}_{gas}" for i in range(1,59)]].to_numpy()*factor
        room_summary[gas] = {"max": float(room.max()), "unit": "percent" if gas == "CO2" else "ppm"}
        corridor_summary[gas] = {"max": float(corridor.max()), "unit": "percent" if gas == "CO2" else "ppm"}
    room_summary["temperature"] = {"max": float(cd.ROOM24_T.max()), "unit": "C"}
    summary["continuous_native_peak_summary"] = {"room24": room_summary, "corridor58": corridor_summary,
         "HRR_output_mean_peak_kW": float(ch.HRR.max()), "not_author_original_or_safety_validation": True}
    native_long = []
    for _, row in cd.iterrows():
        for region in range(1,59):
            native_long.append({"region": f"R{region:02d}", "fire_time_s": row.Time,
                "evacuation_elapsed_s": row.Time-300.0, "CO_ppm": row[f"SS_R{region:02d}_CO"]*1e6,
                "CO2_percent": row[f"SS_R{region:02d}_CO2"]*100, "HCN_ppm": row[f"SS_R{region:02d}_HCN"]*1e6,
                "status": "EXPLORATORY_PHYSICS_NOT_AUTHOR_ORIGINAL"})
    pd.DataFrame(native_long).to_csv(OUT/"clean120_native_gas_long.csv",index=False,encoding="utf-8-sig")
    save(OUT / "analysis_results.json", summary)
    save(OUT / "paired_controls.json", summary["controls"])
    plot_data = {"paired_grid_s": grid.tolist(), "paired_HRR_continuous_kW": hrr_ref.tolist(),
         "paired_HRR_resumed_kW": hrr_res.tolist(), "maximum_temperature_abs_difference_C": np.max(np.abs(temperature_delta),axis=1).tolist(),
         "peak_scaled_gas_max_abs_differences": {gas: value.tolist() for gas,value in max_gas_diffs.items()},
         "continuous_HRR_native": ch[["Time","HRR"]].to_dict("records"),
         "earlier_stop_HRR_native": old_hrr[["Time","HRR"]].to_dict("records"),
         "continuous_devices_for_overview": {"times": cd.Time.tolist(), "room_T_C": cd.ROOM24_T.tolist(),
              "room_CO_ppm": (cd.ROOM24_CO*1e6).tolist(), "room_HCN_ppm": (cd.ROOM24_HCN*1e6).tolist(),
              "corridor_CO_max_ppm": (cd[[f"SS_R{i:02d}_CO" for i in range(1,59)]].max(axis=1)*1e6).tolist(),
              "corridor_HCN_max_ppm": (cd[[f"SS_R{i:02d}_HCN" for i in range(1,59)]].max(axis=1)*1e6).tolist()}}
    save(OUT / "figure_data.json", plot_data)
    blue, green, orange = "#0072B2", "#009E73", "#D55E00"
    with plt.rc_context({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 9}):
        fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
        a = axes[0,0]; a.plot(grid,hrr_ref,color=blue,label="连续计算"); a.plot(grid,hrr_res,"--",color=green,label="正常断点续算")
        a.set(ylabel="HRR输出均值插值（kW）",title="A. 酒店模型：正常断点后的短时比较"); a.legend()
        a = axes[0,1]; a.plot(grid,np.max(np.abs(temperature_delta),axis=1),color=blue)
        a.axhline(2,color=orange,ls="--",label="预先冻结的2℃筛查值"); a.set(ylabel="最大温度绝对差（℃）",title="B. 所有温度通道的最大偏差"); a.legend()
        a = axes[1,0]
        for gas,color in [("CO",blue),("HCN",green),("CO2",orange)]:a.plot(grid,max_gas_diffs[gas]*100,color=color,label=gas)
        a.axhline(5,color="0.4",ls="--",label="预先冻结的5%筛查值")
        a.set(ylabel="按参考峰值与下限归一的偏差（%）",title="C. 177个气体通道的配对比较");a.legend()
        a=axes[1,1]; a.plot(ch.Time,ch.HRR,color=blue,label="本轮正常结束");a.plot(old_hrr.Time,old_hrr.HRR,"--",color=orange,label="旧停止文件续算")
        a.set(xlim=(114,120),ylabel="HRR输出区间均值（kW）",title="D. 原停止附近的轨迹差异");a.axvline(115.49119,color="0.5",ls=":");a.legend()
        for a in axes.flat:a.set(xlabel="火灾时间（秒）");a.grid(alpha=.15);a.spines[["top","right"]].set_visible(False)
        fig.suptitle("120秒酒店探索计算与正常断点续算；筛查通过不等于物理模型已验证",fontsize=12)
        fig.savefig(OUT/"hotel_clean_checkpoint_comparison.png",dpi=180);fig.savefig(OUT/"hotel_clean_checkpoint_comparison.svg");plt.close(fig)
        fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
        a=axes[0,0];a.plot(ch.Time,ch.HRR,color=blue,label="本轮正常结束HRR");a.plot(ch.Time,5000*(ch.Time/326)**2,"--",color=orange,label="规定增长曲线");a.legend();a.set(ylabel="输出区间均值（kW）",title="HRR与规定火源曲线")
        a=axes[0,1];a.plot(cd.Time,cd.ROOM24_T,color=orange);a.set(ylabel="房间测点气温（℃）",title="Room24气温；不是窗玻璃温度")
        for a,gas in [(axes[1,0],"CO"),(axes[1,1],"HCN")]:
            a.plot(cd.Time,cd["ROOM24_"+gas]*1e6,color=blue,label="房内测点")
            a.plot(cd.Time,cd[[f"SS_R{i:02d}_{gas}" for i in range(1,59)]].max(axis=1)*1e6,color=green,label="58区域逐时最大值")
            a.set(ylabel=f"{gas}（ppm，湿基）",title=f"{gas}早期传播");a.legend()
        for a in axes.flat:a.axvline(100,color="0.5",ls=":");a.set(xlabel="火灾时间（秒）",xlim=(0,120));a.grid(alpha=.15);a.spines[["top","right"]].set_visible(False)
        fig.suptitle("正常结束的120秒探索模型：HCN跨材料迁移及几何等假设仍未验证",fontsize=12)
        fig.savefig(OUT/"hotel_clean120_overview.png",dpi=180);fig.savefig(OUT/"hotel_clean120_overview.svg");plt.close(fig)
    print(json.dumps({"status": summary["status"], "screens": screens, "checkpoint_s": checkpoint,
         "HRR_relative_energy_error": relative_energy_error, "max_T_difference_C": maximum_T}, ensure_ascii=True))


if __name__ == "__main__":
    main()
