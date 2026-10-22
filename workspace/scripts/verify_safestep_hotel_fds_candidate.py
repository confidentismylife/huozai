"""Independent bounded checks for the static hotel draft and Eq3 interface.

All gas input is explicitly synthetic; no physical FDS output is generated.
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from safestep_gas_to_fer import convert_file, gas_to_fer, read_fds_gas_csv, validate_parameters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_hotel_fds_candidate_20261016"
CHECKS = []


def check(name, condition, detail=None):
    if not condition:
        raise AssertionError((name, detail))
    CHECKS.append({"name": name, "status": "PASS", "detail": detail})


def rejects(name, function):
    try:
        function()
    except ValueError as error:
        CHECKS.append({"name": name, "status": "PASS_REJECTED", "detail": str(error)})
    else:
        raise AssertionError("Failed to reject: " + name)


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def get(body, name):
    m = re.search(rf"\b{name}\s*=\s*'([^']*)'", body)
    return m.group(1) if m else None


def number(body, name):
    return float(re.search(rf"\b{name}\s*=\s*([+\-\d.eE]+)", body).group(1))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    assert not (OUT / "verification.json").exists(), "Do not overwrite a completed verification"
    spec = load(OUT / "frozen_specification.json")
    source_manifest = load(OUT / "source_manifest.json")
    for name, sha in {**source_manifest["protected_previous_sha256"], **source_manifest["input_sha256"]}.items():
        check("input_unchanged:" + name, digest(ROOT / name) == sha)
    text = (OUT / "safestep_hotel_candidate.fds.in").read_text("utf-8")
    records = [(m.group(1), m.group(2)) for m in re.finditer(r"&([A-Z][A-Z0-9_]*)\b([^/]*?)/", text, re.S)]
    valid = {"HEAD", "TIME", "MESH", "REAC", "DEVC", "SURF", "OBST", "HOLE", "VENT", "SLCF", "TAIL", "MISC", "DUMP", "RAMP"}
    check("all_generator_directives_removed", all(k in valid for k, _ in records))
    check("one_unresolved_HCN_token_no_invented_yield", text.count("{{REQUIRED_HCN_YIELD_KG_PER_KG}}") == 1)
    check("one_HEAD_one_TAIL", sum(k == "HEAD" for k, _ in records) == sum(k == "TAIL" for k, _ in records) == 1)
    check("exactly_six_inherited_mesh_blocks", [b.strip() for k, b in records if k == "MESH"] ==
          [m.group(1).strip() for m in re.finditer(r"&MESH\b([^/]*?)/", (ROOT/spec["source_template"]).read_text("utf-8"), re.S)])
    devices = {get(b, "ID"): b for k, b in records if k == "DEVC"}
    check("unique_all_device_IDs", len(devices) == sum(k == "DEVC" for k, b in records) == 416)
    gas_ids = {f"SS_R{r:02d}_{g}" for r in range(1, 59) for g in ("CO", "CO2", "HCN")}
    check("exactly_174_gas_devices", {i for i in devices if i.startswith("SS_R")} == gas_ids)
    gas_species = {"CO": "CARBON MONOXIDE", "CO2": "CARBON DIOXIDE", "HCN": "HYDROGEN CYANIDE"}
    for key in sorted(gas_ids):
        b = devices[key]
        check("gas_device:" + key, get(b, "QUANTITY") == "VOLUME FRACTION" and
              get(b, "SPEC_ID") == gas_species[key.rsplit("_", 1)[-1]] and
              "TIME_AVERAGED=.FALSE." in b and "DRY=.FALSE." in b)
    controlled_obsts = [b for k, b in records if k == "OBST" and get(b, "DEVC_ID")]
    check("one_whole_door_not_three_overlapping", len(controlled_obsts) == 1 and get(controlled_obsts[0], "DEVC_ID") == "SS_DOOR_OPEN")
    check("door_present_initially_then_removed_at_100s", number(devices["SS_DOOR_OPEN"], "SETPOINT") == 100 and "INITIAL_STATE=.TRUE." in devices["SS_DOOR_OPEN"])
    controlled_holes = [b for k, b in records if k == "HOLE" and get(b, "DEVC_ID")]
    check("one_controlled_window_hole", len(controlled_holes) == 1 and get(controlled_holes[0], "DEVC_ID") == "SS_WINDOW_BREAK")
    check("window_hole_created_when_gas_proxy_exceeds_150C", number(devices["SS_WINDOW_BREAK"], "SETPOINT") == 150 and
          "INITIAL_STATE=.FALSE." in devices["SS_WINDOW_BREAK"] and get(devices["SS_WINDOW_BREAK"], "QUANTITY") == "TEMPERATURE")
    check("external_OPEN_vent_preserved", len([b for k, b in records if k == "VENT" and get(b, "SURF_ID") == "OPEN"]) == 1)
    for kind, body in records:
        identifier = get(body, "DEVC_ID")
        if identifier:
            check("device_reference:" + identifier, identifier in devices)
    defined_surfs = {get(b, "ID") for k, b in records if k == "SURF"} | {"INERT", "OPEN"}
    for kind, body in records:
        for m in re.finditer(r"\bSURF_ID(?:S)?\s*=\s*((?:'[^']*'\s*,?\s*)+)", body):
            refs = re.findall(r"'([^']*)'", m.group(1))
            check("surface_reference", all(ref in defined_surfs for ref in refs), refs)
    check("no_old_fire_ramp_reference", "CURVE_fireramp" not in text and "RAMP_Q='SS_T2'" in text)
    ramp = [(number(b, "T"), number(b, "F")) for k, b in records if k == "RAMP"]
    check("328_monotone_ramp_points", len(ramp) == 328 and all(a[0] < b[0] for a, b in zip(ramp, ramp[1:])))
    check("growth_exact_at_integer_knots_with_format_tolerance", all(abs(f - min(t/326, 1)**2) < 5.1e-13 for t, f in ramp))
    check("peak_and_plateau_explicit", ramp[326] == (326, 1) and ramp[-1] == (900, 1))
    check("burner_area_times_HRRPUA_equals_5000kW", math.isclose((5.4-3.4)*(24.7-23.7)*2500, 5000))
    misc = next(b for k, b in records if k == "MISC")
    dump = next(b for k, b in records if k == "DUMP")
    check("ambient_and_output_contract", number(misc, "TMPA") == 20 and number(misc, "HUMIDITY") == 40 and number(dump, "DT_DEVC") == 5)
    mapping_ast = ast.parse((ROOT/spec["source_mapping"]).read_text("utf-8-sig"))
    mapping = next(ast.literal_eval(n.value) for n in mapping_ast.body if isinstance(n, ast.Assign))
    regions = csv_rows(OUT / "region_mapping_candidate.csv")
    check("region_set_1_to_58", {r["region"] for r in regions} == {f"R{i}" for i in range(1, 59)} and len(regions) == 58)
    cells = {tuple(xy): row["region"] for row in regions for xy in json.loads(row["RL_cells_json"])}
    check("all61_nonexit_author_cells_exactly_preserved", cells == {k: v for k, v in mapping.items() if v.startswith("R")} and len(cells) == 61)
    check("only_R56_R57_R58_shared", {r["region"] for r in regions if int(r["RL_cell_count"]) == 2} == {"R56", "R57", "R58"})
    check("every_region_mapping_labelled_inferred", all(r["mapping_status"] == "INFERRED_NOT_AUTHOR_CONFIRMED" for r in regions))
    screened = csv_rows(OUT / "region_point_geometry_screen.csv")
    check("all58_raw_coordinates_in_mesh_clear_of_raw_solids", len(screened) == 58 and all(r["screen_result"] == "RAW_POINT_CLEAR" for r in screened))
    for row in regions:
        region_no = int(row["region"][1:])
        for gas in ("CO", "CO2", "HCN"):
            body = devices[f"SS_R{region_no:02d}_{gas}"]
            xyz = [float(x) for x in re.search(r"XYZ=([\d.,]+)", body).group(1).rstrip(",").split(",")]
            check("device_matches_region_coordinates", xyz == [float(row[k]) for k in ("x_m", "y_m", "z_m")])
    # Synthetic, explicit convention for arithmetic tests. These are NOT original V/D.
    params = {"ventilation_L_per_min": 30.0, "COHb_incapacitation_percent": 30.0,
              "equation_rate_time_unit": "minute", "formula_concentration_units": {"CO": "ppm", "CO2": "percent", "HCN": "ppm"},
              "convention_status": "synthetic_only", "parameter_provenance": "Arbitrary numbers for unit and arithmetic tests only",
              "evacuation_start_fire_time_s": 300.0}
    unit_ppm = {"CO": "ppm", "CO2": "percent", "HCN": "ppm"}
    fixture = {"CO": 1000, "CO2": 5, "HCN": 50}
    result = gas_to_fer(fixture, unit_ppm, params)
    expected_co = (3.317e-5 * math.pow(1000, 1.036) * math.e) / 60
    expected_hcn = (math.pow(50, 2.36) * math.e / 1.2e6) / 60
    check("synthetic_arithmetic_reference", math.isclose(result["FER_CO_per_s"], expected_co, rel_tol=1e-12) and math.isclose(result["FER_HCN_per_s"], expected_hcn, rel_tol=1e-12), result)
    volume_result = gas_to_fer({"CO": .001, "CO2": .05, "HCN": .00005}, dict.fromkeys(gas_species, "mol/mol"), params)
    check("ppm_percent_fraction_equivalence", all(math.isclose(result[k], volume_result[k], rel_tol=1e-12) for k in result))
    second_result = gas_to_fer(fixture, unit_ppm, {**params, "equation_rate_time_unit": "second"})
    check("explicit_minute_second_factor60", math.isclose(second_result["FER_total_per_s"], 60*result["FER_total_per_s"]))
    zero = gas_to_fer(dict.fromkeys(gas_species, 0), unit_ppm, params)
    check("zero_toxic_gases_zero_rate", zero["FER_total_per_s"] == 0)
    no_co2 = gas_to_fer({**fixture, "CO2": 0}, unit_ppm, params)
    check("5percent_CO2_multiplier_e", math.isclose(result["FER_total_per_s"] / no_co2["FER_total_per_s"], math.e))
    double_v = gas_to_fer(fixture, unit_ppm, {**params, "ventilation_L_per_min": 60})
    check("V_scales_CO_only", math.isclose(double_v["FER_CO_per_s"], 2*result["FER_CO_per_s"]) and double_v["FER_HCN_per_s"] == result["FER_HCN_per_s"])
    rejects("missing_V_rejected", lambda: gas_to_fer(fixture, unit_ppm, {**params, "ventilation_L_per_min": None}))
    rejects("zero_D_rejected", lambda: gas_to_fer(fixture, unit_ppm, {**params, "COHb_incapacitation_percent": 0}))
    rejects("unspecified_time_basis_rejected", lambda: gas_to_fer(fixture, unit_ppm, {**params, "equation_rate_time_unit": None}))
    rejects("missing_HCN_not_zero", lambda: gas_to_fer({"CO": 1000, "CO2": 5}, unit_ppm, params))
    rejects("negative_gas_rejected", lambda: gas_to_fer({**fixture, "HCN": -1}, unit_ppm, params))
    rejects("nonfinite_gas_rejected", lambda: gas_to_fer({**fixture, "HCN": float("nan")}, unit_ppm, params))
    rejects("mass_fraction_not_volume_fraction", lambda: gas_to_fer(fixture, {**unit_ppm, "CO": "kg/kg"}, params))
    rejects("fraction_above_one_rejected", lambda: gas_to_fer({**fixture, "CO2": 101}, unit_ppm, params))
    rejects("incomplete_real_parameter_template_rejected", lambda: validate_parameters(load(OUT / "fer_parameters_REQUIRED.json")))
    tests = OUT / "synthetic_interface_checks"
    tests.mkdir(exist_ok=False)
    save(tests / "synthetic_parameters_NOT_ORIGINAL.json", params)
    gas_header = ["Time"] + sorted(gas_ids)
    input_path = tests / "SYNTHETIC_ONLY_devc.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s"] + ["mol/mol"] * len(gas_ids))
        writer.writerow(gas_header)
        for time in range(0, 901, 5):
            # Same constant fixture in each region, used for end-to-end data plumbing.
            writer.writerow([time] + [{"CO": .001, "CO2": .05, "HCN": .00005}[key.rsplit("_", 1)[-1]] for key in gas_header[1:]])
    metadata = convert_file(input_path, params, tests / "synthetic_export_NOT_PHYSICAL")
    exported = csv_rows(tests / "synthetic_export_NOT_PHYSICAL/fer_long.csv")
    check("full_synthetic58x181_export", metadata["output_rows"] == len(exported) == 10498)
    check("two_time_clocks_preserved", float(exported[0]["evacuation_elapsed_s"]) == -300 and float(exported[-1]["evacuation_elapsed_s"]) == 600)
    check("synthetic_export_value_not_just_row_count", all(math.isclose(float(row["FER_total_per_s"]), result["FER_total_per_s"], rel_tol=1e-12) for row in exported))
    # A constant rate over a known 10 s duration, independent reference integration.
    check("synthetic_constant_rate_10s_integral", math.isclose(sum([result["FER_total_per_s"]*5]*2), (expected_co+expected_hcn)*10))
    missing_path = tests / "SYNTHETIC_MISSING_HCN.csv"
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        keep = [key for key in gas_header if key != "SS_R01_HCN"]
        writer.writerow(["s"] + ["mol/mol"]*(len(keep)-1))
        writer.writerow(keep)
        writer.writerow([0]*len(keep))
    rejects("missing_FDS_HCN_column_rejected", lambda: read_fds_gas_csv(missing_path))
    # Explicitly reject incomplete/irregular timestamps before any output creation.
    short_path = tests / "SYNTHETIC_SHORT_TIME_GRID.csv"
    lines = input_path.read_text("utf-8").splitlines()
    short_path.write_text("\n".join(lines[:4]) + "\n", "utf-8")
    rejects("incomplete_time_grid_rejected_no_resampling", lambda: convert_file(short_path, params, tests/"must_not_exist"))
    check("failed_export_did_not_create_directory", not (tests/"must_not_exist").exists())
    # Repeat preservation audit after all calculations.
    protected = source_manifest["protected_previous_sha256"]
    check("previous102_inputs_unchanged_after_checks", all(digest(ROOT/name) == sha for name, sha in protected.items()), len(protected))
    save(OUT / "verification.json", {
        "status": "PASS_STATIC_AND_SYNTHETIC_INTERFACE_CHECKS_NOT_PHYSICAL_VALIDATION",
        "created_at": datetime.now(timezone.utc).isoformat(), "checks": CHECKS,
        "check_count": len(CHECKS), "physical_regions": 58, "gas_devices": 174,
        "protected_previous_files_unchanged": len(protected),
        "synthetic_export_rows": 10498,
        "ramp_linear_interpolation_max_HRR_error_kW": 5000 / (4 * 326**2),
        "ramp_growth_integral_error_kJ": 5000 / (6 * 326),
        "FDS_parser_or_initialization_runs": 0, "physical_FDS_runs": 0,
        "training_updates": 0, "LLM_requests": 0, "small_case62percent_reproduced": False,
        "figure_review": "Original figure visually inspected; bottom-corridor labels moved below points in v2 to avoid junction label collisions",
        "remaining_blockers": load(OUT/"readiness.json"),
    })
    print(json.dumps({"status": "PASS", "checks": len(CHECKS), "synthetic_rows": 10498, "physical_FDS_runs": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()
