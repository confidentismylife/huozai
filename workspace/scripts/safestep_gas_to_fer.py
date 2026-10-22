"""Explicit-unit SafeStep Eq. 3 interface, independent of FDS's built-in FED.

This implements an equation convention, not a validated human-toxicity model.
No default V, D, coefficient time basis, missing HCN, interpolation or fitting.
The original paper's implementation/units remain to be independently confirmed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

SPECIES = ("CO", "CO2", "HCN")
FORMULA_UNITS = {"CO": "ppm", "CO2": "percent", "HCN": "ppm"}
SCALE_TO_FRACTION = {"mol/mol": 1.0, "ppm": 1e-6, "percent": 0.01}


def finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: explicit numeric value required")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name}: invalid range")
    return float(value)


def validate_parameters(parameters: Mapping):
    v = finite_number(parameters.get("ventilation_L_per_min"), "V", positive=True)
    d = finite_number(parameters.get("COHb_incapacitation_percent"), "D", positive=True)
    if d > 100:
        raise ValueError("D is a percent numeric value in (0,100], not a fraction")
    basis = parameters.get("equation_rate_time_unit")
    if basis not in ("minute", "second"):
        raise ValueError("equation_rate_time_unit must be explicitly minute or second")
    if parameters.get("formula_concentration_units") != FORMULA_UNITS:
        raise ValueError("This interface uses CO/HCN ppm and CO2 percent; confirm this convention explicitly")
    if parameters.get("convention_status") not in ("synthetic_only", "reconstruction_assumption", "original_verified"):
        raise ValueError("Explicit convention_status required")
    if not isinstance(parameters.get("parameter_provenance"), str) or not parameters["parameter_provenance"].strip():
        raise ValueError("Nonempty parameter_provenance required")
    if parameters["convention_status"] == "original_verified" and not parameters.get("original_verification_reference"):
        raise ValueError("original_verified requires an original_verification_reference")
    return v, d, 60.0 if basis == "minute" else 1.0


def gas_to_fer(concentrations: Mapping, input_units: Mapping, parameters: Mapping):
    v, d, seconds_per_formula_unit = validate_parameters(parameters)
    fractions = {}
    for gas in SPECIES:
        value = finite_number(concentrations.get(gas), gas)
        unit = input_units.get(gas)
        if unit not in SCALE_TO_FRACTION:
            raise ValueError(f"{gas}: unsupported/missing unit {unit!r}; mass fraction is not volume fraction")
        fractions[gas] = value * SCALE_TO_FRACTION[unit]
        if fractions[gas] > 1.0:
            raise ValueError(f"{gas}: volume fraction above one")
    if sum(fractions.values()) > 1.0 + 1e-12:
        raise ValueError("Sum of provided gas volume fractions exceeds one")
    co_ppm = fractions["CO"] * 1e6
    hcn_ppm = fractions["HCN"] * 1e6
    co2_percent = fractions["CO2"] * 100.0
    hyperventilation = math.exp(co2_percent / 5.0)
    co = 3.317e-5 * co_ppm ** 1.036 * v / d * hyperventilation
    hcn = hcn_ppm ** 2.36 / 1.2e6 * hyperventilation
    result = {
        "CO_ppm": co_ppm, "CO2_percent": co2_percent, "HCN_ppm": hcn_ppm,
        "CO2_multiplier": hyperventilation,
        "FER_CO_per_s": co / seconds_per_formula_unit,
        "FER_HCN_per_s": hcn / seconds_per_formula_unit,
        "FER_total_per_s": (co + hcn) / seconds_per_formula_unit,
    }
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError("Non-finite conversion result")
    return result


def read_fds_gas_csv(path: Path, *, region_count=58):
    """Read one FDS device CSV with its unit row; reject ambiguous/incomplete data.

    FDS may split device CSVs at high device counts. This candidate has fewer
    than 1,000 devices; split files must be explicitly joined and audited first.
    Accept the usual leading whitespace and quote formatting, never fill NaNs.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            units = [x.strip().strip('"') for x in next(reader)]
            columns = [x.strip().strip('"') for x in next(reader)]
        except StopIteration:
            raise ValueError("FDS unit/header rows missing") from None
        if len(columns) != len(units) or len(set(columns)) != len(columns):
            raise ValueError("Header length mismatch or duplicate columns")
        if not columns or columns[0] != "Time" or units[0] != "s":
            raise ValueError("Expected FDS Time column in seconds")
        indices = {}
        for i in range(1, region_count + 1):
            for gas in SPECIES:
                key = f"SS_R{i:02d}_{gas}"
                if key not in columns:
                    raise ValueError(f"Missing gas device {key}; missing HCN is not zero")
                j = columns.index(key)
                if units[j] != "mol/mol":
                    raise ValueError(f"{key}: expected wet volume fraction mol/mol, got {units[j]!r}")
                indices[(i, gas)] = j
        times, values = [], []
        for line_no, row in enumerate(reader, start=3):
            if not row or all(not v.strip() for v in row):
                continue
            if len(row) != len(columns):
                raise ValueError(f"Line {line_no}: column count mismatch")
            time_s = float(row[0])
            if not math.isfinite(time_s) or time_s < 0 or (times and time_s <= times[-1]):
                raise ValueError(f"Line {line_no}: times must be finite, nonnegative and strictly increasing")
            sample = {}
            for key, j in indices.items():
                try:
                    value = float(row[j])
                except ValueError:
                    raise ValueError(f"Line {line_no}: gas value is not numeric") from None
                sample[key] = finite_number(value, f"line {line_no}, {key}")
            times.append(time_s)
            values.append(sample)
    if not times:
        raise ValueError("No data rows")
    return times, values


def convert_file(input_path: Path, parameters: Mapping, output_directory: Path):
    validate_parameters(parameters)
    times, samples = read_fds_gas_csv(input_path)
    # Strict export contract for this hotel candidate. No interpolation.
    if len(times) != 181 or any(abs(t - i * 5) > 1e-6 for i, t in enumerate(times)):
        raise ValueError("Expected complete 0..900 s at 5 s intervals; no silent resampling")
    evacuation_start = finite_number(parameters.get("evacuation_start_fire_time_s"), "evacuation_start_fire_time_s")
    if evacuation_start != 300:
        raise ValueError("This hotel interface expects evacuation start at fire time 300 s")
    converted = []
    for t, sample in zip(times, samples):
        for i in range(1, 59):
            values = gas_to_fer({g: sample[(i, g)] for g in SPECIES},
                                {g: "mol/mol" for g in SPECIES}, parameters)
            converted.append({"region": f"R{i}", "fire_time_s": t,
                              "evacuation_elapsed_s": t - evacuation_start, **values})
    output_directory.mkdir(parents=True, exist_ok=False)
    target = output_directory / "fer_long.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    metadata = {
        "label": "EQUATION_INTERFACE_EXPORT_NOT_AUTHOR_ORIGINAL_FER",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "parameters": dict(parameters),
        "physical_regions": 58, "time_points": len(times), "output_rows": len(converted),
        "negative_evacuation_elapsed_s": "retained to identify pre-evacuation data",
        "automatic_resampling": False, "built_in_FDS_FED_used": False,
        "original_physical_validation": False,
    }
    (output_directory / "conversion_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text("utf-8-sig"))
    print(json.dumps(convert_file(args.input, parameters, args.output_directory), ensure_ascii=True))


if __name__ == "__main__":
    main()
