"""Meaningful analytic and failure-mode tests of the native-time contract."""
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from safestep_native_time import NativeRateCurve, route_exposure
from safestep_gas_to_fer import gas_to_fer, validate_parameters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_physics_time_contract_20261016"
CHECKS = []


def check(name, passed, detail=None):
    assert passed, (name, detail)
    CHECKS.append({"name": name, "passed": True, "detail": detail})


def rejects(name, fn):
    try:
        fn()
    except ValueError as error:
        CHECKS.append({"name": name, "passed": True, "rejected_with": str(error)})
    else:
        raise AssertionError(name)


def main():
    assert not (OUT/"native_time_verification.json").exists()
    c = NativeRateCurve([0,.37,1.3,2], [2,3.11,5.9,8], 1.0)
    check("nonuniform_linear_rate_matches_analytic_0to2_integral", math.isclose(c.integrate(0,2), 10, rel_tol=1e-12))
    check("off_node_interval_boundaries_split_exactly", math.isclose(c.integrate(.2,1.4), 5.28, rel_tol=1e-12))
    check("splitting_an_interval_conserves_integral", math.isclose(c.integrate(.2,1.4), c.integrate(.2,.8)+c.integrate(.8,1.4), rel_tol=1e-12))
    check("zero_duration_has_zero_exposure", c.integrate(.8,.8) == 0)
    # A spike lies between 5s sample points; native integration must retain it.
    spike = NativeRateCurve([0,1,5], [0,1,0], 4.0)
    grid_t, grid_r = spike.regular_samples(start_s=0, end_s=5, step_s=5)
    coarse_integral = float(np.sum(np.diff(grid_t)*(grid_r[:-1]+grid_r[1:])/2))
    check("native_integration_preserves_spike_missed_by5s_grid", spike.integrate(0,5) == 2.5 and coarse_integral == 0,
          {"synthetic_native_integral":2.5, "incorrect_coarse_grid_integral":0})
    a = NativeRateCurve([0,1,2], [0,2,0], 1.0)
    b = NativeRateCurve([0,.6,1.4,2], [10,10,10,10], 1.0)
    check("route_change_uses_correct_region_and_time", math.isclose(route_exposure({"A":a,"B":b},[("A",0,1),("B",1,2)]), 11))
    rejects("gap_in_route_not_silently_waiting", lambda: route_exposure({"A":a},[("A",0,.5),("A",.6,1)]))
    rejects("overlap_in_route_rejected", lambda: route_exposure({"A":a},[("A",0,1),("A",.5,1.5)]))
    rejects("future300s_not_covered_by120s_pilot", lambda: c.integrate(300,310))
    rejects("extrapolate_left_rejected", lambda: c.sample([-.1]))
    rejects("extrapolate_right_rejected", lambda: c.sample([2.1]))
    rejects("reversed_interval_rejected", lambda: c.integrate(1,.5))
    rejects("duplicate_native_times_rejected", lambda: NativeRateCurve([0,1,1],[0,1,2],1))
    rejects("unsorted_native_times_rejected", lambda: NativeRateCurve([0,2,1],[0,1,2],2))
    rejects("large_native_data_gap_rejected", lambda: NativeRateCurve([0,.5,2],[0,1,2],1))
    rejects("NaN_native_rate_rejected", lambda: NativeRateCurve([0,1],[0,float('nan')],1))
    rejects("negative_native_rate_rejected", lambda: NativeRateCurve([0,1],[0,-1],1))
    rejects("partial_final_regular_interval_rejected", lambda: c.regular_samples(start_s=0,end_s=1.7,step_s=1))
    p = json.loads((OUT/"exploratory_human_reference_NOT_AUTHOR_ORIGINAL.json").read_text("utf-8"))
    validate_parameters(p)
    fixture = {"CO":1000.0,"CO2":0.0,"HCN":0.0}
    result = gas_to_fer(fixture, {"CO":"ppm","CO2":"percent","HCN":"ppm"}, p)
    expected = 3.317e-5*1000**1.036*(25/30)/60
    check("reference_V25_D30_with_minute_to_second_conversion", math.isclose(result["FER_total_per_s"],expected,rel_tol=1e-12))
    # Two declared interpolation models differ under a nonlinear gas->rate law.
    exponent=2.36
    native_rate_linear = (100**exponent+300**exponent)/(2*1.2e6*60)
    linear_gas_then_rate = (300**(exponent+1)-100**(exponent+1))/(200*(exponent+1)*1.2e6*60)
    check("nonlinear_concentration_and_rate_interpolation_not_interchangeable", native_rate_linear > linear_gas_then_rate,
          {"synthetic_linear_rate_integral":native_rate_linear, "synthetic_linear_gas_integral":linear_gas_then_rate,
           "relative_difference":native_rate_linear/linear_gas_then_rate-1})
    data = {"status":"PASS_NATIVE_TIME_CONTRACT_SYNTHETIC_ANALYTIC_CHECKS", "checks":CHECKS, "check_count":len(CHECKS),
            "physical_exposure_conclusions":False, "all_numbers_in_tests_are_synthetic":True,
            "integration_authority":"native_nodes_and_declared_piecewise_linear_rate_model", "resampled5s_table_authority":False}
    (OUT/"native_time_verification.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")
    print(json.dumps({"status":"PASS","checks":len(CHECKS)},ensure_ascii=True))


if __name__=="__main__":
    main()
