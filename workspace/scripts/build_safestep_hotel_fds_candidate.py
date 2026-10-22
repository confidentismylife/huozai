"""Build a separately versioned, intentionally incomplete hotel FDS input draft.

Only local text/metadata/geometry processing. No FDS, training, model calls,
downloads, executable author code, or access to sealed crowd scenarios.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "schemas/safestep_hotel_fds_candidate_20261016.json"
GENERATOR_NAMES = {"MAIN", "PFSL", "PHRC", "PPSD", "IDWT", "MRND", "GDWT", "IFTD", "IMHR", "GFTD", "GMHR", "GHRC", "IFSL", "GFSL"}
FDS_NAMES = {"HEAD", "TIME", "MESH", "REAC", "DEVC", "SURF", "OBST", "HOLE", "VENT", "SLCF", "TAIL", "MISC", "DUMP", "RAMP"}


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


def blocks(text):
    # This template has no slash inside a quoted value. Not a general FDS parser.
    return [(m.group(1).upper(), m.group(2).strip())
            for m in re.finditer(r"&([A-Z][A-Z0-9_]*)\b([^/]*?)/", text, re.S | re.I)]


def get_string(body, key):
    match = re.search(rf"\b{key}\s*=\s*'([^']*)'", body, re.I)
    return match.group(1) if match else None


def get_numbers(body, key, count):
    match = re.search(rf"\b{key}\s*=\s*([+\-\d.Ee,\s]+)", body)
    if not match:
        return None
    vals = [float(x.strip()) for x in match.group(1).strip().rstrip(",").split(",")]
    assert len(vals) == count, (key, body, vals)
    return vals


def inside(point, xb, tolerance=1e-8):
    return all(xb[2*a] - tolerance <= point[a] <= xb[2*a+1] + tolerance for a in range(3))


def overlaps(a, b):
    return all(min(a[2*i+1], b[2*i+1]) > max(a[2*i], b[2*i]) + 1e-8 for i in range(3))


def get_mapping(path):
    tree = ast.parse(path.read_text("utf-8-sig"))
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mapping" for t in n.targets))


def propose_locations(mapping, source_blocks):
    t1 = [(get_string(b, "ID"), get_numbers(b, "XYZ", 3)) for k, b in source_blocks
          if k == "DEVC" and (get_string(b, "ID") or "").startswith("T1_")]
    assert len(t1) == 59  # 58 corridor locations plus Room24
    rows = []
    for region in range(1, 59):
        if region <= 21:
            x, y = 2.0 * region, 27.1
        elif region <= 36:
            x, y = 12.0 + 2 * (region - 22), 12.1
        elif region <= 43:
            x, y = 10.2, 13.1 + 2 * (region - 37)
        elif region <= 50:
            x, y = 33.2, 13.1 + 2 * (region - 44)
        elif region <= 55:
            x, y = 10.2, 11.1 - 2 * (region - 51)
        else:
            x, y = 22.6, {56: 14.7, 57: 17.1, 58: 19.7}[region]
        anchors = [identifier for identifier, xyz in t1 if math.hypot(x - xyz[0], y - xyz[1]) < 1e-8]
        assert len(anchors) == 1, (region, anchors)
        cells = sorted(list(xy) for xy, label in mapping.items() if label == f"R{region}")
        assert len(cells) == (2 if region >= 56 else 1)
        rows.append({"region": f"R{region}", "x_m": x, "y_m": round(y, 4), "z_m": 2.0,
                     "RL_cells_json": json.dumps(cells), "RL_cell_count": len(cells),
                     "template_anchor_ID": anchors[0], "template_anchor_height_m": 0.6,
                     "mapping_status": "INFERRED_NOT_AUTHOR_CONFIRMED"})
    return rows


def geometry_screen(rows, candidate_blocks):
    meshes = [{"ID": get_string(b, "ID"), "XB": get_numbers(b, "XB", 6), "IJK": get_numbers(b, "IJK", 3)}
              for k, b in candidate_blocks if k == "MESH"]
    obsts = [(idx, get_string(b, "ID"), get_numbers(b, "XB", 6)) for idx, (k, b) in enumerate(candidate_blocks) if k == "OBST"]
    holes = [(idx, get_string(b, "ID"), get_numbers(b, "XB", 6), get_string(b, "DEVC_ID"))
             for idx, (k, b) in enumerate(candidate_blocks) if k == "HOLE"]
    screened = []
    for r in rows:
        xyz = [r[k] for k in ("x_m", "y_m", "z_m")]
        mesh_hits = [m for m in meshes if inside(xyz, m["XB"])]
        solid_hits = [idx for idx, _, box in obsts if inside(xyz, box)]
        open_holes = [idx for idx, _, box, control in holes if not control and inside(xyz, box)]
        on_faces = []
        for m in mesh_hits:
            for a, axis in enumerate("xyz"):
                index = (xyz[a] - m["XB"][2*a]) / ((m["XB"][2*a+1] - m["XB"][2*a]) / m["IJK"][a])
                if abs(index - round(index)) < 1e-7:
                    on_faces.append(m["ID"] + ":" + axis)
        screened.append({**r, "containing_meshes": "|".join(m["ID"] for m in mesh_hits),
                         "raw_solid_hits": len(solid_hits), "permanent_hole_hits": len(open_holes),
                         "mesh_grid_plane_axes": "|".join(on_faces),
                         "screen_result": "RAW_POINT_CLEAR" if mesh_hits and not solid_hits else "REVIEW_GEOMETRY"})
    inventory = []
    for idx, identifier, xb in obsts:
        inventory.append({"candidate_block_index": idx, "ID": identifier, "XB": xb,
                          "positive_volume_mesh_overlap": [m["ID"] for m in meshes if overlaps(xb, m["XB"])],
                          "zero_thickness": any(abs(xb[2*a+1] - xb[2*a]) < 1e-8 for a in range(3))})
    return meshes, screened, inventory


def derive_input(source_blocks, rows, config):
    paper = config["paper_confirmed"]
    assumptions = config["candidate_assumptions"]
    output, edits = [], []
    for index, (kind, body) in enumerate(source_blocks):
        identifier = get_string(body, "ID")
        if kind in GENERATOR_NAMES:
            edits.append({"source_block_index": index, "action": "remove_generator_directive", "kind": kind})
            continue
        assert kind in FDS_NAMES, kind
        if kind == "HEAD":
            body = "CHID='safestep_hotel_candidate', TITLE='Unvalidated reconstruction; required HCN yield is unresolved'"
        elif kind == "REAC":
            body += ", HCN_YIELD={{REQUIRED_HCN_YIELD_KG_PER_KG}}"
        elif kind == "DEVC" and identifier in ("dot_timer1", "dot_timer2", "dot_timer3"):
            edits.append({"source_block_index": index, "action": "replace_three_timers_with_one_100s_timer", "kind": kind})
            continue
        elif kind == "OBST" and identifier in ("door24_1", "door24_2", "door24_3"):
            edits.append({"source_block_index": index, "action": "replace_three_overlapping_doors_with_one", "kind": kind})
            continue
        elif kind == "SURF" and identifier == "CURVE":
            body = "ID='SS_FIRE', HRRPUA=2500.0, RAMP_Q='SS_T2', COLOR='RED'"
        elif kind == "OBST" and identifier == "fire source":
            body = "ID='SS_BURNER', XB=" + ",".join(str(x) for x in assumptions["burner_XB_m"]) + ", SURF_IDS='SS_FIRE','INERT','INERT'"
        elif kind == "HOLE" and get_numbers(body, "XB", 6) == [-0.25, 0.0, 23.5, 25.0, 1.3, 2.8]:
            body = "ID='SS_ROOM24_WINDOW', XB=-0.25,0.0,23.5,25.0,1.3,2.8, DEVC_ID='SS_WINDOW_BREAK', COLOR='BLUE'"
        elif kind == "TAIL":
            continue
        if body != source_blocks[index][1]:
            edits.append({"source_block_index": index, "action": "replace_or_extend_block", "kind": kind,
                          "before": source_blocks[index][1], "after": body})
        output.append((kind, body))
    output += [
        ("MISC", "TMPA=20.0, HUMIDITY=40.0"),
        ("DUMP", "DT_DEVC=5.0, DT_HRR=5.0"),
        ("DEVC", "ID='SS_DOOR_OPEN', QUANTITY='TIME', XYZ=10.2,23.7,1.0, SETPOINT=100.0, INITIAL_STATE=.TRUE., LATCH=.TRUE."),
        ("OBST", "ID='SS_DOOR24', XB=9.4,9.6,23.15,24.3,0.0,2.0, SURF_ID='INERT', DEVC_ID='SS_DOOR_OPEN'"),
        ("DEVC", "ID='SS_WINDOW_BREAK', QUANTITY='TEMPERATURE', XYZ=0.2,24.3,2.0, SETPOINT=150.0, INITIAL_STATE=.FALSE., LATCH=.TRUE., TIME_AVERAGED=.FALSE."),
    ]
    ramp = []
    for time in list(range(327)) + [900]:
        factor = min(time / paper["time_to_peak_s"], 1.0) ** 2
        ramp.append({"fire_time_s": time, "factor": factor, "specified_HRR_kW": factor * paper["peak_HRR_kW"]})
        output.append(("RAMP", f"ID='SS_T2', T={time:.1f}, F={factor:.12f}"))
    gases = {"CO": "CARBON MONOXIDE", "CO2": "CARBON DIOXIDE", "HCN": "HYDROGEN CYANIDE"}
    device_blocks = []
    for r in rows:
        region = int(r["region"][1:])
        for short, species in gases.items():
            body = (f"ID='SS_R{region:02d}_{short}', XYZ={r['x_m']:.4f},{r['y_m']:.4f},{r['z_m']:.4f}, "
                    f"QUANTITY='VOLUME FRACTION', SPEC_ID='{species}', TIME_AVERAGED=.FALSE., DRY=.FALSE.")
            output.append(("DEVC", body))
            device_blocks.append(("DEVC", body))
    output.append(("TAIL", ""))
    header = ("! STATIC RECONSTRUCTION CANDIDATE, NOT AN ORIGINAL SAFESTEP INPUT.\n"
              "! Derived from NIST FD-Gen example_commercial.fds; see bundled FD-Gen_LICENSE.md.\n"
              "! Modified 2026-10-16: fixed fire/control assumptions and added gas devices.\n"
              "! REQUIRED_HCN_YIELD token intentionally prevents accidental execution.\n"
              "! See parameter_register.csv, readiness.json and the Chinese report.\n"
              "! Geometry, burner, post-peak curve, window proxy and R mapping are unverified.\n")
    serialize = lambda records: "\n".join(f"&{k} {b} /" for k, b in records) + "\n"
    return header + serialize(output), serialize(device_blocks), ramp, edits, output


def render_figure(out, rows, mapping, meshes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    font = "Microsoft YaHei"
    with plt.rc_context({"font.family": font, "axes.unicode_minus": False, "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
        ax = axes[0]
        for m in meshes:
            x1, x2, y1, y2, _, _ = m["XB"]
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fc="#EDF1F5", ec="#7A8591", lw=1))
        for r in rows:
            ax.plot(r["x_m"], r["y_m"], "o", ms=3, color="#0072B2")
            label = r["region"][1:]
            dx, dy = ((0.35, 0) if int(label) >= 37 else (0, -0.7) if int(label) >= 22 else (0, 0.55))
            ax.text(r["x_m"] + dx, r["y_m"] + dy, label, fontsize=7, ha="left" if dx else "center")
        ax.plot(4.4, 24.2, "*", color="#D55E00", ms=12)
        ax.text(4.4, 22.8, "候选火源", ha="center", color="#A43E00", fontsize=8)
        ax.set(xlabel="物理 x（m）", ylabel="物理 y（m）", title="58个候选气体测点（z=2.0 m）", xlim=(-2, 45), ylim=(0, 30))
        ax.set_aspect("equal")
        ax = axes[1]
        for (x, y), label in mapping.items():
            exit_cell = label.startswith("EXIT")
            shared = label in ("R56", "R57", "R58")
            ax.add_patch(Rectangle((x-.46, y-.46), .92, .92, fc="#FFF1D4" if shared else "#EAF2F8", ec="#616C76", lw=.6))
            ax.text(x, y, label.replace("EXIT_", "出口"), ha="center", va="center", fontsize=7)
            if exit_cell:
                ax.add_patch(Rectangle((x-.46, y-.46), .92, .92, fill=False, ec="#009E73", lw=2))
        ax.set(xlim=(0, 22), ylim=(14, 0), xlabel="RL 网格 x", ylabel="RL 网格 y", title="作者网格：61个非出口格 → 58个区域")
        ax.set_aspect("equal")
        fig.suptitle("酒店区域对应草案：编号关系由模板推断，尚未经作者确认", fontsize=13)
        fig.text(.5, .005, "左图仅画计算网格范围，并非完整墙体图；左右两图不共享比例。R56–R58各由两个RL格共用。", ha="center", fontsize=9)
        fig.savefig(out / "region_mapping_candidate_v2.png", dpi=180)
        fig.savefig(out / "region_mapping_candidate_v2.svg")
        plt.close(fig)


def main():
    config = load(CONFIG)
    out = ROOT / config["output_directory"]
    assert not out.exists(), f"Do not overwrite: {out}"
    source = ROOT / config["source_template"]
    assert digest(source) == config["source_template_sha256"]
    blob = hashlib.sha1(f"blob {source.stat().st_size}\0".encode() + source.read_bytes()).hexdigest()
    assert blob == config["source_template_git_blob"]
    assert load(ROOT / "outputs/safestep_gap_material_search_20261016/fdgen_commit.json")["sha"] == config["source_template_commit"]
    # Record and verify existing frozen inputs before any derived writes.
    historical = load(ROOT / "outputs/safestep_reproduction_package_20261016/source_manifest.json")["input_sha256"]
    prior_gap = load(ROOT / "outputs/safestep_gap_material_search_20261016/completion_manifest.json")["artifact_sha256"]
    protected = {**historical, **prior_gap}
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, name
    input_paths = [CONFIG, source, ROOT / config["source_mapping"],
                   ROOT / "outputs/safestep_paper_numerics_20261015/nist_author_manuscript.bin",
                   ROOT / "outputs/safestep_gap_material_search_20261016/materials/FDS_User_Guide_6.9.1.pdf",
                   ROOT / "outputs/safestep_gap_material_search_20261016/materials/FDS_User_Guide_6.9.1_text.txt",
                   ROOT / "outputs/safestep_gap_material_search_20261016/materials/LICENSE.md"]
    mapping = get_mapping(ROOT / config["source_mapping"])
    source_blocks = blocks(source.read_text("utf-8"))
    rows = propose_locations(mapping, source_blocks)
    draft, devices, ramp, edits, candidate_blocks = derive_input(source_blocks, rows, config)
    meshes, screened, inventory = geometry_screen(rows, candidate_blocks)
    out.mkdir(parents=True)
    save(out / "frozen_specification.json", config)
    save(out / "source_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
         "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in input_paths},
         "protected_previous_sha256": protected, "protected_previous_count": len(protected)})
    save(out / "safestep_hotel_candidate.fds.in", draft)
    save(out / "gas_devices_58x3.fds.fragment", devices)
    csv_save(out / "region_mapping_candidate.csv", rows)
    csv_save(out / "region_point_geometry_screen.csv", screened)
    csv_save(out / "specified_HRR_ramp.csv", ramp)
    save(out / "template_derivation_log.json", edits)
    save(out / "obstruction_mesh_inventory.json", inventory)
    register = []
    for group in ("paper_confirmed", "public_rate_table_confirmed", "candidate_assumptions", "required_unknowns"):
        for key, value in config[group].items():
            register.append({"parameter": key, "value_json": json.dumps(value, ensure_ascii=False), "evidence_group": group})
    csv_save(out / "parameter_register.csv", register)
    shutil.copyfile(ROOT / "outputs/safestep_gap_material_search_20261016/materials/LICENSE.md", out / "FD-Gen_LICENSE.md")
    # Preserve the equation tokens from the manuscript without importing author code.
    ns = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    xml = ET.fromstring(zipfile.ZipFile(input_paths[3]).read("word/document.xml"))
    math_tokens = ["".join(node.itertext()) for node in xml.findall(".//m:oMath", ns)]
    save(out / "manuscript_equation_tokens.json", {"warning": "Token order only, not a complete mathematical serialization; inspect original Eq3",
         "tokens": [s for s in math_tokens if any(k in s for k in ("3.317", "2.36", "exp"))]})
    parameters = {"ventilation_L_per_min": None, "COHb_incapacitation_percent": None,
                  "equation_rate_time_unit": None,
                  "formula_concentration_units": {"CO": "ppm", "CO2": "percent", "HCN": "ppm"},
                  "convention_status": "reconstruction_assumption", "parameter_provenance": "",
                  "evacuation_start_fire_time_s": 300.0,
                  "note": "Proposed interface convention; original units/time basis/V/D are unconfirmed. Null values intentionally fail validation."}
    save(out / "fer_parameters_REQUIRED.json", parameters)
    runtime_blockers = ["Required HCN yield token unresolved; draft is intentionally not executable",
                        "No FDS parser/initialization, mesh-snapped geometry or numerical stability check has been run"]
    reproduction_blockers = ["Exact original geometry and 58-region physical mapping unavailable",
                            "Original burner dimensions/location and post-peak HRR unavailable",
                            "Window gas-temperature surrogate is not the paper's verified glass-temperature model",
                            "Original HCN yield and concentration time series unavailable",
                            "Original Eq3 V, D, gas-unit conventions and coefficient time basis unconfirmed",
                            "Original FDS version, numerical settings and output averaging unconfirmed",
                            "Original small-building case inputs remain unavailable"]
    save(out / "readiness.json", {"status": "STATIC_DRAFT_COMPLETE_NOT_READY_FOR_FIRE_REPRODUCTION",
         "runtime_blockers": runtime_blockers, "exact_reproduction_blockers": reproduction_blockers,
         "scope": config["scope_limits"],
         "next_bounded_step": "After explicit physical scenario parameters are set, run only an initialization/short verification case in a new directory; never label it the author's original simulation"})
    render_figure(out, rows, mapping, meshes)
    save(out / "build_summary.json", {
        "source_block_counts": dict(Counter(k for k, _ in source_blocks)),
        "candidate_block_counts": dict(Counter(k for k, _ in candidate_blocks)),
        "physical_regions": len(rows), "nonexit_RL_cells": sum(r["RL_cell_count"] for r in rows),
        "new_gas_devices": 174, "mesh_cells": sum(math.prod(m["IJK"]) for m in meshes),
        "raw_point_clear_count": sum(r["screen_result"] == "RAW_POINT_CLEAR" for r in screened),
        "mesh_grid_plane_point_count": sum(bool(r["mesh_grid_plane_axes"]) for r in screened),
        "positive_volume_mesh_overlap_obstructions": sum(bool(r["positive_volume_mesh_overlap"]) for r in inventory),
        "no_positive_volume_mesh_overlap_obstructions": sum(not r["positive_volume_mesh_overlap"] for r in inventory),
        "FDS_initialization_performed": False, "protected_previous_files_verified": len(protected),
        "FDS_executable_detected": "C:/Program Files/firemodels/FDS6/bin/fds.exe",
        "FDS_executable_version": None,
        "figure": {"purpose": "internal source/mapping audit", "publisher_compliance_claim": False,
                   "encodings": "sampling coordinates in metres; RL lattice separately scaled; no data fitting"},
    })
    print(json.dumps({"output": out.as_posix(), "draft_written": True,
                      "region_count": len(rows), "raw_point_clear": sum(r["screen_result"] == "RAW_POINT_CLEAR" for r in screened),
                      "FDS_runs": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()
