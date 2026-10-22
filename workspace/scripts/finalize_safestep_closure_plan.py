"""Freeze completed evidence and generate an unexecuted 24-case design matrix.

This program runs no simulator, policy rollout, training, or LLM request.
It reads an explicit prior provenance list, not the sealed human test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/post_safestep_physical_design_20261019.json"
OUT = ROOT / "outputs/safestep_closure_plan_20261019"
CHAPTER = ROOT / "reports/SafeStep复现与预测门控审计_论文章节_v1_2026-10-19.md"
DESIGN_REPORT = ROOT / "reports/SafeStep收尾与新建FDS场景实验方案_v1_2026-10-19.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def safe_old_path(relative: str) -> Path:
    if any(word in relative.lower() for word in ("sealed", "final48")):
        raise ValueError("Refusing to open an explicitly sealed data path")
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def prior_hashes() -> dict[str, str]:
    frozen = read_json(ROOT / "outputs/safestep_robust_gate_20261018/frozen_plan.json")
    expected = dict(frozen["protected_previous_sha256"])
    for relative in (
        "outputs/safestep_robust_gate_20261018/completion_manifest.json",
        "outputs/safestep_robust_gate_analysis_20261018/completion_manifest.json",
    ):
        manifest = read_json(ROOT / relative)
        expected.update(manifest["artifact_sha256"])
        expected[relative] = sha(ROOT / relative)
    return expected


def verify_files(expected: dict[str, str]) -> list[dict[str, str]]:
    problems = []
    for relative, value in expected.items():
        path = safe_old_path(relative)
        if not path.is_file():
            problems.append({"path": relative, "problem": "MISSING"})
        elif sha(path) != value:
            problems.append({"path": relative, "problem": "HASH_MISMATCH"})
    return problems


def inside(point: tuple[float, float], rectangles: list[list[float]]) -> bool:
    x, y = point
    return any(a <= x <= b and c <= y <= d for a, b, c, d in rectangles)


def check_geometry(layout: dict) -> dict:
    """Only check the abstract open-floor union, not FDS geometry or smoke."""
    rectangles = layout["open_rectangles_xy"]
    width, depth, _ = layout["domain_xyz"]
    nodes = {
        (ix, iy)
        for ix in range(int(width * 2) + 1)
        for iy in range(int(depth * 2) + 1)
        if inside((ix / 2, iy / 2), rectangles)
    }
    def key(point):
        return tuple(int(round(value * 2)) for value in point)
    origin = key(layout["exit_centers_xy"][0])
    visited, queue = {origin}, deque([origin])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            node = (x + dx, y + dy)
            # Midpoint checks guard against an edge cutting a solid segment.
            midpoint = ((x + node[0]) / 4, (y + node[1]) / 4)
            if node in nodes and node not in visited and inside(midpoint, rectangles):
                visited.add(node)
                queue.append(node)
    points = layout["start_centers_xy"] + layout["exit_centers_xy"]
    assert all(key(point) in visited for point in points)
    assert all(inside(tuple(point), rectangles) for point in layout["fire_centers_xy"].values())
    assert len(layout["start_centers_xy"]) == 10
    return {
        "abstract_grid_spacing_m": 0.5,
        "open_grid_points": len(nodes),
        "reachable_open_points_from_exit_A": len(visited),
        "all_ten_starts_and_two_exits_connected": True,
        "fire_centers_in_open_floor": True,
        "claim_boundary": "Abstract coordinate connectivity only; FDS obstructions, wall apertures, sensors and flow are not checked.",
    }


def matrix(protocol: dict) -> list[dict]:
    factors = protocol["matrix"]["factors"]
    names = list(factors)
    rows = []
    for levels in itertools.product(*(factors[name] for name in names)):
        row = dict(zip(names, levels))
        row["case_id"] = "_".join(levels)
        row["target_peak_HRR_kW"] = 1000
        row["time_to_peak_s"] = int(row["growth_id"][1:])
        row["status"] = "DESIGNED_NOT_RUN_PHYSICS_PENDING"
        rows.append(row)
    random.Random(protocol["matrix"]["run_order_seed"]).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["run_order"] = index
    return rows


def build() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    protocol = read_json(PROTOCOL)
    expected = prior_hashes()
    problems = verify_files(expected)
    if problems:
        raise RuntimeError(f"Existing frozen evidence changed: {problems[:8]}")
    rows = matrix(protocol)
    factors = protocol["matrix"]["factors"]
    assert len(rows) == protocol["matrix"]["expected_cases"] == 24
    assert len({row["case_id"] for row in rows}) == 24
    balance = {name: dict(Counter(row[name] for row in rows)) for name in factors}
    assert all(len(set(counts.values())) == 1 for counts in balance.values())
    assert protocol["evaluation_precommitment"]["path_pairs_per_method"] == 24 * 10 * 3 * 2
    assert max(protocol["sampling_and_paths"]["departure_fire_times_s"]) + 240 / 0.8 <= 900
    geometry = {key: check_geometry(protocol["geometry_design_assumptions"][key]) for key in ("L1", "L2")}
    OUT.mkdir(parents=True)
    with (OUT / "scenario_matrix.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(OUT / "frozen_factorial_design.json", protocol)
    save_json(OUT / "source_hashes.json", expected)
    save_json(OUT / "design_validation.json", {
        "status": "PASS_STRUCTURAL_DESIGN_CHECKS",
        "factorial_cases": len(rows), "balanced_factors": balance,
        "geometry": geometry, "nominal_paired_routes_per_method": 1440,
        "per_observation_condition_paired_routes_per_method": 720,
        "executable_physics_validated": False,
        "gate_states": protocol["execution_gates"],
    })
    after = verify_files(expected)
    assert not after, after[:8]
    source_manifest = {
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "human_verified": False,
        "verification_status": "Local machine checks only; author scientific review pending",
        "chapter": {"path": CHAPTER.relative_to(ROOT).as_posix(), "sha256": sha(CHAPTER)},
        "design_report": {"path": DESIGN_REPORT.relative_to(ROOT).as_posix(), "sha256": sha(DESIGN_REPORT)},
        "online_sources": protocol["official_sources_checked"],
    }
    save_json(OUT / "evidence_manifest.json", source_manifest)
    artifact_hashes = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted(OUT.iterdir()) if path.is_file()
    }
    for path in (PROTOCOL, CHAPTER, DESIGN_REPORT, Path(__file__).resolve()):
        artifact_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    save_json(OUT / "completion_manifest.json", {
        "status": "COMPLETE_CHAPTER_CLOSURE_AND_FACTORIAL_DESIGN",
        "prior_files_verified_before_and_after": len(expected),
        "artifact_sha256": artifact_hashes,
        "FDS_runs": 0, "new_policy_episodes": 0,
        "training_updates": 0, "LLM_calls": 0, "sealed_final48_accessed": False,
        "full_physics_execution_ready": False,
        "freeze_scope": "Completed prior evidence plus chapter version and nominal factorial design. Unresolved executable physical parameters remain explicitly pending.",
    })
    print(json.dumps({"status": "COMPLETE_DESIGN_NOT_SIMULATION", "cases": 24,
        "protected_files": len(expected), "FDS_runs": 0, "output": str(OUT)}, ensure_ascii=False))


def verify() -> None:
    latest = OUT / "completion_manifest_reviewed.json"
    manifest = read_json(latest if latest.exists() else OUT / "completion_manifest.json")
    errors = verify_files(manifest["artifact_sha256"])
    errors += verify_files(read_json(OUT / "source_hashes.json"))
    if errors:
        raise RuntimeError(errors[:10])
    print(json.dumps({"status": "PASS", "new_artifacts": len(manifest["artifact_sha256"]),
        "prior_files": manifest["prior_files_verified_before_and_after"],
        "physics_ready": False, "FDS_runs": 0}))


def seal_review() -> None:
    """Preserve the first freeze; record a scientific review before any scoring."""
    if (OUT / "completion_manifest_reviewed.json").exists():
        raise FileExistsError("Reviewed closure already exists")
    old = read_json(OUT / "completion_manifest.json")
    expected = read_json(OUT / "source_hashes.json")
    assert not verify_files(expected)
    assert sha(OUT / "pre_review_design_report.md") == old["artifact_sha256"][DESIGN_REPORT.relative_to(ROOT).as_posix()]
    assert sha(OUT / "pre_review_finalizer.py") == old["artifact_sha256"][Path(__file__).resolve().relative_to(ROOT).as_posix()]
    protocol = read_json(PROTOCOL)
    with (OUT / "scenario_matrix.csv").open(encoding="utf-8-sig", newline="") as file:
        stored = list(csv.DictReader(file))
    generated = matrix(protocol)
    assert [row["case_id"] for row in stored] == [row["case_id"] for row in generated]
    checks = read_json(OUT / "design_validation.json")
    checks["review_version"] = "1.1"
    checks["new_paired_failures_required_zero"] = protocol["evaluation_precommitment"]["descriptive_continuation_criteria"]["new_failure_pairs_must_equal_zero"]
    checks["exit_height_aligned_at_all_candidate_meshes"] = all(
        abs(protocol["geometry_design_assumptions"]["exit_aperture_height_m"] / spacing - round(protocol["geometry_design_assumptions"]["exit_aperture_height_m"] / spacing)) < 1e-10
        for spacing in protocol["mesh_plan"]["candidate_spacing_m"])
    assert checks["new_paired_failures_required_zero"] and checks["exit_height_aligned_at_all_candidate_meshes"]
    save_json(OUT / "frozen_factorial_design_reviewed.json", protocol)
    save_json(OUT / "design_validation_reviewed.json", checks)
    save_json(OUT / "final_review_amendment.json", {
        **protocol["review_amendment"],
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "new_FDS_runs": 0, "new_policy_evaluations": 0,
        "original_report_sha256": sha(OUT / "pre_review_design_report.md"),
        "reviewed_report_sha256": sha(DESIGN_REPORT),
        "original_protocol_sha256": sha(OUT / "frozen_factorial_design.json"),
        "reviewed_protocol_sha256": sha(PROTOCOL),
        "original_completion_manifest_role": "Preserved historical initial freeze; changed source paths are superseded by this recorded pre-execution review.",
    })
    hashes = {path.relative_to(ROOT).as_posix(): sha(path)
              for path in sorted(OUT.iterdir()) if path.is_file()}
    for path in (PROTOCOL, CHAPTER, DESIGN_REPORT, Path(__file__).resolve()):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    save_json(OUT / "completion_manifest_reviewed.json", {
        **old, "review_version": "1.1", "artifact_sha256": hashes,
        "active_protocol": "frozen_factorial_design_reviewed.json",
        "evidence_manifest_note": "Initial evidence_manifest preserves first report hash; final_review_amendment and this manifest provide the current report and protocol hashes.",
    })
    print(json.dumps({"status": "PRE_EXECUTION_REVIEW_SEALED", "version": "1.1", "cases": 24, "prior_files": len(expected), "FDS_runs": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seal-review", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    elif args.seal_review:
        seal_review()
    else:
        build()
