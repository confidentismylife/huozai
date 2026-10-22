"""Preserve the first render and move its legend clear of all evidence points."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/candidate_library_coverage_audit_development_v1"
STEM = "missed_candidate_prediction_comparison"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    archive = OUT / "figure_layout_initial"
    assert not archive.exists()
    archive.mkdir()
    preserved = [OUT / (STEM + "." + suffix) for suffix in ("png", "svg", "pdf")]
    preserved += [OUT / "report_assets_manifest.json", ROOT / "scripts/write_candidate_library_coverage_report.py"]
    initial_hashes = {}
    for path in preserved:
        target = archive / path.name
        shutil.copyfile(path, target)
        assert sha(target) == sha(path)
        initial_hashes[target.name] = sha(target)
    states = table(OUT / "state_opportunity_audit.csv")
    plans = table(OUT / "full_library_plan_evidence.csv")
    lookup = {(r["state_id"], int(r["plan_index"])): r for r in plans}
    missed = sorted((r for r in states if r["missed_by_old_proposal"] == "True"), key=lambda r: r["state_id"])
    rows = [lookup[r["state_id"], int(r["empirical_best_index"])] for r in missed]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    yy = np.arange(len(rows))
    collections = [
        ax.scatter([float(r["old_prediction_gain_pp"]) for r in rows], yy + .11,
                   color="#D55E00", marker="x", s=55, label="Frozen prediction for that candidate"),
        ax.scatter([float(r["actual_gain_pp"]) for r in rows], yy - .11,
                   color="#0072B2", s=40, label="Actual 6-future mean (hindsight-selected candidate)")]
    ax.axvline(0, color="#555555", lw=.8)
    ax.set_yticks(yy, [r["state_id"] + ": " + r["plan"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("601 incapacitation minus candidate incapacitation (percentage points)")
    ax.set_title("Missed opportunities in a viewed development subset", fontsize=12)
    ax.grid(axis="x", alpha=.2)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.60, .01), fontsize=8)
    fig.tight_layout(rect=(0, .20, 1, 1))
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    for collection in collections:
        for x, y in ax.transData.transform(collection.get_offsets()):
            assert not (legend_box.x0 - 8 <= x <= legend_box.x1 + 8 and
                        legend_box.y0 - 8 <= y <= legend_box.y1 + 8)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / (STEM + "." + suffix), dpi=200)
    plt.close(fig)
    assets_path = OUT / "report_assets_manifest.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    assets.update({"plots_sha256": {STEM + "." + s: sha(OUT / (STEM + "." + s)) for s in ("png", "svg", "pdf")},
                   "layout_repair_script_sha256": sha(__file__),
                   "initial_figure_and_writer_preserved": initial_hashes,
                   "legend_moved_outside_data_axes": True,
                   "legend_point_overlap_geometry_check_passed": True})
    assets_path.write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    note = {"issue": "Initial lower-left legend obscured the fourth candidate's old-prediction cross.",
            "repair": "Legend placed below the data axes; no scientific data or report text changed.",
            "initial_files_sha256": initial_hashes, "script_sha256": sha(__file__),
            "all_eight_data_points_outside_legend_checked": True,
            "new_simulations": 0, "new_fits": 0, "new_llm_calls": 0}
    (OUT / "figure_layout_repair.json").write_text(json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure_repaired": True, "original_files_preserved": len(initial_hashes),
                      "data_unchanged": True, "new_simulations": 0}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
