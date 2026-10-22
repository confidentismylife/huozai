"""Audit PLOS 2016 choice tables and lock an independent, label-free input set.

Source: Haghani et al., PLOS ONE 11(11): e0166908, supporting tables S1/S3.
The XLS files are CC BY. Install optional reader with `pip install xlrd>=2.0`.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

try:
    import xlrd
except ImportError as exc:
    raise SystemExit("Install xlrd>=2.0 to read the source XLS files") from exc


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public_haghani_2016"
OUTPUT = ROOT / "outputs/haghani_external"
SEED = 20260928
PER_CHOICE_SET_SIZE = 30


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_choice_sets(path: Path, expected_context: int):
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = sheet.row_values(13)
    assert header == ["id", "alti", "cset", "choice", "Dist", "Cong", "Vis", "FltoEx", "Pds", "RPSP"]
    index = 14
    observation = 0
    valid, exclusions = [], []
    while index < sheet.nrows:
        first = sheet.row_values(index)
        size = int(first[2])
        block = [sheet.row_values(j) for j in range(index, min(index + size, sheet.nrows))]
        observation += 1
        reasons = []
        if len(block) != size or [int(r[1]) for r in block] != list(range(1, size + 1)):
            reasons.append("malformed_alternatives")
        if len({int(r[0]) for r in block}) != 1 or len({int(r[2]) for r in block}) != 1:
            reasons.append("mixed_choice_set")
        if sum(r[3] for r in block) != 1 or any(r[3] not in (0, 1) for r in block):
            reasons.append("invalid_choice")
        if any(sheet.cell_type(j, 4) != xlrd.XL_CELL_NUMBER for j in range(index, index + len(block))):
            reasons.append("missing_distance")
        if any(r[6] not in (0, 1) for r in block):
            reasons.append("invalid_visibility")
        if any(r[9] != expected_context for r in block):
            reasons.append("unexpected_context")
        if reasons:
            exclusions.append({"observation_index": observation, "source_excel_row": index + 1, "reasons": reasons})
        else:
            valid.append({
                "observation_index": observation,
                "source_excel_row": index + 1,
                "record_id": int(first[0]),
                "choice_set_size": size,
                "selected_exit": next(int(r[1]) for r in block if r[3] == 1),
                "alternatives": [{
                    "exit": int(r[1]), "distance_m": float(r[4]),
                    "congestion_people": int(r[5]), "visible": bool(r[6]),
                    "flow_to_exit_people": int(r[7]),
                } for r in block],
            })
        index += size
    return valid, exclusions, observation, sheet.nrows - 14


def lock_inputs(realistic):
    """Select without inspecting exit labels; at most one observation per record ID."""
    rng = random.Random(SEED)
    used_ids = set()
    selected = []
    for size in (2, 3, 4):
        eligible = sorted((r for r in realistic if r["choice_set_size"] == size), key=lambda r: r["observation_index"])
        rng.shuffle(eligible)
        chosen = []
        for row in eligible:
            if row["record_id"] in used_ids:
                continue
            chosen.append({key: value for key, value in row.items() if key != "selected_exit"})
            used_ids.add(row["record_id"])
            if len(chosen) == PER_CHOICE_SET_SIZE:
                break
        if len(chosen) != PER_CHOICE_SET_SIZE:
            raise ValueError(f"not enough independent records with {size} exits")
        selected.extend(chosen)
    return sorted(selected, key=lambda r: r["observation_index"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hc_path = DATA / "S1_hypothetical_choices.xls"
    rc_path = DATA / "S3_realistic_choices.xls"
    hypothetical, hc_excluded, hc_total, hc_rows = read_choice_sets(hc_path, 1)
    realistic, rc_excluded, rc_total, rc_rows = read_choice_sets(rc_path, 0)
    locked = lock_inputs(realistic)
    inputs_path = OUTPUT / "locked_input_cases.json"
    if inputs_path.exists():
        locked = json.loads(inputs_path.read_text(encoding="utf-8"))
        valid_by_index = {r["observation_index"]: r for r in realistic}
        if len(locked) != 3 * PER_CHOICE_SET_SIZE or len({r["record_id"] for r in locked}) != len(locked):
            raise RuntimeError("locked input manifest has invalid size or repeated record IDs")
        for row in locked:
            source = valid_by_index.get(row["observation_index"])
            if source is None or row != {k: v for k, v in source.items() if k != "selected_exit"}:
                raise RuntimeError("locked input manifest contains an invalid or changed case")
    else:
        inputs_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "source_article": "https://doi.org/10.1371/journal.pone.0166908",
        "source_files": {hc_path.name: file_hash(hc_path), rc_path.name: file_hash(rc_path)},
        "selection_seed": SEED,
        "selection_rule": "30 valid cases per exit count (2,3,4); at most one per record ID; validity requires one choice but selection does not use which exit was chosen",
        "locked_input_sha256": file_hash(inputs_path),
        "hypothetical": {
            "source_rows": hc_rows, "choice_sets": hc_total, "valid_choice_sets": len(hypothetical),
            "record_ids": len({r["record_id"] for r in hypothetical}),
            "choice_set_sizes": dict(Counter(r["choice_set_size"] for r in hypothetical)),
            "exclusions": hc_excluded,
        },
        "realistic": {
            "source_rows": rc_rows, "choice_sets": rc_total, "valid_choice_sets": len(realistic),
            "record_ids": len({r["record_id"] for r in realistic}),
            "choice_set_sizes": dict(Counter(r["choice_set_size"] for r in realistic)),
            "exclusions": rc_excluded,
        },
        "locked_cases": len(locked),
        "warning": "No smoke or signage field. Realistic record ID may identify a participant-run rather than a unique human across runs.",
    }
    (OUTPUT / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("hypothetical", "realistic", "locked_cases", "locked_input_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
