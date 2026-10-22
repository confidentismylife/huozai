"""Source-only provenance and map-semantics audit. Never score outcomes.

Actual choices, model probabilities and method scores are never used to choose
an interpretation. The absolute-direction candidate stays explicitly unconfirmed.
"""
from __future__ import annotations
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "schemas/bristol_source_semantics_audit_20261014.json"
OUT = ROOT / "outputs/bristol_source_semantics_audit_20261014"
FIELDS = ["left_exit", "right_exit", "knowledge", "start_time", "finish_time"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def table(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def number(value):
    return Decimal(str(value).strip().replace(",", ".")).normalize()


def project(row):
    return tuple(number(row[k]) for k in FIELDS)


def legacy_direction(left, right, code):
    if code == 0: return "unknown"
    if code == 3: return "equal"
    if left == right: return "unresolved"
    majority = "left" if left > right else "right"
    minority = "right" if left > right else "left"
    return majority if code == 1 else minority


def candidate_direction(code):
    return {0: "unknown", 1: "left", 2: "right", 3: "equal"}[code]


def freeze(cfg, source):
    paths = [CFG, Path(__file__), ROOT / cfg["source_archive"], ROOT / cfg["paper_pdf"],
        source / "README.txt", source / ".info.json", source / "information_sheet.pdf",
        source / "consent_form.pdf", source / "processed_data/processed_data.CSV",
        OUT / "readme_current.txt", OUT / "directory_current.html", OUT / "public_source_fetch_log.json"]
    protected = []
    for folder in ["bristol_context_choice_benchmark_20261013", "bristol_prior_mechanism_probe_20261013", "bristol_no_prior_choice_benchmark_20261013"]:
        protected.extend(p for p in (ROOT / "outputs" / folder).iterdir() if p.suffix.lower() in {".json", ".jsonl", ".csv"})
    for folder in ["schemas", "scripts"]:
        protected.extend((ROOT / folder).glob("*bristol*"))
    protected = [p for p in protected if p.is_file() and p not in {CFG, Path(__file__)}]
    protected.extend([ROOT / "reports/无基率LLM真人出口选择完整对照_2026-10-13.md",
        ROOT / "reports/出口选择LLM基率提示机制诊断_2026-10-13.md",
        ROOT / "reports/出口选择失败原因与语义风险复核_2026-10-13.md"])
    snapshot = {"source_files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths},
        "protected_previous_files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(set(protected))},
        "policy": {k: cfg[k] for k in ["scope", "forbidden_for_decoding", "do_not_promote_candidate"]}}
    target = OUT / "source_snapshot_v2.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        assert {k: v for k, v in previous.items() if k != "recorded_at_utc"} == snapshot
        return previous
    snapshot["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(target, snapshot)
    return snapshot


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    source = ROOT / cfg["source_folder"]
    OUT.mkdir(exist_ok=True)
    snapshot = freeze(cfg, source)
    assert (OUT / "readme_current.txt").read_bytes() == (source / "README.txt").read_bytes()
    rows = list(csv.DictReader((source / "processed_data/processed_data.CSV").open(encoding="utf-8-sig")))
    # Only these five non-outcome fields enter downstream audit functions.
    projections = [{key: row[key] for key in FIELDS} for row in rows]
    raw_paths = sorted((source / "raw_data").rglob("*.Txt"))
    raw_lookup = defaultdict(list)
    context_lookup = defaultdict(list)
    raw_counts = Counter()
    raw_hashes = {}
    for path in raw_paths:
        with path.open(encoding="utf-8-sig") as handle:
            first_five = [next(handle).strip() for _ in range(5)]
        start, code, left, right, finish = map(number, first_five)
        key = left, right, code, start, finish
        relative = str(path.relative_to(source)).replace("\\", "/")
        raw_lookup[key].append(relative)
        context_lookup[key[:3]].append((key, relative))
        raw_counts[(int(left), int(right), int(code))] += 1
        raw_hashes[relative] = sha(path)
    processed_counts = Counter((int(r["left_exit"]), int(r["right_exit"]), int(r["knowledge"])) for r in projections)
    assert processed_counts == raw_counts
    matches = []
    ambiguity_count = changed_count = changed_records = pairing_change_records = 0
    changed_conditions = set()
    pairing_changed_conditions = set()
    for index, row in enumerate(projections, 2):
        key = project(row)
        exact_candidates = raw_lookup.get(key, [])
        units = [Decimal(10) ** Decimal(str(row[field]).replace(",", ".")).as_tuple().exponent for field in ["start_time", "finish_time"]]
        interval_candidates = [relative for raw_key, relative in context_lookup[key[:3]]
            if key[3] <= raw_key[3] < key[3] + units[0] and key[4] <= raw_key[4] < key[4] + units[1]]
        candidates = exact_candidates if exact_candidates else interval_candidates
        left, right, code = [int(row[k]) for k in FIELDS[:3]]
        old = legacy_direction(left, right, code)
        hypothetical = candidate_direction(code)
        eligible = not (left == right and code in (1, 2))
        ambiguity_count += not eligible
        direction_changed = eligible and old != hypothetical
        changed_records += direction_changed
        if direction_changed: changed_conditions.add((left, right, code))
        pairing_changed = eligible and code in (1, 2)
        pairing_change_records += pairing_changed
        if pairing_changed: pairing_changed_conditions.add((left, right, code))
        gate = "map_direction_semantics_unresolved" if code in (1, 2) else (
            "equal_map_balanced_count_discrepancy" if code == 3 and left == right else "comparatively_clear_under_README")
        matches.append({"case_key": f"context2023_csvrow_{index}", "source_csv_row": index,
            "assigned_left": left, "assigned_right": right, "knowledge_code": code,
            "start_time": row["start_time"], "finish_time": row["finish_time"],
            "exact_raw_matches_without_choice": len(exact_candidates),
            "truncation_interval_candidates_without_choice": len(interval_candidates),
            "source_link_status": "exact_unique" if len(exact_candidates)==1 else "unique_interval_candidate" if len(candidates)==1 else "ambiguous_interval_candidates" if candidates else "unmatched",
            "raw_relative_files": "|".join(candidates),
            "legacy_direction_README": old, "absolute_candidate_direction_UNCONFIRMED": hypothetical,
            "eligible_in_previous_benchmark": eligible, "hypothetical_direction_change": direction_changed,
            "hypothetical_mirror_pair_change": pairing_changed, "source_gate": gate})
    assert len(rows) == len(raw_paths) == 547
    table("raw_processed_source_links.csv", matches)
    write_json(OUT / "raw_file_sha256.json", raw_hashes)
    counts = [{"left": a, "right": b, "knowledge_code": k, "processed_count": n, "raw_count": raw_counts[(a, b, k)]}
              for (a, b, k), n in sorted(processed_counts.items())]
    table("source_condition_counts.csv", counts)
    # Table comparisons use counts and condition codes only, not exit choices.
    comparison = []
    codes = {"no_map": 0, "equal_map": 3, "left_nearer_map": 1, "right_nearer_map": 2}
    for category, groups in cfg["table_A1"].items():
        for group, paper_samples in groups.items():
            c = cfg["table_1"][group]
            crowds = [c["original"]] + ([c["mirror"]] if c["mirror"] else [])
            candidate_samples = [processed_counts[(a, b, codes[category])] for a, b in crowds]
            relative_samples = []
            for a, b in crowds:
                if category == "no_map": code = 0
                elif category == "equal_map": code = 3
                elif a == b: code = None
                else:
                    desired = "left" if category == "left_nearer_map" else "right"
                    code = 1 if ("left" if a > b else "right") == desired else 2
                relative_samples.append(processed_counts[(a, b, code)] if code is not None else None)
            comparison.append({"paper_category": category, "group": group,
                "paper_samples_original_mirror": json.dumps(paper_samples),
                "absolute_candidate_samples_UNCONFIRMED": json.dumps(candidate_samples),
                "absolute_candidate_unordered_count_match": sorted(candidate_samples) == sorted(paper_samples),
                "absolute_candidate_total_delta": sum(candidate_samples) - sum(paper_samples),
                "README_relative_samples_if_row_utility_is_fixed": json.dumps(relative_samples),
                "README_relative_unordered_match_if_fixed": None if None in relative_samples else sorted(relative_samples) == sorted(paper_samples)})
    table("paper_sample_count_comparison.csv", comparison)
    with zipfile.ZipFile(ROOT / cfg["source_archive"]) as archive:
        assert archive.testzip() is None
        assert archive.read("3kn82h2aamsfw2eplor4v2ewk0/processed_data/processed_data.CSV") == (source / "processed_data/processed_data.CSV").read_bytes()
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        ext = Counter(Path(n).suffix.lower() for n in names)
        program_assets = [n for n in names if Path(n).suffix.lower() in {".cs", ".exe", ".dll", ".unity", ".prefab", ".asset"}]
    metadata = json.loads((source / ".info.json").read_text(encoding="utf-8"))
    metadata_checks = []
    def walk(node):
        if isinstance(node, list):
            for item in node: walk(item)
        elif isinstance(node, dict):
            if "sha1" in node:
                relative = str(node["id"]).split("3kn82h2aamsfw2eplor4v2ewk0/", 1)[1]
                path = source / relative
                actual = hashlib.sha1(path.read_bytes()).hexdigest()
                assert actual == node["sha1"]
                metadata_checks.append({"relative_file": relative, "SHA1_matches_official_metadata": True})
            for value in node.values():
                if isinstance(value, (dict, list)): walk(value)
    walk(metadata)
    table("official_metadata_checksum_checks.csv", metadata_checks)
    fetch_log = json.loads((OUT / "public_source_fetch_log.json").read_text())
    game_text = (OUT / "historical_game_link.html").read_text(encoding="utf-8")
    removed_message = "This link has been removed" in game_text and "access to this document has been removed" in game_text
    gate_counts = Counter(r["source_gate"] for r in matches)
    link_counts = Counter(r["source_link_status"] for r in matches)
    summary = {"status": "source_audit_completed_map_codes_unresolved", "processed_records": 547,
        "raw_files": 547, "raw_processed_condition_counts_match": True,
        "raw_processed_link_status_counts": dict(link_counts),
        "unique_raw_file_candidates": len({r["raw_relative_files"] for r in matches if r["source_link_status"] in ['exact_unique','unique_interval_candidate']}),
        "processed_time_precision_caveat": "Every exact time key failed. Interval links are truncation-compatible source candidates; they do not establish person identity or continuous timing precision.",
        "official_metadata_checksum_matches": len(metadata_checks),
        "current_README_matches_downloaded_README": True, "archive_crc_and_processed_bytes_verified": True,
        "paper_table_A1_total": sum(sum(n) for groups in cfg["table_A1"].values() for n in groups.values()),
        "absolute_candidate_unordered_count_matches": sum(r["absolute_candidate_unordered_count_match"] for r in comparison),
        "count_comparison_groups": len(comparison), "absolute_candidate_is_confirmed": False,
        "two_balanced_count_discrepancies": [r for r in comparison if not r["absolute_candidate_unordered_count_match"]],
        "previous_balanced_code_1_2_quarantined_records": ambiguity_count,
        "hypothetical_candidate_changes_existing_direction_records": changed_records,
        "hypothetical_candidate_changes_existing_direction_conditions": len(changed_conditions),
        "hypothetical_candidate_changes_existing_mirror_pair_records": pairing_change_records,
        "hypothetical_candidate_changes_existing_mirror_pair_conditions": len(pairing_changed_conditions),
        "source_gate_counts": dict(gate_counts),
        "comparatively_clear_conditions": len({(r['assigned_left'],r['assigned_right'],r['knowledge_code']) for r in matches if r['source_gate']=='comparatively_clear_under_README'}),
        "archive_extensions": dict(ext), "program_assets_in_archive": program_assets,
        "historical_game_transcribed_link_reports_removed": removed_message,
        "historical_game_link_caveat": "Manually transcribed case-sensitive URL; server message is for that URL. No executable/source recovered; no authentication attempted.",
        "chosen_new_map_mapping": None, "outcomes_used_for_decoding": False,
        "new_performance_scores_computed": False, "LLM_requests": 0, "PPO_updates": 0,
        "old_final_scenarios_accessed": False, "Liao_confirmation_scores_computed": False,
        "source_snapshot_sha256": sha(OUT / "source_snapshot_v2.json"),
        "decision": "Preserve old scores as conditional on README interpretation; pause directional-map claims and verify code-to-map assignment before new comparisons."}
    write_json(OUT / "summary.json", summary)
    assert all(sha(ROOT / p) == value for p, value in snapshot["protected_previous_files"].items())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
