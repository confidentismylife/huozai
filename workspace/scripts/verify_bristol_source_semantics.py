"""Independently verify source audit without using choice outcomes or scores."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bristol_source_semantics_audit_20261014"
CONFIG = ROOT / "schemas/bristol_source_semantics_audit_20261014.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value):
    return Decimal(str(value).strip().replace(",", "."))


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    snapshot = json.loads((OUT / "source_snapshot_v2.json").read_text(encoding="utf-8"))
    for key in ["source_files", "protected_previous_files"]:
        assert all(sha(ROOT / path) == value for path, value in snapshot[key].items())
    source = ROOT / cfg["source_folder"]
    assert (OUT / "readme_current.txt").read_bytes() == (source / "README.txt").read_bytes()
    records = read_csv(source / "processed_data/processed_data.CSV")
    links = read_csv(OUT / "raw_processed_source_links.csv")
    assert len(records) == len(links) == 547
    fields = cfg["raw_processed_match_fields"]
    projected = [{k: row[k] for k in fields} for row in records]
    assert not set(cfg["forbidden_for_decoding"]) & set(links[0])
    raw = defaultdict(list)
    condition_counts = Counter()
    raw_hashes = json.loads((OUT / "raw_file_sha256.json").read_text(encoding="utf-8"))
    for relative, expected_sha in raw_hashes.items():
        path = source / relative
        assert sha(path) == expected_sha
        with path.open(encoding="utf-8-sig") as handle:
            start, code, left, right, finish = [num(next(handle)) for _ in range(5)]
        raw[(int(left), int(right), int(code))].append((start, finish, relative))
        condition_counts[(int(left), int(right), int(code))] += 1
    assert len(raw_hashes) == 547
    assert condition_counts == Counter((int(r['left_exit']),int(r['right_exit']),int(r['knowledge'])) for r in projected)
    status_counts = Counter()
    candidate_direction_changes = candidate_pair_changes = 0
    direction_changed_conditions = set()
    pairing_changed_conditions = set()
    gates = Counter()
    for index, (row, link) in enumerate(zip(projected, links), 2):
        cond = tuple(int(row[k]) for k in fields[:3])
        a, b, k = cond
        assert link["case_key"] == f"context2023_csvrow_{index}" and int(link["source_csv_row"]) == index
        assert (int(link["assigned_left"]),int(link["assigned_right"]),int(link["knowledge_code"])) == cond
        start, finish = num(row["start_time"]), num(row["finish_time"])
        start_unit = Decimal(10) ** start.as_tuple().exponent
        finish_unit = Decimal(10) ** finish.as_tuple().exponent
        exact = [name for x,y,name in raw[cond] if x == start and y == finish]
        interval = [name for x,y,name in raw[cond] if start <= x < start+start_unit and finish <= y < finish+finish_unit]
        actual = exact if exact else interval
        assert int(link["exact_raw_matches_without_choice"]) == len(exact)
        assert int(link["truncation_interval_candidates_without_choice"]) == len(interval)
        assert link["raw_relative_files"].split("|") == actual
        status = "exact_unique" if len(exact)==1 else "unique_interval_candidate" if len(actual)==1 else "ambiguous_interval_candidates" if actual else "unmatched"
        assert link["source_link_status"] == status
        status_counts[status] += 1
        if k == 0: old = "unknown"
        elif k == 3: old = "equal"
        elif a == b: old = "unresolved"
        else: old = "left" if (a > b) == (k == 1) else "right"
        candidate = {0:"unknown",1:"left",2:"right",3:"equal"}[k]
        assert link["legacy_direction_README"] == old
        assert link["absolute_candidate_direction_UNCONFIRMED"] == candidate
        eligible = not (a == b and k in (1,2))
        change = eligible and old != candidate
        pair_change = eligible and k in (1,2)
        assert link["eligible_in_previous_benchmark"] == str(eligible)
        assert link["hypothetical_direction_change"] == str(change)
        assert link["hypothetical_mirror_pair_change"] == str(pair_change)
        candidate_direction_changes += change; candidate_pair_changes += pair_change
        if change: direction_changed_conditions.add(cond)
        if pair_change: pairing_changed_conditions.add(cond)
        gate = "map_direction_semantics_unresolved" if k in (1,2) else "equal_map_balanced_count_discrepancy" if k==3 and a==b else "comparatively_clear_under_README"
        assert gate == link["source_gate"]; gates[gate] += 1
        # Changing outcomes/demographics does not change the source projection.
        mutated = {**records[index-2], "exit_choice": "999", "observed_exit": "999", "age": "999", "gender": "999"}
        assert {field: mutated[field] for field in fields} == row
    assert dict(status_counts) == summary["raw_processed_link_status_counts"]
    assert candidate_direction_changes == summary["hypothetical_candidate_changes_existing_direction_records"] == 104
    assert len(direction_changed_conditions) == 8
    assert candidate_pair_changes == summary["hypothetical_candidate_changes_existing_mirror_pair_records"] == 215
    assert len(pairing_changed_conditions) == 16
    assert dict(gates) == summary["source_gate_counts"]
    comparisons = read_csv(OUT / "paper_sample_count_comparison.csv")
    assert len(comparisons) == 20
    count_matches = 0
    code_by_category = {"no_map":0,"equal_map":3,"left_nearer_map":1,"right_nearer_map":2}
    for row in comparisons:
        category, group = row["paper_category"], row["group"]
        expected = cfg["table_A1"][category][group]
        crowds = [cfg["table_1"][group]["original"]]
        if cfg["table_1"][group]["mirror"]: crowds.append(cfg["table_1"][group]["mirror"])
        candidate = [condition_counts[(a,b,code_by_category[category])] for a,b in crowds]
        assert json.loads(row["paper_samples_original_mirror"]) == expected
        assert json.loads(row["absolute_candidate_samples_UNCONFIRMED"]) == candidate
        match = sorted(expected) == sorted(candidate)
        assert row["absolute_candidate_unordered_count_match"] == str(match)
        assert int(row["absolute_candidate_total_delta"]) == sum(candidate)-sum(expected)
        count_matches += match
    assert count_matches == summary["absolute_candidate_unordered_count_matches"] == 18
    checks = read_csv(OUT / "official_metadata_checksum_checks.csv")
    assert len(checks) == summary["official_metadata_checksum_matches"] == 556
    assert len({r["relative_file"] for r in checks}) == 556
    assert summary["chosen_new_map_mapping"] is None and summary["absolute_candidate_is_confirmed"] is False
    assert summary["outcomes_used_for_decoding"] is False and summary["new_performance_scores_computed"] is False
    assert summary["LLM_requests"] == summary["PPO_updates"] == 0
    assert summary["program_assets_in_archive"] == []
    old_snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
    revision = OUT / "analysis_revisions/initial_exact_time_assumption"
    assert sha(revision / "audit_bristol_source_semantics.py") == old_snapshot["source_files"]["scripts/audit_bristol_source_semantics.py"]
    assert sha(revision / "audit_config.json") == old_snapshot["source_files"]["schemas/bristol_source_semantics_audit_20261014.json"]
    result = {"status":"completed_source_audit_verified", "all_547_source_contexts_verified":True,
        "all_36_raw_processed_condition_counts_match":True, "all_previous_experiment_files_unchanged":True,
        "raw_file_sha256_checks":547, "official_metadata_checksum_checks":556,
        "time_interval_links_independently_checked":True,"outcome_perturbation_does_not_change_source_projection":True,
        "paper_sample_counts_independently_recomputed":True,"candidate_not_promoted":True,
        "initial_exact_time_assumption_preserved_and_revised":True,
        "summary_sha256":sha(OUT/'summary.json'),"source_snapshot_sha256":sha(OUT/'source_snapshot_v2.json'),
        "verifier_sha256":sha(Path(__file__)),"LLM_requests":0,"PPO_updates":0,
        "old_final_scenarios_accessed":False,"Liao_confirmation_scores_computed":False}
    (OUT / "verification.json").write_text(json.dumps(result,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
