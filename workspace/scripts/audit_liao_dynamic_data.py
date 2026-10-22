"""Reproducible structural/semantic audit; no model fitting or inference.

Source labels are retrospective annotations. Time relative to a track's first
row is a diagnostic convention, not an asserted acoustic-start timestamp.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

FPS = 16
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/public_data_search_20261013"
DEFAULT_OUT = ROOT / "outputs/liao_dynamic_data_audit_20261013"

# Transcribed from visually inspected supplement pp. 15 and 17 (Tables S1/S3).
REPORTED = {
    "A_3": [(8.6, 3.8), (9.0, 4.8), (10.8, 8.4)],
    "A_5": [(8.6, 5.3)], "A_7": [(8.2, 2.9)],
    "A_9": [(22.9, 11.5), (25.4, 12.8), (23.4, 19.9), (24.2, 21.4), (11.9, 2.8)],
    "A_10": [(21.9, 6.5), (25.6, 15.1), (10.6, 4.9)],
    "B_3": [(11.2, 5.9), (11.9, 5.9), (11.9, 7.8)],
    "B_4": [(10.2, 6.8), (13.8, 8.7)],
    "B_5": [(20.9, 10.9), (21.1, 11.2), (15.3, 4.9), (19.3, 5.6), (20.9, 7.1)],
    "B_6": [(19.4, 6.2), (21.3, 17.4), (18.6, 12.4), (16.3, 9.9), (19.0, 13.1), (16.3, 11.2), (19.5, 13.1)],
    "C_1": [(5.9, 3.1), (6.0, 2.9), (7.2, 5.1)],
    "C_2": [(9.0, 4.8), (10.4, 5.7), (10.4, 4.9)],
    "C_3": [(14.9, 4.9), (16.5, 10.9), (16.4, 9.9), (19.2, 17.2), (12.2, 4.5), (8.2, 4.6), (8.6, 6.4)],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_write(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse(source):
    tracks = defaultdict(list)
    run_headers = []
    for filename in ["rsif20160684_si_002.txt", "rsif20160684_si_003.txt", "rsif20160684_si_004.txt"]:
        path = source / filename
        run = None
        for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            match = re.fullmatch(r"#\s*Experiment\s+([ABC]_\d+)", line)
            if match:
                run = match[1]
                run_headers.append(run)
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if run is None or len(parts) != 5:
                raise ValueError(f"Malformed {filename}:{lineno}: {raw}")
            local_id, frame = int(parts[0]), int(parts[1])
            x, y, label = float(parts[2]), float(parts[3]), int(parts[4])
            valid = range(5) if run.startswith("B") else range(3)
            if label not in valid or not all(math.isfinite(v) for v in [x, y]):
                raise ValueError(f"Invalid values {filename}:{lineno}")
            tracks[(run, local_id)].append(dict(frame=frame, x=x, y=y, label=label,
                                                source_file=filename, source_line=lineno))
    if len(run_headers) != len(set(run_headers)):
        raise ValueError("Repeated run header")
    return tracks


def blocks(rows):
    result = []
    for row in rows:
        if not result or result[-1]["label"] != row["label"]:
            result.append(dict(label=row["label"], first_frame=row["frame"],
                               last_frame=row["frame"], source_line=row["source_line"], n=1))
        else:
            result[-1]["last_frame"] = row["frame"]
            result[-1]["n"] += 1
    return result


def conditions(run):
    group, ix = run.split("_")
    ix = int(ix)
    if group == "A":
        n1, n2 = (18, 0) if ix <= 2 else ((40, 0) if ix <= 8 else (48, 90))
        widths = {"1": .7, "2": 1.1}
    elif group == "B":
        n1, n2 = (11, 0) if ix == 1 else ((40, 0) if ix <= 4 else (0, 138))
        widths = {"1": .8, "2": .8, "3": .8 if ix <= 4 else 1.2,
                  "4": .8 if ix <= 4 else 1.2}
    else:
        n1, n2 = (67, 0) if ix == 1 else ((71, 0) if ix == 2 else (90, 48))
        widths = {"1": .8, "2": .8 if ix <= 2 else 1.2}
    return dict(holding_I_n=n1, holding_II_n=n2, true_n=n1+n2, exit_widths_m=widths,
                aggregation_n=69 if run in ["C_1", "C_2"] else n1+n2)


def audit(source, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = [source / f"rsif20160684_si_{i:03d}.{ext}"
             for i, ext in [(1, "pdf"), (2, "txt"), (3, "txt"), (4, "txt")]]
    before = {p.name: sha(p) for p in paths}
    download_audit = json.loads((source / "liao_download_audit.json").read_text(encoding="utf-8"))
    registered = {f["name"]: f["sha256"] for item in download_audit for f in item["files"]}
    if before != registered:
        raise ValueError("Original sources differ from intake hashes")
    tracks = parse(source)
    track_records, segment_records, event_records = [], [], []
    for (run, local_id), rows in sorted(tracks.items(), key=lambda z: (z[0][0][0], int(z[0][0].split('_')[1]), z[0][1])):
        seq = blocks(rows)
        frames = [r["frame"] for r in rows]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError(f"Unordered or duplicate frames {run}/{local_id}")
        nonzero_blocks = [b for b in seq if b["label"] != 0]
        switches = sum(a["label"] != b["label"] for a, b in zip(nonzero_blocks, nonzero_blocks[1:]))
        has_zero_between_choices = any(b["label"] == 0 for b in seq[1:-1])
        track_records.append(dict(run=run, local_id=local_id, rows=len(rows),
                                  first_frame=frames[0], last_frame=frames[-1],
                                  span_s=(frames[-1]-frames[0])/FPS,
                                  first_label=rows[0]["label"], last_label=rows[-1]["label"],
                                  first_nonzero_label=nonzero_blocks[0]["label"] if nonzero_blocks else None,
                                  switches_between_nonzero_labels=switches,
                                  zero_rows=sum(r["label"] == 0 for r in rows),
                                  interior_zero_block=has_zero_between_choices,
                                  frame_gaps=sum(b-a != 1 for a, b in zip(frames, frames[1:])),
                                  source_file=rows[0]["source_file"],
                                  source_first_line=rows[0]["source_line"],
                                  source_last_line=rows[-1]["source_line"],
                                  global_person_id=None, reminder_content=None,
                                  reminder_time_s=None, reminder_received=None,
                                  true_exit_crossing_time_s=None))
        previous_nonzero = None
        for ix, b in enumerate(seq):
            segment_records.append(dict(run=run, local_id=local_id, segment_index=ix,
                                        label=b["label"], first_frame=b["first_frame"],
                                        last_frame=b["last_frame"], rows=b["n"],
                                        duration_s=b["n"]/FPS, source_file=rows[0]["source_file"],
                                        source_first_line=b["source_line"]))
            if b["label"] != 0:
                if previous_nonzero is not None and previous_nonzero["label"] != b["label"]:
                    kind = "nonzero_switch"
                    prior = previous_nonzero["label"]
                elif ix > 0 and seq[ix-1]["label"] == 0:
                    kind = "decision_onset_after_zero"
                    prior = 0
                else:
                    previous_nonzero = b
                    continue
                event_records.append(dict(run=run, local_id=local_id, event_kind=kind,
                                          from_label=prior, to_label=b["label"], frame=b["first_frame"],
                                          frame_div_16_s=b["first_frame"]/FPS,
                                          since_track_first_s=(b["first_frame"]-frames[0])/FPS,
                                          last_frame_div_16_s=frames[-1]/FPS,
                                          track_span_s=(frames[-1]-frames[0])/FPS,
                                          new_label_duration_s=b["n"]/FPS,
                                          nonzero_switch_count=switches,
                                          crosses_zero=ix > 0 and seq[ix-1]["label"] == 0,
                                          source_file=rows[0]["source_file"], source_line=b["source_line"],
                                          verified_table_person_id=None))
                previous_nonzero = b

    run_records = []
    for run in sorted({r for r, _ in tracks}, key=lambda r: (r[0], int(r.split('_')[1]))):
        tr = [r for r in track_records if r["run"] == run]
        ev = [e for e in event_records if e["run"] == run]
        all_rows = [r for (r2, _), rows in tracks.items() if r2 == run for r in rows]
        c = conditions(run)
        run_records.append(dict(run=run, person_runs=len(tr), expected_actual_n=c["true_n"],
                                population_matches=len(tr) == c["true_n"],
                                paper_aggregation_n=c["aggregation_n"],
                                rows=len(all_rows), first_frame=min(r["frame"] for r in all_rows),
                                last_frame=max(r["frame"] for r in all_rows),
                                min_x_m=min(r["x"] for r in all_rows), max_x_m=max(r["x"] for r in all_rows),
                                min_y_m=min(r["y"] for r in all_rows), max_y_m=max(r["y"] for r in all_rows),
                                changed_tracks=sum(r["switches_between_nonzero_labels"] > 0 for r in tr),
                                switch_events=sum(e["event_kind"] == "nonzero_switch" for e in ev),
                                decision_onsets=sum(e["event_kind"] == "decision_onset_after_zero" for e in ev),
                                table_S3_cases=len(REPORTED.get(run, [])),
                                zero_rows=sum(r["zero_rows"] for r in tr),
                                frame_gaps=sum(r["frame_gaps"] for r in tr)))

    # Candidate matching is descriptive, not identity reconstruction. Two timing
    # conventions are tested explicitly. Every table row keeps its absent ID.
    temporal_candidates = []
    for run, cases in REPORTED.items():
        events = [e for e in event_records if e["run"] == run]
        for case_ix, (exit_time, change_time) in enumerate(cases, 1):
            for convention, end_key, change_key in [
                ("absolute_frame_div_16", "last_frame_div_16_s", "frame_div_16_s"),
                ("relative_to_first_track_row", "track_span_s", "since_track_first_s")]:
                distances = sorted(events, key=lambda e: abs(e[end_key]-exit_time)+abs(e[change_key]-change_time))
                for rank, e in enumerate(distances[:3], 1):
                    temporal_candidates.append(dict(run=run, table_case=case_ix,
                        table_egress_s=exit_time, table_change_s=change_time,
                        time_convention=convention, candidate_rank=rank,
                        candidate_local_id=e["local_id"], candidate_event_frame=e["frame"],
                        candidate_event_kind=e["event_kind"], candidate_from=e["from_label"],
                        candidate_to=e["to_label"], candidate_end_s=e[end_key],
                        candidate_change_s=e[change_key], end_error_s=e[end_key]-exit_time,
                        change_error_s=e[change_key]-change_time,
                        max_abs_error_s=max(abs(e[end_key]-exit_time),abs(e[change_key]-change_time)),
                        sum_abs_error_s=abs(e[end_key]-exit_time)+abs(e[change_key]-change_time),
                        source_file=e["source_file"], source_line=e["source_line"],
                        identity_verified=False))

    csv_write(output / "person_run_cases.csv", track_records)
    csv_write(output / "label_segments.csv", segment_records)
    csv_write(output / "label_events.csv", event_records)
    csv_write(output / "run_summary.csv", run_records)
    csv_write(output / "table_s3_temporal_candidates.csv", temporal_candidates)
    dump(output / "source_table_transcription.json", {
        "source": "rsif20160684_si_001.pdf", "visually_verified_pages": [15,17,22],
        "conditions": {r["run"]: conditions(r["run"]) for r in run_records},
        "table_s3": {run: [dict(table_case=i, egress_s=t, change_s=c, local_id=None)
                               for i, (t,c) in enumerate(vals, 1)] for run, vals in REPORTED.items()}})
    after = {p.name: sha(p) for p in paths}
    if after != before:
        raise ValueError("Source changed during audit")
    timing_summary = {}
    for convention in ["absolute_frame_div_16", "relative_to_first_track_row"]:
        nearest = [r for r in temporal_candidates if r["time_convention"] == convention and r["candidate_rank"] == 1]
        timing_summary[convention] = dict(nearest_with_both_errors_le_0_2s=sum(r["max_abs_error_s"] <= .2 for r in nearest),
                                         table_rows=len(nearest),
                                         warning="Nearest candidates are not verified IDs; no one-to-one identity assignment.")
    summary = dict(audit_version=1, sampling_fps=FPS, coordinate_units="metres",
                   rows=sum(len(rows) for rows in tracks.values()), runs=len(run_records),
                   person_run_keys=len(tracks), documented_participant_pool=138,
                   cross_run_person_identity_verified=False,
                   nonzero_changed_tracks=sum(r["switches_between_nonzero_labels"] > 0 for r in track_records),
                   nonzero_switch_events=sum(e["event_kind"] == "nonzero_switch" for e in event_records),
                   decision_onset_events=sum(e["event_kind"] == "decision_onset_after_zero" for e in event_records),
                   tracks_with_zero=sum(r["zero_rows"] > 0 for r in track_records),
                   zero_rows=sum(r["zero_rows"] for r in track_records),
                   interior_zero_tracks=sum(r["interior_zero_block"] for r in track_records),
                   frame_gaps=sum(r["frame_gaps"] for r in track_records),
                   table_S3_cases=sum(len(v) for v in REPORTED.values()),
                   population_matches_all_runs=all(r["population_matches"] for r in run_records),
                   temporal_diagnostic=timing_summary,
                   source_sha256_before=before, source_sha256_after=after,
                   sources_unchanged=True,
                   interpretation={"C_counts": "RESOLVED: actual 67/71; pooled condition 69 per Table S1.",
                     "B5": "UNRESOLVED semantics: 3 switching tracks, 4 switches, 5 table cases; inspect 0->1 onsets as candidates only.",
                     "time_zero": "Frame spacing known; acoustic-start offsets and egress line crossings not independently verified.",
                     "geometry": "Exit label order and widths verified; numeric wall/exit coordinates not digitized.",
                     "future_leakage": "Do not pass future labels, complete paths, egress time, run/local ID or table-derived identity to predictors.",
                     "new_subjects": "Run/layout holdout cannot establish new-person generalization without linked subject IDs.",
                     "scope": "Low-pressure autonomous egress, no observed guidance treatment or smoke."})
    dump(output / "audit_summary.json", summary)
    plot_b5(tracks, output)
    return summary


def plot_b5(tracks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15,6))
    selected = [37,49,63,64,77]
    for col, local_id in enumerate(selected):
        rows = tracks[("B_5", local_id)]
        for label, color in [(0,"gray"),(1,"tab:blue"),(2,"tab:orange"),(3,"tab:green"),(4,"tab:red")]:
            chosen = [r for r in rows if r["label"] == label]
            if chosen:
                axes[0,col].scatter([r['x'] for r in chosen],[r['y'] for r in chosen],s=4,c=color,label=str(label))
        axes[0,col].set_title(f"B_5 local ID {local_id}")
        axes[0,col].set_aspect("equal", adjustable="datalim")
        axes[0,col].set_xlabel("X (m)");axes[0,col].set_ylabel("Y (m)")
        axes[1,col].step([(r['frame']-rows[0]['frame'])/FPS for r in rows],
                          [r['label'] for r in rows],where="post")
        axes[1,col].set_ylim(-.2,4.2);axes[1,col].set_yticks(range(5))
        axes[1,col].set_xlabel("Seconds since first track row")
        axes[1,col].set_ylabel("Annotated exit label")
    fig.suptitle("Audit candidates only: label 0 is undecided; walls and exit crossing lines not reconstructed")
    fig.tight_layout()
    fig.savefig(output / "b5_label_diagnostic.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.source, args.output)
    print(json.dumps({k: result[k] for k in ["rows","runs","person_run_keys","nonzero_changed_tracks",
                "nonzero_switch_events","decision_onset_events","zero_rows","frame_gaps",
                "population_matches_all_runs","sources_unchanged","temporal_diagnostic"]},indent=2))
