"""Build traceable, past-only input records and a separate outcome manifest.

No predictions are produced. This is a draft data-interface preflight, not a
frozen confirmation evaluation. A window's opaque key may join outcomes but
must not be serialized into a model prompt.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_liao_dynamic_data import ROOT, SOURCE, DEFAULT_OUT, parse, sha


def prepare():
    protocol_path = ROOT / "schemas/liao_dynamic_prediction_protocol_20261013.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tracks = parse(SOURCE)
    output = DEFAULT_OUT / "draft_prediction_interface"
    output.mkdir(exist_ok=True)
    history, horizon, stride = (protocol[k] for k in ["history_frames", "horizon_frames", "stride_frames"])
    outcomes, inputs, exclusions = [], [], []
    for (run, local_id), rows in tracks.items():
        lookup = {r["frame"]: r for r in rows}
        first, last = rows[0]["frame"], rows[-1]["frame"]
        group = ("quarantine" if run in protocol["quarantine_runs"] else
                 "proposed_development" if run in protocol["proposed_development_runs"] else
                 "proposed_confirmation" if run in protocol["proposed_confirmation_runs"] else "unassigned")
        for frame in range(first+history, last+1, stride):
            key = hashlib.sha256(f"liao-draft:{run}:{local_id}:{frame}".encode()).hexdigest()[:24]
            trace = dict(window_key=key, run=run, local_id=local_id, current_frame=frame,
                         partition=group, source_file=rows[0]["source_file"])
            if frame+horizon > last:
                exclusions.append({**trace, "reason": "incomplete_future_right_censored"})
                continue
            required = range(frame-history, frame+horizon+1)
            if any(f not in lookup for f in required):
                exclusions.append({**trace, "reason": "frame_gap"})
                continue
            past = [lookup[f] for f in range(frame-history, frame+1)]
            future = [lookup[f] for f in range(frame+1, frame+horizon+1)]
            current_label = lookup[frame]["label"]
            if current_label == 0 or any(r["label"] == 0 for r in future):
                exclusions.append({**trace, "reason": "unresolved_zero_in_current_or_future"})
                continue
            # Input branch reads only positions at or before the cutoff. Future
            # and current labels are read by the separate outcomes branch.
            end = past[-1]
            positions = [dict(relative_time_s=(r["frame"]-frame)/16,
                              x_m=r["x"], y_m=r["y"]) for r in past]
            features = dict(history_s=history/16, forecast_horizon_s=horizon/16,
                            positions=positions,
                            mean_vx_m_s=(end["x"]-past[0]["x"])/(history/16),
                            mean_vy_m_s=(end["y"]-past[0]["y"])/(history/16))
            inputs.append(dict(window_key=key, features=features))
            different = [r for r in future if r["label"] != current_label]
            outcomes.append({**trace, "current_annotated_target_eval_only": current_label,
                             "future_target": future[-1]["label"],
                             "switch_within_horizon": int(bool(different)),
                             "first_different_frame": different[0]["frame"] if different else None,
                             "input_first_source_line": past[0]["source_line"],
                             "input_last_source_line": past[-1]["source_line"],
                             "target_last_source_line": future[-1]["source_line"],
                             "eligible_primary": group != "quarantine"})

    with (output / "past_only_inputs.jsonl").open("w", encoding="utf-8") as f:
        for row in inputs:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, records in [("outcomes_and_provenance.csv", outcomes), ("exclusions.csv", exclusions)]:
        with (output / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    parts = {}
    for part in sorted({r["partition"] for r in outcomes}):
        selected = [r for r in outcomes if r["partition"] == part]
        parts[part] = dict(windows=len(selected), positive_windows=sum(r["switch_within_horizon"] for r in selected),
                          person_run_keys=len({(r['run'], r['local_id']) for r in selected}),
                          positive_person_run_keys=len({(r['run'],r['local_id']) for r in selected if r['switch_within_horizon']}),
                          unique_transition_frames=len({(r['run'],r['local_id'],r['first_different_frame']) for r in selected if r['switch_within_horizon']}))
    assert len({r['window_key'] for r in outcomes}) == len(outcomes)
    assert all(max(v['relative_time_s'] for v in row['features']['positions']) == 0 for row in inputs)
    assert all(min(v['relative_time_s'] for v in row['features']['positions']) == -history/16 for row in inputs)
    summary = dict(status="draft_interface_only_no_predictions", protocol_sha256=sha(protocol_path),
                   partitions=parts, exclusion_counts=dict(Counter(r["reason"] for r in exclusions)),
                   input_records=len(inputs), outcome_records=len(outcomes),
                   note="Includes quarantine for audit only. Strip window_key before prompting. No dataset is described as wholly unseen.")
    (output / "interface_summary.json").write_text(json.dumps(summary, indent=2)+"\n",encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
