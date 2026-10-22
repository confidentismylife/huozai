"""Describe previously saved labels by published event metadata, no new fits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_two_hour_session_20261012"
PAPER = ROOT / "outputs/exit_choice_alternative_design_20261011/Tong_2025_library_exit_choices.txt"
EVENT_TYPES = {"2019-03-02": "false_alarm", "2019-03-06": "drill",
               "2019-09-11": "drill", "2019-10-31": "false_alarm"}
cases = json.loads((ROOT / "outputs/exit_choice_initial_benchmark_20261012/strict_initial_cases.json").read_text(encoding="utf-8"))
dates = []
for date, kind in EVENT_TYPES.items():
    selected = [r for r in cases if r["date"] == date]
    counts = Counter(r["initial_label"] for r in selected)
    dates.append({"date": date, "published_event_type": kind, "strict_records": len(selected),
                  "exit_counts": [counts[k] for k in (1, 2, 3)]})
types = []
for kind in ("drill", "false_alarm"):
    selected = [r for r in cases if EVENT_TYPES[r["date"]] == kind]
    counts = Counter(r["initial_label"] for r in selected)
    types.append({"published_event_type": kind, "events": 2, "strict_records": len(selected),
                  "exit_counts": [counts[k] for k in (1, 2, 3)]})
result = {"source": PAPER.as_posix(), "source_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest(),
          "locator": "Table 2 event type; Table 3 original floor sample sizes; Section 3.2 knowledge of drills",
          "status": "post-result descriptive metadata audit, not causal or independent confirmation",
          "date_rows": dates, "type_rows": types, "event_type_used_as_new_predictive_feature": False,
          "actor_knowledge_of_event_type_verified": False, "new_llm_calls": 0, "new_fits": 0}
(OUT / "published_event_metadata_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"dates": dates, "types": types, "new_fits": 0}, ensure_ascii=False))
