"""Local annotation contract for future, time-aligned exit-choice records.

This is an intake/validation tool, not a simulator or trained behavior model.
No existing dataset is upgraded by inventing timestamps or negative outcomes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

VERSION = "exit-choice-observation-v1"
OBS_KEYS = {"schema_version", "record_id", "event_id", "participant_group_id",
            "source_ref", "data_origin", "decision_time_s", "observation_time_s",
            "current_exit", "available_exit_ids", "exits"}
EXIT_KEYS = {"exit_id", "distance_m", "distance_kind", "previous_choosers",
             "previous_choosers_time_s", "guide_active", "guide_observed_at_s",
             "guide_started_at_s"}
LABEL_KEYS = {"record_id", "event_id", "window_end_s", "fully_observed",
              "chosen_exit_at_end", "route_changed", "change_time_s", "source_ref"}


class ContractError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ContractError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def text_id(value):
    return isinstance(value, str) and bool(value.strip())


def observation_errors(obs):
    """Return one concrete first error; unknown observations remain null."""
    try:
        require(isinstance(obs, dict) and set(obs) == OBS_KEYS, "Observation fields mismatch; outcomes and notes are not allowed")
        require(obs["schema_version"] == VERSION, "Wrong observation schema version")
        for key in ("record_id", "event_id", "source_ref"):
            require(text_id(obs[key]), f"Missing {key}")
        require(obs["participant_group_id"] is None or text_id(obs["participant_group_id"]), "Invalid participant group")
        require(obs["data_origin"] in {"human_annotation", "synthetic_example"}, "Declare human annotation or synthetic example")
        t = obs["decision_time_s"]
        seen = obs["observation_time_s"]
        require(number(t) and number(seen) and seen <= t, "Observation must be available no later than decision")
        require(isinstance(obs["exits"], list) and len(obs["exits"]) >= 2, "Need all mapped exits, including exits with no observed choices")
        ids = []
        for exit_row in obs["exits"]:
            require(isinstance(exit_row, dict) and set(exit_row) == EXIT_KEYS, "Exit fields mismatch")
            eid = exit_row["exit_id"]
            require(text_id(eid), "Missing exit identity")
            ids.append(eid)
            distance = exit_row["distance_m"]
            require(distance is None or number(distance), "Distance must be nonnegative or unknown")
            require(exit_row["distance_kind"] in {"euclidean", "network", "unknown"}, "Distance type missing")
            require((distance is None) == (exit_row["distance_kind"] == "unknown"), "Unknown distance cannot receive a known distance type")
            count = exit_row["previous_choosers"]
            count_time = exit_row["previous_choosers_time_s"]
            require(count is None or type(count) is int and count >= 0, "Previous chooser count must be integer or unknown")
            if count is None:
                require(count_time is None, "Unknown chooser count must have null measurement time")
            else:
                require(number(count_time) and count_time <= seen, "Chooser count uses a future measurement")
            guide = exit_row["guide_active"]
            guide_time = exit_row["guide_observed_at_s"]
            start = exit_row["guide_started_at_s"]
            require(guide is None or type(guide) is bool, "Guidance presence must be true, false, or unknown")
            if guide is None:
                require(guide_time is None and start is None, "Unknown guidance must not invent presence/start times")
            else:
                require(number(guide_time) and guide_time <= seen, "Guidance observation is after the decision observation")
                require(start is None or number(start) and start <= guide_time, "Guidance start uses future information")
        require(len(ids) == len(set(ids)), "Duplicate physical exit IDs")
        available = obs["available_exit_ids"]
        require(isinstance(available, list) and bool(available) and len(available) == len(set(available)), "Need unique available exit IDs")
        require(set(available).issubset(ids), "Available exits must match physical exit IDs")
        require(obs["current_exit"] is None or obs["current_exit"] in ids, "Current route is not mapped")
        return []
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return [str(exc)]


def label_errors(label, obs):
    try:
        require(not observation_errors(obs), "Validate observation before its label")
        require(isinstance(label, dict) and set(label) == LABEL_KEYS, "Label fields mismatch")
        require(label["record_id"] == obs["record_id"] and label["event_id"] == obs["event_id"], "Label/observation identity mismatch")
        end = label["window_end_s"]
        require(number(end) and end > obs["decision_time_s"], "Follow-up window must end after decision")
        require(type(label["fully_observed"]) is bool, "Declare complete or incomplete follow-up")
        require(text_id(label["source_ref"]), "Label source reference is missing")
        choice = label["chosen_exit_at_end"]
        require(choice is None or choice in [e["exit_id"] for e in obs["exits"]], "End choice not mapped")
        changed = label["route_changed"]
        when = label["change_time_s"]
        require(changed is None or type(changed) is bool, "Change label must be true, false, or unknown")
        if changed is True:
            require(obs["current_exit"] is not None, "Cannot label a route change without a known prior route")
            require(number(when) and obs["decision_time_s"] < when <= end, "Route change time is outside follow-up")
        else:
            require(when is None, "Absent/unknown change cannot invent a change time")
        if changed is False:
            require(label["fully_observed"], "Incomplete follow-up must not be labeled as a route stayer")
            require(obs["current_exit"] is not None and choice == obs["current_exit"], "Stayer requires known same route at the end")
        if label["fully_observed"]:
            require(choice is not None, "Complete follow-up needs its end choice")
            if obs["current_exit"] is not None:
                require(changed is not None, "Complete dynamic follow-up needs a verified change/stay label")
        return []
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return [str(exc)]


def model_input(obs):
    errors = observation_errors(obs)
    require(not errors, "; ".join(errors))
    return {key: deepcopy(obs[key]) for key in
            ("observation_time_s", "decision_time_s", "current_exit", "available_exit_ids", "exits")}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_dataset(observations, labels, registry):
    errors = []
    require(isinstance(registry, dict) and set(registry) == {"schema_version", "partitions"}, "Registry fields mismatch")
    require(registry["schema_version"] == VERSION, "Registry version mismatch")
    parts = registry["partitions"]
    require(isinstance(parts, dict) and set(parts) == {"training", "development", "confirmation"}, "Registry needs three explicit partitions")
    require(all(isinstance(ids, list) and all(text_id(x) for x in ids) for ids in parts.values()), "Partition event IDs invalid")
    flat = [eid for ids in parts.values() for eid in ids]
    require(len(flat) == len(set(flat)), "Event appears in more than one partition or is duplicated")
    partition_by_event = {eid: part for part, ids in parts.items() for eid in ids}
    if not observations:
        errors.append({"record_id": None, "reason": "No time-verified observations supplied; template is not a completed dataset"})
    obs_map = {}
    group_parts = {}
    for obs in observations:
        rid = obs.get("record_id") if isinstance(obs, dict) else None
        rid = rid if text_id(rid) else None
        ee = observation_errors(obs)
        if rid in obs_map:
            ee.append("Duplicate observation record")
        if not ee:
            part = partition_by_event.get(obs["event_id"])
            if part is None:
                ee.append("Event is absent from the frozen partition registry")
            group = obs["participant_group_id"]
            if group is not None and part is not None:
                group_parts.setdefault(group, set()).add(part)
            obs_map[rid] = obs
        errors.extend({"record_id": rid, "reason": e} for e in ee)
    for group, partitions in group_parts.items():
        if len(partitions) > 1:
            errors.append({"record_id": None, "reason": f"Known participant group crosses partitions: {group}"})
    label_ids = set()
    eligible = 0
    for label in labels:
        rid = label.get("record_id") if isinstance(label, dict) else None
        rid = rid if text_id(rid) else None
        if rid in label_ids:
            errors.append({"record_id": rid, "reason": "Duplicate follow-up label"})
        label_ids.add(rid)
        if rid not in obs_map:
            errors.append({"record_id": rid, "reason": "Label lacks valid matching observation"})
            continue
        ee = label_errors(label, obs_map[rid])
        errors.extend({"record_id": rid, "reason": e} for e in ee)
        eligible += int(not ee and label["fully_observed"])
    return {"schema_version": VERSION, "passed": not errors, "observations": len(observations),
            "labels": len(labels), "fully_observed_eligible_labels": eligible, "errors": errors,
            "events_without_known_global_identity_do_not_prove_person_independence": True,
            "no_llm_or_simulation_calls": True}


def synthetic_example():
    exits = [{"exit_id": f"E{j}", "distance_m": 10 * j, "distance_kind": "euclidean",
              "previous_choosers": None, "previous_choosers_time_s": None,
              "guide_active": None, "guide_observed_at_s": None, "guide_started_at_s": None}
             for j in (1, 2, 3)]
    obs = {"schema_version": VERSION, "record_id": "SYNTHETIC:r1", "event_id": "SYNTHETIC:event1",
           "participant_group_id": "SYNTHETIC:person1", "source_ref": "synthetic_example_not_human_data",
           "data_origin": "synthetic_example", "decision_time_s": 10, "observation_time_s": 9,
           "current_exit": "E1", "available_exit_ids": ["E1", "E2", "E3"], "exits": exits}
    label = {"record_id": obs["record_id"], "event_id": obs["event_id"], "window_end_s": 20,
             "fully_observed": True, "chosen_exit_at_end": "E1", "route_changed": False,
             "change_time_s": None, "source_ref": "synthetic_example_not_human_data"}
    return obs, label


def self_check():
    obs, label = synthetic_example()
    checks = {}
    checks["valid_synthetic_complete_stayer"] = not observation_errors(obs) and not label_errors(label, obs)
    bad = deepcopy(obs); bad["chosen_exit"] = "E1"
    checks["outcome_in_observation_rejected"] = bool(observation_errors(bad))
    bad = deepcopy(obs); bad["available_exit_ids"] = [{"invalid": "E1"}]
    checks["malformed_exit_identity_returns_error"] = bool(observation_errors(bad))
    bad = deepcopy(obs); bad["observation_time_s"] = 11
    checks["post_decision_observation_rejected"] = bool(observation_errors(bad))
    bad = deepcopy(obs); bad["exits"][0].update(previous_choosers=2, previous_choosers_time_s=12)
    checks["future_crowd_measurement_rejected"] = bool(observation_errors(bad))
    bad = deepcopy(obs); bad["exits"][0].update(guide_active=True, guide_observed_at_s=9, guide_started_at_s=12)
    checks["future_guidance_start_rejected"] = bool(observation_errors(bad))
    bad_label = deepcopy(label); bad_label["fully_observed"] = False
    checks["incomplete_followup_as_negative_rejected"] = bool(label_errors(bad_label, obs))
    payload = model_input(obs)
    checks["unknown_guidance_preserved_as_null"] = payload["exits"][0]["guide_active"] is None
    checks["identity_and_source_excluded_from_model_input"] = not set(payload) & {"record_id", "event_id", "participant_group_id", "source_ref", "chosen_exit_at_end"}
    registry = {"schema_version": VERSION, "partitions": {"training": [obs["event_id"]], "development": [], "confirmation": []}}
    checks["empty_template_not_completed_data"] = not validate_dataset([], [], registry)["passed"]
    another = deepcopy(obs); another.update(record_id="SYNTHETIC:r2", event_id="SYNTHETIC:event2")
    registry["partitions"]["confirmation"] = [another["event_id"]]
    checks["known_person_group_split_leakage_rejected"] = not validate_dataset([obs, another], [], registry)["passed"]
    require(all(checks.values()), "Contract self-check failed")
    return {"checks": checks, "passed": True, "fixture_type": "synthetic_example_only", "human_samples_added": 0}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("self_check"); check.add_argument("--report", type=Path, required=True)
    valid = sub.add_parser("validate")
    for name in ("observations", "labels", "registry", "report"):
        valid.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        write_new(args.output / "partition_registry.json", {"schema_version": VERSION,
                  "partitions": {"training": [], "development": [], "confirmation": []}})
        obs, label = synthetic_example()
        write_new(args.output / "synthetic_example_observation.json", obs)
        write_new(args.output / "synthetic_example_label.json", label)
        for name in ("observations.jsonl", "labels.jsonl"):
            with (args.output / name).open("x", encoding="utf-8") as f:
                f.write("")
        result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(args.output),
                  "human_records": 0, "status": "empty_annotation_template_not_validated_data"}
    elif args.command == "self_check":
        result = self_check(); write_new(args.report, result)
    else:
        result = validate_dataset(read_jsonl(args.observations), read_jsonl(args.labels),
                                  json.loads(args.registry.read_text(encoding="utf-8")))
        write_new(args.report, result)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
