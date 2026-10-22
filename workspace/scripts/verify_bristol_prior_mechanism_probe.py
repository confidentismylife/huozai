"""Independent input, provenance, raw-output and metric verification.

Without responses this performs the pre-inference design audit. With responses
it also checks completeness and recomputes all mechanism metrics independently.
No human outcomes or held-out evacuation files are accessed.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bristol_prior_mechanism_probe_20261013"
CFG = ROOT / "schemas/bristol_prior_mechanism_probe_20261013.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def close(x, y):
    assert math.isclose(x, y, rel_tol=0, abs_tol=1e-12), (x, y)


def verify():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    frozen = OUT / "frozen_manifest.json"
    plan = json.loads(frozen.read_text(encoding="utf-8"))
    assert plan["config_sha256"] == sha(CFG)
    assert plan["runner_sha256"] == sha(ROOT / "scripts/run_bristol_prior_mechanism_probe.py")
    assert all(sha(ROOT / path) == value for path, value in plan["protected_previous_files"].items())
    source = json.loads((ROOT / cfg["source_manifest"]).read_text(encoding="utf-8"))
    originals = {tuple(task["condition"]): task for task in source["tasks"]}
    tasks = plan["tasks"]
    expected = {(tuple(c), a["arm"]): a["prior"] for pair in cfg["mirror_pairs"]
                for c in pair["conditions"] for a in cfg["arms"]}
    assert len(tasks) == len(expected) == 18
    assert {(tuple(t["condition"]), t["arm"]) for t in tasks} == set(expected)
    assert [t["request_order"] for t in tasks] == list(range(1, 19))
    assert len({t["key"] for t in tasks}) == 18
    assert len({t["payload"]["messages"][0]["content"] for t in tasks}) == 1
    scenes = {}
    for task in tasks:
        cond = tuple(task["condition"])
        original = originals[cond]["payload"]
        obs = json.loads(task["payload"]["messages"][1]["content"])
        baseline_obs = json.loads(original["messages"][1]["content"])
        original_scene = {k: v for k, v in baseline_obs.items() if k != "training_base_probability_left"}
        scene = {k: v for k, v in obs.items() if k != "training_base_probability_left"}
        assert scene == original_scene
        assert task["prior"] == expected[(cond, task["arm"])]
        if task["prior"] is None:
            assert set(obs) == set(original_scene)
        else:
            assert set(obs) == set(original_scene) | {"training_base_probability_left"}
            assert obs["training_base_probability_left"] == task["prior"]
        assert task["scene_sha256"] == object_sha(scene)
        assert task["payload_sha256"] == object_sha(task["payload"])
        messages = task["payload"]["messages"]
        assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == original["messages"][0]["content"].replace(
            cfg["system_sentence_original"], cfg["system_sentence_replacement"])
        assert task["prompt_sha256"] == hashlib.sha256((messages[0]["content"] + "\n" + messages[1]["content"]).encode()).hexdigest()
        assert task["payload"]["options"] == original["options"] == cfg["LLM"]["options"]
        for field in ["model", "format", "think", "stream", "keep_alive"]:
            assert task["payload"][field] == original[field]
        assert scene["assigned_left_crowd"] == cond[0] and scene["assigned_right_crowd"] == cond[1]
        assert scene["crowd_state_design"] == ("busy" if max(cond[:2]) >= 20 else "free")
        if cond[2] == 0:
            assert scene["nearer_to_final_exit"] == "unknown" and scene["map_information"] == "no_map"
        else:
            near = "left" if (cond[0] > cond[1]) == (cond[2] == 1) else "right"
            assert scene["nearer_to_final_exit"] == near
        scenes.setdefault(cond, []).append(scene)
    assert len(scenes) == 6 and all(len(values) == 3 and values[0] == values[1] == values[2] for values in scenes.values())
    for pair in cfg["mirror_pairs"]:
        a, b = [scenes[tuple(c)][0] for c in pair["conditions"]]
        assert a["assigned_left_crowd"] == b["assigned_right_crowd"]
        assert a["assigned_right_crowd"] == b["assigned_left_crowd"]
        assert a["nearer_to_final_exit"] == {"left": "right", "right": "left", "unknown": "unknown"}[b["nearer_to_final_exit"]]
        assert a["crowd_state_design"] == b["crowd_state_design"] and a["map_information"] == b["map_information"]
    result = {"design_verified": True, "same_scene_across_arms": True,
        "identical_system_prompt_across_arms": True, "all_scenes_are_source_supported": True,
        "source_mirror_semantics_verified": True, "no_outcome_fields_in_inputs": True,
        "old_benchmark_files_unchanged": True, "requests": 18,
        "manifest_sha256": sha(frozen), "human_accuracy_computed": False,
        "old_final_scenarios_accessed": False, "Liao_confirmation_scores_computed": False, "PPO_updates": 0}
    ledger = OUT / "responses.jsonl"
    if not ledger.exists():
        result["status"] = "pre_inference_design_verified"
        target = OUT / "design_verification.json"
    else:
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == len(tasks) == 18, "Inference is incomplete"
        assert [r["key"] for r in rows] == [t["key"] for t in tasks]
        response_values = {}
        for row, task in zip(rows, tasks):
            assert row["manifest_sha256"] == sha(frozen) and row["model_digest"] == plan["model_digest"]
            assert row["prompt_sha256"] == task["prompt_sha256"] and row["payload_sha256"] == task["payload_sha256"]
            assert datetime.fromisoformat(plan["frozen_at_utc"]) <= datetime.fromisoformat(row["started_at_utc"])
            assert datetime.fromisoformat(row["started_at_utc"]) <= datetime.fromisoformat(row["finished_at_utc"])
            assert row["wall_seconds"] > 0
            if row["valid"]:
                response = row["response"]
                assert response["done"] is True and response["model"] == cfg["LLM"]["model"]
                obj = json.loads(response["message"]["content"])
                assert set(obj) == {"p_left"}
                value = obj["p_left"]
                assert not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                close(value, row["p_left"])
                response_values[(tuple(task["condition"]), task["arm"])] = value
            else:
                assert row["p_left"] is None and "error" in row
        summary_path = OUT / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["status"] == "completed" and summary["completed_requests"] == 18
        assert summary["valid_responses"] == sum(r["valid"] for r in rows)
        assert summary["invalid_responses"] == sum(not r["valid"] for r in rows)
        assert summary["manifest_sha256"] == sha(frozen) and summary["responses_sha256"] == sha(ledger)
        with (OUT / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        assert len(predictions) == 18
        for pred, row, task in zip(predictions, rows, tasks):
            assert pred["key"] == row["key"] and pred["arm"] == task["arm"]
            assert pred["valid"] == str(row["valid"])
            if row["valid"]:
                close(float(pred["p_left"]), row["p_left"])
            else:
                assert pred["p_left"] == ""
        explicit = [(key, prior) for key, prior in expected.items() if prior is not None and key in response_values]
        copies = sum(abs(response_values[key] - prior) <= cfg["exact_copy_tolerance"] for key, prior in explicit)
        assert summary["valid_explicit_prior_responses"] == len(explicit) and summary["exact_prior_copies"] == copies
        sensitivities = []
        for entry in summary["scene_effects"]:
            cond = tuple(entry["condition"])
            for arm in cfg["arms"]:
                assert entry[arm["arm"]] == response_values.get((cond, arm["arm"]))
            lo, hi = entry["prior_20"], entry["prior_80"]
            if lo is not None and hi is not None:
                close(entry["prior_delta"], hi - lo)
                close(entry["prior_sensitivity"], (hi - lo) / 0.6)
                sensitivities.append((hi - lo) / 0.6)
        if sensitivities:
            close(summary["mean_prior_sensitivity"], statistics.mean(sensitivities))
        for effect, pair in zip(summary["mirror_effects"], cfg["mirror_pairs"]):
            assert effect["pair_id"] == pair["pair_id"] and effect["conditions"] == pair["conditions"]
            a, b = map(tuple, pair["conditions"])
            for arm in cfg["arms"]:
                x, y = response_values.get((a, arm["arm"])), response_values.get((b, arm["arm"]))
                actual = effect["same_prior_scene_contrast"][arm["arm"]]
                if x is not None and y is not None:
                    close(actual, x - y)
                else:
                    assert actual is None
            for name, aa, bb in [("no_prior", "no_prior", "no_prior"), ("20_vs_80", "prior_20", "prior_80"), ("80_vs_20", "prior_80", "prior_20")]:
                x, y = response_values.get((a, aa)), response_values.get((b, bb))
                actual = effect["absolute_mirror_error"][name]
                if x is not None and y is not None:
                    close(actual, abs(x + y - 1))
                else:
                    assert actual is None
        no_prior = [value for (condition, arm), value in response_values.items() if arm == "no_prior"]
        if no_prior:
            close(summary["no_prior_scene_range"], max(no_prior) - min(no_prior))
        close(summary["total_request_seconds"], sum(r["wall_seconds"] for r in rows))
        close(summary["mean_request_seconds"], statistics.mean(r["wall_seconds"] for r in rows))
        for name in ["runtime_model.json", "runtime_model_after.json"]:
            runtime = json.loads((OUT / name).read_text(encoding="utf-8"))
            assert runtime["model"]["name"] == cfg["LLM"]["model"] and runtime["model"]["digest"] == plan["model_digest"]
        result.update(status="completed_verified", valid_responses=summary["valid_responses"],
            raw_content_matches_predictions=True, all_metrics_recomputed=True,
            frozen_before_first_inference=True, no_fallbacks=True,
            model_digest_verified_before_and_after=True, responses_sha256=sha(ledger), summary_sha256=sha(summary_path))
        target = OUT / "verification.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
