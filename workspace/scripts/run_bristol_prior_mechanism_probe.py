"""Freeze and run 18 stateless baseline-probability perturbation requests.

This diagnostic reads only the old input manifest, never human outcome tables.
Invalid predictions stay missing. Existing frozen plans and responses cannot be
silently replaced, retried with a new prompt, or combined with another model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "schemas/bristol_prior_mechanism_probe_20261013.json"
OUTPUT = ROOT / "outputs/bristol_prior_mechanism_probe_20261013"
SCENE_FIELDS = ["assigned_left_crowd", "assigned_right_crowd", "map_information",
                "nearer_to_final_exit", "crowd_state_design"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def api(path, payload=None, timeout=90):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request("http://127.0.0.1:11434" + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_content(response):
    obj = json.loads(response["message"]["content"])
    if not isinstance(obj, dict) or set(obj) != {"p_left"}:
        raise ValueError("Output must contain only p_left")
    value = obj["p_left"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Probability must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability outside [0,1]")
    if response.get("done") is not True:
        raise ValueError("Response not complete")
    return float(value)


def prepare():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / cfg["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    lookup = {tuple(task["condition"]): task for task in source["tasks"]}
    tasks = []
    for pair in cfg["mirror_pairs"]:
        for condition in pair["conditions"]:
            original = lookup[tuple(condition)]
            system = original["payload"]["messages"][0]["content"]
            old = cfg["system_sentence_original"]
            if system.count(old) != 1:
                raise ValueError("Original prompt sentence mismatch")
            system = system.replace(old, cfg["system_sentence_replacement"])
            original_obs = json.loads(original["payload"]["messages"][1]["content"])
            obs = {key: original_obs[key] for key in SCENE_FIELDS}
            for arm in cfg["arms"]:
                observation = dict(obs)
                if arm["prior"] is not None:
                    observation["training_base_probability_left"] = arm["prior"]
                content = canonical(observation)
                payload = {
                    "model": cfg["LLM"]["model"],
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": content}],
                    "format": original["payload"]["format"],
                    "think": cfg["LLM"]["think"], "stream": False,
                    "keep_alive": cfg["LLM"]["keep_alive"],
                    "options": cfg["LLM"]["options"]}
                key = "_".join(map(str, condition)) + "__" + arm["arm"]
                tasks.append({"key": key, "pair_id": pair["pair_id"],
                    "condition": condition, "arm": arm["arm"], "prior": arm["prior"],
                    "scene_sha256": object_sha(obs), "payload": payload,
                    "payload_sha256": object_sha(payload),
                    "prompt_sha256": hashlib.sha256((system + "\n" + content).encode()).hexdigest()})
    random.Random(cfg["random_run_order_seed"]).shuffle(tasks)
    for index, task in enumerate(tasks, 1):
        task["request_order"] = index
    if len(tasks) != cfg["expected_requests"] or len({t["key"] for t in tasks}) != len(tasks):
        raise ValueError("Invalid task count")
    protected_paths = [cfg["source_manifest"], "schemas/bristol_context_choice_benchmark_20261013.json",
        "scripts/run_bristol_context_choice_benchmark.py",
        "outputs/bristol_context_choice_benchmark_20261013/responses.jsonl",
        "outputs/bristol_context_choice_benchmark_20261013/summary.json",
        "outputs/bristol_context_choice_benchmark_20261013/verification.json"]
    plan = {"config_sha256": sha(CONFIG), "runner_sha256": sha(__file__),
        "protected_previous_files": {path: sha(ROOT / path) for path in protected_paths},
        "model_digest": cfg["LLM"]["digest"], "expected_requests": len(tasks), "tasks": tasks,
        "human_outcomes_read": False, "old_final_scenarios_accessed": False,
        "Liao_confirmation_scores_computed": False, "PPO_updates": 0}
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "frozen_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        comparison = {key: value for key, value in saved.items() if key != "frozen_at_utc"}
        if comparison != plan:
            raise ValueError("Frozen plan/source/config/code changed; refusing to replace")
        return cfg, saved
    plan["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, plan)
    print(json.dumps({"status": "frozen_before_inference", "requests": len(tasks),
        "manifest_sha256": sha(path), "order": [task["key"] for task in tasks]}), flush=True)
    return cfg, plan


def read_ledger(plan):
    path = OUTPUT / "responses.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    saved = {row["key"]: row for row in rows}
    if len(saved) != len(rows):
        raise ValueError("Duplicate response keys")
    tasks = {task["key"]: task for task in plan["tasks"]}
    for key, row in saved.items():
        if key not in tasks:
            raise ValueError("Unknown response key")
        if row["manifest_sha256"] != sha(OUTPUT / "frozen_manifest.json"):
            raise ValueError("Response/manifest mismatch")
        for field in ["prompt_sha256", "payload_sha256"]:
            if row[field] != tasks[key][field]:
                raise ValueError("Response/request mismatch")
        if row["model_digest"] != plan["model_digest"]:
            raise ValueError("Response/model mismatch")
    return saved


def analyze(cfg, plan):
    saved = read_ledger(plan)
    predictions = []
    for task in plan["tasks"]:
        row = saved.get(task["key"])
        predictions.append({key: task[key] for key in ["request_order", "key", "pair_id", "arm", "prior"]} | {
            "condition": canonical(task["condition"]),
            "valid": bool(row and row["valid"]),
            "p_left": row["p_left"] if row and row["valid"] else None,
            "wall_seconds": row["wall_seconds"] if row else None})
    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)

    def p(condition, arm):
        row = saved.get("_".join(map(str, condition)) + "__" + arm)
        return row["p_left"] if row and row["valid"] else None

    scene_effects = []
    mirror_effects = []
    for pair in cfg["mirror_pairs"]:
        for condition in pair["conditions"]:
            values = {arm["arm"]: p(condition, arm["arm"]) for arm in cfg["arms"]}
            lo, hi = values["prior_20"], values["prior_80"]
            scene_effects.append({"pair_id": pair["pair_id"], "condition": condition, **values,
                "prior_delta": None if lo is None or hi is None else hi - lo,
                "prior_sensitivity": None if lo is None or hi is None else (hi - lo) / 0.6})
        a, b = pair["conditions"]
        contrasts = {arm["arm"]: None if p(a, arm["arm"]) is None or p(b, arm["arm"]) is None
            else p(a, arm["arm"]) - p(b, arm["arm"]) for arm in cfg["arms"]}
        mirror_checks = {}
        for name, aa, bb in [("no_prior", "no_prior", "no_prior"),
                             ("20_vs_80", "prior_20", "prior_80"),
                             ("80_vs_20", "prior_80", "prior_20")]:
            x, y = p(a, aa), p(b, bb)
            mirror_checks[name] = None if x is None or y is None else abs(x + y - 1)
        mirror_effects.append({"pair_id": pair["pair_id"], "conditions": pair["conditions"],
            "same_prior_scene_contrast": contrasts, "absolute_mirror_error": mirror_checks})
    prior_rows = [task for task in plan["tasks"] if task["prior"] is not None
                  and task["key"] in saved and saved[task["key"]]["valid"]]
    no_prior = [entry["no_prior"] for entry in scene_effects if entry["no_prior"] is not None]
    sensitivities = [entry["prior_sensitivity"] for entry in scene_effects if entry["prior_sensitivity"] is not None]
    latencies = [row["wall_seconds"] for row in saved.values()]
    result = {"status": "completed" if len(saved) == plan["expected_requests"] else "partial",
        "planned_requests": plan["expected_requests"], "completed_requests": len(saved),
        "valid_responses": sum(row["valid"] for row in saved.values()),
        "invalid_responses": sum(not row["valid"] for row in saved.values()),
        "valid_explicit_prior_responses": len(prior_rows),
        "exact_prior_copies": sum(abs(saved[task["key"]]["p_left"] - task["prior"]) <= cfg["exact_copy_tolerance"] for task in prior_rows),
        "mean_prior_sensitivity": statistics.mean(sensitivities) if sensitivities else None,
        "no_prior_scene_range": max(no_prior) - min(no_prior) if no_prior else None,
        "scene_effects": scene_effects, "mirror_effects": mirror_effects,
        "total_request_seconds": sum(latencies),
        "mean_request_seconds": statistics.mean(latencies) if latencies else None,
        "manifest_sha256": sha(OUTPUT / "frozen_manifest.json"),
        "responses_sha256": sha(OUTPUT / "responses.jsonl") if saved else None,
        "human_accuracy_computed": False, "old_final_scenarios_accessed": False,
        "Liao_confirmation_scores_computed": False, "PPO_updates": 0,
        "limitations": ["Three selected mirror pairs, one deterministic response per cell.",
            "Perturbed prior values are synthetic; this is not calibrated human prediction evaluation.",
            "One system sentence differs from historical benchmark; only new arms share identical wording.",
            "Mirror symmetry is checked with no prior or complementary priors, never fixed asymmetric priors.",
            "Mechanism behavior under this prompt/model/settings does not establish human modeling or control efficacy."]}
    write_json(OUTPUT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(cfg, plan):
    tags = api("/api/tags", timeout=5)
    model = next(model for model in tags["models"] if model["name"] == cfg["LLM"]["model"])
    if model["digest"] != plan["model_digest"]:
        raise ValueError("Available model digest changed")
    write_json(OUTPUT / "runtime_model.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "model": model})
    saved = read_ledger(plan)
    consecutive_errors = 0
    for task in plan["tasks"]:
        if task["key"] in saved:
            continue
        row = {"key": task["key"], "prompt_sha256": task["prompt_sha256"],
            "payload_sha256": task["payload_sha256"], "model_digest": plan["model_digest"],
            "manifest_sha256": sha(OUTPUT / "frozen_manifest.json"),
            "started_at_utc": datetime.now(timezone.utc).isoformat()}
        before = time.monotonic()
        try:
            row["response"] = api("/api/chat", task["payload"], timeout=90)
            if row["response"].get("model") != cfg["LLM"]["model"]:
                raise ValueError("Response model mismatch")
            row.update(valid=True, p_left=validate_content(row["response"]))
            consecutive_errors = 0
        except Exception as error:
            row.update(valid=False, p_left=None, error=f"{type(error).__name__}: {error}")
            consecutive_errors += 1
        row["wall_seconds"] = time.monotonic() - before
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (OUTPUT / "responses.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        saved[row["key"]] = row
        print(json.dumps({"completed": len(saved), "planned": len(plan["tasks"]), "key": row["key"],
            "valid": row["valid"], "p_left": row["p_left"], "seconds": round(row["wall_seconds"], 2)}), flush=True)
        if consecutive_errors >= cfg["max_consecutive_errors"]:
            break
    ending_tags = api("/api/tags", timeout=5)
    ending_model = next(model for model in ending_tags["models"] if model["name"] == cfg["LLM"]["model"])
    if ending_model["digest"] != plan["model_digest"]:
        raise ValueError("Model changed during execution")
    write_json(OUTPUT / "runtime_model_after.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "model": ending_model})
    analyze(cfg, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "analyze"])
    mode = parser.parse_args().mode
    config, manifest = prepare()
    if mode == "run":
        run(config, manifest)
    elif mode == "analyze":
        analyze(config, manifest)
