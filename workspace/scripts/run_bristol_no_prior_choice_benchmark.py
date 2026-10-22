"""One frozen 34-request scene-only ablation, scored after new inference.

Preserves historical baselines and the six probe responses. No fitting, prompt
search, outcome-fed input, fallback prediction, or held-out evacuation access.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_bristol_prior_mechanism_probe import api, canonical, object_sha, sha, validate_content, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "schemas/bristol_no_prior_choice_benchmark_20261013.json"
OUTPUT = ROOT / "outputs/bristol_no_prior_choice_benchmark_20261013"


def condition(row):
    return tuple(int(row[k]) for k in ["assigned_left_crowd", "assigned_right_crowd", "knowledge_code"])


def family(row):
    a, b, k = condition(row)
    return min(a, b), max(a, b), k


def prepare():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads((ROOT / cfg["source_manifest"]).read_text(encoding="utf-8"))
    probe = json.loads((ROOT / cfg["prompt_source_manifest"]).read_text(encoding="utf-8"))
    systems = {t["payload"]["messages"][0]["content"] for t in probe["tasks"]}
    assert len(systems) == 1
    system = systems.pop()
    template = probe["tasks"][0]["payload"]
    tasks = []
    for old in source["tasks"]:
        obs = json.loads(old["payload"]["messages"][1]["content"])
        del obs["training_base_probability_left"]
        user = canonical(obs)
        payload = {**template, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        tasks.append({"key": old["key"], "condition": old["condition"], "payload": payload,
            "prompt_sha256": hashlib.sha256((system + "\n" + user).encode()).hexdigest(),
            "payload_sha256": object_sha(payload)})
    random.Random(cfg["random_run_order_seed"]).shuffle(tasks)
    assert len(tasks) == len({t["key"] for t in tasks}) == cfg["expected_requests"]
    for index, task in enumerate(tasks, 1):
        task["request_order"] = index
    protected = [cfg["source_manifest"], cfg["prompt_source_manifest"], cfg["evaluation_source"],
        "schemas/bristol_context_choice_benchmark_20261013.json",
        "scripts/run_bristol_context_choice_benchmark.py",
        "scripts/run_bristol_prior_mechanism_probe.py",
        "schemas/bristol_prior_mechanism_probe_20261013.json",
        "outputs/bristol_context_choice_benchmark_20261013/responses.jsonl",
        "outputs/bristol_context_choice_benchmark_20261013/baseline_predictions.csv",
        "outputs/bristol_context_choice_benchmark_20261013/summary.json",
        "outputs/bristol_context_choice_benchmark_20261013/verification.json",
        "outputs/bristol_prior_mechanism_probe_20261013/responses.jsonl",
        "outputs/bristol_prior_mechanism_probe_20261013/summary.json",
        "outputs/bristol_prior_mechanism_probe_20261013/verification.json",
        "outputs/dynamic_information_search_20261013/audit/bristol_context_cases.csv"]
    plan = {"config_sha256": sha(CONFIG), "runner_sha256": sha(__file__),
        "protected_previous_files": {p: sha(ROOT / p) for p in protected},
        "expected_requests": len(tasks), "model_digest": cfg["LLM"]["digest"], "tasks": tasks,
        "evaluation_plan": {k: cfg[k] for k in ["methods", "primary_metric", "secondary_metrics", "decision_rule", "secondary_subset", "inference_reporting"]},
        "outcomes_in_prompts": False, "PPO_updates": 0, "old_final_scenarios_accessed": False,
        "Liao_confirmation_scores_computed": False}
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "frozen_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "frozen_at_utc"} != plan:
            raise ValueError("Frozen inputs/code/config or previous files changed; refusing replacement")
        return cfg, saved
    plan["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, plan)
    print(json.dumps({"status": "frozen_before_inference", "requests": len(tasks), "manifest_sha256": sha(path)}), flush=True)
    return cfg, plan


def ledger(plan):
    path = OUTPUT / "responses.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    saved = {r["key"]: r for r in rows}
    assert len(rows) == len(saved)
    tasks = {t["key"]: t for t in plan["tasks"]}
    for key, row in saved.items():
        assert key in tasks and row["manifest_sha256"] == sha(OUTPUT / "frozen_manifest.json")
        assert row["model_digest"] == plan["model_digest"]
        assert all(row[k] == tasks[key][k] for k in ["prompt_sha256", "payload_sha256"])
    return saved


def evaluate(rows, method):
    cells = defaultdict(list)
    losses, briers, correct = [], [], []
    for row in rows:
        p = min(1 - 1e-9, max(1e-9, float(row[method])))
        y = int(row["observed_exit"]) == 1
        loss = -math.log(p if y else 1 - p)
        brier = (p - y) ** 2
        losses.append(loss); briers.append(brier); correct.append((p >= .5) == y)
        cells[condition(row)].append((p, y, loss, brier))
    cell_loss = [statistics.mean(x[2] for x in values) for values in cells.values()]
    cell_brier = [statistics.mean(x[3] for x in values) for values in cells.values()]
    cell_errors = []
    for values in cells.values():
        assert len({x[0] for x in values}) == 1
        cell_errors.append(abs(values[0][0] - statistics.mean(x[1] for x in values)))
    return {"records": len(rows), "conditions": len(cells),
        "condition_macro_NLL": statistics.mean(cell_loss), "condition_macro_Brier": statistics.mean(cell_brier),
        "participant_NLL": statistics.mean(losses), "participant_accuracy": statistics.mean(correct),
        "mean_absolute_condition_probability_error": statistics.mean(cell_errors)}


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(cfg, plan):
    saved = ledger(plan)
    if len(saved) != cfg["expected_requests"] or not all(r["valid"] for r in saved.values()):
        write_json(OUTPUT / "incomplete_status.json", {"planned": 34, "completed": len(saved),
            "valid": sum(r["valid"] for r in saved.values()), "full_scores_computed": False})
        print("Incomplete/invalid responses preserved; full comparison not scored.", flush=True)
        return
    rows = list(csv.DictReader((ROOT / cfg["evaluation_source"]).open(encoding="utf-8-sig")))
    assert len(rows) == cfg["eligible_records"] and len({condition(r) for r in rows}) == 34
    for row in rows:
        row["qwen3_4b_no_prior"] = saved["_".join(map(str, condition(row)))]["p_left"]
        row["no_prior_LLM_valid"] = True
    methods = {method: evaluate(rows, method) for method in cfg["methods"]}
    historical = json.loads((ROOT / "outputs/bristol_context_choice_benchmark_20261013/summary.json").read_text(encoding="utf-8"))
    for name in cfg["methods"][:-1]:
        for metric, value in methods[name].items():
            assert math.isclose(value, historical["methods"][name][metric], rel_tol=0, abs_tol=1e-12)
    table(OUTPUT / "all_predictions.csv", rows)
    condition_rows = []
    for cond in sorted({condition(r) for r in rows}):
        subset = [r for r in rows if condition(r) == cond]
        for name in cfg["methods"]:
            condition_rows.append({"condition": canonical(cond), "family": canonical(family(subset[0])),
                "method": name, "p_left": float(subset[0][name]),
                "observed_left_fraction": statistics.mean(int(r["observed_exit"]) == 1 for r in subset),
                **evaluate(subset, name)})
    table(OUTPUT / "per_condition_metrics.csv", condition_rows)
    family_rows = []
    for fam in sorted({family(r) for r in rows}):
        subset = [r for r in rows if family(r) == fam]
        for name in cfg["methods"]:
            family_rows.append({"family": canonical(fam), "method": name, **evaluate(subset, name)})
    table(OUTPUT / "per_family_metrics.csv", family_rows)
    mirrors = []
    for fam in sorted({family(r) for r in rows}):
        a, b, k = fam
        for name in cfg["methods"]:
            pa = float(next(r for r in rows if condition(r) == (a, b, k))[name])
            pb = float(next(r for r in rows if condition(r) == (b, a, k))[name])
            mirrors.append({"family": canonical(fam), "method": name, "p_left_original": pa,
                "p_left_mirror": pb, "self_mirror": a == b, "absolute_mirror_error": abs(pa + pb - 1)})
    table(OUTPUT / "mirror_consistency.csv", mirrors)
    new = methods["qwen3_4b_no_prior"]
    comparisons = {}
    for name in cfg["methods"][:-1]:
        pairs = []
        for fam in sorted({family(r) for r in rows}):
            subset = [r for r in rows if family(r) == fam]
            pairs.append(evaluate(subset, "qwen3_4b_no_prior")["condition_macro_NLL"] - evaluate(subset, name)["condition_macro_NLL"])
        old = methods[name]
        comparisons[name] = {"NLL_delta_new_minus_reference": new["condition_macro_NLL"] - old["condition_macro_NLL"],
            "NLL_relative_reduction": 1 - new["condition_macro_NLL"] / old["condition_macro_NLL"],
            "Brier_delta_new_minus_reference": new["condition_macro_Brier"] - old["condition_macro_Brier"],
            "accuracy_delta_percentage_points": 100 * (new["participant_accuracy"] - old["participant_accuracy"]),
            "mirror_families_lower_NLL": sum(d < -1e-12 for d in pairs),
            "mirror_families_higher_NLL": sum(d > 1e-12 for d in pairs),
            "mirror_families_tied_NLL": sum(abs(d) <= 1e-12 for d in pairs)}
    old_probe = [json.loads(line) for line in (ROOT / "outputs/bristol_prior_mechanism_probe_20261013/responses.jsonl").read_text().splitlines()]
    probe_tasks = json.loads((ROOT / cfg["prompt_source_manifest"]).read_text())["tasks"]
    no_prior_tasks = [t for t in probe_tasks if t["arm"] == "no_prior"]
    probe_conds = {tuple(t["condition"]) for t in no_prior_tasks}
    overlaps = []
    for task in no_prior_tasks:
        old_row = next(r for r in old_probe if r["key"] == task["key"])
        key = "_".join(map(str, task["condition"]))
        current = saved[key]
        assert current["prompt_sha256"] == task["prompt_sha256"] and current["payload_sha256"] == task["payload_sha256"]
        overlaps.append({"condition": task["condition"], "previous": old_row["p_left"], "new": current["p_left"],
            "absolute_difference": abs(current["p_left"] - old_row["p_left"])})
    remaining = [r for r in rows if condition(r) not in probe_conds]
    subgroup_definitions = {"remaining_28_conditions": remaining,
        "no_map": [r for r in rows if r["map_information"] == "no_map"],
        "map_available": [r for r in rows if r["map_information"] != "no_map"],
        "busy": [r for r in rows if r["crowd_state_design"] == "busy"],
        "free": [r for r in rows if r["crowd_state_design"] == "free"]}
    # Remaining subset fixed before inference; other subgroups are descriptive diagnostics.
    subgroups = {label: {name: evaluate(subset, name) for name in cfg["methods"]} for label, subset in subgroup_definitions.items()}
    mirror_summary = {name: {"families": 18,
        "mean_absolute_error": statistics.mean(r["absolute_mirror_error"] for r in mirrors if r["method"] == name),
        "max_absolute_error": max(r["absolute_mirror_error"] for r in mirrors if r["method"] == name)} for name in cfg["methods"]}
    frequency = methods["training_frequency"]
    signal = new["condition_macro_NLL"] < frequency["condition_macro_NLL"] and new["condition_macro_Brier"] < frequency["condition_macro_Brier"]
    result = {"status": "completed_exploratory_development_ablation", "records": len(rows), "conditions": 34,
        "mirror_families": 18, "fresh_LLM_requests": 34, "valid_LLM_responses": 34,
        "methods": methods, "comparisons": comparisons, "subgroups": subgroups,
        "mirror_summary": mirror_summary, "diagnostic_repeatability": overlaps,
        "development_signal_over_frequency_on_both_probability_metrics": signal,
        "decision": "Independent confirmation warranted as development follow-up; not deployment evidence" if signal else "Close this fixed 4B scene-only configuration; no further prompt selection",
        "total_request_seconds": sum(r["wall_seconds"] for r in saved.values()),
        "manifest_sha256": sha(OUTPUT / "frozen_manifest.json"), "responses_sha256": sha(OUTPUT / "responses.jsonl"),
        "no_new_fitting": True, "no_fallbacks": True, "no_prompt_tuning": True,
        "PPO_updates": 0, "old_final_scenarios_accessed": False, "Liao_confirmation_scores_computed": False,
        "limitations": [cfg[k] for k in ["development_visibility", "inference_reporting", "fairness_disclosure", "historical_comparison_disclosure"]]}
    write_json(OUTPUT / "summary.json", result)
    print(json.dumps({k: result[k] for k in ["status", "records", "conditions", "methods", "comparisons", "development_signal_over_frequency_on_both_probability_metrics", "decision", "total_request_seconds"]}, indent=2), flush=True)


def run(cfg, plan):
    model = next(m for m in api("/api/tags", timeout=5)["models"] if m["name"] == cfg["LLM"]["model"])
    assert model["digest"] == plan["model_digest"]
    write_json(OUTPUT / "runtime_model_before.json", model)
    saved = ledger(plan)
    errors = 0
    for task in plan["tasks"]:
        if task["key"] in saved:
            continue
        row = {"key": task["key"], "prompt_sha256": task["prompt_sha256"], "payload_sha256": task["payload_sha256"],
            "model_digest": plan["model_digest"], "manifest_sha256": sha(OUTPUT / "frozen_manifest.json"),
            "started_at_utc": datetime.now(timezone.utc).isoformat()}
        start = time.monotonic()
        try:
            row["response"] = api("/api/chat", task["payload"], timeout=90)
            assert row["response"]["model"] == cfg["LLM"]["model"]
            row.update(valid=True, p_left=validate_content(row["response"])); errors = 0
        except Exception as exc:
            row.update(valid=False, p_left=None, error=f"{type(exc).__name__}: {exc}"); errors += 1
        row["wall_seconds"] = time.monotonic() - start
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (OUTPUT / "responses.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        saved[row["key"]] = row
        if len(saved) == 1 or len(saved) % 5 == 0 or not row["valid"]:
            print(json.dumps({"completed": len(saved), "planned": 34, "valid": row["valid"]}), flush=True)
        if errors >= cfg["max_consecutive_errors"]:
            break
    ending = next(m for m in api("/api/tags", timeout=5)["models"] if m["name"] == cfg["LLM"]["model"])
    assert ending["digest"] == plan["model_digest"]
    write_json(OUTPUT / "runtime_model_after.json", ending)
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
