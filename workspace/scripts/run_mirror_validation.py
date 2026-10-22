"""Validation-only left/right mirror audit of the already tested route prompt.

The 41 test participants are deliberately never loaded into the scoring path.
This is an exploratory prompt sensitivity analysis, not a new independent test.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.public_route_validation import participant_splits, read_public_trials, route_prompt, score_predictions
from run_local_route_validation import OUTPUT_SCHEMA, api_json, write_csv

VERSION = "snopkova_text_corridor_v1_mirrored"


def swapped(trial):
    return replace(
        trial,
        left_width_m=trial.right_width_m,
        left_length_m=trial.right_length_m,
        left_stairs=trial.right_stairs,
        right_width_m=trial.left_width_m,
        right_length_m=trial.left_length_m,
        right_stairs=trial.left_stairs,
    )


def paired_interval(rows, candidate, reference, column, *, samples=5000, seed=20260928):
    lookup = {(r["participant_id"], r["task_id"], r["method"]): r for r in rows}
    people = sorted({r["participant_id"] for r in rows})
    sums, counts = [], []
    for person in people:
        tasks = {t for p, t, m in lookup if p == person and m == candidate}
        diffs = [
            float(lookup[person, t, candidate][column]) - float(lookup[person, t, reference][column])
            for t in tasks if (person, t, reference) in lookup
        ]
        sums.append(sum(diffs))
        counts.append(len(diffs))
    sums, counts = np.array(sums), np.array(counts)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(people), (samples, len(people)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "candidate": candidate, "reference": reference, "metric": column,
        "participants": int((counts > 0).sum()), "decisions": int(counts.sum()),
        "difference": float(sums.sum() / counts.sum()),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    source = ROOT / "data/public_snopkova/Snopkova_Isovists.zip"
    original_dir = ROOT / "outputs/local_llm_validation/response_cache"
    output_dir = ROOT / "outputs/mirror_validation"
    cache_dir = output_dir / "response_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trials, counts = read_public_trials(str(source))
    split = participant_splits(trials)
    tasks = {t.task_id: t for t in trials}
    tags = api_json(f"{args.base_url}/api/tags")
    models = [m for m in tags.get("models", []) if m.get("name") == args.model]
    if not models:
        raise SystemExit(f"Ollama model not found: {args.model}")
    digest = models[0]["digest"]
    original, mirrored = {}, {}
    failures = []
    new_calls = 0
    for task_id, trial in sorted(tasks.items()):
        old = json.loads((original_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        if old.get("model_digest") != digest or old.get("prompt_hash") != hashlib.sha256(route_prompt(trial).encode()).hexdigest():
            raise ValueError(f"original cache mismatch: {task_id}")
        original[task_id] = float(old["p_left"])
        prompt = route_prompt(swapped(trial))
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        path = cache_dir / f"{task_id}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if cached and all(cached.get(k) == v for k, v in {
            "model_digest": digest, "prompt_hash": prompt_hash, "prompt_version": VERSION,
        }.items()):
            mirrored[task_id] = float(cached["p_left"])
            continue
        if args.max_new_calls is not None and new_calls >= args.max_new_calls:
            continue
        payload = {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": 0, "seed": 20260928, "num_predict": 64},
            "think": False, "stream": False,
        }
        try:
            response = api_json(f"{args.base_url}/api/chat", payload, args.timeout)
            raw = response["message"]["content"]
            probability = float(json.loads(raw)["p_left"])
            if not 0 <= probability <= 1:
                raise ValueError("p_left outside [0, 1]")
        except Exception as exc:
            failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {task_id}: {exc}", flush=True)
            continue
        path.write_text(json.dumps({
            "task_id": task_id, "model": args.model, "model_digest": digest,
            "prompt_version": VERSION, "prompt_hash": prompt_hash, "prompt": prompt,
            "raw_response": raw, "p_left": probability,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_ns": response.get("total_duration"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        mirrored[task_id] = probability
        new_calls += 1
        print(f"{len(mirrored)}/{len(tasks)} {task_id}: mirrored p_left={probability:.3f}", flush=True)

    # Only these participants contribute labels. The original test CSV is not read.
    original_rows, _ = score_predictions(trials, split, original, evaluation_split="validation")
    original_rows = [r for r in original_rows if r["task_id"] in mirrored]
    mirror_rows = []
    for row in original_rows:
        if row["method"] != "local_llm":
            continue
        task = row["task_id"]
        target = float(row["actual_choice"] == "left")
        for method, probability in (
            ("mirror_back", 1 - mirrored[task]),
            ("symmetrized", (original[task] + 1 - mirrored[task]) / 2),
        ):
            mirror_rows.append({
                **row, "method": method, "p_left": probability,
                "correct": float((probability >= 0.5) == bool(target)),
                "brier": (probability - target) ** 2,
            })
    rows = original_rows + mirror_rows
    for row in rows:
        probability = float(row["p_left"])
        row["correct_tie_half"] = (
            0.5 if abs(probability - 0.5) < 1e-12 else float(row["correct"])
        )
    methods = ("local_llm", "mirror_back", "symmetrized", "global_majority", "geometry_rule", "train_task_rate")
    metrics = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        metrics.append({
            "method": method, "participants": len({r["participant_id"] for r in subset}),
            "decisions": len(subset), "accuracy": float(np.mean([r["correct"] for r in subset])),
            "accuracy_ties_half": float(np.mean([r["correct_tie_half"] for r in subset])),
            "ties": sum(abs(float(r["p_left"]) - 0.5) < 1e-12 for r in subset),
            "brier": float(np.mean([r["brier"] for r in subset])),
            "mean_p_left": float(np.mean([r["p_left"] for r in subset])),
            "predicted_left_rate": float(np.mean([r["p_left"] >= 0.5 for r in subset])),
        })
    comparisons = [
        paired_interval(rows, candidate, reference, metric)
        for candidate, reference in (("symmetrized", "local_llm"), ("symmetrized", "train_task_rate"), ("mirror_back", "local_llm"))
        for metric in ("correct", "correct_tie_half", "brier")
    ] if mirror_rows else []
    by_choice = [{
        "method": method, "actual_choice": choice, "decisions": len(subset),
        "accuracy": float(np.mean([r["correct"] for r in subset])),
        "mean_p_left": float(np.mean([r["p_left"] for r in subset])),
    } for method in methods for choice in ("left", "right")
      if (subset := [r for r in rows if r["method"] == method and r["actual_choice"] == choice])]
    symmetry = [{
        "task_id": task, "original_p_left": original[task],
        "mirror_p_left": mirrored[task], "mirror_back_p_left": 1 - mirrored[task],
        "symmetry_error": original[task] + mirrored[task] - 1,
    } for task in sorted(mirrored)]
    write_csv(rows, output_dir / "validation_predictions.csv")
    write_csv(metrics, output_dir / "validation_metrics.csv")
    write_csv(by_choice, output_dir / "validation_by_choice.csv")
    write_csv(comparisons, output_dir / "participant_bootstrap.csv")
    write_csv(symmetry, output_dir / "task_symmetry.csv")
    write_csv(failures, output_dir / "failures.csv")
    (output_dir / "metadata.json").write_text(json.dumps({
        "analysis": "exploratory validation-only mirror audit; original test results have already been inspected",
        "source_counts": counts, "split_seed": 20260928, "model": args.model,
        "model_digest": digest, "original_prompt_version": "snopkova_text_corridor_v1",
        "mirror_prompt_version": VERSION, "mirrored_tasks": len(mirrored),
        "failed_tasks": failures,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mirrored_tasks": len(mirrored), "metrics": metrics, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
