from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.public_route_validation import (
    participant_bootstrap_differences,
    participant_splits,
    read_public_trials,
    route_prompt,
    score_predictions,
)


SOURCE_URL = (
    "https://springernature.figshare.com/articles/dataset/"
    "Predictors_of_evacuation_behavior_dataset_on_respondents_route_choice_and_web_interaction/27705402"
)
PROMPT_VERSION = "snopkova_text_corridor_v1"
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"p_left": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["p_left"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local Ollama model against held-out human route choices.")
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=ROOT / "data" / "public_snopkova" / "Snopkova_Isovists.zip",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "local_llm_validation")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-calls", type=int, default=None, help="For an interface smoke test.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--split-seed", type=int, default=20260928)
    return parser.parse_args()


def api_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "response_cache"
    cache_dir.mkdir(exist_ok=True)

    trials, counts = read_public_trials(str(args.source_zip))
    split = participant_splits(trials, seed=args.split_seed)
    write_csv(
        [{"participant_id": person, "split": name} for person, name in sorted(split.items())],
        args.output_dir / "participant_split.csv",
    )
    tasks = {trial.task_id: trial for trial in trials}

    tags = api_json(f"{args.base_url}/api/tags")
    matching = [item for item in tags.get("models", []) if item.get("name") == args.model]
    if not matching:
        raise SystemExit(f"model {args.model!r} is not installed in Ollama")
    model_digest = matching[0].get("digest", "")

    llm_by_task: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    new_calls = 0
    for task_id, trial in sorted(tasks.items()):
        prompt = route_prompt(trial)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{task_id}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if cached is not None and all(
            cached.get(key) == value
            for key, value in (
                ("model", args.model),
                ("model_digest", model_digest),
                ("prompt_hash", prompt_hash),
                ("prompt_version", PROMPT_VERSION),
            )
        ):
            llm_by_task[task_id] = float(cached["p_left"])
            continue
        if args.max_new_calls is not None and new_calls >= args.max_new_calls:
            continue

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": 0, "seed": 20260928, "num_predict": 64},
            "think": False,
            "stream": False,
        }
        try:
            response = api_json(f"{args.base_url}/api/chat", payload, timeout=args.timeout)
            raw = response["message"]["content"]
            p_left = float(json.loads(raw)["p_left"])
            if not 0.0 <= p_left <= 1.0:
                raise ValueError("p_left outside [0, 1]")
        except Exception as exc:
            failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {task_id}: {exc}", flush=True)
            continue

        cached = {
            "task_id": task_id,
            "model": args.model,
            "model_digest": model_digest,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "raw_response": raw,
            "p_left": p_left,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_ns": response.get("total_duration"),
        }
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
        llm_by_task[task_id] = p_left
        new_calls += 1
        print(f"{len(llm_by_task)}/{len(tasks)} {task_id}: p_left={p_left:.3f}", flush=True)

    predictions, metrics = score_predictions(trials, split, llm_by_task)
    comparisons = participant_bootstrap_differences(predictions)
    write_csv(predictions, args.output_dir / "test_predictions.csv")
    write_csv(metrics, args.output_dir / "test_metrics.csv")
    write_csv(comparisons, args.output_dir / "participant_bootstrap.csv")
    write_csv(failures, args.output_dir / "failures.csv")
    metadata = {
        "source_url": SOURCE_URL,
        "source_zip": str(args.source_zip.resolve()),
        "source_counts": counts,
        "split_seed": args.split_seed,
        "participant_counts": {name: sum(value == name for value in split.values()) for name in ("train", "validation", "test")},
        "model": args.model,
        "model_digest": model_digest,
        "prompt_version": PROMPT_VERSION,
        "task_predictions": len(llm_by_task),
        "failed_tasks": failures,
        "warning": (
            "Text summaries of an 8-second visual virtual-building task; no measured smoke, "
            "fire, crowd, or signage effects. Pilot predictive comparison only."
        ),
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "task_predictions": len(llm_by_task), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
