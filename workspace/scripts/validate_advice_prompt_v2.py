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

from fire_evac.advice_protocol_v2 import (
    PROMPT_VERSION,
    advice_prompt_v2,
    expected_semantic_target,
    mirror_scenario,
)
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.dynamic_llm_rl import NORTH, parse_dynamic_advice


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_exit": {"type": "string", "enum": ["north", "south"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "avoid_predicted_smoke",
                    "early_guidance",
                    "reduce_exposure",
                    "balance_congestion",
                ],
            },
        },
    },
    "required": ["target_exit", "confidence", "reason_codes"],
}


def api_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--development-scenarios",
        type=Path,
        default=ROOT / "outputs" / "confirmatory_protocol" / "development_scenarios.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "confirmatory_v2_2_prompt_development"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.development_scenarios.read_text(encoding="utf-8"))
    originals = tuple(DynamicConflictScenario(**item) for item in payload)
    mirrors = tuple(mirror_scenario(item) for item in originals)
    scenarios = originals + mirrors
    tags = api_json(f"{args.base_url}/api/tags", timeout=10)
    match = [x for x in tags.get("models", []) if x.get("name") == args.model]
    if not match:
        raise RuntimeError(f"missing Ollama model {args.model}")
    digest = str(match[0].get("digest", ""))
    cache = args.output / "llm_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    parsed_by_id = {}
    for index, scenario in enumerate(scenarios, 1):
        prompt = advice_prompt_v2(scenario)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        path = cache / f"{scenario.scenario_id}.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if stored and stored.get("model_digest") == digest and stored.get("prompt_hash") == prompt_hash:
            raw_payload = stored["parsed"]
        else:
            response = api_json(
                f"{args.base_url}/api/chat",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": SCHEMA,
                    "options": {"temperature": 0, "seed": args.seed, "num_predict": 96},
                    "think": False,
                    "stream": False,
                },
                args.timeout,
            )
            raw_payload = json.loads(response["message"]["content"])
            path.write_text(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "model": args.model,
                        "model_digest": digest,
                        "prompt_version": PROMPT_VERSION,
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "parsed": raw_payload,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        advice = parse_dynamic_advice(raw_payload)
        parsed_by_id[scenario.scenario_id] = advice
        expected = expected_semantic_target(scenario)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "source_id": scenario.scenario_id.removesuffix("M"),
                "is_mirror": int(scenario.scenario_id.endswith("M")),
                "conflict_type": scenario.conflict_type,
                "forecast_source": "north" if scenario.forecast_source_branch == NORTH else "south",
                "target_exit": "north" if advice.target_branch == NORTH else "south",
                "expected_exit": "north" if expected == NORTH else "south",
                "semantic_correct": int(advice.target_branch == expected),
                "confidence": advice.confidence,
                "reason_codes": "|".join(advice.reason_codes),
            }
        )
        print(f"Validated {index}/{len(scenarios)} {scenario.scenario_id}", flush=True)
    path = args.output / "development_semantic_audit.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    original_accuracy = sum(r["semantic_correct"] for r in rows if not r["is_mirror"]) / len(originals)
    mirror_accuracy = sum(r["semantic_correct"] for r in rows if r["is_mirror"]) / len(mirrors)
    clear = [r for r in rows if r["conflict_type"] == "clear"]
    clear_accuracy = sum(r["semantic_correct"] for r in clear) / len(clear)
    symmetry = sum(
        parsed_by_id[s.scenario_id].target_branch != parsed_by_id[f"{s.scenario_id}M"].target_branch
        for s in originals
    ) / len(originals)
    passed = original_accuracy >= 0.95 and mirror_accuracy >= 0.95 and clear_accuracy == 1.0 and symmetry >= 0.95
    summary = {
        "status": "passed" if passed else "failed",
        "freeze_allowed": passed,
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "model_digest": digest,
        "development_scenarios": len(originals),
        "mirrored_scenarios": len(mirrors),
        "original_semantic_accuracy": original_accuracy,
        "mirror_semantic_accuracy": mirror_accuracy,
        "clear_semantic_accuracy": clear_accuracy,
        "mirror_symmetry_rate": symmetry,
        "passing_thresholds": {
            "original_semantic_accuracy": 0.95,
            "mirror_semantic_accuracy": 0.95,
            "clear_semantic_accuracy": 1.0,
            "mirror_symmetry_rate": 0.95,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
