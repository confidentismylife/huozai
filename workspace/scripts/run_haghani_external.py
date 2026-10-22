"""Low-cost external choice prediction on locked Haghani 2016 mock-evacuation cases.

The 90 input cases were frozen without labels by prepare_haghani_external.py.
This tests distance, visibility and crowd response, NOT fire/smoke/signage response.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_haghani_external import DATA, OUTPUT, file_hash, read_choice_sets
from run_local_route_validation import api_json, write_csv

PROMPT_VERSION = "haghani_visible_attributes_v1"


def features(alternatives):
    """Only cues represented in the text prompt, with unseen congestion masked."""
    return np.array([
        [
            alt["distance_m"] * alt["visible"],
            alt["congestion_people"] * alt["visible"],
            float(alt["visible"]),
            alt["flow_to_exit_people"] * alt["visible"],
            alt["flow_to_exit_people"] * (not alt["visible"]),
        ]
        for alt in alternatives
    ], dtype=float)


def fit_choice_model(hypothetical):
    design = np.stack([features(row["alternatives"]) for row in hypothetical])
    target = np.array([row["selected_exit"] - 1 for row in hypothetical])
    mean = design.reshape(-1, design.shape[-1]).mean(axis=0)
    std = design.reshape(-1, design.shape[-1]).std(axis=0)
    std[std < 1e-12] = 1
    scaled = (design - mean) / std

    def loss(beta):
        utilities = scaled @ beta
        return float((logsumexp(utilities, axis=1) - utilities[np.arange(len(target)), target]).mean())

    result = minimize(loss, np.zeros(design.shape[-1]), method="BFGS", options={"maxiter": 1000})
    if not np.isfinite(result.fun):
        raise RuntimeError("multinomial logit fit did not produce a finite loss")
    return {"beta": result.x, "mean": mean, "std": std, "train_log_loss": result.fun,
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message)}


def fitted_probabilities(model, alternatives):
    matrix = (features(alternatives) - model["mean"]) / model["std"]
    return softmax(matrix @ model["beta"]).tolist()


def route_prompt(case):
    lines = [
        "一项模拟建筑紧急疏散实验中，一名成年人需要从多个出口中选一个。"
        "只根据下面提供的信息，预测类似参与者选择各出口的概率。"
        "没有给出火、烟、指示牌或其他信息，不要自行补充。"
    ]
    for alt in case["alternatives"]:
        if alt["visible"]:
            lines.append(
                f"出口{alt['exit']}：可见，距离约{alt['distance_m']:.1f}米，"
                f"出口附近拥堵{alt['congestion_people']}人，"
                f"正朝这个出口移动{alt['flow_to_exit_people']}人。"
            )
        else:
            lines.append(
                f"出口{alt['exit']}：目前被障碍物遮挡，不可见；出口附近拥堵人数未知，"
                f"正朝该方向移动{alt['flow_to_exit_people']}人。"
            )
    lines.append(
        f"只返回JSON对象，键probs对应按出口1到出口{len(case['alternatives'])}排序的概率数组；"
        "每项在0到1之间，总和为1。"
    )
    return "\n".join(lines)


def score_case(probs, selected_exit):
    p = np.asarray(probs, dtype=float)
    target = selected_exit - 1
    maximum = p.max()
    winners = np.flatnonzero(np.isclose(p, maximum, atol=1e-12, rtol=0))
    return {
        "top1_ties_fractional": float((target in winners) / len(winners)),
        "brier_multiclass": float(np.square(p - np.eye(len(p))[target]).sum()),
        "log_loss": float(-np.log(max(p[target], 1e-12))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-new-calls", type=int, default=None)
    args = parser.parse_args()
    locked_path = OUTPUT / "locked_input_cases.json"
    audit = json.loads((OUTPUT / "data_audit.json").read_text(encoding="utf-8"))
    if file_hash(locked_path) != audit["locked_input_sha256"]:
        raise RuntimeError("locked input manifest hash mismatch")
    cases = json.loads(locked_path.read_text(encoding="utf-8"))
    if len(cases) != 90 or any("selected_exit" in row for row in cases):
        raise RuntimeError("locked input cases are incomplete or contain labels")
    hc, excluded, _, _ = read_choice_sets(DATA / "S1_hypothetical_choices.xls", 1)
    if excluded:
        raise RuntimeError("unexpected HC exclusions")
    baseline_model = fit_choice_model(hc)
    baseline_public = {
        "feature_names": ["visible_distance", "visible_congestion", "visibility", "visible_flow", "hidden_flow"],
        "beta": baseline_model["beta"].tolist(), "mean": baseline_model["mean"].tolist(),
        "std": baseline_model["std"].tolist(), "train_log_loss": baseline_model["train_log_loss"],
        "optimizer_success": baseline_model["optimizer_success"],
        "optimizer_message": baseline_model["optimizer_message"],
    }
    (OUTPUT / "hc_logit_fit.json").write_text(json.dumps(baseline_public, indent=2), encoding="utf-8")

    tags = api_json(f"{args.base_url}/api/tags")
    matching = [m for m in tags.get("models", []) if m.get("name") == args.model]
    if not matching:
        raise SystemExit(f"Ollama model not found: {args.model}")
    digest = matching[0]["digest"]
    cache_dir = OUTPUT / "response_cache"
    cache_dir.mkdir(exist_ok=True)
    llm = {}
    failures = []
    new_calls = 0
    for case in cases:
        key = case["observation_index"]
        prompt = route_prompt(case)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{key}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if cached and all(cached.get(k) == v for k, v in {
            "model_digest": digest, "prompt_hash": prompt_hash, "prompt_version": PROMPT_VERSION,
        }.items()):
            llm[key] = cached["probs"]
            continue
        if args.max_new_calls is not None and new_calls >= args.max_new_calls:
            continue
        n = case["choice_set_size"]
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {"probs": {
                "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": n, "maxItems": n,
            }}, "required": ["probs"],
        }
        payload = {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "format": schema, "options": {"temperature": 0, "seed": 20260928, "num_predict": 128},
            "think": False, "stream": False,
        }
        try:
            response = api_json(f"{args.base_url}/api/chat", payload, args.timeout)
            raw = response["message"]["content"]
            probs = np.asarray(json.loads(raw)["probs"], dtype=float)
            if len(probs) != n or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or probs.sum() <= 0:
                raise ValueError("invalid probability vector")
            probs = (probs / probs.sum()).tolist()
        except Exception as exc:
            failures.append({"observation_index": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {key}: {exc}", flush=True)
            continue
        cache_path.write_text(json.dumps({
            "observation_index": key, "model": args.model, "model_digest": digest,
            "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash, "prompt": prompt,
            "raw_response": raw, "probs": probs,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_ns": response.get("total_duration"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        llm[key] = probs
        new_calls += 1
        print(f"{len(llm)}/{len(cases)} observation {key}", flush=True)

    (OUTPUT / "call_status.json").write_text(json.dumps({
        "model": args.model, "model_digest": digest, "prompt_version": PROMPT_VERSION,
        "locked_input_sha256": audit["locked_input_sha256"],
        "completed": len(llm), "total": len(cases), "failures": failures,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(llm) != len(cases):
        print(f"Predictions incomplete ({len(llm)}/{len(cases)}); labels have not been loaded or scored.")
        return

    # Opening the external outcomes is deliberately delayed until every prediction is cached.
    realistic, exclusions, _, _ = read_choice_sets(DATA / "S3_realistic_choices.xls", 0)
    truth = {row["observation_index"]: row["selected_exit"] for row in realistic}
    predictions = []
    for case in cases:
        index = case["observation_index"]
        size = case["choice_set_size"]
        options = case["alternatives"]
        models = {
            "local_llm": llm[index],
            "hc_logit": fitted_probabilities(baseline_model, options),
            "uniform": [1 / size] * size,
        }
        for method, probability in models.items():
            predictions.append({
                "observation_index": index, "record_id": case["record_id"],
                "choice_set_size": size, "selected_exit": truth[index], "method": method,
                "probabilities": json.dumps(probability),
                **score_case(probability, truth[index]),
            })
    metrics = []
    for method in ("local_llm", "hc_logit", "uniform"):
        for size in (0, 2, 3, 4):
            subset = [r for r in predictions if r["method"] == method and (size == 0 or r["choice_set_size"] == size)]
            metrics.append({
                "method": method, "choice_set_size": size if size else "all",
                "cases": len(subset),
                **{column: float(np.mean([r[column] for r in subset])) for column in (
                    "top1_ties_fractional", "brier_multiclass", "log_loss"
                )},
            })
    write_csv(predictions, OUTPUT / "external_predictions.csv")
    write_csv(metrics, OUTPUT / "external_metrics.csv")
    keyed = {(r["observation_index"], r["method"]): r for r in predictions}
    strata = {size: [c["observation_index"] for c in cases if c["choice_set_size"] == size] for size in (2, 3, 4)}
    rng = np.random.default_rng(20260928)
    draws = np.concatenate([
        rng.choice(ids, size=(5000, len(ids)), replace=True) for ids in strata.values()
    ], axis=1)
    comparisons = []
    for reference in ("hc_logit", "uniform"):
        for metric in ("top1_ties_fractional", "brier_multiclass", "log_loss"):
            deltas = {case["observation_index"]: (
                keyed[case["observation_index"], "local_llm"][metric]
                - keyed[case["observation_index"], reference][metric]
            ) for case in cases}
            values = np.array([np.mean([deltas[int(index)] for index in draw]) for draw in draws])
            comparisons.append({
                "candidate": "local_llm", "reference": reference, "metric": metric,
                "cases": len(cases), "difference": float(np.mean(list(deltas.values()))),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
            })
    write_csv(comparisons, OUTPUT / "paired_bootstrap.csv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
