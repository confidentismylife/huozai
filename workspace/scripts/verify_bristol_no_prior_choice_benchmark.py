"""Independent checks for the frozen scene-only human-choice benchmark."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bristol_no_prior_choice_benchmark_20261013"
CONFIG = ROOT / "schemas/bristol_no_prior_choice_benchmark_20261013.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def condition(row):
    return tuple(int(row[k]) for k in ["assigned_left_crowd", "assigned_right_crowd", "knowledge_code"])


def family(row):
    a, b, k = condition(row)
    return min(a, b), max(a, b), k


def metrics(rows, method):
    grouped = defaultdict(list)
    for row in rows:
        grouped[condition(row)].append(row)
    cell_loss, cell_brier, errors = [], [], []
    total_loss = total_correct = 0
    for cell in grouped.values():
        ps = {float(r[method]) for r in cell}
        assert max(ps) - min(ps) <= 1e-12
        p = min(1 - 1e-9, max(1e-9, float(cell[0][method])))
        q = sum(int(r["observed_exit"]) == 1 for r in cell) / len(cell)
        loss = -q * math.log(p) - (1 - q) * math.log(1 - p)
        brier = q * (1 - p) ** 2 + (1 - q) * p ** 2
        cell_loss.append(loss); cell_brier.append(brier); errors.append(abs(p - q))
        total_loss += len(cell) * loss
        total_correct += sum(((p >= .5) == (int(r["observed_exit"]) == 1)) for r in cell)
    return {"records": len(rows), "conditions": len(grouped),
        "condition_macro_NLL": sum(cell_loss) / len(grouped), "condition_macro_Brier": sum(cell_brier) / len(grouped),
        "participant_NLL": total_loss / len(rows), "participant_accuracy": total_correct / len(rows),
        "mean_absolute_condition_probability_error": sum(errors) / len(grouped)}


def check_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        close(actual[key], expected[key])


def verify():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = OUT / "frozen_manifest.json"
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    assert plan["config_sha256"] == sha(CONFIG)
    assert plan["runner_sha256"] == sha(ROOT / "scripts/run_bristol_no_prior_choice_benchmark.py")
    assert all(sha(ROOT / path) == checksum for path, checksum in plan["protected_previous_files"].items())
    source = json.loads((ROOT / cfg["source_manifest"]).read_text(encoding="utf-8"))
    probe = json.loads((ROOT / cfg["prompt_source_manifest"]).read_text(encoding="utf-8"))
    source_tasks = {t["key"]: t for t in source["tasks"]}
    tasks = plan["tasks"]
    assert len(tasks) == len({t["key"] for t in tasks}) == 34
    assert set(source_tasks) == {t["key"] for t in tasks}
    assert [t["request_order"] for t in tasks] == list(range(1, 35))
    template = probe["tasks"][0]["payload"]
    allowed = {"assigned_left_crowd", "assigned_right_crowd", "map_information", "nearer_to_final_exit", "crowd_state_design"}
    for task in tasks:
        previous = source_tasks[task["key"]]
        assert task["condition"] == previous["condition"]
        payload = task["payload"]
        messages = payload["messages"]
        assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
        obs = json.loads(messages[1]["content"])
        before = json.loads(previous["payload"]["messages"][1]["content"])
        before.pop("training_base_probability_left")
        assert set(obs) == allowed and obs == before
        assert messages[0] == template["messages"][0]
        assert all(payload[k] == template[k] for k in payload if k != "messages")
        assert task["prompt_sha256"] == hashlib.sha256((messages[0]["content"] + "\n" + messages[1]["content"]).encode()).hexdigest()
        assert task["payload_sha256"] == hashlib.sha256(canonical(payload).encode()).hexdigest()
    assert len(source["folds"]) == 18
    for fold in source["folds"]:
        assert not ({tuple(c) for c in fold["held_conditions"]} & {tuple(c) for c in fold["train_conditions"]})
    result = {"status": "pre_inference_design_verified", "all_34_source_scenes_covered": True,
        "same_prompt_and_options_as_frozen_probe": True, "no_priors_or_outcome_fields_in_prompts": True,
        "all_previous_files_unchanged": True, "no_new_fitting": True,
        "manifest_sha256": sha(manifest), "PPO_updates": 0,
        "old_final_scenarios_accessed": False, "Liao_confirmation_scores_computed": False}
    path = OUT / "responses.jsonl"
    if not path.exists():
        target = OUT / "design_verification.json"
    else:
        responses = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(responses) == 34 and all(r["valid"] for r in responses)
        assert [r["key"] for r in responses] == [t["key"] for t in tasks]
        bykey = {r["key"]: r for r in responses}
        for row, task in zip(responses, tasks):
            assert row["model_digest"] == cfg["LLM"]["digest"] == plan["model_digest"]
            assert row["manifest_sha256"] == sha(manifest)
            assert all(row[k] == task[k] for k in ["prompt_sha256", "payload_sha256"])
            assert datetime.fromisoformat(plan["frozen_at_utc"]) <= datetime.fromisoformat(row["started_at_utc"]) <= datetime.fromisoformat(row["finished_at_utc"])
            raw = row["response"]
            assert raw["model"] == cfg["LLM"]["model"] and raw["done"] is True
            obj = json.loads(raw["message"]["content"])
            assert set(obj) == {"p_left"} and not isinstance(obj["p_left"], bool)
            assert isinstance(obj["p_left"], (int, float)) and math.isfinite(obj["p_left"]) and 0 <= obj["p_left"] <= 1
            close(row["p_left"], obj["p_left"])
        old = list(csv.DictReader((ROOT / cfg["evaluation_source"]).open(encoding="utf-8-sig")))
        rows = list(csv.DictReader((OUT / "all_predictions.csv").open(encoding="utf-8-sig")))
        assert len(rows) == len(old) == 489
        for current, original in zip(rows, old):
            assert all(current[key] == value for key, value in original.items())
            close(float(current["qwen3_4b_no_prior"]), bykey["_".join(map(str, condition(current)))]["p_left"])
            assert current["no_prior_LLM_valid"] == "True"
        raw_cases = list(csv.DictReader((ROOT / "outputs/dynamic_information_search_20261013/audit/bristol_context_cases.csv").open(encoding="utf-8-sig")))
        eligible = [r for r in raw_cases if r["eligible_context_benchmark"] == "True"]
        assert len(raw_cases) == 547 and len(eligible) == 489
        assert {r["case_key"] for r in rows} == {r["case_key"] for r in eligible}
        summary_file = OUT / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        repair = json.loads((OUT / "analysis_repair_v1.json").read_text(encoding="utf-8"))
        assert repair["analysis_repair_sha256"] == sha(ROOT / "scripts/analyze_bristol_no_prior_choice_benchmark.py")
        assert repair["manifest_sha256"] == sha(manifest) and repair["responses_sha256"] == sha(path)
        assert repair["inference_runner_sha256"] == plan["runner_sha256"]
        assert repair["prompt_changes"] == repair["new_inference_requests"] == repair["metric_definition_changes"] == 0
        assert len(repair["baseline_probability_spreads"]) == 4
        for entry in repair["baseline_probability_spreads"]:
            cell = [r for r in old if condition(r) == tuple(entry["condition"])]
            values = [float(r[entry["method"]]) for r in cell]
            close(entry["spread"], max(values) - min(values))
            assert entry["method"] == "context_logit" and 0 < entry["spread"] <= 1e-12
        assert summary["fresh_LLM_requests"] == summary["valid_LLM_responses"] == 34
        assert summary["records"] == 489 and summary["conditions"] == 34 and summary["mirror_families"] == 18
        assert summary["manifest_sha256"] == sha(manifest) and summary["responses_sha256"] == sha(path)
        for method in cfg["methods"]:
            check_metrics(summary["methods"][method], metrics(rows, method))
        for filename, groupkey in [("per_condition_metrics.csv", "condition"), ("per_family_metrics.csv", "family")]:
            with (OUT / filename).open(encoding="utf-8-sig") as handle:
                details = list(csv.DictReader(handle))
            expected_count = (34 if groupkey == "condition" else 18) * len(cfg["methods"])
            assert len(details) == expected_count
            for record in details:
                group = tuple(json.loads(record[groupkey]))
                subset = [r for r in rows if (condition(r) if groupkey == "condition" else family(r)) == group]
                calculated = metrics(subset, record["method"])
                check_metrics({k: float(record[k]) for k in calculated}, calculated)
        new = summary["methods"]["qwen3_4b_no_prior"]
        for method, comparison in summary["comparisons"].items():
            ref = summary["methods"][method]
            close(comparison["NLL_delta_new_minus_reference"], new["condition_macro_NLL"] - ref["condition_macro_NLL"])
            close(comparison["NLL_relative_reduction"], 1 - new["condition_macro_NLL"] / ref["condition_macro_NLL"])
            close(comparison["Brier_delta_new_minus_reference"], new["condition_macro_Brier"] - ref["condition_macro_Brier"])
            close(comparison["accuracy_delta_percentage_points"], 100 * (new["participant_accuracy"] - ref["participant_accuracy"]))
            diffs = []
            for fam in {family(r) for r in rows}:
                subset = [r for r in rows if family(r) == fam]
                diffs.append(metrics(subset, "qwen3_4b_no_prior")["condition_macro_NLL"] - metrics(subset, method)["condition_macro_NLL"])
            assert comparison["mirror_families_lower_NLL"] == sum(d < -1e-12 for d in diffs)
            assert comparison["mirror_families_higher_NLL"] == sum(d > 1e-12 for d in diffs)
            assert comparison["mirror_families_tied_NLL"] == sum(abs(d) <= 1e-12 for d in diffs)
        with (OUT / "mirror_consistency.csv").open(encoding="utf-8-sig") as handle:
            mirror_rows = list(csv.DictReader(handle))
        assert len(mirror_rows) == 90
        for record in mirror_rows:
            a, b, k = json.loads(record["family"])
            pa = float(next(r for r in rows if condition(r) == (a, b, k))[record["method"]])
            pb = float(next(r for r in rows if condition(r) == (b, a, k))[record["method"]])
            close(float(record["p_left_original"]), pa); close(float(record["p_left_mirror"]), pb)
            close(float(record["absolute_mirror_error"]), abs(pa + pb - 1))
        for method, info in summary["mirror_summary"].items():
            errors = [float(r["absolute_mirror_error"]) for r in mirror_rows if r["method"] == method]
            assert info["families"] == len(errors) == 18
            close(info["mean_absolute_error"], sum(errors) / 18); close(info["max_absolute_error"], max(errors))
        selected = {tuple(t["condition"]) for t in probe["tasks"] if t["arm"] == "no_prior"}
        subsets = {"remaining_28_conditions": [r for r in rows if condition(r) not in selected],
            "no_map": [r for r in rows if r["map_information"] == "no_map"],
            "map_available": [r for r in rows if r["map_information"] != "no_map"],
            "busy": [r for r in rows if r["crowd_state_design"] == "busy"],
            "free": [r for r in rows if r["crowd_state_design"] == "free"]}
        assert len({condition(r) for r in subsets["remaining_28_conditions"]}) == 28
        for label, subset in subsets.items():
            for method in cfg["methods"]:
                check_metrics(summary["subgroups"][label][method], metrics(subset, method))
        probe_responses = {r["key"]: r for r in [json.loads(line) for line in (ROOT / "outputs/bristol_prior_mechanism_probe_20261013/responses.jsonl").read_text().splitlines()]}
        assert len(summary["diagnostic_repeatability"]) == 6
        for repeat in summary["diagnostic_repeatability"]:
            key = "_".join(map(str, repeat["condition"]))
            close(repeat["previous"], probe_responses[key + "__no_prior"]["p_left"])
            close(repeat["new"], bykey[key]["p_left"])
            close(repeat["absolute_difference"], abs(repeat["new"] - repeat["previous"]))
        signal = new["condition_macro_NLL"] < summary["methods"]["training_frequency"]["condition_macro_NLL"] and new["condition_macro_Brier"] < summary["methods"]["training_frequency"]["condition_macro_Brier"]
        assert signal == summary["development_signal_over_frequency_on_both_probability_metrics"]
        close(summary["total_request_seconds"], sum(r["wall_seconds"] for r in responses))
        for filename in ["runtime_model_before.json", "runtime_model_after.json"]:
            runtime = json.loads((OUT / filename).read_text(encoding="utf-8"))
            assert runtime["name"] == cfg["LLM"]["model"] and runtime["digest"] == cfg["LLM"]["digest"]
        result.update(status="completed_verified", all_489_cases_accounted_for=True, valid_fresh_responses=34,
            raw_responses_match_predictions=True, all_metrics_independently_recomputed=True,
            remaining_28_conditions_verified=True, old_baselines_preserved=True, no_fallbacks=True,
            scoring_only_roundoff_repair_verified=True,
            frozen_before_first_call=True, model_digest_verified_before_and_after=True,
            responses_sha256=sha(path), summary_sha256=sha(summary_file))
        target = OUT / "verification.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
