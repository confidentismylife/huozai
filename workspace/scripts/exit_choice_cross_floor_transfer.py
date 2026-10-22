"""Frozen, source-only cross-floor development test. No PPO or sealed scenes.

Ground Floor is the source; Floor 1 (attachment Level 3) is the target.
All source records belonging to a target date are excluded from fitting and
LLM context. This is not independent-event or independently blind validation.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import openpyxl
import psutil
import exit_choice_initial_benchmark as old
import exit_choice_calibration_campaign as calibration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/exit_choice_cross_floor_transfer_20261012"
SRC = ROOT / "outputs/exit_choice_initial_benchmark_20261012"
CTX = ROOT / "outputs/exit_choice_supervised_context_20261012"
XLSX = Path("C:/Users/摩天轮大盗/Downloads/Exit choice.xlsx")
BLOCKS = {"06-03-19 EC": [139, 159], "11-09-19 EC": [119, 137],
          "31-10-19 EC": [142, 156, 164]}
MODES = ["zero_same_settings", "source_training_records"]
METHODS = ["nearest", "familiarity_only", "portable_linear", "portable_nonlinear",
           "source_exit_frequency_diagnostic", "source_index_logit_diagnostic"]
RIDGES = [.01, .1, 1., 10.]
TEMPERATURES = [1., 2., 5., 10., 20., 50.]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    p = OUT / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "frozen_plan.json").exists():
        print("Existing protocol retained; no preparation repeated"); return
    protected = {r["path"]: r["sha256"] for r in read(ROOT / "outputs/exit_choice_calibration_campaign_20261012/preservation_before.json")}
    for folder in (CTX, ROOT / "outputs/exit_choice_order_stability_20261012",
                   ROOT / "outputs/exit_choice_training_table_probe_20261012",
                   ROOT / "outputs/exit_choice_two_hour_session_20261012"):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in {".json", ".csv", ".bib"}:
                protected[str(p)] = sha(p)
    for r in read(ROOT / "outputs/exit_choice_two_hour_session_20261012/completion_manifest.json")["files"]:
        protected[r["path"]] = r["sha256"]
    assert all(sha(p) == digest for p, digest in protected.items())
    save("preservation_before.json", [{"path": p, "sha256": s} for p, s in protected.items()])
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=False)
    raw = []; headers = []
    for sheet, selected_headers in BLOCKS.items():
        rows = list(wb[sheet].iter_rows(values_only=True)); active = None
        date = datetime.strptime(sheet[:8], "%d-%m-%y").date().isoformat()
        for rownum, vals in enumerate(rows, 1):
            starts = [i for i, v in enumerate(vals) if isinstance(v, str) and v.strip().lower() == "dist"]
            if starts:
                active = None
                if rownum in selected_headers:
                    assert len(starts) == 3
                    preceding = rows[rownum-2]
                    labels = [preceding[i] for i in starts]
                    assert "nw next" in str(labels[0]).lower() and "elevator" in str(labels[1]).lower() and "stairs" in str(labels[2]).lower()
                    assert any("level 3" in str(v).lower() for v in preceding[:3])
                    choice_col = [i for i, v in enumerate(vals) if isinstance(v, str) and v.strip().lower() == "choice"]
                    assert len(choice_col) == 1
                    active = {"starts": starts, "choice_col": choice_col[0], "header_row": rownum}
                    headers.append({"sheet": sheet, **active, "exit_labels": labels,
                                    "floor": "Floor 1 / attachment Level 3"})
                continue
            if active is None or vals[0] is None:
                continue
            ident = old.canonical_id(vals[0])
            if not re.fullmatch(r"\d+[A-Za-z]?", ident):
                continue
            initial, final, kind = old.choice(vals[active["choice_col"]])
            desc = " ".join(str(v) for v in vals[1:active["starts"][0]] if v is not None)
            raw.append({"record_id": f"{date}:r{rownum}", "person_key": f"{date}:{ident}",
                        "date": date, "row": rownum, "sheet": sheet, "header_row": active["header_row"],
                        "initial_label": initial, "final_label": final, "label_kind": kind,
                        "staff_description_flag": bool(re.search(r"\bstaff\b", desc, re.I)),
                        "exits": [{"exit": j+1, **dict(zip(["dist", "people", "staff", "fam"], vals[i:i+4]))}
                                  for j, i in enumerate(active["starts"])],
                        "label_time_verified": False})
    wb.close()
    grouped = defaultdict(list)
    for r in raw: grouped[r["person_key"]].append(r)
    cases = []; excluded = []
    for key, rows in grouped.items():
        first = min(rows, key=lambda r: r["row"]); reasons = []
        if len(rows) != 1 or first["label_kind"] != "single_numeric": reasons.append("sequence_or_non_single_choice")
        if any(r["staff_description_flag"] for r in rows): reasons.append("staff_description_conservative_exclusion")
        for field in ("dist", "people", "staff", "fam"):
            vv = [e[field] for e in first["exits"]]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in vv):
                reasons.append("missing_or_censored_" + field)
            elif field in ("fam", "staff") and any(v not in (0, 1) for v in vv):
                reasons.append("unresolved_nonbinary_" + field)
        if reasons: excluded.append({"person_key": key, "rows": [r["row"] for r in rows], "reasons": reasons})
        else: cases.append(first)
    cases.sort(key=lambda r: r["record_id"])
    source = read(SRC / "strict_initial_cases.json")
    assert set(r["record_id"] for r in cases).isdisjoint(r["record_id"] for r in source)
    assert cases and len(cases) * 2 <= 150
    prior_plan = read(CTX / "frozen_plan.json")
    contexts = {(r["test_date"], r["mode"]): r for r in read(CTX / "contexts.json")}
    tasks = []
    for date in sorted({r["date"] for r in cases}):
        for mode in MODES:
            previous_mode = "all_training_records" if mode == "source_training_records" else "zero_same_settings"
            cc = contexts[(date, previous_mode)]
            assert all(i.split(":")[0] != date for i in cc["train_ids"])
            for case in [c for c in cases if c["date"] == date]:
                target = {"exits": [{"exit": f"E{e['exit']}", "distance": e["dist"],
                                    "previous_choosers": e["people"], "commonly_used": e["fam"]} for e in case["exits"]]}
                payload = {"model": prior_plan["model"], "messages": [
                    {"role": "system", "content": prior_plan["system_prompt"]},
                    {"role": "user", "content": "其他事件训练资料：" + cc["context"]},
                    {"role": "user", "content": "当前个体：" + json.dumps(target, ensure_ascii=False, separators=(",", ":"))}],
                    "format": prior_plan["schema"], "options": prior_plan["options"], "think": False,
                    "stream": False, "keep_alive": "2m"}
                tasks.append({"case_id": case["record_id"], "date": date, "mode": mode,
                              "training_ids": cc["train_ids"], "payload": payload})
    save("target_cases.json", cases); save("target_row_audit.json", raw)
    save("exclusions.json", excluded); save("block_mapping.json", headers); save("requests.json", tasks)
    plan = {"frozen_at_utc": now(), "question": "Transfer source-floor behavior prediction to a different floor with different exit identity/familiarity and class coverage",
            "status": "exploratory_cross_floor_development; same original events previously inspected",
            "source": "103 strict Ground Floor cases", "target": "strict single records on Floor 1 / attachment Level 3",
            "whole_target_date_excluded_from_source_learning": True, "target_labels_used_for_tuning": False,
            "methods": METHODS, "llm_modes": MODES, "llm_model": prior_plan["model"],
            "digest": read(ROOT / "outputs/local_llm_selection.json")["digest"],
            "system_prompt": prior_plan["system_prompt"], "options": prior_plan["options"],
            "llm_prompt_and_settings_preserved_from_previous_context_study": True,
            "portable_features": "Dist/10, People/10, Fam; nonlinear adds log1p(People), no exit-index intercepts",
            "fam_limit": "Fam effect is confounded with source E1 identity; portability is a tested assumption, not causal identification",
            "index_baselines": "source exit frequency and index Logit are diagnostics; local E1 is a different physical exit on each floor",
            "ridge_grid": RIDGES, "nearest_temperature_grid": TEMPERATURES,
            "parameter_selection": "source dates only, inner leave-one-source-date-out, date-equal NLL",
            "fallback": "uniform probabilities, separately reported", "maximum_new_llm_calls": len(tasks),
            "maximum_runtime_seconds_per_inference_run": 1800, "timeout_seconds": 180, "retries": 0,
            "stop_after_consecutive_failures": 3, "cold_load_minimum_available_GiB": 6.5,
            "primary_comparison": "source-context Qwen vs source-only portable nonlinear utility",
            "metrics": ["date-equal NLL", "pooled NLL", "accuracy", "Brier", "per-exit recall", "exit probability shares"],
            "no_ppo_training_or_sealed_scene_access": True, "label_timing": "recorded choice, first-choice time unverified",
            "target_cases_sha256": sha(OUT / "target_cases.json"), "requests_sha256": sha(OUT / "requests.json"),
            "source_cases_sha256": sha(SRC / "strict_initial_cases.json"), "script_sha256": sha(__file__)}
    save("frozen_plan.json", plan)
    (OUT / "source_snapshot").mkdir(exist_ok=True)
    (OUT / "source_snapshot/exit_choice_cross_floor_transfer.py").write_bytes(Path(__file__).read_bytes())
    save("data_summary.json", {"raw_rows": len(raw), "person_keys": len(grouped), "strict_cases": len(cases),
        "excluded_keys": len(excluded), "dates": dict(Counter(c["date"] for c in cases)),
        "exit_choices": dict(Counter(c["initial_label"] for c in cases)),
        "familiarity_vectors": dict(Counter(str([e["fam"] for e in c["exits"]]) for c in cases)),
        "new_human_events": 0, "calls": len(tasks)})
    print(json.dumps(read(OUT / "data_summary.json"), ensure_ascii=False), flush=True)


def features(cases, method):
    dist = np.array([[e["dist"] / 10 for e in c["exits"]] for c in cases])
    people = np.array([[e["people"] / 10 for e in c["exits"]] for c in cases])
    fam = np.array([[e["fam"] for e in c["exits"]] for c in cases])
    if method == "source_index_logit_diagnostic": return old.design(cases, False)
    if method == "familiarity_only": return fam[:, :, None]
    columns = [dist, people, fam]
    if method == "portable_nonlinear":
        columns.append(np.array([[np.log1p(e["people"]) for e in c["exits"]] for c in cases]))
    return np.stack(columns, axis=2)


def predict(train, test, method, parameter):
    if method == "nearest":
        return old.softmax(-np.array([[e["dist"] for e in c["exits"]] for c in test]) / parameter), {}
    if method == "source_exit_frequency_diagnostic":
        return np.repeat(old.frequency(train)[None, :], len(test), axis=0), {}
    x = features(train, method); tx = features(test, method)
    theta, detail = calibration.optimize(x, np.zeros(x.shape[:2]), old.targets(train), parameter)
    return old.softmax(np.einsum("njd,d->nj", tx, theta)), {"coefficients": theta.tolist(), **detail}


def baselines():
    plan = read(OUT / "frozen_plan.json"); source = read(SRC / "strict_initial_cases.json"); target = read(OUT / "target_cases.json")
    assert sha(SRC / "strict_initial_cases.json") == plan["source_cases_sha256"]
    assert sha(OUT / "target_cases.json") == plan["target_cases_sha256"]
    predictions = []; fits = []
    for date in sorted({r["date"] for r in target}):
        train = [r for r in source if r["date"] != date]; test = [r for r in target if r["date"] == date]
        for method in METHODS:
            grid = TEMPERATURES if method == "nearest" else ([1.] if method == "source_exit_frequency_diagnostic" else RIDGES)
            tuning = []
            for parameter in grid:
                folds = []
                for inner_date in sorted({r["date"] for r in train}):
                    tr = [r for r in train if r["date"] != inner_date]; te = [r for r in train if r["date"] == inner_date]
                    p, _ = predict(tr, te, method, parameter)
                    folds.append({"validation_date": inner_date, "train_dates": sorted({r["date"] for r in tr}),
                                  "log_loss": old.metrics(te, p)["log_loss"]})
                tuning.append({"parameter": parameter, "source_macro_loss": float(np.mean([r["log_loss"] for r in folds])), "folds": folds})
            best = min(tuning, key=lambda r: (r["source_macro_loss"], r["parameter"]))["parameter"]
            p, detail = predict(train, test, method, best)
            fits.append({"date": date, "method": method, "parameter": best,
                         "training_ids": [r["record_id"] for r in train], "inner_tuning": tuning, **detail})
            for case, pp in zip(test, p):
                predictions.append({"case_id": case["record_id"], "date": date, "method": method,
                                    "true_exit": case["initial_label"], "probabilities": pp.tolist(), "fallback": False})
    save("baseline_predictions.json", predictions); save("source_fits.json", fits)
    print(json.dumps({"statistical_predictions": len(predictions), "target_records": len(target), "target_label_fits": 0}), flush=True)


def run():
    plan = read(OUT / "frozen_plan.json"); tasks = read(OUT / "requests.json")
    assert sha(OUT / "requests.json") == plan["requests_sha256"]
    tag = next(r for r in old.api("/api/tags")["models"] if r["name"] == plan["llm_model"])
    assert tag["digest"] == plan["digest"]
    available = psutil.virtual_memory().available / 2**30
    active = old.api("/api/ps")["models"]
    assert any(r.get("name") == plan["llm_model"] for r in active) or available >= plan["cold_load_minimum_available_GiB"], "Insufficient available memory for a cold 8B load"
    save("runtime.json", {"at_utc": now(), "tag": tag, "available_GiB": available, "version": old.api("/api/version")})
    started = time.monotonic(); failures = 0
    for i, task in enumerate(tasks, 1):
        name = "responses/" + task["case_id"].replace(":", "_") + "_" + task["mode"] + ".json"
        if (OUT / name).exists(): continue
        if time.monotonic() - started > plan["maximum_runtime_seconds_per_inference_run"]:
            save("run_status.json", {"state": "runtime_bound_reached", "at_utc": now()}); break
        start = time.monotonic(); record = {**task, "started_at_utc": now()}
        try:
            response = old.api("/api/chat", task["payload"], plan["timeout_seconds"])
            record["response"] = response
            parsed = json.loads(response["message"]["content"])
            assert set(parsed) == {"p1", "p2", "p3"}
            assert all(type(v) in (int, float) for v in parsed.values())
            pp = np.asarray([parsed[k] for k in ("p1", "p2", "p3")], dtype=float)
            assert np.isfinite(pp).all() and (pp >= 0).all() and (pp <= 1).all() and abs(pp.sum()-1) <= .01
            assert response["done"] and response.get("done_reason") == "stop"
            assert response["prompt_eval_count"] + response["eval_count"] < plan["options"]["num_ctx"]
            record.update(valid=True, probabilities=(pp/pp.sum()).tolist()); failures = 0
        except Exception as exc:
            record.update(valid=False, error=f"{type(exc).__name__}: {exc}", probabilities=[1/3]*3); failures += 1
        record["wall_seconds"] = time.monotonic()-start; save(name, record)
        save("run_status.json", {"state": "running", "at_utc": now(), "saved": len(list((OUT/"responses").glob("*.json"))), "expected": len(tasks)})
        print(json.dumps({"index": i, "total": len(tasks), "mode": task["mode"], "valid": record["valid"], "seconds": round(record["wall_seconds"], 2)}), flush=True)
        if failures >= plan["stop_after_consecutive_failures"]: break
    else:
        save("run_status.json", {"state": "completed", "at_utc": now(), "saved": len(tasks), "expected": len(tasks)})


def report():
    target = read(OUT / "target_cases.json"); lookup = {c["record_id"]: c for c in target}
    predictions = read(OUT / "baseline_predictions.json")
    responses = [read(p) for p in (OUT / "responses").glob("*.json")]
    for r in responses:
        predictions.append({"case_id": r["case_id"], "date": r["date"], "method": "qwen_" + r["mode"],
                            "true_exit": lookup[r["case_id"]]["initial_label"], "probabilities": r["probabilities"], "fallback": not r["valid"]})
    metrics = []
    for method in sorted({r["method"] for r in predictions}):
        rr = [r for r in predictions if r["method"] == method]
        for date in sorted({r["date"] for r in rr}) + ["pooled"]:
            rows = [r for r in rr if date == "pooled" or r["date"] == date]
            y = np.array([r["true_exit"]-1 for r in rows]); p = np.array([r["probabilities"] for r in rows]); pred = p.argmax(1)
            metrics.append({"method": method, "date": date, **old.metrics([lookup[r["case_id"]] for r in rows], p),
                            "recalls": [float((pred[y == j] == j).mean()) if np.any(y == j) else None for j in range(3)],
                            "fallbacks": sum(r["fallback"] for r in rows)})
        dd = [r for r in metrics if r["method"] == method and r["date"] != "pooled"]
        metrics.append({"method": method, "date": "date_macro", "n": len(rr),
                        **{k: float(np.mean([r[k] for r in dd])) for k in ("accuracy", "log_loss", "brier", "exit_share_tv")}})
    save("predictions.json", predictions); save("metrics.json", metrics)
    complete = len(responses) == read(OUT/"frozen_plan.json")["maximum_new_llm_calls"]
    save("summary.json", {"complete": complete, "saved": len(responses), "valid": sum(r["valid"] for r in responses),
                          "target_records": len(target), "new_events": 0, "new_ppo_runs": 0})
    labels = {"nearest": "最近出口", "familiarity_only": "源域熟悉代理模型", "portable_linear": "可迁移线性效用",
              "portable_nonlinear": "可迁移非线性效用", "source_exit_frequency_diagnostic": "源域出口编号频率（诊断）",
              "source_index_logit_diagnostic": "源域编号截距Logit（诊断）", "qwen_zero_same_settings": "同设置零样本Qwen",
              "qwen_source_training_records": "源域真人记录Qwen"}
    summary = read(OUT / "data_summary.json")
    lines = ["# 从地面层到另一楼层的出口选择迁移", "", "2026年10月12日。已初查的同一图书馆事件上的跨楼层开发实验，不是新的独立事件或最终确认。", "",
             f"目标为附件Level 3／论文Floor 1：{len(target)}条严格单次记录，来源原始{summary['raw_rows']}行。训练及LLM实例仅来自Ground Floor，整个目标日期排除。",
             f"本轮保存{len(responses)}条本地响应，合法{sum(r['valid'] for r in responses)}条，完整={complete}。", "",
             "| 方法 | 日期等权NLL | 合并准确率 | E1召回 | E2召回 | E3召回 | 回退 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in labels:
        pooled = next((r for r in metrics if r["method"] == method and r["date"] == "pooled"), None)
        macro = next((r for r in metrics if r["method"] == method and r["date"] == "date_macro"), None)
        if pooled and macro:
            recall = ["未定义" if v is None else f"{v:.2%}" for v in pooled["recalls"]]
            lines.append(f"| {labels[method]} | {macro['log_loss']:.6f} | {pooled['accuracy']:.2%} | {' | '.join(recall)} | {pooled['fallbacks']} |")
    lines.extend(["", "模型输出是显式生成概率。统计效用用Dist、People、Fam及预设非线性，不使用跨楼层不相同的出口编号截距。源域Fam与E1身份仍共线，因此其跨楼层语义是待检验的假设。编号频率及编号截距仅作失效诊断，不作为主要公平迁移对照。",
                  "", "所有参数和提示先冻结，正则与最近出口温度只用源域日期内部选择。目标楼层标签没有用于拟合、参数选择或LLM例子。原始资料和过去结果保持不变，最终48个疏散场景未读取，没有新PPO训练。",
                  "", "目标记录不具备已核实的逐人首次选择时间，仍只称记录选择预测；Staff描述角色及原论文完整样本尚未全部对齐。案例在同事件、同行群体中可能相关，调用数不等于独立人类或建筑样本。",
                  "", "只有全部请求完成后才进行完整队列比较。该结果不能代替新日期或新建筑验证，也不能推出真人引导因果效应或疏散安全收益。",
                  "", f"完整方案、来源行、排除表、学习来源及回答见[结果目录]({OUT.as_posix()})。"])
    (ROOT / "reports/Exit_choice跨楼层迁移实验_2026-10-12.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(read(OUT / "summary.json"), ensure_ascii=False), flush=True)


def audit():
    plan = read(OUT / "frozen_plan.json"); assert sha(OUT / "requests.json") == plan["requests_sha256"]
    source = read(SRC / "strict_initial_cases.json"); dates = {c["record_id"]: c["date"] for c in source}
    for request in read(OUT / "requests.json"):
        assert all(dates[k] != request["date"] for k in request["training_ids"])
    for fit in read(OUT / "source_fits.json"):
        assert all(dates[k] != fit["date"] for k in fit["training_ids"])
        for choice in fit["inner_tuning"]:
            assert all(fit["date"] not in fold["train_dates"] and fold["validation_date"] != fit["date"] for fold in choice["folds"])
    protected = read(OUT / "preservation_before.json")
    for r in protected: assert sha(r["path"]) == r["sha256"], r["path"]
    result = {"dates_disjoint": True, "source_only_tuning": True, "protected_files_unchanged": len(protected),
              "reserved_scenes_accessed": 0, "target_label_fits": 0, "checked_at_utc": now()}
    save("verification.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "baselines", "run", "report", "audit"])
    {"prepare": prepare, "baselines": baselines, "run": run, "report": report, "audit": audit}[parser.parse_args().command]()
