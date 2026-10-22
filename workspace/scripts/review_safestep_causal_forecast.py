"""Independent cache replay and post-hoc interpretation; no new policy sweep.

Keeps the completed experiment untouched. Quadrature here does not use the
RateField primitive; forecast coefficients are reconstructed by least squares,
and chosen-action optimality is checked by forward graph search.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_public_benchmark import Graph, future_min_fed_policy, rollout
from fire_evac.safestep_causal_forecast import available_prefix, make_forecast, predicted_interval_costs

RUN = ROOT / "outputs/safestep_causal_forecast_20261018"
OUT = ROOT / "outputs/safestep_causal_forecast_review_20261018"
REPORT = ROOT / "reports/SafeStep预测规划实验_结果解读与独立核查_2026-10-18.md"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    assert not p.exists(), p
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(p, rows):
    assert not p.exists() and rows
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify(hashes):
    for rel, expected in hashes.items():
        assert sha(ROOT / rel) == expected, rel


def integral(times, rates, a, b):
    """Independent trapezoids after splitting at every native knot."""
    assert times[0] - 1e-9 <= a <= b <= times[-1] + 1e-9
    knots = np.concatenate(([a], times[(times > a) & (times < b)], [b]))
    y = np.interp(knots, times, rates)
    return float(np.sum((y[:-1] + y[1:]) * np.diff(knots) / 2))


def independent_forecast_costs(times, rates, decision, dt, remain, spec):
    observed = times <= decision + 1e-9
    t, y = times[observed].copy(), rates[observed].copy()
    selected = t >= t[-1] - spec["window_s"] - 1e-9
    tx = t[selected] - t[-1]
    design = np.column_stack((np.ones(len(tx)), tx))
    slope = np.linalg.lstsq(design, y[selected], rcond=None)[0][1] * spec["slope_gain"]
    anchor = y[-1]
    cap = float(spec["projection_cap_s"])
    costs = np.zeros((remain, rates.shape[1]))
    for k in range(remain):
        a = decision + k * dt - t[-1]
        b = a + dt
        for i in range(rates.shape[1]):
            knots = [a, b]
            if a < cap < b:
                knots.append(cap)
            if slope[i] < 0:
                crossing = -anchor[i] / slope[i]
                if a < crossing < b and crossing < cap:
                    knots.append(float(crossing))
            x = np.asarray(sorted(knots))
            yy = np.maximum(0., anchor[i] + slope[i] * np.minimum(x, cap))
            costs[k, i] = np.sum((yy[:-1] + yy[1:]) * np.diff(x) / 2)
    return costs


def forward_value(graph, costs, start, first_action=None):
    if first_action is None:
        queue, best = [(0., 0, start)], {(0, start): 0.}
    else:
        dest = graph.destination(start, first_action)
        first = float(costs[0, start])
        queue, best = [(first, 1, dest)], {(1, dest): first}
    while queue:
        value, k, state = heapq.heappop(queue)
        if value != best[k, state]:
            continue
        if state in graph.exit_ids:
            return value
        if k == len(costs):
            continue
        for _, dest in graph.edges[state]:
            new = value + float(costs[k, state])
            key = (k + 1, dest)
            if new < best.get(key, math.inf):
                best[key] = new
                heapq.heappush(queue, (new, k + 1, dest))
    return math.inf


def main():
    assert not OUT.exists() and not REPORT.exists()
    cfg = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))["configuration"]
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    completed = json.loads((RUN / "completion_manifest.json").read_text("utf-8"))
    protected = dict(frozen["protected_previous_sha256"])
    protected.update(frozen["input_sha256"])
    protected.update(completed["artifact_sha256"])
    protected[(RUN / "completion_manifest.json").relative_to(ROOT).as_posix()] = sha(RUN / "completion_manifest.json")
    verify(protected)
    OUT.mkdir()
    save(OUT / "analysis_plan.json", dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        status="POST_HOC_REVIEW_OF_COMPLETED_CACHED_DEVELOPMENT_RESULTS",
        prior_inspection="All main tables, changed-action logs, and selected routes were already inspected",
        work=["independent native-knot quadrature for every cached move",
              "legal actions, cache agreement, first-exit and episode counts",
              "independent least-squares and forward search at all main divergent states plus R24",
              "fixed-plan ranking at selected benefit/harm cases, without new closed-loop policy rollouts",
              "descriptive effect concentration, parameter sensitivity, and comparison with archived DQN"],
        new_closed_loop_rollouts=0, new_training=0, new_LLM_calls=0, final48_access=False,
        protected_sha256=protected, review_script_sha256=sha(Path(__file__))))
    data = load_author_inputs(ROOT / cfg["source_directory"])
    starts = [tuple(c) for c in data["empty"]]
    graph = Graph(starts + list(data["exits"]), data["exits"])
    times = data["rates"]["Time"]
    measured = np.column_stack([data["rates"][data["mapping"][c]] if c not in graph.exits
        else np.zeros(len(times)) for c in graph.cells])
    dt, horizon = cfg["movement"]["dt_s"], cfg["movement"]["maximum_actions"]
    runs = json.loads((RUN / "rollouts.json").read_text("utf-8"))
    costs = {}
    policies = {}
    for departure in cfg["departure_fire_times_s"]:
        costs[departure] = np.asarray([[integral(times, measured[:, i],
            departure + k * dt, departure + (k + 1) * dt)
            for i in range(len(graph.cells))] for k in range(horizon)])
        policies[departure] = {}
        for stage in ("primary", "sensitivity"):
            path = RUN / "policy_tables" / f"{stage}_t{departure}.npz"
            if path.exists():
                with np.load(path) as bundle:
                    policies[departure].update({m: bundle[m].copy() for m in bundle.files})
    replay = []
    max_segment_error = 0.
    for run in runs:
        departure = run["departure_fire_s"]
        path = [tuple(c) for c in run["path"]]
        assert len(path) == len(run["executed_steps"]) + 1 == run["steps"] + 1
        assert len(path) - 1 <= horizon
        assert all(c not in graph.exits for c in path[:-1])
        assert (path[-1] in graph.exits) == run["reached"]
        dose = 0.
        for k, step in enumerate(run["executed_steps"]):
            state = graph.index[path[k]]
            assert tuple(step["source"]) == path[k]
            assert tuple(step["destination"]) == path[k + 1]
            assert graph.cells[graph.destination(state, step["action"])] == path[k + 1]
            if run["method"] != "full_future_reference":
                assert step["action"] == policies[departure][run["method"]][k, state]
            dose += costs[departure][k, state]
            err = abs(float(costs[departure][k, state]) - step["FED_increment"])
            assert err < 1e-10
            max_segment_error = max(max_segment_error, err)
            assert abs(dose - step["FED_cumulative"]) < 1e-10
        error = abs(dose - run["FED"])
        assert error < 1e-10
        replay.append(dict(case_id=run["case_id"], method=run["method"], moves=run["steps"],
            cached_FED=run["FED"], independent_FED=float(dose), absolute_difference=error))
    write_csv(OUT / "independent_episode_replay.csv", replay)

    # Same cached action, reconstructed without the online forecast integral code.
    main_changes = [r for r in csv_rows(RUN / "changed_action_counterfactuals.csv")
                    if r["method"] == "trend_w30_c60"]
    points = {(int(r["departure_fire_s"]), int(r["step"]),
               graph.index[int(r["cell_x"]), int(r["cell_y"])]) for r in main_changes}
    points.add((300, 0, graph.index[8, 9]))
    independent = []
    for departure, k, state in sorted(points):
        decision = departure + k * dt
        for spec in cfg["primary_methods"]:
            rebuilt = independent_forecast_costs(times, measured, decision, dt, horizon - k, spec)
            history = available_prefix(times, measured, decision)
            model = make_forecast(history, spec["window_s"], spec["projection_cap_s"], spec["slope_gain"])
            original_costs = predicted_interval_costs(model, decision, dt, horizon - k)
            cost_error = float(np.max(np.abs(rebuilt - original_costs)))
            assert cost_error < 1e-10
            selected = int(policies[departure][spec["id"]][k, state])
            best = forward_value(graph, rebuilt, state)
            chosen = forward_value(graph, rebuilt, state, selected)
            assert abs(best - chosen) < 1e-10
            independent.append(dict(departure_fire_s=departure, step=k,
                cell=graph.cells[state], method=spec["id"], max_interval_cost_error=cost_error,
                independently_optimal_predicted_FED=best, chosen_action_predicted_FED=chosen))
    save(OUT / "independent_forecast_and_action_checks.json", independent)

    # Fixed-plan comparison: this is retrospective explanation, not another test.
    selected_cases = [(300, 0, (6, 9), "R22_benefit"),
                      (300, 0, (8, 9), "R24_remaining_gap"),
                      (450, 4, (5, 1), "R1_transient_detour"),
                      (600, 1, (5, 1), "R4_wrong_exit")]
    rankings, details = [], []
    main_spec = cfg["primary_methods"][1]
    for departure, k, cell, label in selected_cases:
        decision = departure + k * dt
        state = graph.index[cell]
        history = available_prefix(times, measured, decision)
        forecast = make_forecast(history, main_spec["window_s"], main_spec["projection_cap_s"], 1.)
        predicted = predicted_interval_costs(forecast, decision, dt, horizon - k)
        held = np.tile(history.rates[-1] * dt, (horizon - k, 1))
        real = costs[departure][k:]
        case = dict(label=label, departure_fire_s=departure, step=k, decision_fire_s=decision,
            latest_measurement_s=float(history.times[-1]), cell=cell, plans=[])
        for origin, planning_costs in (("snapshot_open_loop", held),
                                        ("trend_open_loop", predicted), ("future_reference", real)):
            actions, _, _ = future_min_fed_policy(graph, planning_costs)
            plan = rollout(graph, actions, planning_costs, state)
            ids = [graph.index[tuple(x)] for x in plan["path"][:-1]]
            truth_dose = float(sum(real[j, i] for j, i in enumerate(ids)))
            predicted_dose = float(sum(predicted[j, i] for j, i in enumerate(ids)))
            held_dose = float(sum(held[j, i] for j, i in enumerate(ids)))
            row = dict(case=label, plan_source=origin, decision_fire_s=decision,
                moves=plan["steps"], exit_region=data["mapping"][tuple(plan["exit"])],
                snapshot_predicted_FED=held_dose, trend_predicted_FED=predicted_dose,
                actual_fixed_plan_FED=truth_dose, revisits=plan["revisits"])
            rankings.append(row)
            case["plans"].append(dict(**row, path=plan["path"],
                region_path=[data["mapping"][tuple(c)] for c in plan["path"]]))
        details.append(case)
    write_csv(OUT / "selected_fixed_plan_rankings.csv", rankings)
    save(OUT / "selected_fixed_plan_details.json", details)
    by_case_method = {(r["case_id"], r["method"]): r for r in runs}
    main_pairs = [r for r in csv_rows(RUN / "paired_case_differences.csv") if r["method"] == "trend_w30_c60"]
    half_pairs = [r for r in csv_rows(RUN / "paired_case_differences.csv") if r["method"] == "trend_w30_c60_half"]
    all_pairs = csv_rows(RUN / "paired_case_differences.csv")
    method_summary = []
    for spec in cfg["primary_methods"][1:] + cfg["sensitivity_methods"]:
        pp = [r for r in all_pairs if r["method"] == spec["id"]]
        if not pp:
            continue
        by_difference = [float(r["difference_FED"]) for r in pp]
        method_summary.append(dict(method=spec["id"],
            helpful=sum(r["classification"] == "helpful" for r in pp),
            harmful=sum(r["classification"] == "harmful" for r in pp),
            neutral=sum(r["classification"] == "neutral" for r in pp),
            failure=sum(r["classification"] == "failure" for r in pp),
            equal_weight_case_mean_FED_difference=float(np.mean(by_difference)),
            ratio_of_means_reduction=-sum(by_difference) / sum(float(r["snapshot_FED"]) for r in pp)))
    write_csv(OUT / "all_departures_descriptive_effects.csv", method_summary)
    main_effect = next(r for r in method_summary if r["method"] == "trend_w30_c60")
    half_effect = next(r for r in method_summary if r["method"] == "trend_w30_c60_half")
    old = csv_rows(ROOT / cfg["previous_benchmark"] / "episode_metrics.csv")
    dqn = {r["case_id"]: float(r["FED"]) for r in old if r["method"] == "author_DQN_legal_mask"}
    dqn_comparison = []
    for departure in cfg["departure_fire_times_s"]:
        main = [r for r in runs if r["method"] == "trend_w30_c60" and r["departure_fire_s"] == departure]
        d = [r["FED"] - dqn[r["case_id"]] for r in main]
        mean_dqn = float(np.mean([dqn[r["case_id"]] for r in main]))
        dqn_comparison.append(dict(departure_fire_s=departure,
            trend_mean_FED=float(np.mean([r["FED"] for r in main])),
            archived_DQN_mean_FED=mean_dqn, trend_relative_increase_vs_DQN=float(np.mean(d)) / mean_dqn,
            trend_lower=sum(x < -1e-10 for x in d), equal=sum(abs(x) <= 1e-10 for x in d),
            trend_higher=sum(x > 1e-10 for x in d),
            scope="Context only: DQN inference inputs and prior training differ"))
    write_csv(OUT / "archived_DQN_context.csv", dqn_comparison)
    focus = []
    for case_id, reason in [("t0300_x06_y09", "only_primary_300s_gain"),
                            ("t0300_x08_y09", "original_R24_gap_not_solved"),
                            ("t0450_x01_y01", "transient_detour_and_backtrack"),
                            ("t0600_x04_y01", "shorter_route_but_higher_dose")]:
        snap = by_case_method[case_id, "snapshot_hold"]
        trend = by_case_method[case_id, "trend_w30_c60"]
        oracle = by_case_method[case_id, "full_future_reference"]
        diff = trend["FED"] - snap["FED"]
        focus.append(dict(case_id=case_id, reason=reason,
            snapshot_FED=snap["FED"], trend_FED=trend["FED"], oracle_FED=oracle["FED"],
            difference_FED=diff, relative_FED_increase=diff / snap["FED"],
            snapshot_steps=snap["steps"], trend_steps=trend["steps"],
            snapshot_path=snap["path"], trend_path=trend["path"],
            snapshot_revisits=snap["revisits"], trend_revisits=trend["revisits"]))
    save(OUT / "focus_case_results.json", focus)
    summary = dict(status="PASS_INDEPENDENT_CACHE_REVIEW", cached_episodes_verified=len(replay),
        cached_moves_verified=sum(r["moves"] for r in replay),
        maximum_episode_FED_error=max(r["absolute_difference"] for r in replay),
        maximum_segment_FED_error=max_segment_error,
        independent_forecast_and_action_checks=len(independent),
        descriptive_effects=method_summary, post_hoc=True,
        raw_region_count_interpretation="58 distinct region labels; not statistically independent fires or observations",
        new_closed_loop_rollouts=0, training_updates=0, LLM_requests=0, FDS_runs=0,
        sealed_final48_accessed=False)
    save(OUT / "review_results.json", summary)
    make_case_figure(focus)
    make_report(focus, main_effect, half_effect, dqn_comparison, method_summary, summary, details)
    # All original artifacts must still match, including the first report.
    verify(protected)
    links = re.findall(r"\]\((D:/[^)]+)\)", REPORT.read_text("utf-8"))
    assert all(Path(p).is_file() for p in links)
    files = [p for p in OUT.rglob("*") if p.is_file()] + [REPORT, Path(__file__)]
    save(OUT / "completion_manifest.json", dict(status="COMPLETE", protected_files_verified=len(protected),
        local_report_links_verified=len(links),
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", episodes_verified=len(replay),
        moves_verified=summary["cached_moves_verified"], independent_checks=len(independent),
        main=main_effect, half=half_effect), ensure_ascii=True), flush=True)


def make_case_figure(focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        titles = ["300秒／R22：唯一改善起点", "300秒／R24：主方案仍未解决",
                  "450秒／R1：折返增加暴露", "600秒／R4：更短路线仍可能更危险"]
        for ax, case, title in zip(axes.flat, focus, titles):
            for field, color, line, marker, label in [
                    ("snapshot_path", "#0072B2", "-", "o", "快照"),
                    ("trend_path", "#D55E00", "--", "s", "趋势")]:
                # Spatial paths can overlap; paired dose bars below provide exact values.
                # Use cumulative dose versus elapsed time to expose timing and backtracking costs.
                runs = json.loads((RUN / "rollouts.json").read_text("utf-8"))
                method = "snapshot_hold" if field == "snapshot_path" else "trend_w30_c60"
                run = next(r for r in runs if r["case_id"] == case["case_id"] and r["method"] == method)
                x = [0.] + [(s["step"] + 1) * 2 / .3 for s in run["executed_steps"]]
                y = [0.] + [s["FED_cumulative"] for s in run["executed_steps"]]
                ax.plot(x, y, color=color, linestyle=line, marker=marker, markersize=3, label=label)
            ax.set_title(title)
            ax.set_xlabel("出发后经过时间／秒")
            ax.set_ylabel("累计FED")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("同一已知火灾的事后案例解释：改善、未解决与伤害均保留", fontsize=11)
        fig.savefig(OUT / "four_case_cumulative_dose.png", dpi=180)
        fig.savefig(OUT / "four_case_cumulative_dose.svg")
        plt.close(fig)


def make_report(focus, main, half, dqn, sensitivity, summary, details):
    base = (ROOT / "reports/SafeStep历史观测预测与路径规划开发实验_2026-10-18.md").as_posix()
    p300 = next(r for r in dqn if r["departure_fire_s"] == 300)
    lookup = {r["reason"]: r for r in focus}
    r22, r24 = lookup["only_primary_300s_gain"], lookup["original_R24_gap_not_solved"]
    r1, r4 = lookup["transient_detour_and_backtrack"], lookup["shorter_route_but_higher_dose"]
    focus_table = "\n".join(f"| {r['case_id']} | {r['snapshot_FED']:.6f} | {r['trend_FED']:.6f} | "
        f"{r['relative_FED_increase']*100:+.2f}% | {r['snapshot_steps']}→{r['trend_steps']} |" for r in focus)
    sens_table = "\n".join(f"| {r['method']} | {r['helpful']} | {r['neutral']} | {r['harmful']} | "
        f"{r['ratio_of_means_reduction']*100:.3f}% |" for r in sensitivity)
    plans = []
    for case in details:
        for plan in case["plans"]:
            plans.append(f"| {case['label']} | {plan['plan_source']} | {plan['moves']} | {plan['exit_region']} | "
                f"{plan['trend_predicted_FED']:.6f} | {plan['actual_fixed_plan_FED']:.6f} |")
    content = f'''# SafeStep预测规划实验：结果解读与独立核查

日期：2026-10-18。建议从本文阅读本轮结论；[主实验完整报告]({base})保留方案、全部分时刻结果及计时。

**这一轮支持一个有限结论：简单历史预测有时能改善选路，但收益集中在少数起点，预测也会引发有害改道。还没有形成稳定超过既有DQN或跨火灾有效的方法。**

## 实际完成

冻结方案后运行61个起点、9个出发时刻、7种在线参数设定及1个完整未来参照，共4,392条轨迹、43,532个执行动作。四种敏感性设定按预先约定条件触发。全部轨迹到达网格出口；这是549个起点—时刻组合的重复方法比较，仍只有一个火灾。

主方案仅用最近30秒历史估计趋势，最多延伸60秒，然后保持。两种主要在线方法使用相同的原生5秒观测、地图、动作约束和规划器。没有调用LLM、训练PPO/DQN或运行FDS；48个最终人群场景继续封存。

## 收获一：预测存在局部用途，但不能把平均改善说成普遍改善

300秒出发时，快照平均FED为0.145939，趋势为0.143378，下降1.75%，平均行走时间68.31→66.56秒。这全部来自R22这一个起点：23步改为7步，FED从{r22['snapshot_FED']:.6f}降到{r22['trend_FED']:.6f}，降低{-r22['relative_FED_increase']*100:.2f}%。另外60个起点的FED没有变化。

在所有549个组合中，主方案{main['helpful']}个降低、{main['neutral']}个相同、{main['harmful']}个增加。九个出发时刻中七个平均降低、一个增加、一个不变。按549个组合等权计算的“均值之比”下降约{main['ratio_of_means_reduction']*100:.3f}%，受所选时刻和起点权重影响；不是独立火灾效应量。

## 收获二：最初提出问题的R24仍未被主方案解决

300秒R24的快照与主趋势都走21步B路线，FED均为{r24['trend_FED']:.6f}，完整未来参照为{r24['oracle_FED']:.6f}。这说明“未来信息有用”与“最近30秒直线趋势足以提供它”是两个问题。某些敏感性设定在该条件改善，不能据此把最佳参数改成新的确认性结果。

## 收获三：预测错误可能造成折返，也可能把人引向更短但更危险的出口

| 起点—时刻 | 快照FED | 趋势FED | 趋势相对变化（正值更差） | 步数变化 |
|---|---:|---:|---:|---:|
{focus_table}

450秒R1：主方案走到(5,1)后先转入(5,2)，下一步又回到(5,1)，再继续原通道。相比快照多走两步、约13.33秒，FED增加{r1['relative_FED_increase']*100:.2f}%。这是从已执行轨迹直接观察到的局部折返；不能仅据此归因于某个物理参数。

600秒R4：趋势把11步路线改为5步，但FED反而增加{r4['relative_FED_increase']*100:.2f}%。因此，“路线更短”“更早到出口”和“暴露更低”要分别评价。此处没有失能阈值或真实人群模型，不能将这些数值当作伤亡风险变化。

![四个案例的累积暴露]({(OUT/'four_case_cumulative_dose.png').as_posix()})

## 收获四：平均预测误差和决策后果需要同时看

主趋势在10／30／60秒预测上的平均绝对误差均比保持快照低，但均方根误差均更高。以60秒为例，MAE从0.004788降到0.003686，RMSE从0.008663升到0.010939。这表示它降低了绝对误差的平均值，却出现足以拉高平方误差的偏差；现有汇总不能单独定位这些误差发生在哪个区域。

半斜率版本在本开发数据中较保守：{half['helpful']}个改善、{half['neutral']}个相同、{half['harmful']}个有害。但它没有保留主方法在500秒R23的收益。因此“本轮没有有害回合”也不能解释成可部署的安全保证。

主方法逐动作反事实为16个变化机会、12个有益、4个有害；逐完整回合为11个改善、4个有害。两种计数不同，因为一个回合可有多个变化动作，后续动作也可能部分补偿前一步损失。

## 参数敏感性与DQN背景

以下按549个组合等权呈现，每个参数都是开发设定；没有将最好的一项改为主方法。

| 方法 | 更低FED | 相同 | 更高FED | 相对快照的均值之比下降 |
|---|---:|---:|---:|---:|
{sens_table}

300秒时既有DQN平均FED约{p300['archived_DQN_mean_FED']:.6f}，主趋势约{p300['trend_mean_FED']:.6f}，趋势仍高{100*p300['trend_relative_increase_vs_DQN']:.3f}%。这仅用于理解规模：DQN的推理输入和既有训练经历不同，不是公平训练对照。

## 独立复算检查

本次审计逐一重放全部{summary['cached_episodes_verified']}条已缓存轨迹和{summary['cached_moves_verified']}个动作。使用原生采样断点上的梯形积分重新计算，不调用主评分器的积分函数。最大整回合FED偏差{summary['maximum_episode_FED_error']:.3g}，在1e-10容差内。合法移动、轨迹长度、首次出口终止和动作缓存均吻合。

在主方法全部不同动作状态以及R24起点，共完成{summary['independent_forecast_and_action_checks']}项独立核对：用最小二乘另算斜率、按零点和外推截止点分段积分，再用正向最短路核查缓存动作在其预测目标下是否最优。均通过。说明当前偏差不是这些核查范围内的求解器选错动作；预测模型和滚动执行造成的后果仍需研究。

补充说明：主实验报告中的“58个独立区域标签”应理解为58个不同的区域编号，用于避免共享网格重复加权。它们在同一火灾中有空间和时间相关性，不是58个统计独立样本。

## 对固定计划的事后排序核查

下面把每个选定状态的快照最优计划、趋势最优计划及完整未来计划固定下来，分别给出趋势预测剂量和实际剂量。它用于解释当时的计划排序；实际闭环执行可能随新观测改道，不能把固定计划结果替换成主实验闭环结果。未来计划只在审计器中使用。

| 案例 | 计划来源 | 步数 | 出口 | 趋势预测FED | 实际固定计划FED |
|---|---|---:|---|---:|---:|
{chr(10).join(plans)}

## 下一步决定

本轮到此收尾。已经验证“简单历史预测有局部价值”，同时查明“收益稀疏、主方案未解决R24、部分预测引发折返或错误出口、参数敏感”这些限制。

现有证据不足以直接扩大训练或加入LLM。后续若继续方法研究，优先问题应是：在观测误差和趋势改变时，如何表达预测不确定性并减少有害路线切换？任何改法都应另立开发版本，保留普通预测基线，并在独立物理条件或明确标注的合成压力测试中评价。没有独立数据时，可把这轮并入既有复现章节的探索性扩展。

本轮数值主运行约9秒，完整主运行含绘图与写盘另有少量耗时；实现、设计和独立审计另计。已有危险率与小图动态规划可以快速计算，没有为用满1—2小时重复训练或重复推理。

## 复查入口

- [主实验报告]({base})
- [独立复算结果]({(OUT/'review_results.json').as_posix()})
- [每条轨迹的独立剂量核对]({(OUT/'independent_episode_replay.csv').as_posix()})
- [选定固定计划的完整路线]({(OUT/'selected_fixed_plan_details.json').as_posix()})
- [参数与收益计数]({(OUT/'all_departures_descriptive_effects.csv').as_posix()})
- [既有DQN背景比较]({(OUT/'archived_DQN_context.csv').as_posix()})
'''
    REPORT.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
