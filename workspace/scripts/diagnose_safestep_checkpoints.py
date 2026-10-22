"""Bounded read-only checkpoint evaluation, followed by verified reporting.

Run prepare, evaluate, the separate verifier, and finalize in that order.
No optimizer, training entry point, external service, or held-out crowd data.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from fire_evac.safestep_author_numerics import (
    AuthorDQN, author_rollout, import_reviewed_source, load_author_inputs,
)

CONFIG = ROOT / "schemas/safestep_checkpoint_diagnosis_20261016.json"
ARMS = {"full_anchor": "A 四输出锚定", "selected": "B 单动作", "selected_quarter": "C 单动作÷4"}
CATEGORIES = ("exit_arrival", "blocked_self_loop", "multi_cell_cycle")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(p):
    return Path(p).relative_to(ROOT).as_posix()


def csv_rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(p, rows):
    assert rows
    with Path(p).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_directory(cfg, arm, seed):
    base = cfg["baseline_directory"] if arm == "full_anchor" else cfg["loss_directory"]
    source_arm = "every_step" if arm == "full_anchor" else arm
    return ROOT / base / "runs" / f"{source_arm}_seed{seed}"


def frozen_inputs(out):
    frozen = read(out / "frozen_manifest.json")
    for name, expected in frozen["input_sha256"].items():
        assert digest(ROOT / name) == expected, f"Changed input: {name}"
    return frozen


def prepare(cfg, out):
    out.mkdir(exist_ok=False)
    inputs = [CONFIG, Path(__file__), ROOT / "scripts/verify_safestep_checkpoints.py",
              ROOT / "src/fire_evac/safestep_author_numerics.py"]
    source = ROOT / cfg["source_directory"]
    inputs += [source / n for n in ("main.py", "maze.py", "coordinates2areanumber.py",
        "FED rate - room24.xlsx", "fire hazard zones FED - room24.xlsx")]
    inputs.append(source.parent / "numeric_run/results.json")
    inputs.append(ROOT / cfg["loss_directory"] / "nine_model_comparison.csv")
    historic = {}
    for folder in (cfg["baseline_directory"], cfg["loss_directory"]):
        p = ROOT / folder / "completion_manifest.json"
        inputs.append(p)
        historic.update({Path(k).as_posix(): v for k, v in read(p)["artifact_sha256"].items()})
    missing, checked = [], 0
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            d = run_directory(cfg, arm, seed)
            for name in ["episodes.csv", "budget_training_complete.json", "evaluation_traces.json",
                         "summary.json", "budget_final_model.pth"] + [
                         f"checkpoint_step{s:06d}.pth" for s in cfg["checkpoints_at_steps"]]:
                p = d / name
                if not p.exists():
                    missing.append(relative(p))
                    continue
                key = relative(p)
                assert key in historic, f"Input absent from previous completion manifest: {key}"
                assert digest(p) == historic[key], f"Prior input hash mismatch: {key}"
                checked += 1
                inputs.append(p)
    known = csv_rows(ROOT / cfg["loss_directory"] / "nine_model_comparison.csv")
    record = dict(status="FROZEN_BEFORE_EARLIER_CHECKPOINT_SCORING",
        created_at=datetime.now(timezone.utc).isoformat(),
        prior_manifest_input_hashes_checked=checked,
        known_final_arrivals=[dict(arm=r["loss"], seed=int(r["seed"]), arrivals=int(r["arrivals"])) for r in known],
        earlier_checkpoint_scores_inspected=False, missing_inputs=missing,
        input_sha256={relative(p): digest(p) for p in inputs})
    write(out / "frozen_manifest.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "input_sha256"}), flush=True)


def classify(trace):
    if trace["status"] == "WIN":
        return "exit_arrival", None, 0
    path = list(map(tuple, trace["path"]))
    seen = {}
    for i, xy in enumerate(path):
        if xy in seen:
            entry, period = seen[xy], i - seen[xy]
            assert all(path[j] == path[entry + (j - entry) % period] for j in range(entry, len(path)))
            return "blocked_self_loop" if period == 1 else "multi_cell_cycle", entry, period
        seen[xy] = i
    raise AssertionError("A failed stationary rollout has no cycle within the original timeout")


def training_metadata(directory, step):
    rows = csv_rows(directory / "episodes.csv")
    complete = [r for r in rows if int(r["global_steps"]) <= step and r["episode_status"] != "BUDGET_CUT"]
    epsilon = float(complete[-1]["epsilon_after"]) if complete else 1.0
    recent = complete[-100:]
    return dict(completed_episodes=len(complete), epsilon_at_checkpoint=epsilon,
        last_completed_episode_global_step=int(complete[-1]["global_steps"]) if complete else 0,
        recent_completed_episode_count=len(recent),
        recent_training_WIN_fraction=sum(r["episode_status"] == "WIN" for r in recent) / len(recent) if recent else None)


class CachedPolicy:
    def __init__(self, actions, coordinates, action_class):
        self.table = {xy: action_class(a) for xy, a in zip(coordinates, actions)}

    def predict(self, state):
        return self.table[tuple(map(int, state))]


def evaluate(cfg, out):
    frozen_inputs(out)
    assert not (out / "evaluation_complete.json").exists()
    started = time.monotonic()
    torch.set_num_threads(1)
    source = ROOT / cfg["source_directory"]
    data = load_author_inputs(source)
    maze = import_reviewed_source(source / "maze.py", "cached_checkpoint_author_maze")
    maze.Maze.reward_exit = 10.0
    reference = read(source.parent / "numeric_run/results.json")["stored_weight_runs"]
    (out / "checkpoints").mkdir(exist_ok=False)
    summaries, start_rows, missing = [], [], []
    identity = torch.eye(61)
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            d = run_directory(cfg, arm, seed)
            train = read(d / "budget_training_complete.json")
            assert list(map(tuple, train["coordinate_order"])) == data["empty"] + list(data["exits"])
            for step in cfg["checkpoints_at_steps"]:
                assert time.monotonic() - started < cfg["hard_timeout_seconds"], "Hard evaluation limit exceeded"
                p = d / f"checkpoint_step{step:06d}.pth"
                if not p.exists():
                    missing.append(relative(p))
                    continue
                model = AuthorDQN(61)
                model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
                model.eval()
                with torch.no_grad():
                    q = torch.stack([model(identity[i]) for i in range(61)]).numpy()
                assert q.shape == (61, 4) and np.isfinite(q).all()
                actions = q.argmax(axis=1)
                policy = CachedPolicy(actions, data["empty"], maze.Action)
                traces = []
                for xy in data["empty"]:
                    t = author_rollout(maze, data, policy, xy)
                    t.pop("integration_trace")
                    category, entry, period = classify(t)
                    t.update(start=xy, category=category, cycle_entry_step=entry, cycle_period=period)
                    traces.append(t)
                    start_rows.append(dict(arm=arm, seed=seed, checkpoint_step=step, start_x=xy[0], start_y=xy[1],
                        category=category, status=t["status"], steps=t["steps"], FED=t["FED"],
                        invalid_steps=t["invalid_steps"], revisits=t["revisits"], cycle_entry_step=entry, cycle_period=period))
                count = Counter(t["category"] for t in traces)
                summary = dict(arm=arm, seed=seed, checkpoint_step=step, arrivals=count["exit_arrival"],
                    blocked_starts=count["blocked_self_loop"], cycle_starts=count["multi_cell_cycle"],
                    invalid_steps=sum(t["invalid_steps"] for t in traces), revisits=sum(t["revisits"] for t in traces),
                    **training_metadata(d, step))
                for i, expected in enumerate(reference, 1):
                    t = next(t for t in traces if list(t["start"]) == expected["path"][0])
                    match = list(map(list, t["path"])) == expected["path"]
                    if match:
                        assert round(t["FED"], 4) == expected["paper_FED"]
                    summary.update({f"start{i}_status": t["status"], f"start{i}_steps": t["steps"],
                        f"start{i}_FED": t["FED"], f"start{i}_paper_path_match": match})
                summary["both_paper_routes_match"] = all(summary[f"start{i}_paper_path_match"] for i in (1, 2))
                tag = f"{arm}_seed{seed}_step{step:06d}"
                write(out / "checkpoints" / (tag + ".json"), dict(summary=summary, traces=traces,
                    scalar_float32_Q=q.tolist(), greedy_actions=actions.tolist(), checkpoint_sha256=digest(p)))
                summaries.append(summary)
                write(out / "progress.json", dict(status="evaluating", completed_checkpoints=len(summaries),
                    planned_checkpoints=27, rollouts=len(start_rows), elapsed_seconds=time.monotonic() - started))
                print(json.dumps({k: summary[k] for k in ("arm", "seed", "checkpoint_step", "arrivals", "blocked_starts", "cycle_starts")}), flush=True)
    csv_write(out / "checkpoint_metrics.csv", summaries)
    csv_write(out / "start_outcomes.csv", start_rows)
    write(out / "evaluation_complete.json", dict(status="CACHED_EVALUATION_COMPLETE", checkpoints=len(summaries),
        rollouts=len(start_rows), missing_checkpoints=missing, new_training_updates=0, LLM_requests=0,
        wall_seconds=time.monotonic() - started))
    frozen_inputs(out)


def progression(values):
    a, b = values[1] - values[0], values[2] - values[1]
    if a == b == 0:
        return "持平"
    if a >= 0 and b >= 0:
        return "非递减"
    if a <= 0 and b <= 0:
        return "非递增"
    return "先升后降" if a > 0 else "先降后升"


def finalize(cfg, out):
    verify = read(out / "verification.json")
    assert verify["status"] == "PASS"
    frozen_inputs(out)
    rows = csv_rows(out / "checkpoint_metrics.csv")
    for r in rows:
        for k in ("seed", "checkpoint_step", "arrivals", "blocked_starts", "cycle_starts", "completed_episodes"):
            r[k] = int(r[k])
        r["epsilon_at_checkpoint"] = float(r["epsilon_at_checkpoint"])
    groups = {(a, s): sorted([r for r in rows if r["arm"] == a and r["seed"] == s], key=lambda r: r["checkpoint_step"])
              for a in cfg["arms"] for s in cfg["seeds"]}
    trends = []
    for (a, s), rr in groups.items():
        if len(rr) != 3:
            continue
        v = [r["arrivals"] for r in rr]
        trends.append(dict(arm=a, seed=s, arrivals_50k=v[0], arrivals_100k=v[1], arrivals_200k=v[2],
            delta_50k_to_100k=v[1]-v[0], delta_100k_to_200k=v[2]-v[1], pattern=progression(v)))
    csv_write(out / "model_progressions.csv", trends)
    transitions = []
    for (arm, seed), rr in groups.items():
        for left, right in zip(rr, rr[1:]):
            def traces(step):
                return read(out / "checkpoints" / f"{arm}_seed{seed}_step{step:06d}.json")["traces"]
            old, new = traces(left["checkpoint_step"]), traces(right["checkpoint_step"])
            counts = Counter((x["category"], y["category"]) for x, y in zip(old, new))
            assert all(x["start"] == y["start"] for x, y in zip(old, new))
            for c1 in CATEGORIES:
                for c2 in CATEGORIES:
                    transitions.append(dict(arm=arm, seed=seed, from_step=left["checkpoint_step"],
                        to_step=right["checkpoint_step"], from_category=c1, to_category=c2, starts=counts[c1, c2]))
    csv_write(out / "adjacent_outcome_transitions.csv", transitions)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
             "axes.unicode_minus": False, "font.size": 10, "svg.fonttype": "none"}
    palette = ["#0072B2", "#D55E00", "#009E73"]
    with plt.rc_context(style):
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.3), layout="constrained", sharey=True)
        for ax, seed in zip(axs, cfg["seeds"]):
            for arm, color, marker in zip(cfg["arms"], palette, ["o", "s", "^"]):
                rr = groups[arm, seed]
                xx, yy = [r["checkpoint_step"] / 1000 for r in rr], [r["arrivals"] for r in rr]
                ax.plot(xx, yy, marker=marker, color=color, label=ARMS[arm], linewidth=1.7)
                for x, y in zip(xx, yy):
                    ax.annotate(str(y), (x, y), xytext=(4, 5), textcoords="offset points", color=color, fontsize=9)
            ax.set(title=f"种子 {seed}", xlabel="训练控制步（千步）", xticks=[50, 100, 200], ylim=(-2, 64))
            ax.axhline(61, color="0.5", ls="--", lw=.8)
            ax.grid(axis="y", alpha=.2)
        axs[0].set_ylabel("成功到达出口的起点数 / 61")
        axs[1].legend(loc="upper center", frameon=False, fontsize=9)
        fig.suptitle("已有检查点的确定性评价：全部九个模型，三个固定时点")
        fig.savefig(out / "checkpoint_arrivals.png", dpi=220)
        fig.savefig(out / "checkpoint_arrivals.pdf")
        plt.close(fig)
        fig, axs = plt.subplots(3, 3, figsize=(12.5, 9), sharex=True, sharey=True, layout="constrained")
        for i, arm in enumerate(cfg["arms"]):
            for j, seed in enumerate(cfg["seeds"]):
                ax, rr = axs[i, j], groups[arm, seed]
                bottom = np.zeros(len(rr))
                for key, label, color in zip(["arrivals", "blocked_starts", "cycle_starts"],
                        ["到达", "原地循环", "多格循环"], ["#0072B2", "#D55E00", "#999999"]):
                    vals = [r[key] for r in rr]
                    ax.bar(range(len(rr)), vals, bottom=bottom, width=.6, label=label, color=color,
                           edgecolor="white", linewidth=.5)
                    bottom += vals
                ax.set(title=f"{ARMS[arm]} · 种子{seed}", ylim=(0, 64), xticks=range(len(rr)),
                       xticklabels=["5万", "10万", "20万"][:len(rr)])
                if j == 0:
                    ax.set_ylabel("起点数 / 61")
        axs[0, 1].legend(ncol=3, frameon=False, fontsize=8)
        fig.suptitle("所有起点的结果分类；每根柱合计61")
        fig.savefig(out / "checkpoint_failure_types.png", dpi=200)
        fig.savefig(out / "checkpoint_failure_types.pdf")
        plt.close(fig)

    patterns = Counter(t["pattern"] for t in trends)
    totals = {step: {k: sum(r[k] for r in rows if r["checkpoint_step"] == step)
                    for k in ("arrivals", "blocked_starts", "cycle_starts")} for step in cfg["checkpoints_at_steps"]}
    decision = dict(status="BOUNDED_DIAGNOSIS_CLOSED_NO_AUTOMATIC_TRAINING", patterns=dict(patterns),
        counts_by_checkpoint=totals,
        scope="Same hotel,9 trained policies,3 repeated checkpoints,61 dependent start cells. Descriptive only.",
        no_best_checkpoint_selection=True, new_training_updates=0, LLM_requests=0,
        old_final48_crowd_scenarios_accessed=False,
        next_action="Consolidate reproduction results. Any future training-stability test needs a separately frozen mechanism and full budget.")
    write(out / "decision.json", decision)
    table = ["| 损失 | 种子 | 5万步到达 | 10万步到达 | 20万步到达 | 阶段变化 |", "|---|---:|---:|---:|---:|---|"]
    table += [f"| {ARMS[t['arm']]} | {t['seed']} | {t['arrivals_50k']}/61 | {t['arrivals_100k']}/61 | {t['arrivals_200k']}/61 | {t['pattern']} |" for t in trends]
    failures = ["| 时点 | 到达 | 原地循环 | 多格循环 |", "|---|---:|---:|---:|"]
    failures += [f"| {s//10000}万步 | {totals[s]['arrivals']} | {totals[s]['blocked_starts']} | {totals[s]['cycle_starts']} |" for s in cfg["checkpoints_at_steps"]]
    metadata = ["| 损失 / 种子 | 5万步回合数 / ε | 10万步回合数 / ε | 20万步回合数 / ε |", "|---|---:|---:|---:|"]
    metadata += [f"| {ARMS[a]} / {s} | " + " | ".join(f"{r['completed_episodes']} / {r['epsilon_at_checkpoint']:.3f}" for r in rr) + " |" for (a, s), rr in groups.items()]
    paper = ["| 损失 / 种子 / 时点 | 起点1：结果 / 步数 / FED | 起点2：结果 / 步数 / FED | 两路线匹配 |", "|---|---|---|---|"]
    for r in rows:
        paper.append(f"| {ARMS[r['arm']]} / {r['seed']} / {r['checkpoint_step']//10000}万 | " + " | ".join(
            f"{r[f'start{i}_status']} / {r[f'start{i}_steps']} / {float(r[f'start{i}_FED']):.6f}" for i in (1, 2)) + f" | {r['both_paper_routes_match']} |")
    loss_count = sum(t["starts"] for t in transitions if t["from_category"] == "exit_arrival" and t["to_category"] != "exit_arrival")
    gain_count = sum(t["starts"] for t in transitions if t["from_category"] != "exit_arrival" and t["to_category"] == "exit_arrival")
    report = f"""## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；scientific-visualization。
- Origin Mode：run + validate（已有检查点的离线重评）。
- Origin Date：2026-10-16。
- Verification Status：VERIFIED，限27检查点轨迹重建、FED积分、终点对齐和输入未变核查。
- Version Label：safestep_checkpoint_diagnosis_v1。

# SafeStep训练阶段缓存诊断：结果与收尾决定

本轮完成{len(rows)}个已有检查点、{len(rows)*61:,}条确定性回放；没有新增训练。阶段模式为{dict(patterns)}。**本轮诊断结束，停止自动扩展训练，转入复现结果整理。**

## 1. 逐模型结果

{chr(10).join(table)}

![九模型三个时点的到达数](../outputs/safestep_checkpoint_diagnosis_20261016/checkpoint_arrivals.png)

图中线段仅连接三个已测时点，不代表时点之间的连续学习曲线。全部种子均展示，没有挑选最佳检查点作为最终模型。

## 2. 错误类型与起点变化

{chr(10).join(failures)}

每个时点合计549次起点评价，来自九个模型各61个起点。不是549个独立建筑样本，也不能据此计算跨建筑成功概率。

![全部起点的到达与循环类型](../outputs/safestep_checkpoint_diagnosis_20261016/checkpoint_failure_types.png)

在两个相邻时段、九个模型中，共有{loss_count}次“到达→失败”和{gain_count}次“失败→到达”。同一起点可在两个时段重复计数；这些是配对状态变化次数，不是独立行人或独立实验数。详细3×3转移表保留在adjacent_outcome_transitions.csv。

原地循环指策略最终反复执行一个受阻动作；多格循环指最终周期包含至少两个格子。所有失败均按原程序运行到第301步，未因发现循环提前停止。循环是最终策略的可观察行为，不能单凭其存在定位训练中的唯一原因。

## 3. 训练进程与探索率

{chr(10).join(metadata)}

回合数只统计该时点之前已经完成的回合；ε采用最近已完成回合后的记录。评价时ε为0，即始终取最大Q动作。训练中的探索率与本轮确定性评价不能混为一谈。不同损失会改变轨迹、回合数和探索进程，因此相同控制步预算不意味着相同经验或探索率。

这些模型的20万步仅覆盖约1,800至2,600个完成回合；较早公开程序成功运行完成了约11,000至13,500回合。不能直接据此断言本轮训练已充分，也不能承诺延长就能成功。

## 4. 论文两个起点

{chr(10).join(paper)}

LOSE行的高FED来自原程序继续积分到超时，不是有效疏散方案或真实人员可继续行动的证据。两条路线匹配按预先固定的逐格路线核对；到达、剂量与路线匹配分别报告。

## 5. 执行和核查范围

- 在查看5万、10万步分数之前，冻结27个时点、全部种子、评价方法、决定规则及输入哈希；九个20万步分数事先已知，已明确登记，属于探索性诊断。
- 采用作者环境逐步执行，并以位置输入的逐状态float32推理缓存动作。没有更改动作、奖励、危险率或终止时刻。
- 独立核查用格点转移表重建{verify['verified_rollouts']:,}条完整路径、状态、循环、重访及FED；核对全部{verify['verified_checkpoints']}个检查点的Q值和动作。
- 九个20万步检查点与上轮保存的最终路径、状态、步数、FED、非法动作和重访全部吻合。独立重评是重复核查，不增加独立证据数。
- {verify['unchanged_frozen_inputs']}个冻结输入文件哈希未变；未重新核查与本轮无关的全部历史训练转移或优化器更新。
- 新训练更新0，LLM/API请求0；48个旧人群最终场景没有读取。

## 6. 解释与停止决定

单个模型在某个相邻时段下降，只能证明该两个固定时点的确定性策略表现下降；不能证明完整训练过程必然退化。若不同模型方向不同，不能选择最符合解释的一条曲线归因。即使某段持续改善，也不能证明最终会收敛。

本轮没有通过干预独立改变学习率、状态、奖励或终止机制，不据此确认唯一故障原因。**结束这条有限预算诊断；不自动增加训练轮数、种子或新的损失变体。**

后续交付聚焦原作者酒店案例：四个关键FED数值已按论文精度复现；公开训练程序的六次运行均61/61可达，其中两种设置的论文双路线匹配分别1/3与2/3。将成功复现和修改后的训练诊断分别呈现。原论文约62%改善所需的小建筑危险率数据仍缺失，无法用本轮酒店实验补足。

## 7. 统计误读检查（11/11）

按种子分列，防止混合趋势；不外推真人或建筑群体；全部模型与起点纳入，避免按成功或终点筛选产生Berkson/碰撞偏差；明确61格分母；不把一次反弹当作可靠提升；失败轨迹保留；不搜索显著结果或最佳时点；预先固定27个时点与输出；阶段相关不作唯一因果；不倒推原因。没有显著性检验、独立样本假定或未经支持的置信区间。

## 8. 可追溯文件

- [冻结方案](../schemas/safestep_checkpoint_diagnosis_20261016.json)
- [评分前输入登记](../outputs/safestep_checkpoint_diagnosis_20261016/frozen_manifest.json)
- [27检查点指标](../outputs/safestep_checkpoint_diagnosis_20261016/checkpoint_metrics.csv)
- [1,647条起点结果](../outputs/safestep_checkpoint_diagnosis_20261016/start_outcomes.csv)
- [逐模型变化](../outputs/safestep_checkpoint_diagnosis_20261016/model_progressions.csv)
- [独立核查](../outputs/safestep_checkpoint_diagnosis_20261016/verification.json)
- [机器可读决定](../outputs/safestep_checkpoint_diagnosis_20261016/decision.json)
"""
    report_path = ROOT / cfg["report"]
    assert not report_path.exists()
    report_path.write_text(report, encoding="utf-8")
    write(out / "figure_provenance.json", dict(source="checkpoint_metrics.csv", population="Same hotel,9 policies,3 checkpoints",
        transformations="No smoothing, interpolation, error bars, or exclusions. Lines connect only fixed observed checkpoints.",
        uncertainty="Not estimated:61 start cells are dependent repeated evaluations,3 seeds per arm",
        png_dpi=[220, 200], pdf="Vector", publisher_requirements="No specific journal formatting claimed",
        versions=dict(torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__)))
    frozen_inputs(out)
    artifacts = list(out.rglob("*")) + [report_path]
    write(out / "completion_manifest.json", dict(status="CHECKPOINT_DIAGNOSIS_COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        checkpoints=len(rows), rollouts=len(rows)*61, new_training_updates=0,
        old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={relative(p): digest(p) for p in artifacts if p.is_file() and p.name != "completion_manifest.json"}))
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "evaluate", "finalize"])
    args = parser.parse_args()
    cfg = read(CONFIG)
    out = ROOT / cfg["output_directory"]
    {"prepare": prepare, "evaluate": evaluate, "finalize": finalize}[args.phase](cfg, out)


if __name__ == "__main__":
    main()
