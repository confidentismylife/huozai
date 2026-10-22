"""Presentation-only revision: avoid overlapping labels/legends; no rescoring.

Preserve all first-version artifacts and emit separately versioned figures,
report and hashes. Counts and decisions are copied from verified results.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_checkpoint_diagnosis_20261016"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    previous_manifest = OUT / "completion_manifest.json"
    previous_hash = digest(previous_manifest)
    previous = json.loads(previous_manifest.read_text("utf-8"))
    for p, expected in previous["artifact_sha256"].items():
        assert digest(ROOT / p) == expected, p
    with (OUT / "checkpoint_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    arms = ["full_anchor", "selected", "selected_quarter"]
    names = ["A 四输出锚定", "B 单动作", "C 单动作÷4"]
    seeds = [141, 142, 143]
    groups = {(a, s): sorted([r for r in records if r["arm"] == a and int(r["seed"]) == s],
                             key=lambda r: int(r["checkpoint_step"])) for a in arms for s in seeds}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(12.5, 4.8), layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[1, .12])
        axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
        for ax, seed in zip(axes, seeds):
            for a, label, color, marker in zip(arms, names, ["#0072B2", "#D55E00", "#009E73"], ["o", "s", "^"]):
                rr = groups[a, seed]
                ax.plot([int(r["checkpoint_step"])/1000 for r in rr], [int(r["arrivals"]) for r in rr],
                        label=label, color=color, marker=marker, linewidth=1.8, markersize=6)
            ax.set(title=f"种子 {seed}", xlabel="训练控制步（千步）", xticks=[50, 100, 200],
                   xlim=(40, 210), ylim=(0, 64), yticks=[0, 10, 20, 30, 40, 50, 61])
            ax.axhline(61, ls="--", color=".5", lw=.8)
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("成功到达出口的起点数 / 61")
        legend = fig.add_subplot(gs[1, :])
        legend.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        legend.legend(handles, labels, loc="center", ncol=3, frameon=False)
        fig.suptitle("九个已有模型的阶段表现：三个固定检查点")
        for ext in ("png", "pdf"):
            p = OUT / f"checkpoint_arrivals_v2.{ext}"
            assert not p.exists()
            fig.savefig(p, dpi=220)
        plt.close(fig)

        fig = plt.figure(figsize=(12.5, 9.6), layout="constrained")
        gs = fig.add_gridspec(4, 3, height_ratios=[.12, 1, 1, 1])
        legend = fig.add_subplot(gs[0, :])
        legend.axis("off")
        axes = []
        for i, a in enumerate(arms):
            for j, seed in enumerate(seeds):
                ax = fig.add_subplot(gs[i+1, j])
                axes.append(ax)
                bottom = np.zeros(3)
                for key, label, color, hatch in zip(["arrivals", "blocked_starts", "cycle_starts"],
                        ["到达", "原地循环", "多格循环"], ["#0072B2", "#D55E00", "#999999"], ["", "//", ".."]):
                    vals = [int(r[key]) for r in groups[a, seed]]
                    ax.bar(range(3), vals, bottom=bottom, width=.6, label=label, color=color,
                           hatch=hatch, edgecolor="white", linewidth=.6)
                    bottom += vals
                assert (bottom == 61).all()
                ax.set(title=f"{names[i]} · 种子{seed}", ylim=(0, 64), xticks=range(3),
                       xticklabels=["5万", "10万", "20万"], yticks=[0, 20, 40, 61])
                if j == 0:
                    ax.set_ylabel("起点数 / 61")
        handles, labels = axes[0].get_legend_handles_labels()
        legend.legend(handles, labels, loc="center", ncol=3, frameon=False)
        fig.suptitle("全部起点的结果分类：每根柱合计61")
        for ext in ("png", "pdf"):
            p = OUT / f"checkpoint_failure_types_v2.{ext}"
            assert not p.exists()
            fig.savefig(p, dpi=200)
        plt.close(fig)

    lower = sum(int(rr[-1]["arrivals"]) < int(rr[0]["arrivals"]) for rr in groups.values())
    stage_counts = Counter()
    for rr in groups.values():
        for left, right in zip(rr, rr[1:]):
            d = int(right["arrivals"]) - int(left["arrivals"])
            stage_counts["decline" if d < 0 else "increase" if d > 0 else "equal"] += 1
    assert lower == 7 and stage_counts == {"decline": 11, "increase": 7}
    source = ROOT / "reports/SafeStep训练阶段缓存诊断_结果与收尾决定_2026-10-16.md"
    report = source.read_text("utf-8").replace("safestep_checkpoint_diagnosis_v1。", "safestep_checkpoint_diagnosis_v2（仅完善呈现；评价与决定不变）。")
    report = report.replace("阶段模式为{'非递增': 2, '先降后升': 4, '先升后降': 3}。",
        "九个模型中，七个20万步的可达起点数低于5万步；两个连续下降，另外七个先升后降或先降后升。")
    marker = "## 1. 逐模型结果"
    explanation = """本轮更支持“早期已有部分可达能力，但后续未稳定保留”的描述。以A组种子141为例，可达起点从40/61降至20/61，再降至4/61；A组种子142从35/61降至22/61，再降至10/61。其余运行存在回升，不能把整段训练描述为全部单调恶化。九模型共18段相邻变化中，11段下降、7段上升。

这不是“完整学会后又完全遗忘”的证明：27个时点均未达到61/61。三个检查点也不能排除更长训练后来恢复，但足以说明不能把“再加训练”当成可靠修复措施。阶段变化未识别出唯一原因，停止本轮后不追加机制试验。

"""
    report = report.replace(marker, explanation + marker)
    report = report.replace("checkpoint_arrivals.png", "checkpoint_arrivals_v2.png").replace("checkpoint_failure_types.png", "checkpoint_failure_types_v2.png")
    report = report.replace("../outputs/", ROOT.as_posix() + "/outputs/").replace("../schemas/", ROOT.as_posix() + "/schemas/")
    target = source.with_name(source.stem + "_v2.md")
    assert not target.exists()
    target.write_text(report, encoding="utf-8")
    assert digest(previous_manifest) == previous_hash
    artifacts = [OUT / f"{name}_v2.{ext}" for name in ("checkpoint_arrivals", "checkpoint_failure_types") for ext in ("png", "pdf")]
    artifacts += [target, Path(__file__)]
    manifest = dict(status="PRESENTATION_REVISION_V2", previous_completion_manifest_sha256=previous_hash,
        changes="Separate legend areas, no crowded point labels, redundant category hatching, explanatory paragraph from existing verified counts; no evaluation, training, or selection changes.",
        metric_source_sha256=digest(OUT / "checkpoint_metrics.csv"), lower_200k_than_50k_models=lower,
        adjacent_count_changes=dict(stage_counts), previous_artifacts_preserved=True,
        artifact_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts})
    (OUT / "presentation_revision_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k != "artifact_sha256"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
