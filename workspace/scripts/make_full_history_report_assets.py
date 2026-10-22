from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/full_history_report_assets"
OUT.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")


def box(ax, xy, text, color="#e9f0f5", width=2.15, height=.85, size=12):
    x, y = xy
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.08", facecolor=color, edgecolor="#416174", linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontproperties=FONT, fontsize=size)


def arrow(ax, start, end, label=None, label_xy=None):
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#416174"})
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontproperties=FONT, fontsize=10, color="#345164")


fig, ax = plt.subplots(figsize=(12, 5.3))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.3); ax.axis("off")
box(ax, (1.4, 3.65), "房间A\n人员等待与暴露")
box(ax, (1.4, 1.7), "房间B\n人员等待与暴露")
box(ax, (5.7, 2.65), "受控路口\n行为模型选择方向", "#dceee9", width=2.5, height=1.15)
box(ax, (10.1, 3.65), "北侧出口走廊\n容量与危险约束")
box(ax, (10.1, 1.7), "南侧出口走廊\n容量与危险约束")
arrow(ax, (2.6, 3.65), (4.35, 2.9)); arrow(ax, (2.6, 1.7), (4.35, 2.4))
ax.text(3.35, 3.8, "房间总放行\n默认20人/步\n可调25%–100%或等待", ha="center", fontproperties=FONT, fontsize=10)
arrow(ax, (7.05, 2.9), (8.9, 3.65)); arrow(ax, (7.05, 2.4), (8.9, 1.7))
ax.text(8.1, 3.95, "分支默认12人/步\n出口默认8人/步", ha="center", fontproperties=FONT, fontsize=10)
box(ax, (5.7, 4.7), "控制器：方向＋放行容量", "#faf0da", width=3.5, height=.6, size=11)
arrow(ax, (5.7, 4.28), (5.7, 3.3))
ax.text(6, .42, "五分区容量流示意｜非真实建筑图｜默认240人，5秒/步，最大120步｜危险在全部分区随时间变化",
        ha="center", fontproperties=FONT, fontsize=10, color="#555555")
fig.tight_layout(); fig.savefig(OUT / "environment.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
rows = [
    (4.8, "原始研究方向", ["真实人员数据\n验证与校准", "LLM人员行为\n基线独立比较", "RL动态引导\n行为错配检验"], "#e9f0f5"),
    (2.95, "低成本控制实验", ["场景预测文字\n本地LLM缓存", "目标建议与置信度\nPPO实时纠错", "相同9动作对照\n新场景与多种子"], "#e6eee7"),
    (1.1, "后期开发诊断", ["复制仿真状态\n离线候选教师", "49维可观测学生\n软评分与排序", "新开发家庭审计\n不作为硬安全概率"], "#faf0da"),
]
for y, label, labels, color in rows:
    ax.text(.15, y, label, va="center", fontproperties=FONT, fontsize=11, color="#345164")
    for x, text in zip((3.3, 6.6, 10), labels):
        box(ax, (x, y), text, color, width=2.6, height=1.1, size=11)
    arrow(ax, (4.7, y), (5.2, y)); arrow(ax, (8, y), (8.6, y))
ax.text(6, 5.75, "三条链路的证据与作用不同，后期仿真教师不是LLM人员行为验证的替代证据", ha="center", fontproperties=FONT, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "research_tracks.png", dpi=180); plt.close(fig)
print(OUT)
