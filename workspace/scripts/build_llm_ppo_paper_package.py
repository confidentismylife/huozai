from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "llm_ppo_paper_package_20261020"
FIG = OUT / "figures"


def _setup_plotting() -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _match(df: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in conditions.items():
        mask &= df[key].eq(value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {conditions}, got {len(rows)}")
    return rows.iloc[0]


def _result_row(
    evidence: str,
    dataset: str,
    episodes: int,
    comparison: str,
    metric: str,
    unit: str,
    row: pd.Series,
    threshold: str,
    passed: str,
    interpretation: str,
    scale: float = 1.0,
) -> dict[str, object]:
    return {
        "证据等级": evidence,
        "数据集或分析单元": dataset,
        "回合或样本数": episodes,
        "比较": comparison,
        "指标": metric,
        "单位": unit,
        "效应值": float(row["mean_difference"]) * scale,
        "95%CI下限": float(row["ci95_low"]) * scale,
        "95%CI上限": float(row["ci95_high"]) * scale,
        "预设门槛": threshold,
        "是否通过": passed,
        "解释": interpretation,
    }


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    blind = pd.read_csv(ROOT / "outputs/final_blind_evaluation_20261020/paired_effects.csv")
    pipe = pd.read_csv(ROOT / "outputs/final_pipeline_context_20261020/paired_effects.csv")
    mech = pd.read_csv(ROOT / "outputs/final_mechanism_attribution_20261020/mechanism_effects.csv")
    pred = pd.read_csv(ROOT / "outputs/final_mechanism_attribution_20261020/predictability_metrics.csv")
    dec = pd.read_csv(ROOT / "outputs/direction_capacity_decoupling_dev_20261020_v2/paired_effects.csv")

    rows: list[dict[str, object]] = []

    for baseline, label in [("pure_ppo", "同控制器零建议"), ("rule_advice_ppo", "规则建议")]:
        for metric, zh, unit, scale in [
            ("incapacitation_rate", "失能率", "百分点", 100.0),
            ("mean_cumulative_fed", "人均累计FED", "FED", 1.0),
            ("evacuation_time_seconds", "回合终止时间", "秒", 1.0),
            ("cumulative_congestion", "累计拥堵", "指数", 1.0),
        ]:
            source = _match(
                blind,
                behavior_model="calibrated_logit",
                noise_level="none",
                candidate="llm_advice_ppo",
                baseline=baseline,
                metric=metric,
            )
            rows.append(
                _result_row(
                    "结局盲测；共享控制器消融",
                    "48个SCF场景×5训练种子；主要条件240回合",
                    240,
                    f"LLM建议−{label}",
                    zh,
                    unit,
                    source,
                    "失能率与FED为主要终点；对规则比较用于判断LLM增量",
                    "不适用" if baseline == "pure_ppo" else "未显示优于规则",
                    (
                        "零建议会改变已训练控制器的输入分布，不能等同独立纯PPO"
                        if baseline == "pure_ppo"
                        else "区间跨0或次要代价增加，未建立LLM增量优势"
                    ),
                    scale,
                )
            )

    for baseline, label in [("trained_pure_ppo", "独立纯PPO"), ("trained_rule_advice_ppo", "规则建议PPO")]:
        for metric, zh, unit, scale in [
            ("incapacitation_rate", "失能率", "百分点", 100.0),
            ("mean_cumulative_fed", "人均累计FED", "FED", 1.0),
            ("evacuation_time_seconds", "回合终止时间", "秒", 1.0),
        ]:
            source = _match(
                pipe,
                behavior_model="calibrated_logit",
                noise_level="none",
                candidate="trained_llm_advice_ppo",
                baseline=baseline,
                metric=metric,
            )
            rows.append(
                _result_row(
                    "解封后探索性；完整训练管线",
                    "48个SCF场景×5训练种子；主要条件240回合",
                    240,
                    f"LLM建议PPO−{label}",
                    zh,
                    unit,
                    source,
                    "95%CI应排除0，且安全与代价方向一致",
                    "未通过",
                    "点估计不能排除无改善或变差；不构成确认性优势",
                    scale,
                )
            )

    for metric, zh, unit, scale in [
        ("mean_requested_release_fraction_nonwait", "平均请求放行档位", "比例", 1.0),
        ("low_release_fraction_nonwait", "低档位动作比例", "比例", 1.0),
        ("rooms_clear_seconds", "房间清空时间", "秒", 1.0),
        ("evacuated_50_seconds", "50%撤离时间", "秒", 1.0),
        ("evacuated_90_seconds", "90%撤离时间", "秒", 1.0),
    ]:
        source = _match(
            mech,
            behavior_model="calibrated_logit",
            noise_level="none",
            candidate="trained_llm_advice_ppo",
            baseline="trained_pure_ppo",
            metric=metric,
        )
        rows.append(
            _result_row(
                "解封后探索性；动作级重放",
                "原2,880回合动作级重放；主要条件240配对单元",
                2880,
                "LLM建议PPO−独立纯PPO",
                zh,
                unit,
                source,
                "机制归因，无确认性优越门槛",
                "机制证据",
                "用于解释时间代价，不作单一因果证明",
                scale,
            )
        )

    for _, source in pred.iterrows():
        baseline = "独立纯PPO" if source["baseline"] == "trained_pure_ppo" else "规则建议PPO"
        features = "公开观测" if source["feature_set"] == "public_observation" else "公开观测＋LLM字段"
        rows.append(
            {
                "证据等级": "解封后探索性；场景分组交叉验证",
                "数据集或分析单元": "192个场景×行为×噪声单元；48个场景，8折",
                "回合或样本数": int(source["rows"]),
                "比较": f"预测LLM相对{baseline}是否有益；{features}",
                "指标": "ROC AUC",
                "单位": "AUC",
                "效应值": float(source["roc_auc"]),
                "95%CI下限": float(source["roc_auc_ci95_low"]),
                "95%CI上限": float(source["roc_auc_ci95_high"]),
                "预设门槛": "两种基线均要求AUC下限>0.60且Brier改善≥5%",
                "是否通过": "未通过",
                "解释": f"Brier相对频率基线改善{float(source['brier_improvement_percent']):.1f}%",
            }
        )

    for baseline, label in [("trained_llm_joint", "原LLM联合控制"), ("trained_pure_ppo", "独立纯PPO")]:
        for metric, zh, unit, scale in [
            ("incapacitation_rate", "失能率", "百分点", 100.0),
            ("mean_cumulative_fed", "人均累计FED", "FED", 1.0),
            ("evacuation_time_seconds", "回合终止时间", "秒", 1.0),
            ("mean_release_fraction", "平均放行档位", "比例", 1.0),
        ]:
            source = _match(
                dec,
                candidate="llm_policy_direction_pure_capacity",
                baseline=baseline,
                metric=metric,
            )
            rows.append(
                _result_row(
                    "开发实验；旧V2FE场景",
                    "60个开发场景×5人口重复×5训练种子",
                    1500,
                    f"LLM动态方向＋纯PPO容量−{label}",
                    zh,
                    unit,
                    source,
                    "时间CI上限<0；失能率CI上限<+0.5百分点；FED CI上限<+0.01",
                    "仅时间通过" if baseline == "trained_llm_joint" else "未显示增量",
                    "容量解耦恢复速度，但安全非劣门槛未通过",
                    scale,
                )
            )

    result_table = pd.DataFrame(rows)
    result_table.to_csv(OUT / "论文主表_统一结果.csv", index=False, encoding="utf-8-sig")

    registry = pd.DataFrame(
        [
            {
                "实验": "共享控制器最终盲测",
                "证据等级": "结局盲测；确认性主分析",
                "场景": "48个SCF场景；6类冲突，每类8个",
                "训练种子": "601–605，共5个",
                "人群与噪声": "2种人群×2档噪声",
                "回合或样本": "2,880回合；主要条件每方法240回合",
                "新增训练": "0",
                "新增LLM请求": "48个qwen3:4b建议，结局运行前冻结",
                "回答的问题": "固定同一LLM管线权重时，建议输入相对置零或规则输入的影响",
                "主要限制": "零建议不是独立训练纯PPO，存在输入分布变化",
            },
            {
                "实验": "三条完整训练管线补充对照",
                "证据等级": "解封后探索性",
                "场景": "同48个已解封SCF场景",
                "训练种子": "601–605，共5个",
                "人群与噪声": "2种人群×2档噪声",
                "回合或样本": "新增1,920＋复用960＝2,880回合",
                "新增训练": "0；使用既有独立训练权重",
                "新增LLM请求": "0",
                "回答的问题": "训练和部署模式一致时，LLM、规则、纯PPO三管线表现",
                "主要限制": "协议在共享控制器结果解封后冻结",
            },
            {
                "实验": "动作机制归因",
                "证据等级": "解封后探索性",
                "场景": "同48个已解封SCF场景",
                "训练种子": "601–605，共5个",
                "人群与噪声": "2种人群×2档噪声",
                "回合或样本": "2,880回合动作级重放",
                "新增训练": "0",
                "新增LLM请求": "0",
                "回答的问题": "时间代价来自等待、方向还是放行容量",
                "主要限制": "归因基于相关与配对机制指标，不是单因素随机试验",
            },
            {
                "实验": "收益可预测性",
                "证据等级": "解封后探索性",
                "场景": "48个场景，按场景分组8折",
                "训练种子": "先对5种子平均",
                "人群与噪声": "2种人群×2档噪声",
                "回合或样本": "192个分类单元",
                "新增训练": "4个固定逻辑回归预测器",
                "新增LLM请求": "0",
                "回答的问题": "决策时可见信息能否识别LLM何时有益",
                "主要限制": "最终场景已解封，仅能生成假设",
            },
            {
                "实验": "方向与容量解耦",
                "证据等级": "开发实验",
                "场景": "60个历史V2FE开发场景；6类冲突",
                "训练种子": "601–605，共5个",
                "人群与噪声": "5个人口重复",
                "回合或样本": "新增4,500＋复用3,000＝7,500回合",
                "新增训练": "0",
                "新增LLM请求": "0",
                "回答的问题": "保留LLM方向、交由纯PPO控制容量能否恢复速度且保持安全",
                "主要限制": "开发场景多次使用；综合门槛未通过",
            },
        ]
    )
    registry.to_csv(OUT / "实验登记与证据等级表.csv", index=False, encoding="utf-8-sig")
    return result_table, registry


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 1.4, 2.0, 1.2, "现场观测\n34维状态＋时间戳", "#DCEEFF"),
        (3.0, 2.35, 2.2, 1.1, "LLM／规则\n高层建议", "#FFE7C2"),
        (3.0, 0.55, 2.2, 1.1, "PPO\n基础联合动作", "#E3F4E1"),
        (6.0, 1.4, 2.2, 1.2, "建议通道与检查\n出口、容量、时效", "#F2E3F9"),
        (9.0, 1.4, 1.8, 1.2, "共同执行约束\n方向＋放行档位", "#E8E8E8"),
        (11.4, 1.4, 1.3, 1.2, "人群与\n烟气演化", "#FFDADA"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#455A64", linewidth=1.2
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
    arrows = [
        ((2.3, 2.0), (3.0, 2.9)),
        ((2.3, 2.0), (3.0, 1.1)),
        ((5.2, 2.9), (6.0, 2.15)),
        ((5.2, 1.1), (6.0, 1.85)),
        ((8.2, 2.0), (9.0, 2.0)),
        ((10.8, 2.0), (11.4, 2.0)),
        ((12.05, 1.4), (1.3, 1.35)),
    ]
    for start, end in arrows:
        connection = "arc3,rad=-0.22" if start[0] > end[0] else "arc3,rad=0"
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#455A64", connectionstyle=connection))
    ax.text(6.5, 3.75, "研究对象：LLM建议能否在相同信息、动作与预算下提供稳定增量价值", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(6.7, 0.08, "PPO与LLM建议均受相同约束；新观测进入下一控制步", ha="center", va="bottom", fontsize=9.5, color="#455A64")
    fig.tight_layout()
    fig.savefig(FIG / "图1_系统流程.png", bbox_inches="tight")
    plt.close(fig)


def draw_primary_forest() -> None:
    pipe = pd.read_csv(ROOT / "outputs/final_pipeline_context_20261020/paired_effects.csv")
    specs = [
        ("trained_pure_ppo", "LLM−纯PPO", "#1f77b4"),
        ("trained_rule_advice_ppo", "LLM−规则PPO", "#d95f02"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    metrics = [
        ("incapacitation_rate", "失能率差（百分点）", 100.0),
        ("mean_cumulative_fed", "人均累计FED差", 1.0),
        ("evacuation_time_seconds", "回合终止时间差（秒）", 1.0),
    ]
    for ax, (metric, title, scale) in zip(axes, metrics):
        for y, (baseline, label, color) in enumerate(specs):
            row = _match(pipe, behavior_model="calibrated_logit", noise_level="none", candidate="trained_llm_advice_ppo", baseline=baseline, metric=metric)
            mean = row["mean_difference"] * scale
            lo = row["ci95_low"] * scale
            hi = row["ci95_high"] * scale
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4, markersize=7)
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_yticks([0, 1], [s[1] for s in specs])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        if metric != "incapacitation_rate":
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("图2  三条完整训练管线的主要条件效应（95% CI）", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "负值表示LLM管线指标更低；分析在共享控制器结果解封后完成，属于探索性证据。", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(FIG / "图2_完整管线主要效应.png", bbox_inches="tight")
    plt.close(fig)


def draw_mechanism() -> None:
    mech = pd.read_csv(ROOT / "outputs/final_mechanism_attribution_20261020/mechanism_effects.csv")
    def val(metric: str) -> pd.Series:
        return _match(mech, behavior_model="calibrated_logit", noise_level="none", candidate="trained_llm_advice_ppo", baseline="trained_pure_ppo", metric=metric)

    release = val("mean_requested_release_fraction_nonwait")
    clear = val("rooms_clear_seconds")
    t50 = val("evacuated_50_seconds")
    t90 = val("evacuated_90_seconds")
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    nodes = [
        (0.3, 1.8, 2.5, 1.5, "LLM管线请求更低放行档位", f"差 {release.mean_difference:+.3f}\n95% CI [{release.ci95_low:+.3f}, {release.ci95_high:+.3f}]", "#FFE0B2"),
        (3.7, 1.8, 2.5, 1.5, "人员在房间滞留更久", f"清空晚 {clear.mean_difference:.1f} 秒\n95% CI [{clear.ci95_low:.1f}, {clear.ci95_high:.1f}]", "#FFF3CD"),
        (7.1, 2.75, 2.5, 1.35, "前半程几乎不变", f"50%撤离 {t50.mean_difference:+.2f} 秒\n95% CI [{t50.ci95_low:.2f}, {t50.ci95_high:.2f}]", "#DDEBF7"),
        (7.1, 0.65, 2.5, 1.35, "尾部撤离明显变慢", f"90%撤离 +{t90.mean_difference:.2f} 秒\n95% CI [{t90.ci95_low:.2f}, {t90.ci95_high:.2f}]", "#F8D7DA"),
        (10.6, 1.8, 2.5, 1.5, "总体时间代价", "LLM−纯PPO +22.50 秒\n房间清空差与总时间差\nSpearman ρ=0.896", "#E2E3E5"),
    ]
    for x, y, w, h, title, body, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#555", linewidth=1.2))
        ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", fontsize=9.2)
    for start, end in [((2.8, 2.55), (3.7, 2.55)), ((6.2, 2.55), (7.1, 3.42)), ((6.2, 2.55), (7.1, 1.32)), ((9.6, 3.42), (10.6, 2.8)), ((9.6, 1.32), (10.6, 2.3))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#555", linewidth=1.4))
    ax.text(6.75, 4.75, "图3  时间损失的机制链：容量保守主要拖慢尾部", ha="center", fontsize=14, fontweight="bold")
    ax.text(6.75, 0.12, "动作重放支持这一解释链，但相关系数与中介顺序不等于严格因果证明。", ha="center", fontsize=9.5, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "图3_容量与尾部机制.png", bbox_inches="tight")
    plt.close(fig)


def draw_gate_and_decoupling() -> None:
    pred = pd.read_csv(ROOT / "outputs/final_mechanism_attribution_20261020/predictability_metrics.csv")
    dec = pd.read_csv(ROOT / "outputs/direction_capacity_decoupling_dev_20261020_v2/paired_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    labels = []
    for _, row in pred.iterrows():
        base = "对纯PPO" if row["baseline"] == "trained_pure_ppo" else "对规则"
        feat = "公开观测" if row["feature_set"] == "public_observation" else "＋LLM字段"
        labels.append(f"{base}｜{feat}")
    y = np.arange(len(pred))
    auc = pred["roc_auc"].to_numpy()
    lo = pred["roc_auc_ci95_low"].to_numpy()
    hi = pred["roc_auc_ci95_high"].to_numpy()
    axes[0].errorbar(auc, y, xerr=[auc - lo, hi - auc], fmt="o", capsize=4, color="#2a6fbb", markersize=7)
    axes[0].axvline(0.5, color="#777", linestyle="--", linewidth=1, label="随机水平")
    axes[0].axvline(0.6, color="#c44e52", linestyle=":", linewidth=1.5, label="门槛下限要求")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.4, 0.8)
    axes[0].set_xlabel("ROC AUC（95% CI）")
    axes[0].set_title("A. 收益预测：4/4均未通过", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=8)

    metrics = [
        ("evacuation_time_seconds", "时间（秒）", -20.4967, -36.0875, -3.8656, "通过"),
        ("incapacitation_rate", "失能率（百分点）", 0.8042, -0.0081, 2.0787, "未通过"),
        ("mean_cumulative_fed", "人均FED", 0.001661, -0.006499, 0.011650, "未通过"),
    ]
    yy = np.arange(len(metrics))
    colors = ["#55a868", "#c44e52", "#c44e52"]
    for i, (_, _, mean, low, high, _) in enumerate(metrics):
        axes[1].errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="o", capsize=4, color=colors[i], markersize=7)
    axes[1].axvline(0, color="#777", linestyle="--", linewidth=1)
    axes[1].set_yticks(yy, [f"{m[1]}｜{m[5]}" for m in metrics])
    axes[1].invert_yaxis()
    axes[1].set_xscale("symlog", linthresh=0.02)
    axes[1].set_xlabel("混合方案−原LLM联合控制（95% CI；对称对数轴）")
    axes[1].set_title("B. 方向/容量解耦：仅时间门槛通过", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("图4  为什么不继续扩大实时LLM控制实验", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "收益预测无法可靠筛选有益场景；容量解耦虽提速，但安全非劣未通过。", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(FIG / "图4_收益预测与解耦门槛.png", bbox_inches="tight")
    plt.close(fig)


def draw_composite() -> None:
    files = [
        ("图1_系统流程.png", "A  控制闭环"),
        ("图2_完整管线主要效应.png", "B  完整管线效应"),
        ("图3_容量与尾部机制.png", "C  时间损失机制"),
        ("图4_收益预测与解耦门槛.png", "D  停止扩大实验的依据"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5))
    for ax, (filename, label) in zip(axes.flat, files):
        ax.imshow(plt.imread(FIG / filename))
        ax.axis("off")
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold", pad=6)
    fig.suptitle("LLM建议＋PPO火灾疏散控制：效果、机制与失败边界", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "图5_论文组合主图.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _setup_plotting()
    build_tables()
    draw_architecture()
    draw_primary_forest()
    draw_mechanism()
    draw_gate_and_decoupling()
    draw_composite()
    print(OUT)


if __name__ == "__main__":
    main()
