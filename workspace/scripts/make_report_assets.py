from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fill: str) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline="#35566f", width=2)
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    f = font(29, bold=True)
    heights = [draw.textbbox((0, 0), line, font=f)[3] for line in lines]
    total = sum(heights) + 10 * (len(lines) - 1)
    y = (y1 + y2 - total) / 2
    for line, h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=f)
        x = (x1 + x2 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=f, fill="#102a3a")
        y += h + 10


def architecture() -> None:
    image = Image.new("RGB", (1800, 760), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    draw.text((80, 40), "真实行为校准与动态引导的闭环实验架构", font=title_font, fill="#111111")

    boxes = [
        (80, 190, 360, 400, "真实行为数据\n参与者级划分", "#dceaf5"),
        (430, 190, 710, 400, "LLM行为模型\n独立验证与冻结", "#e6f2e3"),
        (780, 190, 1060, 400, "行为代理模型\n事件驱动决策", "#fff0cf"),
        (1130, 190, 1410, 400, "FDS与JuPedSim\n火灾和运动", "#f4e4eb"),
        (1480, 190, 1740, 400, "RL动态标志\nPPO稳健训练", "#e8e4f5"),
    ]
    for x1, y1, x2, y2, text, fill in boxes:
        rounded_box(draw, (x1, y1, x2, y2), text, fill)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][2] + 10
        x2 = boxes[i + 1][0] - 10
        y = 295
        draw.line((x1, y, x2, y), fill="#35566f", width=5)
        draw.polygon([(x2, y), (x2 - 18, y - 12), (x2 - 18, y + 12)], fill="#35566f")

    rounded_box(draw, (410, 500, 1390, 650), "锁定测试：行为模型交叉 × 未见火源 × 传感器扰动", "#edf1f4")
    draw.line((870, 400, 870, 500), fill="#35566f", width=5)
    draw.polygon([(870, 500), (858, 482), (882, 482)], fill="#35566f")
    image.save(OUT / "architecture.png", quality=95)


def prototype_chart() -> None:
    summary_path = ROOT / "outputs" / "prototype_summary.csv"
    rows = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["behavior_model"] == "mixed_logit":
                rows.append(row)
    order = ["hazard_aware_rule", "neutral", "q_learning", "static_north"]
    rows.sort(key=lambda r: order.index(r["controller"]))

    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "合成原型中的Mixed-logit人群结果", font=font(42, True), fill="#111111")
    draw.text((90, 105), "用于验证代码闭环，不作为论文科学结论", font=font(27), fill="#7a2f2f")

    left, top, right, bottom = 170, 220, 1500, 760
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    labels = {
        "hazard_aware_rule": "危险感知规则",
        "neutral": "中性标志",
        "q_learning": "Q-learning",
        "static_north": "固定指北",
    }
    colors = ["#2f6f8f", "#659b76", "#d59b3a", "#b85c5c"]
    bar_width = 210
    gap = 95
    for i, row in enumerate(rows):
        evac = float(row["evacuation_rate_mean"])
        incapac = float(row["incapacitation_rate_mean"])
        loss = 1.0 - evac
        height = max(3, int(loss * 8000))
        x1 = left + 100 + i * (bar_width + gap)
        x2 = x1 + bar_width
        y1 = bottom - height
        draw.rectangle((x1, y1, x2, bottom), fill=colors[i], outline="#333333")
        draw.text((x1 + 25, y1 - 45), f"未撤离/失能 {loss:.3%}", font=font(23), fill="#111111")
        draw.text((x1 + 35, bottom + 25), labels[row["controller"]], font=font(25), fill="#111111")
        draw.text((x1 + 30, bottom + 66), f"FED {float(row['mean_cumulative_fed']):.3f}", font=font(22), fill="#444444")
        draw.text((x1 + 30, bottom + 102), f"失能 {incapac:.3%}", font=font(22), fill="#444444")
    draw.text((35, 430), "失败比例\n放大显示", font=font(24), fill="#333333")
    image.save(OUT / "prototype_results.png", quality=95)


def ppo_chart() -> None:
    summary_path = ROOT / "outputs" / "ppo" / "ppo_summary.csv"
    rows = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["behavior_model"] == "calibrated_logit":
                rows.append(row)
    order = ["hazard_aware_rule", "ppo", "neutral", "static_north"]
    rows.sort(key=lambda r: order.index(r["controller"]))

    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "合成校准Logit人群中的PPO结果", font=font(42, True), fill="#111111")
    draw.text((90, 105), "训练50000步；每种控制器评估300个配对场景；不作为真实人员结论", font=font(27), fill="#7a2f2f")
    left, top, right, bottom = 170, 220, 1500, 760
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    labels = {
        "hazard_aware_rule": "危险感知规则",
        "ppo": "PPO",
        "neutral": "中性标志",
        "static_north": "固定指北",
    }
    colors = ["#2f6f8f", "#7b63a6", "#659b76", "#b85c5c"]
    bar_width = 210
    gap = 95
    for i, row in enumerate(rows):
        evac = float(row["evacuation_rate_mean"])
        incapac = float(row["incapacitation_rate_mean"])
        loss = 1.0 - evac
        height = max(3, int(loss * 13000))
        x1 = left + 100 + i * (bar_width + gap)
        x2 = x1 + bar_width
        y1 = bottom - height
        draw.rectangle((x1, y1, x2, bottom), fill=colors[i], outline="#333333")
        draw.text((x1 + 25, y1 - 45), f"未撤离/失能 {loss:.3%}", font=font(23), fill="#111111")
        draw.text((x1 + 35, bottom + 25), labels[row["controller"]], font=font(25), fill="#111111")
        draw.text((x1 + 30, bottom + 66), f"FED {float(row['mean_cumulative_fed']):.3f}", font=font(22), fill="#444444")
        draw.text((x1 + 30, bottom + 102), f"失能 {incapac:.3%}", font=font(22), fill="#444444")
    draw.text((35, 430), "失败比例\n放大显示", font=font(24), fill="#333333")
    image.save(OUT / "ppo_results.png", quality=95)


def multiseed_chart() -> None:
    summary_path = ROOT / "outputs" / "ppo_multiseed" / "multiseed_per_seed_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["behavior_model"] == "calibrated_logit"]
    ppo_rows = sorted(
        [row for row in rows if row["controller"] == "ppo"], key=lambda row: int(row["training_seed"])
    )
    neutral = next(float(row["incapacitation_rate_mean"]) for row in rows if row["controller"] == "neutral")
    hazard = next(
        float(row["incapacitation_rate_mean"]) for row in rows if row["controller"] == "hazard_aware_rule"
    )
    values = [float(row["incapacitation_rate_mean"]) for row in ppo_rows]

    image = Image.new("RGB", (1600, 850), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "五个PPO训练种子的失能率", font=font(42, True), fill="#111111")
    draw.text((90, 105), "合成校准Logit人群；每个种子评估300个配对场景", font=font(27), fill="#7a2f2f")
    left, top, right, bottom = 180, 200, 1500, 700
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    maximum = 0.018

    def y_of(value: float) -> float:
        return bottom - value / maximum * (bottom - top)

    for tick in (0.0, 0.005, 0.010, 0.015):
        y = y_of(tick)
        draw.line((left - 8, y, right, y), fill="#dddddd", width=1)
        draw.text((55, y - 15), f"{tick:.1%}", font=font(22), fill="#444444")
    draw.line((left, y_of(neutral), right, y_of(neutral)), fill="#659b76", width=5)
    draw.text((right - 310, y_of(neutral) - 37), f"中性 {neutral:.3%}", font=font(23), fill="#477452")
    draw.line((left, y_of(hazard), right, y_of(hazard)), fill="#2f6f8f", width=5)
    draw.text((right - 390, y_of(hazard) - 37), f"危险感知规则 {hazard:.3%}", font=font(23), fill="#244f66")
    points = []
    for i, (row, value) in enumerate(zip(ppo_rows, values)):
        x = left + 130 + i * 245
        y = y_of(value)
        points.append((x, y))
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#7b63a6", outline="#3f315c", width=3)
        draw.text((x - 36, bottom + 25), f"种子{row['training_seed']}", font=font(23), fill="#222222")
        draw.text((x - 42, y - 48), f"{value:.3%}", font=font(21), fill="#3f315c")
    if len(points) > 1:
        draw.line(points, fill="#7b63a6", width=4)
    mean_value = sum(values) / len(values)
    draw.text((90, 765), f"PPO均值 {mean_value:.3%}；种子间差异必须纳入论文不确定性分析", font=font(25), fill="#333333")
    image.save(OUT / "ppo_multiseed_results.png", quality=95)


def behavior_cross_heatmap() -> None:
    matrix_path = ROOT / "outputs" / "behavior_cross" / "cross_matrix.csv"
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_order = ["logit", "ensemble", "rule", "llm"]
    test_order = ["calibrated_logit", "llm_surrogate_unvalidated", "mixed_logit", "rule"]
    train_labels = {"logit": "PPO-Logit", "ensemble": "PPO-Ensemble", "rule": "PPO-Rule", "llm": "PPO-LLM"}
    test_labels = {
        "calibrated_logit": "校准Logit",
        "llm_surrogate_unvalidated": "LLM代理",
        "mixed_logit": "Mixed-logit",
        "rule": "规则人员",
    }
    lookup = {(row["training_variant"], row["test_behavior"]): row for row in rows}
    image = Image.new("RGB", (1700, 1020), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), "PPO训练行为 × 测试行为失能率", font=font(42, True), fill="#111111")
    draw.text((80, 100), "4步状态历史；3个训练种子；每格200个配对场景", font=font(27), fill="#7a2f2f")
    left, top = 380, 260
    cell_w, cell_h = 290, 145
    for column, test_name in enumerate(test_order):
        x = left + column * cell_w
        bbox = draw.textbbox((0, 0), test_labels[test_name], font=font(25, True))
        draw.text((x + (cell_w - (bbox[2] - bbox[0])) / 2, top - 65), test_labels[test_name], font=font(25, True), fill="#222222")
    maximum = 0.022
    for row_index, train_name in enumerate(train_order):
        y = top + row_index * cell_h
        draw.text((110, y + 48), train_labels[train_name], font=font(27, True), fill="#222222")
        for column, test_name in enumerate(test_order):
            x = left + column * cell_w
            record = lookup[(train_name, test_name)]
            value = float(record["incapacitation_rate_mean"])
            seed_std = float(record["incapacitation_rate_between_seed_std"])
            intensity = min(1.0, value / maximum)
            fill = (
                int(232 + 18 * intensity),
                int(242 - 105 * intensity),
                int(246 - 120 * intensity),
            )
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline="#ffffff", width=4)
            main_text = f"{value:.2%}"
            sub_text = f"种子SD {seed_std:.2%}"
            main_box = draw.textbbox((0, 0), main_text, font=font(31, True))
            sub_box = draw.textbbox((0, 0), sub_text, font=font(20))
            draw.text((x + (cell_w - (main_box[2] - main_box[0])) / 2, y + 32), main_text, font=font(31, True), fill="#222222")
            draw.text((x + (cell_w - (sub_box[2] - sub_box[0])) / 2, y + 86), sub_text, font=font(20), fill="#444444")
    draw.text((85, 880), "颜色越深表示失能率越高。PPO-Logit的平均与最坏情况均最佳；PPO-LLM训练波动最大。", font=font(25), fill="#333333")
    image.save(OUT / "behavior_cross_heatmap.png", quality=95)


def recurrent_comparison_chart() -> None:
    comparison_path = ROOT / "outputs" / "recurrent_comparison" / "architecture_comparison.csv"
    with comparison_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    behavior_order = ["calibrated_logit", "llm_surrogate_unvalidated", "mixed_logit", "rule"]
    labels = {
        "calibrated_logit": "校准Logit",
        "llm_surrogate_unvalidated": "LLM代理",
        "mixed_logit": "Mixed-logit",
        "rule": "规则人员",
    }
    lookup = {(row["architecture"], row["test_behavior"]): float(row["incapacitation_rate_mean"]) for row in rows}
    image = Image.new("RGB", (1700, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), "4步历史PPO与RecurrentPPO失能率", font=font(42, True), fill="#111111")
    draw.text((80, 100), "相同训练行为、种子、步数和测试场景", font=font(27), fill="#7a2f2f")
    left, top, right, bottom = 190, 220, 1580, 720
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    maximum = 0.020

    def y_of(value: float) -> float:
        return bottom - value / maximum * (bottom - top)

    for tick in (0.0, 0.005, 0.010, 0.015, 0.020):
        y = y_of(tick)
        draw.line((left - 8, y, right, y), fill="#dddddd", width=1)
        draw.text((60, y - 15), f"{tick:.1%}", font=font(22), fill="#444444")
    group_width, bar_width = 315, 105
    for i, behavior in enumerate(behavior_order):
        center = left + 185 + i * group_width
        history_value = lookup[("history_stack_4", behavior)]
        recurrent_value = lookup[("recurrent_lstm", behavior)]
        for offset, value, color in ((-bar_width, history_value, "#587fa3"), (10, recurrent_value, "#a56278")):
            x1 = center + offset
            x2 = x1 + bar_width
            y = y_of(value)
            draw.rectangle((x1, y, x2, bottom), fill=color, outline="#333333")
            draw.text((x1 + 5, y - 35), f"{value:.2%}", font=font(20), fill="#222222")
        label_box = draw.textbbox((0, 0), labels[behavior], font=font(23, True))
        draw.text((center - (label_box[2] - label_box[0]) / 2, bottom + 25), labels[behavior], font=font(23, True), fill="#222222")
    draw.rectangle((230, 800, 270, 825), fill="#587fa3")
    draw.text((285, 795), "4步历史PPO", font=font(23), fill="#222222")
    draw.rectangle((520, 800, 560, 825), fill="#a56278")
    draw.text((575, 795), "RecurrentPPO", font=font(23), fill="#222222")
    draw.text((950, 795), "当前预算下LSTM在四类测试行为上均未改善", font=font(23), fill="#7a2f2f")
    image.save(OUT / "recurrent_comparison.png", quality=95)


def robust_curriculum_chart() -> None:
    with (ROOT / "outputs" / "robust_ensemble" / "curriculum_log.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    series = [
        ("rule", "规则人员", "#587fa3"),
        ("mixed_logit", "Mixed-logit", "#d28a3b"),
        ("calibrated_logit", "校准Logit", "#a56278"),
        ("llm_surrogate_unvalidated", "LLM代理", "#659b76"),
    ]
    phases = sorted({int(row["phase"]) for row in rows})
    image = Image.new("RGB", (1700, 950), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), "自适应训练的行为采样权重", font=font(42, True), fill="#111111")
    draw.text((80, 100), "3个训练种子的阶段均值；每阶段5000步", font=font(27), fill="#7a2f2f")
    left, top, right, bottom = 190, 200, 1580, 730
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = bottom - tick / 0.8 * (bottom - top)
        draw.line((left, y, right, y), fill="#e3e3e3", width=1)
        draw.text((85, y - 15), f"{tick:.0%}", font=font(23), fill="#444444")
    for key, label, color in series:
        points = []
        for phase in phases:
            values = [float(row[f"weight_{key}"]) for row in rows if int(row["phase"]) == phase]
            value = sum(values) / len(values)
            x = left + 110 + (phase - 1) * 230
            y = bottom - value / 0.8 * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
    for phase in phases:
        draw.text((left + 80 + (phase - 1) * 230, bottom + 20), f"阶段{phase}", font=font(22), fill="#222222")
    for i, (_, label, color) in enumerate(series):
        x = 160 + i * 365
        draw.line((x, 835, x + 55, 835), fill=color, width=7)
        draw.text((x + 70, 815), label, font=font(24), fill="#222222")
    image.save(OUT / "robust_curriculum.png", quality=95)


def robust_methods_chart() -> None:
    with (ROOT / "outputs" / "robust_ensemble" / "robust_methods_ranking.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["training_method"]: row for row in rows}
    methods = [
        ("logit_only", "只用校准Logit"),
        ("equal_ensemble", "等权行为集成"),
        ("adaptive_worst_case", "自适应最坏情况"),
    ]
    image = Image.new("RGB", (1700, 930), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), "三种训练方法的平均与最坏失能率", font=font(42, True), fill="#111111")
    draw.text((80, 100), "合成行为；3个训练种子；每类行为每种子200个场景", font=font(27), fill="#7a2f2f")
    left, top, right, bottom = 190, 215, 1580, 710
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    maximum = 0.016
    for tick in (0.0, 0.004, 0.008, 0.012, 0.016):
        y = bottom - tick / maximum * (bottom - top)
        draw.line((left, y, right, y), fill="#e3e3e3", width=1)
        draw.text((65, y - 15), f"{tick:.1%}", font=font(22), fill="#444444")
    for i, (key, label) in enumerate(methods):
        row = by_method[key]
        center = left + 250 + i * 450
        for offset, column, color in (
            (-95, "mean_incapacitation_rate", "#587fa3"),
            (25, "worst_behavior_incapacitation_rate", "#a56278"),
        ):
            value = float(row[column])
            x1 = center + offset
            y = bottom - value / maximum * (bottom - top)
            draw.rectangle((x1, y, x1 + 110, bottom), fill=color, outline="#333333")
            draw.text((x1 - 5, y - 37), f"{value:.2%}", font=font(21), fill="#222222")
        label_box = draw.textbbox((0, 0), label, font=font(24, True))
        draw.text((center - (label_box[2] - label_box[0]) / 2, bottom + 25), label, font=font(24, True), fill="#222222")
    draw.rectangle((290, 815, 335, 845), fill="#587fa3")
    draw.text((350, 810), "四类行为平均", font=font(23), fill="#222222")
    draw.rectangle((735, 815, 780, 845), fill="#a56278")
    draw.text((795, 810), "四类行为中的最坏值", font=font(23), fill="#222222")
    image.save(OUT / "robust_methods.png", quality=95)


def equation_images() -> None:
    equations = {
        "equation_state.png": ["s(t) = [D(t), Q(t), H(t), V(t), F(t), A(t−1), Ĉ(t)]"],
        "equation_reward.png": [
            "r(t) = w_e·ΔN_exit(t)/N − w_d·ΔN_incap(t)/N",
            "       − w_f·mean[ΔFED(t)] − w_c·C(t) − w_s·S(t)",
        ],
    }
    for filename, equation_lines in equations.items():
        width = 1700 if "reward" in filename else 1100
        height = 260 if "reward" in filename else 180
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        f = font(43)
        y = 45
        for equation in equation_lines:
            bbox = draw.textbbox((0, 0), equation, font=f)
            x = max(20, (width - (bbox[2] - bbox[0])) // 2)
            draw.text((x, y), equation, font=f, fill="#111111")
            y += 82
        image.save(OUT / filename)


if __name__ == "__main__":
    architecture()
    prototype_chart()
    ppo_chart()
    multiseed_chart()
    behavior_cross_heatmap()
    recurrent_comparison_chart()
    robust_curriculum_chart()
    robust_methods_chart()
    equation_images()
    print(OUT)

