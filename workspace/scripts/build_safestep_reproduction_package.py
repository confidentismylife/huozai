"""Assemble a source-linked report and local baseline index; no new training."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_reproduction_package_20261016"
NUM = ROOT / "outputs/safestep_paper_numerics_20261015"
TRAIN = ROOT / "outputs/safestep_author_training_20261015_v2"
FREQ = ROOT / "outputs/safestep_frequency_ablation_20261015"
LOSS = ROOT / "outputs/safestep_loss_ablation_20261015"
CHECK = ROOT / "outputs/safestep_checkpoint_diagnosis_20261016"
TIMEOUT = ROOT / "outputs/safestep_paper_stepwise_timeout_audit_20261015"
REPORT = ROOT / "reports/SafeStep论文复现正式整合报告_2026-10-16.md"
BRIEF = ROOT / "reports/SafeStep复现组会汇报_五分钟版_2026-10-16.md"


def read(p):
    return json.loads(p.read_text("utf-8-sig"))


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(p, value):
    assert not p.exists(), p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_save(p, value):
    assert not p.exists(), p
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(value[0]))
        writer.writeheader()
        writer.writerows(value)


def link(p, label):
    return f"[{label}]({p.as_posix()})"


def main():
    assert read(OUT / "baseline_replay/verification.json")["status"] == "PASS"
    manifests = [d / "completion_manifest.json" for d in (NUM, TRAIN, FREQ, LOSS, CHECK, TIMEOUT)]
    historic = {}
    for p in manifests:
        m = read(p)
        for k, v in m["artifact_sha256"].items():
            key = Path(k).as_posix()
            if key in historic:
                assert historic[key] == v, key
            historic[key] = v
    revision = CHECK / "presentation_revision_v2.json"
    for k, v in read(revision)["artifact_sha256"].items():
        historic[Path(k).as_posix()] = v
    evidence = {
        "E1": ("酒店四路线数值", "VERIFIED_FIXED_CASE", NUM / "numeric_run/results.json"),
        "E2": ("四路线可读对照", "VERIFIED_FIXED_CASE", NUM / "numeric_run/fixed_route_comparison.csv"),
        "E3": ("固定权重独立核查", "HISTORICAL_PASS", NUM / "numeric_run/verification.json"),
        "E4": ("六次公开程序训练", "COMPLETED_ALL_SIX", TRAIN / "six_run_comparison.csv"),
        "E5": ("六次训练独立核查", "HISTORICAL_PASS", TRAIN / "verification.json"),
        "E6": ("正文重建中断审计", "INTERIM_ONLY", TIMEOUT / "audit_summary.json"),
        "E7": ("更新频率有限预算对照", "DIAGNOSTIC_ONLY", FREQ / "frequency_comparison.csv"),
        "E8": ("九模型损失对照", "DIAGNOSTIC_THREE_REUSED", LOSS / "nine_model_comparison.csv"),
        "E9": ("27个固定阶段检查点", "EXPLORATORY_REEVALUATION", CHECK / "checkpoint_metrics.csv"),
        "E10": ("检查点独立核查", "PASS", CHECK / "verification.json"),
        "E11": ("九模型阶段变化", "EXPLORATORY_REEVALUATION", CHECK / "model_progressions.csv"),
        "E12": ("有限诊断停止决定", "CLOSED", CHECK / "decision.json"),
        "E13": ("本次可复用入口重放", "REPLAY_PASS", OUT / "baseline_replay/verification.json"),
    }
    sources = [p for _, _, p in evidence.values()] + manifests + [revision]
    sources += [ROOT / "schemas/safestep_author_numerics_20261015.json", ROOT / "src/fire_evac/safestep_author_numerics.py",
                ROOT / "schemas/safestep_author_training_20261015_v2.json"]
    sources += [NUM / "author_repository" / n for n in ("main.py", "maze.py", "maze_dql.pth", "coordinates2areanumber.py",
                "FED rate - room24.xlsx", "fire hazard zones FED - room24.xlsx", "DQL_NetworkModel.py")
                if (NUM / "author_repository" / n).exists()]
    for d in sorted((TRAIN / "runs").iterdir()):
        if d.is_dir():
            sources.extend([d / "final_model.pth", d / "training_complete.json", d / "summary.json"])
    sources = sorted(set(sources))
    old_checked = 0
    for p in sources:
        assert p.exists(), p
        key = p.relative_to(ROOT).as_posix()
        if key in historic:
            assert digest(p) == historic[key], key
            old_checked += 1
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    save(OUT / "source_manifest.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
        verification_scope="Selected referenced records, original inputs and six final models; not a repeat audit of all old optimizer logs",
        matched_historical_manifest_entries=old_checked, input_sha256=source_hashes))
    csv_save(OUT / "evidence_index.csv", [dict(id=e, claim=c, status=s, source=p.relative_to(ROOT).as_posix(), sha256=digest(p))
                                          for e, (c, s, p) in evidence.items()])
    numeric = read(NUM / "numeric_run/results.json")
    training = rows(TRAIN / "six_run_comparison.csv")
    frequency = rows(FREQ / "frequency_comparison.csv")
    losses = rows(LOSS / "nine_model_comparison.csv")
    stages = rows(CHECK / "model_progressions.csv")
    assert len(training) == 6 and len(frequency) == 6 and len(losses) == 9 and len(stages) == 9
    assert sum(int(r["episodes"]) for r in training) == 72500
    assert sum(int(r["transitions"]) for r in training) == 3343272
    assert all(int(r["all_start_success"]) == 61 for r in training)
    assert [sum(r["both_paper_routes_match"] == "True" for r in training if r["arm"] == arm)
            for arm in ("public_beta", "paper_listed_parameters")] == [1, 2]
    assert sum(int(r["arrivals_200k"]) < int(r["arrivals_50k"]) for r in stages) == 7
    key_rows = []
    route_labels = ["起点1→B（推荐）", "起点1→A（备选）", "起点2→B（推荐）", "起点2→A（备选）"]
    for label, r in zip(route_labels, numeric["fixed_routes"]):
        target = r["target"]
        assert round(r["FED"], 4) == target["FED"]
        key_rows.append(dict(route=label, steps=r["steps"], time_s=r["time_s"], paper_FED=target["FED"],
            replayed_FED=r["FED"], absolute_difference_from_rounded_paper=abs(r["FED"]-target["FED"])))
    csv_save(OUT / "hotel_key_numbers.csv", key_rows)
    csv_save(OUT / "six_training_runs.csv", training)
    csv_save(OUT / "loss_final_models.csv", losses)
    csv_save(OUT / "nine_checkpoint_progressions.csv", stages)
    number_table = ["| 路线 | 步数 | 时间/s | 论文FED | 复算FED | 四位小数吻合 |", "|---|---:|---:|---:|---:|---|"]
    number_table += [f"| {r['route']} | {r['steps']} | {r['time_s']:.1f} | {r['paper_FED']:.4f} | {r['replayed_FED']:.9f} | 是 |" for r in key_rows]
    train_table = ["| 设置 | 种子 | 回合数 | 起点1：步数/FED | 起点2：步数/FED | 全起点到达 | 两路线同时匹配 |",
                   "|---|---:|---:|---|---|---:|---|"]
    for r in training:
        label = "公开Beta" if r["arm"] == "public_beta" else "正文参数＋公开程序"
        train_table.append(f"| {label} | {r['seed']} | {int(r['episodes']):,} | {r['start1_steps']} / {float(r['start1_FED']):.6f} | {r['start2_steps']} / {float(r['start2_FED']):.6f} | 61/61 | {'是' if r['both_paper_routes_match']=='True' else '否'} |")
    stage_table = ["| 损失 | 种子 | 5万步 | 10万步 | 20万步 |", "|---|---:|---:|---:|---:|"]
    arm_labels = {"full_anchor": "A 四输出锚定", "selected": "B 单动作", "selected_quarter": "C 单动作÷4"}
    stage_table += [f"| {arm_labels[r['arm']]} | {r['seed']} | {r['arrivals_50k']}/61 | {r['arrivals_100k']}/61 | {r['arrivals_200k']}/61 |" for r in stages]
    reductions = [(key_rows[i+1]["replayed_FED"]-key_rows[i]["replayed_FED"])/key_rows[i+1]["replayed_FED"]*100 for i in (0, 2)]
    references = "\n".join(f"- **[{e}] {c}**：{link(p, p.name)}。状态：{s}。" for e, (c, s, p) in evidence.items())
    environment = read(OUT / "baseline_replay/environment.json")
    report = f"""## Material Passport

- Origin Skill：academic-research-suite；academic-paper写作流程，依据已有实验记录整合。
- Origin Mode：报告整理、证据核对、已有基线推理重放。
- Origin Date：2026-10-16。
- Verification Status：引用记录与历史清单核对；基线入口重放PASS。对原始训练的结论沿用逐轮已完成的独立核查，本轮没有重训或重新重放全部优化器日志。
- Version Label：safestep_consolidated_reproduction_v1。
- 资料范围：SafeStep酒店案例与本地训练诊断；与早期五分区人群LLM＋PPO研究分别报告。

# SafeStep论文复现正式整合报告

## 摘要

本研究复核SafeStep公开版本在酒店火灾案例中的关键数值与训练可重复性。使用作者归档数据和保存权重，四条指定路线的累计暴露剂量FED全部与论文报告的四位小数吻合，两个论文推荐路径逐格一致，61个非出口起点均可到达出口。采用作者公开训练程序开展两种参数设置、每种三个训练种子的运行，六次均满足全起点到达、无非法动作和无重访；两条论文推荐路线同时匹配的运行数分别为1/3和2/3。

公开程序与论文描述在更新时机、损失以及超时自举等方面存在差异。按正文描述重建的三个运行均在预设时间上限内未完成，最终结果不可评价；后续固定20万控制步的更新频率与损失诊断未产生可靠的修正方案。对九个已有模型的27个固定检查点进行1,647条回放后，七个模型在20万步时的可达起点数低于5万步时，说明这些运行没有稳定保留早期的部分可达能力。该观察不能确定唯一故障机制，也不能推断更长训练必然失败。

当前完成范围为酒店案例的已保存权重数值复核、公开训练程序的有限种子重复，以及明确标注的训练诊断。小建筑案例约62%的改善、上游FDS模拟和浓度到危险率转换仍未复现。本轮将已有结果整理为可追溯基线，结束自动扩大训练的诊断路线。

## 1. 研究目标与来源

目标论文为Fang等的 *Development of a tenability-based path planning model for building fire evacuations using reinforcement learning*，Journal of Building Engineering 118（2026），115132，{link_url('10.1016/j.jobe.2025.115132')}。归档作者仓库为[SafeStep_v1固定版本](https://github.com/hqfang3/SafeStep_v1/tree/573f613420e21e9a6731bc56fa36ea0b3f3a05f6)，提交号`573f613420e21e9a6731bc56fa36ea0b3f3a05f6`。本文依据本地归档，不声称检查了仓库最新状态。

本次复现回答三个问题：作者数据和权重能否恢复酒店关键数值；从随机初始化执行公开程序能否恢复可达策略与论文路线；公开程序与正文描述的差异能否解释复现不稳定。前两项已有实证结果，第三项目前只有有限诊断证据，未识别唯一原因。

## 2. 环境、模型与评价方法

### 2.1 酒店案例

作者地图数组为15×23，包含61个非出口移动格、两个出口和58个不同危险区域编号。危险率表覆盖5至900秒，每5秒一行，共180个有效时刻；疏散从起火后300秒开始。部分区域编号映射到多个相邻格，区域数和网络输入维度不可混用。[E1]

每格移动距离2米，设定速度0.3米/秒，单步时间Δt=2/0.3≈6.6667秒。论文两个起点为(5,1)和(10,9)；出口A为(6,4)，出口B为(14,2)，坐标按作者代码的(col,row)约定。该环境控制一个路径决策对象，没有显式模拟多人排队、服从率、从众、工作人员提醒或真实人群轨迹。

### 2.2 DQN策略

输入为当前位置的61维独热编码，网络为61→61→4，单隐藏层ReLU，四个动作对应左、右、上、下。推理直接选择最大Q值动作；本复现保持作者原有动作执行与终止逻辑。位置输入没有显式包含时间、累计FED及访问历史，奖励却依赖其中部分量，这是后续解释学习难度时必须说明的结构性限制。

### 2.3 暴露积分

为了核对作者数值，保留原程序的取样约定：初始化时加入起点在300秒的危险率乘Δt；第k步走到新格后，采用索引floor(kΔt/5)对应的危险率乘Δt；到达出口时该步剂量为0；超过率表范围时沿用末行。另用独立积分复核路径FED。没有在看到终点数值后改用插值或调整参数。[E1–E3]

FED为本实现使用的累计暴露指标。数值对齐验证计算一致性，不直接验证上游烟气预测或真实人员生理反应。作者环境在高剂量后仍可继续至出口或第301步，超时失败路线的高FED不能作为真实人员仍可行走的证据。

### 2.4 分开的验收条件

第一项是指定路线FED的四位小数一致，并核对步数和论文保留一位小数的时间。第二项是已保存策略从论文起点逐格恢复推荐路线。第三项是从全部61个非出口格到达、无非法动作、无重访。达到出口、低剂量、匹配论文路线及全局最优是不同要求；本文未对所有可能路线给出全局最优证明。

所有种子与失败均保留。61个起点来自同一地图，不能当作61栋独立建筑；阶段检查点是重复评价，不能当作独立训练。未对三个种子作显著性或泛化概率推断。

## 3. 原作者权重的关键数值结果

{chr(10).join(number_table)}

四个FED在论文精度下均吻合。作者保存权重从两处起点均选择出口B的推荐路线，分别10步和19步；全部61个起点均可到达，轨迹无非法动作或重访。[E1–E3]

在这两组指定路线内部，推荐路线相对备选路线的FED分别低{reductions[0]:.2f}%和{reductions[1]:.2f}%。这两个百分比仅描述酒店中的指定路线配对，不等于论文摘要的小建筑约62%改善，也不代表所有疏散策略的平均收益。推荐路线在这里用更长行走时间换取更低计算剂量，说明单独最短时间不足以描述这一案例的评价目标。

## 4. 从头训练：公开程序能做到什么

### 4.1 两种设置

两组均使用作者公开训练函数、酒店数据、61→61→4网络、Adam、折扣0.95、回放容量1,000和批量64。公开Beta设置的学习率为0.01，出口奖励参数为0，最多20,000回合，并按公开规则检查提前停止。另一组把正文明列的学习率0.008、出口奖励参数10和固定12,000回合应用于公开程序。出口奖励参数经过原步行权重0.1缩放。

第二种设置同时改变多项参数，不能作为单因素实验，也不能称为正文训练伪代码的完整复现。每组均使用141、142、143三个种子；保存全部预定最终模型。[E4–E5]

### 4.2 六次完整结果

{chr(10).join(train_table)}

六次共完成72,500回合、3,343,272个训练转移和72,500次优化更新。公开Beta组的论文双路线匹配为1/3，正文参数结合公开程序组为2/3；两组全起点到达检查均3/3。不能将两种不同设置合并成总体成功概率，也不能把全起点到达称为稳定恢复论文低剂量路线。

例如，正文参数组种子142在起点2选择11步、FED0.322977的备选路线，而非19步、FED0.230021的论文推荐路线。因此，“能够到达”这一条件仍会掩盖具体路线质量差异。全部模型保留供后续基线使用，不按本次成绩挑选一个模型代表原方法。[E4]

### 4.3 原版与正文的差异

| 项目 | 公开程序的实际行为 | 正文描述重建时采用的行为 |
|---|---|---|
| 更新频率 | 每回合结束一次优化 | 预热后每个控制步优化 |
| 损失 | 四输出MSE，被选动作替换TD目标，其他输出以目标网络锚定 | 被选动作TD误差 |
| 超时终止 | WIN不自举，LOSE超时仍自举 | WIN与LOSE均不自举 |
| 目标网络同步 | 回合末检查步数间隔 | 每50个全局控制步同步 |

公开版本与正文之间有差异，但现有材料不能还原作者当年唯一的最终训练历史。本文使用明确标注的运行版本，避免把公开Beta、正文参数适配和正文流程重建混为同一算法实例。

## 5. 修改流程后的诊断结果

### 5.1 正文流程重建：三个运行均中断

三个计划12,000回合的运行都触发预先冻结的一小时上限，0/3完成，没有正式最终模型。按完整日志覆盖条件预先选定的中途检查点分别为9,500、8,000、9,000回合，可达起点数为2/61、11/61、13/61。这些是中途诊断，不能当作完整训练终点成绩。[E6]

该版本同时更改四个机制，无法从失败唯一归因。后续曾观察到大量撞墙自循环及两格往返，以及相同位置动作在不同累计暴露下具有不同奖励。它们是问题线索，不足以证明某个机制是全部失败的必要或充分原因。

### 5.2 更新频率：固定交互预算下没有可靠控制器

每种调度三个种子，均200,000控制步。回合末更新组可达数为0、1、1；每步更新组为4、10、31。每步组有199,936次更新，回合末组只有935、934、932次，更新数约相差214倍。[E7]

这检验固定交互预算下调度的总体效果，优化次数和实际探索进程并不相同。它既不能证明更新更频繁就能稳定解决问题，也不能把最近较少回合的训练与较早完整公开程序运行直接当成同阶段比较。

### 5.3 损失拆分：没有一致修正方向

令D为选中动作TD平方误差的批均值，U为三个未选动作锚定误差平方和的批均值。A=(D+U)/4，B=D，C=D/4。C−A在相同被选项系数下移除锚定；B−C检验整体目标缩放。三组固定每步更新和20万控制步。

A复用上一轮全部三个每步模型，事先已知分数；B和C新增六次运行。九个比较对象来自本轮六个新增运行及三个复用模型，不能把它们说成九次新训练。C−A的到达数差为+12、+3、−11；B−C为+1、−6、−2，均未形成跨种子的同向修正。[E8]

Adam的矩归一化使损失整体乘四不能解释成参数更新或学习率乘四。旧缓存单步探针首版的跨分支优化器状态未隔离，其结果已作废保留；后续有效v2深拷贝分支状态。正式训练每个运行有独立优化器，这个探针问题不影响正式九模型结果。本文不使用无效首版比值支持结论。

### 5.4 阶段缓存诊断：已有部分能力未稳定保留

{chr(10).join(stage_table)}

![九模型阶段变化]({(CHECK/'checkpoint_arrivals_v2.png').as_posix()})

九个模型均评价三个固定时点，合计27个检查点、1,647条完整回放。七个模型20万步时的可达数低于5万步时；两个在两个相邻时段均下降，另外七个发生方向反转。18段相邻变化中11段下降、7段上升。所有27个时点都未达到61/61。[E9–E11]

这支持“部分早期能力没有稳定保留”，不支持“模型完全学会后又完全遗忘”，也不能排除更长训练恢复。三个离散检查点不能确定循环首次形成时间，更不能单独识别学习率、奖励、观测或终止机制的因果贡献。本轮已按有限范围结束，停止自动增加训练和损失变体。[E12]

## 6. 讨论：现在能说明什么

### 6.1 已经有可用的计算基线

归档酒店数据、作者权重和公开程序足以建立可追溯的路径规划基线。本次新增的推理重放入口在当前项目路径再次得到四个数值一致、两条推荐路径一致和61/61可达，且全部61条轨迹与归档结果逐项一致。这是已知案例的软件重放核对，不增加跨场景证据。[E13]

从头训练结果表明，可达策略能够被公开程序重新学得，但低剂量路线存在种子差异。因此后续使用该基线时应报告全部固定种子，并将到达与剂量分别评价；不得挑出本次路线最吻合的一个种子宣称方法稳定最优。

### 6.2 最近失败没有推翻已完成的复现

最近有限预算实验使用了不同优化调度、随机数流适配或其他流程处理；相应模型常只完成约1,800至2,600回合。它们不能直接代替较早11,000至13,500回合的公开程序终点，也不能用于宣称原论文不可复现。反过来，较早六次全部可达也不能证明最新修改可靠或整个原论文都已恢复。

### 6.3 对LLM＋RL课题的实际作用

SafeStep验证了在作者给定危险率条件下进行路径决策的计算链条。本环境没有提供人群响应、引导接收和改道轨迹，因此不能据此证明LLM能够模拟真人，也不能把这里的DQN路径分数直接与早期五分区人群PPO控制分数比较。

若以后接回LLM＋RL方向，需要先确定一个行为任务及决策前可用信息，用真人数据检验其相对简单统计模型的价值；然后在共同的人群环境、动作、观测、约束和预算下比较控制策略。现阶段没有足够依据恢复联合PPO训练。

## 7. 未完成部分与停止边界

| 部分 | 当前状态 | 所需条件 |
|---|---|---|
| 酒店四条指定路线FED | 已复核 | 保留归档输入、积分规则和代码 |
| 作者权重两推荐路径 | 已复核 | 同一酒店与起始时刻 |
| 公开程序从头训练 | 六次完成；路线匹配随种子变化 | 使用明确标注的版本与预算 |
| 正文流程完整终点 | 未完成 | 另立预算与实现假设，当前不追加 |
| 小建筑四个剂量及约62%改善 | 未复现 | 对应建筑、危险率序列和评价定义 |
| FDS生成危险场 | 未复现 | 上游模型、边界、火源与输出设置 |
| 浓度到危险率转换 | 未复现 | 原始气体序列与具体转换实现 |
| 真实人群与跨建筑泛化 | 未验证 | 对应数据和独立场景评价 |

小建筑案例记录的目标剂量0.0441/0.1185与0.0695/0.5853目前没有对应原始序列可重算。酒店案例不能代替该建筑的数据。后续工作按信息缺口推进，不再把重复增加酒店训练当成补足原始数据的办法。

## 8. 可复用代码、模型和环境

本次新增入口为{link(ROOT/'scripts/replay_safestep_hotel_baseline.py','replay_safestep_hotel_baseline.py')}，强制指定一个尚不存在的输出目录。它只进行四路线与61起点推理复算，并核对归档轨迹。输入文件、作者权重和历史结果保持原样；已有输出目录会被拒绝。使用方法见{link(OUT/'README.md','基线使用说明')}。

作者权重位于{link(NUM/'author_repository/maze_dql.pth','maze_dql.pth')}；六个从头训练最终模型均列在{link(OUT/'trained_model_index.csv','模型索引')}，不设最佳种子默认值。完整训练实现、配置和日志继续保存在原目录，本轮不启动训练。

本次成功重放环境：Python {environment['python']}，NumPy {environment['packages']['numpy']}，PyTorch {environment['packages']['torch']}，openpyxl {environment['packages']['openpyxl']}，Matplotlib {environment['packages']['matplotlib']}。这是已运行环境快照，不声称是通用安装锁文件或跨设备逐位确定性的保证。

## 9. 核查与研究透明度

本报告表格从归档CSV/JSON生成，并与原完成清单核对。{old_checked}项本轮选定文件匹配历史哈希；完整本轮引用输入另记于source_manifest.json。原来的独立验证已经覆盖训练转移、回放抽样及预先指定的Adam更新，本轮沿用这些记录，不把文档整理称为再次完成数百万步训练核查。

本次基线入口重新推理61个起点并执行四条固定路线，共65条回放。原始固定权重验证及最新检查点核查各有独立计算；新增入口主要确认路径迁移后可用，不宣称又一次全新外部验证。所有统计描述限定在单一酒店和既有种子，未计算显著性、置信区间或复现完成百分比。

本轮训练更新0、LLM/API请求0；48个旧人群最终场景未读取。旧代码、失败输出及无效探针作为历史资料保留；本整合报告给出截至2026-10-16的结论，较早报告中的“尚未找到作者资料”等状态只代表其写作时点。

## 10. 可直接用于论文的结论表述

使用作者公开酒店数据与保存的DQN权重，本研究恢复了两处起点的推荐路线，并在论文报告精度下复算得到四条路线的累计FED。采用公开训练程序的两种设置、每种三个种子进行重复训练，全部运行从61个非出口起点可达；两条推荐路线同时恢复的运行数分别为1/3与2/3，显示路径质量存在训练随机性。公开程序与正文描述存在若干实现差异，本文对修改流程的有限诊断尚未找到可靠修正机制。该结果支持酒店固定案例的部分数值复现，不构成原论文全部案例、上游火灾模拟或真实人群控制效果的完整验证。

## 11. 文献与本地证据

目标文献：[Fang等，DOI 10.1016/j.jobe.2025.115132](https://doi.org/10.1016/j.jobe.2025.115132)。代码来源：[SafeStep_v1固定提交](https://github.com/hqfang3/SafeStep_v1/tree/573f613420e21e9a6731bc56fa36ea0b3f3a05f6)。

{references}
"""
    save(REPORT, report)
    brief = f"""# SafeStep论文复现：五分钟组会汇报

日期：2026-10-16。依据：{link(REPORT,'正式整合报告')}。本材料汇报已有证据，不是新方法优越性证明。

## 0:00—0:45　这篇论文做什么，我们复现什么

SafeStep研究的是火灾中如何选择暴露剂量更低的路径。它使用DQN，根据当前位置选择上下左右四个方向。我们使用作者公开的酒店地图、危险率数据和网络权重，先检查论文数值能否算出来，再从随机权重重新训练。这里是路径规划模型，没有模拟多人排队或LLM人员行为。

## 0:45—1:45　最确定的成果：关键数值已经对上

酒店案例有两个起点、两个出口。论文四条指定路线的FED是0.1471、0.1671、0.2300和0.3230，我们复算后全部在四位小数上吻合。作者保存模型选择的两条推荐路线分别是10步、约66.7秒，以及19步、约126.7秒；61个非出口起点都能走到出口。

推荐路线比备选路线走得更久，计算暴露剂量却更低。这说明这个案例中只看距离或时间不够。酒店内两组路线的剂量差约为{reductions[0]:.1f}%和{reductions[1]:.1f}%，不能拿来替代论文另一个建筑的约62%结果。

## 1:45—2:45　从头训练：能学会到达，但路线不完全稳定

我们做了两种设置，每种三个种子，共六次完整训练。六次都从61个起点到达，且没有非法动作和重访；但两条论文推荐路线同时恢复，第一组只有1/3次，第二组2/3次。

所以现在能说“公开程序能够重新训练出可达策略”，也能说“存在成功恢复论文路线的运行”；还不能说“每个种子都稳定得到低剂量路线”。全部模型都保留，没有挑最好的一个来代表结果。

## 2:45—3:45　为什么后来一直诊断

公开程序与论文文字在更新频率、损失和超时处理上有差异。我们重建正文流程后，三个运行都触发原定时间上限，最终训练没有完成。随后有限预算的拆分实验，也没有找到稳定修正。

最近读取九个模型的5万、10万、20万步检查点，完成27个检查点、1,647条回放。七个模型最后的可达起点数低于早期，例如40降到20再降到4。说明早期的部分能力没有稳定保留，但三个时点还不足以确定原因，也不能证明更长训练必然失败。

## 3:45—4:30　现在不能宣称什么

酒店案例的数值复核成立，整篇论文还没有完整复现。小建筑案例约62%改善缺少对应原始危险率；FDS模拟、气体浓度转换和真实人群行为没有被这套结果验证。这里的DQN也不能直接和我们早期五分区PPO的人群控制成绩作比较。

## 4:30—5:00　现在的决定

结束自动增加训练的诊断路线，把作者数据、权重、六次训练和失败边界整理成基线。已有一个可重复运行的推理入口，本轮在迁移后的项目目录再次核对通过。

对老师提出的“LLM模拟人、RL引导”方向，下一阶段应先定义一个能用真人数据检验的行为问题，比较简单统计模型与LLM。证据足够后，再考虑共同环境中的控制实验。当前这份复现材料可以作为路径规划基线与方法边界说明。

## 汇报时可展示的图

- {link(CHECK/'checkpoint_arrivals_v2.png','九个模型三个阶段的到达数')}
- {link(CHECK/'checkpoint_failure_types_v2.png','到达、原地循环与多格循环分类')}

## 老师可能追问

- **到底有没有复现成功？** 酒店固定权重关键数值成功，公开程序有从头训练成功运行；整篇论文全部案例尚未完成。
- **为什么不能直接继续训练？** 已经观察到阶段性下降，但原因不唯一；继续加预算不是已验证的修复方法。
- **下一步的价值是什么？** 先把已有计算基线固定下来，后续用它检验共同条件下的新方法，而不是不断重复同一项失败诊断。
"""
    save(BRIEF, brief)
    model_index = []
    for r in training:
        d = TRAIN / "runs" / f"{r['arm']}_seed{r['seed']}"
        p = d / "final_model.pth"
        model_index.append(dict(setting=r["arm"], seed=r["seed"], episodes=r["episodes"],
            all_start_arrivals=r["all_start_success"], both_paper_routes_match=r["both_paper_routes_match"],
            model=p.relative_to(ROOT).as_posix(), sha256=digest(p), selected_as_default=False))
    csv_save(OUT / "trained_model_index.csv", model_index)
    readme = f"""# SafeStep酒店基线与复现材料入口

本目录是本地证据索引与已知案例重放结果。作者源数据和六个训练模型继续保存在原归档目录；本目录不是独立可迁移的软件发行包。当前项目根目录为`{ROOT.as_posix()}`。

## 阅读材料

- {link(REPORT,'正式整合报告：方法、结果、讨论、限制')}
- {link(BRIEF,'五分钟组会讲稿')}
- {link(OUT/'hotel_key_numbers.csv','四个酒店关键数值')}
- {link(OUT/'six_training_runs.csv','六个从头训练结果')}
- {link(OUT/'trained_model_index.csv','全部六个最终模型索引')}
- {link(OUT/'evidence_index.csv','结论与来源索引')}
- {link(OUT/'baseline_replay/verification.json','本次入口重放核对')}

## 再运行一次酒店固定权重基线

在PowerShell中执行。输出目录名每次换一个；已有目录会被拒绝，不覆盖记录。

```powershell
Set-Location -LiteralPath '{ROOT.as_posix()}'
& 'D:/miniconda3/python.exe' scripts/replay_safestep_hotel_baseline.py --output outputs/safestep_hotel_replay_manual_001
```

预期`verification.json`显示`status=PASS`、`fixed_route_FED_matches=4`、`stored_weight_paper_path_matches=2`、`all_start_arrivals=61`、`invalid_steps=0`、`revisits=0`。程序同时比较全部61条路径与归档记录。重复推理核对已知案例，不训练模型，也不增加泛化证据。

该入口默认从项目中读取{link(NUM/'author_repository','作者固定版本目录')}。配置和依赖快照见{link(OUT/'baseline_replay/environment.json','成功重放环境')}与{link(ROOT/'schemas/safestep_author_numerics_20261015.json','原数值复现配置')}。本轮未安装、更新或下载依赖。

## 历史训练入口

{link(ROOT/'scripts/run_safestep_author_training_v2.py','原公开程序训练启动器')}用于已经完成的冻结实验，原输出目录现已存在。它不是本包默认执行入口；若未来需要新训练，应另立版本、输出目录、预算和验收条件。最新检查点诊断的决定是结束自动扩展训练。

## 证据范围

酒店指定路线数值与作者权重推理已核对；两组训练均3/3全起点到达，论文双路线匹配分别1/3、2/3。小建筑约62%结果、FDS与浓度到危险率的上游没有在本基线中复现。所有原始失败与无效探针均保留，索引不将其列为正面效果证据。
"""
    save(OUT / "README.md", readme)
    # Check report arithmetic, all local Markdown links and input immutability.
    checked_links = 0
    for doc in (REPORT, BRIEF, OUT / "README.md"):
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text("utf-8")):
            if target.startswith("https://"):
                continue
            assert Path(target).exists(), (doc.name, target)
            checked_links += 1
    for key, expected in source_hashes.items():
        assert digest(ROOT / key) == expected, key
    verification = dict(status="PASS", selected_source_files_unchanged=len(source_hashes),
        historical_hashes_matched=old_checked, local_document_links_checked=checked_links,
        fixed_route_rows=4, from_scratch_runs=6, checkpoint_progression_rows=9,
        arrival_gate_counts_by_setting={"public_beta": "3/3", "paper_listed_parameters": "3/3"},
        paper_route_match_counts_by_setting={"public_beta": "1/3", "paper_listed_parameters": "2/3"},
        baseline_replay=read(OUT / "baseline_replay/verification.json"),
        scope="Document arithmetic and selected provenance checks; historical full training audits reused, not re-executed",
        new_training_updates=0, LLM_requests=0, old_final48_crowd_scenarios_accessed=False)
    save(OUT / "package_verification.json", verification)
    artifacts = [p for p in OUT.rglob("*") if p.is_file()] + [REPORT, BRIEF, Path(__file__), ROOT / "scripts/replay_safestep_hotel_baseline.py"]
    save(OUT / "completion_manifest.json", dict(status="REPRODUCTION_REPORT_AND_BASELINE_INDEX_COMPLETE",
        created_at=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts}))
    print(json.dumps({k:v for k,v in verification.items() if k != "baseline_replay"}, ensure_ascii=True, indent=2))


def link_url(doi):
    return f"[DOI：{doi}](https://doi.org/{doi})"


if __name__ == "__main__":
    main()
