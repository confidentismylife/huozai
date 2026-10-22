"""Create manuscript v5 from v4 using only frozen local results.

The revision adds a post-hoc precision audit and previously unused action
diagnostics. It does not train, simulate, or call an LLM.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "artifacts/llm_ppo_paper_v4_20261021"
OUT = ROOT / "artifacts/llm_ppo_paper_v5_20261021"
OLD_NAME = "固定LLM建议与PPO疏散控制_期刊式改写稿_v4.md"
NAME = "固定LLM建议与PPO疏散控制_期刊式改写稿_v5.md"
EPISODES = ROOT / "outputs/final_pipeline_context_20261020/combined_episode_results.csv"


def replace_once(text: str, old: str, new: str, label: str, changes: list[str]) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    changes.append(label)
    return text.replace(old, new, 1)


def load_main_rows() -> list[dict[str, str]]:
    with EPISODES.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if row["behavior_model"] == "calibrated_logit" and row["noise_level"] == "none"
    ]


def action_diagnostics(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    result = {}
    for method, group in groups.items():
        waits = [float(row["wait_fraction"]) for row in group]
        matches = [float(row["requested_actual_joint_match_rate"]) for row in group]
        result[method] = {
            "episodes": len(group),
            "mean_wait_fraction": sum(waits) / len(waits),
            "episodes_with_wait": sum(value > 0 for value in waits),
            "total_control_steps": sum(int(row["steps"]) for row in group),
            "mean_requested_actual_joint_match_rate": sum(matches) / len(matches),
        }
    return result


def precision_audit(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    candidate = "trained_llm_advice_ppo"
    baseline = "trained_pure_ppo"
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        cells[(row["scenario_id"], row["training_seed"])][row["method"]] = float(row["incapacitation_rate"])
    scenarios = sorted({key[0] for key in cells})
    seeds = sorted({key[1] for key in cells})
    grid = [[cells[(scene, seed)][candidate] - cells[(scene, seed)][baseline] for seed in seeds] for scene in scenarios]
    estimate = sum(sum(row) for row in grid) / (len(scenarios) * len(seeds))
    rng = random.Random(20261021)
    draws = []
    for _ in range(10_000):
        sampled_scenes = rng.choices(range(len(scenarios)), k=len(scenarios))
        sampled_seeds = rng.choices(range(len(seeds)), k=len(seeds))
        draws.append(
            sum(grid[i][j] for i in sampled_scenes for j in sampled_seeds)
            / (len(sampled_scenes) * len(sampled_seeds))
        )
    draws.sort()
    low = draws[250]
    high = draws[9750]
    se = (high - low) / (2 * 1.96)
    mde = (1.96 + 0.8416212335729143) * se
    z = abs(estimate) / se
    normal = NormalDist()
    effect_power = normal.cdf(z - 1.96) + normal.cdf(-z - 1.96)
    flat = [value for row in grid for value in row]
    iid_var_of_mean = (sum(value * value for value in flat) / len(flat) - estimate * estimate) / len(flat)
    design_effect = (se / math.sqrt(iid_var_of_mean)) ** 2
    return {
        "status": "post_hoc_planning_sensitivity_not_confirmatory_evidence",
        "scenarios": len(scenarios),
        "training_seeds": len(seeds),
        "episodes": len(scenarios) * len(seeds),
        "bootstrap_draws": len(draws),
        "bootstrap_seed": 20261021,
        "mean_difference_fraction": estimate,
        "bootstrap_ci95_low_fraction": low,
        "bootstrap_ci95_high_fraction": high,
        "bootstrap_derived_se_percentage_points": se * 100,
        "normal_approx_mde_80pct_percentage_points": mde * 100,
        "normal_approx_power_at_observed_effect": effect_power,
        "iid_equivalent_effective_n_heuristic": round(len(flat) / design_effect),
        "caution": "MDE and effective n are planning heuristics; they do not change the actual 48-scene crossed design or exclude effects inside the reported CI.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = (PREV / OLD_NAME).read_text(encoding="utf-8")
    changes: list[str] = []

    source = replace_once(
        source,
        "## 基于合成冲突场景的对照与机制分析",
        "## 基于合成冲突场景的对照 精度与机制分析",
        "副标题加入精度审计",
        changes,
    )

    old_abstract = source.split("**摘要**\n\n", 1)[1].split("\n\n", 1)[0]
    new_abstract = (
        "固定LLM出口建议能否为动态疏散控制带来增量收益，取决于建议是否提供新信息、是否作用于关键控制变量以及评价精度。"
        "本文在五分区、两出口、240人的合成环境中，将场景级Qwen3 4B缓存建议输入近端策略优化控制器。主要条件含48个场景和5个训练种子；"
        "LLM建议PPO相对独立纯PPO的失能率差为−0.6632个百分点［−2.3160，0.4132］，人均合成累计暴露差为0.000669［−0.012287，0.013746］，尚未建立增量优势或等效。"
        "事后精度审计显示，80%功效对应的最小可检出效应约为1.9个百分点。LLM目标由PPO已接收的预测火源决定，48份置信度均为0.99；"
        "LLM建议策略在240回合、7460个控制步中未选择等待，较长回合与更多低档位请求相一致。容量混合虽缩短回合时间，但未建立安全非劣。"
        "因此，本研究识别的是固定建议表示及其训练响应，不能据此判断LLM在开放信息、在线规划或容量调度中的一般价值。"
    )
    source = replace_once(source, old_abstract, new_abstract, "摘要加入精度和等待动作证据", changes)

    source = replace_once(
        source,
        "本文的工作包括三个方面。首先，通过共享权重输入消融与独立训练管线比较，区分建议通道依赖和系统级控制收益。其次，结合动作记录、撤离阈值达标情况及方向与容量混合，分析时间代价与控制行为之间的关系。最后，使用按场景分组的收益预测及特征敏感性分析，考察初始信息能否识别LLM建议的受益条件。所有分析保留原有证据等级，完整管线及机制结果均作为解封后的探索性证据报告。",
        "本文的工作包括四个方面。首先，通过共享权重输入消融与独立训练管线比较，区分建议通道依赖和系统级控制收益。其次，结合动作记录、撤离阈值达标情况及方向与容量混合，分析时间代价与控制行为之间的关系。再次，对主要比较进行事后精度审计，说明当前设计能够辨别的效应尺度。最后，使用按场景分组的收益预测及特征敏感性分析，考察初始信息能否识别LLM建议的受益条件。所有分析保留原有证据等级；精度审计、完整管线及机制结果均作为解封后的探索性证据报告。",
        "引言增加事后精度审计定位",
        changes,
    )

    source = replace_once(
        source,
        "配对效应在相同场景、训练种子、人群与噪声条件下计算，方向定义为LLM建议PPO减去基线；失能率差以百分点报告。主要条件每方法有240回合，但独立场景仅48个。原盲测及完整管线将主要条件的配对差构成48场景×5种子矩阵，每次分别有放回抽取场景和种子，再对交叉子矩阵求均值，以10,000次重采样的2.5%与97.5%分位形成95%区间；机制分析使用5,000次。原盲测主要检验按场景平均种子后进行符号翻转，并对两项主要终点的两个主要比较实施Holm校正。完整管线及后续分析不改写为预注册确认性结果。",
        "配对效应在相同场景、训练种子、人群与噪声条件下计算，方向定义为LLM建议PPO减去基线；失能率差以百分点报告。主要条件每方法有240回合，但独立场景仅48个。原盲测及完整管线将主要条件的配对差构成48场景×5种子矩阵，每次分别有放回抽取场景和种子，再对交叉子矩阵求均值，以10,000次重采样的2.5%与97.5%分位形成95%区间；机制分析使用5,000次。原盲测主要检验按场景平均种子后进行符号翻转，并对两项主要终点的两个主要比较实施Holm校正。完整管线及后续分析不改写为预注册确认性结果。结果解封后另作规划性精度审计：由固定种子分层重采样区间宽度近似标准误，并在双侧α=0.05的正态近似下计算80%功效的最小可检出效应。观测效应对应功效和IID等价有效样本量仅作为后续研究的启发式量，不用于证明等效、排除区间内效应或改变实际样本结构。",
        "统计方法限定事后精度审计的用途",
        changes,
    )

    source = replace_once(
        source,
        "图2给出完整管线的差值及95%区间。相对独立纯PPO，LLM建议PPO失能率差为−0.6632个百分点［−2.3160，0.4132］，FED差为0.000669［−0.012287，0.013746］，时间差为22.50 s［−0.65，40.52］。失能率区间同时包含改善与变差的可能性，当前证据尚未建立增量优势，也不足以认定等效。",
        "图2给出完整管线的差值及95%区间。相对独立纯PPO，LLM建议PPO失能率差为−0.6632个百分点［−2.3160，0.4132］，FED差为0.000669［−0.012287，0.013746］，时间差为22.50 s［−0.65，40.52］。失能率区间同时包含改善与变差的可能性，当前证据尚未建立增量优势，也不足以认定等效。事后精度审计得到约0.69个百分点的重采样标准误；按正态近似，80%功效对应的最小可检出效应约为1.9个百分点，观测到0.6632个百分点差异时的对应功效约为16%。IID等价有效样本量约为72，但它只是由标准误比值换算的规划指标。现有区间仍包含超过2个百分点的潜在改善，因而不能把1.9个百分点误写为本研究已经排除的效应界限。",
        "主要结果增加精度与正确解释",
        changes,
    )

    source = replace_once(
        source,
        "48个LLM目标中有6个违反提示指定映射，最终运行器保留原始解析目标，没有调用目标纠正函数。违反指令与造成疏散危害具有不同含义，不能直接对应。LLM与规则在42/48个场景中选择相同方向，但其中41个场景置信度不同。LLM置信度全部为0.99，规则均值约0.832；这些分数未经过概率校准，不能视为建议正确率。\n\n方向一致并不意味着完整输入相同。除置信度外，建议存在标志和随观测变化的矛盾标志也会进入策略。因此，下文比较识别的是建议通道整体及其训练响应，现有设计没有单独分离语言推理、方向编码或置信度的作用。",
        "48个LLM目标中有6个违反提示指定映射，最终运行器保留原始解析目标，没有调用目标纠正函数。违反指令与造成疏散危害具有不同含义，不能直接对应。LLM与规则在42/48个场景中选择相同方向，但其中41个场景置信度不同。LLM置信度全部为0.99，规则均值约0.832；这些分数未经过概率校准，不能视为建议正确率。\n\n提示要求的目标是预测火源的对侧出口，而预测火源已经属于三种方法共享的9维原始预测字段；置信度在LLM缓存中也没有场景间变异。建议存在标志以及由当前危险与建议目标计算的矛盾标志能够改变输入表示，但没有引入独立外部证据。六个映射错误又使实际通道并非严格的确定性复制。因而，下文比较识别的是带有模型错误的结构化建议表示及其训练响应，不能单独归因于语言推理、方向编码、置信度或存在标志。",
        "信息分析区分新证据与输入表示",
        changes,
    )

    source = replace_once(
        source,
        "主要条件下，LLM建议PPO的平均非等待请求放行档位相对独立纯PPO的差为−0.1079［−0.1784，−0.0167］，低档位动作比例差为0.1374［0.0386，0.2188］。低档位定义为放行比例不超过0.5，分母为非等待控制步。它描述策略请求容量，不等同于实际通过人数。\n\n活动人员房间清空时间差为25.52 s［−1.15，47.27］，其场景级差值与回合时间差的Spearman相关系数为0.896。房间清空包含失能人员移出，不能解释为全员成功放行。较低放行档位与较长回合的关联支持容量可能参与时间代价形成，但相关性和时间先后关系没有识别唯一因果机制。",
        "主要条件下，LLM建议PPO的平均非等待请求放行档位相对独立纯PPO的差为−0.1079［−0.1784，−0.0167］，低档位动作比例差为0.1374［0.0386，0.2188］。低档位定义为放行比例不超过0.5，分母为非等待控制步。它描述策略请求容量，不等同于实际通过人数。\n\n逐回合聚合记录还显示，LLM建议PPO的平均等待比例为0，240个回合和7460个控制步中均未出现等待；独立纯PPO为0.0231，19/240个回合出现等待；规则建议PPO为0.0720，55/240个回合出现等待。请求动作与实际执行联合动作的一致率依次为0.9237、0.9335和0.9401；不一致包含方向最短保持两步造成的阻断，但现有聚合记录没有分解所有来源。这些量为描述性诊断，未另作多重比较校正。\n\n活动人员房间清空时间差为25.52 s［−1.15，47.27］，其场景级差值与回合时间差的Spearman相关系数为0.896。房间清空包含失能人员移出，不能解释为全员成功放行。LLM建议策略没有通过等待动作造成较长回合；结合较低放行档位，其时间代价更符合持续低档放行的解释。但这些指标来自完整策略轨迹，相关性和时间先后关系仍未识别唯一因果机制。",
        "补报等待动作与请求执行一致率",
        changes,
    )

    source = replace_once(
        source,
        "共享控制器消融揭示较大的建议输入效应，而独立完整管线中的差异明显缩小。这说明评价基线如何训练，会直接影响对LLM作用的解释。输入分布变化是零建议退化的可能原因，但当前实验未单独分离这一机制；建议方向先验、置信度分布以及训练时的通道依赖也可能共同作用。",
        "共享控制器消融揭示较大的建议输入效应，而独立完整管线中的差异明显缩小。这说明评价基线如何训练，会直接影响对LLM作用的解释。零建议改变了训练时输入分布，建议目标主要复述已有预测，置信度又恒定为0.99，因此策略可能利用了通道存在、固定尺度或方向先验，而不一定利用了语言推理。现有实验没有分别置换目标、固定置信度或单独消融存在标志，尚不能确认策略究竟学到了哪一种信号。",
        "讨论中将通道依赖改为待检验机制",
        changes,
    )

    source = replace_once(
        source,
        "当前提示规定把预测火源映射到对侧出口，并禁止容量和封闭信息改变目标。该映射本身可以直接计算；原始预测也已提供给纯PPO。因此，本设计检验的是对既有信息增加结构化建议通道的效果，不能将其理解为LLM发现了控制器不可见的新信息。LLM独立权衡等待、分流和改道的能力没有在这里得到检验。六份违反映射的响应反映指令遵循问题，其安全后果仍取决于环境和后续控制。",
        "当前提示规定把预测火源映射到对侧出口，并禁止容量和封闭信息改变目标。该映射本身可以直接计算；原始预测也已提供给纯PPO。方向只占九个动作中的一个维度，四档放行量与等待由PPO决定。建议又在回合开始前冻结，运行中的封闭、拥堵和危险反转均不能触发LLM重规划。因此，本设计评价的是LLM处于固定方向提示位置时的作用，不能扩展为LLM在动态容量、节奏或在线改道中的一般能力。六份违反映射的响应反映指令遵循问题，其安全后果仍取决于环境和后续控制。",
        "讨论明确方向 容量和时序三层限制",
        changes,
    )

    source = replace_once(
        source,
        "在这一证据范围内，继续研究应先明确额外信息来源、工程上有意义的收益标签和独立验证方案，再决定是否增加训练规模。若转向真实建筑应用，还需补充物理危险输入、连续人群响应和部署条件，不能把本合成实验直接视为现实安全证明。",
        "在这一证据范围内，下一项实验应先改变可辨识问题，而不是扩大当前训练。可预先固定三条同信息路径：将新增信息数值化后直接输入PPO、由LLM转化为分段容量计划、由人工规则转化为同一计划格式；三者使用相同场景信息和预算。LLM只在封闭生效、危险差反转或交汇区拥堵越线时触发，每回合设置调用上限，并先用独立提示集验证解析成功率。未来确认性研究还应在观察结果前确定工程相关界值；2个百分点可作为样本量规划的候选值，但当前研究没有预先设定或验证这一界值。若转向真实建筑应用，还需补充物理危险输入、连续人群响应和部署条件，不能把本合成实验直接视为现实安全证明。",
        "后续工作改为预设同信息三臂与事件触发方案",
        changes,
    )

    source = replace_once(
        source,
        "本文在合成五分区火灾疏散环境中，评价了固定LLM建议与PPO联合控制的作用。共享控制器消融表明建议通道能够显著改变既有策略行为，但独立完整管线比较尚未建立相对纯PPO或规则建议PPO的稳定增量优势。动作记录揭示较低放行档位与时间代价相关；方向与容量混合缩短了回合时间，却未建立安全非劣。修正未达标时间后，撤离比例和条件时间需要分开解释。去除实验身份标签后，当前固定收益预测器仍未达到既有开发判据。\n\n这些发现支持对LLM建议通道进行明确限定和公平评价。其结论适用于当前模型、强制映射提示、缓存建议及合成分布，不扩展为LLM普遍无效。进一步建立控制增益，需要在新的预设任务中区分方向、置信度和动态响应作用，并使用独立场景验证安全与效率的共同收益。",
        "本文在合成五分区火灾疏散环境中，评价了固定LLM建议与PPO联合控制的作用。建议目标主要由PPO已见的预测火源确定，置信度没有场景间变异，LLM也不参与回合内的容量与等待决策。共享控制器消融表明建议通道能够明显改变既有策略行为，但独立完整管线尚未建立相对纯PPO或规则建议PPO的稳定增量优势；当前精度对约0.66个百分点的效应辨别能力有限。240个LLM建议回合没有等待动作，较长回合与持续低档放行相一致；容量混合能够提速，却未建立安全非劣。\n\n结论适用于当前模型、强制映射提示、缓存建议及合成分布。它说明LLM在当前固定方向提示位置没有显示出可确认的系统增量，不能据此推断LLM在疏散控制中普遍无效。后续实验需要预先赋予LLM可区分的新增信息或动态容量任务，并用同信息基线、独立场景和预定安全效率判据检验其转化价值。",
        "结论聚焦当前任务位置而非LLM一般价值",
        changes,
    )

    rows = load_main_rows()
    actions = action_diagnostics(rows)
    precision = precision_audit(rows)
    (OUT / NAME).write_text(source, encoding="utf-8")

    for folder in ["figures", "data"]:
        target = OUT / folder
        target.mkdir(exist_ok=True)
        for item in (PREV / folder).iterdir():
            if item.is_file():
                target.joinpath(item.name).write_bytes(item.read_bytes())
    for filename in ["方法与来源附录.md", "分层结果溯源.json"]:
        (OUT / filename).write_bytes((PREV / filename).read_bytes())

    audit = {
        "source_csv": str(EPISODES.relative_to(ROOT)),
        "filters": {"behavior_model": "calibrated_logit", "noise_level": "none"},
        "action_diagnostics": actions,
        "precision_audit": precision,
        "new_training": 0,
        "new_simulation_episodes": 0,
        "new_llm_calls": 0,
    }
    (OUT / "新增精度与动作诊断.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    parent_hashes = {
        str((PREV / OLD_NAME).relative_to(ROOT)): hashlib.sha256((PREV / OLD_NAME).read_bytes()).hexdigest(),
        str(EPISODES.relative_to(ROOT)): hashlib.sha256(EPISODES.read_bytes()).hexdigest(),
    }
    revision = {
        "parent_and_sources": parent_hashes,
        "changes": changes,
        "scientific_cautions": [
            "The precision audit is post hoc and planning-oriented.",
            "The 1.9 percentage point MDE is not an exclusion bound.",
            "The effective n value is an IID-equivalent heuristic, not the observed number of independent units.",
            "The 2 percentage point relevance threshold is proposed only for future preregistration and has not been externally validated.",
            "Action diagnostics are descriptive and do not identify a unique causal mechanism.",
        ],
        "new_experiments": 0,
    }
    (OUT / "revision_sources.json").write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8")

    note = (
        "# v5修改说明\n\n"
        "v5在v4基础上补充事后精度审计和既有动作诊断，并重写任务设计局限与后续实验定位。"
        "本轮没有新增训练、疏散仿真或LLM调用。\n\n"
        "## 主要修改\n\n" + "\n".join(f"- {item}" for item in changes) +
        "\n\n## 解释边界\n\n"
        "- 最小可检出效应约1.9个百分点是事后规划量，不是已经排除的效应界限。\n"
        "- 约72的有效样本量是IID等价精度换算，实际设计仍为48场景乘5种子。\n"
        "- 2个百分点只作为未来研究的候选工程界值，当前研究未预先设定，也没有外部验证。\n"
        "- 等待比例和请求执行一致率来自既有逐回合聚合记录，只支持描述性机制解释。\n\n"
        "## 仍需作者确认\n\n"
        "作者、机构、贡献、资助、利益冲突、公开归档、目标期刊及AI使用披露仍需由作者确认。\n"
    )
    (OUT / "修改说明.md").write_text(note, encoding="utf-8")
    (OUT / "README.md").write_text(
        "# LLM加PPO火灾疏散论文 v5\n\n"
        "本目录包含中文期刊式写作稿、Word稿、证据记录和验证结果。v5聚焦任务信息增量、决策位置、时序限制、事后精度与动作执行诊断。\n\n"
        "所有结果来自既有冻结输出；本轮新增训练、仿真和LLM请求均为0。\n",
        encoding="utf-8",
    )
    print(json.dumps({"changes": len(changes), "abstract_chars": len(new_abstract), "output": str(OUT / NAME)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
