"""Integrate retrospective evidence only; do not run or change experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
NAME = 'LLM_RL火灾疏散研究全程整合报告_2026-10-08'
ASSETS = ROOT / 'artifacts/integrated_report_assets_20261008'
PROTECTED = {
    'outputs/local_llm_selection.json': '302c6db2c857811dd41d1ed36248d1d9a71f3e32108cd870ecfe02aa520ff721',
    'outputs/third_round_development_status.json': 'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0',
}
# The exact preserved state digest is checked against the previous independent audit.
PROTECTED['outputs/third_round_development_status.json'] = 'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(label, path):
    p = ROOT / path
    assert p.exists(), p
    return f'[{label}]({p.as_posix()})'


def sections(text):
    parts = re.split(r'(?m)^## (.+)\n', text)
    return parts[0], list(zip(parts[1::2], parts[2::2]))


STAGES = [
    (17, '教师世界聚合与规划预算：增加反事实计算是否值得',
     '这里比较的是离线教师的世界聚合与抽样预算，教师可以复制隐藏人员状态。规划收益不能归为LLM增益，也不能与后来的可观测学生当成同一控制器。',
     [('教师世界聚合与规划预算全新开发验证报告.md', None)]),
    (18, '真实改善的效果量、集中度与研究投入',
     '本章把一次开发收益换算成绝对差、相对差和计算成本。每千模拟人员的失能事件差只是归一化，不能叫现实救人数；前瞻样本量是敏感性估算，并非正式最终样本量。',
     [('继续研究的价值与投入产出量化评估_2026-10-03.md', ['1.', '2.', '3.', '4.', '5.', '6.', '7.'])]),
    (19, '教师标签歧义与人口、随机未来的交叉诊断',
     '先检查相同输入为何给出不同答案，再用人口×规划随机流的交叉设计拆开两个同时变化的因素。标签不一致、主失能最优不一致和实际控制失败分别报告。',
     [('可观测教师标签与覆盖收益审计报告_2026-10-03.md', ['1.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '12.']),
      ('人口与规划随机流交叉诊断报告_2026-10-03.md', ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '10.'])]),
    (20, '集合监督：允许多个好答案，仍未获得可靠覆盖',
     '针对唯一教师标签可能过严的问题，比较集合与唯一标签，并交叉比较有无经验收益余量。两个有门控方法全回退601是行为一致，不是已经证明了收益判断或安全。',
     [('集合监督与覆盖不确定性开发实验报告_2026-10-03.md', ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '9.'])]),
    (21, '统一提名与收益门控：修正决策语义并做交叉拟合',
     '把“选哪个计划”和“是否值得覆盖”的分数统一，解决逻辑冲突；然后在家庭隔离预测中检查其准确性。代码语义修好与疏散效果改善是两个完成条件。',
     [('统一提名与覆盖收益交叉拟合诊断报告_2026-10-03.md', ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.'])]),
    (22, '补足提案价值分类与条件校准证据',
     '固定提案器后补采真正的提案状态，分别检查有益、有害及微小结果。分类器只是额外拒绝条件，不会自动找回被收益预测低估的方案。',
     [('覆盖价值预测与新增条件校准实验报告_2026-10-03.md', ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.'])]),
    (23, '收益目标对齐与十二随机未来复核',
     '分别比较两策略最坏世界损失，与要求每个支持世界都不伤害，并不是相同目标。本章也检查有限未来采样能否解释有益低分、有害高分。',
     [('收益目标对齐与十二未来复核诊断报告_2026-10-03.md', ['结论', '1.', '2.', '3.', '4.', '5.', '6.', '8.'])]),
    (24, '收益标签与轻量模型的四组配对对照',
     '通过固定R/M收益目标与线性/核模型的2×2比较，区分目标和拟合方式。没有有益提案时，有益召回分母为零，应记不可估计，不记作0%或100%。',
     [('收益标签与轻量模型四组对照实验报告_2026-10-03.md', None)]),
    (25, '完整十七候选库审计：好方案是否被提案器漏掉',
     '旧流程只评估601和已经被提名的方案，因此无法判断未提名计划是否有益。完整库审计发现遗漏机会，但离线特权教师的机会不等于在线收益。',
     [('完整候选计划覆盖审计报告_2026-10-03.md', ['结论', '1.', '2.', '3.', '4.', '5.', '6.', '8.'])]),
    (26, '五PPO候选集合与全候选标签接口修复',
     '该固定子集中的五个PPO保留了17候选库的全部已观察收益机会。新接口为每个状态完整输出五候选标签，避免旧提案先筛掉训练正例；没有把旧缓存当作新独立验证。',
     [('五PPO候选集合与全候选标签接口审计报告_2026-10-03.md', ['结论', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'])]),
    (27, '五PPO全新家庭的轻量选择验证',
     '修复数据覆盖后仍使用新训练、校准和留出家庭检验。全部余量版本回退601，说明候选库有机会与模型能识别机会之间仍有缺口。',
     [('五PPO全新家庭轻量选择实验报告_2026-10-04.md', None)]),
    (28, '相似初始观测下的收益反转：缓存、机制与预测偏差',
     'F5TE0018与F5TR0006是当前度量下的近邻，并非同一观测。容量、增长、起火和背景烟气差异可见；相似度不能当可信度，更不能把距离贡献当物理因果贡献。',
     [('相似初始观测收益反转缓存审计报告_2026-10-05.md', None)]),
    (29, '自动实验与独立真实LLM建议对照',
     '自动主流程因预先规定的收益条件未通过，跳过行为/噪声支线；后来另行冻结的真实LLM独立分支已完成。两个流程的48家庭与当前保留最终48情景不是同一封存角色。',
     [('疏散自动实验最终复核与通俗说明_2026-10-05.md', None)]),
    (30, 'LLM语义、任务目标、信息确定性与输出交付诊断',
     '本章按同题配对检查解析、目标翻转、信息等价表示、确认与预测条件、JSON约束及思考请求。简单题参考答案只适用于其定义，不能套给容量不等的复杂疏散情景。',
     [('LLM建议语义诊断结果与下一步_2026-10-05.md', None),
      ('LLM任务目标与字段角色对照实验报告_2026-10-06.md', None),
      ('LLM信息确定性结果复核与下一步_2026-10-06.md', None),
      ('LLM输出机制结果复核与通俗说明_2026-10-06.md', None)]),
    (31, '8B、14B、27B本地模型与运行成本',
     '模型规模实验是功能和本机运行诊断，不是疏散方法公平消融。8B/14B复用旧题，27B分阶段改变运行设置；32次请求不等于32个独立问题家庭。',
     [('LLM_8B模型复测实验报告_2026-10-06.md', None),
      ('LLM_14B模型复测实验报告_校订版_2026-10-06.md', None),
      ('Qwen3.5_27B纯文本压缩与完整测试说明_2026-10-06.md', None)]),
    (32, '时间戳现场流程、约束一致性与27B提速边界',
     '现场观测→建议→时效与规则检查→PPO执行的代码链已经实现，但27B驻留提速没有完成受控验证。2次重复计时不替代32题诊断、16题确认或疏散效果实验。',
     [('27B提速与LLM_PPO时间戳流程执行报告_2026-10-07.md', None),
      ('27B直接计时与内存压力实测_2026-10-07.md', None),
      ('论文开发实验_方向覆盖与等待反事实审计_2026-10-07.md', None)]),
    (33, '真实Qwen3 14B缓存接入及完整疏散配对实验',
     '这里才回到真实14B响应的疏散效果。即时与真实耗时回放分别报告；当前只有初始建议，没有滚动刷新建议。无建议组为同权重消融，不是新训练的纯PPO。',
     [('Qwen3_14B缓存与PPO配对实验报告_2026-10-07.md', None)]),
    (34, '14B建议到实际动作：方向、容量与剩余损失归因',
     '先核实建议是否真的改变动作，再在同状态比较合法动作，并用新人口验证方向提示修正。保留校订版归因，后续分区域机制验证在下一章继续补足。',
     [('14B建议作用与候选收益审计报告_2026-10-07.md', None),
      ('14B动作审计与出口提示修正验证_2026-10-07.md', None),
      ('14B剩余负效应归因总审计_校订版_2026-10-07.md', ['1.', '2.', '3.', '4.', '5.', '6.', '8.'])]),
    (35, '方向、放行容量、保持时间与重复建议的机制实验',
     '五个情景共用同一五区域图。旧报告出现的“五个建筑”在本整合版本中统一为“五个参数情景”；新人口、权重、模式与未来副本均不算独立建筑。',
     [('方向容量与保持时间机制实验报告_2026-10-07.md', ['1.', '3.', '4.', '5.', '7.']),
      ('方向容量机制实验_结论与后续计划_2026-10-07.md', ['1.', '2.', '3.', '4.', '5.'])]),
    (36, '当前是否值得再次覆盖PPO：新人口确认与误拒审计',
     '这是截至本次整合最近一个已完整执行并审计的控制实验。它同时报告有害拦截和有益误拒，避免把几乎不动作的门控称为准确收益判断。',
     [('再次覆盖PPO门控确认实验报告_2026-10-07.md', ['1.', '2.', '6.']),
      ('再次覆盖PPO_结果解释与独立审计_2026-10-07.md', ['1.', '2.', '3.', '4.', '5.', '7.'])]),
]


HEADER = """# LLM与强化学习火灾疏散研究：全程整合报告

**整合日期：2026年10月8日（北京时间）。研究内容包括此前原型、公开行为数据、所有已完成控制与诊断阶段；最新人群响应收益预测尚未产生实验结果。**

用途：导师讨论、论文选题收敛、方法与结果复盘、代码实践交接。性质：有证据索引的阶段研究报告；不是已经证明现实消防效能的终稿。

## 阅读说明与当前结论

已经完成的是可运行的研究平台、多训练种子与行为/噪声对照、公开人类选择预测、本地真实LLM响应、反事实教师和收益选择器的多轮开发，以及逐步实际动作和统计审计。尚未建立当前LLM＋PPO稳定优于强规则或纯PPO的证据；没有真实火灾、真实建筑、FDS耦合或连续轨迹层的完整验证。

研究推进并不是单调提高分数。前期解决“系统能否运行”，中期解决“比较是否公平”，后期把失败分解为语义、观测、目标、容量、时序和收益预测问题。负结果、未执行分支与失败恢复记录都保留，不依据新结果把探索性方法改称确认性主方法。

最近一个完整控制实验完成2,080回合、2,480终点反事实、248个建议机会和102,632个控制步。即时模式24个再次覆盖机会中，13有害、5有益、6微小；两种门控拒绝全部再次机会，所以同时拦住13/13有害与误拒5/5有益。当前门控未达到有效区分的目标。

人群响应预测这一新阶段只完成特征、轻量岭回归与测试初稿，没有执行测试、拟合、校准或新人口验证。本次报告整理不运行训练、LLM或疏散模拟，也不打开保留最终情景。

### 读数与角色统一

1. 失能率用百分比，两个失能率差用百分点；FED为本合成环境累计危险增量，不写成真实人体毒性剂量。
2. 场景家庭、人口、行为模型、训练权重、监督初始化、未来种子和时刻是不同层次。大量回合不等于大量独立建筑。
3. 反复使用的旧20场景统一称开发集。历史两轮确认性银行只在各自当时冻结条件下具有确认角色，已查看后不再充当修改方法的未见最终测试。
4. 自动主流程与LLM独立分支各有自己的开发家庭；“48家庭”不能与近期继续封存的48最终情景混为同一数据集。
5. CalibratedLogit名称不代表真人校准。当前控制行为参数来源标记为synthetic_demo_not_human_evidence；公开人类选择验证属于尚未连接到控制实验的另一条证据链。
6. 早期LLM-surrogate是占位系数模型，不是实时LLM。后期真实LLM建议属于高层建议模块，不能反写成已验证的人类行为模型。
7. 601—605在不同目录和训练臂中不是同一个模型集合。五纯PPO候选选择阶段与近期raw_llm_ppo同权重消融阶段必须按路径区分。
8. 单步强制后统一PPO的终点收益、恢复父策略的终点收益和完整回合差异是三个量；局部收益不可直接相加。

### 如何阅读

先读第1—3章了解研究来源、系统和数据；第4—16章是截至10月2日的详细早期史；第17—29章是教师、收益预测、候选覆盖和自动实验；第30—36章是LLM语义、模型成本和最新控制机制；第37—40章给出准确的当前状态、综合判断、下一步设计与复现方法。

后续章节保留原报告的具体实验日期、统计定义、冻结参数和结果。章节内“本轮”均指该段来源的那一轮；原报告内的后续建议是当时判断，其实际执行进展以本整合报告后续章节及第37—40章为准。旧报告没有被覆盖。

"""


CLOSING = """
## 37. 人群响应观测与覆盖收益预测：当前代码阶段

### 37.1 为什么转向这个问题

最近查表门控拦住有害建议，却也拒绝全部有益再次覆盖；更细的当前观测减少了分格混合，但仍发现一组完整公开历史相同、收益符号相反的病例。这不能证明任何合法传感器都无法预测，却说明仅凭现有格表不能稳定识别。

AULL0019、权重604、第15秒：概率行为人口1497101301的候选减PPO FED为+0.001212877177；Rule人口1497101302为-0.009771520627。失能差均为0。两者既有34维观测和完整公开动作、放行历史一致，背后人员组成及行为模型不同；在线控制器不能通过读取行为编号或隐藏人员属性区分。

### 37.2 已写代码和未完成事项

已新增crowd_response_gain.py和test_crowd_response_gain.py。模块包含公开字段投影、只追加已完成步的历史、三种特征表示、家庭等权岭回归、两个独立收益目标、经验校准余量和接受判据。代码是开发初稿，没有训练PPO，也尚未拟合收益模型。

| 工作 | 截至本报告的状态 |
|---|---|
| 决策前公开历史与响应特征初稿 | 已写，未运行验证 |
| 当前观测／公开历史／响应三种表示 | 已写接口，尚无比较结果 |
| 失能差和FED差的岭回归实现 | 已写函数，尚未拟合 |
| 基于校准家庭残差的经验余量 | 已写函数，尚未校准 |
| 信息泄漏、时序、烟气边界和门控测试 | 已写5项，尚未运行 |
| 新实验执行脚本与冻结清单 | 尚未创建 |
| 全新人口回合及反事实验证 | 尚未启动 |
| 相对原观测预测器的误差与控制收益 | 尚无结果 |

### 37.3 响应指标应怎样解释

两侧实际过路人数、队列消退速度、指示方向上的过路比例可以由摄像计数或区域传感器构想获得；本仿真缓存已有聚合计数。但“沿指示方向过路的人数比例”同时受烟气、拥堵、人数构成与容量截断影响，不能直接当个体服从率，更不是已识别的因果服从参数。

初稿读取当前34维公开观测、基线与候选动作、既往建议及覆盖次数。公开历史增加人数变化、近三步人数、方向占比、放行与切换；响应表示增加两侧流量、无流量、沿指示流量比例、容量截断标志、路口清除比例和出口吞吐。所有历史仅使用当前决策前已完成的步骤，不读取终点FED、失能计数、个体偏好或未来源真值。

旧缓存没有直接记录每侧出口人数。初稿只在累计公开最大烟气增量仍小于仿真失能阈值1时，用“前走廊人数＋本步进入人数−后走廊人数”推断离开人数；否则关闭该推断并给出不可用标志。这个条件依赖当前模拟器初始FED为0及其更新规则，接入现实烟气模型时不能直接沿用。

现有早期公开人数闭合可能已经隐含这些流量，因此新增指标可能只是更易学习的历史表示，未必提供新信息。必须同时比较当前观测、普通公开历史和响应表示，才能分清“新增传感器信息”与“历史特征工程”的效果。

### 37.4 后续需要验证的收益目标

针对当前具体候选a_L和同状态PPO动作a_P，定义两个代价差：ΔI=I(a_L→共同后续PPO)-I(a_P→共同后续PPO)，ΔF=FED(a_L→共同后续PPO)-FED(a_P→共同后续PPO)。负差表示候选改善，不能与先前“601损失减候选损失”的正向收益符号混用。

当前代码初稿采用分目标预测，没有把失能和暴露任意合成一个分数。点预测门控只有预测失能不增加，并且失能或FED降低达到0.001时才允许候选；保守版本另加经验不利残差余量。阈值仅是开发描述和初稿规则，尚未冻结成已验证方案。

经验残差带不是经过认证的风险上界，也不是LLM自报confidence的概率校准。没有数据验证前，不能报告预测准确率、有害拦截率或控制收益。

## 38. 把所有阶段连起来：当前能说明什么

### 38.1 已完成成果

工程上，研究已具备危险场、人群选择、拥堵、指示控制、联合容量动作、PPO训练、本地LLM缓存、时间戳检查、共同约束、反事实终点评价、轻量监督与统计审计链。代码、原始回合、模型、响应和报告均按阶段保存，能够区分请求与实际执行。

方法上，已经建立强基线、相同动作/信息/预算的历史公平对照、多训练种子、行为×噪声矩阵，以及先冻结再验证的新开发家庭或人口。查表、监督高准确率和零违规没有被直接当作有效控制证据。

机制上，已经有条件于本模拟器的证据说明：静态照建议执行可以制造虚假的“动态纠正”；总体准确率可被复制上一动作支配；合法JSON不保证任务语义；预测安全出口可能因低吞吐而增加滞留；方向与容量存在交互；同一建议在第一次与再次采纳时收益可变；旧提名筛选可漏掉候选正例；有限未来、隐藏人口和观测分布变化共同影响标签。

### 38.2 主要效果证据与限制

| 结果 | 已观察到的量 | 可以支持的判断 |
|---|---|---|
| 第一轮公平LLM建议对纯PPO | 失能+1.34个百分点，区间[+0.67,+2.10] | 该模型/提示/环境版本出现退化 |
| 第二轮护栏LLM对纯PPO | +1.02个百分点，区间[-1.57,+4.55] | 未建立稳定优势；也不能称等效 |
| PTDEV24离线worst3教师对601 | -0.25752个百分点，探索性区间[-0.57006,-0.03183] | 有限开发银行存在离线规划机会 |
| 自动独立LLM同控制器对规则建议 | +0.237847个百分点，区间[-0.085077,+0.597305] | 改善信号未通过 |
| 最近首次门控对持续提示，即时 | FED -0.000725，失能-0.0521个百分点 | 局部开发均值改善，仍误拒有益再次动作 |
| 最近查表门控再次覆盖 | 拦住13/13有害，误拒5/5有益 | 全拒绝基线行为，未识别净收益 |

表内条件、独立单位、候选权重和计算资源不同，因此不能按这些数值跨阶段排名，更不能画成一条持续进步的性能曲线。离线特权教师不等于可部署LLM＋PPO。最近两行只涉及五个选过的同图参数情景及新人口，不作独立建筑显著性声明。

### 38.3 尚未完成的研究环节

未建立当前主方法的稳定LLM独立增益；未通过原硬护栏资格；未完成第三轮至少十个新PPO训练种子及新最终评估；当前保留48最终情景未读取。没有FDS危险转换和真实楼层校核，没有JuPedSim连续轨迹外部验证；没有把包含烟气、引导、封闭和拥堵的真实逐人数据校准到当前控制模型。

“零限制违规”只说明本次模拟操作约束符合；不能等同零失能。回合终止可以因为失能，应只在全部成功退出时报告全员成功疏散时间。辅助监督网络初始化、人口和未来也不能代替新PPO训练种子。

### 38.4 当前最合理的论文主线

建议将主问题收敛为“在不确定人群响应下，何时让高层建议覆盖一个已有闭环控制器”。主张应建立在观测可得性、具体候选净收益、有害拦截与有益误拒，以及全程效果上，而不是模型规模或实验数量上。

已有失败机制可作为有价值的研究结果，但新颖性仍需针对目标会议/期刊核对；本整合没有进行新的穷尽文献检索。不能把LLM＋RL的组合名称本身当作已证明创新，也不能把负结果泛化为所有LLM不适合疏散。

## 39. 下一阶段完整设计与推进条件

本章为建议性设计，尚未冻结、采集或完成。它不替代前述已完成实验，也不在本次文档整理中执行。

### 39.1 先做数据与信息审计

从旧44机制状态和最近持续提示轨迹抽取当前具体候选相对同状态PPO的两项终点差。概率行为先按同状态配对未来取均值，不把未来当独立样本。相同人口、行为、权重、状态和完整路径的时序副本去重；仅以持续提示共同轨迹作主要比较，其他门控自身轨迹另列。

首先运行初稿测试，核实历史严格早于决策、传感器投影排除身份与结局、出口推断在可能失能时停用，并检查收益反转对在新增表示下是否仍相同。这一步可能得到“新指标没有新增信息”的负结果，也应保留。

### 39.2 拟合、校准与新人口隔离

按情景和人口家族划分拟合与校准，同行为、权重、时序和状态副本保持同组；新人口验证在特征、固定正则、门控和脚本冻结后才运行。候选初始分法可用147旧人口与149人口重复0/1拟合，149重复2/3校准，150新前缀验证；这些仅为拟定命名空间，实际冻结前还需检查冲突。

比较三种表示，使用同一种固定预算岭回归与相同两项收益标签，不在验证集搜索正则、窗口或阈值。当前观测、普通公开历史、响应特征均不输入情景编号、行为编号、人口种子、训练权重编号或隐藏个体属性。

### 39.3 完整回合和同状态反事实

建议控制臂为无建议同权重PPO、持续提示、仅首次覆盖、当前观测收益点门控、响应收益点门控、响应收益余量门控，加同信息规则。五情景×四新人口×两行为×五权重×两时序×六PPO控制=2,400回合，规则另80，共拟定2,480回合；该数是预算方案，不是已经完成量。

记录所有建议覆盖机会，分别强制PPO和候选一步后共同PPO直到终点；另恢复父门控检查后续交互。原实际未来下强制实际动作并恢复父门控应逐步精确重现父轨迹，否则先修执行与随机配对，再解释结果。

### 39.4 主要读数和决策

分别报告失能差与FED差的MAE、偏差和经验误差带覆盖；按首次/再次、行为、情景和权重分解。有害拦住/有害总数、有益误拒/有益总数、接受的有害与有益、实际覆盖次数和完整回合差异一起报告。无分母项目记不可估计；全回退不能称预测成功。

新增响应若只改善开发拟合，却不改善新人口预测与控制，不采用；如果收益仅来自一个情景或一套权重，应报告集中度并再冻结外部条件验证。若有害下降但有益全部被拒，保持诊断性质。没有前序稳定证据前继续暂停扩大PPO，不读取最终48情景、不调整Ollama、不请求新LLM。

## 40. 代码、数据、复现与交付

### 40.1 文件如何对应研究问题

src/fire_evac为环境、行为、控制、约束、教师和预测模块；scripts为训练、采集、分析和审计；outputs为逐回合、响应、模型和冻结清单；reports为固定方案与解释；artifacts为论文、Word/PDF和报告图。附录证据索引提供可点击的实际路径。

历史模型的训练输入、动作和权重随轮次改变。复现时应使用对应manifest、源码快照、模型和随机流；不能用当前默认参数替代旧配置，也不能将同编号不同训练臂模型混用。

### 40.2 读报告不会触发训练

本次仅读取与整合证据，生成文档和两张描述性图，不运行新的火灾回合、拟合、PPO更新或LLM请求。既有结果、模型选择和开发台账保持原哈希。报告生成与质量核验脚本为新的文件，不改旧脚本。

本机近期实验使用D:/miniconda3/python.exe。报告的环境版本与耗时取各轮记录，并不声明旧轮都在同一软件版本、同一负载下运行。时区统一以北京时间解释本次日期；目录中的UTC时间戳保留原值。

### 40.3 本报告怎样用于导师讨论

可以围绕三项具体决定讨论：是否把论文主线收敛到建议覆盖价值；哪些聚合响应在目标场所确实能测到；用什么独立行为或布局检验模型。在有这些答案之前，现有结果适合提交为完整实践、方法诊断和负结果证据，不写成已实现现实伤亡下降的产品系统。

"""


def preserve_snapshot():
    ASSETS.mkdir(parents=True, exist_ok=True)
    target = ASSETS / 'preservation_before.json'
    assert not target.exists(), 'Use a new report version rather than replacing the preservation snapshot.'
    files = set()
    for folder in ('reports', 'src', 'scripts', 'tests'):
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.md', '.py', '.json') and '__pycache__' not in str(p))
    files.update(ROOT / p for p in PROTECTED)
    files.update((ROOT/'outputs/guarded_confirmatory_v2/models').glob('seed_*_raw_llm_ppo.zip'))
    for folder in ('repeat_override_confirmation_full_20261007T144941461333Z', 'repeat_override_confirmation_continuations_20261007T145327415464Z',
                   'repeat_override_confirmation_analysis_20261007T145724470047Z', 'repeat_override_independent_audit_20261007T150059095758Z',
                   'direction_release_mechanism_full_20261007T142136292884Z', 'direction_release_mechanism_continuations_20261007T142443133563Z'):
        files.update(p for p in (ROOT/'outputs'/folder).iterdir() if p.is_file())
    records = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    for p, h in PROTECTED.items(): assert records[p] == h, p
    target.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records, reserved_final_scenarios_read=0),ensure_ascii=False,indent=2),encoding='utf-8')
    return len(records)


def insert_stage(n, title, interpretation, sources, inventory):
    out = [f'## {n}. {title}\n\n', interpretation+'\n\n']
    index = 0
    for name, selections in sources:
        text = (REPORTS/name).read_text(encoding='utf-8')
        intro, parts = sections(text)
        inventory.append(dict(chapter=n, source=name, sha256=sha(REPORTS/name), section_selection=selections))
        index += 1
        label = re.sub(r'(?m)^# (.+)$',r'\1',intro).strip().splitlines()[0]
        out += [f'### {n}.{index} 证据来源：{label}\n\n',link('原始报告及其原始结果入口',f'reports/{name}')+'。以下保留该轮口径；其后续建议按当时判断阅读。\n\n']
        intro_body = '\n'.join(intro.strip().splitlines()[1:]).strip()
        if not parts:
            # Reports with no level-two headings contain their full concise narrative.
            intro_body = re.sub(r'(?m)^(这轮能说明什么|这轮不能说明什么|下一步|完成的工作)$',r'**\1**',intro_body)
            out.append(intro_body+'\n\n')
            continue
        if intro_body: out.append(intro_body+'\n\n')
        sub = 0
        for heading, body in parts:
            if selections is not None and not any(heading.startswith(prefix) for prefix in selections): continue
            sub += 1
            heading = re.sub(r'^\d+[.、]\s*','',heading)
            if '下一步' in heading or '下一项' in heading: heading = '当时的后续安排：'+heading
            body = re.sub(r'(?m)^### (.+)$',lambda m:'**'+re.sub(r'^\d+(?:\.\d+)*\s*','',m[1])+'**',body)
            body = re.sub(r'(?m)^#### (.+)$',r'**\1**',body)
            out += [f'#### {n}.{index}.{sub} {heading}\n\n',body.strip()+'\n\n']
    return ''.join(out)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axs[0].bar(['有害被拒绝','有益被拒绝','微小被拒绝'],[13,5,6],color=['#2a7182','#c0644c','#8a97a0'])
    axs[0].set_ylim(0,16);axs[0].set_ylabel('建议机会数');axs[0].set_title('即时模式：24次再次机会全部拒绝')
    for i,v in enumerate([13,5,6]):axs[0].text(i,v+.35,str(v),ha='center')
    axs[1].bar(['有害拦截率','有益误拒率'],[100,100],color=['#2a7182','#c0644c'])
    axs[1].set_ylim(0,120);axs[1].set_ylabel('%');axs[1].set_title('必须同时看拦截与误拒')
    for i,s in enumerate(['13 / 13','5 / 5']):axs[1].text(i,103,s,ha='center')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    p=ASSETS/'repeat_override_tradeoff.png';fig.savefig(p,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
    boxes=[(0.2,3.8,'可运行平台\n烟气、人群、方向和容量'),(3.5,3.8,'公平效能检验\n强规则、PPO、LLM建议'),(6.8,3.8,'失败机制诊断\n语义、目标、时序、响应'),(0.2,1.0,'公开人类选择验证\n尚未校准当前控制行为'),(3.5,1.0,'当前待执行\n响应观测＋净收益预测'),(6.8,1.0,'独立验证与现实耦合\n最终场景、FDS、轨迹')]
    for x,y,s in boxes:ax.text(x+1.3,y,s,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.7',facecolor='#e7f0f4' if y>2 else '#f4eee4',edgecolor='#426478'))
    for a,b in [((2.85,3.8),(3.45,3.8)),((6.1,3.8),(6.75,3.8)),((8.1,3.1),(4.8,1.7)),((6.1,1),(6.75,1))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#426478'))
    ax.text(5,4.75,'研究路线：工程完成、控制增益未确立、外部证据尚待连接',ha='center',fontsize=13)
    p2=ASSETS/'integrated_research_route.png';fig.savefig(p2,dpi=180);plt.close(fig)
    return p,p2


def build():
    before=preserve_snapshot()
    old=(REPORTS/'LLM_RL火灾疏散全流程实验研究总报告_2026-10-02.md').read_text(encoding='utf-8')
    core=old[old.index('## 1. '):old.index('## 17. ')]
    # The word latest remains local to a dated phase, rather than current project state.
    core=core.replace('## 16. 最新2×2开发实验：','## 16. 截至10月2日的2×2开发实验：')
    core=core.replace('以下是根据现有证据提出的后续方案，尚未执行','以下为该阶段的后续设想，进展见后续章节')
    inventory=[]
    content=HEADER+core+'\n'+''.join(insert_stage(*stage,inventory) for stage in STAGES)+CLOSING
    content=content.replace('新人口仍在这五个选择过的建筑上','新人口仍在这五个选择过的参数情景上')
    content=content.replace('五个选过的建筑','五个选过的同图参数情景')
    p,p2=figures()
    content=content.replace('### 读数与角色统一',f'![研究路线与未完成证据连接]({p2.as_posix()})\n\n### 读数与角色统一')
    content=content.replace('## 37. 人群响应观测',f'![再次覆盖门控：有害拦截与有益误拒]({p.as_posix()})\n\n来源：最近冻结实验的同状态反事实；两种门控在此再次子集的决定相同。这是24个机会的描述图，不是24个独立场景。\n\n## 37. 人群响应观测')
    # Keep the old glossary and E01-E25 evidence anchors, not its superseded closing recommendations.
    _, old_sections=sections(old)
    appendices={title:body for title,body in old_sections if title.startswith('附录')}
    content+='## 附录A. 通俗术语与读数\n\n'+appendices[next(t for t in appendices if t.startswith('附录A'))]
    content+='\n## 附录B. 早期核心证据E01—E25\n\n'+appendices[next(t for t in appendices if t.startswith('附录B'))]
    content+='\n## 附录C. 后期阶段来源与可追溯清单\n\n|章|来源报告|复核口径|\n|---|---|---|\n'
    for item in inventory:
        content+=f"|{item['chapter']}|{link(item['source'], 'reports/'+item['source'])}|旧报告与冻结结果；本次只读整合|\n"
    content+='\n补充证据：'+link('最新独立审计','outputs/repeat_override_independent_audit_20261007T150059095758Z/summary.json')+'；'+link('原始行为数据来源标记','outputs/behavior_pipeline/metrics.json')+'；'+link('人群响应代码初稿','src/fire_evac/crowd_response_gain.py')+'；'+link('尚未运行的新测试','tests/test_crowd_response_gain.py')+'。\n'
    content+='\n## 附录D. 历史方案、版本与未完成状态说明\n\n下列清单覆盖整合时已有阶段报告与方案。实验、修订、方案和解释稿不是新的独立实验，不能按文件数累计工作量。校订版优先用于新解释，旧版仍保全。\n\n|文档|角色提示|\n|---|---|\n'
    for path in sorted(REPORTS.glob('*.md')):
        if path.stem==NAME or path.name.startswith('总报告素材'): continue
        role='固定方案/准备记录，不代表执行完成' if any(w in path.name for w in ('方案','protocol','草稿','素材')) else '实验/诊断/解释记录，执行状态见对应正文'
        content+=f'|{link(path.name,path.relative_to(ROOT))}|{role}|\n'
    content+='\n## 附录E. 本次报告整理与核验边界\n\n本次沿用10月2日总报告核验过的早期证据，再读取后期报告、清单、源代码与最新回合及反事实结果。没有重新查看导师临时截图或重新进行全文文献检索，导师与论文部分以既有源材料核对记录为依据。两篇目标论文已下载全文，Tong论文仍只有已核实文章页/摘要，不能补写为已取得全文。\n\n'
    content+='文档整理不触发新模拟、训练或推理。核验覆盖来源哈希保全、核心近期计数与均差重算、本地链接、章节、Word正文/表格/图片、PDF文本与分页及图表目检；不声称重新运行了全部历史统计或全项目测试。完整核验日志与来源哈希放在'+link('本次报告资产目录',ASSETS.relative_to(ROOT))+'。\n'
    path=REPORTS/(NAME+'.md');assert not path.exists();path.write_text(content,encoding='utf-8')
    record=dict(report=str(path),created_at_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-10-08',client_timezone='Asia/Shanghai',source_character_count=len(content),chinese_character_count=len(re.findall(r'[\u4e00-\u9fff]',content)),chapters=40,retained_old_core_chapters=16,post_oct2_source_reports=inventory,preserved_snapshot_files=before,new_evacuations=0,new_fits=0,new_ppo_updates=0,new_llm_requests=0,reserved_final_scenarios_read=0,crowd_response_stage='code_draft_not_tested_not_fitted_not_evaluated')
    (ASSETS/'assembly_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='post_oct2_source_reports'},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');build()
