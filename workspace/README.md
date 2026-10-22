# 行为感知的火灾疏散强化学习原型

这个仓库把论文方案落成一个可运行的研究闭环：

1. 合成火灾危险场随时间变化；
2. 规则、mixed-logit 和 LLM-surrogate 三类疏散者在路口选择出口；
3. 动态指示牌控制器观察人群、拥堵和危险；
4. Q-learning 和 PPO 学习标志方向，并与静态和危险感知规则基线比较；
5. 参与者级行为数据按人划分，拟合层次logit并输出锁定测试指标；
6. 所有仿真结果按 episode 输出，便于后续统计分析。

当前版本是工程原型，不是论文结果。`LLMSurrogateBehavior` 只是接口和实验流程占位，必须使用真实人员数据校准并在锁定测试集上验证后，才能解释为人类行为模型。

截至2026年10月2日的完整研究经过见[全流程实验研究总报告](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/LLM_RL火灾疏散全流程实验研究总报告_2026-10-02.md)，另有[Word版](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/artifacts/LLM_RL火灾疏散全流程实验研究总报告_2026-10-02.docx)和[PDF版](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/artifacts/LLM_RL火灾疏散全流程实验研究总报告_2026-10-02.pdf)。报告覆盖原型、真实数据预测、动态门控、两轮公平确认性实验、护栏与最新计划蒸馏，并保留负结果和证据边界；当前尚未建立稳定LLM控制增益，正式第三轮仍未启动。

## 目录

- `src/fire_evac/`：仿真环境、行为模型、控制器、Q-learning 与 LLM 结构化决策接口。
- `schemas/llm_decision.schema.json`：LLM 路线选择响应格式。
- `schemas/behavior_observation.schema.json`：参与者级路线选择数据格式。
- `schemas/fds_zone_timeseries.schema.json`：FDS分区危险时间序列格式。
- `scripts/run_prototype.py`：训练并评估第一版策略。
- `scripts/prepare_behavior_data.py`：校验和划分行为数据，拟合行为基线。
- `scripts/train_ppo.py`：训练PPO并执行跨行为模型评估。
- `scripts/run_multiseed_ppo.py`：五随机种子训练、最佳模型选择与分层Bootstrap。
- `scripts/run_behavior_cross_experiment.py`：PPO-Rule、PPO-Logit、PPO-LLM和PPO-Ensemble交叉测试。
- `scripts/analyze_behavior_cross.py`：策略平均、最坏情况与训练稳定性排名。
- `scripts/run_recurrent_comparison.py`：训练RecurrentPPO并与4步历史PPO对照。
- `scripts/analyze_recurrent_comparison.py`：循环策略配对Bootstrap与稳健性排名。
- `scripts/run_robust_ensemble.py`：分阶段增加当前较差行为模型的训练抽样权重，并与PPO-Logit、等权集成对照。
- `scripts/run_local_route_validation.py`：使用本地Ollama模型预测公开虚拟疏散实验中的左右走廊选择，按参与者锁定测试。
- `scripts/run_lu_warden_validation.py`：使用论文公开系数复现 Lu et al. (2025) 的12个引导员—群众冲突场景。
- `scripts/run_li_group_validation.py`：复现 Lu et al. (2026) 的8个三出口场景、4组效用函数和群体拆分分布。
- `scripts/run_conflict_llm_rl.py`：在烟雾、群众、标志与引导员冲突场景中比较论文系数基线、PPO、本地LLM和LLM+PPO。
- `tests/`：基本验证与护栏运行一致性测试，完整检查使用 `python -m pytest -q`。
- `reports/`：研究报告 Markdown 源文件。
- `artifacts/`：Word 报告等交付文件。
- `outputs/`：实验输出。

## 运行

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\run_prototype.py --train-episodes 3000 --eval-episodes 300
python scripts\prepare_behavior_data.py
python scripts\train_ppo.py --timesteps 50000 --eval-episodes 300
python scripts\analyze_ppo_results.py
python scripts\run_multiseed_ppo.py --seeds 41,42,43,44,45 --timesteps 50000 --eval-episodes 300
python scripts\run_behavior_cross_experiment.py --variants rule,logit,llm,ensemble --seeds 51,52,53
python scripts\analyze_behavior_cross.py
python scripts\run_recurrent_comparison.py --seeds 51,52,53 --timesteps 30000 --eval-episodes 200
python scripts\analyze_recurrent_comparison.py
python scripts\run_robust_ensemble.py --seeds 51,52,53 --timesteps 30000 --phases 6 --validation-episodes 30 --eval-episodes 200
python scripts\run_lu_warden_validation.py --draws-per-scenario 10000
python scripts\run_li_group_validation.py --draws 10000
python scripts\run_conflict_llm_rl.py --scenarios 48 --train-scenarios 36 --timesteps 10000
python -m unittest discover -s tests -v
```

## 最低成本本地LLM行为验证

从[Snopková等人公开数据](https://springernature.figshare.com/articles/dataset/Predictors_of_evacuation_behavior_dataset_on_respondents_route_choice_and_web_interaction/27705402)下载 `Snopkova_Isovists.zip` 到 `data/public_snopkova/`。在本机Ollama服务中准备 `qwen3:4b` 后运行：

```powershell
ollama pull qwen3:4b
python scripts\run_local_route_validation.py
```

脚本根据论文公开的任务编码提取走廊宽度、长度和楼梯信息，跳过训练题、未作答及无法可靠解码的任务；按参与者固定划分训练、验证和测试；每种独立情境只调用一次本地模型，缓存模型摘要、提示词、原始回复和预测概率。输出位于 `outputs/local_llm_validation/`。对照包括总体频率、固定几何规则以及训练参与者在相同任务上的选择频率。

原实验让参与者在8秒内查看可旋转的虚拟场景，而这里给LLM的是属性文字。数据没有真实烟气、动态指示牌或火场行为，因此此实验只检验文本条件下的虚拟路线选择预测，不能当作真实火灾行为验证。

不传 `prepare_behavior_data.py --input` 时只生成明确标记的合成演示数据。接入真实数据时使用：

```powershell
python scripts\prepare_behavior_data.py --input data\真实路线选择数据.csv
```

如使用 Codex 工作区自带 Python：

```powershell
$python = "C:\Users\摩天轮大盗\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "$PWD\src"
& $python scripts\run_prototype.py
```

## 下一步接口

- 将 `SyntheticHazardProvider` 替换为 FDS 网格或分区 CSV 输入；
- 将当前分区流模型替换为 JuPedSim 连续轨迹；
- 用真实数据和冻结的 LLM 响应替换当前合成校准产物；
- 用更多训练种子和配对统计检验自适应最坏情况训练，并加入传感器扰动；
- 保留相同的训练—测试行为交叉矩阵。

## 第三轮开发审计状态（2026-09-30）

最新报告：[第三轮护栏审计与605模型干预报告](reports/第三轮护栏审计与605模型干预报告.md)。护栏按真实空集合回退规则的危险动作拦截率为92.15%，没有达到95%门槛；第三轮正式训练与最终银行尚未启动。605模型的全放行干预未改善诊断子集的平均失能率，不能继续将“放行过慢”当作已确认根因。

- `scripts/audit_counterfactual_shield.py`：保留原开发模型，审计运行掩码、各冲突指标及20步/完整回合/不同随机未来的标签变化。
- `scripts/diagnose_guarded_release.py`：复现旧模型结果，记录固定出口指令和闭环容量干预。属于事后诊断。
- 现有结果位于 `outputs/conditional_shield_audit_v1/` 和 `outputs/guarded_seed605_diagnosis_v2/`。重新运行时用 `--output` 指定新目录，避免覆盖。

后续完整回合开发试验已完成，见[完整回合反事实护栏开发实验报告](reports/完整回合反事实护栏开发实验报告.md)。主试验499状态、26,946次模拟，显式三分类并比较有/无可观测历史。两组均未达到冻结门槛；不确定自动放行会漏掉危险动作。事后取消该规则仍未达标，不替换原结果。

- `scripts/build_terminal_shield.py`：六类新开发场景、三个随机未来和两种继续计划、分组训练/校准/审计，支持清单一致的 `--resume`。
- `scripts/diagnose_terminal_shield.py`：事后诊断弃权规则及相同观测、不同隐藏火源的标签歧义。
- `src/fire_evac/terminal_shield_runtime.py`：记录请求、筛选、实际动作的94维运行接口，尚未用于正式PPO训练。
- 输出位于 `outputs/terminal_shield_development_v1/` 与 `outputs/terminal_shield_diagnosis_v1/`；第三轮正式最终银行仍未生成。

可能火源与动态计划开发已完成，见[可能火源与动态计划开发实验报告](reports/可能火源与动态计划开发实验报告.md)。重放249个状态，比较等待、实时改道和标志时段分流；两火源参考平均失能7.94%，没有优于单火源参考7.83%。同一初始人群与检验随机未来的事后既有纯PPO平均为3.31%，表明手写计划库仍弱，不能直接作PPO强制监督教师。

- `scripts/run_source_world_planning.py`：规划与检验随机流分离的十二计划、九组开发参考比较；支持一致清单的续跑。
- `scripts/check_source_planner_ppo_reference.py`：既有五个纯PPO的事后能力参照，不是公平训练消融。
- 输出 `outputs/source_world_planning_development_v1/` 与 `outputs/source_world_planning_frozen_ppo_context_v1/`。下一开发阶段先将强PPO闭环候选纳入教师；正式最终银行与训练继续暂缓。

五PPO扩展候选库开发已完成，见[PPO扩展参考教师开发实验报告](reports/PPO扩展参考教师开发实验报告.md)。保留12手写计划并加入全部五个冻结纯PPO，共17候选，换用新规划及独立检验随机流。36家庭、249状态共22,599次完整回合模拟。扩展最坏世界教师初始失能率1.29%，强固定PPO 601为2.04%，原手写最坏世界教师7.85%；相对601差−0.747个百分点，探索性家庭Bootstrap区间[−1.464,−0.148]。12家庭改善、8变差、16持平，最坏世界选择未优于单火源选择。

- `scripts/run_ppo_extended_world_planning.py`：17候选、15执行组、五固定模型全部保留，清单完全一致时允许续跑。
- `src/fire_evac/ppo_reference_planning.py`：只复制环境的完整闭环PPO候选；保留原34输入和九动作，模型不再训练。
- `scripts/analyze_ppo_extended_world_planning.py`：观测、哈希、种子、选计划及90,783步实际执行审计，生成报告与图。
- 输出 `outputs/ppo_extended_world_planning_development_v1/`，90项测试通过。下一步尝试可观测输入的候选计划软建议蒸馏，并用新开发家庭留出审计；当前结果是离线特权参考，不能证明LLM增益或护栏资格通过，正式训练与最终银行仍未启动。

可观测计划软评分蒸馏开发已完成，见[可观测计划软评分蒸馏开发实验报告](reports/可观测计划软评分蒸馏开发实验报告.md)。旧36家庭分为24训练/12校准，训练五个49输入学生和五个仅预测字段学生，模型固定后审计24个新开发家庭。168状态共16,407次完整回合模拟。完整学生平均初始失能3.88%，强固定601为3.94%，离线教师2.68%；学生相对601差−0.056个百分点，探索性区间[−1.221,+1.119]，没有保留教师的主要收益，不能替代强固定对照。

- `src/fire_evac/plan_distillation.py`：观测输入、六项相对优先级、按家庭隔离的训练和校准、17候选软评分检查点。
- `scripts/run_observable_plan_distillation.py`：先固定十个监督模型，再运行新开发家庭；所有17候选和五固定PPO保留。
- `scripts/analyze_observable_plan_distillation.py`：实际疏散结果、家庭配对区间、观测推荐复核及66,835步执行记录审计。
- 输出 `outputs/observable_plan_distillation_development_v1/`，95项测试通过；`pytest.ini`限制只收集当前tests，避免归档副本干扰。后续先预声明更广训练家庭和相对601的代价/排序目标，使用新的开发审计；这24家庭已用于开发审计，不能重新称为独立最终集。当前不进入硬护栏或正式PPO训练。

扩展训练与相对PPO排序开发已完成（2026-10-02），见[扩展训练与相对PPO排序开发实验报告](reports/扩展训练与相对PPO排序开发实验报告.md)。固定2×2方案比较24/84训练家庭与MSE/相对601排序目标，四组均重新训练五个配对监督初始化；另保留五个历史模型。新增60训练家庭的424状态只采规划标签，所有25学生固定后才审计36个全新开发家庭的248状态。38执行组共55,152次唯一完整回合模拟，117项测试通过，108,879步实际执行审核通过。

四组平均初始失能率分别为小MSE 1.4857%、小排序1.4151%、大MSE 1.4722%、大排序1.4074%；强固定PPO601为1.3194%。大排序相对601差+0.088个百分点，探索性36家庭Bootstrap区间[−0.018,+0.230]，3家庭改善、3变差、30持平。四项因素对照区间均包含零，未建立扩充训练或目标方案的稳定收益。本轮最坏世界教师1.3002%，相对601优势也很小且未稳定复现，不能将结果全部归因于学生没有学好教师。

- `src/fire_evac/relative_plan_ranking.py`：新增相对601的带符号分数、差距加权排序；两个目标共用旧校准初始状态的教师失能遗憾选检查点，同值保留最早轮次。
- `scripts/run_relative_plan_distillation.py`：新训练标签、四组模型、历史复制、训练初始状态全局对照、冻结后新审计和精确清单续跑。
- `scripts/analyze_relative_plan_distillation.py`：家庭配对结果、完整四因素和交互、所有初始化、逐状态/标签/归一化/哈希/实际动作审核及中文报告。
- 输出 `outputs/baseline_relative_plan_distillation_development_v1/`。两个训练全局对照均选择601；大排序与历史学生的差异也包含检查点规则改变，不能单独归因于排序。保留所有版本，下一步先检查教师误选和学生的错误覆盖及标签歧义；不依据本轮挑最好初始化。PRAU36已成为开发数据，正式最终银行、至少十PPO训练种子、LLM公平消融和硬护栏资格仍未完成。

## 教师诊断与新开发验证（2026-10-03）

本次结果与研究判断见[教师诊断与新开发验证解读](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/教师诊断与新开发验证结果解读_2026-10-03.md)。[旧缓存诊断报告](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/反事实教师误选诊断开发报告.md)发现，旧PRAU两个变差家庭在规划时已接受真实火源损失，以降低另一世界的最坏损失；初始主结果与后续可达状态分别报告，不能认定为唯一失败原因。

随后按预先保存的[2×2方案](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/教师世界聚合与规划预算配对验证方案.md)，固定17候选和五个纯PPO，在24个新开发家庭比较最坏/等权平均世界与3/12规划未来。[原始完整结果报告](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/教师世界聚合与规划预算全新开发验证报告.md)记录21,150次完整回合模拟：worst3与worst12失能率均为0.9346%，601为1.1921%，配对差−0.2575个百分点，探索性区间[−0.5701,−0.0318]。两组有7个起点选择不同，不能因失能相同就认为策略相同；聚合与预算因素均未建立额外失能收益，mean12相对mean3的暴露指标有升高的探索性信号。

- `scripts/run_teacher_factorial_pilot.py`：保留的冻结采集与原分析入口；采集后存在已记录的NumPy整数JSON导出错误。
- `scripts/analyze_teacher_factorial_pilot_cached.py`：从完整缓存修复JSON导出，不重跑仿真；失败分析版本保存在输出的`analysis_revisions/`。
- `scripts/verify_teacher_factorial_pilot.py`：独立重算汇总和配对区间，核对缓存、六CSV修复前后相同、原输入与历史报告保全；无新仿真。

当前完整缓存只需运行分析修复入口。完整复现的两条命令和新目录要求见解读报告，不重复覆盖原采集进度。本轮12项现有规划测试通过，未重新运行上一轮117项完整测试。没有新学生/PPO训练或LLM调用；PTDEV24已成为查看过的开发数据。下一步先审查实际可观测输入的教师标签歧义，再另存泛化方案；正式第三轮、最终银行和硬护栏资格仍未完成。

