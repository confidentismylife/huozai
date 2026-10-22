# 证据与论文主张对照

2026年10月12日。用途：组会及论文写作的可追溯检查。此表由代理核对本地结果，不代表作者已完成人工来源审查或批准投稿。新人体数据验证、原始录像时点核实及新颖性检索仍待完成。

| 主张ID | 可以写出的主张 | 原始证据及定位 | 必须同时保留的边界 |
|---|---|---|---|
| C01 | 103条严格单次记录，四日期人数19、24、16、44 | [数据清单](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_initial_benchmark_20261012/strict_initial_cases.json)、[独立审核](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/evidence_audit.json)，human_records/date_counts | 严格记录数不等于经录像确认的独立人；跨日期身份未核对 |
| C02 | 主记录出口选择为81、22、0 | 独立审核recorded_exit_counts；E3缺项表示本集合0例 | 未出现出口3选择者不证明该出口封闭，也不证明无法选它 |
| C03 | 原始零样本Qwen合并准确率61.17%，观测Logit93.20% | [首轮结果](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_initial_benchmark_20261012/all_predictions.json)、[首轮报告](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Exit_choice三出口数据清洗与首轮实验_2026-10-12.md)，strict/primary | 初始Qwen没有真人例子，Logit使用其他日期标签；不是所有LLM的结论 |
| C04 | 偏差校准改善原始Qwen概率损失 | [校准指标](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_calibration_campaign_20261012/metrics.json)，primary/date_macro/raw_qwen与bias_calibration | 0.832789到0.710472，仍落后观测Logit；校准可主要修正出口偏好 |
| C05 | 加Qwen分数未建立稳健额外预测价值 | 同一指标，hybrid_qwen_logit、observables_logit、observables_nonlinear；[打乱检查](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_calibration_campaign_20261012/negative_control_summary.json) | 真实基础损失0.339956，仅较普通Logit低0.002953，落后非线性0.227353；打乱不是正式p值 |
| C06 | 同设置给予全部真人记录使合并准确率增加7.77个百分点 | [学习资料指标](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_supervised_context_20261012/metrics.json)，pooled/zero_same_settings及all_training_records | 同设置50.49%到58.25%；合并NLL变差，不能挑准确率独报 |
| C07 | 上下文实例效果因日期而异 | [逐日期配对表](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/context_per_date_comparison.json) | 两日期NLL改善、两日期退化；同楼四日期不能当作多建筑验证 |
| C08 | 本提示存在出口展示顺序敏感性 | [顺序检查](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_order_stability_20261012/summary.json)，12例/两channel | 哈希选例，24次调用；只检验此模型此提示和这12例，未偷偷重排得分 |
| C09 | 直接计数复述正确，而训练表重新计数不可靠 | [8题结果](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/counting_probe_details.json) | 4/4与0/4，不是行为预测样本；不唯一解释预测失败 |
| C10 | 本轮341次新调用输出合法 | [独立审核](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/evidence_audit.json)，total_new_local_llm_calls | 309学习资料、24顺序、8事实题；合法不保证语义、稳定性或真实性 |
| C11 | 114行指标独立重算一致，保护文件未变 | 同一审核，calibration_reconciliation/context_reconciliation/protected_source_files_unchanged | 不替代人工录像、伦理及论文审查；不能证明所有潜在设计偏差不存在 |
| C12 | 暂不能正式训练是否/何时改道模型 | [改道核查](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Exit_choice改道序列与工作人员编码核查_2026-10-12.md) | 15条全为1改2，缺保持者风险集与个体时间；静态103条不能自动当负例 |
| C13 | 新动态接入工具能实施指定时间与标签边界检查 | [代码](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/scripts/exit_choice_observation_contract.py)、[11项检查](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/observation_contract_self_check_final.json) | 仅合成检查，无新真人数据，无动态模型，无安全认证 |
| C14 | 旧控制证据尚未建立稳定LLM额外收益 | [旧研究证据](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/研究主线与证据一页说明_2026-10-11.md) | 不同轮次不合并；简化同布局模拟，不是新三出口地图控制结果 |
| C15 | 总体准确率改善伴随少数出口退化及部分过度自信错误 | [缓存错误分解](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/cached_error_decomposition.json)、[逐例配对](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_two_hour_session_20261012/context_casewise_changes.json) | 纠正15例、破坏7例，E2识别12/22降到7/22；结果后的描述性分析，没有据此改提示、筛案例或称独立确认 |
| C16 | 事件背景与本严格子集的选择分布不同，原论文样本口径尚未完全对齐 | [事件类型核查](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Exit_choice事件类型与信息可得性核查_2026-10-12.md)、[原论文](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/exit_choice_alternative_design_20261011/Tong_2025_library_exit_choices.pdf)，表2—3及3.2 | 误报63例全E1，演习40例含22个E2；事后类型不是自动可用输入；121条原样本与118个保留键未逐一对齐 |
| C17 | 提示意图为初次选择，但标签时点未逐人核实 | [冻结提示来源](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/scripts/exit_choice_initial_benchmark.py:439)、[字段审计](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Exit_choice三出口字段与清洗规则_2026-10-12.md) | 成绩只称记录选择预测；没有改变已保存提示、标签或得分，也不能保证标签都是首次选择 |

## 应删除或改写的说法

| 不合适的写法 | 改为 |
|---|---|
| LLM已经模拟了真人疏散，PPO验证了有效引导 | 目前分别做了记录选择预测和旧仿真控制；两条证据链尚未贯通 |
| 提供真人数据后模型可靠性显著提升 | 同设置准确率提高，但合并概率损失退化，跨日期效果不一致 |
| Logit证明人会从众、工作人员导致改道 | Logit拟合这些记录属性与选择的关联，Guide时点与引导因果效应未确认 |
| 逆序失败是数组位置映射错误，修正后成功 | 位置绑定假设只能解释部分不稳定；缓存重排是诊断，未作正式修复 |
| 341次结果代表341个人 | 341次为同一批开发数据上的重复请求与事实题，严格主记录仍为103 |
| 封存最终验证已通过 | 48个旧最终疏散场景本轮没有读取，新的真人最终评价没有执行 |
| 动态观察模板完成了新数据实验 | 只完成格式和校验代码，真实时间标注与模型实验尚未运行 |
| LLM技术在疏散中无用 | 当前Qwen3 8B使用方式在此数据与任务上未达到统计基线 |

## 写作和引用状态

报告可以在组会上讨论结果和限制。作为提交期刊的正文，仍需作者逐条核实数据来源、原论文定位、版本方法与所有声明。本表不捏造作者、作者顺序、伦理许可、数据授权范围或目标期刊政策。

当前最强的论文主线仍可整理为“有效性与失效机制”，真人行为验证作为明确分开的补充证据。是否改成主要的行为预测论文，应由新的独立事件证据和完整相关研究核查决定，不能把负结果换一个标题就当作新贡献已经成立。
