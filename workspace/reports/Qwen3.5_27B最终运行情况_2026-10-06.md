# Qwen3.5 27B分阶段完成情况

日期：2026-10-06（北京时间）。

状态：complete。共覆盖32/32条，其中符合数学目标24条。

使用同一27B Q4_K_M文本权重。第一阶段3条使用原定义，第二阶段13条使用无视觉组件的纯文本定义，第三阶段1条使用num_batch64逐条卸载；第四阶段剩余15条使用num_batch32、num_ctx2048逐条卸载。没有重复已交付题目。

纯文本模型名qwen35-27b-text-q4:latest；没有projector。去掉视觉组件不改变文本权重精度。第三阶段更改计算批次与驻留策略；第四阶段进一步将num_ctx设为2048、num_batch设为32。题目与schema未删减，temperature0、seed2026100601、num_predict1024和think=false保持原计划。

这是一套固定题目的分阶段覆盖，不是同一配置下的32题基准。计算批次、缓存和后台内存状态不同，不能将结果差异直接归因于视觉移除或参数规模。

第四阶段采样最低可用内存527.0MB；整机分页文件使用量最高2.27GB。

| 条件 | 已交付 | 符合目标 |
| --- | --- | --- |
| minimize_hazard | 8 | 8 |
| maximize_hazard | 8 | 4 |
| minimize_clear | 8 | 4 |
| maximize_clear | 8 | 8 |

各阶段使用独立原始响应、预算、内存采样、卸载记录和完整性台账。无付费API、拟合、疏散模拟或PPO更新。原8B、14B记录、默认模型选择与全局研究状态不变。

[第四阶段固定方案](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Qwen3.5_27B小上下文续测方案_2026-10-06.md)；[全阶段逐题结果](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/qwen35_27b_text_low_memory_20261006/all_phases_results.csv)。
