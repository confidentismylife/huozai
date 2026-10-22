# Qwen3.5 27B本机试运行结果说明

日期：2026-10-06（北京时间）。

27B Q4_K_M已下载到D盘，实际成功加载并返回3条正确答案。第三条完成后，下一条请求开始前可用内存低于预设1GiB门槛，资源保护停止了后续测试。随后已卸载模型。

这次结果是“可以勉强运行”，尚不能认定适合长期批量实验；没有完成整套32题，也不能判断它是否比8B或14B整体更准确。

实际模型26,895,998,464参数，GGUF Q4_K_M；Ollama报告加载大小17.37GB，CPU推理（size_vram=0），上下文4096。

采样最低可用内存约20.5MB；系统分页文件使用量最高约4.61GB。这是整机观测，不把分页使用量全部归因于模型。采样不保证精确峰值。

| 已交付题目 | 条件 | 参考方向 | 实际方向 | 耗时秒 |
| --- | --- | --- | --- | --- |
| GR03_north_minimize_hazard | minimize_hazard | south | south | 58.25 |
| GR03_south_minimize_hazard | minimize_hazard | north | north | 25.88 |
| GR03_south_maximize_clear | maximize_clear | south | south | 26.27 |

首条58.25秒包含加载，其后两条分别25.88和26.27秒；三条各生成22个token。仅包含两条minimize_hazard和一条maximize_clear，没有覆盖全部目标与角色条件。未执行的29条不算答错。

提示、schema、顺序、temperature0、seed2026100601、1024输出预算与4096上下文保持原计划，仅model字段变化。模型代际、架构、训练与模板不同，不能作纯参数量因果比较。无模拟、PPO更新、付费API或额外推理。

原试运行记录与报告已经封存，本说明作为单独的补充记录；旧8B、14B台账、全局研究状态和默认模型选择均已核验未改。

如果继续测试27B，应先释放更多后台内存，再另行冻结从第4条起的续测记录；不能降低已声明的内存门槛来强行扩大运行。当前配置下，日常批量实验继续使用已验证的较小模型更合适。

[原始试运行报告](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Qwen3.5_27B本机试运行报告_2026-10-06.md)；[独立核验记录](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/qwen35_27b_partial_closure_20261006/audit.json)。
