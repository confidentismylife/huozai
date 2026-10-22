# Qwen3.5 27B Q4_K_M纯文本续测报告

日期：2026-10-06（北京时间）。

状态：stopped_after_failure。本次交付13/29条，连同原批次3条，共覆盖16/32条，其中符合数学目标12条。

这是两阶段覆盖结果：前3条使用原模型定义，后续使用纯文本定义；不能冒充同一纯文本配置下32条复测。

纯文本模型名：qwen35-27b-text-q4:latest。无projector，仍使用同一份16,547,399,008字节的27B Q4_K_M文本权重。模板、参数映射、renderer、parser和thinking设置与原模型相同。

本次从第4条开始，无重复前三条，无自动推理重试。消息、schema、temperature0、seed2026100601、num_predict1024、num_ctx4096、think=false保持原计划。

Ollama报告本次加载大小17.11GB，原定义为17.37GB；本次size_vram=0。

加载大小是服务报告值。原纯文字请求未必加载视觉编码器，后台占用和缓存也不同，不把运行内存差异直接归因于去掉视觉组件。

采样最低可用内存26.3MB，系统分页文件使用量最高3.09GB。

每次请求间保留1GiB可用内存门槛；一秒采样不保证捕获精确峰值。分页使用量为整机数值，不能全部归因于模型。

停止原因：

```text
Available memory below 1GiB; stop remaining requests.
```

| 条件 | 两阶段交付数 | 符合目标数 |
| --- | --- | --- |
| minimize_hazard | 4 | 4 |
| maximize_hazard | 4 | 2 |
| minimize_clear | 4 | 2 |
| maximize_clear | 4 | 4 |

8个固定案例来自4个镜像家庭；数学标签比较不是实际疏散控制或伤亡结果，不能据此扩大PPO训练。原报告、模型、默认研究选择与全局状态保持不变。

[固定方案](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Qwen3.5_27B纯文本续测方案_2026-10-06.md)；[逐题记录](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/qwen35_27b_text_continuation_20261006/two_phase_coverage.csv)。
