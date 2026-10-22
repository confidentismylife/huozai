# Qwen3.5 27B Q4_K_M本机试运行报告

日期：2026-10-06（北京时间）。

状态：stopped_after_failure。本地聊天请求开始3次，成功交付3/32。

模型：qwen3.5:27b；实际参数量26,895,998,464；量化Q4_K_M；Ollama 0.35.1。

清单digest：`64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59`。使用官方GGUF包，文本权重16,547,399,008字节，视觉组件931,145,984字节；仅文本测试。

本机为Ryzen 7 8845HS、24GB内存、Radeon 780M核显。运行设备、加载大小和上下文以loaded_model_snapshot.json为准。

题目、消息、JSON schema、temperature0、seed2026100601、1024输出预算、4096上下文与旧轮次保持一致；只更改请求model。Qwen3.5的架构、训练、模板和内置参数可能不同，不能归因于纯参数量效果。最大化条件只是数学诊断，不是疏散建议。

采样最低可用内存0.02GB；最大系统分页使用量4.61GB；Ollama进程及子进程RSS之和最大16.68GB。RSS可能重复计算共享页，采样不保证精确峰值；系统分页使用量不等于全部由模型引起。

本次停止原因：

```text
Available memory below 1GiB; stop remaining requests.
```

该结果只说明本次服务版本、模型文件、运行设置和当时资源条件下的尝试结果。未交付的题目不记为语义答错，也不输出伪造的全套正确率。

旧实验、默认模型选择、全局研究状态和保留疏散场景保持原字节。无付费API、模拟、拟合或PPO更新。

[固定方案](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/reports/Qwen3.5_27B本机试运行固定方案_2026-10-06.md)；[原始记录](C:/Users/摩天轮大盗/Documents/ChatGPT/LLM探索/outputs/qwen35_27b_trial_20261006)。
