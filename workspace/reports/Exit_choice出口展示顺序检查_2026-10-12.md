# 出口展示顺序的配对检查

2026年10月12日。每个日期按固定编号哈希选3例，12个既有真人案例，两种信息条件，共24次新本地调用。

只把E1,E2,E3的展示顺序改成E3,E2,E1；物理身份、属性、原始提示、模型、采样参数、schema和种子保持一致。选择案例时没有使用标签或模型错误。

| 条件 | 有效配对 | 平均概率总变差 | 最大概率出口改变 | 原顺序正确 | 逆序正确 |
|---|---:|---:|---:|---:|---:|
| primary | 12 | 27.08% | 10 | 6 | 1 |
| guide_conditional | 12 | 31.25% | 9 | 8 | 0 |

这是12个案例的表达一致性检查，不是新增独立人类样本，也不根据结果修改提示或选取更有利的顺序。输出合法不能证明出口语义或概率稳定。

[相关方法依据：Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions，NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.130/)。本轮仅报告本模型与本样本的结果。
