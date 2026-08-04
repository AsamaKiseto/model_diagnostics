# 通用诊断可视化参考

本目录仅保留旧 dashboard 的颜色、排版和纯展示样式，供新 renderer 设计时人工参考。

它不属于 active diagnostics runtime：训练、评估、`model_diagnostics.base`、
`model_diagnostics.extensions`、adapter、CLI 和报告生成器都不得 import、读取或调用这里的内容。
这里不保存旧 checkpoint loader、指标计算、旧 JSONL reader、领域执行器或兼容逻辑，
也不承诺示例样式可独立执行。

参考文件：

- `dashboard-reference.css`：light/dark palette、状态色、卡片和表格布局。
- `matplotlib_reference.py`：ranked bar、trend line、网格和紧凑图例样式。
