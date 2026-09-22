# 三阶段诊断手册

## 训练过程

训练会自动启用 flight recorder。训练页分别显示 `objective_terms` 中的 `Loss Components`、
`Per-Output Loss`、实际 backward 的整体目标和 `Global Gradient Norm`，用来查看目标项或输出
通道是否长期失衡、整体梯度是否突变或非有限、microbatch 是否被拒绝。高密度训练图只
保留每条曲线，不生成逐 update 点详情，因此不会因数千个交互点拖慢页面。

训练页所有功能图固定使用同一套全宽大图布局。任意图放大后仍保留主图的悬停规则；tooltip
位于放大 dialog 内部，因此不会被 modal top layer 遮挡。技术标题使用稳定的领域术语，
例如 `Loss Components`、`Per-Output Loss`、`Global Gradient Norm`、`Raw Training Loss`
和 `Backward Training Loss`；标题只清理多余空白，不再按单词逐项翻译。

`objective_terms` 保留宿主 ledger 的原始名称；其中的 loss、权重或其它调度量不会由通用
报告依据字符串重新分类。最新 artifact 中的逐项 loss 都会显示，权重项也以原名保留，避免
静默隐藏训练事实。

全局梯度范数属于实际 backward 总目标，不能从标量 loss 数值拆成逐 loss 梯度范数。
flight recorder 不为每个命名 loss 追加 `autograd.grad`；需要逐输出直接梯度时，在逐检查点
页面查看 `output_gradient_rms`。该阶段不用于比较模块重要性。

## 逐检查点

扫描宿主 checkpoint provider 声明的全部训练检查点：

```bash
python -m my_project.diagnostics checkpoint-sweep --run-dir <run_dir>
```

固定 cohort 为按 group 确定性分层的 32 个 test window，至少覆盖 8 个 group；不满足时
整项标记证据不足。同一个 cohort 用于全部 checkpoint。图中每个输出通道的响应和直接输出梯度、
参数叶子层梯度或 Activation/Norm 观测点各自形成一条随 checkpoint update 变化的线。
每个 checkpoint 的训练 provenance 提供横轴；不能跨 checkpoint 合并为一个总体分布。
非有限值以状态记录，不会中止后续 checkpoint。

## 最终模型

对已选定的唯一 checkpoint 运行：

```bash
python -m my_project.diagnostics final \
  --run-dir <run_dir> \
  --checkpoint <checkpoint>
```

final 固定使用 8 个 test window、覆盖 8 个不同 group，并运行 catalog 中全部 final
analyzer。模块分析先按通用 PyTorch 结构生成 Stage/Block 候选，再在 3–8 次成功
final-cohort forward 上确认实际执行路径；报告同时列出 coverage gap。宿主显式 module
site 只补充自动结构无法表达的边界，不关闭自动发现。
输入与输出 catalog 沿宿主 `RuntimeComponentRef` 声明的 tensor 轴逐项枚举；
有语义标签时显示标签，否则显示 component group 与稳定 index，不会把
整个 component group 合并成一个通道。输入分析覆盖 capability 声明的全部
model-visible input，不在诊断侧猜测任务语义。
输入×输出分析在这 8 个 group 上计算逐样本配对响应，但 artifact 只保存每个
input × output × method 的 cohort 均值、最小值、最大值、support 与缺失计数，不保存
逐样本笛卡尔积。

最终响应类分析固定使用 `eval()` mode。parameter 在 response-only 分支中保持只读，逐分支
恢复 registered buffers，整个 analyzer 退出时完整恢复模型状态。输入影响先在同一
checkpoint、sample 和随机流下重复一次未改变的 baseline，按输出通道汇总为恒等对照；mean replacement 才按每个输入
通道执行。报告不会把恒等对照显示成一种可筛选干预，而是在每行实际干预旁显示对照均值
和范围。support、改动元素、样本覆盖和未干预路径核对仍保留在原始 artifact，只有异常时
进入折叠有效性表。

`Stage/Block Influence` 不是单一模块结果。报告先显示全部已确认模块的覆盖数量，并提供
`Stage` / `Block` 层级筛选和具体模块筛选；主表、缺失核对表及折叠色块矩阵共享同一筛选。
每个实际干预记录仍保留明确的 `hierarchy_level`、`module_path` 和输出通道身份。

输入扰动敏感度使用 task profile 固定的多个模型输入坐标幅度。对同一个
`sample × input`，全部尺度复用同一 Rademacher 方向并执行正负配对；只有两侧都成功的
样本才进入 `S_ij(s)`。报告的尺度表格保留全部 input×output 组合，选择表格单元格或
输入/输出下拉框可查看幅度—对称相对输出响应曲线。每个方向先用扰动前后输出自身的
RMS 计算 `2×RMS(y_d-y_0)/(RMS(y_d)+RMS(y_0))`，再对正负方向和 cohort 做等样本 RMS。
该指标范围为 0–2，不使用训练集标准差、逐 shot 特征或输入扰动 RMS 作为分母；曲线
近似水平只表示相对响应在该尺度范围内接近不变，不能据此推断物理因果。

报告按“训练过程监测 / 逐检查点异常监测 / 最终模型分析”分栏。检查点图按对象显示从
update 0 开始的趋势；输入/模块影响以表格为主、折叠色块矩阵为补充；自由滚动按输出通道
和 horizon 分开显示。原始数据页用于核对 support、identity、commit 与 claim boundary。
