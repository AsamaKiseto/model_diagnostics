# 检查点引擎

## `checkpoint_sweep`

用于大量 checkpoint 的训练异常监测。固定 cohort 为 32 个 test window，并要求至少
覆盖 8 个 group；支持不足时整项标记 `insufficient_evidence`。每个样本只执行一次
baseline forward/backward：

- 每个输出通道的验证响应、直接输出梯度 RMS/零值/非有限比例与 support；
- raw/backward objective 和有限性；
- 模型整体与参数叶子层的 gradient health；
- 激活函数和归一化层的 input/output 分布。

sweep 的逐输出梯度来自同一次 total-objective backward 对预测张量保留的梯度，不增加
反向次数。sweep 不运行任何输入、模块、表示、序列记忆、Oracle、rollout 或 optimizer
模拟。

## `final_selected`

强制只选择一个 checkpoint。Base 先从通用 `nn.Module` 结构生成
`All → Model → Stage → Block` 候选，再在固定 final cohort 的 3–8 次成功 forward 上
确认真实调用并冻结非重叠 Block frontier。Linear、Conv、Norm、Activation、Dropout、
Embedding 与纯容器不自动成为 Block；宿主 `ModuleSite` 只作为结构无法表达边界的增量
override。hierarchy catalog、coverage gap 和调用确认状态进入 cohort/analysis identity，
因此站点变化不会错误复用旧 analysis。每个样本共享一次 baseline backward；condition
分支只 forward。默认条件为：

- `identity`；
- `output_scale=0`；
- `mean_patch`。

每个样本只运行一次真实训练 objective backward，用于所有确认模块的 local Taylor；
输出干预和 8-group training mean reference 只运行 response-only forward，不生成或比较
训练 loss delta。

最终阶段固定执行 capability 可支持的 `final_channel_influence`、
`final_objective_conflict` 和 `final_rollout`，不提供 analyzer 选择参数。固定 cohort
为 8 个 test window，并要求覆盖 8 个不同 group。input×output 结果按 cohort 充分统计
收口为每个组合一行，避免持久化逐样本笛卡尔积。这些结果分别表示预测依赖、局部
优化关系和自由滚动表现，不表示物理因果。
