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

宿主可在 composition root 对 `GenericCheckpointAdapter` 显式启用
`reuse_compatible_checkpoint_session`。启用后，adapter 只在同一次
`checkpoint_sweep` 内、相邻 checkpoint 的 `ModelSpec` 完全相等时复用已构造模型和
宿主 data runtime；每个 checkpoint 仍重新执行 `CheckpointProvider.load()` 与严格
`restore()`，重新绑定 checkpoint state/training provenance，清空 parameter gradients，
并把非持久 registered buffers 恢复到首次构造基线。结构 identity 不同则关闭旧 session
并回到完整构造；registered module hook catalog 漂移时同样重建，防止 analyzer hook 泄漏
到下一个 checkpoint。restore 失败立即释放当前 session。`final_selected` 和默认未启用
路径保持每个 loaded model 独立关闭；实际复用 session 统一由 adapter `close()` 收口。

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
训练 loss delta。所有 response-only baseline、恒等对照和实际干预固定在 `eval()` mode
下运行，并从同一 canonical checkpoint state 与配对随机流开始。response-only 分支不
执行 backward/optimizer，parameter 保持只读；每个分支前只恢复 registered buffers，
analyzer 退出时再完整恢复一次 parameter/buffer state，避免为数千个分支重复加载全部
权重。Taylor backward 仍使用真实 checkpoint objective 的 module mode。

最终阶段固定执行 capability 可支持的 `final_channel_influence`、
`final_input_sensitivity`、`final_objective_conflict` 和 `final_rollout`，不提供
analyzer 选择参数。输入敏感度在同一 final cohort 上使用 task profile 声明的 scale
和正负配对确定性扰动，只保存 input×output×scale 的对称相对输出响应。固定 cohort
为 8 个 test window，并要求覆盖 8 个不同 group。输入恒等对照与输入通道无关，因此
每个样本只重复执行一次 baseline，并按输出通道聚合；mean replacement 仍按
input×output 收口为每个组合一行。这样避免把同一对照复制成输入笛卡尔积。这些结果分别表示预测依赖、局部
优化关系和自由滚动表现，不表示物理因果。
