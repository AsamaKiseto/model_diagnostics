# 训练期运行时

训练期只保留 checkpoint 无法可靠重建的事实：

- raw/backward objective、normalization、loss cap；
- actual ledger 中的逐输出 objective、support 和覆盖状态；
- `AttemptExecutor` 已计算的全局梯度范数与有限性；
- microbatch commit/rollback、optimizer/GradScaler/scheduler/auxiliary step 终态；
- 非有限梯度出现时、`zero_grad()` 之前的参数 owner 定位；
- checkpoint milestone 和失败事实 bundle。

训练固定使用 flight recorder，不提供关闭或 profile 选择。flight recorder 不注册
module hook、不发起额外 DDP collective、不重复 forward/backward，也不重新计算全局
gradient norm。训练期不计算模块 local Taylor、模块干预、输入 replacement、目标梯度
几何或 rollout。

失败 bundle 仅保存已经发生的事实，不提供 replay CLI，也不声称可重放根因。

Stage/Block 调用确认不在训练期执行。`final_selected` 加载 model-only checkpoint 后，
才在固定 final cohort 的 3–8 次成功 forward 上确认通用结构候选；该步骤不 backward、
不更新参数，也不改变 flight recorder 的低开销契约。
