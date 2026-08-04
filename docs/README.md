# Model Diagnostics 文档

当前诊断系统采用三个互斥阶段：

1. 训练过程监测：宿主 `AttemptExecutor` 通过低开销 flight recorder 记录真实
   objective、逐输出 objective、梯度有限性和 update 终态。
2. 逐检查点异常监测：`checkpoint_sweep` 在固定小 cohort 上对每个 checkpoint
   只执行一次 baseline forward/backward，并监测逐输出响应/直接输出梯度、参数叶子层
   梯度以及 Activation/Norm 分布。
3. 最终模型分析：`final_selected` 强制只选择一个 checkpoint，才允许运行输入通道、
   自定义模块、输出目标关系和 free rollout 分析。

现行设计见：

- [架构总览](architecture/overview.md)
- [Host Runtime 契约](architecture/host-runtime.md)
- [训练期运行时](architecture/training-runtime.md)
- [检查点引擎](architecture/checkpoint-engine.md)
- [保留的 extensions](architecture/extensions.md)
- [使用手册](handbook.md)
- [三阶段决策](adr/ADR-0013-three-stage-diagnostics.md)

所有可视化结果只用于人工判断方向。局部 Taylor、replacement、梯度几何和 rollout
均不构成物理因果证据。
