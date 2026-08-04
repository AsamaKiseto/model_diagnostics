# ADR-0013：三阶段低开销诊断

## 状态

Accepted

## 决策

诊断执行划分为训练期 flight recorder、逐 checkpoint anomaly sweep 和单一最终
checkpoint influence analysis。三者执行路径互斥。

批量 sweep 只允许一次 baseline forward/backward、同一次 backward 的逐输出直接梯度、
参数叶子层 gradient health 和 Activation/Norm 分布；固定 cohort 为 32 个 test
window 且至少覆盖 8 个 group。
所有会随输入数、模块数、输出数或 horizon 成乘法增长的分析移到 `final_selected`。
final 固定使用 8 个 test window 且覆盖 8 个不同 group。该阶段用于给出排查方向，
不承担逐样本审计或总体效应的精确区间估计；input×output artifact 只保存 cohort
充分统计，不保存逐样本笛卡尔积。
模块分析把训练 objective local Taylor 与评估输出干预分开：每个样本一次真实 backward，
干预和 8-group mean reference 只做 response-only forward，不报告干预后的训练 loss。
最终通道分析沿宿主 component catalog 声明的轴逐项展开，不把整个
component group 当作通道。最终模块分析从通用 PyTorch 结构生成 Stage/Block 候选，并只在固定
final cohort 的 3–8 次成功 forward 上确认真实执行路径；训练期与 checkpoint sweep
仍不注册 hierarchy hook。宿主 module site 只作为自动结果的增量 override。

删除当前生产 catalog 中的 representation、sequence memory、2×2 feedback、
conditional donor、optimizer transform、fresh optimization branch 和 failure replay。

## 原因

逐 checkpoint 重复运行影响与关联分析不能帮助监测训练异常，却使 100 个 checkpoint
的代价扩大到数百小时。阶段互斥并使用固定执行策略，比靠用户维护一组 recipe 更可靠。
2026-07-30 的运行门槛复核进一步发现，64-sample final 会让单 checkpoint 模块条件和
input×output 序列化失去交互式诊断价值，因此收紧为 8 个不同 group，并明确把精确总体
估计留给独立研究实验。

## 影响

训练过程不再给出模块级机制曲线；逐 checkpoint 不再给出通道/模块影响。影响与关联只
描述最终选定模型。训练和 sweep 的纵向证据分别以 `update`、`checkpoint_update` 为固定
横轴，不跨时间聚合；单个时间位置绘制单点，不按数据量降级成其它图形。
所有 replacement、local Taylor、gradient geometry 和 rollout 结论仍不得解释为物理
因果。
