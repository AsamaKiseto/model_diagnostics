# ADR-0008 通用稳定窗口确认 Block 与分布 Tap 分层

## 状态

Superseded

ADR-0013 已删除训练期 hierarchy discovery。当前逐 checkpoint 仅对
Activation/Norm 建立分布 Tap；final-selected 在固定 cohort 上用 3–8 次成功 forward
确认静态 Stage/Block 候选，不恢复本文的训练期 controller、distribution Tap 分层或可调
plan。下文仍只保留被替代的训练期设计。

## 背景

仅依赖静态 PyTorch 结构推断会把某些参数容器误认为可执行 Block，也可能错过由父模块
直接调用的更深子模块。单个 update 的调用轨迹又可能遗漏 curriculum、条件分支或
activation checkpointing 下较晚出现的路径。逐模型维护路径名单会让组件替换与诊断配置
耦合。另一方面，Norm 与 Activation 的输入输出分布对定位特征消失、死亡和饱和很重要，
但不需要承担 Block 级 CUDA timing、参数梯度扫描或长期 Tensor 保留成本。

## 决策

- `All → Model → Stage → Block` 只有一种自动发现行为：先生成完整的 registered-module 候选，
  再在多个成功 logical update 的轻量 forward 调用轨迹上选择非重叠、真实执行的
  Block frontier。跨 rank 使用调用并集。
- 默认 `discovery_min_updates=3`、`discovery_max_updates=8`、
  `stability_patience=3`。满足 minimum 后，跨 rank frontier 连续三次不变即可提前冻结；
  否则第八个成功 update 强制冻结。失败 attempt 不推进发现窗口。
- 未调用的候选从最终 hierarchy 移除；父候选未调用时可下钻到实际调用的子候选，深层
  候选未调用但父 wrapper 实际执行时可回退到父边界。没有可靠 Block 的 Stage 保留
  Stage 级监控并写入 `coverage_gaps`。
- `HierarchyDiscoveryPolicy` 只定义发现稳定窗口，不限制层深或 Block 数。
  `HierarchyOverride` 只用于极晚条件分支或结构无法表达的少数边界，并始终是自动结果
  的增量补充；发现 hooks 不会为等待极晚分支而无限保留。
- hierarchy 对每个候选记录 first/last observed update、called rank count、alias paths、
  repeated invocation count、coverage gap 和 late/unobserved override 状态。
- Norm/Activation 使用独立 distribution Tap，不进入参数 hierarchy。Tap 通过
  `owner_node_id` 关联最终最深 Model/Stage/Block。
- Tap 在 forward pre/post hook 内对完整 Tensor 计算 device-scalar sufficient
  statistics，不跨 logical update 保留完整输入输出 Tensor，也不采集参数梯度、local Taylor、
  backward timing 或 CUDA memory。
- 在线 artifact 写入 `diagnostics/v2/online/sessions/<session_id>/`。setup 阶段不把
  静态猜测当成最终 hierarchy；稳定窗口收口后写入 session 内唯一 `hierarchy.json`，
  其中包含发现窗口、`coverage_gaps`、采样策略和 `distribution_taps`。
- checkpoint analyzer 只读取同一 v2 contract；没有可验证的运行时 hierarchy 时可以调用
  同一静态 discoverer，但必须标记来源，condition 执行期对不可调用目标作 structured
  skip。旧 `diagnostics/online/` 与 `diagnostics/checkpoint_conditioned/` 不参与发现或
  fallback。

## 影响

- 普通模型和可替换组件无需维护诊断 module path 名单；标准 `nn.Module.forward()` 组合
  即可参与调用确认。发现窗口增加少量低成本 call-count hook 生命周期，但降低了单一
  update 条件分支造成的永久漏报。
- 直接存在于 Python 局部变量、无法由 `named_modules()` 定位的 Tensor 仍不是可 hook
  边界；需要长期监控时应将其抽成稳定命名的语义子模块。
- 报告分别展示 Block 指标、Stage coverage gap 和 Tap 分布，并按
  `owner_node_id + update` 关联。
- 完整 Tensor、optimizer、scheduler、scaler、RNG 和 loader cursor 都不进入诊断产物；
  checkpoint-conditioned 分析不表示原始 AdamW 轨迹复现。
