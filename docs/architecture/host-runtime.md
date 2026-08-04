# Host Task Runtime

## 模块职责

`model_diagnostics.host_runtime` 定义训练、评估与运行后工具可共同使用的任务中立宿主执行层。它随
`model-diagnostics` 发行，但不依赖 diagnostics Base、extensions 或任何宿主代码。它把
一个任务组合为：

```text
RuntimeDescriptor
├── objective_executor_version
├── CheckpointObjectiveSpec
├── device
├── ModelProvider
├── CheckpointProvider
├── BatchProvider
├── ObjectiveExecutor
├── AttemptExecutor
└── CapabilityCatalog
```

普通任务只需提供模型、batch 与 objective 的正常运行实现；checkpoint 和高级
capability 按场景显式提供。观察或诊断机制消费相同 runtime 事实，不得在 adapter
中复制 loss、gradient accumulation、AMP 或 optimizer boundary。

## 文件归属

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | `RuntimeDescriptor`、`BatchEnvelope`、sample/checkpoint/failure ref、objective unit/execution 与 provider protocols |
| `composition.py` | provider/capability 组合，以及 live/checkpoint session 的资源所有权 |
| `execution.py` | 流式 training attempt、instrumentation、同步服务和 optimizer/scheduler/auxiliary 终态 |
| `registry.py` | 无 import-time discovery 的显式 `TaskDefinition`/`TaskRegistry` |
| `__init__.py` | 稳定公共 import 入口 |

## 核心执行契约

`ObjectiveExecutor.definition_version` 是正整数公共契约。`TaskRegistry` 在 runtime
创建时把它写入 `RuntimeDescriptor.objective_executor_version`，composition 再校验
descriptor 与实际绑定 executor 完全一致；缺失或不一致均 fail-fast，不允许退化为
`unspecified` identity。

checkpoint 分支的 objective defaults 与 execution precision 使用类型化
`CheckpointObjectiveSpec`，权重对应的
训练位置使用 `CheckpointState.training_provenance:
CheckpointTrainingProvenance | None`。opaque `metadata` 不承载这两项隐式协议，generic
consumer 不得从 metadata 猜 objective、device 或训练位置。
当前 provenance 只表达单个训练段内的 `local_completed_update / training_updates`
及其 objective 坐标；段间 checkpoint relay 不合并或继承全局训练进度。

`ObjectiveExecutor.execute()` 返回 `ObjectiveExecution`。其中 `units` 是单次消费的
lazy iterable：需要保持 forward 与 backward 邻接的实现必须在 iterator 消费时才做
forward，不得在 `execute()` 中预先物化全部 chunk。

每个 `ObjectiveUnit` 同时提供：

- `raw_total` 与实际 AMP scale 前的 `backward_total`；
- actual `objective_identity` 与 request 侧 `ObjectiveContext`；
- normalization divisor、loss cap、AMP scale 与 schedule provenance；
- ledger、terms、chunk/segment identity。

非标量 `backward_total` 必须显式携带同 shape/device/dtype 的 finite detached
`backward_cotangent` 与 `cotangent_identity`；runtime 只执行
`sum(backward_total * backward_cotangent)`，禁止隐式求和。

checkpoint consumer 必须使用 spec 声明的 precision，executor 同时返回 requested
precision、实际 parameter dtype、device 与 autocast provenance；不得把固定的 `fp32`
标签写到采用其它策略执行的分支。

`TrainingAttemptExecutor` 只执行 `backward_total`，不重新计算 cap 或 divisor。它按
microbatch 流式消费输入，支持 gradient accumulation、可选 scaler/clip、optimizer、
scheduler 和 commit 前 auxiliary steps。一个尚未 backward 的 nonfinite microbatch
会独立 rollback；若 chunk 已有 unit backward 后才失败，则整个 attempt 失败并清空
梯度。

`TrainingAttemptExecutor` 只反向传播 `ObjectiveExecutor` 返回的实际 backward
objective，不提供替换训练梯度的诊断或 research hook。多目标梯度关系只在最终
checkpoint 的隔离分析中计算，不写回训练 optimizer。

## 分布式边界

通用 executor 不 import 仓库 distributed runtime，也不直接调用
`torch.distributed`。宿主通过 `AttemptSynchronization` 提供：

- 覆盖 lazy forward 与 backward 的 unit context；
- 跨 rank objective finite 决策；
- manual gradient synchronization；
- 全局 post-unscale、pre-clip gradient summary。

`ObjectiveExecution.metadata` 可声明宿主的同步策略，例如 chunk execution 是否需要
manual gradient sync；具体解释归同步服务。单进程默认使用
`LocalAttemptSynchronization`。

## Session 所有权

- `HostTaskRuntime.bind_live_model()` 返回非 owning `ExecutionSession`；关闭它不释放
  训练模型。
- `HostTaskRuntime.open_session()` 通过 checkpoint provider 构造并恢复模型，返回
  owning `LoadedTaskSession`；restore 失败立即释放，`close()` 幂等。
- checkpoint locator 与序列化布局完全由 provider 解释。
- 若 batch selection/materialization 依赖 checkpoint model context，provider 额外实现
  `SessionAwareBatchProvider.activate_session()`；调用方必须在每次相应操作前显式激活
  session。多个 loaded checkpoint 可以并存，禁止依赖“最后加载的 session”。

## 失败事实边界

训练 runtime 只向 flight recorder 发送已经发生的 objective、梯度和 step outcome。
非有限梯度会在清理前按 parameter owner 定位。当前不提供 replay capability，也不保存
用于重放的大型 model/input/RNG attachment；失败事实只能用于排查方向，不能声称已复现
根因。

## Capability 与 registry

`CapabilityCatalog` 只保存显式 `CapabilityDescriptor + implementation` binding。
缺失 capability 由调用方 fail-fast 或 structured skip，不触发自动 import。每个
`TaskDefinition` 固定 definition id/version 以及 provider/capability 组合；
`TaskRegistry` 在构造期拒绝重复 id，不维护可变全局状态。

### Component 与逐输出评价

需要逐组件分析的 task 通过三项相互独立的 task-neutral capability 暴露事实：

- `ComponentCatalogCapability` 枚举当前 payload 的全部 `RuntimeInputRef` 与
  `RuntimeOutputRef`；
- `OutputEvaluationCapability` 从同一个 `ObjectiveUnit` 的输出一次性评价传入的完整
  output catalog，返回逐 output 的 scalar response、support、normalization 与
  `higher_is_better`；
- `OutputObjectiveCapability` 从 actual objective ledger 提取逐 output 的可微
  objective slice，并保留 objective identity、support、normalization、partition
  status 与 coverage。

`RuntimeComponentRef.component_id` 标识一次执行中的具体输入或输出位置，
`semantic_id` 连接语义相同但位置不同的 component；`display_label` 只用于展示，不参与
identity。input/output ref 另外保存 tensor 坐标，output ref 还保存 normalization
identity。runtime core 不解释这些坐标对应图像通道、token、状态变量还是其它模态。

catalog 必须完整、稳定且无重复；task 已知真实标签时不得退化成行号或 hash 别名。
逐输出评价消费已经完成的同一次 forward，不能按 output 数重复执行模型。逐输出
objective 必须来自实际 ledger；当非线性聚合、loss cap 或未分配 shared term 使
partition 无法闭合时，调用方只能跳过依赖 actual objective 的 gradient/Taylor/conflict
证据，不能用另写的 diagnostic loss 冒充。

具体任务的 label、tensor 轴、validity broadcasting 和 normalization 都属于宿主
capability。Host Runtime 只保存已物化的稳定 component identity 和对应 provenance，
不按图像、token、序列或其它任务形态分支。

跨 run 稳定的任务分析事实由具体 task 的只读 profile 拥有。task composition 在构造
`TaskDefinition`/capability catalog 时绑定 profile，并把 id/version/digest 写入
metadata/descriptor；runtime core 不解释 profile 内容。Generic diagnostics
不能运行期重配 capability；checkpoint sweep、final cohort 与 analyzer 集合均由固定
阶段契约决定。未执行的重型
capability 不激活 execution requirements；被选择时只调用 task-neutral、无参数、
幂等的 `ExecutionPreparationCapability.prepare_execution()`。

## 相邻层边界

- task/model implementation 拥有 payload、模型构造和 objective 数学。
- pipeline 拥有训练 stage 编排、control 到通用 request/options 的翻译和日志展示。
- runtime 拥有通用 attempt mechanics 与真实执行 provenance。
- diagnostics package 可实现 `AttemptInstrumentation`，但 runtime 不反向依赖它。
- rollout、输入组件和多目标 partition 等不是必选核心，只能作为显式 capability。

## External contract

公共入口是 `model_diagnostics.host_runtime`。主要对象为：

```text
TaskDefinition -> TaskRegistry -> HostTaskRuntime
HostTaskRuntime + ModelHandle -> ExecutionSession
BatchEnvelope + ObjectiveContext -> ObjectiveExecution
TrainingAttemptRequest -> TrainingAttemptResult
```

schema 或 weight-boundary 语义变化必须提升相应 definition/version，并同步本文件。

## 最高风险

1. objective provider 与 training pipeline 各自实施 cap/divisor，形成两个数学事实源；
2. 预先 materialize chunk 或 microbatch，改变显存峰值、DDP no-sync 范围或 RNG 顺序；
3. 分布式宿主误用 local synchronization，造成 rank decision 或 grad norm 不一致；
4. optimizer 已部分更新后把失败错误记录为 `optimizer_step_applied=False`；
5. 把任务字段、artifact schema 或 analyzer default 推入本目录，使 runtime 退化为新
   God Object。

## 最小验证

```bash
pytest -q -c pytest.ini tests/host_runtime
```

测试使用普通 `nn.Linear`，覆盖流式 accumulation、finite/nonfinite microbatch、
objective provenance、显式 registry、session 所有权及 import dependency boundary。
