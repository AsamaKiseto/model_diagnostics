# Portable Host Task Runtime 本地规则

## Scope

本目录拥有统一宿主运行时的任务中立 contracts、显式 provider/capability composition、
训练 attempt mechanics 与 registry。它作为独立发行包中的任务中立运行边界服务训练、
评估和诊断，但不依赖 diagnostics Base/extensions，也不解释具体任务 payload。

## Invariants

- 只依赖 Python 标准库与 PyTorch；不得 import `model_diagnostics` 或仓库任务类型。
- contract 不假设时序、自回归、多目标、特定数据模态、特定 checkpoint 布局或任务字段。
- objective 数学只来自 `ObjectiveExecutor`；`TrainingAttemptExecutor` 不重建 loss。
- objective executor 必须有稳定正整数版本；checkpoint objective defaults、device 与
  training provenance 使用类型化字段，不得由 consumer 解释 opaque metadata。
- `ObjectiveExecution.units` 必须支持流式 `forward -> backward`，不得为统计预先物化。
- 分布式 finite、gradient norm、no-sync 和 manual sync 只通过
  `AttemptSynchronization` 实现，不直接调用全局 distributed runtime。
- registry 与 capability 均显式组合，不允许 import-time discovery 或可变全局注册。
- live session 不接管训练模型；checkpoint session 必须恢复失败即清理、关闭幂等。

## Escalation

- provider ownership、attempt weight boundary、objective provenance 或 capability contract
  发生变化时，同步 `model_diagnostics/docs/architecture/host-runtime.md`。
- 如需加入任务名词、训练 control 解析、artifact 写入或诊断 analyzer，停止并把职责留给
  task implementation、pipeline 或 integration 层。
