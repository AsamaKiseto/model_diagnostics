# Portable Checkpoint Integration 本地规则

## 范围

本目录拥有任务中立 Host Runtime 到 checkpoint engine 的 DTO 翻译、analyzer binding
组合和逐 component 证据配对。

## 不变量

- 只依赖 `model_diagnostics.host_runtime`、Base、extensions、标准库和 PyTorch。
- 不 import `src`，不解释 checkpoint 文件名、宿主 batch、任务通道或 rollout 语义。
- `GenericCheckpointAdapter` 只委托 providers/capabilities，不复制 objective、数据装配或
  rollout。
- 缺失 optional capability 时只省略对应 analyzer，不生成伪证据。

## 升级条件

- 需要读取宿主 control、模型类型或数据对象时，必须把该逻辑留在宿主 composition。
