# 架构总览

任务中立层全部随 `model-diagnostics` 发行，依赖方向为：

```text
model_diagnostics.host_runtime
          ↑                         ↑
host task implementation    model_diagnostics.integration
                                      ↓
model_diagnostics.base ← model_diagnostics.extensions
```

`host_runtime` 只依赖标准库和 PyTorch，定义 model/checkpoint/batch/objective
provider、session、attempt executor 和可选 capability。它不 import diagnostics 算法，
因此训练、评估和诊断可共用同一 objective 与 rollout 事实源。

`base` 只依赖标准库和 PyTorch，拥有 checkpoint condition isolation、通用 module
intervention、artifact/validator 和统一报告。`integration` 把 Host Runtime DTO 翻译为
Base checkpoint protocol，并按显式 capability 组合 analyzer；它不选择具体任务。
`extensions` 只保留最终
checkpoint 使用的三类算法：

- `input_dependence`：identity/mean replacement 的输入×输出响应，以及多尺度正负配对
  扰动的对称相对输出响应；
- `multi_objective`：输出目标与其余目标的整体梯度 cosine、norm ratio 和分区校验；
- `rollout`：显式 free rollout trace 的误差与局部失稳统计。

宿主 composition 只从自身 task catalog 选择 `HostTaskRuntime`，再调用 portable
`create_checkpoint_adapter()`。训练路径只使用 flight recorder；checkpoint
sweep 与 final analyzer 共用宿主真实 objective，不复制 loss 或 rollout。

外部仓库不需要重写 recipe、checkpoint engine、artifact 事务或 report 流程：
它只实现 providers/capabilities，用 `create_checkpoint_adapter()` 组合 runtime，
再将 factory 交给 `run_diagnostics_cli()`。

三阶段是执行边界，不是展示过滤器。`checkpoint_sweep` recipe 无法选择影响 analyzer；
`final_selected` recipe 无法选择 sweep，且 checkpoint 数必须等于一。
