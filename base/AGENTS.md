# model_diagnostics.base 本地规则

## 适用范围

适用于只依赖 Python 标准库与 PyTorch 的任务无关诊断基础设施。

## 本地不变量

- 禁止 import `model_diagnostics.extensions`、`src` 或任何宿主 integration。
- 禁止假设序列、history、自回归、rollout、多 objective、必需样本分组或
  checkpoint 目录布局；通用 DTO 可携带调用方显式提供的可选 `group_id`。
- analyzer catalog 只包含 Base analyzer；extension 与宿主通过显式组合扩展 catalog。
- artifact wire identity 与 Python package 路径分离。
- runtime memory 只表示 CPU/CUDA 运行资源，不表示序列记忆。

## 升级条件

- 新增任务术语、隐式 analyzer 注册、宿主默认值或 Base 到 extension 的依赖。
- 变更 flight recorder 事件、checkpoint branch isolation、artifact transaction 或 report contract。
