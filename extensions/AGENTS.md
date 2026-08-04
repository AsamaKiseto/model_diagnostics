# model_diagnostics.extensions 本地规则

## 适用范围

适用于在任务无关 Base 之上按需组合的高级诊断算法。

## 本地不变量

- extension 只允许依赖 Python 标准库、PyTorch 与 `model_diagnostics.base`。
- extension 之间默认不互相 import；确有共享需要时必须通过显式、最小依赖表达。
- 只能 import `model_diagnostics.base`、Python 标准库和 PyTorch；禁止 import
  `src`、宿主 integration 或其它已废弃 façade。
- 禁止内置数据 role、设备、实验分组、checkpoint 布局、rollout horizon 或其它宿主默认值。
- 每个 package 通过 `ANALYZER_DEFINITIONS` 和 `ANALYZER_CATALOG` 暴露本地 catalog；
  不执行全局注册，不修改 Base catalog。
- probe、gradient、permutation、factorial response 与 rollout 指标不得声明物理因果。

## 升级条件

- 新增跨 extension import、宿主术语、隐式注册、领域默认值或 Base 之外的依赖。
- 修改 extension 的公开数学口径、证据边界或 catalog descriptor。
