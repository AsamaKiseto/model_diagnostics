# model_diagnostics 本地规则

## 适用范围

适用于可整体迁移到其它 PyTorch 仓库的 `model_diagnostics/` 核心 package。

## 本地不变量

- 只允许依赖 Python 标准库、PyTorch 和本 package 的显式子模块，禁止 import `src.*` 或其它宿主模块。
- `base` 不拥有宿主数据集、checkpoint 布局、任务 role、样本分组、rollout horizon 或实验 cohort 默认值；宿主事实通过公开 contract、adapter 或 catalog 注入。
- `host_runtime` 只定义任务中立执行契约，不依赖 `base`、`extensions` 或
  `integration`；`integration` 可依赖 `host_runtime + base + extensions`。
- 宿主 task implementation 只实现 `host_runtime` provider/capability；任务选择与
  composition root 留在宿主仓库。顶层 `__init__.py` 不做 eager re-export。
- `base/artifacts` 拥有跨执行路径的标准库原语、writer、transaction 与 validator。
- `base/reporting` 独占统一报告与 HTML renderer；runtime、checkpoint 和 extensions 不复制报告实现。
- 当前生产 catalog 只保留 `checkpoint_sweep`、`final_module_influence`、
  `final_channel_influence`、`final_objective_conflict` 和 `final_rollout`；批量
  checkpoint 与 final-selected analyzer 必须互斥。
- artifact wire-format identity 与 Python import path 分离；package 迁移不得隐式改变 v2 digest 或 format identity。
- portable 回归只放在 `model_diagnostics/tests/`，并由本目录的 `pytest.ini` 独立收集；
  测试同样禁止 import `src.*` 或宿主 fixture。
- 通用说明、架构、contract 和 ADR 只放在 `model_diagnostics/docs/`；仓库级文档只保留
  package 索引，不复制诊断正文。
- `model_diagnostics/` 是独立 PEP 517 build root；wheel 只含 runtime，sdist 可携带
  portable tests/docs，但两者都不得打包宿主源码、cache、`__pycache__` 或 `.pyc`。
- 发行版本、artifact wire-format 和 analyzer descriptor version 独立演进；当前
  `LicenseRef-Proprietary` 不能被误写成开源授权。

## 升级条件

- 新增宿主 import、领域分支、任务默认值、跨子包依赖或顶层 re-export façade。
- 变更公开 contract、CLI、artifact schema/layout、transaction、report validation 或生命周期。
