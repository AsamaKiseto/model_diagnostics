# ADR-0011 Portable model diagnostics package 边界

## 状态

Accepted

## 背景

训练期诊断和 checkpoint-conditioned 诊断容易把调用方的数据类型、文件布局、实验默认值
与通用 PyTorch hook、autograd 和 artifact 基础设施混在一起，导致 package 无法迁移，
新调用方也必须复制大量执行代码。

## 决策

使用单一 `model_diagnostics` package，并固定依赖方向：

```text
base ← extensions ← host integration
```

- Base 只依赖 Python 标准库和 PyTorch；
- Base 不假设输入形状、执行场景、多 objective、必需样本分组或 checkpoint 布局；
- extension 只依赖 Base、Python 标准库和 PyTorch；
- extension 通过显式 descriptor catalog 与 analyzer binding catalog 组合，不执行全局
  注册或名称分发；
- host integration 提供 checkpoint 发现、样本物化、objective、scenario 和 donor 语义；
- checkpoint adapter 只有 catalog、configure、checkpoint/model/cohort/sample/objective
  和 close 七项必需边界，其余能力按 analyzer 显式提供；
- 顶层 package 不提供跨层 re-export façade；
- CLI 归实际 owner，并通过动态 adapter factory 接入调用方；
- package identity 与 `diagnostics/v2` wire identity 分离；
- 仅移动 Python package、测试或文档不升级 artifact schema。
- portable tests 与文档分别只放在 `model_diagnostics/tests/` 和
  `model_diagnostics/docs/`，并在阻断宿主 import 的环境中运行。
- `model_diagnostics/` 本身是 PEP 517 build root；复制该目录无需宿主仓库即可构建
  wheel 和 sdist；
- wheel 只包含 runtime package，sdist 另外包含 portable tests 与文档；两者都不得携带
  宿主 `src/`、cache、`__pycache__` 或 `.pyc`；
- package release version、artifact wire format 和 analyzer descriptor version 分别
  演进，不因另一个 identity 改变而自动升级；
- 当前发行 metadata 使用 `LicenseRef-Proprietary`。这不是猜测的开源许可证；权利人
  必须在任何外部分发前确认或替换许可证和 metadata；
- 发行 conformance 必须从复制后的独立目录构建，在无仓库 `PYTHONPATH` 的 clean
  environment 安装 wheel，并以 wheel 外部的普通 MLP/Transformer adapter 执行
  checkpoint identity/null control 与 report。

## 影响

普通在线诊断只需接入公开 lifecycle。checkpoint 与高级 extension 需要调用方实现显式
adapter 或 capability；宿主只把 capability 完整的 optional analyzer 组合进 effective
catalog，缺失时在 recipe parse 阶段拒绝，condition-local 证据不足才 structured skip。
任务专属实现不能进入 portable package，portable package 也不能反向 import host
integration。声明的依赖范围不等同于全部版本组合都已验证；实际发布门槛由兼容性矩阵和
conformance 结果共同决定。
