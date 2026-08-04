# 当前兼容性

本实现不保留旧 Python import、旧 analyzer 名称、旧 recipe 字段或旧 artifact reader。
输入必须使用当前 `stage` 和当前 analyzer catalog；非当前 schema 直接报
`unsupported`。

保留 `diagnostics/v2/` 根目录名称只是当前 wire contract，不表示兼容任何历史实现。
当前 artifact 仍保留 schema/descriptor identity，用于 commit 校验、analysis identity
和 fail-fast；这些字段不能删除。

portable `model_diagnostics.base` 只依赖标准库和 PyTorch。宿主信息只通过
`HostTaskRuntime` integration 输入。
