# ADR-0010 通用诊断 v2 事务产物与 session 生命周期

## 状态

Accepted

事务、commit、hash、resume 与报告部分继续生效。ADR-0013 已删除本文曾描述的在线
controller capture、hierarchy discovery 和规模 benchmark；这些历史段落不再代表当前
公共入口。

## 背景

逐文件追加 JSONL 无法区分完整 update 与 queue overflow 后的部分 update，也无法可靠处理
进程中断、ENOSPC、writer hang、重复运行和 run directory 复用。checkpoint condition
若先写 summary、后写 detail，还可能把未完成分支误判为成功。统一报告只有先验证来源，
才能避免把截断或身份混杂的 artifact 汇总成可信结论。

## 决策

- 唯一活动布局是 `diagnostics/v2/`。writer、analyzer 和 report 不读取、不转换，也不因
  旧 `diagnostics/online/` 或 `diagnostics/checkpoint_conditioned/` 存在而失败。
- 每次训练创建新的 `session_id`，写入
  `online/sessions/<session_id>/`。queue 元素是完整 `logical_update`、
  `failed_attempt` 或 `controller_event` transaction；queue 满时丢弃整个 transaction。
- 每个在线 transaction 以 commit 收口，commit 包含 transaction identity、各 stream row
  count 和 payload SHA-256，且 commit 必须是 transaction 的最后一行。segment 每 256 个
  transaction 或 32 MiB 未压缩数据轮转。
- 普通 transaction 追加不执行逐 update `fsync`。segment 轮转或 writer 关闭时才对当前
  segment 执行 `flush` 与 `fsync`；terminal manifest 只能在 writer thread 完成该
  durability boundary 后写为 `complete`。
- `running` manifest 每 256 个已提交 transaction 检查点一次，并在 segment 轮转后强制
  检查点。periodic checkpoint 只把当前 segment 的 Python buffer 刷入内核以供 live
  validator 读取；原子 manifest 写入同时 `fsync` 文件及父目录。`running` manifest
  永远不能产生 `success`，即使其 commit/hash 完整也只能是 `partial`；崩溃留下的非持久
  segment 由 validator 判为 `partial` 或 `corrupt`。
- 上述批处理只定义 artifact metadata 与 durability cadence，不改变诊断 sampling cadence、
  queue capacity 或 queue 的 transaction 原子单位。
- writer 首次不可恢复错误后拒绝新 transaction；`close()` 最多等待配置的 30 秒。manifest
  状态仅允许 `running`、`complete`、`degraded` 和 `corrupt`，并记录缺口范围。
- checkpoint `analysis_id` 在 `checkpoint_diagnostics_v4` 中由 canonical recipe、
  analyzer definitions、checkpoint identity、cohort identity/sample refs，以及调用方
  显式提供的 runtime descriptor 共同派生。runtime descriptor 绑定
  `runtime_descriptor_digest`、`task_definition_id/version`、
  `objective_executor_version` 和 canonical capability descriptors。每个 condition 按
  detail streams、terminal row、commit 的顺序写入；resume 只承认 hash、row count 和
  完整 v4 runtime identity 均合法的 commit，orphan row 不进入 summary。
- checkpoint writer、validator 与 report 只接受 `checkpoint_diagnostics_v4`。其它
  format 在创建 shard、读取 commit 或汇总前 fail closed；不提供读取、resume 或转换。
  checkpoint schema 与 `diagnostics/v2/` 布局及 `diagnostics_v2` root format identity
  独立。
- adapter analyzer 的 transaction `condition_id` 由 engine 持有，领域 condition identity
  使用 `analyzer_local_condition_id`；rollout 使用 `rollout_condition_id`。detail row
  的 terminal-like `status` 不触发 commit；唯一且位于最后的
  `record_kind=analyzer_terminal` 声明 detail row count/status counts 并触发提交。manifest
  以 `condition_transaction_contract=explicit_analyzer_terminal_v2` 标识该契约。
- Base 从 analyzer descriptor 构造 canonical analysis plan，并按 plan 顺序在 rank 间
  确定性分片；不同 rank/generation 的同名 stream 可以指向不同 shard。resume/report
  逐 commit 读取其 `stream_files`，按 condition 独立重算 row count 与 payload hash，
  不能把同名 stream 预先折叠成 analysis 级单一路径。`execution_mode=aggregate` 的单个
  plan 获得完整 stable cohort，不能把 cohort 本身跨 rank 拆开，也不以 analyzer 名称
  写死 rank 0 特例。
- rank 0 finalize 只从合法 commit 重建最终 analyzer streams。manifest 的
  `finalized_streams` 为 report-visible allow-list，每项绑定
  `stream_id/path/row_count/bytes/sha256`；统一 report 不递归消费 shard、orphan 或未声明
  JSON/JSONL。analyzer terminal 只保留在 committed shard 中供校验，不写入
  report-visible analyzer records。manifest 未声明的 payload-invalid 残留 commit 作为
  orphan warning；manifest 声明的 invalid commit 或未声明的 payload-valid commit 使
  analysis 为 `corrupt`。
- 每个 rank 在 condition 执行前持久化 rank-local expected condition inventory。
  finalize 只有在所有 rank plan 完整、condition owner 唯一且 expected IDs 与合法 commit
  IDs 精确相等时，才允许写 `complete/degraded`；缺失或意外 condition 均 fail closed 为
  `corrupt`。该 inventory 同时参与 complete analysis 的幂等恢复校验。
- 已完成且完整 identity 相同的 analysis 重跑幂等返回；recipe、analyzer definition、
  checkpoint、cohort、task definition、objective executor 或 capability identity 改变时
  生成新的 analysis，不覆盖旧分析。
- report 只读取 `diagnostics/v2`，先验证 manifest、commit、hash、duplicate ID 和 hierarchy
  identity，再做流式汇总。只有完整且校验通过的来源使用 `success`；其它来源显式标成
  `partial`、`degraded` 或 `corrupt`。仅存在 `run_state.json` 而没有 terminal manifest
  的中断 checkpoint analysis 固定为 `partial`。
- 报告始终生成唯一 self-contained HTML。validated source 分块压缩后全部内嵌，单个
  压缩 member 不超过 4 MiB；不生成 linked bundle、外部 data chunk 或第二份 HTML。
  renderer 不截断 series point 或自动发现的 numeric metric。Python 汇总仍不同时物化
  全部 JSONL。
- HTML 进入某个功能页后才由内嵌 Web Worker 解压对应 source，并只向主线程返回当前
  metric 和必要 identity 的投影。最终模型分析按通道影响、模块影响、目标关系与 rollout
  分页；输入×输出和模块×输出使用每页 100 行的筛选列表，色块矩阵只在用户展开补充证据
  时创建。以上只改变展示与浏览器资源占用，不改变 validated source 或 artifact identity。
- report renderer 只按 analyzer descriptor 的 `evidence_kind` 选择展示定义，不按
  analyzer name 分发。Base 与 extension 各自暴露无副作用 catalog，由调用方显式组合；
  已注册 evidence 的 analyzer definition version 必须与当前 renderer 严格相等，否则
  fail-fast；只有未知 evidence 使用有界 generic scalar fallback。figure 必须保留 claim
  boundary、null control、categorical overflow 与 insufficient-evidence 状态。report
  format version `18` 记录 renderer provenance、metric visualization guide、source
  evidence kind 与 validated failure source 摘要；attachment 内容不进入统一报告。
  HTML 主界面按诊断功能分栏，并按原始 observation 保留
  module/condition/checkpoint/update/sample identity；默认不作跨 identity 均值。
  时间比较轴只有一个 distinct value 时仍在固定坐标中显示单点，不据此声称存在趋势。
  每个有限 scalar/series 都可以在对应栏目中绘制，identity/status 用于分组，
  hash/commit/schema identity 仅用于后台校验。报告不自动输出异常、根因或行动建议。
- online session manifest 和 hierarchy 各自持久化相同的完整 stream analyzer
  descriptors；validator 在读取 row 前验证 descriptor schema、definition version 和
  两份事实一致性。hierarchy/row schema version `10` 是 controller、validator 与 benchmark
  的单一事实源；该版本按 ADR-0012 持久化 capture profile provenance，root/transaction
  identity 保持不变。report 不再合成 online local-Taylor descriptor。
- `model_diagnostics.base` 各 owner 的 public API 按符号 owner 延迟导入，CLI 在解析子命令后再加载
  对应 owner。`report` 子命令不加载 PyTorch、controller 或 benchmark owner，使报告
  resource contract 不受在线训练 runtime 初始化污染。
- 规模验收先以轻量 launcher `exec` 清空 benchmark 地址空间，再由 launcher 派生第二代
  report subprocess，并在第二代进程中读取
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`。该两级边界避免 Linux 的首次
  fork/exec 把含 PyTorch 父进程的 RSS high-water 带入被测进程；门槛仍使用 report
  进程的绝对 high-water，父进程 RSS 只作补充观测。

## 影响

- 在线提交粒度变大，queue overflow 会损失整次 transaction，但不会产生无法识别的半条
  logical update；manifest 提供可审计的缺口范围。
- durability I/O 以 segment 为单位批处理，避免把每个训练 update 变成 `fsync`；代价是
  运行中的 periodic manifest 只是一致性快照，不构成终态成功证明。
- checkpoint resume 需要额外 commit/hash IO，但能够把重复条件、截断文件和 orphan rows
  从统计支持集中排除。
- runtime identity 增加少量 manifest 内容和调用方接线，但阻止 task/objective/capability
  实现变化后错误复用结构相同但执行事实不同的 analysis。
- run directory 可安全复用，因为 session 与 analysis identity 不混合；报告可以同时展示
  多个合法来源并保留各自状态。
- report 的延迟导入与两级隔离 RSS 测量使规模门槛对应 report 自身，但 benchmark 需要
  额外 launcher、被测 subprocess 生命周期与 resource measurement artifact。
