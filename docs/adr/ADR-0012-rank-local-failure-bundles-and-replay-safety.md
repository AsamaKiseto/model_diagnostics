# ADR-0012 Rank-local failure bundle 与 replay 安全边界

> 本 ADR 已由 ADR-0013 替代。当前生产路径只保留失败事实包，不提供 `research`
> 在线模式或 failure replay CLI。下文仅记录被替代设计。

## 状态

Superseded

## 背景

普通 online transaction 适合有界 scalar 和充分统计，但 failure-time model state、RNG、
输入 payload 与 gradient attachment 可能很大。若把它们放入 bounded queue，queue
overflow、writer thread 故障或 DDP collective 已损坏时可能丢失唯一的失败现场。反过来，
仅依赖每 200 步等低频 checkpoint 又无法恢复 checkpoint 之间的失败 attempt。

failure replay 还存在独立风险：optimizer 可能已经部分写入，某个 rank 可能缺失，OOM
可能依赖 allocator 状态，输入或随机变换也可能无法精确重建。完整性通过不等于允许重放，
重现一次失败更不等于已经定位计算根因。

## 决策

- `ModuleDiagnosticsPlan` schema 升级为 `3`，并成为不可由用户 JSON 覆盖的内部执行策略。
  宿主只通过 `diagnostics_mode=off|flight_recorder|research` 选择 instrumentation；
  portable 完整 `Controller` 继续作为 research engine。
- online hierarchy/row schema `9` 记录 plan v3 provenance。顶层
  `diagnostics_v2`、online source `training_diagnostics_v2` 与 transaction
  `diagnostics_transaction_v1` identity 不变。
- hard failure 的大型现场写入 online session 下的独立 `failure_bundles/<failure_id>/`
  sidecar。普通 failed-attempt transaction 只携带 `FailureBundleReference`，不内嵌大
  Tensor。
- 每个 rank 只拥有 `ranks/rank_<rank>/`。writer 构造时立即同步原子写 `running`
  manifest；attachment 使用 rank-local `fsync + replace`；final manifest 写完后才写
  terminal commit。该路径不使用 online bounded queue，也不要求 failure-time
  collective。
- rank-local commit 只证明单个 rank 的 durability，不代表整个 bundle 已终结。每个
  rank 完成自身 commit 后都可调用任务无关的 `finalize_failure_bundle()`；只有全部预期
  rank commit 均存在、identity 一致且验证通过时，才依次原子写 bundle 根
  `manifest.json` 和最后的根 `commit.json`。并发调用者基于不可变 rank commit 生成
  相同根 manifest，因此不需要 collective、queue 或共享进程锁。
- rank commit 绑定 manifest size/SHA-256 与 attachment path、size、SHA-256。
  attachment 只允许 canonical POSIX relative path。validator 不读取未声明 orphan，也不把
  缺 rank commit、缺 bundle 根 commit、缺 rank 或 hash 不一致提升为完整 evidence。
  根 manifest 绑定每个 rank commit 的 path、size 与 SHA-256；根 commit 再绑定根
  manifest。没有合法根 commit 时，即使所有 rank-local commit 已完成也只能是
  `partial/degraded`，绝不进入 replay executor。
- replay safety 固定区分：
  `exact_forward_backward_replayable`、`approximate_replayable`、
  `post_primary_state_only`、`unsafe_partial_optimizer`、
  `unsafe_distributed_state`、`unsafe_allocator_failure` 和
  `insufficient_evidence`。
- Base CLI 只提供 `failure replay` 协议骨架。它先验证 bundle 与全部 rank safety，再动态
  调用宿主 `FailureReplayExecutor`。Base 不 import 宿主，不物化 task batch，不实现
  objective，也不授权 optimizer replay。
- replay 结果只能是：
  `reproduced_and_localized`、`reproduced_but_not_localized`、
  `signature_mismatch`、`non_reproducible_failure`、`unsafe_to_replay` 或
  `insufficient_evidence`。只有宿主隔离 replay 才能产生前四种执行结果。

## 影响

- failure header 和 attachment 写入位于异常路径，会增加失败时同步 IO，但不会把普通
  update 变成逐步 `fsync`。
- rank-local ownership 允许 collective 已失效时仍保留局部证据；缺 rank bundle 只能是
  `degraded`，默认不得进入 replay executor。
- bundle finalize 依赖共享文件系统对同目录 `os.replace` 的原子性，以及已关闭 rank
  commit 的可见性。具有延迟目录可见性的分布式挂载可能让最后一次即时尝试暂时返回
  `None`；此时 bundle 保持 `partial/degraded`，必须在文件可见后安全地重试 finalize，
  不能绕过根 commit 验证。
- v10 reader 明确知道 plan v3 的内部执行 provenance，但原有 v2 root、transaction commit
  和 segment 轮转语义不变。
- 当前 CLI 不承诺历史 optimizer、scheduler、GradScaler、DataLoader cursor 或外部系统
  状态重放。后续若要支持 optimizer replay，必须新增独立 capability、状态 contract 和
  ADR，不能扩张本协议的含义。

## 被否决方案

- **把 failure payload 放入普通 online queue**：无法保证 queue overflow 或 writer
  failure 下保留关键现场。
- **每写一个 attachment 就更新共享 bundle manifest**：需要锁或 collective，恰好依赖
  故障时最不可靠的同步机制。当前根 manifest 只在全部不可变 rank commit 可验证后写入，
  且并发 finalizer 生成相同内容。
- **缺少精确状态时自动用最近 checkpoint 重放**：只能产生 checkpoint-conditioned
  实验，不能冒充历史失败 attempt。
- **默认重跑 optimizer step**：部分写入和缺失 optimizer state 会让重放既不安全也不可
  证伪。
