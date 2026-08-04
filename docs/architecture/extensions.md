# 保留的 Extensions

当前只保留 final-selected 阶段需要的 extensions：

| Extension | 输入 | 输出 | 明确不包含 |
|---|---|---|---|
| `input_dependence` | 通用输入/输出 identity、identity/mean replacement | input×output effect | donor、conditional permutation、时序匹配 |
| `multi_objective` | actual objective A、complement、shared terms | model-wide cosine、norm ratio、partition/supervision coverage、共同活跃参数比例 | PCGrad、CAGrad、GradNorm、Recon、fresh branch |
| `rollout` | 显式 `ScenarioSpec` 与 free `RolloutTrace` | RMSE/sRMSE、逐步 mean/q90 误差、最大局部斜率、episode、cohort 覆盖 | 2×2 feedback、Oracle、隐式 horizon、AUC、中位 slope、boundary jump、bootstrap |

representation probe、sequence memory/ECL 和 autoregressive feedback extension 已从当前
生产 catalog 删除。若未来重新引入，必须有独立运行预算和面向用户问题的验收，不得仅因
已有算法实现而恢复。

`multi_objective` 的 `gradient_dot`、两个 objective gradient norm、negative-dot、
gradient numel 和 partition residual/tolerance 只作为复算与有效性原始字段保留，不注册
成独立图。报告使用同一输出行的四列矩阵展示 cosine、norm ratio、partition coverage 和
joint active parameter fraction；后两列用于约束前两列的可解释性，监督 `support_count`
保留在悬停和原始记录中。旧 `effective_supervision_coverage` 实际等于 support count，并非
比例，当前 producer 和 renderer 均已删除。以上指标都不表示训练优劣或物理因果。

`RolloutTrace` 不携带只服务于已删除算法的 boundary step 或 bootstrap group。局部斜率
窗口和不稳定阈值属于解释现有指标所需的口径，随汇总 row 持久化；逐 item threshold time
和 episode 明细不进入 artifact，因为当前页面只需要时间曲线、最大斜率、episode 数和
达到阈值时间的中位数。`cohort_item_ids` 保留作身份审计，但不单独可视化。

最终 rollout 字段按以下唯一用途保留，不再为同一事实保留第二份计数：

| 字段 | 报告位置 | 保留原因 |
|---|---|---|
| `rmse`、`srmse` | 单通道汇总与全通道色块 | 比较绝对误差和训练尺度归一化误差 |
| `timestep_mean_absolute_error`、`timestep_q90_absolute_error` | 单通道逐步曲线 | 查看平均误差和尾部误差随 rollout step 的增长 |
| `rolling_local_slope_max`、`unstable_episode_count` | 单通道汇总与全通道色块 | 查看最快局部增长和阈值片段数量 |
| `time_to_threshold_median` | 单通道汇总 | 查看达到显式阈值的典型时间 |
| `candidate_item_count`、`item_count`、`fixed_complete_item_count`、`valid_element_count` | cohort 覆盖表 | 区分候选、当前有效、跨条件完整和逐元素支持 |
| `skip_reason`、两类 exclusion count | cohort 覆盖表 | 解释 structured skip 和 cohort 缺失 |
| `local_slope_window`、`instability_threshold` | cohort 覆盖表 | 公开 slope 与 episode 的计算口径 |
| scenario、horizon、condition、response 与 cohort identity | 筛选、分组和表格 | 保证不同实验条件不被混合 |
| `cohort_item_ids`、checkpoint/commit identity、claim boundary | 原始证据 | 配对、复核、去重和解释边界，不适合作数值图 |

`available_item_count_by_condition` 不再复制到每个最终汇总 row；该 row 的 `item_count`
就是当前 condition 的 available/fixed 支持数。AUC、中位局部斜率、boundary jump、
cluster bootstrap、逐 item threshold time 和 episode 明细既不计算，也没有 renderer。
