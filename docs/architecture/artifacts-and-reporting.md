# Artifact 与报告

目录仍使用 `diagnostics/v2/` 的 transaction/commit/hash 布局。Python package 重构不改变
根目录 identity；当前 writer 只写当前 schema，reader 不兼容旧 schema。

报告先校验 manifest、commit、SHA-256、duplicate identity 和 analyzer descriptor，再
读取 evidence。主页面只按三个阶段组织：

`diagnostics/v2/index.json` 记录当前 checkpoint runtime descriptor digest。报告只消费
该 generation 下的 sweep/final analyses，避免代码或 capability contract 更新后把旧
模块清单、旧通道身份和新 evidence 混到同一张图；旧目录仍可审计，但不进入当前报告。

checkpoint finalize 对 detail stream 使用显式 `evidence_key` 去重；没有该键的上下文记录
只去除内容完全相同的副本。`condition_id` 不能作为 detail 去重键，因为一个 condition
可以同时产生多个输出组件和 metric 的独立证据。所有成功计算且进入报告的数值记录显式
写入 `status=success`。

- 训练过程监测；
- 逐检查点异常监测；
- 最终模型分析。

趋势量必须以 `update` 或 `checkpoint_update` 为固定横轴，每个输出、参数叶子层或观测点
一条线，不能跨 update 聚合。图形类型和坐标由 metric guide 固定声明：一个时间位置显示
单点，多个时间位置连接成线；数据量不触发图形切换。缺少全部时间坐标时只显示无数据，
禁止改画成分布图、横向点图、单值卡或其它替代图。训练与 checkpoint 趋势图的折线是
同一训练步的均值，不再叠加会挤压纵轴的逐样本散点；数值完全重合的指标使用不同线型叠加并显式说明，
不能让后画曲线遮住先画曲线。

训练页把目标信息固定拆成三组：`objective_terms` 中的全部命名训练目标项、
`component_objective_value` 的逐输出通道 loss，以及 `raw_objective_sum /
backward_objective_sum` 的整体目标。三组都只复用 flight recorder 已保存的真实训练 ledger，
不增加 forward 或 backward。报告保留每个 term 的原始名称，使 loss、调度权重和其它 ledger
项能够被区分，而不在通用报告中按名称猜测任务语义。

训练页的 `gradient_norm` 是实际 backward 总目标对全部参数的全局梯度范数。标量
`objective_terms` 不能反推出每一项的参数梯度；逐 loss 参数梯度需要为每个目标项单独执行
`autograd.grad`/VJP，会显著增加训练时间与图保留成本，因此 flight recorder 不计算也不绘制
这类曲线。逐输出张量的直接梯度由 checkpoint sweep 的 `output_gradient_rms` 提供，不能与
逐 loss 参数梯度混称。

所有趋势图在主页面固定使用线性纵轴；点击图表进入放大窗口后，才提供“线性”和
“对称对数”两种纵轴尺度。对称对数使用
`sign(y) * log10(1 + abs(y) / threshold)`，其中 `threshold` 是当前纵轴最大绝对值的
`1e-6`；它保留零值和负值，不会像普通对数轴一样丢弃梯度余弦、响应差等带符号测量。
尺度切换只改变坐标映射，不改变或筛除诊断观测。线性轴主刻度使用
`1/2/2.5/5 × 10ⁿ` 的整洁间隔；对称对数轴主刻度固定为带符号的 `10ⁿ` 和零，不显示
由变换后等距反算得到的零碎小数。

主图和放大图使用同一份交互契约：鼠标靠近观测点时优先显示点信息，位于曲线而不靠近点时
显示整条曲线信息。放大窗口使用其 dialog 内部的 tooltip，避免浏览器 top-layer 隔离使提示
被遮挡。训练过程监测的功能图固定为全宽、等高卡片，不因指标数量或本次数据形态改成半宽图。
技术标题保留完整的领域术语，例如 `Loss Components`、`Per-Output Loss`、
`Global Gradient Norm`、`Raw Training Loss` 和 `Backward Training Loss`。受控标题不再经过
逐词翻译，只压缩多余空白；中文用于描述指标关系、有效性和阅读方式。

激活与归一化层分布趋势按 `module_path + tap_id` 建立曲线身份。discovery 找到的每个
Activation、BatchNorm、LayerNorm、GroupNorm 或 InstanceNorm `nn.Module` 都单独成线；
不存在对应模块类型时不生成虚假曲线。`output_rms`、`output_std`、`output_abs_max`、
`output_zero_fraction` 和 `output_nonfinite_fraction` 使用相同层身份，便于逐层对照。
只有同时缺少输出通道身份、`module_path`、`tap_id` 和 `node_id` 的普通 runtime 记录才
允许汇总成总体均值；带层身份的观测不得跨层合并。

输入×输出和模块×输出固定使用可筛选分页列表。主表只列实际干预方法、输入或模块、
输出通道、cohort 均值、实际响应范围、恒等对照平均响应和恒等对照响应范围；恒等对照
不再作为可筛选的干预类型。输入对照按输出通道共享，模块对照按模块站点与输出通道
配对。`support_count`、`changed_element_count / affected_value_element_count`、
`unavailable_sample_count / sample_count` 和 `preserved_paths_verified` 继续完整保存在 artifact，
但只有缺失、未配对或核对失败时才进入折叠的有效性表。列表保留全部实际干预组合，
但每页只创建 100 行 DOM。每个影响页保留一个默认折叠、展开时才创建
的色块矩阵，作为表格的补充证据；关闭页面时不会预先创建大型笛卡尔积 SVG。
模块影响页额外持久显示已覆盖的 `Stage` / `Block` 数量，并按 `hierarchy_level` 与
`module_path` 提供两个独立筛选器。列表和矩阵默认包含所有已确认模块，而不是只选择首个
模块；矩阵使用当前筛选，避免把全部 Block 强行挤进同一视图。
rollout 使用完整时间序列及少量汇总；不会为每个充分统计单独生成图。主视图之外还提供
全部输出通道 × horizon × RMSE/sRMSE/最大局部斜率/episode 数的概览色块，以及
success/structured skip、候选样本、有效样本、有效元素和排除原因表。`cohort_item_ids`
只用于原始证据审计，不生成没有比较意义的逐 ID 图。

影响列表中的“平均响应”是同一干预方法、输入或模块、输出通道下所有有限
`effect_value` 的算术平均；生产者已经按 cohort 汇总时，一条记录就是 cohort 均值。
“响应最小值/最大值”是同一组合底层有效配对响应的范围，不是模型输出范围或置信区间。
恒等对照两列是在相同 checkpoint、sample、`eval()` mode 与随机流下重复执行未改变
condition 的 cohort 均值和范围，用来呈现重复执行噪声，而不是从实际干预响应中线性扣除。
响应正值固定表示干预后任务指标
变差，负值表示改善；这些值只描述模型干预响应，不表示物理因果。

`effect_value` 由基线指标 `B` 和干预后指标 `C` 成对计算：对于越小越好的指标取
`C - B`，对于越大越好的指标取 `B - C`，从而固定正值为变差、负值为改善。
`normalized_effect = effect_value / max(abs(B), 1e-12)`，只用于相同指标契约下比较相对
响应。`objective_normalized_local_taylor` 定义为
`sum(abs(a * dJ/da)) / max(abs(J), 1e-12)`；它只描述当前 checkpoint 附近的局部一阶
敏感度，不能替代真实干预，也不是 Integrated Gradients、Conductance 或物理因果证据。

指标说明使用固定的“指标字段、如何计算、诊断用途、参考与对照、不可解释条件”结构。
主区采用固定两列网格：普通单指标图占一列，组合趋势、矩阵、影响列表和最终模型概览
占满整行。卡片跨度只由功能和图形契约决定，不根据本次观测数、标签长度或缺失情况
临时改变。同一类型图在同一栏目保持相同宽度；长矩阵按完整 SVG 高度直接展开，不设置
卡片内部滚动条，也不把几十个或上百个类别压缩成细长图。

“输出目标之间的优化关系”固定合并为一张对齐矩阵：每行是同一个输出通道，四列依次为
梯度余弦、梯度范数比、目标分区覆盖率和共同活跃参数比例。四列共享输出身份和行高；
单元格显示有效样本均值，悬停显示最小值、最大值、有限观测数和有效监督 support。梯度余弦使用
`[-1, 1]` 零中心色阶，梯度范数比以 `1` 为中心使用对数方向色阶，分区与监督覆盖率以
`1` 为完整参考。共同活跃参数比例说明梯度几何实际建立在多少共同参数上，不代表没有冲突。
样本或 group identity 不作为矩阵轴。

报告页眉只显示报告类型和“模型诊断”标题，不显示完整 `run_dir`，也不重复展示结论边界
说明。运行路径仍保存在内嵌 manifest 和“证据与原始数据”中，供需要时追溯。

训练执行状态只把 `rejected_microbatch_count` 注册为诊断指标，不再单独展示
`processed_microbatch_count`。在线 heartbeat 和 failure bundle 也不再持久化 processed、
committed、finite 或 rolled-back 总数；训练执行器内部仍可使用这些计数完成 update 控制，
但它们不属于诊断 artifact。microbatch 异常只持久化 `rejected_microbatch_count` 和原因。

## 指标与计算字段去向

正式 metric guide 必须满足“生产端仍计算、报告有固定入口、说明声明解释边界”三项条件。
当前正式目录中的每个指标均有可视化入口；“补充证据”也是按需加载的真实图或表，不是
占位。以下表逐项给出正式指标的位置与作用。

| 阶段 | 指标 | 展示位置 | 作用 |
|---|---|---|---|
| 训练 | `objective_terms` | 各命名训练目标项折线 | 比较总目标、AR、chunk、shape 等命名项的尺度和训练趋势；保留原名区分 loss 与权重 |
| 训练 | `component_objective_value` | 各输出通道训练目标折线 | 查看通道 loss 长期失衡和突变 |
| 训练 | `raw_numerator_sum` | 训练目标补充折线 | 核对 support/normalization 前的目标分子 |
| 训练 | `raw_objective_sum`、`backward_objective_sum` | 同一组合折线 | 核对实际反向目标是否因 cap/divisor 与原始目标分离 |
| 训练 | `loss_cap_hit_fraction` | 有效性折线 | 显示 loss cap 实际介入的频率 |
| 训练 | `gradient_norm` | 整体梯度折线 | 查看训练 update 的全局梯度尺度变化 |
| 训练 | `rejected_microbatch_count` | 执行状态折线 | 显示本 update 是否发生 microbatch 回滚 |
| 检查点 | `response_value` | 每输出通道检查点折线 | 比较固定 cohort 上各输出指标的训练轨迹 |
| 检查点 | `raw_objective`、`backward_objective` | 同一组合折线 | 核对检查点目标与反向目标是否重合或分离 |
| 检查点 | `output_gradient_rms`、`output_gradient_zero_fraction`、`output_gradient_nonfinite_fraction` | 每输出通道折线 | 查看直接输出梯度的尺度、消失和非有限值 |
| 检查点 | `relative_grad_rms` | 每参数叶子层折线 | 比较梯度相对参数尺度，筛查消失或爆炸方向 |
| 检查点 | `grad_abs_max` | 每参数叶子层折线 | 补充 RMS 容易掩盖的孤立尖峰 |
| 检查点 | `gradient_energy_share` | 每参数叶子层折线 | 查看梯度能量是否集中到少数层 |
| 检查点 | `grad_zero_fraction`、`grad_nonfinite_fraction` | 每参数叶子层折线 | 查看零梯度与 NaN/Inf |
| 检查点 | `no_grad_parameter_tensor_count` | 每参数叶子层折线 | 区分 `grad is None` 与数值为零的梯度 |
| 检查点 | `output_rms`、`output_std`、`output_abs_max` | 每 Activation/Norm 层折线 | 查看输出尺度、离散程度和峰值漂移 |
| 检查点 | `output_zero_fraction`、`output_nonfinite_fraction` | 每 Activation/Norm 层折线 | 查看稀疏/饱和迹象和非有限值 |
| 检查点 | `normalization_running_mean_rms`、`normalization_running_var_mean` | 每归一化层折线 | 查看 running statistics 漂移 |
| 检查点 | `elapsed_seconds` | 每 checkpoint 折线 | 核对批量诊断总运行预算 |
| 最终输入 | `effect_value` | 分页列表与折叠色块矩阵 | 比较每个输入通道对每个输出的实际干预响应 |
| 最终输入 | `normalized_effect` | 补充列表与色块矩阵 | 在同一指标契约下补充比较相对基线响应 |
| 最终模块 | `effect_value` | 分页列表与折叠色块矩阵 | 比较已确认 Stage/Block 对每个输出的实际干预响应 |
| 最终模块 | `normalized_effect` | 补充列表与色块矩阵 | 补充比较不同输出量级下的相对响应 |
| 最终模块 | `objective_normalized_local_taylor` | 模块比较图 | 仅筛查当前参数点附近的局部一阶敏感度 |
| 最终目标 | `gradient_cosine`、`norm_ratio` | 四列关系矩阵 | 查看目标梯度方向和尺度关系 |
| 最终目标 | `partition_coverage`、`joint_active_parameter_fraction` | 同一关系矩阵 | 核对上述几何是否建立在完整目标和共同参数支持上；监督 support 在悬停中显示 |
| 最终滚动 | `rmse`、`srmse` | 单通道汇总与全通道概览 | 比较绝对误差和训练尺度归一化误差 |
| 最终滚动 | `timestep_mean_absolute_error`、`timestep_q90_absolute_error` | 每 horizon 独立时间曲线 | 查看典型与尾部误差随闭环步数增长 |
| 最终滚动 | `rolling_local_slope_max`、`unstable_episode_count` | 单通道汇总与全通道概览 | 查看最快局部增长与超阈值片段数量 |
| 最终滚动 | `time_to_threshold_median` | 单通道汇总 | 查看达到显式阈值的典型时间 |

仍在 artifact 中计算但不注册为独立指标的字段按用途分为以下四类。它们都能在“证据与
原始数据”中查询；只有直接影响主结果有效性的字段进入主表，避免重复图表。

| 字段 | 是否单独可视化 | 原因与当前位置 |
|---|---|---|
| `gradients_finite`、optimizer/scaler/scheduler/auxiliary outcome、attempt/update/transaction identity | 否 | 离散执行事实，用于 failure/commit 审计；原始记录比数值图更准确 |
| objective unit count | 否 | 用于计算 loss-cap 命中比例并核对 objective observation 完整性；只保留 `rejected_microbatch_count` 作为 microbatch 异常方向指标 |
| objective identity、cap/divisor/AMP/precision/chunk provenance | 否 | 定义图中目标数值的口径，适合详情和原始记录，不适合另画趋势；`objective_terms` 的 loss 数值已进入训练页 |
| `grad_rms`、`parameter_rms`、`grad_l2`、gradient/parameter numel | 否 | 是 `relative_grad_rms`、`gradient_energy_share` 的充分统计或分母；单独成图会重复尺度事实 |
| activation/gradient 的 count、sum、square sum、finite/nonfinite count | 否 | 用于跨调用/rank 正确聚合 RMS、std、fraction；属于可复算充分统计 |
| baseline/condition value、`raw_delta` | 否 | 是 `effect_value` 的成对计算依据；主表已展示 effect、范围和恒等对照 |
| baseline/condition support、`support_count` | 不单独作图 | 进入影响表“有效输出元素”或悬停，判断结果是否有足够支持 |
| affected value/validity、changed element、sample/unavailable、preserved path | 不单独作图 | 进入输入/模块影响表的有效性列，核对干预范围、样本缺失和路径隔离 |
| `local_taylor_abs_sum`、捕获元素/调用计数 | 否 | 是归一化局部 Taylor 的分子和覆盖依据；原始记录用于复核捕获范围 |
| gradient dot、目标 A/B norm、negative-dot、gradient numel、partition residual/tolerance | 否 | 是 cosine、norm ratio、coverage 的计算依据；四列关系矩阵给出可读主结果，原始记录保留复算条件 |
| rollout candidate/fixed/available/support/threshold/window/skip/exclusion | 表格 | 进入 cohort 覆盖与缺失原因表，不能用缺少身份的独立折线替代 |
| sample/group/component/checkpoint/cohort/condition identity、commit/hash/claim boundary | 否 | 只用于配对、分组、去重、完整性与解释边界 |

已经停止计算的旧量不再保留 producer 占位、renderer、功能栏目或 fallback 标签：模块
`intervention_objective`/`objective_delta` 及其固定空匹配字段、名为
`effective_supervision_coverage` 但实际等于 `support_count` 的伪覆盖率、rollout AUC、中位局部
斜率、boundary jump、cluster bootstrap、逐 item threshold time 与 episode 明细，以及已删除
extension 的 probe、ECL、2×2 feedback、donor/conditional permutation。恢复其中任一项必须
同时恢复 producer、metric guide、固定可视化位置和运行预算，不能只恢复一个空页面。

报告正文可以中文化指标说明，但 `module_path`、`tensor_path`、`state_*d` 等代码身份必须
逐字保留，禁止在一个 identifier 内把 `norm`、`output` 等片段替换为中文。

网页固定写为 `diagnostics/v2/diagnostics-report.html`，全部 validated source 压缩后
内嵌；不生成 linked bundle、外部 data chunk 或第二份 HTML。每个 gzip member 使用
独立的不可执行 `<script type="application/octet-stream">` 承载，页面启动时不把全部
Base64 解析成 JSON。用户进入栏目后，内嵌 Web Worker 才解压、解析并展平该栏目需要的
source；Worker 只把当前 metric 的数值和必要身份字段投影给主线程，并按 source 复用
解压结果。主线程的完整 source LRU 只服务原始记录查看器。最终模型分析再按诊断功能
分页，进入通道影响、模块影响、目标关系或 rollout 页时才加载对应 evidence。

checkpoint manifest 在网页内使用已经过 validator 校验的紧凑副本：成功 evidence key
不重复嵌入，只保留计数、inventory digest 和权威 manifest SHA-256；missing、failed、
skipped key 仍逐项保留。analysis 目录中的权威 manifest 不被修改，成功 key 仍可由
finalized stream 的 `evidence_key` 重建。这项压缩只删除报告副本中的重复验证事实，不
删除诊断观测。

数值序列和自动发现的 metric 不做点数/数量截断。训练页的 `Loss Components`、
`Per-Output Loss`、
整体目标和全局梯度范数把完整点集合编码为每条曲线一个 SVG path，只保留整条曲线的
悬停说明，不建立逐点 DOM、逐点空间索引或逐点详情；原始 update 数值仍完整保存在 artifact。
其它需要精确点选的趋势图仍使用单一 hover overlay，不为每个观测创建独立交互层。
最终 rollout 先选择一个输出通道和 RMSE/sRMSE 汇总口径，再把
每个显式 horizon 固定画在相互独立且等尺寸的分面中；例如 50 步和 500 步不会共享曲线、
坐标区或含混的条件标签。曲线固定表示原始尺度 mean absolute error 与 q90；汇总口径选择
不会重命名或缩放该曲线。每个分面右侧列出所选 RMSE/sRMSE、最大局部误差斜率、不稳定
片段数和达到阈值的时间中位数。全通道色块中同一指标跨 horizon 共用色阶，缺失组合显示
斜纹。覆盖表明确分开 fixed-complete 与 available cohort。放大趋势图从相同数据重新
渲染，不克隆密集节点。报告不自动输出异常结论，只说明每个图的阅读方式和证据失效条件。
