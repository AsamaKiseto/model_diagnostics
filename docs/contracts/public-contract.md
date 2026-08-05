# Public Contract

## 操作员入口

```bash
python -m my_project.diagnostics checkpoint-sweep --run-dir <run_dir>
python -m my_project.diagnostics final --run-dir <run_dir> --checkpoint <path>
python -m my_project.diagnostics report --run-dir <run_dir>
```

其中 `my_project.diagnostics` 只需将宿主 factory 绑定到下列独立 distribution API：

```python
from model_diagnostics.host_runtime import HostTaskRuntime, TaskDefinition
from model_diagnostics.integration import create_checkpoint_adapter
from model_diagnostics.cli import run_diagnostics_cli
```

外部仓库实现任务正常运行需要的 providers/capabilities，将组合后的
`HostTaskRuntime` 交给 `create_checkpoint_adapter()`，再将宿主 factory 交给
`run_diagnostics_cli()`。宿主不复制 recipe、condition loop、commit/resume、validator
或 report 流程。

factory 契约固定为：

```python
def create_adapter(*, run_dir: str | Path) -> GenericCheckpointAdapter: ...
```

adapter 持有宿主 runtime 时必须用 `owns_runtime=True` 组合；标准 CLI 会在
成功或失败后统一 `close()`。

`DiagnosticsRecipe` 只由 `checkpoint_sweep()` 或 `final_selected()` factory 在代码内
构造。不存在可编辑 recipe、analyzer 列表、selector 或 per-analyzer option。

操作员可修改项完整限定为：

| 参数 | 保留原因 |
|---|---|
| `run_dir` | 标识需要读取或写入的 run |
| final `checkpoint` | 模型选择是实验决策，诊断系统不能代替用户决定 |

以下值仍出现在 Python DTO 或 artifact 中，但不是手工配置：`model_handle`、`optimizer`、
`rank`、`world_size`、`session_id`、`device`、执行 `precision`、checkpoint identity/update/
weight boundary、objective identity/schedule/cap/divisor/AMP、component catalog、task
scenario、capability descriptor、sample/group identity。它们由 Host Runtime 注入或由实际
执行产生，删除会丢失诊断口径与可复现性。

## 当前 Analyzer

| 名称 | 阶段 | 证据 |
|---|---|---|
| `checkpoint_sweep` | 逐 checkpoint | 输出响应/直接输出梯度、objective、参数叶子层梯度、Activation/Norm 分布 |
| `final_module_influence` | final | runtime-confirmed Stage/Block 的 local Taylor 与 identity/scale-zero/mean-patch 输出响应 |
| `final_channel_influence` | final | identity/mean replacement 的 input×output 响应 |
| `final_input_sensitivity` | final | 多尺度正负配对扰动的 input×output 对称相对输出响应 |
| `final_objective_conflict` | final | objective A 与 complement 的 model-wide 梯度关系 |
| `final_rollout` | final | 显式 scenario 的 free rollout 稳定性 |

## 训练 Port

训练 bootstrap 固定创建 flight recorder，不注册 module hook、不增加 collective、
不额外 forward/backward；`TrainingBridgeConfig` 不接受启用/关闭参数。

## 固定统计契约

| 项目 | 固定值 |
|---|---|
| sweep checkpoint | 宿主 `CheckpointProvider` 声明的全部 sweep checkpoint |
| sweep cohort | test 32 个 window，至少 8 个 group，确定性分层 |
| final cohort | test 8 个 window、8 个不同 group，确定性分层 |
| module mean reference | train 8 个 window、8 个不同 group，response-only |
| input sensitivity | task profile 声明 scale；同一方向的正负配对；final cohort |
| 分布观测 | 全部 Activation/Norm 输入与输出 |
| objective partition tolerance | `1e-6` |
| rollout local slope | `min(20, horizon)`，下限 2 |
| rollout normalized threshold | `1.0` |
| report | 不截断 metric/series；唯一自包含 HTML |
| online writer | degrade、queue 64、close timeout 30 秒 |

固定支持不足时只允许 structured skip，不允许减小样本数或 group 数后伪装为充分证据。

## 参数处置清单

下表覆盖诊断执行、统计、干预、artifact 和展示曾经可能形成配置面的项目。除前两项
操作员入口外，其余要么固定在代码中，要么只能由 Host Runtime 提供真实任务事实。

| 参数族 | 处置 | 生效规则 |
|---|---|---|
| `run_dir` | 保留操作员参数 | 决定 run、checkpoint 和 artifact 根目录 |
| final checkpoint path | 保留操作员参数 | 必须明确指定一个模型选择结果；相对路径按 `run_dir` 解析 |
| stage | 改为命令 | 只允许 `checkpoint-sweep`、`final`、`report` 三个明确命令 |
| adapter/factory 字符串 | 删除 | 宿主 composition root 在代码中绑定 factory，不作为操作员参数 |
| recipe JSON/path | 删除 | 不读取外部 recipe |
| analyzer list/options | 删除 | 每个阶段固定执行其 catalog 中全部 analyzer |
| checkpoint glob/kind/update filter/max count | 删除 | sweep 读取 provider 声明的全部 checkpoint |
| cohort mode/partition/count/seed | 固定 | sweep/test/32；final/test/8；确定性 group 分层，不使用 RNG seed |
| minimum group count | 固定 | sweep 8、final 8、module mean 8 |
| module mean reference count | 固定 | train 8，response-only forward |
| Activation/Norm include/exclude/path/type | 删除 | 自动覆盖全部 PyTorch Activation/Norm 输入与输出 |
| online enable/profile/cadence/sample budget | 删除 | 训练始终使用无 module hook 的 flight recorder |
| writer failure mode/queue/timeout/segment | 固定 | degrade、64、30 秒、256 transaction 或 32 MiB |
| module site selection | 固定发现 | 通用结构候选在 final cohort 的 3–8 次成功 forward 上确认；宿主声明只作增量 override |
| module intervention method/scale | 固定 | identity、`output_scale=0`、training mean patch |
| input target list | 任务事实 | Host capability 枚举全部 model-visible input |
| output target list | 任务事实 | Host capability 枚举全部 objective output |
| input replacement method | 固定 | identity 与 training mean replacement |
| input×output 持久化 | 固定 | 每个 input×output×method 保存 cohort 均值、范围、support、干预元素范围、样本缺失与路径隔离核对 |
| input sensitivity scale | 任务事实 | 由 task profile 声明模型输入坐标中的多个扰动幅度，诊断侧不可覆盖 |
| input sensitivity direction | 固定 | 同一 sample×input 在所有尺度复用同一确定性 Rademacher 方向，并执行正负配对 |
| input sensitivity metric | 固定 | 每个方向先计算 `2×RMS(y_d-y_0)/(RMS(y_d)+RMS(y_0))`，再对正负方向与 cohort 做等样本 RMS；只保存 `symmetric_relative_output_response` 与必要 support/provenance |
| donor/permutation/matching/kNN 参数 | 删除 | 当前精简系统不执行 donor 或 permutation |
| objective identity/schedule/cap/divisor/AMP/precision | 运行事实 | 必须复用真实 checkpoint objective，不允许诊断侧覆盖 |
| objective partition tolerance | 固定 | `1e-6 × (1 + abs(training objective))` |
| PCGrad/CAGrad/GradNorm/Recon 选择与参数 | 删除 | 当前精简系统不执行 optimizer transform |
| rollout scenario/horizon | 任务事实 | 只使用 task profile 声明的场景；诊断侧不可覆盖 |
| rollout condition | 固定 | free closed-loop |
| local slope window | 固定推导 | `max(2, min(20, horizon))` |
| instability threshold | 固定 | normalized error `1.0` |
| report metric/series/category limits | 删除 | 完整保留数值字段、series 点、类别值与标签 |
| report HTML/chunk/link mode | 固定 | 唯一自包含 HTML；内部压缩 member 不超过 4 MiB |
| report source loading | 固定 | source 按当前功能页在内嵌 Web Worker 中解压；只向主线程投影当前 metric 与必要身份字段 |
| dense trend rendering | 固定 | 训练/checkpoint 页按比较对象显示同一 update 的均值折线；横轴从 0 开始且只显示整数刻度 |
| overlapping trend rendering | 固定 | 数值完全重合的曲线使用不同颜色和线型叠加，并在图下注明重合数量 |
| final analysis loading | 固定 | 通道、模块、目标关系和 rollout 按功能分页；只加载当前页 |
| input/module influence view | 固定 | 全部组合进入可筛选分页列表，每页 100 行；色块矩阵仅在展开补充证据时创建 |
| input sensitivity view | 固定 | scale 可切换的完整 `S_ij` 表格与指定 input/output 的尺度—增益曲线；全部 scale 共用色阶 |
| objective relationship view | 固定 | 每个输出一行，同时展示 cosine、norm ratio、partition coverage 和共同活跃参数比例；support 进入悬停 |
| final rollout channel view | 固定 | 选择输出通道和 RMSE/sRMSE 汇总口径；各显式 horizon 分面展示原始尺度 mean–q90 时间曲线 |
| final rollout overview | 固定 | 全部输出通道按 horizon 对齐展示 RMSE、sRMSE、最大局部斜率和 episode 数；相同指标跨 horizon 共用色阶 |
| final rollout coverage | 固定 | success 与 structured skip、fixed-complete 与 available、候选/有效样本、有效元素和排除原因进入同一审计表 |
| UI 栏目、筛选、搜索、分页、缩放 | 保留展示交互 | 只改变当前查看方式，不改变计算、统计或 artifact |

`rank/world_size/session_id`、checkpoint identity/update、sample/group/component identity、
tensor path、support、validity、task capability descriptor 等字段必须保留。它们不是算法
调参，而是防止跨 rank、样本、输出、模块或 objective 错误聚合所需的身份与有效性事实。

## Claim Boundary

局部 Taylor 是局部一阶敏感度；replacement/patch 是特定干预下的预测依赖；gradient
cosine 是局部优化关系；rollout 是指定 cohort/scenario 下的闭环表现。以上都不是物理
因果。
