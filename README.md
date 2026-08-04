# Model Diagnostics

`model_diagnostics` 是面向通用 PyTorch `nn.Module`、autograd 与 checkpoint 的独立诊断
package。Base 只依赖 Python 标准库和 PyTorch；当前 extensions 只保留最终 checkpoint
需要的 input-dependence、multi-objective 和 rollout 算法。具体任务事实由调用方
runtime/capability 注入。

快速验证：

```bash
python -m pip install .
python -I -m model_diagnostics.base --help
python -m model_diagnostics.base report --run-dir <run-dir>
python -m model_diagnostics.extensions report --run-dir <run-dir>
pytest -q -c pytest.ini tests
```

## 安装发布版

发布产物位于私有 GitHub 仓库的 release。已获授权的环境可下载 wheel 后安装：

```bash
gh release download v0.1.0 \
  --repo AsamaKiseto/model_diagnostics \
  --pattern 'model_diagnostics-0.1.0-py3-none-any.whl'
python -m pip install ./model_diagnostics-0.1.0-py3-none-any.whl
```

也可以通过已有 SSH 凭据直接安装固定 tag：

```bash
python -m pip install \
  'model-diagnostics @ git+ssh://git@github.com/AsamaKiseto/model_diagnostics.git@v0.1.0'
```

参与 package 开发时，推荐把本仓库与宿主仓库放在同一父目录；宿主可优先解析该源码树，
缺失时再使用已安装 distribution。该优先级由宿主入口实现，本 package 不读取宿主路径。

Base report 只启用 Base renderer；extension report 通过显式、无全局注册的 catalog
factory 加入 multi-objective、input-dependence 和 rollout renderer。生成的
`diagnostics-report.html` 按“训练过程监测、逐检查点异常监测、最终模型分析、证据与
原始数据”组织，而不是把全部字段放进同一个筛选器。当前 run 已采集的全部注册指标
直接绘制；未运行的 analyzer 不生成替代数据。

训练和逐 checkpoint 图以从 0 开始的整数 update 为横轴，每个比较对象一条折线；最终
输入/模块影响以完整分页表为主，并可按需展开色块矩阵；最终 rollout 先选择输出通道和
显式 horizon，再展示对应误差曲线。只有一个 observation 时仍在固定坐标中显示单点。
artifact、完整性校验和原始字段集中在“证据与
原始数据”栏目。页面展示每项指标的定义、比较方法、参考口径和失效条件，但不自动输出
异常、根因或行动建议。

栏目显示“未运行”时，表示对应阶段或宿主 capability 没有产生该 evidence，而不是指标结果为 0；analyzer
已执行但没有合法有限 observation 时显示“无有效 evidence”。单文件交付固定写入：

```text
<run_dir>/diagnostics/v2/diagnostics-report.html
```

该文件内嵌全部 validated evidence，可单独下载和打开，不依赖 `figures/`、`data/` 或
其它 report 文件。图表不是静态素材：点击任意语义图或 evidence summary 可打开大图，
并使用按钮/滚轮缩放和拖动。全部 numeric/series 与 categorical artifact path 还可在
“完整字段可视化”中按 source 选择查看；未注册字段只展示原始值或频数，不自动解释。

## 新任务如何接入

推荐让训练、评估与诊断共同消费一个任务运行时，而不是为诊断另写一套 loss、checkpoint
loader 或 rollout loop。接入分为“每个任务实现一次”和“每次运行选择一次”两层。

### 每个任务实现一次

一个新任务至少提供以下正常运行组件：

| 组件 | 职责 | 基础诊断是否必需 |
| --- | --- | --- |
| `ModelProvider` | 构造、恢复和释放 `nn.Module` | 是 |
| `CheckpointProvider` | 解析任意 checkpoint 布局并返回稳定 identity | 是 |
| `BatchProvider` | 选择样本、物化 batch、复制 intervention 输入 | 是 |
| `ObjectiveExecutor` | 执行与训练相同的 objective，返回 raw/backward objective provenance | 是 |
| `TrainingAttemptExecutor` | 拥有 backward、AMP、clip、optimizer 与 scheduler 顺序 | 在线诊断需要 |

这些组件应先服务任务自身的训练或评估。诊断只桥接同一执行事实，不得在 adapter 中复制
objective、gradient accumulation、AMP、checkpoint 发现或 batch 重建。

基础 artifact、checkpoint sweep 和通用 module intervention 只需要上述核心组件。
当前 final-selected 阶段只组合下列高级能力：

| Capability | 任务只需提供的事实 |
| --- | --- |
| multi-objective | 同一真实 objective 的 `objective_a/objective_b/shared_terms` |
| input-dependence | tensor target、identity/mean replacement 与修改 provenance |
| rollout | 显式 `ScenarioSpec` 对应的 `RolloutTrace` |
| failure facts | 已发生 objective、梯度、step outcome 与参数非有限定位 |

任务不支持的 capability 不注册。当前不提供 representation probe、sequence memory、
autoregressive 2×2 feedback、conditional donor、optimizer transform 或 failure replay。

### 通用组合与标准 CLI

Host Runtime contract 和 Generic checkpoint adapter 都在本 distribution 中：

```python
from model_diagnostics.integration import create_checkpoint_adapter


def create_adapter(*, run_dir):
    runtime = create_task_runtime(run_dir)  # 宿主唯一的 task composition
    return create_checkpoint_adapter(
        runtime=runtime,
        run_dir=run_dir,
        owns_runtime=True,
    )
```

宿主 CLI 只需把该 factory 绑定到标准 runner，不重写 recipe、engine、
artifact 或 report 流程：

```python
from model_diagnostics.cli import run_diagnostics_cli
from my_project.diagnostics_runtime import create_adapter


if __name__ == "__main__":
    raise SystemExit(run_diagnostics_cli(create_adapter))
```

因此新仓库需要实现的是任务本身需要的 providers/capabilities 和一个
composition factory，而不是一份诊断执行脚本。公共对象和资源所有权见
[Host Runtime 契约](docs/architecture/host-runtime.md)。

### 三阶段入口

训练期固定启用低开销 flight recorder，不提供关闭、profile 或 cadence 参数。逐
checkpoint 只运行训练健康扫描：

```bash
python -m my_project.diagnostics checkpoint-sweep --run-dir <run_dir>
```

最终分析必须显式选定唯一 checkpoint：

```bash
python -m my_project.diagnostics final \
  --run-dir <run_dir> \
  --checkpoint <checkpoint>
```

不存在 recipe JSON、analyzer 选择、checkpoint glob/filter/count、cohort size/seed、
分布 include/exclude 或 rollout 统计参数。sweep 遍历宿主 provider 显式声明的检查点，
使用 32 个 test window 且至少覆盖 8 个 group；final 固定使用 8 个 test window 且覆盖
8 个不同 group，并运行 catalog 中全部 final analyzer。支持不足时输出
`insufficient_evidence`，不会降低阈值或随机回退。最终 rollout horizon 由宿主正式评估
profile 声明。

### 新模型与新任务的代码量边界

- 同一任务 contract 下新增模型：应为零行 diagnostics-specific 代码。
- 新的数据形态或 objective：实现正常的 provider/executor，并复用
  `model_diagnostics.integration`。
- 只有新增任务语义时才实现相应 optional capability。
- 框架外的普通 PyTorch 项目实现 Host Runtime providers 后可直接使用 Generic
  adapter；仍应复用真实 objective，而不是构造诊断专用 surrogate loss。

接入验收至少包括：Base cold import 不加载宿主或 extensions；diagnostics on/off 的
weights、optimizer、scheduler、RNG 与 sample identity 等价；checkpoint objective 和
训练 objective/gradient 对齐；identity control 为数值零；缺少 capability 时不产生
伪成功；runtime/capability descriptor 变化会改变 analysis identity。

local Taylor、replacement、gradient geometry 与 rollout 都只是限定协议下的诊断
证据，不得解释为物理因果。

完整架构、公共契约、兼容性矩阵和结果解释边界见
[docs/README.md](docs/README.md)。

当前发行 metadata 使用 `LicenseRef-Proprietary`。这不是开源许可；任何外部分发前都必须
由权利人确认或替换 [LICENSE](LICENSE)。
