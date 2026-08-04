# ADR-0009 通用 objective 与 faithfulness 契约

## 状态

Superseded

三阶段精简后的 active 能力与统计契约由 `ADR-0013-three-stage-diagnostics.md` 和
`../contracts/public-contract.md` 定义。下文只保留被取代的完整 faithfulness 设计，
不得据此恢复已删除 analyzer 或参数入口。

## 背景

activation-gradient product、probe、replacement、feedback contrast 和 rollout metric
分别观察不同数学对象。若 objective provenance、control 或替换分布不明确，它们容易被
错误地合并成单一“重要性”或因果结论。

## 决策

- local Taylor 只使用实际 backward objective，并保持 numerator/denominator 同口径；
- scalar objective 直接使用自身；vector objective 必须由调用方提供与真实 backward
  完全一致的 cotangent 与稳定 identity，diagnostics 只记录显式 VJP contraction；
- 多个 active microbatch 以 execution-local token 隔离，functional Tensor 只通过显式
  `TensorSite` 接点观察；
- local Taylor 明确不是 Integrated Gradients、Conductance 或 activation patching；
- probe 必须配对 grouped nested validation、random representation 与 permuted-target
  control；
- probe decodability 与 intervention effect 分开报告；
- representation cohort 必须保持稳定 target identity 与 shape；缺少精确
  activation-site identity 或 target-aligned response 时，不生成 model-use pairing；
- replacement 必须包含 identity、修改范围、replacement provenance 和 OOD 风险；
- attribution context 与 performance context 使用不同字段；
- multi-objective geometry 只有在 objective partition 闭合后才计算；
- equal-parameter control 只验证参数量和实验控制协议；通用扩展不执行调用方 training
  profile，收益判断必须 structured skip；
- fixed cohort 与 available cohort 分开输出；
- analyzer descriptor 固定声明 evidence kind 和 claim boundary；
- 缺少 null control、coverage 或合法 transaction 时只输出
  `insufficient_evidence` 或 structured skip。

## 影响

新增指标不能单独提升 evidence 等级。faithfulness 通过可证伪配对验证：一阶 attribution
对有限 intervention、decodability 对 model use、donor protocol 对 overlap、objective
geometry 对真实 objective、cohort summary 对固定支持集合。

这些诊断只描述模型、数据和协议下的证据，不声明物理因果。
