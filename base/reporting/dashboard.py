"""
提供按诊断功能分栏、自动选择比较维度的交互报告前端模板。
"""

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN" data-report-format="__REPORT_FORMAT_VERSION__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型诊断</title>
<style>
:root{color-scheme:light;--bg:#f6f8fc;--panel:#fff;--ink:#15213d;--muted:#52627d;--line:#d7e0ee;--accent:#2457f5;--soft:#e9efff;--soft-cyan:#dcf8ff;--soft-amber:#fff0c2;--bad:#e11d48;--viz-1:#2457f5;--viz-2:#00a6c8;--viz-3:#f28e00;--viz-4:#e83e8c;--viz-5:#00a86b;--viz-6:#7c3aed;--viz-7:#0088f0;--viz-8:#e11d48;--viz-negative:#e11d48;--scale-1:#7c3aed;--scale-2:#3157e8;--scale-3:#00a6c8;--scale-4:#16b77a;--scale-5:#f4b400}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,sans-serif;font-variant-numeric:tabular-nums}
main{width:min(1580px,96vw);margin:auto;padding:26px 0 70px}h1{margin:0 0 6px}h2{margin:0 0 5px}h3{margin:0}.muted{color:var(--muted)}.error{color:var(--bad)}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:var(--soft);color:var(--accent);font-size:12px;font-weight:700}
button,input,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}button{cursor:pointer}button:hover{border-color:var(--accent)}
.section-nav{display:flex;gap:7px;overflow:auto;padding:18px 0 12px}.section-nav button{white-space:nowrap}.section-nav button[aria-selected="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:start}.support{white-space:nowrap;color:var(--muted)}
.purpose-section{margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}.purpose-description{max-width:1100px}
.metric-list{display:flex;flex-wrap:wrap;gap:5px}.metric-token{font-size:12px;padding:3px 6px;border-radius:5px;background:var(--panel);border:1px solid var(--line)}.metric-token.missing{opacity:.5}
.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-items:start;gap:10px;margin-top:10px}.metric-panel{border:1px solid var(--line);border-radius:11px;padding:10px;min-width:0;overflow:hidden}.metric-panel.layout-full{grid-column:1/-1}
.metric-head{display:flex;justify-content:space-between;align-items:start;gap:14px;min-width:0}.metric-head>div{min-width:0}.metric-head code{font-size:11px;color:var(--muted);max-width:48%;overflow-wrap:anywhere;text-align:right}.metric-panel svg,.contract svg{display:block;width:100%;height:auto;margin-top:8px}.zoomable{cursor:zoom-in}
.chart-viewport{min-width:0;overflow:visible}.chart-viewport>svg{display:block;width:100%;height:auto;max-height:300px;margin:8px auto 0}.metric-panel.layout-full .chart-viewport>svg{max-height:430px}.metric-panel.layout-expanded .chart-viewport>svg,.metric-panel.layout-matrix .chart-viewport>svg{max-height:none}.metric-panel .influence-matrix svg{max-height:none}.influence-matrix{margin-top:10px}.influence-matrix>summary{cursor:pointer;font-weight:700}.influence-matrix p{margin:6px 0 8px}
.axis{stroke:var(--muted);stroke-width:1}.gridline{stroke:var(--line);stroke-width:1}.label{fill:var(--muted);font:11px ui-sans-serif,system-ui,sans-serif}
.series-line{fill:none;stroke-width:2.1;opacity:.92;stroke-linejoin:round;stroke-linecap:round;pointer-events:none}.line-hit{fill:none;stroke:transparent;stroke-width:14;pointer-events:stroke}.point-cloud{fill:none;stroke-width:1.6;stroke-linecap:round;opacity:.7;pointer-events:none}.attention-cloud{fill:none;stroke:var(--bad);stroke-width:4.8;stroke-linecap:round;pointer-events:none}.attention-segment{fill:none;stroke:var(--bad);stroke-width:2.2;stroke-linecap:round;pointer-events:none}.hover-point{fill:var(--panel);stroke:var(--accent);stroke-width:2.2;pointer-events:none;filter:drop-shadow(0 0 4px #2457f588)}.attention-point{fill:var(--bad);stroke:var(--panel);stroke-width:1}.reference-line{stroke:var(--muted);stroke-width:1.3;stroke-dasharray:5 5}.evidence-note{fill:var(--muted);font:11px ui-sans-serif,system-ui,sans-serif}.evidence-note[y="48"]{display:none}.raw-point{opacity:.5}.mean-point,.matrix-dot{stroke:var(--panel);stroke-width:1}.legend-label{fill:var(--muted);font:10px ui-sans-serif,system-ui,sans-serif}.matrix-guide{stroke:var(--line);stroke-width:1}.bar-mark{opacity:.92}.box-mark{fill:var(--soft);stroke:var(--viz-4);stroke-width:1.8}.tail-band{opacity:.17;pointer-events:none}.tail-boundary{fill:none;stroke-width:1.35;stroke-dasharray:5 4;opacity:.78;pointer-events:none}.direction-stem{stroke-width:1.6;stroke-linecap:round;opacity:.78}.facet-title{fill:var(--ink);font:600 12px ui-sans-serif,system-ui,sans-serif}.control-note{fill:var(--muted);font:11px ui-sans-serif,system-ui,sans-serif}
.matrix-cell{stroke:var(--panel);stroke-width:2}.matrix-cell-missing{stroke:var(--line);stroke-width:1}.matrix-missing-label{fill:var(--muted);font:12px ui-sans-serif,system-ui,sans-serif;pointer-events:none}
.interactive-mark{transition:opacity .1s ease,filter .1s ease,transform .1s ease;transform-box:fill-box;transform-origin:center}.chart-interactive.has-hover .interactive-mark:not(.is-linked):not(.is-hovered){opacity:.12}.chart-interactive .interactive-mark.is-linked,.chart-interactive .interactive-mark.is-hovered{opacity:1;filter:drop-shadow(0 0 3px var(--accent))}.chart-interactive circle.interactive-mark.is-hovered,.chart-interactive path.mean-point.is-hovered,.chart-interactive path.raw-point.is-hovered{transform:scale(1.8)}.chart-tooltip{position:fixed;z-index:1000;pointer-events:none;max-width:min(420px,80vw);padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);box-shadow:0 8px 30px #0003;white-space:pre-line;font-size:12px}
.single-value{display:grid;place-items:center;min-height:250px;font-size:32px}.single-value small{display:block;font-size:13px;color:var(--muted);text-align:center;max-width:390px}.status-strip{display:flex;align-items:center;gap:14px;min-height:72px;margin-top:8px;padding:10px 14px;border-left:5px solid var(--viz-5);background:color-mix(in srgb,var(--viz-5) 8%,transparent)}.status-strip.attention{border-left-color:var(--bad);background:color-mix(in srgb,var(--bad) 9%,transparent)}.status-value{font-size:24px;font-weight:700}.status-strip small{color:var(--muted)}
.metric-meta{font-size:12px;color:var(--muted);margin-top:6px}.chart-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 2px}.chart-controls[hidden]{display:none!important}.chart-controls select{min-width:180px;flex:1}.metric-panel details{margin-top:8px}.metric-panel dl{display:grid;grid-template-columns:78px 1fr;gap:6px 8px;font-size:12px}.metric-panel dt{color:var(--muted);font-weight:700}.metric-panel dd{margin:0}
.purpose-tabs{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;padding-bottom:10px;border-bottom:1px solid var(--line)}.purpose-tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}.purpose-tab-host{min-height:220px}.influence-list{margin-top:10px}.influence-list .toolbar{position:sticky;top:0;z-index:2;background:var(--panel);padding:4px 0}.influence-list .toolbar input{min-width:240px;flex:1}.influence-list .toolbar select{min-width:150px;max-width:360px}.influence-list td.effect-positive{color:#b45309;font-weight:700}.influence-list td.effect-negative{color:#047857;font-weight:700}.influence-summary{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;color:var(--muted);font-size:12px}.influence-summary span{padding:4px 7px;border-radius:999px;background:var(--soft)}
.secondary{margin-top:15px}.secondary>summary{cursor:pointer;font-weight:700}.support-details{padding-top:12px;border-top:1px dashed var(--line)}.support-details>summary{display:list-item}.support-details .chart-grid{margin-top:12px}.support-note{max-width:1100px;margin:7px 0 0}.supplementary-group{padding:12px 0;border-top:1px dashed var(--line)}.supplementary-group:first-of-type{border-top:0}.supplementary-group h4{margin:0 0 4px}.detail-chart-host{margin-top:10px}.empty{padding:32px 12px;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:10px}
.sensitivity-layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(420px,.65fr);gap:12px;margin-top:10px;align-items:start}.sensitivity-pane{min-width:0}.sensitivity-pane h4{margin:0 0 6px}.sensitivity-table{max-height:720px}.sensitivity-table th:first-child,.sensitivity-table td:first-child{position:sticky;left:0;background:var(--panel);z-index:2}.sensitivity-table th:first-child{z-index:3}.sensitivity-table td[data-sensitivity-cell]{cursor:pointer;text-align:right;font-weight:650}.sensitivity-table td[data-sensitivity-cell]:hover{outline:2px solid var(--accent);outline-offset:-2px}.sensitivity-table td.selected{outline:3px solid var(--accent);outline-offset:-3px}.sensitivity-curve svg{min-height:360px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;align-items:center}.toolbar select{max-width:min(720px,100%);flex:1}.table{overflow:auto;max-height:650px;border:1px solid var(--line);border-radius:9px}
table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;max-width:440px;overflow:hidden;text-overflow:ellipsis}th{position:sticky;top:0;background:var(--panel);z-index:1}tr:hover{background:var(--soft)}
.contracts{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:9px}.contract{border:1px solid var(--line);border-radius:9px;padding:11px}.boundary{padding-left:18px;font-size:12px}
.tree{display:grid;gap:5px}.tree div{padding:6px 9px;border-left:3px solid var(--accent);background:var(--soft);margin-left:calc(var(--depth)*18px)}pre{white-space:pre-wrap;word-break:break-word;max-height:30rem;overflow:auto}
dialog{width:min(96vw,1700px);height:min(94vh,1100px);padding:0;border:1px solid var(--line);border-radius:13px;background:var(--panel);color:var(--ink)}dialog::backdrop{background:#000a}.zoom-toolbar{display:flex;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}.zoom-toolbar strong{margin-right:auto}.zoom-axis-control{display:flex;align-items:center;gap:6px}.zoom-axis-control[hidden]{display:none}.zoom-axis-control select{padding:6px 8px}.zoom-viewport{height:calc(100% - 58px);overflow:auto;background:var(--bg);cursor:grab}.zoom-viewport.dragging{cursor:grabbing}.zoom-content{width:1400px;transform-origin:0 0;padding:20px}.zoom-content svg{display:block;width:1360px;max-height:none;height:auto;background:var(--panel)}
@media(max-width:1100px){.sensitivity-layout{grid-template-columns:1fr}}@media(max-width:900px){.chart-grid{grid-template-columns:1fr}.metric-panel.layout-full{grid-column:1}.section-head{display:block}.support{margin-top:6px}.metric-head{display:block}.metric-head code{display:block;max-width:none;margin-top:5px;text-align:left}}
</style>
</head>
<body><main>
<header><span class="badge">诊断报告 · <span id="delivery"></span></span><h1>模型诊断</h1></header>
<nav id="section-nav" class="section-nav" aria-label="诊断栏目"></nav>
<section id="diagnostic-workspace" class="panel">
  <div class="section-head"><div><h2 id="section-title"></h2><p id="section-question" class="muted"></p></div><div id="section-support" class="support"></div></div>
  <div id="section-status" class="muted">正在读取当前栏目所需证据。</div>
  <p class="muted">主区只展示能指向排查方向的组合视图；派生统计、对照、覆盖率和来源信息可在“补充证据”中按需绘图。点击任意图表可放大、缩放和拖动。</p>
  <div id="purpose-sections"></div>
</section>
<section id="evidence-workspace" class="panel" hidden>
  <h2>证据与原始数据</h2><p class="muted">这里只检查来源完整性、模块层级、结论边界和原始记录，不与模型指标混在同一栏目。</p>
  <h3>可视化覆盖</h3><div id="visualization-coverage"></div>
  <h3 style="margin-top:18px">完整字段可视化</h3>
  <p class="muted">注册诊断量在对应功能栏目中提供主图或按需图表；这里还可以为每个数值或序列字段绘制原始观测图，为类别字段绘制频数图，但不会自动赋予诊断语义。</p>
  <div class="toolbar"><select id="explorer-source"></select><select id="explorer-field"></select><button id="render-field">绘制字段</button></div>
  <div id="field-explorer" class="metric-panel"><div class="empty">选择来源和字段。</div></div>
  <h3 style="margin-top:18px">不可独立绘图字段的保留规则</h3><div id="field-retention-policy" class="table"></div>
  <h3>来源完整性</h3><div id="inventory" class="table"></div>
  <h3 style="margin-top:18px">模块层级</h3><div id="hierarchy" class="tree"></div>
  <h3 style="margin-top:18px">证据契约</h3><div id="contracts" class="contracts"></div>
  <h3 style="margin-top:18px">原始记录</h3>
  <div class="toolbar"><select id="source"></select><button id="load-source">加载</button><input id="search" placeholder="搜索当前 artifact"><select id="page-size"><option>25</option><option selected>100</option><option>512</option></select><button id="previous">上一页</button><button id="next">下一页</button></div>
  <div id="stats" class="muted">选择 artifact 后可逐行检查。</div><div id="data" class="table"></div><pre id="detail"></pre>
  <details><summary>字段可视化审计</summary><p class="muted">注册数值诊断量进入功能栏目，其余数值、序列和类别字段进入完整字段查看器；身份与状态字段用于比较和分组，摘要、提交记录和格式标识只用于校验追溯。</p><div id="field-audit" class="table"></div></details>
</section>
</main>
<div id="chart-tooltip" class="chart-tooltip" role="status" hidden></div>
<dialog id="chart-dialog"><div class="zoom-toolbar"><strong id="zoom-title">图表</strong><label id="zoom-y-scale-control" class="zoom-axis-control" hidden>纵轴尺度<select id="zoom-y-scale"><option value="linear">线性</option><option value="symlog">对称对数</option></select></label><button id="zoom-out">−</button><span id="zoom-level">100%</span><button id="zoom-in">＋</button><button id="zoom-reset">重置</button><button id="zoom-close">关闭</button></div><div id="zoom-viewport" class="zoom-viewport"><div id="zoom-content" class="zoom-content"></div></div><div id="zoom-chart-tooltip" class="chart-tooltip" role="status" hidden></div></dialog>
<script id="boot" type="application/json">__BOOTSTRAP_JSON__</script>
__PAYLOAD_ELEMENTS__
<script>
"use strict";
const BOOT=JSON.parse(document.getElementById("boot").textContent),CACHE=new Map(),PENDING=new Map();
const COMPONENTS_BY_SCOPE=new Map(Object.entries(BOOT.component_catalogs||{}).map(([scope,catalog])=>{const components=Array.isArray(catalog.components)?catalog.components:[];const byId=new Map(components.map(component=>{const label=String(component.display_label||`channel ${component.component_index??"?"}`),group=String(component.component_group||"").trim(),index=component.component_index,segment=String(component.metadata?.segment||"").trim(),segmentLabels={history:"历史输入",target:"预测窗口输入",rollout:"滚动窗口输入"},base=group?(label===group?`${group} / channel ${index??"?"}`:`${group} / ${label}`):label,qualified=component.component_kind==="input"&&segment?`${segmentLabels[segment]||segment} / ${base}`:base;return[String(component.component_id),{...component,display:qualified}]}));return[scope,byId]}));
const COLORS=["var(--viz-1)","var(--viz-2)","var(--viz-3)","var(--viz-4)","var(--viz-5)","var(--viz-6)","var(--viz-7)","var(--viz-8)"],SCALE_COLORS=["var(--scale-1)","var(--scale-2)","var(--scale-3)","var(--scale-4)","var(--scale-5)"];
const CHART={width:1000,height:430,left:96,right:28,top:68,bottom:92};
const SECTIONS=[
  {id:"training",label:"训练过程监测",question:"只展示训练时真实发生且 checkpoint 无法重建的目标、梯度和执行状态。"},
  {id:"checkpoint",label:"逐检查点异常监测",question:"在固定小样本集合上比较各检查点的输出表现、梯度健康和 Activation/Norm 分布漂移。"},
  {id:"final",label:"最终模型分析",question:"只对最终选定的单一检查点展示通道影响、模块影响、目标关系和自由滚动表现。"},
  {id:"evidence",label:"证据与原始数据",question:"检查产物完整性、模块层级、结论边界与原始记录。"}
];
const ROLE_LABELS={primary:"主要结果",intervention:"实际干预",control:"对照与基线",context:"有效性与覆盖"};
const METRIC_LABELS={
  objective_terms:"Loss Components",
  component_objective_value:"Per-Output Loss",
  raw_numerator_sum:"Pre-Normalization Output Loss",
  raw_objective_sum:"Raw Training Loss",
  backward_objective_sum:"Backward Training Loss",
  loss_cap_hit_fraction:"Loss-Cap Hit Fraction",
  gradient_norm:"Global Gradient Norm",
  rejected_microbatch_count:"Rejected Microbatches",
  grad_rms:"参数梯度 RMS",
  grad_abs_max:"参数梯度绝对最大值",
  no_grad_parameter_tensor_count:"无梯度参数张量数",
  relative_grad_rms:"相对梯度 RMS",
  gradient_energy_share:"梯度能量占比",
  grad_nonfinite_fraction:"梯度非有限比例",
  output_gradient_rms:"逐输出梯度 RMS",
  output_gradient_zero_fraction:"逐输出零梯度比例",
  output_gradient_nonfinite_fraction:"逐输出非有限梯度比例",
  objective_normalized_local_taylor:"目标归一化局部 Taylor 敏感度",
  local_taylor_abs_sum:"局部 Taylor 绝对和",
  gradient_cosine:"梯度余弦",
  norm_ratio:"梯度范数比",
  partition_coverage:"目标分区覆盖率",
  joint_active_parameter_fraction:"共同活跃参数比例",
  symmetric_relative_output_response:"对称相对输出响应",
  rmse:"RMSE",
  srmse:"sRMSE",
  rolling_local_slope_max:"局部误差斜率最大值",
  unstable_episode_count:"不稳定片段数",
  time_to_threshold_median:"达到阈值的时间中位数",
  timestep_mean_absolute_error:"逐时间步平均绝对误差",
  timestep_q90_absolute_error:"逐时间步绝对误差 q90"
};
const PROSE_REPLACEMENTS=[
  [/\bwhole line\b/gi,"整条曲线"],
  [/\bvalidity state at reference\b/gi,"有效性检查：参考值"],
  [/\bobjective identity\b/gi,"目标函数标识"],
  [/\bobjective-normalized\b/gi,"目标归一化"],
  [/\braw\/backward objective\b/gi,"原始/反向传播目标函数"],
  [/\bactivation-gradient\b/gi,"激活梯度"],
  [/\breused module\b/gi,"重复调用模块"],
  [/\benergy share\b/gi,"能量占比"],
  [/\babsolute maximum\b/gi,"绝对最大值"],
  [/\babsolute error\b/gi,"绝对误差"],
  [/\btime-normalized\b/gi,"时间归一化"],
  [/\btime-to-threshold\b/gi,"达到阈值的时间"],
  [/\bunstable episode count\b/gi,"不稳定片段数"],
  [/\brolling slope\b/gi,"滚动窗口斜率"],
  [/\bcontext-loss\b/gi,"上下文损失"],
  [/\bcontext length\b/gi,"上下文长度"],
  [/\battribution context50\/context90\b/gi,"归因上下文 50%/90%"],
  [/\bcontext50\b/gi,"上下文 50%"],
  [/\bcontext90\b/gi,"上下文 90%"],
  [/\bhalf-life\b/gi,"半衰期"],
  [/\bclaim boundary\b/gi,"结论边界"],
  [/\bdescriptor-driven summary\b/gi,"描述符驱动摘要"],
  [/\bno physical-causality inference\b/gi,"不作物理因果推断"],
  [/\blocal approximation\b/gi,"局部近似"],
  [/\bmulti-objective\b/gi,"多目标"],
  [/\bmean range\b/gi,"均值范围"],
  [/\btotal support\b/gi,"总有效观测"],
  [/\battention candidate(s)?\b/gi,"需关注位置"],
  [/\battention\b/gi,"需关注"],
  [/\bobservation(s)?\b/gi,"观测值"],
  [/\bresidual\b/gi,"残差"],
  [/\bcategory\b/gi,"类别"],
  [/\bseries index\b/gi,"序列索引"],
  [/\bindex\b/gi,"索引"],
  [/\bfinal\b/gi,"末端"],
  [/\bdirection\b/gi,"方向"],
  [/\broot output\b/gi,"根输出"],
  [/\ball input\b/gi,"全部输入"],
  [/\ball\b/gi,"全部"],
  [/\bsource scope\b/gi,"独立数据范围"],
  [/\brank-local\b/gi,"单计算卡本地"],
  [/\bCUDA allocated memory\b/gi,"CUDA 已分配显存"],
  [/\blocal Taylor\b/gi,"局部 Taylor"],
  [/\bpredictive dependence\b/gi,"预测依赖"],
  [/\bmatching feature\b/gi,"匹配特征"],
  [/\bcandidate count\b/gi,"候选数量"],
  [/\bdonor distance\b/gi,"供体距离"],
  [/\bOOD replacement\b/gi,"分布外替换"],
  [/\btensor path\b/gi,"张量路径"],
  [/\bpreserved path\b/gi,"保持不变的路径"],
  [/\bidentity-control\b/gi,"恒等对照"],
  [/\bjoint active parameter fraction\b/gi,"共同活跃参数比例"],
  [/\bnegative-dot rate\b/gi,"负点积比例"],
  [/\bparameter scope\b/gi,"参数范围"],
  [/\bphysical causality\b/gi,"物理因果"],
  [/\bfree rollout\b/gi,"自由闭环滚动预测"],
  [/\bnull control\b/gi,"空效应对照"],
  [/\bnull effect\b/gi,"空效应"],
  [/\bfixed_complete\b/gi,"固定完整样本集"],
  [/\beffect\b/gi,"响应"],
  [/\braw objective\b/gi,"原始目标函数"],
  [/\bbackward objective\b/gi,"反向传播目标函数"],
  [/\bloss cap\b/gi,"损失截断"],
  [/\bforward\b/gi,"前向传播"],
  [/\bbackward\b/gi,"反向传播"],
  [/\bobjective\b/gi,"目标函数"],
  [/\braw\b/gi,"原始"],
  [/\babsolute\b/gi,"绝对"],
  [/\brelative\b/gi,"相对"],
  [/\benergy\b/gi,"能量"],
  [/\bshare\b/gi,"占比"],
  [/\berror\b/gi,"误差"],
  [/\bloss\b/gi,"损失"],
  [/\bcap\b/gi,"截断"],
  [/\btimeline\b/gi,"时间趋势"],
  [/\bcomplete\b/gi,"完整"],
  [/\bfixed\b/gi,"固定"],
  [/\bnormalization\b/gi,"归一化"],
  [/\bnormalized\b/gi,"归一化"],
  [/\bactivation\b/gi,"激活值"],
  [/\bgradient\b/gi,"梯度"],
  [/\bcosine\b/gi,"余弦"],
  [/\bnorm\b/gi,"范数"],
  [/\bdot\b/gi,"点积"],
  [/\bintervention\b/gi,"干预"],
  [/\bdelta\b/gi,"变化量"],
  [/\bsum\b/gi,"总和"],
  [/\bestimate\b/gi,"估计"],
  [/\breplacement\b/gi,"替换"],
  [/\bdonor\b/gi,"供体"],
  [/\bcoverage\b/gi,"覆盖率"],
  [/\bsupport\b/gi,"有效观测"],
  [/\bprovenance\b/gi,"来源信息"],
  [/\bvalidity\b/gi,"有效性"],
  [/\bcontrol(s)?\b/gi,"对照"],
  [/\bidentity\b/gi,"恒等"],
  [/\bsample\b/gi,"样本"],
  [/\btarget\b/gi,"目标"],
  [/\bgroup(ed)?\b/gi,"分组"],
  [/\bcondition\b/gi,"条件"],
  [/\bsite\b/gi,"位置"],
  [/\binvocation\b/gi,"调用"],
  [/\boutput\b/gi,"输出"],
  [/\bfeature\b/gi,"特征"],
  [/\bparent\b/gi,"父层"],
  [/\bsibling(s)?\b/gi,"同级模块"],
  [/\bsingleton\b/gi,"单一成员"],
  [/\bnonfinite\b/gi,"非有限"],
  [/\bfinite\b/gi,"有限"],
  [/\bfraction\b/gi,"比例"],
  [/\bstatistics\b/gi,"统计"],
  [/\baffected\b/gi,"计划干预"],
  [/\bchanged\b/gi,"实际改变"],
  [/\belement(s)?\b/gi,"元素"],
  [/\battribution\b/gi,"归因"],
  [/\bpairing\b/gi,"配对"],
  [/\bpartition\b/gi,"分区"],
  [/\bratio\b/gi,"比值"],
  [/\bmaximum\b/gi,"最大值"],
  [/\bmax\b/gi,"最大值"],
  [/\bmedian\b/gi,"中位数"],
  [/\brolling\b/gi,"滚动窗口"],
  [/\bunstable\b/gi,"不稳定"],
  [/\bepisode(s)?\b/gi,"片段"],
  [/\bcount\b/gi,"数量"],
  [/\bmemory\b/gi,"记忆"],
  [/\blength\b/gi,"长度"],
  [/\bstate\b/gi,"状态"],
  [/\bmodule\b/gi,"模块"],
  [/\bnull\b/gi,"空效应"],
  [/\bprobe\b/gi,"探针"],
  [/\bbaseline\b/gi,"基线"],
  [/\bthreshold\b/gi,"阈值"],
  [/\bslope\b/gi,"斜率"],
  [/\bmass\b/gi,"质量"],
  [/\bfitted\b/gi,"拟合"],
  [/\bfit\b/gi,"拟合"],
  [/\binteraction\b/gi,"交互效应"],
  [/\bmatrix\b/gi,"矩阵"],
  [/\bcausality\b/gi,"因果关系"],
  [/\bcausal\b/gi,"因果"],
  [/\bCI\b/g,"置信区间"],
  [/\blag\b/gi,"滞后步"],
  [/\bcontext\b/gi,"上下文"],
  [/\brollout\b/gi,"滚动预测"],
  [/\btimestep\b/gi,"时间步"],
  [/\bhorizon\b/gi,"预测长度"],
  [/\bcohort\b/gi,"样本集合"],
  [/\bfeedback\b/gi,"反馈"],
  [/\bfactor\b/gi,"因子"],
  [/\bcell(s)?\b/gi,"组合"],
  [/\bresponse\b/gi,"响应"],
  [/\breference\b/gi,"参考值"],
  [/\bseries\b/gi,"曲线"],
  [/\bmean\b/gi,"均值"],
  [/\bupdate\b/gi,"训练步"],
  [/\brow(s)?\b/gi,"记录"],
  [/\bsource\b/gi,"来源"],
  [/\bscope\b/gi,"范围"],
  [/\bmodel\b/gi,"模型整体"],
  [/\buse\b/gi,"使用"],
  [/\bscore\b/gi,"分数"],
  [/\bscale\b/gi,"尺度"],
  [/\blayout\b/gi,"布局"],
  [/\bpath\b/gi,"路径"],
  [/\bsplit\b/gi,"划分"],
  [/\bleakage\b/gi,"泄漏"],
  [/\bmeasurement(s)?\b/gi,"测量"],
  [/\bsensitivity\b/gi,"敏感度"],
  [/\bcadence\b/gi,"采样频率"],
  [/\bprecision\b/gi,"精度"],
  [/\boptimizer\b/gi,"优化器"],
  [/\bonline\b/gi,"训练期"],
  [/\bcheckpoint\b/gi,"检查点"],
  [/\bsuccess\b/gi,"成功"],
  [/\bskipped?\b/gi,"已跳过"],
  [/\bdegraded\b/gi,"已降级"],
  [/\bcorrupt\b/gi,"已损坏"],
  [/\bunavailable\b/gi,"不可用"],
  [/\bevidence\b/gi,"证据"],
  [/\banalyzer\b/gi,"分析器"],
  [/\brun\b/gi,"运行"]
];
function localizeProse(value){let text=String(value??""),identifiers=[];text=text.replace(/\b(?:[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b/g,identifier=>{const token=`\uE000${identifiers.length}\uE001`;identifiers.push(identifier);return token});for(const [pattern,replacement] of PROSE_REPLACEMENTS)text=text.replace(pattern,replacement);return text.replace(/\uE000(\d+)\uE001/g,(_,index)=>identifiers[Number(index)]??"")}
const displayTitle=value=>String(value??"").replace(/\s+/g," ").trim();
function displayGuide(guide){return{...guide,label:displayTitle(METRIC_LABELS[guide.path]||guide.label),description:localizeProse(guide.description),reading:localizeProse(guide.reading),reference:localizeProse(guide.reference),invalid_when:localizeProse(guide.invalid_when)}}
const PURPOSES={};
PURPOSES.training=[
  {id:"objective",title:"Training Loss",description:"分别查看 objective_terms 中的 loss components、逐输出 loss 和实际 backward 总 loss。所有曲线都来自真实训练 ledger，不增加 forward 或 backward。高密度训练曲线只绘制折线路径，不创建逐点详情。",paths:["objective_terms","component_objective_value","raw_numerator_sum","raw_objective_sum","backward_objective_sum","loss_cap_hit_fraction"]},
  {id:"gradient",title:"Gradient Health",description:"查看实际 backward 总 loss 的全局参数梯度范数随训练步的变化。现有 flight recorder 没有执行逐 loss 的额外 autograd，因此不能把这条曲线拆成各 loss component 的参数梯度范数。非有限梯度、optimizer、GradScaler 和 scheduler 终态作为原始事务审计字段保留。",paths:["gradient_norm"]},
  {id:"execution",title:"Training Execution",description:"查看 update 是否发生 microbatch 回滚。处理数、提交数和 milestone 是事务完整性事实，只在原始证据中审计。",paths:["rejected_microbatch_count"]}
];
PURPOSES.checkpoint=[
  {id:"output",title:"Per-Output Checkpoint Trends",description:"固定同一小样本集合，每个输出通道一条线，比较验证指标与直接输出梯度。重点看 loss/gradient 的持续失衡、突变和相对收敛差异。",paths:["response_value","output_gradient_rms","output_gradient_zero_fraction","output_gradient_nonfinite_fraction","raw_objective","backward_objective"]},
  {id:"gradient",title:"Parameter Gradient Health",description:"每个参数叶子层一条线，查看相对梯度尺度、孤立尖峰、能量集中、零/非有限梯度和 no-grad；不对容器模块重复统计。",paths:["relative_grad_rms","grad_abs_max","gradient_energy_share","grad_nonfinite_fraction","grad_zero_fraction","no_grad_parameter_tensor_count"]},
  {id:"distribution",title:"Activation and Normalization Drift",description:"自动监测每个实际 Activation/Norm nn.Module；module_path 与 tap_id 共同标识层，每个层各一条曲线。查看 RMS、标准差、峰值、零值比例、非有限比例及运行统计随检查点的变化。",paths:["output_rms","output_std","output_abs_max","output_zero_fraction","output_nonfinite_fraction","normalization_running_mean_rms","normalization_running_var_mean"]},
  {id:"runtime",title:"Checkpoint Diagnostics Runtime",description:"显示每个 checkpoint 的实际诊断耗时。所有 checkpoint 耗时之和用于直接核对 100 个 checkpoint 不超过两小时的目标。",paths:["elapsed_seconds"]}
];
PURPOSES.final=[
  {id:"channel",title:"Input-to-Output Influence",description:"在相同样本和输出指标下，只替换一个输入通道，再与未替换的基线比较。effect_value 已统一方向：正值表示替换后输出指标变差，负值表示改善。列表给出全部输入×输出组合，色块矩阵用于概览；它反映模型在该干预下的预测依赖，不表示物理因果。",categories:["input"],paths:["effect_value","normalized_effect"]},
  {id:"sensitivity",title:"Input Perturbation Sensitivity",description:"对每个输入通道在模型实际输入坐标中施加多个幅度的正负配对扰动，以扰动前后输出自身的 RMS 作为相对尺度。尺度表格用于查看完整 Sij，曲线用于查看指定输入—输出组合的相对响应如何随扰动幅度变化；它描述模型预测敏感度，不表示物理因果。",categories:["input"],paths:["symmetric_relative_output_response"]},
  {id:"module",title:"Stage/Block Influence",description:"展示 final cohort 上全部已确认 Stage 和 Block，而不是只展示一个模块。可按层级、具体模块、干预方法和输出通道筛选；effect_value 正值表示输出指标变差，负值表示改善。相对响应放在补充证据，真实逐输出干预与局部 Taylor 分开展示。",categories:["module"],paths:["effect_value","normalized_effect","objective_normalized_local_taylor"]},
  {id:"objective",title:"Multi-Objective Gradient Geometry",description:"比较每个输出目标与其余目标的梯度余弦、范数比，并同时核对目标分区和共同活跃参数覆盖。有效监督 support 在单元格悬停和原始记录中核对。它只描述最终参数点附近的一阶优化关系。",categories:["multi_objective"],paths:["gradient_cosine","norm_ratio","partition_coverage","joint_active_parameter_fraction"]},
  {id:"rollout",title:"Closed-Loop Rollout",description:"只运行显式配置的 free rollout。先选择输出通道，再查看该通道随时间步变化的均值与 q90 尾部误差，避免把全部通道叠成不可读曲线。",categories:["rollout"],paths:["rmse","srmse","timestep_mean_absolute_error","timestep_q90_absolute_error","rolling_local_slope_max","unstable_episode_count","time_to_threshold_median"]}
];
// 主页面只保留能直接指向排查方向的测量；其余充分统计、controls 与 provenance
// 仍保留在 artifact 和下钻清单中，避免把“保留证据”误写成“一字段一张图”。
const DIRECTION_PATHS=new Set([
  "objective_terms",
  "effect_value",
  "symmetric_relative_output_response",
  "response_value",
  "raw_objective",
  "backward_objective",
  "gradient_norm",
  "output_gradient_rms",
  "output_gradient_zero_fraction",
  "rejected_microbatch_count",
  "rmse",
  "srmse",
  "timestep_mean_absolute_error",
  "timestep_q90_absolute_error",
  "rolling_local_slope_max",
  "unstable_episode_count",
  "time_to_threshold_median",
  "output_rms",
  "output_std",
  "output_abs_max",
  "output_zero_fraction",
  "objective_normalized_local_taylor",
  "relative_grad_rms",
  "grad_abs_max",
  "grad_zero_fraction",
  "no_grad_parameter_tensor_count",
  "gradient_energy_share",
  "normalization_running_mean_rms",
  "normalization_running_var_mean",
  "elapsed_seconds",
  "gradient_cosine",
  "norm_ratio"
]);
const STATUS_PATHS=new Set([
  "loss_cap_hit_fraction",
  "output_nonfinite_fraction",
  "output_gradient_nonfinite_fraction",
  "grad_nonfinite_fraction",
  "partition_coverage",
  "joint_active_parameter_fraction"
]);
const RAW_ONLY_PATHS=new Set();
const COMPOSITE_VIEWS={
  "training.objective":[
    {kind:"objective_terms",title:"Loss Components",paths:["objective_terms"]},
    {kind:"trend",title:"Per-Output Loss",coordinate:"update",paths:["component_objective_value"]},
    {kind:"trend",title:"Total Loss",coordinate:"update",paths:["raw_objective_sum","backward_objective_sum"]}
  ],
  "checkpoint.output":[
    {kind:"trend",title:"Checkpoint Loss",coordinate:"checkpoint_update",paths:["raw_objective","backward_objective"]},
    {kind:"trend",title:"Invalid Output-Gradient Fraction",coordinate:"checkpoint_update",paths:["output_gradient_zero_fraction","output_gradient_nonfinite_fraction"]}
  ],
  "final.objective":[
    {kind:"objective_overview",title:"Multi-Objective Gradient Geometry",paths:["gradient_cosine","norm_ratio","partition_coverage","joint_active_parameter_fraction"]}
  ],
  "final.sensitivity":[
    {kind:"input_sensitivity",title:"Multi-Scale Input-to-Output Sensitivity",paths:["symmetric_relative_output_response"]}
  ],
  "final.rollout":[
    {kind:"rollout_overview",title:"Closed-Loop Rollout by Horizon",paths:["rmse","srmse","rolling_local_slope_max","unstable_episode_count","time_to_threshold_median","timestep_mean_absolute_error","timestep_q90_absolute_error"]},
    {kind:"rollout_heatmap",title:"Rollout Metric Overview",paths:["rmse","srmse","rolling_local_slope_max","unstable_episode_count"]},
    {kind:"rollout_coverage",title:"Rollout Cohort Coverage",paths:["rmse"]}
  ]
};
const DIMENSIONS={
  training:["update","checkpoint_update","response_component_id","node_id","module_path"],
  module:["update","response_component_id","module_site_id","__condition__","node_id","module_path","branch","sample_id","invocation_index"],
  parameter:["update","__condition__","node_id","module_path","sample_id"],
  input:["scale","response_component_id","intervened_component_id","__condition__","sample_id","group_id","position"],
  multi_objective:["response_component_id","scope","node_id","module_path","group_id","sample_id"],
  rollout:["timestep","horizon","rollout_condition_id","scenario_id","response_component_id","sample_id","position"],
  runtime:["update","node_id","module_path","rank"],
  explorer:["update","timestep","lag","horizon","__condition__","intervened_component_id","response_component_id","module_site_id","node_id","module_path","scope","scenario_id","sample_id","group_id","position","rank","invocation_index"]
};
const DIMENSION_LABELS={
  update:"训练步",
  checkpoint_update:"检查点训练步",
  timestep:"时间步",
  lag:"滞后步",
  context_length:"上下文长度",
  horizon:"预测长度",
  position:"样本位置",
  intervened_component_id:"被干预输入组件",
  response_component_id:"响应输出组件",
  module_site_id:"模块位置",
  activation_site_id:"激活位置",
  module_path:"模块路径",
  node_id:"模块节点",
  output_path:"输出张量路径",
  invocation_index:"调用序号",
  method:"干预方法",
  scale:"缩放系数",
  condition_id:"实验条件",
  rollout_condition_id:"滚动预测条件",
  scenario_id:"场景",
  cohort_policy:"样本集合策略",
  sample_id:"样本",
  group_id:"样本组",
  scope:"参数范围",
  rank:"进程编号",
  status:"状态",
  skip_reason:"跳过原因",
  __condition__:"实验条件",
  __module_site__:"模块位置",
  __input_condition__:"输入组件与干预条件",
  __module_condition__:"模块位置与干预条件",
  __row_index__:"记录序号",
  "series index":"序列索引",
  "context length":"上下文长度"
};
const VALUE_LABELS={
  free:"自由反馈",
  fixed_complete:"固定完整样本集",
  available:"可用样本集",
  identity:"恒等对照",
  output_scale:"输出缩放",
  detach:"停止梯度",
  constant_patch:"常数替换",
  mean:"均值替换",
  mean_patch:"均值替换",
  explicit_donor_patch:"显式供体替换",
  training_mean_patch:"训练均值替换",
  unconditional_resample_patch:"无条件重采样替换",
  regime_matched_resample_patch:"工况匹配重采样替换",
  conditional_knn_resample_patch:"条件近邻重采样替换",
  success:"成功",
  insufficient_evidence:"证据不足",
  empty_eligible_cohort:"没有符合条件的样本",
  skipped:"已跳过",
  failed:"失败",
  missing:"缺失"
};
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const show=v=>typeof v==="object"?JSON.stringify(v):String(v??"null");
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const leaf=path=>String(path).split(".").pop();
const unique=values=>[...new Set(values.filter(value=>value!==null&&value!==undefined).map(value=>String(value)))];
const uniqueNumbers=values=>[...new Set(values.map(value=>Number(value)).filter(value=>Number.isFinite(value)))];
const dimensionLabel=name=>DIMENSION_LABELS[name]||localizeProse(String(name??""));
const displayValue=value=>VALUE_LABELS[String(value)]||String(value??"");
const COLUMN_LABELS={metric:"诊断量",path:"字段名",usage:"保留原因",availability:"本次状态",item:"覆盖项目",count:"数量",meaning:"说明",classification:"字段类型",visualization:"展示方式",reason:"保留原因",kind:"字段类型",retain:"保留理由",group:"来源类别",content_kind:"内容类型",label:"名称",format:"格式",row_count:"记录数",source_bytes:"原始字节数",available:"可用"};
const DELIVERY_LABELS={self_contained_gzip_embedded:"自包含单文件"};
function extent(values){let low=Math.min(...values),high=Math.max(...values);if(!Number.isFinite(low)||!Number.isFinite(high))return[0,1];if(low===high){const margin=Math.max(Math.abs(low)*.1,1e-6);return[low-margin,high+margin]}return[low,high]}
function table(rows,columns){if(!rows.length)return'<div class="empty">没有记录</div>';return`<table><thead><tr>${columns.map(c=>`<th>${esc(COLUMN_LABELS[c]||c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r,i)=>`<tr data-row="${i}">${columns.map(c=>`<td title="${esc(show(r[c]))}">${esc(show(r[c]))}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
const ZOOM_DIALOG=document.getElementById("chart-dialog"),ZOOM_VIEWPORT=document.getElementById("zoom-viewport"),ZOOM_CONTENT=document.getElementById("zoom-content"),ZOOM_Y_SCALE_CONTROL=document.getElementById("zoom-y-scale-control"),ZOOM_Y_SCALE=document.getElementById("zoom-y-scale"),CHART_TOOLTIP=document.getElementById("chart-tooltip"),ZOOM_CHART_TOOLTIP=document.getElementById("zoom-chart-tooltip");let zoomScale=1,zoomSource=null,zoomTitle="";
function applyZoom(){ZOOM_CONTENT.style.transform=`scale(${zoomScale})`;document.getElementById("zoom-level").textContent=`${Math.round(zoomScale*100)}%`}
function positionChartTooltip(event,tooltip){const gap=14,box=tooltip.getBoundingClientRect(),left=Math.min(event.clientX+gap,window.innerWidth-box.width-gap),top=Math.min(event.clientY+gap,window.innerHeight-box.height-gap);tooltip.style.left=`${Math.max(gap,left)}px`;tooltip.style.top=`${Math.max(gap,top)}px`}
function clearChartHover(svg){const state=svg.__interactionState;if(state){if(state.hovered)state.hovered.classList.remove("is-hovered");state.linked.forEach(mark=>mark.classList.remove("is-linked"));state.hovered=null;state.linked=[];if(state.overlay)state.overlay.setAttribute("visibility","hidden");state.tooltip.hidden=true}svg.classList.remove("has-hover")}
function interactiveMarkAtPointer(event,svg){const candidates=[],seen=new Set(),elements=typeof document.elementsFromPoint==="function"?document.elementsFromPoint(event.clientX,event.clientY):[event.target];for(const element of elements){const mark=element.closest?.(".interactive-mark");if(!mark||!svg.contains(mark)||seen.has(mark))continue;seen.add(mark);candidates.push(mark)}const point=candidates.find(mark=>mark.tagName.toLowerCase()==="circle");if(point)return point;const line=candidates.find(mark=>mark.classList.contains("line-hit"));return line??candidates[0]??event.target.closest?.(".interactive-mark")}
// 密集图只保留一个 hover DOM 点；按 SVG 横坐标分桶后查找最近原始观测。
function registerHoverPoints(svg,points){const buckets=new Map();points.forEach(point=>{const bucket=Math.round(point.x);if(!buckets.has(bucket))buckets.set(bucket,[]);buckets.get(bucket).push(point)});svg.__pointBuckets=buckets}
function nearestHoverPoint(event,svg){const buckets=svg.__pointBuckets;if(!buckets?.size)return null;const matrix=svg.getScreenCTM();if(!matrix)return null;const local=svg.createSVGPoint();local.x=event.clientX;local.y=event.clientY;const point=local.matrixTransform(matrix.inverse()),bounds=svg.getBoundingClientRect(),viewBox=(svg.getAttribute("viewBox")||"0 0 1000 430").split(/\s+/).map(Number),threshold=7*(viewBox[2]||1000)/Math.max(bounds.width,1),radius=Math.ceil(threshold);let best=null,bestDistance=threshold*threshold;for(let offset=-radius;offset<=radius;offset++){for(const candidate of buckets.get(Math.round(point.x)+offset)||[]){const distance=(candidate.x-point.x)**2+(candidate.y-point.y)**2;if(distance<=bestDistance){best=candidate;bestDistance=distance}}}return best}
function bindChartInteractions(svg){svg.classList.add("chart-interactive");const bySeries=new Map();svg.querySelectorAll("[data-series-key]").forEach(mark=>{const key=mark.dataset.seriesKey;if(!bySeries.has(key))bySeries.set(key,[]);bySeries.get(key).push(mark)});const tooltip=svg.closest("dialog")?ZOOM_CHART_TOOLTIP:CHART_TOOLTIP,state={bySeries,hovered:null,linked:[],overlay:svg.querySelector(".hover-point"),pending:null,frame:0,tooltip};svg.__interactionState=state;const showHover=(event,point,mark)=>{if(state.hovered)state.hovered.classList.remove("is-hovered");state.linked.forEach(item=>item.classList.remove("is-linked"));state.hovered=mark||null;state.linked=[];const key=point?.seriesKey||mark?.dataset.seriesKey;if(mark)mark.classList.add("is-hovered");if(key){state.linked=state.bySeries.get(key)||[];state.linked.forEach(item=>item.classList.add("is-linked"))}if(point&&state.overlay){state.overlay.setAttribute("cx",point.x);state.overlay.setAttribute("cy",point.y);state.overlay.setAttribute("stroke",point.attention?"var(--bad)":"var(--accent)");state.overlay.setAttribute("visibility","visible")}else if(state.overlay)state.overlay.setAttribute("visibility","hidden");const detail=point?.tooltip||mark?.dataset.tooltip||"";if(!detail){clearChartHover(svg);return}svg.classList.add("has-hover");tooltip.textContent=detail;tooltip.hidden=false;positionChartTooltip(event,tooltip)};svg.onpointermove=event=>{state.pending={clientX:event.clientX,clientY:event.clientY,target:event.target};if(state.frame)return;state.frame=requestAnimationFrame(()=>{state.frame=0;const pending=state.pending;if(!pending)return;const point=nearestHoverPoint(pending,svg);if(point){showHover(pending,point,null);return}const mark=interactiveMarkAtPointer(pending,svg);if(mark&&svg.contains(mark))showHover(pending,null,mark);else clearChartHover(svg)})};svg.onpointerleave=()=>{if(state.frame)cancelAnimationFrame(state.frame);state.frame=0;state.pending=null;clearChartHover(svg)}}
function renderZoomChart(){if(!zoomSource)return;ZOOM_CONTENT.innerHTML="";let rendered;if(typeof zoomSource.__zoomRender==="function"){rendered=document.createElementNS("http://www.w3.org/2000/svg","svg");rendered.setAttribute("role","img");rendered.setAttribute("aria-label",zoomTitle);ZOOM_CONTENT.appendChild(rendered);zoomSource.__zoomRender(rendered,ZOOM_Y_SCALE.value)}else{rendered=zoomSource.cloneNode(true);rendered.removeAttribute("id");rendered.classList.remove("zoomable");rendered.removeAttribute("tabindex");ZOOM_CONTENT.appendChild(rendered);bindChartInteractions(rendered)}const viewBox=(rendered.getAttribute("viewBox")||"0 0 1000 430").split(/\s+/).map(Number),contentWidth=Math.max(1360,Number.isFinite(viewBox[2])?viewBox[2]:1000);rendered.style.width=`${contentWidth}px`;ZOOM_CONTENT.style.width=`${contentWidth+40}px`}
function openZoom(svg,title){zoomSource=svg;zoomTitle=title||svg.getAttribute("aria-label")||"图表";const supportsYScale=Boolean(svg.__supportsYScale&&typeof svg.__zoomRender==="function");CHART_TOOLTIP.hidden=true;ZOOM_CHART_TOOLTIP.hidden=true;ZOOM_Y_SCALE_CONTROL.hidden=!supportsYScale;ZOOM_Y_SCALE.value="linear";renderZoomChart();document.getElementById("zoom-title").textContent=zoomTitle;zoomScale=1;applyZoom();ZOOM_VIEWPORT.scrollTo(0,0);ZOOM_DIALOG.showModal()}
function bindZoom(host,title){host.querySelectorAll("svg").forEach(svg=>{svg.classList.add("zoomable");svg.setAttribute("tabindex","0");svg.setAttribute("title","点击放大；放大后可滚轮缩放和拖动");svg.onclick=()=>openZoom(svg,title||svg.getAttribute("aria-label"));svg.onkeydown=event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();openZoom(svg,title||svg.getAttribute("aria-label"))}}})}
document.getElementById("zoom-in").onclick=()=>{zoomScale=Math.min(4,zoomScale*1.25);applyZoom()};document.getElementById("zoom-out").onclick=()=>{zoomScale=Math.max(.35,zoomScale/1.25);applyZoom()};document.getElementById("zoom-reset").onclick=()=>{zoomScale=1;applyZoom();ZOOM_VIEWPORT.scrollTo(0,0)};document.getElementById("zoom-close").onclick=()=>ZOOM_DIALOG.close();ZOOM_DIALOG.addEventListener("close",()=>{const svg=ZOOM_CONTENT.querySelector("svg");if(svg)clearChartHover(svg);ZOOM_CHART_TOOLTIP.hidden=true});ZOOM_VIEWPORT.onwheel=event=>{event.preventDefault();zoomScale=Math.max(.35,Math.min(4,zoomScale*(event.deltaY<0?1.12:.89)));applyZoom()};let drag=null;ZOOM_VIEWPORT.onpointerdown=event=>{drag={x:event.clientX,y:event.clientY,left:ZOOM_VIEWPORT.scrollLeft,top:ZOOM_VIEWPORT.scrollTop};ZOOM_VIEWPORT.setPointerCapture(event.pointerId);ZOOM_VIEWPORT.classList.add("dragging")};ZOOM_VIEWPORT.onpointermove=event=>{if(!drag)return;ZOOM_VIEWPORT.scrollLeft=drag.left-(event.clientX-drag.x);ZOOM_VIEWPORT.scrollTop=drag.top-(event.clientY-drag.y)};ZOOM_VIEWPORT.onpointerup=()=>{drag=null;ZOOM_VIEWPORT.classList.remove("dragging")};
ZOOM_Y_SCALE.onchange=()=>{renderZoomChart();ZOOM_VIEWPORT.scrollTo(0,0)};
function flatten(value,prefix="",out={},depth=0){if(depth>6)return out;if(Array.isArray(value)){out[prefix]=value;return out}if(value&&typeof value==="object"){for(const [key,item] of Object.entries(value))flatten(item,prefix?`${prefix}.${key}`:key,out,depth+1);return out}if(prefix)out[prefix]=value;return out}
function enrichComponentIdentity(row,source){const catalog=COMPONENTS_BY_SCOPE.get(String(source.scope_id||""));if(!catalog)return row;let enriched=row;for(const field of["intervened_component_id","response_component_id"]){const id=row[field],component=id==null?null:catalog.get(String(id));if(component){if(enriched===row)enriched={...row};enriched[`__raw_${field}`]=id;enriched[field]=component.display}}return enriched}
// 解压、JSON 解析和可选 flatten 全部在 Worker 中完成，避免大 source 阻塞页面交互。
const DATA_WORKER_SOURCE=`"use strict";
function flatten(value,prefix="",out={},depth=0){if(depth>6)return out;if(Array.isArray(value)){out[prefix]=value;return out}if(value&&typeof value==="object"){for(const [key,item] of Object.entries(value))flatten(item,prefix?prefix+"."+key:key,out,depth+1);return out}if(prefix)out[prefix]=value;return out}
async function gunzip(bytes){if(typeof DecompressionStream==="undefined")throw new Error("浏览器不支持 DecompressionStream");const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));return new Uint8Array(await new Response(stream).arrayBuffer())}
function parseRows(text,format){if(format==="json")return text.trim()?[JSON.parse(text)]:[];if(format==="csv"){const lines=text.split(/\\r?\\n/).filter(Boolean),head=(lines.shift()||"").split(",");return lines.map(line=>Object.fromEntries(line.split(",").map((value,index)=>[head[index],value])))}return text.split(/\\r?\\n/).filter(line=>line.trim()).map(line=>JSON.parse(line))}
const CACHE=new Map(),INFLIGHT=new Map();let cacheBytes=0;const CACHE_LIMIT=128*1024*1024;
function leaf(path){const parts=String(path).split(".");return parts[parts.length-1]}
function pathMatches(metricPath,candidate){return candidate===metricPath||leaf(candidate)===metricPath}
async function decodeRows(data,flat){const decoder=new TextDecoder(),parts=[];for(let index=0;index<data.chunks.length;index++){const encoded=data.chunks[index],packed=Uint8Array.from(atob(encoded),character=>character.charCodeAt(0));parts.push(decoder.decode(await gunzip(packed),{stream:index+1<data.chunks.length}))}parts.push(decoder.decode());const rows=parseRows(parts.join(""),data.format);return flat?rows.map(row=>flatten(row)):rows}
function cacheRows(key,rows,bytes){if(CACHE.has(key)){cacheBytes-=CACHE.get(key).bytes;CACHE.delete(key)}CACHE.set(key,{rows,bytes});cacheBytes+=bytes;while(cacheBytes>CACHE_LIMIT&&CACHE.size>1){const oldest=CACHE.keys().next().value,entry=CACHE.get(oldest);CACHE.delete(oldest);cacheBytes-=entry.bytes}}
async function flatRows(data){const key=String(data.sourceKey||"");if(CACHE.has(key)){const entry=CACHE.get(key);CACHE.delete(key);CACHE.set(key,entry);return entry.rows}if(INFLIGHT.has(key))return INFLIGHT.get(key);const pending=decodeRows(data,true);INFLIGHT.set(key,pending);try{const rows=await pending;cacheRows(key,rows,Number(data.sourceBytes||0));return rows}finally{INFLIGHT.delete(key)}}
function valueByLeaf(row,name){const path=Object.keys(row).find(candidate=>leaf(candidate)===name);return path?row[path]:null}
function projectObservations(rows,data){const keep=new Set(data.identityLeaves||[]),observations=[];for(let rowIndex=0;rowIndex<rows.length;rowIndex++){const row=rows[rowIndex],rowEvidenceKind=valueByLeaf(row,"evidence_kind");if(rowEvidenceKind&&data.evidenceKind&&rowEvidenceKind!==data.evidenceKind)continue;const projected={};for(const [path,value]of Object.entries(row)){if(keep.has(leaf(path)))projected[path]=value}for(const [path,value]of Object.entries(row)){if(!pathMatches(data.metricPath,path))continue;if(Array.isArray(value)){value.forEach((item,seriesIndex)=>{if(Number.isFinite(item)||data.includeMissing&&item==null)observations.push({value:item,missing:!Number.isFinite(item),row:projected,rowIndex,seriesIndex,path})})}else if(Number.isFinite(value)||data.includeMissing&&value==null)observations.push({value,missing:!Number.isFinite(value),row:projected,rowIndex,seriesIndex:null,path})}}return observations}
function projectTrainingSeries(rows,data){const paths=new Set(data.metricPaths||[]),coordinateLeaf=String(data.coordinateLeaf||"update"),records=[];for(const row of rows){const x=Number(valueByLeaf(row,coordinateLeaf));if(!Number.isFinite(x))continue;for(const [path,value]of Object.entries(row)){if(paths.has(path)&&Number.isFinite(value))records.push([path,x,value])}}return records}
self.onmessage=async event=>{const data=event.data,{id,mode}=data;try{if(mode==="observations"){const rows=await flatRows(data);self.postMessage({id,rows:projectObservations(rows,data)});return}if(mode==="training_series"){const rows=await flatRows(data);self.postMessage({id,rows:projectTrainingSeries(rows,data)});return}const rows=mode==="flat"?await flatRows(data):await decodeRows(data,false);self.postMessage({id,rows})}catch(error){self.postMessage({id,error:String(error?.message||error)})}}`;
const DATA_WORKER=new Worker(URL.createObjectURL(new Blob([DATA_WORKER_SOURCE],{type:"text/javascript"}))),WORKER_REQUESTS=new Map();let workerRequestId=0,cacheSourceBytes=0;
DATA_WORKER.onmessage=event=>{const pending=WORKER_REQUESTS.get(event.data.id);if(!pending)return;WORKER_REQUESTS.delete(event.data.id);if(event.data.error)pending.reject(new Error(event.data.error));else pending.resolve(event.data.rows)};
DATA_WORKER.onerror=event=>{for(const pending of WORKER_REQUESTS.values())pending.reject(new Error(event.message||"诊断数据 worker 失败"));WORKER_REQUESTS.clear()};
function parseEmbeddedRows(descriptor,mode,options={}){const chunks=descriptor.chunks.map(reference=>{const element=document.getElementById(reference.embedded_key);if(!element)throw new Error(`缺少内嵌数据块 ${reference.embedded_key}`);return element.textContent.trim()}),id=++workerRequestId;return new Promise((resolve,reject)=>{WORKER_REQUESTS.set(id,{resolve,reject});DATA_WORKER.postMessage({id,chunks,format:descriptor.format,mode,sourceKey:descriptor.name,sourceBytes:Number(descriptor.embedded_source_bytes||descriptor.source_bytes||0),...options})})}
function cachedRows(key){const entry=CACHE.get(key);if(!entry)return null;CACHE.delete(key);CACHE.set(key,entry);return entry.rows}
function storeRows(key,rows,bytes){if(CACHE.has(key)){cacheSourceBytes-=CACHE.get(key).bytes;CACHE.delete(key)}CACHE.set(key,{rows,bytes});cacheSourceBytes+=bytes;while(cacheSourceBytes>96*1024*1024&&CACHE.size>1){const oldest=CACHE.keys().next().value,entry=CACHE.get(oldest);CACHE.delete(oldest);cacheSourceBytes-=entry.bytes}}
async function load(name,mode="flat"){const key=`${mode}:${name}`,cached=cachedRows(key);if(cached)return cached;if(PENDING.has(key))return PENDING.get(key);const descriptor=BOOT.artifacts.find(item=>item.name===name);if(!descriptor||!descriptor.available)return[];const pending=parseEmbeddedRows(descriptor,mode);PENDING.set(key,pending);try{const rows=await pending;storeRows(key,rows,Number(descriptor.embedded_source_bytes||descriptor.source_bytes||0));return rows}finally{PENDING.delete(key)}}
const GUIDES=[...(BOOT.metric_guides||[])],AVAILABLE=BOOT.artifacts.filter(item=>item.available),STREAMS=AVAILABLE.filter(item=>item.content_kind==="evidence_stream");
const EXECUTED_KINDS=new Set((BOOT.evidence_figures||[]).map(item=>item.evidence_kind)),EXECUTED_ANALYZERS=[...(BOOT.executed_analyzers||[])];
const COMPONENT_GUIDE_CATEGORIES=new Set(["input","module","multi_objective","rollout"]);
function pathMatches(metricPath,candidate){return candidate===metricPath||leaf(candidate)===metricPath}
function objectiveTermPath(path){return/(^|\.)objective_terms\.[^.]+$/.test(String(path))}
function guideCoversPath(guide,path){return guide.path==="objective_terms"?objectiveTermPath(path):pathMatches(guide.path,path)}
function sourceSupportsGuide(source,guide){if(!source.numeric_paths.some(path=>guideCoversPath(guide,path)))return false;if(guide.evidence_kind)return source.evidence_kinds.includes(guide.evidence_kind);return source.group==="online"}
function guideSources(guide){const candidates=STREAMS.filter(source=>sourceSupportsGuide(source,guide)),online=candidates.filter(source=>source.group==="online"),checkpoint=candidates.filter(source=>source.group==="checkpoint");if(!checkpoint.length)return online;const byScope=new Map();checkpoint.forEach(source=>{const scope=String(source.scope_id||source.name);if(!byScope.has(scope))byScope.set(scope,[]);byScope.get(scope).push(source)});const selected=[...byScope.values()].sort((left,right)=>right.reduce((total,source)=>total+Number(source.row_count||0),0)-left.reduce((total,source)=>total+Number(source.row_count||0),0))[0]||[];return[...online,...selected]}
const guideAvailable=guide=>guideSources(guide).length>0;
const CHECKPOINT_EVIDENCE_KINDS=new Set(["checkpoint_training_health"]);
const FINAL_EVIDENCE_KINDS=new Set(["checkpoint_conditioned_activation_intervention","predictive_dependence_intervention","predictive_input_sensitivity","objective_gradient_geometry","rollout_stability_and_cohort_comparison"]);
function purposePathSet(section){return new Set((PURPOSES[section]||[]).flatMap(item=>item.paths||[]))}
function guidesForSection(category){if(category==="training"){const paths=purposePathSet("training");return GUIDES.filter(guide=>!guide.evidence_kind&&paths.has(guide.path))}if(category==="checkpoint")return GUIDES.filter(guide=>CHECKPOINT_EVIDENCE_KINDS.has(guide.evidence_kind));if(category==="final")return GUIDES.filter(guide=>FINAL_EVIDENCE_KINDS.has(guide.evidence_kind));return GUIDES.filter(guide=>guide.category===category)}
function categoryState(category){const guides=guidesForSection(category),available=guides.filter(guideAvailable).length,kinds=unique(guides.map(guide=>guide.evidence_kind).filter(Boolean)),executed=kinds.some(kind=>EXECUTED_KINDS.has(kind));if(available)return{label:"已有数据",kind:"available",available,total:guides.length,kinds};if(kinds.length&&!executed)return{label:"未运行",kind:"not_run",available:0,total:guides.length,kinds};return{label:"无可用证据",kind:"no_evidence",available:0,total:guides.length,kinds}}
function metricPaths(row,guide,includeMissing=false){return Object.keys(row).filter(path=>pathMatches(guide.path,path)&&(finite(row[path])||Array.isArray(row[path])&&row[path].some(finite)||includeMissing&&row[path]==null))}
function dimensionValue(row,name){if(name==="__condition__"){const method=Object.keys(row).find(path=>leaf(path)==="method"),scale=Object.keys(row).find(path=>leaf(path)==="scale"),condition=Object.keys(row).find(path=>leaf(path)==="condition_id");if(method&&row[method]!=null)return`${displayValue(row[method])}${scale&&row[scale]!=null?`：${row[scale]}`:""}`;return condition?displayValue(row[condition]):null}if(name==="__module_site__"){const module=dimensionValue(row,"module_path")??dimensionValue(row,"node_id")??dimensionValue(row,"module_site_id")??dimensionValue(row,"activation_site_id"),output=dimensionValue(row,"output_path"),invocation=dimensionValue(row,"invocation_index");return[module??"模型整体",output==null?null:`输出路径：${output}`,invocation==null?null:`第 ${invocation} 次调用`].filter(Boolean).join(" · ")}if(name==="__input_condition__"){const component=dimensionValue(row,"intervened_component_id"),condition=dimensionValue(row,"__condition__");return[component??"输入组件",condition].filter(Boolean).join(" · ")}if(name==="__module_condition__"){const site=dimensionValue(row,"__module_site__"),condition=dimensionValue(row,"__condition__");return[site,condition].filter(Boolean).join(" · ")}const path=Object.keys(row).find(candidate=>leaf(candidate)===name);return path?row[path]:null}
function chooseDimension(rows,category,excluded=null){for(const candidate of DIMENSIONS[category]||[]){if(candidate===excluded)continue;const raw=rows.map(row=>dimensionValue(row,candidate)).filter(value=>value!==null&&value!==undefined),values=unique(raw),numeric=raw.length>0&&raw.every(value=>finite(value));if(values.length>=2)return candidate}return rows.length>=2?"__row_index__":null}
function identityText(row){return["update","intervened_component_id","response_component_id","module_path","module_type","tap_id","method","scale","condition_id","scenario_id","sample_id","group_id","position","rank"].map(name=>{const value=dimensionValue(row,name);return value==null?null:`${dimensionLabel(name)}：${displayValue(value)}`}).filter(Boolean).join(" · ")}
const OBSERVATION_IDENTITY_LEAVES=["evidence_kind","update","checkpoint_update","timestep","lag","horizon","response_component_id","intervened_component_id","module_site_id","module_path","module_type","tap_id","node_id","scope","output_path","invocation_index","method","scale","condition_id","rollout_condition_id","scenario_id","cohort_policy","sample_id","group_id","position","rank","control_status","skip_reason","status","branch","partition","metric_id","support_count","baseline_support_count","condition_support_count","effect_value_min","effect_value_max","normalized_effect_min","normalized_effect_max","affected_value_element_count","affected_validity_element_count","changed_element_count","preserved_paths_verified","sample_count","unavailable_sample_count","instability_threshold","local_slope_window","candidate_item_count","fixed_complete_item_count","fixed_complete_horizon","item_count","valid_element_count"];
async function observationsForGuide(guide){const includeMissing=["matrix","heatmap"].includes(guide.visualization),observations=[];await Promise.all(guideSources(guide).map(async source=>{const descriptor=BOOT.artifacts.find(item=>item.name===source.name);if(!descriptor)return;const projected=await parseEmbeddedRows(descriptor,"observations",{metricPath:guide.path,evidenceKind:guide.evidence_kind||null,includeMissing,identityLeaves:OBSERVATION_IDENTITY_LEAVES});for(const item of projected)observations.push({...item,row:enrichComponentIdentity(item.row,source),source})}));return observations}
function scale(values,lowPixel,highPixel){const range=extent(values),span=range[1]-range[0];return{range,map:value=>lowPixel+(value-range[0])/span*(highPixel-lowPixel)}}
function orderedDomain(values){const domain=unique(values.map(String)),numeric=domain.length&&domain.every(value=>Number.isFinite(Number(value)));return numeric?domain.sort((a,b)=>Number(a)-Number(b)):domain}
function formatTick(value){const absolute=Math.abs(value);if(value!==0&&(absolute>=1e5||absolute<1e-3))return value.toExponential(2);return Number(value.toPrecision(4)).toString()}
function niceStep(span,target=4){const raw=Math.max(span/target,Number.MIN_VALUE),power=10**Math.floor(Math.log10(raw)),fraction=raw/power,nice=fraction<=1?1:fraction<=2?2:fraction<=2.5?2.5:fraction<=5?5:10;return nice*power}
function niceLinearTicks(domain){const step=niceStep(domain[1]-domain[0]),start=Math.ceil(domain[0]/step-1e-12)*step,end=Math.floor(domain[1]/step+1e-12)*step,ticks=[];for(let value=start;value<=end+step*1e-9;value+=step)ticks.push(Math.abs(value)<step*1e-10?0:Number(value.toPrecision(12)));return ticks.length?ticks:[domain[0],domain[1]]}
function powerTicks(domain,threshold){const ticks=[],append=(sign,minimum,maximum)=>{if(maximum<=0)return;const low=Math.max(minimum,threshold),first=Math.ceil(Math.log10(low)-1e-12),last=Math.floor(Math.log10(maximum)+1e-12),step=Math.max(1,Math.ceil((last-first+1)/8));for(let exponent=first;exponent<=last;exponent+=step)ticks.push(sign*10**exponent);if(first<=last&&(last-first)%step!==0)ticks.push(sign*10**last)};if(domain[0]<0)append(-1,domain[1]<0?Math.abs(domain[1]):threshold,Math.abs(domain[0]));if(domain[0]<=0&&domain[1]>=0)ticks.push(0);if(domain[1]>0)append(1,domain[0]>0?domain[0]:threshold,domain[1]);return unique(ticks.filter(value=>value>=domain[0]-threshold*1e-6&&value<=domain[1]+threshold*1e-6)).map(Number).sort((left,right)=>left-right)}
function formatPowerTick(value){if(value===0)return"0";const sign=value<0?"−":"",exponent=Math.round(Math.log10(Math.abs(value)));return`${sign}1e${exponent}`}
function shortLabel(value,limit=22){const text=String(value);return text.length<=limit?text:`${text.slice(0,limit-1)}…`}
function quantile(values,q){const sorted=[...values].sort((a,b)=>a-b),position=(sorted.length-1)*q,low=Math.floor(position),high=Math.ceil(position);return low===high?sorted[low]:sorted[low]*(high-position)+sorted[high]*(position-low)}
function paddedExtent(values,includeZero=false){let [low,high]=extent(includeZero?[...values,0]:values),padding=(high-low)*.06;if(!Number.isFinite(padding)||padding===0)padding=Math.max(Math.abs(low)*.06,1e-6);return[low-padding,high+padding]}
function boundedPaddedExtent(values){const [actualLow,actualHigh]=extent(values),domain=paddedExtent(values);if(actualLow>=0&&domain[0]<0)domain[0]=0;if(actualHigh<=0&&domain[1]>0)domain[1]=0;return domain}
function linear(domain,start,end){const span=domain[1]-domain[0];return value=>start+(value-domain[0])/span*(end-start)}
function scalarDimensionValues(rows,name){return rows.map(row=>dimensionValue(row,name)).filter(value=>value!==null&&value!==undefined&&!Array.isArray(value))}
function chooseFrom(rows,candidates,{minimum=2,maximum=Infinity,numeric=false,excluded=[]}={}){for(const candidate of candidates){if(excluded.includes(candidate))continue;const raw=scalarDimensionValues(rows,candidate),values=unique(raw);if(values.length<minimum||values.length>maximum)continue;if(numeric&&!raw.every(finite))continue;return candidate}return null}
function hashLike(value){return/^[0-9a-f]{32,}$/i.test(String(value??""))}
function readableIdentity(value,index,prefix){const text=String(value??"");if(!text)return"(model)";if(!hashLike(text))return text;const labels={sample:"样本",output:"输出",condition:"条件",series:"曲线",item:"对象"};return`${labels[prefix]||"对象"} ${index+1}`}
function aliasDomain(values,prefix){const domain=unique(values),aliases=new Map();domain.forEach((value,index)=>aliases.set(value,readableIdentity(value,index,prefix)));return aliases}
function compactPath(value){const text=String(value??"");if(!text)return"模型整体";const parts=text.split(".").filter(Boolean);if(parts.length<=3)return parts.join(" / ");return[parts[0],parts.at(-2),parts.at(-1)].join(" / ")}
function estimatedTextLength(text){return[...String(text)].reduce((total,character)=>total+(/[^\x00-\xff]/.test(character)?11:6.5),0)}
function fittedLabel(text,available){const estimated=Math.max(1,estimatedTextLength(text)),length=Math.min(estimated,Math.max(1,available));return estimated>available?` textLength="${length}" lengthAdjust="spacingAndGlyphs"`:""}
function labelMargin(labels){const estimated=Math.max(0,...labels.map(estimatedTextLength));return Math.max(160,Math.min(650,estimated+40))}
function configureSvg(svg,height,geometry){svg.setAttribute("viewBox",`0 0 ${geometry.width||CHART.width} ${height}`);svg.dataset.plotLeft=String(geometry.left);svg.dataset.plotRight=String(geometry.right);svg.dataset.plotTop=String(geometry.top);svg.dataset.plotBottom=String(geometry.bottom)}
function updateAxisTicks(domain){const maximum=Math.max(1,Math.ceil(domain[1])),rough=maximum/4,power=10**Math.floor(Math.log10(Math.max(rough,1))),step=Math.max(1,Math.ceil(rough/power)*power),ticks=[];for(let value=0;value<=maximum;value+=step)ticks.push(value);if(ticks.at(-1)!==maximum)ticks.push(maximum);return ticks}
function drawNumericXAxis(parts,domain,geometry,label){const sx=linear(domain,geometry.left,geometry.right),isUpdate=["update","checkpoint_update"].includes(label),ticks=isUpdate?updateAxisTicks(domain):Array.from({length:5},(_,index)=>domain[0]+(domain[1]-domain[0])*index/4);ticks.forEach(value=>{const x=sx(value);parts.push(`<line x1="${x}" y1="${geometry.top}" x2="${x}" y2="${geometry.bottom}" class="gridline"/><text x="${x}" y="${geometry.bottom+20}" text-anchor="middle" class="label">${isUpdate?String(Math.round(value)):formatTick(value)}</text>`)});parts.push(`<line x1="${geometry.left}" y1="${geometry.bottom}" x2="${geometry.right}" y2="${geometry.bottom}" class="axis"/><text x="${(geometry.left+geometry.right)/2}" y="${geometry.height-16}" text-anchor="middle" class="label">${esc(dimensionLabel(label))}</text>`);return sx}
// 对称对数固定保留 0 和负值；线性区阈值为当前纵轴最大绝对值的 10^-6。
function drawNumericYAxis(parts,domain,geometry,label,scaleMode="linear"){const maximum=Math.max(Math.abs(domain[0]),Math.abs(domain[1]),1e-30),threshold=maximum*1e-6,transform=value=>scaleMode==="symlog"?Math.sign(value)*Math.log10(1+Math.abs(value)/threshold):value,transformedDomain=domain.map(transform),mapped=linear(transformedDomain,geometry.bottom,geometry.top),sy=value=>mapped(transform(value)),ticks=scaleMode==="symlog"?powerTicks(domain,threshold):niceLinearTicks(domain);ticks.forEach(value=>{const y=sy(value);parts.push(`<line x1="${geometry.left}" y1="${y}" x2="${geometry.right}" y2="${y}" class="gridline"/><text x="${geometry.left-9}" y="${y+4}" text-anchor="end" class="label">${scaleMode==="symlog"?formatPowerTick(value):formatTick(value)}</text>`)});const suffix=scaleMode==="symlog"?"（对称对数）":"";parts.push(`<line x1="${geometry.left}" y1="${geometry.top}" x2="${geometry.left}" y2="${geometry.bottom}" class="axis"/><text x="22" y="${(geometry.top+geometry.bottom)/2}" transform="rotate(-90 22 ${(geometry.top+geometry.bottom)/2})" text-anchor="middle" class="label">${esc(dimensionLabel(label)+suffix)}</text>`);sy.scaleNote=scaleMode==="symlog"?`纵轴为对称对数；主刻度固定为 10 的整数次幂，0 与负值保留`:"纵轴为线性尺度；主刻度使用 1/2/2.5/5 × 10ⁿ 的整洁间隔";return sy}
function drawFullCategoryLabels(parts,labels,geometry,rowHeight){labels.forEach((label,index)=>{const y=geometry.top+(index+.5)*rowHeight,available=geometry.left-40;parts.push(`<text x="${geometry.left-16}" y="${y+4}" text-anchor="end" class="label"${fittedLabel(label,available)}>${esc(label)}</text>`)});parts.push(`<line x1="${geometry.left}" y1="${geometry.top}" x2="${geometry.left}" y2="${geometry.bottom}" class="axis"/>`)}
function seriesDash(index){return["","8 5","2 4","10 4 2 4"][index%4]}
function drawLegend(parts,groups,left=CHART.left){if(groups.length<2||groups.length>6)return;let x=left,y=28;groups.forEach((group,index)=>{const label=String(group),labelWidth=Math.min(260,estimatedTextLength(label)),width=42+labelWidth;if(x+width>CHART.width-CHART.right){x=left;y+=18}const color=COLORS[index%COLORS.length],dash=seriesDash(index);parts.push(`<line x1="${x}" y1="${y}" x2="${x+18}" y2="${y}" stroke="${color}" stroke-width="2"${dash?` stroke-dasharray="${dash}"`:""}/><circle cx="${x+9}" cy="${y}" r="3.5" fill="${color}"/><text x="${x+24}" y="${y+4}" class="legend-label"${fittedLabel(label,labelWidth)}>${esc(label)}</text>`);x+=width})}
function tooltipText(guide,lines,item=null){return[guide.label,...lines,item?identityText(item.row):null].filter(Boolean).join("\n")}
function markAttributes(tooltip,key=null){return`data-tooltip="${esc(tooltip)}"${key?` data-series-key="${esc(key)}"`:""}`}
// 功能图的坐标和图型属于 metric contract；观测数量只决定点和线的数量。
function timelineCoordinate(guide){if(guide.evidence_kind==="checkpoint_training_health")return"checkpoint_update";if(["training","parameter","runtime","module"].includes(guide.category))return"update";if(guide.category==="rollout")return guide.value_kind==="series"?"series index":"horizon";return null}
// update 本身已经标识 checkpoint/time condition；曲线身份只能保留跨 update 稳定的对象。
function temporalSeriesIdentity(row,category,componentAliases,coordinateName){if(category==="rollout"){const horizon=dimensionValue(row,"horizon"),response=dimensionValue(row,"response_component_id"),condition=dimensionValue(row,"rollout_condition_id")??dimensionValue(row,"scenario_id")??"condition";return[`h${horizon}`,componentAliases.get(String(response))??String(response??"output"),String(condition)].join(" · ")}const tap=dimensionValue(row,"tap_id"),module=dimensionValue(row,"module_path")??tap??dimensionValue(row,"node_id")??dimensionValue(row,"scope")??dimensionValue(row,"response_component_id")??"all",tapIdentity=tap!=null&&String(tap)!==String(module)?`tap=${tap}`:null,output=dimensionValue(row,"output_path"),invocation=dimensionValue(row,"invocation_index"),temporalCondition=!["update","checkpoint_update"].includes(coordinateName),condition=temporalCondition?dimensionValue(row,"__condition__"):null;return[String(module)||"(model)",tapIdentity,output==null?null:`output=${output}`,invocation==null?null:`call=${invocation}`,condition==null?null:`condition=${condition}`].filter(Boolean).join(" · ")}
function seriesFamily(label){const base=String(label).split(" · ")[0];if(base==="(model)"||base==="all"||base==="mean")return base;return base.split(/[.[]/,1)[0]||base}
function naturalReference(path){if(["grad_nonfinite_fraction","output_nonfinite_fraction","output_gradient_nonfinite_fraction","loss_cap_hit_fraction","gradient_cosine"].includes(path))return 0;if(["partition_coverage","norm_ratio"].includes(path))return 1;return null}
function deterministicAttention(path,value){if(["grad_nonfinite_fraction","output_nonfinite_fraction","output_gradient_nonfinite_fraction"].includes(path)&&value>0)return"非有限值比例大于 0";if(path==="loss_cap_hit_fraction"&&value>0)return"损失截断已生效";if(path==="partition_coverage"&&value<1)return"覆盖率小于 1";if(path==="gradient_cosine"&&value<0)return"梯度余弦小于 0";return null}
function robustAttention(points,index,allPositive){if(index<8)return null;const transform=value=>allPositive?Math.log(Math.max(value,1e-30)):value,history=points.slice(Math.max(0,index-12),index).map(point=>transform(point.mean)),center=quantile(history,.5),deviations=history.map(value=>Math.abs(value-center)),mad=quantile(deviations,.5),current=transform(points[index].mean);if(mad>0&&Math.abs(current-center)/(1.4826*mad)>=6)return"相对水平偏离不小于 6 MAD";const historyDiffs=history.slice(1).map((value,position)=>value-history[position]);if(historyDiffs.length<7)return null;const diffCenter=quantile(historyDiffs,.5),diffMad=quantile(historyDiffs.map(value=>Math.abs(value-diffCenter)),.5),currentDiff=current-transform(points[index-1].mean);return diffMad>0&&Math.abs(currentDiff-diffCenter)/(1.4826*diffMad)>=6?"相对跳变不小于 6 MAD":null}
function pointCloudPath(points){return points.map(point=>`M ${point.x} ${point.y} h .01`).join(" ")}
function drawTrendRecords(svg,guide,records,xName,yScaleMode="linear",lightweight=false){
  const groups=unique(records.map(record=>record.group)),allX=orderedDomain(records.map(record=>record.x)).map(Number),series=groups.map(group=>{const byX=new Map();records.filter(record=>record.group===group).forEach(record=>{if(!byX.has(record.x))byX.set(record.x,[]);byX.get(record.x).push(record)});return{group,aggregates:[...byX].sort((left,right)=>left[0]-right[0]).map(([x,points])=>({x,points,mean:points.reduce((total,record)=>total+record.value,0)/points.length}))}}),aggregateValues=series.flatMap(item=>item.aggregates.map(point=>point.mean)),reference=naturalReference(guide.path),rawY=reference==null?aggregateValues:[...aggregateValues,reference],yDomain=guide.path==="gradient_cosine"?[-1,1]:rawY.length?boundedPaddedExtent(rawY):[0,1],xValues=records.map(record=>record.x),isUpdate=["update","checkpoint_update"].includes(xName),xDomain=xValues.length?(isUpdate?[0,Math.max(1,Math.ceil(Math.max(...xValues)))]:["lag","context length","timestep"].includes(xName)&&Math.min(...xValues)>=0?[0,Math.max(...xValues)]:paddedExtent(xValues)):[0,1],geometry={left:96,right:940,top:68,bottom:338,height:440},parts=[],sx=drawNumericXAxis(parts,xDomain,geometry,xName),sy=drawNumericYAxis(parts,yDomain,geometry,guide.label,yScaleMode),families=unique(groups.map(seriesFamily)),hoverPoints=[],curveSignatures=[];
  let attentionCount=0;
  configureSvg(svg,geometry.height,geometry);
  drawLegend(parts,families,geometry.left);
  if(reference!=null&&reference>=yDomain[0]&&reference<=yDomain[1]){
    const y=sy(reference);
    parts.push(`<line x1="${geometry.left}" y1="${y}" x2="${geometry.right}" y2="${y}" class="reference-line"/><text x="${geometry.right-4}" y="${y-6}" text-anchor="end" class="evidence-note">reference ${formatTick(reference)}</text>`);
  }
  series.forEach(({group,aggregates},groupIndex)=>{
    const key=`series-${groupIndex}`,family=seriesFamily(group),familyIndex=Math.max(0,families.indexOf(family)),color=COLORS[familyIndex%COLORS.length],dash=seriesDash(groupIndex),summary=tooltipText(guide,[`series=${group}`,`whole line`,`${xName}: ${formatTick(aggregates[0].x)} → ${formatTick(aggregates.at(-1).x)}`,`mean range: ${formatTick(Math.min(...aggregates.map(point=>point.mean)))} → ${formatTick(Math.max(...aggregates.map(point=>point.mean)))}`,`total support=${aggregates.reduce((total,point)=>total+point.points.length,0)}`]),coordinates=aggregates.map(point=>`${sx(point.x)},${sy(point.mean)}`).join(" "),attentionPoints=[],attentionSegments=[],allPositive=aggregates.every(point=>point.mean>0);
    curveSignatures.push(aggregates.map(point=>`${point.x}:${point.mean}`).join("|"));
    if(aggregates.length>=2)parts.push(`<polyline class="series-line interactive-mark" data-series-key="${key}" points="${coordinates}" stroke="${color}"${dash?` stroke-dasharray="${dash}"`:""}/><polyline class="line-hit interactive-mark" ${markAttributes(summary,key)} points="${coordinates}"/>`);
    aggregates.forEach((point,index)=>{
      const reason=deterministicAttention(guide.path,point.mean)??robustAttention(aggregates,index,allPositive),x=sx(point.x),y=sy(point.mean),tooltip=tooltipText(guide,[`series=${group}`,`${xName}=${formatTick(point.x)}`,`mean=${formatTick(point.mean)}`,`support=${point.points.length}`,reason?`attention=${reason}`:null],point.points[0].item);
      if(!lightweight){
        hoverPoints.push({x,y,tooltip,seriesKey:key,attention:Boolean(reason)});
        parts.push(`<circle class="mean-point interactive-mark" ${markAttributes(tooltip,key)} cx="${x}" cy="${y}" r="3.2" fill="${color}"/>`);
      }else if(aggregates.length===1){
        parts.push(`<circle class="mean-point interactive-mark" ${markAttributes(summary,key)} cx="${x}" cy="${y}" r="3.2" fill="${color}"/>`);
      }
      if(reason){
        attentionCount+=1;
        if(!lightweight)attentionPoints.push({x,y});
        if(index>0)attentionSegments.push(`M ${sx(aggregates[index-1].x)} ${sy(aggregates[index-1].mean)} L ${x} ${y}`);
      }
    });
    if(attentionPoints.length)parts.push(`<path class="attention-cloud interactive-mark" data-series-key="${key}" d="${pointCloudPath(attentionPoints)}"/>`);
    if(attentionSegments.length)parts.push(`<path class="attention-segment interactive-mark" data-series-key="${key}" d="${attentionSegments.join(" ")}"/>`);
  });
  if(!lightweight)parts.push('<circle class="hover-point" visibility="hidden" r="5"/>');
  svg.innerHTML=parts.join("");
  if(!lightweight)registerHoverPoints(svg,hoverPoints);
  bindChartInteractions(svg);
  svg.__supportsYScale=true;
  svg.__zoomRender=(target,scaleMode=yScaleMode)=>drawTrendRecords(target,guide,records,xName,scaleMode,lightweight);
  const overlapCount=curveSignatures.length-new Set(curveSignatures).size,overlapNote=overlapCount?`；${overlapCount} 条均值曲线与其它曲线完全重合，已用不同线型叠加`:"";
  const interactionNote=lightweight?"；高密度训练趋势只保留整条曲线悬停，不创建逐点图元或逐点详情":"，点为同一横轴位置的观测均值";
  return{support:records.length,xCount:allX.length,groups:groups.length,wide:false,attentionCount,note:`每个稳定观测对象一条折线${interactionNote}；${sy.scaleNote}${overlapNote}；${attentionCount} 个红色关注片段`};
}
function drawTimeline(svg,guide,observations,yScaleMode="linear"){const rows=observations.map(item=>item.row),xName=timelineCoordinate(guide);if(!xName)throw new Error(`时间趋势图 ${guide.path} 没有声明固定横坐标`);const series=xName==="series index",coordinate=item=>Number(series?item.seriesIndex:dimensionValue(item.row,xName)),usable=observations.map(item=>({item,x:coordinate(item)})).filter(record=>Number.isFinite(record.x)),componentAliases=aliasDomain(rows.map(row=>dimensionValue(row,"response_component_id")).filter(value=>value!=null),"output"),hasOutputIdentity=rows.some(row=>dimensionValue(row,"response_component_id")!=null),hasSiteIdentity=rows.some(row=>dimensionValue(row,"module_path")!=null||dimensionValue(row,"tap_id")!=null||dimensionValue(row,"node_id")!=null),collapse=["training","runtime"].includes(guide.category)&&!hasOutputIdentity&&!hasSiteIdentity,rawRecords=usable.map(({item,x})=>({value:item.value,x,group:collapse?"总体均值":temporalSeriesIdentity(item.row,guide.category,componentAliases,xName),item})),groupAliases=aliasDomain(rawRecords.map(record=>record.group),"series"),records=rawRecords.map(record=>({...record,group:groupAliases.get(String(record.group))})),lightweight=guide.category==="training"&&xName==="update";return drawTrendRecords(svg,guide,records,xName,yScaleMode,lightweight)}
function heatmapDimensions(rows,category){if(["input","module"].includes(category)){const y=category==="input"?"__input_condition__":"__module_condition__",hasResponse=scalarDimensionValues(rows,"response_component_id").length>0,hasRows=scalarDimensionValues(rows,y).length>0;if(hasResponse&&hasRows)return{x:"response_component_id",y}}const xCandidates=category==="multi_objective"?["response_component_id","group_id","sample_id","update"]:category==="rollout"?["horizon","response_component_id","group_id","sample_id"]:["update","group_id","sample_id","__condition__","horizon"],yCandidates=category==="rollout"?["response_component_id","rollout_condition_id","scenario_id"]:["module_path","scope","node_id","response_component_id","__condition__","sample_id"],x=chooseFrom(rows,xCandidates,{minimum:2,maximum:80})??xCandidates.find(candidate=>scalarDimensionValues(rows,candidate).length),y=chooseFrom(rows,yCandidates,{minimum:2,maximum:120,excluded:[x]})??yCandidates.find(candidate=>candidate!==x&&scalarDimensionValues(rows,candidate).length);return{x,y}}
function influenceObject(row,category){if(category==="module"){const value=dimensionValue(row,"module_path")??dimensionValue(row,"module_site_id")??dimensionValue(row,"node_id");return compactPath(value)}const value=dimensionValue(row,"intervened_component_id");return value==null?"输入组件":displayValue(value)}
function hierarchyLevel(row){const value=String(dimensionValue(row,"hierarchy_level")??"").trim();if(/^stage$/i.test(value))return"Stage";if(/^block$/i.test(value))return"Block";return value||"未声明层级"}
function influenceCondition(row){return dimensionValue(row,"__condition__")??"未命名干预"}
function identityControl(row){const method=dimensionValue(row,"method"),scale=Number(dimensionValue(row,"scale")),status=dimensionValue(row,"control_status");return method==="identity"||status==="identity_control"||method==="output_scale"&&scale===1}
function influenceRecords(guide,observations){const responseRaw=observations.map(item=>dimensionValue(item.row,"response_component_id")??"输出"),responseAliases=aliasDomain(responseRaw,"output"),minimumField=`${guide.path}_min`,maximumField=`${guide.path}_max`;return observations.map((item,index)=>{const minimum=dimensionValue(item.row,minimumField),maximum=dimensionValue(item.row,maximumField),object=influenceObject(item.row,guide.category),response=responseAliases.get(String(responseRaw[index])),rawModule=dimensionValue(item.row,"module_path")??dimensionValue(item.row,"module_site_id")??dimensionValue(item.row,"node_id")??object;return{value:item.value,minimum:finite(minimum)?minimum:item.value,maximum:finite(maximum)?maximum:item.value,item,level:guide.category==="module"?hierarchyLevel(item.row):null,object,condition:influenceCondition(item.row),response,control:identityControl(item.row),controlKey:guide.category==="module"?`${rawModule}\u0000${response}`:response,affectedValues:dimensionValue(item.row,"affected_value_element_count"),affectedValidity:dimensionValue(item.row,"affected_validity_element_count"),changedValues:dimensionValue(item.row,"changed_element_count"),preservedPaths:dimensionValue(item.row,"preserved_paths_verified"),sampleCount:dimensionValue(item.row,"sample_count"),unavailableSamples:dimensionValue(item.row,"unavailable_sample_count")}})}

// 高密度 input/module × output 证据固定使用分页列表；保留全部组合，但只创建当前页 DOM。
function drawInfluenceList(svg,guide,records){
  const aggregateRecords=source=>{
    const aggregates=new Map();
    source.forEach(record=>{
      const status=displayValue(dimensionValue(record.item.row,"status")??(finite(record.value)?"success":"missing"));
      const reason=dimensionValue(record.item.row,"skip_reason")??"";
      const key=[record.level,record.condition,record.object,record.response,status,reason].join("\u0000");
      if(!aggregates.has(key))aggregates.set(key,{level:record.level,condition:record.condition,object:record.object,response:record.response,controlKey:record.controlKey,status,reason,values:[],minimums:[],maximums:[],affectedValues:[],changedValues:[],preservedPaths:[],sampleCounts:[],unavailableSamples:[],count:0});
      const aggregate=aggregates.get(key);
      aggregate.count+=1;
      if(finite(record.value))aggregate.values.push(record.value);
      if(finite(record.minimum))aggregate.minimums.push(record.minimum);
      if(finite(record.maximum))aggregate.maximums.push(record.maximum);
      if(finite(record.affectedValues))aggregate.affectedValues.push(Number(record.affectedValues));
      if(finite(record.changedValues))aggregate.changedValues.push(Number(record.changedValues));
      if(typeof record.preservedPaths==="boolean")aggregate.preservedPaths.push(record.preservedPaths);
      if(finite(record.sampleCount))aggregate.sampleCounts.push(Number(record.sampleCount));
      if(finite(record.unavailableSamples))aggregate.unavailableSamples.push(Number(record.unavailableSamples));
    });
    return[...aggregates.values()].map(row=>({...row,mean:row.values.length?row.values.reduce((total,value)=>total+value,0)/row.values.length:null,minimum:row.minimums.length?Math.min(...row.minimums):null,maximum:row.maximums.length?Math.max(...row.maximums):null,finiteCount:row.values.length,affectedValueCount:row.affectedValues.length?Math.max(...row.affectedValues):null,changedValueCount:row.changedValues.length?Math.max(...row.changedValues):null,preservedPathsStatus:row.preservedPaths.length?(row.preservedPaths.every(Boolean)?"通过":"未通过"):null,totalSamples:row.sampleCounts.length?Math.max(...row.sampleCounts):null,unavailableSampleCount:row.unavailableSamples.length?Math.max(...row.unavailableSamples):null}));
  };
  const controlRows=aggregateRecords(records.filter(record=>record.control)),controlByKey=new Map();controlRows.forEach(row=>{if(!controlByKey.has(row.controlKey))controlByKey.set(row.controlKey,[]);controlByKey.get(row.controlKey).push(row)});
  const rows=aggregateRecords(records.filter(record=>!record.control)).map(row=>{const controls=(controlByKey.get(row.controlKey)||[]).filter(control=>finite(control.mean)),controlValues=controls.map(control=>control.mean),controlMinimums=controls.map(control=>control.minimum).filter(finite),controlMaximums=controls.map(control=>control.maximum).filter(finite);return{...row,controlMean:controlValues.length?controlValues.reduce((total,value)=>total+value,0)/controlValues.length:null,controlMinimum:controlMinimums.length?Math.min(...controlMinimums):null,controlMaximum:controlMaximums.length?Math.max(...controlMaximums):null,controlMatched:controls.length>0}});
  const moduleView=guide.category==="module";
  const responses=unique(rows.map(row=>row.response)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));
  const conditions=unique(rows.map(row=>row.condition)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));
  const levels=moduleView?unique(rows.map(row=>row.level)).sort((left,right)=>left.localeCompare(right,"en",{numeric:true})):[];
  const objects=unique(rows.map(row=>row.object)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));
  const levelCoverage=levels.map(level=>`${level}: ${unique(rows.filter(row=>row.level===level).map(row=>row.object)).length}`).join(" · ");
  const id=`influence-list-${Math.random().toString(36).slice(2)}`,objectLabel=moduleView?"Stage/Block":"输入通道",relative=guide.path==="normalized_effect",meanLabel=relative?"平均相对响应":"平均响应",finiteRows=rows.filter(row=>finite(row.mean)),controlValues=controlRows.filter(row=>finite(row.mean)).map(row=>Math.abs(row.mean)),controlMaximum=controlValues.length?Math.max(...controlValues):null;
  const issues=rows.flatMap(row=>{const found=[];if(row.status!=="成功"&&row.status!=="success")found.push(row.reason||row.status);if(!row.controlMatched)found.push("缺少恒等对照");if((row.unavailableSampleCount??0)>0)found.push(`${row.unavailableSampleCount} 个样本不可用`);if(row.preservedPathsStatus==="未通过")found.push("未干预路径核对失败");if(guide.category==="input"&&row.condition!=="恒等对照"&&row.changedValueCount===0)found.push("实际改动元素为 0");return found.length?[{...row,issue:found.join("；")}]:[]});
  const levelControl=moduleView?`<select data-filter="level"><option value="">全部层级</option>${levels.map(value=>`<option value="${esc(value)}">${esc(value)}</option>`).join("")}</select><select data-filter="object"><option value="">全部 Stage/Block</option>${objects.map(value=>`<option value="${esc(value)}">${esc(value)}</option>`).join("")}</select>`:"";
  const levelHeader=moduleView?"<th>层级</th>":"";
  const validityLevelHeader=moduleView?"<th>层级</th>":"";
  const validityRows=issues.map(row=>`<tr>${moduleView?`<td>${esc(row.level)}</td>`:""}<td>${esc(row.condition)}</td><td>${esc(row.object)}</td><td>${esc(row.response)}</td><td>${esc(row.issue)}</td></tr>`).join("");
  svg.outerHTML=`<div id="${id}" class="influence-list"><div class="influence-summary"><span>实际干预组合：${rows.length.toLocaleString()}</span><span>有限组合：${finiteRows.length.toLocaleString()}</span>${moduleView?`<span>Stage/Block coverage：${objects.length.toLocaleString()}（${esc(levelCoverage)}）</span>`:""}<span>输出通道：${responses.length.toLocaleString()}</span><span>已配对恒等对照：${rows.filter(row=>row.controlMatched).length.toLocaleString()}</span><span>恒等对照最大绝对响应：${controlMaximum==null?"无有限值":esc(formatTick(controlMaximum))}</span></div><div class="toolbar">${levelControl}<select data-filter="condition"><option value="">全部干预方法</option>${conditions.map(value=>`<option value="${esc(value)}">${esc(value)}</option>`).join("")}</select><select data-filter="response"><option value="">全部输出通道</option>${responses.map(value=>`<option value="${esc(value)}">${esc(value)}</option>`).join("")}</select><input data-filter="search" placeholder="搜索${esc(objectLabel)}或输出通道"><select data-filter="sort"><option value="magnitude">按绝对影响降序</option><option value="positive">按影响值降序</option><option value="object">按${esc(objectLabel)}名称</option></select><button type="button" data-page="previous">上一页</button><button type="button" data-page="next">下一页</button></div><div data-role="summary" class="muted"></div><div class="table"><table><thead><tr>${levelHeader}<th>干预方法</th><th>${esc(objectLabel)}</th><th>输出通道</th><th>${meanLabel}</th><th>响应范围</th><th>恒等对照平均响应</th><th>恒等对照响应范围</th></tr></thead><tbody></tbody></table></div><details class="influence-validity"><summary>查看缺失与有效性核对</summary>${issues.length?`<div class="table"><table><thead><tr>${validityLevelHeader}<th>干预方法</th><th>${esc(objectLabel)}</th><th>输出通道</th><th>问题</th></tr></thead><tbody>${validityRows}</tbody></table></div>`:'<p class="muted">当前实际干预组合没有缺失或有效性核对失败。</p>'}</details><details class="influence-matrix"><summary>查看当前筛选的色块影响矩阵</summary><p class="muted">矩阵使用与主表相同的层级、模块、干预方法和输出通道筛选；颜色以 0 为中心，单元格悬停信息同时显示对应恒等对照。仅在展开时创建图形。</p><svg role="img" aria-label="${esc(objectLabel)}对输出通道的色块影响矩阵"></svg></details></div>`;
  const host=document.getElementById(id),tbody=host.querySelector("tbody"),summary=host.querySelector('[data-role="summary"]'),levelFilter=host.querySelector('[data-filter="level"]'),objectFilter=host.querySelector('[data-filter="object"]'),conditionFilter=host.querySelector('[data-filter="condition"]'),responseFilter=host.querySelector('[data-filter="response"]'),search=host.querySelector('[data-filter="search"]'),sort=host.querySelector('[data-filter="sort"]'),matrixDetails=host.querySelector(".influence-matrix"),matrixSvg=matrixDetails.querySelector("svg"),pageSize=100;let page=0,searchTimer=0,matrixDrawn=false;
  const matches=(row,includeControl=false)=>{const query=search.value.trim().toLowerCase();return(!levelFilter?.value||row.level===levelFilter.value)&&(!objectFilter?.value||row.object===objectFilter.value)&&(!conditionFilter.value||includeControl||row.condition===conditionFilter.value)&&(!responseFilter.value||row.response===responseFilter.value)&&(!query||`${row.level??""} ${row.object} ${row.response}`.toLowerCase().includes(query))};
  const invalidateMatrix=()=>{matrixDrawn=false;matrixSvg.innerHTML="";if(matrixDetails.open)drawMatrix()};
  const drawMatrix=()=>{const selected=records.filter(record=>matches(record,record.control)),selectedResponses=unique(selected.filter(record=>!record.control).map(record=>record.response)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));if(!selectedResponses.length){configureSvg(matrixSvg,260,{width:1000,height:260,left:0,right:1000,top:0,bottom:260});matrixSvg.innerHTML='<text x="500" y="130" text-anchor="middle" class="label">当前筛选没有可绘制的实际干预矩阵。</text>';matrixDrawn=true;return}drawInfluenceMatrixFacets(matrixSvg,guide,selected,selectedResponses,selected.filter(record=>record.control));bindZoom(matrixDetails,`${objectLabel}对输出通道的色块影响矩阵`);matrixDrawn=true};
  const range=(minimum,maximum)=>minimum==null||maximum==null?"—":`${formatTick(minimum)} ～ ${formatTick(maximum)}`;
  const render=()=>{const filtered=rows.filter(row=>matches(row)),magnitude=row=>finite(row.mean)?Math.abs(row.mean):-Infinity,numeric=row=>finite(row.mean)?row.mean:-Infinity;filtered.sort((left,right)=>sort.value==="object"?left.object.localeCompare(right.object,"zh-CN",{numeric:true}):sort.value==="positive"?numeric(right)-numeric(left):magnitude(right)-magnitude(left));const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);const visible=filtered.slice(page*pageSize,(page+1)*pageSize),columnCount=moduleView?8:7;tbody.innerHTML=visible.map(row=>`<tr>${moduleView?`<td>${esc(row.level)}</td>`:""}<td>${esc(row.condition)}</td><td title="${esc(row.object)}">${esc(row.object)}</td><td>${esc(row.response)}</td><td class="${row.mean==null?"":row.mean>=0?"effect-positive":"effect-negative"}">${row.mean==null?"—":esc(formatTick(row.mean))}</td><td>${esc(range(row.minimum,row.maximum))}</td><td>${row.controlMean==null?"—":esc(formatTick(row.controlMean))}</td><td>${esc(range(row.controlMinimum,row.controlMaximum))}</td></tr>`).join("")||`<tr><td colspan="${columnCount}">当前筛选没有实际干预记录。</td></tr>`;const visibleObjects=unique(filtered.map(row=>row.object)).length;summary.textContent=`${filtered.length.toLocaleString()} 个实际干预组合 · ${visibleObjects.toLocaleString()} 个${objectLabel} · 第 ${page+1}/${pages} 页 · 正值表示干预后误差增加，负值表示干预后误差降低`};
  [levelFilter,objectFilter,conditionFilter,responseFilter,sort].filter(Boolean).forEach(control=>control.onchange=()=>{page=0;render();invalidateMatrix()});search.oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render();invalidateMatrix()},120)};host.querySelector('[data-page="previous"]').onclick=()=>{page=Math.max(0,page-1);render()};host.querySelector('[data-page="next"]').onclick=()=>{page+=1;render()};matrixDetails.ontoggle=()=>{if(matrixDetails.open&&!matrixDrawn)drawMatrix()};render();
  return{support:finiteRows.length,xCount:responses.length,groups:unique(rows.map(row=>row.object)).length,wide:false,visualKind:"分页实际干预列表与折叠色块矩阵",note:"恒等对照按模块/输出或输出通道配对到每一行；审计字段只在异常核对与原始证据中展示"};
}

// 输出很多时才使用矩阵；干预方法不再拼入行标签，且每个面板用稳健色阶。
function drawInfluenceMatrixFacets(svg,guide,records,responses,controls){const interventions=records.filter(record=>!record.control),conditions=unique(interventions.map(record=>record.condition)),allObjects=unique(interventions.map(record=>record.object)),left=Math.min(430,Math.max(180,labelMargin(allObjects))),cellWidth=Math.max(54,Math.min(104,(1180-left)/responses.length)),right=left+responses.length*cellWidth,rowHeight=36,parts=[],facets=[];let cursor=70;conditions.forEach(condition=>{const conditionRecords=interventions.filter(record=>record.condition===condition),objects=unique(conditionRecords.map(record=>record.object)).map(object=>{const values=conditionRecords.filter(record=>record.object===object&&finite(record.value)).map(record=>Math.abs(record.value));return{object,score:values.length?Math.max(...values):-Infinity}}).sort((a,b)=>b.score-a.score).map(item=>item.object),top=cursor+28,bottom=top+objects.length*rowHeight;facets.push({condition,conditionRecords,objects,top,bottom});cursor=bottom+140});const width=Math.max(1000,right+46),height=Math.max(360,cursor),geometry={width,left,right,top:70,bottom:height-30,height};configureSvg(svg,height,geometry);const controlValues=controls.filter(record=>finite(record.value)).map(record=>record.value),controlMaximum=controlValues.length?Math.max(...controlValues.map(Math.abs)):null;parts.push(`<text x="${right}" y="28" text-anchor="end" class="control-note">恒等对照：${controlValues.length} 条${controlMaximum==null?"，无有限数值":`，最大 |响应|=${formatTick(controlMaximum)}`}</text>`);facets.forEach(facet=>{parts.push(`<text x="${left}" y="${facet.top-12}" class="facet-title">${esc(facet.condition)}</text>`);facet.objects.forEach((object,rowIndex)=>{const y=facet.top+(rowIndex+.5)*rowHeight;parts.push(`<text x="${left-14}" y="${y+4}" text-anchor="end" class="label"${fittedLabel(object,left-38)}>${esc(object)}</text>`)});responses.forEach((response,columnIndex)=>{const x=left+(columnIndex+.5)*cellWidth,y=facet.bottom+12;parts.push(`<text x="${x}" y="${y}" transform="rotate(-42 ${x} ${y})" text-anchor="end" class="label"${fittedLabel(response,118)}>${esc(response)}</text>`)});const cellMeans=[];facet.objects.forEach(object=>responses.forEach(response=>{const cell=facet.conditionRecords.filter(record=>record.object===object&&record.response===response&&finite(record.value));if(cell.length)cellMeans.push(cell.reduce((total,record)=>total+record.value,0)/cell.length)}));const magnitude=Math.max(quantile(cellMeans.map(Math.abs),.95)||0,1e-12);facet.objects.forEach((object,rowIndex)=>responses.forEach((response,columnIndex)=>{const cell=facet.conditionRecords.filter(record=>record.object===object&&record.response===response),finiteCell=cell.filter(record=>finite(record.value)),x=left+columnIndex*cellWidth+1,y=facet.top+rowIndex*rowHeight+1,w=cellWidth-2,h=rowHeight-2;if(!finiteCell.length){parts.push(`<rect class="matrix-cell-missing" x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="transparent"/>`);return}const mean=finiteCell.reduce((total,record)=>total+record.value,0)/finiteCell.length,matchingControls=controls.filter(record=>record.controlKey===finiteCell[0].controlKey&&finite(record.value)),controlMean=matchingControls.length?matchingControls.reduce((total,record)=>total+record.value,0)/matchingControls.length:null,controlMinimumValues=matchingControls.map(record=>record.minimum).filter(finite),controlMaximumValues=matchingControls.map(record=>record.maximum).filter(finite),controlMinimum=controlMinimumValues.length?Math.min(...controlMinimumValues):null,controlMaximumValue=controlMaximumValues.length?Math.max(...controlMaximumValues):null,clipped=Math.abs(mean)>magnitude,tooltip=[guide.label,`干预方法：${facet.condition}`,`${guide.category==="module"?"模块":"输入"}：${object}`,`输出：${response}`,`均值：${formatTick(mean)}`,controlMean==null?"恒等对照：缺失":`恒等对照均值：${formatTick(controlMean)}`,controlMinimum==null||controlMaximumValue==null?null:`恒等对照范围：${formatTick(controlMinimum)} ～ ${formatTick(controlMaximumValue)}`,`有效观测：${finiteCell.length}`,clipped?"颜色已按 95% 分位截断":null].filter(Boolean).join("\n"),textColor=Math.abs(mean)>=magnitude*.56?"#fff":"var(--ink)";parts.push(`<rect class="matrix-cell interactive-mark" ${markAttributes(tooltip)} x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${matrixColor(mean,magnitude,true)}"${clipped?' stroke="var(--accent)" stroke-width="2"':""}/><text x="${x+w/2}" y="${y+h/2+4}" text-anchor="middle" font-size="10" font-weight="650" fill="${textColor}" pointer-events="none">${esc(formatTick(mean))}</text>`)}));parts.push(`<text x="${right}" y="${facet.top-12}" text-anchor="end" class="control-note">零中心色阶：±${formatTick(magnitude)}（|单元格均值| 的 95% 分位）</text>`)});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:interventions.filter(record=>finite(record.value)).length,xCount:responses.length,groups:allObjects.length,wide:true,visualKind:"按干预方法分面的影响矩阵",note:`${conditions.length} 种实际干预分别展示；恒等对照与每个单元格配对；每格直接标出实际干预均值`}}
function drawInfluenceFacets(svg,guide,observations){const records=influenceRecords(guide,observations);return drawInfluenceList(svg,guide,records)}
function drawMatrix(svg,guide,observations){if(["input","module"].includes(guide.category))return drawInfluenceFacets(svg,guide,observations);const rows=observations.map(item=>item.row),{x,y}=heatmapDimensions(rows,guide.category);if(!x||!y)return drawMissingMatrix(svg,guide,observations);const xRaw=observations.map(item=>String(dimensionValue(item.row,x)??"all")),yRaw=observations.map(item=>String(dimensionValue(item.row,y)??"all")),xAliases=aliasDomain(xRaw,x),yAliases=aliasDomain(yRaw,y),xDomain=orderedDomain(xRaw).map(value=>xAliases.get(value)),yDomain=unique(yRaw).map(value=>yAliases.get(value)),records=observations.map(item=>({xLabel:xAliases.get(String(dimensionValue(item.row,x)??"all")),yLabel:yAliases.get(String(dimensionValue(item.row,y)??"all")),value:item.value,missing:!finite(item.value),item}));return drawDotMatrixRecords(svg,guide,records,xDomain,yDomain,x,y)}
function aggregateCells(records){const cells=new Map();records.forEach(record=>{const key=`${record.xLabel}\u0000${record.yLabel}`;if(!cells.has(key))cells.set(key,[]);cells.get(key).push(record)});return cells}
function matrixColor(value,magnitude,signed){const ratio=Math.max(0,Math.min(1,signed?Math.abs(value)/magnitude:value/magnitude)),base=signed&&value<0?"var(--viz-negative)":signed?"var(--scale-5)":"var(--viz-1)",weight=Math.round(12+82*ratio);return`color-mix(in srgb,${base} ${weight}%,var(--panel))`}
function drawMissingMatrix(svg,guide,observations){svg.outerHTML=`<div class="empty">${esc(guide.label)} 的组合清单存在，但没有有限数值；缺失证据不会按零展示。</div>`;return{support:0,xCount:0,groups:0,missingCount:observations.length,visualKind:"缺失矩阵",note:`${observations.length} 个缺失组合`}}
function drawDotMatrixRecords(svg,guide,records,xDomain,yDomain,xLabel,yLabel){const xAxisLabel=dimensionLabel(xLabel),yAxisLabel=dimensionLabel(yLabel),rowHeight=34,left=labelMargin(yDomain),minimumCellWidth=26,width=Math.max(CHART.width,left+80+xDomain.length*minimumCellWidth),bottomPadding=Math.min(190,Math.max(82,...xDomain.map(estimatedTextLength))*.72+54),height=Math.max(360,64+yDomain.length*rowHeight+bottomPadding),geometry={width,left,right:width-52,top:60,bottom:60+yDomain.length*rowHeight,height},parts=[],cellWidth=(geometry.right-geometry.left)/Math.max(xDomain.length,1),finiteRecords=records.filter(record=>finite(record.value)),values=finiteRecords.map(record=>record.value),[low,high]=values.length?extent(values):[0,0],signed=DIRECTION_PATHS.has(guide.path)||low<0,magnitude=Math.max(signed?Math.max(Math.abs(low),Math.abs(high)):Math.max(high,Math.abs(low)),1e-12),cells=aggregateCells(records),patternId=`matrix-missing-${Math.random().toString(36).slice(2)}`;configureSvg(svg,height,geometry);parts.push(`<defs><pattern id="${patternId}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="var(--line)" stroke-width="2"/></pattern></defs>`);drawFullCategoryLabels(parts,yDomain,geometry,rowHeight);xDomain.forEach((label,index)=>{const x=geometry.left+(index+.5)*cellWidth,y=geometry.bottom+10,available=Math.max(70,bottomPadding-22);parts.push(`<text x="${x}" y="${y}" transform="rotate(-55 ${x} ${y})" text-anchor="end" class="label"${fittedLabel(label,available)}>${esc(label)}</text>`)});const legend=signed?`以 0 为中心：${formatTick(-magnitude)} → 0 → ${formatTick(magnitude)}`:`取值范围：0 → ${formatTick(magnitude)}`;parts.push(`<text x="${geometry.left}" y="30" class="label">纵轴：${esc(yAxisLabel)}</text><text x="${(geometry.left+geometry.right)/2}" y="${height-15}" text-anchor="middle" class="label">${esc(xAxisLabel)}</text><text x="${geometry.right}" y="30" text-anchor="end" class="label">${esc(legend)}；斜纹表示缺失</text>`);let missingCount=0;for(const [yIndex,yValue] of yDomain.entries()){for(const [xIndex,xValue] of xDomain.entries()){const cell=cells.get(`${xValue}\u0000${yValue}`)||[],finiteCell=cell.filter(record=>finite(record.value)),x=geometry.left+xIndex*cellWidth+1,y=geometry.top+yIndex*rowHeight+1,w=Math.max(2,cellWidth-2),h=rowHeight-2;if(!finiteCell.length){missingCount+=1;const status=cell.map(record=>dimensionValue(record.item.row,"status")).filter(Boolean).map(displayValue).join("、")||"缺失",reason=cell.map(record=>dimensionValue(record.item.row,"skip_reason")).filter(Boolean).join("、"),tooltip=tooltipText(guide,[`${yAxisLabel}：${yValue}`,`${xAxisLabel}：${xValue}`,"数值：缺失",`状态：${status}`,reason?`原因：${reason}`:null],cell[0]?.item);parts.push(`<rect class="matrix-cell-missing interactive-mark" ${markAttributes(tooltip)} x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="url(#${patternId})"/><text class="matrix-missing-label" x="${x+w/2}" y="${y+h/2+4}" text-anchor="middle">—</text>`);continue}const mean=finiteCell.reduce((total,record)=>total+record.value,0)/finiteCell.length,tooltip=tooltipText(guide,[`${yAxisLabel}：${yValue}`,`${xAxisLabel}：${xValue}`,`均值：${formatTick(mean)}`,`有效观测：${finiteCell.length}`],finiteCell[0].item);parts.push(`<rect class="matrix-cell interactive-mark" ${markAttributes(tooltip)} x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${matrixColor(mean,magnitude,signed)}"/>`)}}svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:finiteRecords.length,xCount:xDomain.length,groups:yDomain.length,missingCount,wide:true,visualKind:"影响矩阵",note:`完整 ${yDomain.length} × ${xDomain.length} 组合；${missingCount} 个缺失单元以斜纹显示`}}
function categoryDimension(rows,category){const candidates={module:["response_component_id","module_site_id","module_path","__condition__","node_id","sample_id"],parameter:["module_path","node_id","sample_id"],input:["intervened_component_id","response_component_id","__condition__","sample_id"],multi_objective:["response_component_id","module_path","scope","node_id","group_id"],rollout:["response_component_id","rollout_condition_id","horizon","scenario_id"],runtime:["module_path","node_id","rank"]}[category]||DIMENSIONS[category]||[];return chooseFrom(rows,candidates,{minimum:2,maximum:120})}
function conditionFamily(value){const text=String(value??"all"),parts=text.split(":");if(parts[0]==="feedback")return`feedback · ${parts.at(-1)}`;if(parts[0]==="input")return`input · ${parts.at(-1)}`;return readableIdentity(text,0,"condition")}
function directionalIdentity(row,category,componentAliases){const module=dimensionValue(row,"module_site_id")??dimensionValue(row,"module_path")??dimensionValue(row,"node_id"),scope=dimensionValue(row,"scope"),response=dimensionValue(row,"response_component_id"),responseLabel=componentAliases.get(String(response))??readableIdentity(response,0,"output"),intervened=dimensionValue(row,"intervened_component_id"),condition=dimensionValue(row,"__condition__")??dimensionValue(row,"rollout_condition_id");if(category==="module")return[module??scope??"(model)",response==null?null:responseLabel].filter(Boolean).join(" · ");if(category==="parameter")return seriesFamily(module??scope??"(model)");if(category==="input")return[intervened==null?null:intervened,response==null?null:responseLabel,condition==null?null:conditionFamily(condition)].filter(Boolean).join(" → ")||"all input";if(category==="multi_objective")return[response==null?null:responseLabel,seriesFamily(module??scope??"(model)")].filter(Boolean).join(" · ");if(category==="rollout")return[response==null?null:responseLabel,condition==null?null:conditionFamily(condition),dimensionValue(row,"horizon")==null?null:`h${dimensionValue(row,"horizon")}`].filter(Boolean).join(" · ")||"rollout";return null}
function categoryRecords(guide,observations){const rows=observations.map(item=>item.row),componentAliases=aliasDomain(rows.map(row=>dimensionValue(row,"response_component_id")).filter(value=>value!=null),"output"),directional=observations.map(item=>directionalIdentity(item.row,guide.category,componentAliases));if(directional.some(Boolean))return{name:"direction",records:observations.map((item,index)=>({label:directional[index]||"all",value:item.value,item}))};const name=categoryDimension(rows,guide.category),raw=observations.map((item,index)=>String(name?dimensionValue(item.row,name)??"all":index)),aliases=aliasDomain(raw,name||"item");return{name:name||"observation",records:observations.map((item,index)=>({label:aliases.get(raw[index]),value:item.value,item}))}}
function horizontalGeometry(labels){const rowHeight=30,left=labelMargin(labels),height=Math.max(260,58+labels.length*rowHeight+72);return{rowHeight,left,right:940,top:58,bottom:58+labels.length*rowHeight,height}}
function drawDotPlot(svg,guide,observations){const{name,records}=categoryRecords(guide,observations),reference=naturalReference(guide.path)??0,allLabels=unique(records.map(record=>record.label)),aggregates=allLabels.map(label=>{const points=records.filter(record=>record.label===label),mean=points.reduce((total,record)=>total+record.value,0)/points.length;return{label,points,mean,score:Math.abs(mean-reference)}}).sort((left,right)=>right.score-left.score),visibleAggregates=aggregates,labels=visibleAggregates.map(record=>record.label),visible=visibleAggregates.flatMap(record=>record.points),geometry=horizontalGeometry(labels),domain=paddedExtent([...visible.map(record=>record.value),reference],true),parts=[],sx=drawNumericXAxis(parts,domain,geometry,guide.label),referenceX=sx(reference);configureSvg(svg,geometry.height,geometry);drawFullCategoryLabels(parts,labels,geometry,geometry.rowHeight);parts.push(`<line x1="${referenceX}" y1="${geometry.top}" x2="${referenceX}" y2="${geometry.bottom}" class="reference-line"/><text x="${referenceX+5}" y="${geometry.top-10}" class="evidence-note">reference ${formatTick(reference)}</text>`);visibleAggregates.forEach((aggregate,labelIndex)=>{const center=geometry.top+(labelIndex+.5)*geometry.rowHeight,reason=deterministicAttention(guide.path,aggregate.mean),color=reason?"var(--bad)":COLORS[labelIndex%COLORS.length],key=`direction-${labelIndex}`;parts.push(`<line class="direction-stem interactive-mark" data-series-key="${key}" x1="${referenceX}" y1="${center}" x2="${sx(aggregate.mean)}" y2="${center}" stroke="${color}"/>`);aggregate.points.forEach((record,index)=>{const offset=(index-(aggregate.points.length-1)/2)*Math.min(2.5,geometry.rowHeight*.4/Math.max(aggregate.points.length,1)),tooltip=tooltipText(guide,[`${name}=${aggregate.label}`,`observation=${formatTick(record.value)}`],record.item);parts.push(`<circle class="raw-point interactive-mark" ${markAttributes(tooltip,key)} cx="${sx(record.value)}" cy="${center+offset}" r="1.25" fill="${color}"/>`)});const tooltip=tooltipText(guide,[`${name}=${aggregate.label}`,`mean=${formatTick(aggregate.mean)}`,`reference=${formatTick(reference)}`,`support=${aggregate.points.length}`,reason?`attention=${reason}`:null],aggregate.points[0].item);parts.push(`<circle class="${reason?"attention-point":"mean-point"} interactive-mark" ${markAttributes(tooltip,key)} cx="${sx(aggregate.mean)}" cy="${center}" r="${reason?3.2:2.6}" fill="${color}"/>`)});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:records.length,xCount:labels.length,groups:labels.length,wide:geometry.left>360,note:"按偏离 reference 排序，展示全部比较对象"}}
function drawBars(svg,guide,observations){const{name,records}=categoryRecords(guide,observations),labels=unique(records.map(record=>record.label)),aggregates=labels.map(label=>{const points=records.filter(record=>record.label===label);return{label,points,mean:points.reduce((total,record)=>total+record.value,0)/points.length}}),geometry=horizontalGeometry(labels),domain=paddedExtent(aggregates.map(record=>record.mean),true),parts=[],sx=drawNumericXAxis(parts,domain,geometry,guide.label),zero=sx(0);configureSvg(svg,geometry.height,geometry);drawFullCategoryLabels(parts,labels,geometry,geometry.rowHeight);aggregates.forEach((record,index)=>{const y=geometry.top+index*geometry.rowHeight+6,x=sx(record.mean),tooltip=tooltipText(guide,[`${name}=${record.label}`,`mean=${formatTick(record.mean)}`,`support=${record.points.length}`],record.points[0].item);parts.push(`<rect class="bar-mark interactive-mark" ${markAttributes(tooltip)} x="${Math.min(x,zero)}" y="${y}" width="${Math.max(1,Math.abs(x-zero))}" height="${geometry.rowHeight-12}" fill="${record.mean<0?"var(--viz-negative)":COLORS[index%COLORS.length]}"/>`)});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:records.length,xCount:labels.length,groups:labels.length,wide:geometry.left>360,note:"水平条形长度为同一类别均值"}}
function drawDistribution(svg,guide,observations){const{name,records}=categoryRecords(guide,observations),labels=unique(records.map(record=>record.label)),geometry=horizontalGeometry(labels),domain=paddedExtent(records.map(record=>record.value)),parts=[],sx=drawNumericXAxis(parts,domain,geometry,guide.label);configureSvg(svg,geometry.height,geometry);drawFullCategoryLabels(parts,labels,geometry,geometry.rowHeight);labels.forEach((label,index)=>{const points=records.filter(record=>record.label===label),values=points.map(record=>record.value),minimum=Math.min(...values),maximum=Math.max(...values),q25=quantile(values,.25),median=quantile(values,.5),q75=quantile(values,.75),y=geometry.top+(index+.5)*geometry.rowHeight,tooltip=tooltipText(guide,[`${name}=${label}`,`min=${formatTick(minimum)}`,`q25=${formatTick(q25)}`,`median=${formatTick(median)}`,`q75=${formatTick(q75)}`,`max=${formatTick(maximum)}`,`support=${values.length}`],points[0].item);parts.push(`<g class="interactive-mark" ${markAttributes(tooltip)}><line x1="${sx(minimum)}" y1="${y}" x2="${sx(maximum)}" y2="${y}" stroke="${COLORS[0]}"/><rect class="box-mark" x="${sx(q25)}" y="${y-7}" width="${Math.max(1,sx(q75)-sx(q25))}" height="14"/><line x1="${sx(median)}" y1="${y-9}" x2="${sx(median)}" y2="${y+9}" stroke="${COLORS[0]}" stroke-width="2"/></g>`) });svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:records.length,xCount:labels.length,groups:labels.length,wide:geometry.left>360,note:"箱线图展示 min、q25、median、q75、max"}}
function chartKind(guide){return({timeline:"timeline",series:"timeline",heatmap:"matrix",matrix:"matrix",bar:"bar",distribution:"distribution",scatter:"dot"}[guide.visualization]||"dot")}
function drawMetric(svg,guide,observations,yScaleMode="linear"){const kind=chartKind(guide),finiteObservations=observations.filter(item=>finite(item.value));if(kind==="matrix"&&observations.length){const result=drawMatrix(svg,guide,observations);return{...result,note:`${localizeProse(result.visualKind||kind)} · ${result.note}`}}if(!finiteObservations.length){svg.outerHTML='<div class="empty">当前诊断产物没有有限观测；未生成替代图。</div>';return{support:0,xCount:0,groups:0,note:"未采集"}}observations=finiteObservations;const result=kind==="timeline"?drawTimeline(svg,guide,observations,yScaleMode):kind==="bar"?drawBars(svg,guide,observations):kind==="distribution"?drawDistribution(svg,guide,observations):drawDotPlot(svg,guide,observations);return{...result,note:`${localizeProse(result.visualKind||kind)} · ${result.note}`}}
function drawCategorical(svg,path,rawValues){const values=rawValues.flatMap(value=>Array.isArray(value)?value:[value]).filter(value=>value!==null&&value!==undefined),counts=new Map();values.forEach(value=>{const label=String(value);counts.set(label,(counts.get(label)||0)+1)});const entries=[...counts].sort((a,b)=>b[1]-a[1]);if(!entries.length){svg.outerHTML='<div class="empty">该字段没有可计数的 categorical observation。</div>';return{support:0,categories:0}}const labels=entries.map(entry=>entry[0]),geometry=horizontalGeometry(labels),domain=paddedExtent(entries.map(entry=>entry[1]),true),parts=[],sx=drawNumericXAxis(parts,domain,geometry,"count"),zero=sx(0);configureSvg(svg,geometry.height,geometry);drawFullCategoryLabels(parts,labels,geometry,geometry.rowHeight);entries.forEach(([label,count],index)=>{const y=geometry.top+index*geometry.rowHeight+6,tooltip=`${path}\ncategory=${label}\ncount=${count}`;parts.push(`<rect class="bar-mark interactive-mark" ${markAttributes(tooltip)} x="${zero}" y="${y}" width="${Math.max(1,sx(count)-zero)}" height="${geometry.rowHeight-12}" fill="${COLORS[index%COLORS.length]}"/>`)});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:values.length,categories:entries.length}}

function drawCompositeTrend(svg,title,entries,coordinateName,yScaleMode="linear"){if(!coordinateName)throw new Error(`组合趋势图 ${title} 没有声明固定横坐标`);const records=[];for(const{guide,observations}of entries){const rows=observations.map(item=>item.row),componentAliases=aliasDomain(rows.map(row=>dimensionValue(row,"response_component_id")).filter(value=>value!=null),"output"),hasOutputIdentity=rows.some(row=>dimensionValue(row,"response_component_id")!=null),hasSiteIdentity=rows.some(row=>dimensionValue(row,"module_path")!=null||dimensionValue(row,"tap_id")!=null||dimensionValue(row,"node_id")!=null),collapse=(guide.category==="training"||guide.category==="runtime")&&!hasOutputIdentity&&!hasSiteIdentity;observations.forEach(item=>{const x=Number(dimensionValue(item.row,coordinateName));if(!Number.isFinite(x))return;const temporal=temporalSeriesIdentity(item.row,guide.category,componentAliases,coordinateName),identity=collapse?guide.label:entries.length===1?temporal:`${guide.label} · ${temporal}`;records.push({value:item.value,x,group:identity,item})})}return drawTrendRecords(svg,{label:title,path:"composite_trend"},records,coordinateName,yScaleMode,coordinateName==="update")}
function comparisonIdentity(row,category,componentAliases){const module=dimensionValue(row,"module_site_id")??dimensionValue(row,"module_path")??dimensionValue(row,"node_id"),response=dimensionValue(row,"response_component_id"),responseLabel=componentAliases.get(String(response))??readableIdentity(response,0,"output"),condition=dimensionValue(row,"__condition__")??dimensionValue(row,"rollout_condition_id"),scope=dimensionValue(row,"scope");if(category==="rollout")return[response==null?null:responseLabel,dimensionValue(row,"horizon")==null?null:`h${dimensionValue(row,"horizon")}`].filter(Boolean).join(" · ")||"rollout";return[seriesFamily(module??scope??responseLabel??"all"),condition==null?null:conditionFamily(condition)].filter(Boolean).join(" · ")}
function drawCompositeComparison(svg,title,entries){const allObservations=entries.flatMap(entry=>entry.observations),componentAliases=aliasDomain(allObservations.map(item=>dimensionValue(item.row,"response_component_id")).filter(value=>value!=null),"output"),fixedAvailable=allObservations.some(item=>dimensionValue(item.row,"cohort_policy")==="fixed_complete"),horizons=allObservations.map(item=>Number(dimensionValue(item.row,"horizon"))).filter(Number.isFinite),maximumHorizon=horizons.length?Math.max(...horizons):null,records=[];entries.forEach(({guide,observations},metricIndex)=>observations.filter(item=>!fixedAvailable||dimensionValue(item.row,"cohort_policy")==="fixed_complete").filter(item=>maximumHorizon==null||!Number.isFinite(Number(dimensionValue(item.row,"horizon")))||Number(dimensionValue(item.row,"horizon"))===maximumHorizon).forEach(item=>records.push({metric:guide.label,metricIndex,label:comparisonIdentity(item.row,guide.category,componentAliases),value:item.value,item})));if(!records.length){svg.outerHTML='<div class="empty">组合比较没有有限 observation。</div>';return{support:0,xCount:0,groups:0,note:"no finite comparison"}}const metrics=entries.map(entry=>entry.guide.label),allLabels=unique(records.map(record=>record.label)),primaryMetric=metrics[0],ranked=allLabels.map(label=>{const values=records.filter(record=>record.label===label&&record.metric===primaryMetric).map(record=>record.value),mean=values.length?values.reduce((total,value)=>total+value,0)/values.length:null;return{label,score:mean==null?-Infinity:Math.abs(mean)}}).sort((left,right)=>right.score-left.score),labels=ranked.map(item=>item.label),visible=records.filter(record=>labels.includes(record.label)),geometry=horizontalGeometry(labels),domain=paddedExtent(visible.map(record=>record.value),visible.some(record=>record.value<0)),parts=[],sx=drawNumericXAxis(parts,domain,geometry,title);configureSvg(svg,geometry.height,geometry);drawFullCategoryLabels(parts,labels,geometry,geometry.rowHeight);drawLegend(parts,metrics,geometry.left);if(domain[0]<=0&&domain[1]>=0){const zero=sx(0);parts.push(`<line x1="${zero}" y1="${geometry.top}" x2="${zero}" y2="${geometry.bottom}" class="reference-line"/>`)}labels.forEach((label,rowIndex)=>{const center=geometry.top+(rowIndex+.5)*geometry.rowHeight,key=`comparison-${rowIndex}`,byMetric=metrics.map((metric,metricIndex)=>{const points=visible.filter(record=>record.label===label&&record.metric===metric),mean=points.length?points.reduce((total,record)=>total+record.value,0)/points.length:null;return{metric,metricIndex,points,mean}}).filter(item=>item.mean!==null);if(byMetric.length>=2){const xs=byMetric.map(item=>sx(item.mean));parts.push(`<line class="direction-stem interactive-mark" data-series-key="${key}" x1="${Math.min(...xs)}" y1="${center}" x2="${Math.max(...xs)}" y2="${center}" stroke="var(--muted)"/>`)}byMetric.forEach(item=>{const tooltip=[title,label,`${item.metric}=${formatTick(item.mean)}`,`support=${item.points.length}`].join("\n");parts.push(`<circle class="mean-point interactive-mark" ${markAttributes(tooltip,key)} cx="${sx(item.mean)}" cy="${center+(item.metricIndex-(metrics.length-1)/2)*3}" r="2.6" fill="${COLORS[item.metricIndex%COLORS.length]}"/>`)})});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:visible.length,xCount:metrics.length,groups:labels.length,wide:geometry.left>360,note:"按模块族/输出聚合，展示全部比较对象"}}

// 多目标关系共享同一输出行，避免三个独立图因标签和样本维度不同而出现尺寸失衡。
function drawObjectiveOverview(svg,title,entries){
  const paths=["gradient_cosine","norm_ratio","partition_coverage","joint_active_parameter_fraction"],byPath=new Map(entries.map(entry=>[entry.guide.path,entry])),allObservations=entries.flatMap(entry=>entry.observations),outputs=unique(allObservations.map(item=>dimensionValue(item.row,"response_component_id")).filter(value=>value!=null)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));
  if(!outputs.length||paths.some(path=>!byPath.has(path))){
    svg.outerHTML='<div class="empty">输出目标关系需要梯度方向、尺度、目标分区和共同参数覆盖四项配对证据。</div>';
    return{support:0,xCount:0,groups:0,note:"缺少固定配对指标"};
  }
  const rowHeight=22,width=1480,left=Math.min(430,Math.max(260,labelMargin(outputs))),right=1450,top=112,bottom=top+outputs.length*rowHeight,height=bottom+52,columnGap=12,columnWidth=(right-left-columnGap*(paths.length-1))/paths.length,parts=[],aggregates=new Map(),metricTitles={gradient_cosine:"梯度余弦",norm_ratio:"梯度范数比",partition_coverage:"目标分区覆盖率",joint_active_parameter_fraction:"共同活跃参数比例"},metricNotes={gradient_cosine:"−1 到 1；0 表示局部正交",norm_ratio:"以 1 为尺度平衡中心",partition_coverage:"1 表示训练目标分区完整",joint_active_parameter_fraction:"两个目标都有梯度的参数比例"};
  configureSvg(svg,height,{width,left,right,top,bottom,height});
  paths.forEach(path=>{
    const entry=byPath.get(path);
    outputs.forEach(output=>{
      const points=entry.observations.filter(item=>dimensionValue(item.row,"response_component_id")===output&&finite(item.value)),values=points.map(item=>item.value),mean=values.length?values.reduce((total,value)=>total+value,0)/values.length:null;
      aggregates.set(`${path}\u0000${output}`,{points,values,mean,minimum:values.length?Math.min(...values):null,maximum:values.length?Math.max(...values):null});
    });
  });
  const ratioLogs=outputs.map(output=>aggregates.get(`norm_ratio\u0000${output}`)?.mean).filter(value=>finite(value)&&value>0).map(value=>Math.abs(Math.log10(value))),ratioMagnitude=Math.max(quantile(ratioLogs,.95)||0,1e-12);
  parts.push(`<text x="${left}" y="30" class="facet-title">${esc(title)}</text><text x="${left}" y="52" class="control-note">每行对应一个输出通道；单元格为同一输出在有效样本上的均值，悬停可查看范围和观测数。</text>`);
  drawFullCategoryLabels(parts,outputs,{left,top,bottom},rowHeight);
  paths.forEach((path,columnIndex)=>{
    const x=left+columnIndex*(columnWidth+columnGap);
    parts.push(`<text x="${x+columnWidth/2}" y="78" text-anchor="middle" class="facet-title">${metricTitles[path]}</text><text x="${x+columnWidth/2}" y="96" text-anchor="middle" class="control-note">${metricNotes[path]}</text>`);
    outputs.forEach((output,rowIndex)=>{
      const aggregate=aggregates.get(`${path}\u0000${output}`),y=top+rowIndex*rowHeight+1,cellX=x+1,cellWidth=columnWidth-2,cellHeight=rowHeight-2;
      if(!aggregate||aggregate.mean==null){
        parts.push(`<rect class="matrix-cell-missing" x="${cellX}" y="${y}" width="${cellWidth}" height="${cellHeight}" rx="3" fill="transparent"/>`);
        return;
      }
      const mean=aggregate.mean,isCoverage=["partition_coverage","joint_active_parameter_fraction"].includes(path),supportValues=aggregate.points.map(point=>Number(dimensionValue(point.row,"support_count"))).filter(Number.isFinite),supportText=supportValues.length?formatTick(supportValues.reduce((total,value)=>total+value,0)/supportValues.length):"未提供",color=path==="gradient_cosine"?matrixColor(mean,1,true):path==="norm_ratio"?matrixColor(Math.log10(Math.max(mean,1e-12)),ratioMagnitude,true):`color-mix(in srgb,var(--viz-5) ${Math.round(12+82*Math.max(0,Math.min(1,mean)))}%,var(--panel))`,interpretation=path==="gradient_cosine"?"负值表示该参数点附近与其余目标的梯度方向相反":path==="norm_ratio"?"1 表示目标与其余目标的梯度尺度相等":path==="partition_coverage"?"1 表示目标分区完整":"值越高表示两个目标共同作用的参数比例越大",tooltip=[metricTitles[path],`输出通道：${output}`,`均值：${formatTick(mean)}`,`最小值：${formatTick(aggregate.minimum)}`,`最大值：${formatTick(aggregate.maximum)}`,`有限观测：${aggregate.values.length}`,`平均有效监督 support：${supportText}`,interpretation].join("\n"),textColor=isCoverage&&mean>.65||path==="gradient_cosine"&&Math.abs(mean)>.55||path==="norm_ratio"&&Math.abs(Math.log10(Math.max(mean,1e-12)))>ratioMagnitude*.6?"#fff":"var(--ink)";
      parts.push(`<rect class="matrix-cell interactive-mark" ${markAttributes(tooltip)} x="${cellX}" y="${y}" width="${cellWidth}" height="${cellHeight}" rx="3" fill="${color}"/><text x="${cellX+cellWidth/2}" y="${y+cellHeight/2+3.5}" text-anchor="middle" font-size="9.5" font-weight="650" fill="${textColor}" pointer-events="none">${esc(formatTick(mean))}</text>`);
    });
  });
  svg.innerHTML=parts.join("");
  bindChartInteractions(svg);
  return{support:allObservations.filter(item=>finite(item.value)).length,xCount:paths.length,groups:outputs.length,note:"四个指标共用同一输出行和相同单元格尺寸；悬停同时显示有效监督 support"};
}

function rolloutScalar(entries,path,output,horizon,cohortPolicy){
  const entry=entries.find(candidate=>candidate.guide.path===path);
  if(!entry)return null;
  const values=entry.observations.filter(item=>dimensionValue(item.row,"rollout_condition_id")==="free"&&dimensionValue(item.row,"cohort_policy")===cohortPolicy&&dimensionValue(item.row,"response_component_id")===output&&Number(dimensionValue(item.row,"horizon"))===horizon&&finite(item.value)).map(item=>item.value);
  return values.length?values.reduce((total,value)=>total+value,0)/values.length:null;
}

// 同一 evidence source 只加载一次完整 rollout row；概览图和覆盖表共享这些行。
async function rolloutEvidenceRows(entries){
  const sources=new Map();
  entries.forEach(({guide})=>guideSources(guide).forEach(source=>sources.set(source.name,source)));
  const batches=await Promise.all([...sources.values()].map(async source=>(await load(source.name,"rows")).map(row=>enrichComponentIdentity(row,source))));
  return batches.flat().filter(row=>row.record_kind==="rollout_stability"||row.analyzer==="final_rollout");
}

function rolloutCohortPolicy(rows,horizon){
  return rows.some(row=>Number(row.horizon)===horizon&&row.cohort_policy==="fixed_complete"&&row.status==="success")?"fixed_complete":"available";
}

// 每个预测长度使用独立坐标区和汇总栏；汇总指标选择只改变明确标注的 scalar，不会把 MAE 曲线冒充 RMSE。
function drawRolloutOverview(svg,title,entries,selectedOutput=null,selectedSummaryMetric="rmse",yScaleMode="linear"){
  const meanEntry=entries.find(entry=>entry.guide.path==="timestep_mean_absolute_error"),q90Entry=entries.find(entry=>entry.guide.path==="timestep_q90_absolute_error");
  if(!meanEntry||!q90Entry){
    svg.outerHTML='<div class="empty">自由滚动概览需要逐时间步平均绝对误差和 q90 两项证据。</div>';
    return{support:0,xCount:0,groups:0,note:"缺少固定配对指标",availableGroups:[]};
  }
  const summaryMetric=selectedSummaryMetric==="srmse"?"srmse":"rmse",summaryLabel=summaryMetric==="srmse"?"sRMSE":"RMSE";
  const freeMean=meanEntry.observations.filter(item=>dimensionValue(item.row,"rollout_condition_id")==="free"),outputs=unique(freeMean.map(item=>dimensionValue(item.row,"response_component_id")).filter(value=>value!=null)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true})),activeOutput=outputs.includes(selectedOutput)?selectedOutput:outputs[0];
  if(!activeOutput){
    svg.outerHTML='<div class="empty">没有可绘制的自由反馈输出通道。</div>';
    return{support:0,xCount:0,groups:0,note:"没有自由反馈证据",availableGroups:[]};
  }
  const outputMean=freeMean.filter(item=>dimensionValue(item.row,"response_component_id")===activeOutput),horizons=unique(outputMean.map(item=>dimensionValue(item.row,"horizon")).filter(value=>finite(Number(value)))).map(Number).sort((left,right)=>left-right),width=1200,facetHeight=356,height=34+horizons.length*facetHeight,plotLeft=86,plotRight=850,summaryLeft=890,right=1170,parts=[],hoverPoints=[];
  configureSvg(svg,height,{width,left:plotLeft,right,top:34,bottom:height-24,height});
  parts.push(`<text x="${plotLeft}" y="24" class="facet-title">${esc(activeOutput)} · 自由反馈</text><text x="${right}" y="24" text-anchor="end" class="control-note">汇总误差指标：${summaryLabel}；曲线始终为原始尺度 mean |error| 与 q90</text>`);
  let support=0,attentionCount=0;
  horizons.forEach((horizon,horizonIndex)=>{
    const candidates=outputMean.filter(item=>Number(dimensionValue(item.row,"horizon"))===horizon),fixedCandidates=candidates.filter(item=>dimensionValue(item.row,"cohort_policy")==="fixed_complete"),cohortPolicy=fixedCandidates.length?"fixed_complete":"available",meanMap=new Map(),q90Map=new Map(),eligible=item=>dimensionValue(item.row,"rollout_condition_id")==="free"&&dimensionValue(item.row,"cohort_policy")===cohortPolicy&&dimensionValue(item.row,"response_component_id")===activeOutput&&Number(dimensionValue(item.row,"horizon"))===horizon;
    meanEntry.observations.filter(eligible).forEach(item=>meanMap.set(`${item.source.name}:${item.rowIndex}:${item.seriesIndex}`,item));
    q90Entry.observations.filter(eligible).forEach(item=>q90Map.set(`${item.source.name}:${item.rowIndex}:${item.seriesIndex}`,item));
    const points=[];
    for(const[key,mean]of meanMap){
      const q90=q90Map.get(key);
      if(q90&&mean.seriesIndex!=null)points.push({x:Number(mean.seriesIndex)+1,mean:mean.value,q90:q90.value,item:mean});
    }
    points.sort((left,right)=>left.x-right.x);
    const cursor=34+horizonIndex*facetHeight,plotTop=cursor+52,plotBottom=cursor+276,facetGeometry={left:plotLeft,right:plotRight,top:plotTop,bottom:plotBottom,height:cursor+326},color=COLORS[horizonIndex%COLORS.length];
    parts.push(`<line x1="${plotLeft}" y1="${cursor+8}" x2="${right}" y2="${cursor+8}" class="gridline"/><text x="${plotLeft}" y="${cursor+34}" class="facet-title">预测 ${horizon} 步</text><text x="${plotRight}" y="${cursor+34}" text-anchor="end" class="control-note">${esc(displayValue(cohortPolicy))}</text>`);
    if(!points.length){
      parts.push(`<text x="${(plotLeft+plotRight)/2}" y="${(plotTop+plotBottom)/2}" text-anchor="middle" class="control-note">该预测长度没有可配对的 mean/q90 证据</text>`);
      return;
    }
    const values=points.flatMap(point=>[point.mean,point.q90]),xDomain=[0,Math.max(...points.map(point=>point.x))],yDomain=boundedPaddedExtent(values),sx=drawNumericXAxis(parts,xDomain,facetGeometry,"timestep"),sy=drawNumericYAxis(parts,yDomain,facetGeometry,"absolute error",yScaleMode),key=`rollout-${horizon}`,band=[...points.map(point=>`${sx(point.x)},${sy(point.q90)}`),...points.slice().reverse().map(point=>`${sx(point.x)},${sy(point.mean)}`)].join(" "),meanCoordinates=points.map(point=>`${sx(point.x)},${sy(point.mean)}`).join(" "),q90Coordinates=points.map(point=>`${sx(point.x)},${sy(point.q90)}`).join(" ");
    parts.push(`<polygon class="tail-band" points="${band}" fill="${color}"/><polyline class="tail-boundary" points="${q90Coordinates}" stroke="${color}"/><polyline class="series-line interactive-mark" data-series-key="${key}" points="${meanCoordinates}" stroke="${color}"/><polyline class="line-hit interactive-mark" ${markAttributes([title,`输出=${activeOutput}`,`预测长度=${horizon}`,`${displayValue(cohortPolicy)} · 自由反馈`,`末端均值=${formatTick(points.at(-1).mean)}`,`末端 q90=${formatTick(points.at(-1).q90)}`].join("\n"),key)} points="${meanCoordinates}"/>`);
    const normalPoints=[],attentionPoints=[];
    points.forEach((point,index)=>{
      const thresholdValue=dimensionValue(point.item.row,"instability_threshold"),threshold=thresholdValue==null?null:Number(thresholdValue),reason=threshold!=null&&Number.isFinite(threshold)&&point.q90>=threshold?`q90 达到配置阈值 ${formatTick(threshold)}`:robustAttention(points,index),x=sx(point.x),y=sy(point.mean),tooltip=[title,`输出=${activeOutput}`,`预测长度=${horizon}`,`时间步=${point.x}`,`平均绝对误差=${formatTick(point.mean)}`,`绝对误差 q90=${formatTick(point.q90)}`,reason?`需关注=${reason}`:null].filter(Boolean).join("\n");
      if(reason){attentionCount+=1;attentionPoints.push({x,y})}else normalPoints.push({x,y});
      hoverPoints.push({x,y,tooltip,seriesKey:key,attention:Boolean(reason)});
    });
    if(normalPoints.length)parts.push(`<path class="point-cloud interactive-mark" data-series-key="${key}" d="${pointCloudPath(normalPoints)}" stroke="${color}"/>`);
    if(attentionPoints.length)parts.push(`<path class="attention-cloud interactive-mark" data-series-key="${key}" d="${pointCloudPath(attentionPoints)}"/>`);
    const summaries=[[summaryLabel,rolloutScalar(entries,summaryMetric,activeOutput,horizon,cohortPolicy)],["最大局部误差斜率",rolloutScalar(entries,"rolling_local_slope_max",activeOutput,horizon,cohortPolicy)],["不稳定片段数",rolloutScalar(entries,"unstable_episode_count",activeOutput,horizon,cohortPolicy)],["达到阈值的时间中位数",rolloutScalar(entries,"time_to_threshold_median",activeOutput,horizon,cohortPolicy)]];
    parts.push(`<text x="${summaryLeft}" y="${plotTop+6}" class="facet-title">同一输出的汇总</text>`);
    summaries.forEach(([label,value],index)=>{const y=plotTop+36+index*38;parts.push(`<text x="${summaryLeft}" y="${y}" class="control-note">${esc(label)}</text><text x="${right}" y="${y}" text-anchor="end" class="facet-title">${value==null?"—":esc(formatTick(value))}</text>`)});
    parts.push(`<line x1="${summaryLeft-18}" y1="${plotTop}" x2="${summaryLeft-18}" y2="${plotBottom}" class="gridline"/>`);
    support+=points.length*2;
  });
  parts.push('<circle class="hover-point" visibility="hidden" r="5"/>');
  svg.innerHTML=parts.join("");
  registerHoverPoints(svg,hoverPoints);
  bindChartInteractions(svg);
  svg.__supportsYScale=true;
  svg.__zoomRender=(target,scaleMode=yScaleMode)=>drawRolloutOverview(target,title,entries,activeOutput,summaryMetric,scaleMode);
  return{support,xCount:horizons.length,groups:1,attentionCount,note:`${horizons.map(horizon=>`预测 ${horizon} 步`).join("、")} 使用相互独立的坐标区；实线为 mean |error|，色带上界为 q90，汇总显示 ${summaryLabel}`,availableGroups:outputs,selectedGroup:activeOutput,selectedSummaryMetric:summaryMetric};
}

// 两个 horizon 共享输出行和每项指标的色阶，便于同时查看全通道排名、缺失和长度效应。
function drawRolloutHeatmap(svg,title,entries,rows){
  const paths=["rmse","srmse","rolling_local_slope_max","unstable_episode_count"],labels={rmse:"RMSE",srmse:"sRMSE",rolling_local_slope_max:"最大局部斜率",unstable_episode_count:"不稳定片段数"},freeRows=rows.filter(row=>row.rollout_condition_id==="free"),horizons=unique(freeRows.map(row=>Number(row.horizon)).filter(Number.isFinite)).sort((left,right)=>left-right),policies=new Map(horizons.map(horizon=>[horizon,rolloutCohortPolicy(freeRows,horizon)])),outputs=unique(freeRows.map(row=>row.response_component_id).filter(value=>value!=null)).sort((left,right)=>left.localeCompare(right,"zh-CN",{numeric:true}));
  if(!horizons.length||!outputs.length){
    svg.outerHTML='<div class="empty">没有可用于全通道概览的 free rollout 记录。</div>';
    return{support:0,xCount:0,groups:0,note:"没有 rollout rows"};
  }
  const width=1500,left=Math.min(430,Math.max(260,labelMargin(outputs))),right=1460,top=126,rowHeight=19,bottom=top+outputs.length*rowHeight,height=bottom+50,groupGap=24,groupWidth=(right-left-groupGap*Math.max(0,horizons.length-1))/horizons.length,cellWidth=groupWidth/paths.length,parts=[],patternId=`rollout-missing-${Math.random().toString(36).slice(2)}`,magnitudes=new Map();
  configureSvg(svg,height,{width,left,right,top,bottom,height});
  parts.push(`<defs><pattern id="${patternId}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="var(--line)" stroke-width="2"/></pattern></defs><text x="${left}" y="28" class="facet-title">${esc(title)}</text><text x="${left}" y="50" class="control-note">每行一个输出通道；相同指标在不同预测长度间共用色阶；斜纹表示该组合没有有限证据。</text>`);
  drawFullCategoryLabels(parts,outputs,{left,top,bottom},rowHeight);
  paths.forEach(path=>{
    const values=freeRows.filter(row=>row.status==="success"&&row.cohort_policy===policies.get(Number(row.horizon))&&finite(row[path])).map(row=>Number(row[path]));
    const scaleValues=path==="rolling_local_slope_max"?values.map(Math.abs):values;
    magnitudes.set(path,Math.max(quantile(scaleValues,.95)||0,path==="unstable_episode_count"?1:1e-12));
  });
  let support=0,missingCount=0;
  horizons.forEach((horizon,horizonIndex)=>{
    const groupLeft=left+horizonIndex*(groupWidth+groupGap),policy=policies.get(horizon);
    parts.push(`<text x="${groupLeft+groupWidth/2}" y="76" text-anchor="middle" class="facet-title">预测 ${horizon} 步</text><text x="${groupLeft+groupWidth/2}" y="94" text-anchor="middle" class="control-note">${esc(displayValue(policy))}</text>`);
    paths.forEach((path,pathIndex)=>{
      const x=groupLeft+pathIndex*cellWidth,magnitude=magnitudes.get(path);
      parts.push(`<text x="${x+cellWidth/2}" y="116" text-anchor="middle" class="label">${esc(labels[path])}</text>`);
      outputs.forEach((output,rowIndex)=>{
        const row=freeRows.find(candidate=>Number(candidate.horizon)===horizon&&candidate.cohort_policy===policy&&candidate.response_component_id===output),cellX=x+1,y=top+rowIndex*rowHeight+1,w=cellWidth-2,h=rowHeight-2;
        if(!row||row.status!=="success"||!finite(row[path])){
          missingCount+=1;
          const tooltip=[`输出通道：${output}`,`预测长度：${horizon}`,`指标：${labels[path]}`,"数值：缺失",row?.skip_reason?`原因：${displayValue(row.skip_reason)}`:null].filter(Boolean).join("\n");
          parts.push(`<rect class="matrix-cell-missing interactive-mark" ${markAttributes(tooltip)} x="${cellX}" y="${y}" width="${w}" height="${h}" rx="2" fill="url(#${patternId})"/>`);
          return;
        }
        support+=1;
        const value=Number(row[path]),signed=path==="rolling_local_slope_max",clipped=Math.abs(value)>magnitude,colorValue=signed?Math.max(-magnitude,Math.min(value,magnitude)):Math.min(value,magnitude),tooltip=[`输出通道：${output}`,`预测长度：${horizon}`,`样本集合：${displayValue(policy)}`,`${labels[path]}：${formatTick(value)}`,`有效样本：${row.item_count??"—"}`,`有效元素：${row.valid_element_count??"—"}`,clipped?"颜色按全部通道的 95% 分位截断":null].filter(Boolean).join("\n");
        parts.push(`<rect class="matrix-cell interactive-mark" ${markAttributes(tooltip)} x="${cellX}" y="${y}" width="${w}" height="${h}" rx="2" fill="${matrixColor(colorValue,magnitude,signed)}"${clipped?' stroke="var(--accent)" stroke-width="1.5"':""}/>`);
      });
    });
  });
  svg.innerHTML=parts.join("");
  bindChartInteractions(svg);
  return{support,xCount:horizons.length*paths.length,groups:outputs.length,note:`${horizons.length} 个预测长度 × ${paths.length} 项指标；${missingCount} 个缺失单元`};
}

function countSummary(value,prefix=""){
  if(value==null)return"";
  if(typeof value!=="object")return`${prefix}${value}`;
  return Object.entries(value).flatMap(([key,item])=>typeof item==="object"?countSummary(item,prefix?`${prefix}${key}/`:`${key}/`):Number(item)>0?[`${prefix}${key}：${item}`]:[]).join("；");
}
function rolloutExclusionSummary(row){
  if(row.cohort_policy==="available"){
    return countSummary(row.exclusion_reason_count_by_condition?.[row.rollout_condition_id]);
  }
  return countSummary(row.fixed_complete_exclusion_reason_counts);
}

// 覆盖表保留所有 success/skip row；分页只限制 DOM 数量，不删除任何 artifact 记录。
function drawRolloutCoverage(svg,title,rows){
  const id=`rollout-coverage-${Math.random().toString(36).slice(2)}`,freeRows=rows.filter(row=>row.rollout_condition_id==="free").sort((left,right)=>Number(left.horizon)-Number(right.horizon)||String(left.response_component_id).localeCompare(String(right.response_component_id),"zh-CN",{numeric:true})||String(left.cohort_policy).localeCompare(String(right.cohort_policy))),horizons=unique(freeRows.map(row=>row.horizon)).sort((left,right)=>Number(left)-Number(right)),policies=unique(freeRows.map(row=>row.cohort_policy)),statuses=unique(freeRows.map(row=>row.status));
  svg.outerHTML=`<div id="${id}" class="rollout-coverage"><div class="influence-summary"><span>记录：${freeRows.length.toLocaleString()}</span><span>成功：${freeRows.filter(row=>row.status==="success").length.toLocaleString()}</span><span>缺失：${freeRows.filter(row=>row.status!=="success").length.toLocaleString()}</span><span>输出通道：${unique(freeRows.map(row=>row.response_component_id)).length.toLocaleString()}</span></div><div class="toolbar"><select data-filter="horizon"><option value="">全部预测长度</option>${horizons.map(value=>`<option value="${esc(value)}">预测 ${esc(value)} 步</option>`).join("")}</select><select data-filter="policy"><option value="">全部样本集合</option>${policies.map(value=>`<option value="${esc(value)}">${esc(displayValue(value))}</option>`).join("")}</select><select data-filter="status"><option value="">全部状态</option>${statuses.map(value=>`<option value="${esc(value)}">${esc(displayValue(value))}</option>`).join("")}</select><input data-filter="search" placeholder="搜索输出通道或缺失原因"><button type="button" data-page="previous">上一页</button><button type="button" data-page="next">下一页</button></div><div data-role="summary" class="muted"></div><div class="table"><table><thead><tr><th>输出通道</th><th>场景</th><th>预测长度</th><th>样本集合</th><th>状态</th><th>候选样本</th><th>有效样本</th><th>全条件完整样本</th><th>完整样本判定长度</th><th>有效元素</th><th>局部斜率窗口</th><th>不稳定阈值</th><th>缺失原因</th><th>排除原因计数</th></tr></thead><tbody></tbody></table></div></div>`;
  const host=document.getElementById(id),tbody=host.querySelector("tbody"),summary=host.querySelector('[data-role="summary"]'),horizonFilter=host.querySelector('[data-filter="horizon"]'),policyFilter=host.querySelector('[data-filter="policy"]'),statusFilter=host.querySelector('[data-filter="status"]'),search=host.querySelector('[data-filter="search"]'),pageSize=100;let page=0,searchTimer=0;
  const render=()=>{const query=search.value.trim().toLowerCase(),filtered=freeRows.filter(row=>(!horizonFilter.value||String(row.horizon)===horizonFilter.value)&&(!policyFilter.value||row.cohort_policy===policyFilter.value)&&(!statusFilter.value||row.status===statusFilter.value)&&(!query||`${row.response_component_id} ${row.scenario_id??""} ${row.skip_reason??""} ${rolloutExclusionSummary(row)}`.toLowerCase().includes(query))),pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);const visible=filtered.slice(page*pageSize,(page+1)*pageSize);tbody.innerHTML=visible.map(row=>{const exclusions=rolloutExclusionSummary(row);return`<tr><td>${esc(row.response_component_id)}</td><td>${esc(row.scenario_id??"—")}</td><td>${esc(row.horizon)}</td><td>${esc(displayValue(row.cohort_policy))}</td><td>${esc(displayValue(row.status))}</td><td>${esc(row.candidate_item_count??"—")}</td><td>${esc(row.item_count??"—")}</td><td>${esc(row.fixed_complete_item_count??"—")}</td><td>${esc(row.fixed_complete_horizon??"—")}</td><td>${esc(row.valid_element_count??"—")}</td><td>${esc(row.local_slope_window??"—")}</td><td>${esc(row.instability_threshold??"—")}</td><td>${esc(row.skip_reason?displayValue(row.skip_reason):"—")}</td><td title="${esc(exclusions)}">${esc(exclusions||"—")}</td></tr>`}).join("")||'<tr><td colspan="14">当前筛选没有记录。</td></tr>';summary.textContent=`${filtered.length.toLocaleString()} 条记录 · 第 ${page+1}/${pages} 页 · 固定完整样本集与可用样本集不混合`};
  [horizonFilter,policyFilter,statusFilter].forEach(control=>control.onchange=()=>{page=0;render()});search.oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render()},120)};host.querySelector('[data-page="previous"]').onclick=()=>{page=Math.max(0,page-1);render()};host.querySelector('[data-page="next"]').onclick=()=>{page+=1;render()};render();
  return{support:freeRows.length,xCount:horizons.length,groups:unique(freeRows.map(row=>row.response_component_id)).length,note:"完整列出 success、structured skip、样本支持和排除原因；分页不裁剪源数据"};
}

function drawParityComposite(svg,title,entries){if(entries.length<2){svg.outerHTML='<div class="empty">parity view 需要两个配对指标。</div>';return{support:0,xCount:0,groups:0,note:"missing pair"}}const keyed=entries.map(({observations})=>{const map=new Map();observations.forEach(item=>map.set(`${item.source.name}:${item.rowIndex}`,item));return map}),pairs=[];for(const [key,left] of keyed[0]){const right=keyed[1].get(key);if(right)pairs.push({x:left.value,y:right.value,left,right})}if(!pairs.length){svg.outerHTML='<div class="empty">两个指标没有可验证的同 row pairing。</div>';return{support:0,xCount:0,groups:0,note:"no paired rows"}}const values=pairs.flatMap(pair=>[pair.x,pair.y]),domain=paddedExtent(values,true),geometry={left:112,right:930,top:58,bottom:350,height:440},parts=[],sx=drawNumericXAxis(parts,domain,geometry,entries[0].guide.label),sy=drawNumericYAxis(parts,domain,geometry,entries[1].guide.label);configureSvg(svg,geometry.height,geometry);parts.push(`<line x1="${sx(domain[0])}" y1="${sy(domain[0])}" x2="${sx(domain[1])}" y2="${sy(domain[1])}" class="reference-line"/>`);pairs.forEach(pair=>{const residual=pair.y-pair.x,tooltip=[title,`${entries[0].guide.label}=${formatTick(pair.x)}`,`${entries[1].guide.label}=${formatTick(pair.y)}`,`residual=${formatTick(residual)}`,identityText(pair.left.row)].filter(Boolean).join("\n");parts.push(`<circle class="raw-point interactive-mark" ${markAttributes(tooltip)} cx="${sx(pair.x)}" cy="${sy(pair.y)}" r="1.35" fill="${COLORS[0]}"/>`)});svg.innerHTML=parts.join("");bindChartInteractions(svg);return{support:pairs.length,xCount:pairs.length,groups:1,note:"同 row 配对；对角线 y=x 表示一阶预测与真实干预一致"}}
function influenceColumnDetails(guide){
  if(!["input","module"].includes(guide.category)||!["effect_value","normalized_effect"].includes(guide.path))return"";
  const relative=guide.path==="normalized_effect",response=relative?"相对响应":"响应",normalization=relative?"相对响应先用同一输出的 baseline 绝对值归一化；baseline 接近 0 时该列不能解释。":"";
  return`<dt>平均${response}</dt><dd>同一干预方法、输入或模块、输出通道下，有限 ${guide.path} 记录的算术平均。生产者若已经按 cohort 汇总，一条记录就代表该 cohort 的均值。</dd><dt>${response}范围</dt><dd>同一组合下有效配对样本或底层记录的最小值到最大值；它不是模型输出取值范围，也不是置信区间。${esc(normalization)}</dd><dt>恒等对照平均响应</dt><dd>不改变输入或模块输出，使用相同 checkpoint、样本、eval mode 和随机流重复计算得到的平均 ${response}。输入干预按输出通道共享该对照；模块干预按模块与输出通道配对。</dd><dt>恒等对照响应范围</dt><dd>恒等对照在有效配对样本上的最小响应到最大响应。实际干预只有明显超过这一区间时，才有证据表明响应不是重复执行噪声。</dd><dt>缺失与有效性核对</dt><dd>主表不再堆叠 support、改动元素、样本覆盖和路径核对列。仅当这些审计字段显示缺失、未配对或核对失败时，才进入折叠的“缺失与有效性核对”；全部原始字段仍保留在 artifact 和原始证据中。</dd><dt>正负方向</dt><dd>正值表示干预后任务指标变差，负值表示干预后任务指标改善，0 表示该指标下没有可测响应；这是模型干预响应，不是物理因果效应。</dd>`;
}
function guideDetails(guide){return`<details><summary>指标说明</summary><dl><dt>指标字段</dt><dd><code>${esc(guide.path)}</code></dd><dt>如何计算</dt><dd>${esc(guide.description)}</dd><dt>诊断用途</dt><dd>${esc(guide.reading)}</dd><dt>参考与对照</dt><dd>${esc(guide.reference)}</dd><dt>不可解释条件</dt><dd>${esc(guide.invalid_when)}</dd>${influenceColumnDetails(guide)}</dl><p class="muted">指标只回答本段声明的问题；完整样本、模块、条件和来源身份请在“证据与原始数据”栏目审查。</p></details>`}
let renderToken=0;
// 卡片跨度由功能和图形契约固定，不再根据本次数据行数临时改变。
function metricPanelLayout(guide,mode){
  if(mode==="detail")return"layout-full";
  if(guide.category==="training")return"layout-full";
  if(chartKind(guide)==="matrix")return"layout-full layout-matrix";
  if(["input","module","multi_objective","rollout"].includes(guide.category))return"layout-full";
  return"";
}
async function renderMetricPanel(guide,host,token,purpose,mode="direction"){
  const localized=displayGuide(guide),id=`metric-${Math.random().toString(36).slice(2)}`,badge=mode==="status"?"有效性检查":mode==="detail"?"补充证据":ROLE_LABELS[guide.role]||localizeProse(guide.role),layout=metricPanelLayout(localized,mode);
  host.insertAdjacentHTML("beforeend",`<article class="metric-panel ${layout}"><div class="metric-head"><div><span class="badge">${esc(displayTitle(purpose.title))} · ${esc(badge)}</span><h3>${esc(localized.label)}</h3></div></div><div class="chart-viewport"><svg id="${id}" viewBox="0 0 1000 430" role="img" aria-label="${esc(localized.label)}"></svg></div><div class="metric-meta">正在加载对应证据。</div>${guideDetails(localized)}</article>`);
  const panel=document.getElementById(id).closest(".metric-panel");
  try{
    const observations=await observationsForGuide(guide);
    if(token!==renderToken)return;
    const svg=document.getElementById(id),sources=guideSources(guide),update=()=>{
      const result=drawMetric(svg,localized,observations,"linear");
      panel.querySelector(".metric-meta").textContent=`${result.support.toLocaleString()} 条观测 · ${result.xCount} 个横轴位置 · ${result.groups} 个比较对象 · ${localizeProse(result.note)} · 来源：${unique(sources.map(source=>source.group)).map(localizeProse).join("、")}`;
    };
    update();
    if(panel.querySelector("svg"))bindZoom(panel,localized.label);
  }catch(error){
    panel.querySelector(".metric-meta").innerHTML=`<span class="error">${esc(error.message||error)}</span>`;
  }
}

function sensitivityRecords(observations){
  return observations.map(item=>({item,input:String(dimensionValue(item.row,"intervened_component_id")??"未命名输入"),output:String(dimensionValue(item.row,"response_component_id")??"未命名输出"),scale:Number(dimensionValue(item.row,"scale")),value:Number(item.value)})).filter(record=>Number.isFinite(record.scale)&&Number.isFinite(record.value));
}

async function renderInputSensitivityPanel(view,guides,host,token,purpose){
  // 固定矩阵和尺度曲线共享同一批记录，不根据数据形态切换图形。
  const guide=displayGuide(guides[0]),id=`sensitivity-${Math.random().toString(36).slice(2)}`;
  host.insertAdjacentHTML("beforeend",`<article id="${id}" class="metric-panel layout-full layout-expanded"><div class="metric-head"><div><span class="badge">${esc(displayTitle(purpose.title))}</span><h3>${esc(displayTitle(view.title))}</h3></div></div><div class="chart-controls"><label>扰动幅度</label><select data-role="scale" aria-label="选择扰动幅度"></select></div><div class="sensitivity-layout"><section class="sensitivity-pane"><h4>输入通道 × 输出通道的 S<sub>ij</sub></h4><p class="muted">色阶固定为理论范围 0–2；点击单元格可同步选择右侧曲线。</p><div data-role="matrix" class="table sensitivity-table"></div></section><section class="sensitivity-pane sensitivity-curve"><h4>扰动幅度—相对响应曲线</h4><div class="chart-controls"><label>输入通道</label><select data-role="input"></select><label>输出通道</label><select data-role="output"></select></div><div class="chart-viewport"><svg viewBox="0 0 1000 430" role="img" aria-label="扰动幅度与对称相对输出响应"></svg></div></section></div><div class="metric-meta">正在加载多尺度敏感度证据。</div>${guideDetails(guide)}</article>`);
  const panel=document.getElementById(id),scaleSelector=panel.querySelector('[data-role="scale"]'),inputSelector=panel.querySelector('[data-role="input"]'),outputSelector=panel.querySelector('[data-role="output"]'),matrixHost=panel.querySelector('[data-role="matrix"]'),svg=panel.querySelector("svg"),meta=panel.querySelector(".metric-meta");
  try{
    const observations=await observationsForGuide(guides[0]);
    if(token!==renderToken)return;
    const records=sensitivityRecords(observations),scales=uniqueNumbers(records.map(record=>record.scale)).sort((a,b)=>a-b),inputs=unique(records.map(record=>record.input)).sort((a,b)=>a.localeCompare(b,"zh-CN",{numeric:true})),outputs=unique(records.map(record=>record.output)).sort((a,b)=>a.localeCompare(b,"zh-CN",{numeric:true})),magnitude=2,cells=new Map();
    records.forEach(record=>{const key=`${record.scale}\u0000${record.input}\u0000${record.output}`;if(!cells.has(key))cells.set(key,[]);cells.get(key).push(record)});
    scaleSelector.innerHTML=scales.map(value=>`<option value="${value}">${esc(formatTick(value))}</option>`).join("");
    inputSelector.innerHTML=inputs.map((value,index)=>`<option value="${index}">${esc(value)}</option>`).join("");
    outputSelector.innerHTML=outputs.map((value,index)=>`<option value="${index}">${esc(value)}</option>`).join("");
    const drawCurve=()=>{
      const input=inputs[Number(inputSelector.value)]??inputs[0],output=outputs[Number(outputSelector.value)]??outputs[0],curve=records.filter(record=>record.input===input&&record.output===output).map(record=>({value:record.value,x:record.scale,group:`${input} → ${output}`,item:record.item}));
      const result=drawTrendRecords(svg,guide,curve,"scale","linear");
      meta.textContent=`${records.length.toLocaleString()} 个有限组合 · ${scales.length} 个扰动尺度 · ${inputs.length} 个输入通道 · ${outputs.length} 个输出通道 · 当前曲线 ${result.support} 个点`;
    };
    const drawMatrix=()=>{
      const scale=Number(scaleSelector.value),rows=inputs.map((input,inputIndex)=>`<tr><th title="${esc(input)}">${esc(input)}</th>${outputs.map((output,outputIndex)=>{const values=(cells.get(`${scale}\u0000${input}\u0000${output}`)||[]).map(record=>record.value),value=values.length?values.reduce((total,item)=>total+item,0)/values.length:null,selected=inputIndex===Number(inputSelector.value)&&outputIndex===Number(outputSelector.value);return value==null?'<td>—</td>':`<td data-sensitivity-cell data-input-index="${inputIndex}" data-output-index="${outputIndex}" class="${selected?"selected":""}" style="background:${matrixColor(value,magnitude,false)}" title="输入：${esc(input)}&#10;输出：${esc(output)}&#10;尺度：${esc(formatTick(scale))}&#10;Sij：${esc(formatTick(value))}">${esc(formatTick(value))}</td>`}).join("")}</tr>`).join("");
      matrixHost.innerHTML=`<table><thead><tr><th>输入通道</th>${outputs.map(output=>`<th title="${esc(output)}">${esc(output)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
      matrixHost.querySelectorAll("[data-sensitivity-cell]").forEach(cell=>cell.onclick=()=>{inputSelector.value=cell.dataset.inputIndex;outputSelector.value=cell.dataset.outputIndex;drawMatrix();drawCurve()});
    };
    const redrawSelection=()=>{drawMatrix();drawCurve()};
    scaleSelector.onchange=drawMatrix;
    inputSelector.onchange=redrawSelection;
    outputSelector.onchange=redrawSelection;
    drawMatrix();
    drawCurve();
    bindZoom(panel,"扰动幅度与对称相对输出响应");
  }catch(error){
    meta.innerHTML=`<span class="error">${esc(error.message||error)}</span>`;
  }
}
// 每个在线 source 只在 Worker 中投影一次全部 term，避免按 term 重复解压和扫描大事务流。
async function trainingObjectiveTermRecords(guide){
  const records=[];
  await Promise.all(guideSources(guide).map(async source=>{
    const descriptor=BOOT.artifacts.find(item=>item.name===source.name),metricPaths=(source.numeric_paths||[]).filter(objectiveTermPath);
    if(!descriptor||!metricPaths.length)return;
    const projected=await parseEmbeddedRows(descriptor,"training_series",{metricPaths,coordinateLeaf:"update"});
    for(const[path,x,value]of projected)records.push({value:Number(value),x:Number(x),group:leaf(path),item:null});
  }));
  return records.filter(record=>Number.isFinite(record.value)&&Number.isFinite(record.x));
}
async function renderObjectiveTermsPanel(view,guides,host,token,purpose){
  const guide=displayGuide(guides[0]),id=`objective-terms-${Math.random().toString(36).slice(2)}`,title=displayTitle(view.title);
  host.insertAdjacentHTML("beforeend",`<article class="metric-panel layout-full"><div class="metric-head"><div><span class="badge">${esc(displayTitle(purpose.title))}</span><h3>${esc(title)}</h3></div></div><div class="chart-viewport"><svg id="${id}" viewBox="0 0 1000 430" role="img" aria-label="${esc(title)}"></svg></div><div class="metric-meta">正在读取训练 heartbeat 中的 loss components。</div>${guideDetails(guide)}</article>`);
  const panel=document.getElementById(id).closest(".metric-panel");
  try{
    const records=await trainingObjectiveTermRecords(guides[0]);
    if(token!==renderToken)return;
    const svg=document.getElementById(id);
    if(!records.length){
      svg.outerHTML='<div class="empty">训练 artifact 没有可绘制的 objective_terms 数值项。</div>';
      panel.querySelector(".metric-meta").textContent="0 条观测";
      return;
    }
    const result=drawTrendRecords(svg,guide,records,"update","linear",true);
    panel.querySelector(".metric-meta").textContent=`${result.support.toLocaleString()} 条观测 · ${result.xCount} 个训练步 · ${result.groups} 个命名目标项（原名称区分 loss、权重和其它 ledger 项） · ${localizeProse(result.note)}`;
    bindZoom(panel,title);
  }catch(error){
    panel.querySelector(".metric-meta").innerHTML=`<span class="error">${esc(error.message||error)}</span>`;
  }
}
// 组合视图统一占满一行；rollout 详情选择输出与汇总尺度，各 horizon 始终分面展示。
async function renderCompositePanel(view,guides,host,token,purpose){
  if(view.kind==="input_sensitivity")return renderInputSensitivityPanel(view,guides,host,token,purpose);
  if(view.kind==="objective_terms")return renderObjectiveTermsPanel(view,guides,host,token,purpose);
  const localizedGuides=guides.map(displayGuide),id=`composite-${Math.random().toString(36).slice(2)}`,title=displayTitle(view.kind==="trend"&&localizedGuides.length===1?localizedGuides[0].label:view.title),controlId=`${id}-controls`;
  const controls=view.kind==="rollout_overview"?`<div id="${controlId}" class="chart-controls" hidden><label>输出通道</label><select data-role="output" aria-label="选择输出通道"></select><label>汇总误差指标</label><select data-role="summary-metric" aria-label="选择 RMSE 或 sRMSE"><option value="rmse">RMSE</option><option value="srmse">sRMSE</option></select></div>`:"",expanded=["rollout_overview","rollout_heatmap","objective_overview"].includes(view.kind)?"layout-expanded":"";
  host.insertAdjacentHTML("beforeend",`<article class="metric-panel layout-full ${expanded}"><div class="metric-head"><div><span class="badge">${esc(displayTitle(purpose.title))}</span><h3>${esc(title)}</h3></div></div>${controls}<div class="chart-viewport"><svg id="${id}" viewBox="0 0 1000 430" role="img" aria-label="${esc(title)}"></svg></div><div class="metric-meta">正在组合同一诊断目的的证据。</div><details><summary>本图使用的证据</summary><dl>${localizedGuides.map(guide=>`<dt>${esc(guide.label)}</dt><dd>${esc(guide.description)}</dd>`).join("")}</dl></details></article>`);
  const panel=document.getElementById(id).closest(".metric-panel");
  try{
    const entries=await Promise.all(guides.map(async(guide,index)=>({guide:localizedGuides[index],observations:await observationsForGuide(guide)})));
    if(token!==renderToken)return;
    const svg=document.getElementById(id),controlsElement=view.kind==="rollout_overview"?document.getElementById(controlId):null,sources=unique(guides.flatMap(guide=>guideSources(guide).map(source=>source.scope_id||source.name))),rolloutRows=["rollout_heatmap","rollout_coverage"].includes(view.kind)?await rolloutEvidenceRows(entries):[];
    if(token!==renderToken)return;
    const draw=(output,summaryMetric="rmse")=>view.kind==="trend"?drawCompositeTrend(svg,title,entries,view.coordinate,"linear"):view.kind==="parity"?drawParityComposite(svg,title,entries):view.kind==="objective_overview"?drawObjectiveOverview(svg,title,entries):view.kind==="rollout_overview"?drawRolloutOverview(svg,title,entries,output,summaryMetric,"linear"):view.kind==="rollout_heatmap"?drawRolloutHeatmap(svg,title,entries,rolloutRows):view.kind==="rollout_coverage"?drawRolloutCoverage(svg,title,rolloutRows):drawCompositeComparison(svg,title,entries),updateMeta=result=>{panel.querySelector(".metric-meta").textContent=`${result.support.toLocaleString()} 条观测 · ${result.xCount} 个横轴位置 · ${result.groups} 个比较对象 · ${localizeProse(result.note)} · ${sources.length} 个独立数据范围`};
    let result=draw(null,"rmse");
    updateMeta(result);
    if(view.kind==="rollout_overview"&&result.availableGroups?.length){
      const outputSelector=controlsElement.querySelector('[data-role="output"]'),metricSelector=controlsElement.querySelector('[data-role="summary-metric"]');
      outputSelector.innerHTML=result.availableGroups.map(value=>`<option value="${esc(value)}"${value===result.selectedGroup?" selected":""}>${esc(value)}</option>`).join("");
      const redraw=()=>{result=draw(outputSelector.value,metricSelector.value);updateMeta(result)};
      outputSelector.onchange=redraw;
      metricSelector.onchange=redraw;
      controlsElement.hidden=false;
    }
    if(panel.querySelector("svg"))bindZoom(panel,title);
  }catch(error){
    panel.querySelector(".metric-meta").innerHTML=`<span class="error">${esc(error.message||error)}</span>`;
  }
}
// 功能分组必须覆盖完整 guide catalog；未知新指标保留可见 fallback，避免静默漏图。
function purposeGroups(category,guides){const definitions=PURPOSES[category]||[],assigned=new Set(),groups=definitions.map(definition=>{const categories=new Set(definition.categories||[]),metrics=guides.filter(guide=>definition.paths.includes(guide.path)&&(!categories.size||categories.has(guide.category)));metrics.forEach(metric=>assigned.add(metric));return{...definition,metrics}}).filter(group=>group.metrics.length);const unassigned=guides.filter(guide=>!assigned.has(guide));if(unassigned.length&&category!=="component")groups.push({id:"other",title:"其它注册诊断证据",description:"这些指标已注册明确语义，但当前报告尚未为它们声明更细的功能分组；逐图使用指标说明进行审查。",metrics:unassigned});return groups}
function groupPresentation(category,group,categoryGuides=group.metrics){const available=categoryGuides.filter(guideAvailable),specs=COMPOSITE_VIEWS[`${category}.${group.id}`]||[],views=specs.map(spec=>{const guides=spec.paths.map(path=>available.find(guide=>guide.path===path)).filter(Boolean),minimum=spec.kind==="objective_overview"?4:spec.kind==="rollout_heatmap"?4:["rollout_overview","parity","comparison"].includes(spec.kind)?2:1;return{spec,guides,valid:guides.length>=minimum}}).filter(view=>view.valid),combinedPaths=new Set(views.flatMap(view=>view.guides.map(guide=>guide.path))),ownAvailable=group.metrics.filter(guideAvailable),direction=ownAvailable.filter(guide=>DIRECTION_PATHS.has(guide.path)&&!combinedPaths.has(guide.path)&&!RAW_ONLY_PATHS.has(guide.path)),status=ownAvailable.filter(guide=>STATUS_PATHS.has(guide.path)&&!combinedPaths.has(guide.path)),details=ownAvailable.filter(guide=>!combinedPaths.has(guide.path)&&!direction.includes(guide)&&!status.includes(guide));return{views,direction,status,details,mainCount:views.length+direction.length}}
function retentionReason(guide){if(RAW_ONLY_PATHS.has(guide.path))return"复核抽样估计、干预范围或未归一化量";if(guide.role==="control")return"判断主要结果是否超过对照或满足解释条件";if(guide.role==="context")return"说明覆盖率、有效观测或实验条件";return"补充主要图表的尺度、尾部或派生口径"}
function detailInventory(guides,hostId){const body=guides.map(guide=>{const localized=displayGuide(guide),available=guideAvailable(guide),index=GUIDES.indexOf(guide);return`<tr><td><strong>${esc(localized.label)}</strong></td><td>${esc(retentionReason(guide))}</td><td>${available?"可绘图":"本次未采集"}</td><td>${available?`<button type="button" data-detail-guide="${index}" data-detail-host="${esc(hostId)}">查看图表</button>`:"—"}</td></tr>`}).join("");return`<div class="table"><table><thead><tr><th>诊断量</th><th>为什么保留</th><th>本次状态</th><th>操作</th></tr></thead><tbody>${body}</tbody></table></div><div id="${esc(hostId)}" class="detail-chart-host chart-grid"></div>`}
function bindDetailButtons(token){document.querySelectorAll("#purpose-sections [data-detail-guide]").forEach(button=>{button.onclick=async()=>{const guide=GUIDES[Number(button.dataset.detailGuide)],host=document.getElementById(button.dataset.detailHost);if(!guide||!host)return;host.innerHTML="";button.textContent="已打开";await renderMetricPanel(guide,host,token,{title:"补充证据"},"detail")}})}
async function renderPurposeGroup(sectionId,group,presentation,host,token){const primaryId=`purpose-${sectionId}-${group.id}-${token}-primary`,detailId=`purpose-${sectionId}-${group.id}-${token}-detail`;host.insertAdjacentHTML("beforeend",`<section class="purpose-section"><h3>${esc(displayTitle(group.title))}</h3><p class="muted purpose-description">${esc(localizeProse(group.description))}</p><div id="${primaryId}" class="chart-grid"></div>${presentation.details.length?`<details class="secondary support-details"><summary>查看补充证据</summary><p class="muted support-note">这些诊断量用于核对对照、覆盖率、派生口径或实验条件；有数据时可直接打开图表。</p>${detailInventory(presentation.details,detailId)}</details>`:""}</section>`);const primaryHost=document.getElementById(primaryId);for(const view of presentation.views)await renderCompositePanel(view.spec,view.guides,primaryHost,token,group);await Promise.all(presentation.direction.map(guide=>renderMetricPanel(guide,primaryHost,token,group)));await Promise.all(presentation.status.map(guide=>renderMetricPanel(guide,primaryHost,token,group,"status")));bindDetailButtons(token)}
// 主页面不再为只有补充字段的功能目的创建空容器；这些字段统一进入一个可按需作图的折叠区。
async function renderSection(sectionId){renderToken+=1;const token=renderToken;document.querySelectorAll("#section-nav button").forEach(button=>button.setAttribute("aria-selected",String(button.dataset.section===sectionId)));const diagnostic=document.getElementById("diagnostic-workspace"),evidence=document.getElementById("evidence-workspace");if(sectionId==="evidence"){diagnostic.hidden=true;evidence.hidden=false;return}diagnostic.hidden=false;evidence.hidden=true;const section=SECTIONS.find(item=>item.id===sectionId),state=categoryState(sectionId),guides=guidesForSection(sectionId).sort((a,b)=>displayGuide(a).label.localeCompare(displayGuide(b).label)),groups=purposeGroups(sectionId,guides),presentations=groups.map(group=>groupPresentation(sectionId,group,guides)),renderable=groups.map((group,index)=>({group,presentation:presentations[index]})).filter(item=>item.presentation.mainCount+item.presentation.status.length>0),supplementary=groups.map((group,index)=>({group,presentation:presentations[index]})).filter(item=>item.presentation.mainCount+item.presentation.status.length===0&&item.presentation.details.length);document.getElementById("section-title").textContent=section.label;document.getElementById("section-question").textContent=section.question;document.getElementById("section-support").textContent=state.kind==="available"?"":state.label;const host=document.getElementById("purpose-sections");host.innerHTML="";const status=document.getElementById("section-status");if(state.kind==="available")status.textContent=sectionId==="final"?"最终分析按功能分页；只加载当前页证据，切换分页不会丢失其它结果。":"主要图表用于查看排查方向；其它已采集诊断量可在下方“补充证据”中按需绘图。";else if(state.kind==="not_run")status.textContent=`本次运行未执行这个栏目对应的分析器；所需证据类型：${state.kinds.join(", ")}。已执行分析器：${EXECUTED_ANALYZERS.join(", ")||"无"}。`;else status.textContent="对应分析器已有记录，但没有通过校验且可绘图的有限证据。";
  if(sectionId==="final"&&(renderable.length||supplementary.length)){const tabItems=[...renderable,...(supplementary.length?[{supplementary:true}]:[])];host.innerHTML=`<nav class="purpose-tabs" aria-label="最终模型分析功能"></nav><div class="purpose-tab-host"></div>`;const tabs=host.querySelector(".purpose-tabs"),tabHost=host.querySelector(".purpose-tab-host");tabs.innerHTML=tabItems.map((item,index)=>`<button type="button" data-purpose-index="${index}" aria-selected="${index===0}">${item.supplementary?"补充证据":esc(displayTitle(item.group.title))}</button>`).join("");const show=async index=>{renderToken+=1;const purposeToken=renderToken;tabs.querySelectorAll("button").forEach((button,buttonIndex)=>button.setAttribute("aria-selected",String(buttonIndex===index)));tabHost.innerHTML='<div class="empty">正在加载当前功能所需证据。</div>';const item=tabItems[index];tabHost.innerHTML="";if(!item.supplementary){await renderPurposeGroup(sectionId,item.group,item.presentation,tabHost,purposeToken);return}const groupsHtml=supplementary.map(({group,presentation},groupIndex)=>{const detailId=`purpose-${sectionId}-${group.id}-${purposeToken}-supplementary-${groupIndex}`;return`<section class="supplementary-group"><h4>${esc(displayTitle(group.title))}</h4><p class="muted purpose-description">${esc(localizeProse(group.description))}</p>${detailInventory(presentation.details,detailId)}</section>`}).join("");tabHost.innerHTML=`<p class="muted support-note">这些诊断量用于核对实验条件、覆盖率和派生口径；仅在打开某项时读取对应证据。</p>${groupsHtml}`;bindDetailButtons(purposeToken)};tabs.querySelectorAll("button").forEach((button,index)=>button.onclick=()=>show(index));await show(0);return}
  for(const item of renderable)await renderPurposeGroup(sectionId,item.group,item.presentation,host,token);
  if(supplementary.length){const groupsHtml=supplementary.map(({group,presentation},index)=>{const detailId=`purpose-${sectionId}-${group.id}-${token}-supplementary-${index}`;return`<section class="supplementary-group"><h4>${esc(displayTitle(group.title))}</h4><p class="muted purpose-description">${esc(localizeProse(group.description))}</p>${detailInventory(presentation.details,detailId)}</section>`}).join("");host.insertAdjacentHTML("beforeend",`<details class="secondary support-details"><summary>补充证据</summary><p class="muted support-note">以下诊断量不会单独占据主页面，但仍用于核对实验是否有效、解释主要图表或复现比较；每一项有数据时都可按需绘图。</p>${groupsHtml}</details>`);bindDetailButtons(token)}
}
function options(element,values){element.innerHTML=values.map(([value,label])=>`<option value="${esc(value)}">${esc(label)}</option>`).join("")}
function fieldClass(path,values){const name=path.toLowerCase(),sample=values.find(value=>value!==null&&value!==undefined);if(/sha256|digest|transaction_id|session_id|format_version|schema|commit/.test(name))return["后台校验与追溯","不作为模型诊断图","用于完整性、去重、恢复或来源追溯"];if(finite(sample)||Array.isArray(sample)&&sample.some(finite))return["数值字段","功能图或完整字段查看器","未注册路径不会自动获得诊断语义"];if(/(^|\.)(status|kind|reason|identity|.*_id|.*_path|rank|partition|condition|scenario|target|group)(_|$|\.)/.test(name))return["筛选与分组上下文","比较轴、悬停信息或类别频数图","防止不同身份数据被错误混合"];return["详情与协议字段","类别频数图、搜索和原始详情","记录实验协议或非数值来源信息"]}
let rawRows=[],flatRows=[],page=0;
function renderFieldAudit(){const paths=[...new Set(flatRows.flatMap(row=>Object.keys(row)))].sort(),rows=paths.map(path=>{const [classification,visualization,reason]=fieldClass(path,flatRows.map(row=>row[path]));return{path,classification,visualization,reason}});document.getElementById("field-audit").innerHTML=table(rows,["path","classification","visualization","reason"])}
function updateExplorerFields(){const source=STREAMS.find(item=>item.name===document.getElementById("explorer-source").value),fields=source?[...(source.numeric_paths||[]).map(path=>[`N|${path}`,`数值或序列 · ${path}`]),...(source.categorical_paths||[]).map(path=>[`C|${path}`,`类别 · ${path}`])]:[];options(document.getElementById("explorer-field"),fields.length?fields:[["","没有可视化字段"]])}
// Generic explorer 只保证字段可见；点击图表复用同一 modal，不复制功能图的诊断语义。
async function renderExplorerField(){const source=STREAMS.find(item=>item.name===document.getElementById("explorer-source").value),encoded=document.getElementById("explorer-field").value,host=document.getElementById("field-explorer");if(!source||!encoded){host.innerHTML='<div class="empty">选择来源和字段。</div>';return}const kind=encoded.slice(0,1),path=encoded.slice(2),rows=await load(source.name,"flat"),id=`field-${Math.random().toString(36).slice(2)}`;host.innerHTML=`<div class="metric-head"><div><span class="badge">${kind==="N"?"数值或序列字段":"类别字段"}</span><h3>${esc(path)}</h3></div></div><svg id="${id}" viewBox="0 0 1000 430" role="img" aria-label="${esc(path)}"></svg><div class="metric-meta"></div><p class="muted">这是原始字段图；未注册为诊断量的字段不提供好坏、因果或诊断语义。点击图表可放大、缩放和拖动。</p>`;const svg=document.getElementById(id),meta=host.querySelector(".metric-meta");if(kind==="N"){const observations=[];rows.forEach((row,rowIndex)=>{const value=row[path];if(Array.isArray(value))value.forEach((item,seriesIndex)=>{if(finite(item))observations.push({value:item,row,rowIndex,seriesIndex,source,path})});else if(finite(value))observations.push({value,row,rowIndex,seriesIndex:null,source,path})});const result=drawMetric(svg,{path,label:path,category:"explorer"},observations);meta.textContent=`${result.support} 条观测 · ${result.xCount} 个横轴位置 · ${result.groups} 个比较对象 · ${localizeProse(result.note)}`}else{const result=drawCategorical(svg,path,rows.map(row=>row[path]));meta.textContent=`${result.support} 条观测 · ${result.categories} 个类别`}const rendered=host.querySelector("svg");if(rendered)bindZoom(host,path)}
function renderVisualizationCoverage(){const separator="::numeric-path::",numeric=[...new Set(STREAMS.flatMap(source=>(source.numeric_paths||[]).map(path=>`${source.name}${separator}${path}`)))],categorical=[...new Set(STREAMS.flatMap(source=>(source.categorical_paths||[]).map(path=>`${source.name}${separator}${path}`)))],matched=numeric.filter(item=>{const [name,path]=item.split(separator),source=STREAMS.find(candidate=>candidate.name===name);return GUIDES.some(guide=>guideCoversPath(guide,path)&&sourceSupportsGuide(source,guide))}),rawOnly=numeric.filter(item=>!matched.includes(item)).map(item=>item.split(separator)[1]),availableGuides=GUIDES.filter(guideAvailable),mainGuides=new Set();for(const section of SECTIONS.filter(item=>item.id!=="evidence")){const guides=guidesForSection(section.id);for(const group of purposeGroups(section.id,guides)){const presentation=groupPresentation(section.id,group,guides);presentation.views.flatMap(view=>view.guides).forEach(guide=>mainGuides.add(guide));presentation.direction.forEach(guide=>mainGuides.add(guide));presentation.status.forEach(guide=>mainGuides.add(guide))}}const onDemand=availableGuides.filter(guide=>!mainGuides.has(guide)).length,rows=[{item:"本次已采集的注册诊断量",count:availableGuides.length,meaning:"全部都有主要图表或按需图表入口"},{item:"主页面直接使用的注册诊断量",count:mainGuides.size,meaning:"用于回答当前排查方向或检查结果是否可解释"},{item:"补充证据中的按需诊断量",count:onDemand,meaning:"默认不占页面空间，需要时可直接打开语义图"},{item:"本次未采集的注册诊断量",count:GUIDES.length-availableGuides.length,meaning:"没有数值，不能生成图表"},{item:"未注册但可绘图的数值或序列路径",count:rawOnly.length,meaning:"可检查原始观测，但不会自动解释"},{item:"可查看频数的类别字段路径",count:categorical.length,meaning:"用于筛选、分组或检查状态分布"}];document.getElementById("visualization-coverage").innerHTML=`${table(rows,["item","count","meaning"])}<details><summary>查看没有注册诊断语义的数值路径</summary><p class="muted">它们仍可在完整字段查看器中绘图；其中包含有效观测数、元素数、采样间隔、格式标识和事务计数，不能仅因其是数值就当成模型诊断量。</p><pre>${esc(rawOnly.sort().join("\n")||"无")}</pre></details>`}
function renderRaw(){const query=document.getElementById("search").value.toLowerCase(),matching=query?rawRows.filter(row=>JSON.stringify(row).toLowerCase().includes(query)):rawRows,size=Number(document.getElementById("page-size").value),pages=Math.max(1,Math.ceil(matching.length/size));page=Math.min(page,pages-1);const rows=matching.slice(page*size,(page+1)*size),columns=[...new Set(rows.flatMap(row=>Object.keys(row)))],host=document.getElementById("data");document.getElementById("stats").textContent=`${matching.length.toLocaleString()} / ${rawRows.length.toLocaleString()} 条记录 · 第 ${page+1}/${pages} 页`;host.innerHTML=table(rows,columns);host.querySelectorAll("tbody tr").forEach((element,index)=>element.onclick=()=>document.getElementById("detail").textContent=JSON.stringify(rows[index],null,2))}
async function renderRawSource(){const source=document.getElementById("source");if(!source.value)return;rawRows=await load(source.value,"raw");flatRows=rawRows.map(row=>flatten(row));page=0;renderRaw();renderFieldAudit()}
document.getElementById("delivery").textContent=DELIVERY_LABELS[BOOT.delivery]||BOOT.delivery;
document.getElementById("section-nav").innerHTML=SECTIONS.map(section=>{const state=categoryState(section.id),stateSuffix=section.id==="evidence"||state.kind==="available"?"":`（${state.label}）`;return`<button data-section="${section.id}" aria-selected="false">${esc(section.label+stateSuffix)}</button>`}).join("");document.querySelectorAll("#section-nav button").forEach(button=>button.onclick=()=>renderSection(button.dataset.section));
renderVisualizationCoverage();
document.getElementById("field-retention-policy").innerHTML=table([{kind:"数值或序列测量",visualization:"功能图或原始观测图",retain:"保留唯一测量事实；重复派生值应删除"},{kind:"状态或低基数类别",visualization:"频数图",retain:"保留，用于区分成功、跳过、降级和对照条件"},{kind:"样本、模块和条件身份",visualization:"作为横轴、分组和悬停信息",retain:"必须保留，否则无法配对、过滤或复现比较"},{kind:"摘要、提交与格式标识",visualization:"不作为模型图表",retain:"必须保留，用于完整性、去重、恢复和快速失败"},{kind:"路径、错误与来源文本",visualization:"搜索、表格和详情",retain:"用于解释证据缺失或失败；重复描述可删除"}],["kind","visualization","retain"]);
options(document.getElementById("explorer-source"),STREAMS.length?STREAMS.map(item=>[item.name,`${localizeProse(item.group)} · ${localizeProse(item.label)}`]):[["","没有证据流"]]);updateExplorerFields();document.getElementById("explorer-source").onchange=updateExplorerFields;document.getElementById("render-field").onclick=renderExplorerField;
document.getElementById("inventory").innerHTML=table(BOOT.artifacts,["group","content_kind","label","format","row_count","source_bytes","available"]);
document.getElementById("hierarchy").innerHTML=BOOT.hierarchy_index.map(node=>`<div style="--depth:${({All:0,Model:1,Stage:2,Block:3}[node.hierarchy_level]??0)}"><strong>${esc(({All:"全部",Model:"模型",Stage:"阶段",Block:"模块块"})[node.hierarchy_level]||node.hierarchy_level)}</strong> · ${esc(node.module_path||node.model_name||node.node_id)}</div>`).join("")||'<span class="muted">不可用</span>';
document.getElementById("contracts").innerHTML=(BOOT.evidence_figures||[]).map(figure=>`<article class="contract"><span class="badge">${esc(figure.evidence_kind)}</span><h3>${esc(displayTitle(figure.title))}</h3><div class="muted">${Number(figure.row_count||0).toLocaleString()} 条记录 · ${esc(localizeProse(figure.status))} · 空效应对照 ${esc(localizeProse(figure.null_control_status))}</div>${(figure.claim_boundaries||[]).length?`<ul class="boundary">${figure.claim_boundaries.map(value=>`<li>${esc(localizeProse(value))}</li>`).join("")}</ul>`:""}</article>`).join("")||'<span class="muted">没有通过校验的分析器证据。</span>';
options(document.getElementById("source"),[["","选择诊断产物"],...AVAILABLE.map(item=>[item.name,`${localizeProse(item.group)} · ${localizeProse(item.label)} · ${Number(item.row_count).toLocaleString()} 条记录`])]);document.getElementById("load-source").onclick=renderRawSource;let searchTimer=0;document.getElementById("search").oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(renderRaw,180)};document.getElementById("page-size").onchange=()=>{page=0;renderRaw()};document.getElementById("previous").onclick=()=>{page=Math.max(0,page-1);renderRaw()};document.getElementById("next").onclick=()=>{page+=1;renderRaw()};
const firstSection=SECTIONS.find(section=>section.id!=="evidence"&&guidesForSection(section.id).some(guideAvailable))?.id||"evidence";renderSection(firstSection);
</script></body></html>"""


def render_dashboard_document(
    *,
    report_format_version: int,
    bootstrap_json: str,
    payload_elements: str,
) -> str:
    """注入启动信息和独立压缩 member，避免启动时解析全部 Base64。"""

    return (
        _HTML_TEMPLATE.replace(
            "__REPORT_FORMAT_VERSION__",
            str(report_format_version),
        )
        .replace("__BOOTSTRAP_JSON__", bootstrap_json)
        .replace("__PAYLOAD_ELEMENTS__", payload_elements)
    )


__all__ = ["render_dashboard_document"]
