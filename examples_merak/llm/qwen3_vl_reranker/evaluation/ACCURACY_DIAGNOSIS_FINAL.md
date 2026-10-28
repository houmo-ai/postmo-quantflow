# 精度损失诊断与配置选择

更新：2026-10-28。本文件合并早期视觉对照、六精度组件排查和后续语言激活筛选的结论。
正式任务成绩以[五模型 head A8/A16 报告](HEAD_A8_A16_713_REPORT.md)和
[三套局部 A16 报告](ACTIVATION_A16_713_COMPARISON_20261028.md)为准；这里不重复整套成绩表。
运行批次、完成或停止状态见[历史记录](HISTORY.md)。

## 当前结论

已取得直接证据的主要损失来源是**评分头 A8 激活，以及语言 Linear 的 A8 激活**。
视觉和 Attention 量化也会改变候选间隔，并与其他组件的数值误差交互；
不能把全部损失归为 W4/W8 权重量化，也不能仅凭某个组件的平均误差预测检索得分。

当前配置保留 head W8A16，并提供全语言 W4A16 和两节点 A16 两种选择。
两节点方案只将 `model.language_model.layers.26.mlp.up_proj`、
`model.language_model.layers.27.mlp.gate_proj` 激活升至 A16，语言权重仍为 W4G64，
其余 194 个语言 Linear 仍为 A8。视觉 W8A8 和 FlashAttention Q/K/V/P 8 bit 保持不变。
这项选择来自完整 713 查询的最终检索指标，而不是只按诊断间隔误差排序。

W16A16＋head W8A16 是本轮的高精度 HMONNX 参照，三个模态与 BF16 接近，
支持已覆盖数据上的适配与评测链路基本一致；它仍使用 W8 视觉、8-bit FlashAttention 和 SDK 数值格式，
不等于全组件原生 BF16。当前证据将主要差异指向量化配置，尚未逐算子归因全部残余误差。

## 已核验的输入与实现

早期完整性审计覆盖六轮 4,278 查询、400,854 候选评分：冻结候选、标签、顺序、协议和
分片到最终分数的对应关系均通过；另外五轮相对 W4A8 的 334,045 个有效输入 token SHA 一致。
三个局部 A16 最终方案也分别完成 713 查询、66,809 个候选对，输入审计缺失和不一致均为 0。

- 六轮原 713 集的最长输入为 8,080 tokens，未触发 32,768 截断；因此保护式截断不是这批数据的掉点原因。
  该结论不替代独立的超长输入回归。
- 视频重放相同的源帧、顺序、时间戳和补帧规则；视觉 A8/A16 使用独立缓存 namespace，未发现跨精度误用。
- 已核对末有效 token、M-RoPE、yes-minus-no 评分及稳定排序；这批结果没有 Top-1 同分或 0/1 概率饱和。
- BF16 仍保持官方视觉输入预算；HMONNX 多档位视觉输入的损失计入比较，不把不同预算说成全量像素一致。

这些检查没有发现能解释当前主要掉点的候选错配、缓存串用或模板处理错误；
它们不构成所有输入与运行路径都绝无 bug 的证明。

## 组件隔离证据

### 评分头：保留 W8 权重，将激活升为 A16

在九个特意选择的 W16A16 丢失案例上，固定原模型隐藏态，比较正负候选 logit 间隔：

| 隔离的误差 | 相对间隔平均绝对差 |
| --- | ---: |
| W8A8 HMONNX 对反量化 W8 的 FP32 投影 | 0.187865 |
| 反量化 W8 对源权重的 FP32 投影 | 0.002434 |
| W8A16 HMONNX 对反量化 W8 的 FP32 投影 | 0.000649 |
| FP16 yes-minus-no 融合对 FP32 差分投影 | 0.000016 |

仅提高激活位宽就显著减少头部误差，支持 head A8 激活是这批错误的主要来源，
而不是评分头 W8 权重或一次 FP16 差分融合舍入。头部 A16 修复其中 8/9 个指定候选对；
这是定向机制定位，不是完整 Top-100 命中率。随后五模型 head A8/A16 的完整配对评测验证了配置收益。
进一步升 head 权重至 W16 的批次只完成 W4A8、W8A8 两轮，未显示同等幅度收益；不将中止的三轮算作完成。

### 语言激活与 Attention：同权重比较限制了结构错误的解释

在早期相同 W8 语言权重、源 FP16 视觉和源评分投影下，改变语言激活与 FlashAttention 精度：

| 语言激活 / FlashAttention Q/K/V/P | 对同权重原生 FP16 的概率平均绝对差 |
| --- | ---: |
| A8 / 8 bit | 0.05263 |
| A16 / 8 bit | 0.01667 |
| A8 / 16 bit | 0.03382 |
| A16 / 16 bit | 0.00139 |

后续固定 head W8A16 的 W4A8/W4A16 对照中，语言图 520 个 initializer payload 完全一致，
变化仅在语言 Linear 激活位宽。该单变量证据比直接比较不同权重的 W4/W8 总分更能支持激活敏感性的判断。

另一次同权重参考将 196 个导出 W16 语言权重放回官方图；HMONNX 换源视觉、Flash16 和源 FP32 头后，
36 个候选的概率平均绝对差为 0.003017、最大 0.008793。主要排序错误随低精度组件替换而消失，
不支持将它们主要归为候选拼接或视频位置计算错误；其余数值格式和累加等差异仍未完全拆分。

### 视觉：误差存在，但提高视觉精度不保证检索指标单调改善

对相同 patch、grid 和档位的 93 个视觉输入，最终视觉特征相对 RMSE 从 W8A8 的 15.48%
降至 W8A16 的 7.04%，平均 cosine 从 0.98693 升至 0.99737。原有 102 对候选的有效输入、
Merak/直接运行时评分与五档视觉 Golden 均通过一致性检查。

视觉 A16 恢复了 23 个选定失败候选对中的 10 个，源 FP16 视觉恢复 16 个，说明视觉量化会影响排序。
但同一混合视觉版本在 CIRR 的正式 Hit@1 从 63 降至 56，而全候选概率 MAE 从 0.043335 降至 0.042651。
固定视觉 A16 后再提高语言激活，能修复九个新增失败对中的八个，支持多组件误差交互，
不能据此认定 A16 视觉实现错误或 A8 视觉更准确。组件恢复数不能相加解释整套损失比例。

## 三套局部 A16 的筛选与适用范围

定位使用原 713 条中的 135 查询、每查询六个候选；整层扫描 28 层后，对敏感的 L5/L26/L27
再比较 21 个 Linear。最终正式验证使用三套方案各自完整的原冻结候选集，不能以六候选定位分数代替。
层号从 0 开始；“整层 A16”只指该层七个 Linear，不包含自动提高 FlashAttention 位宽。

间隔误差以全语言 W4A16 参照计算：每查询取最大相关 logit 减最大不相关 logit，
对有效查询平均其与参照间隔的绝对差。`原 W4A8 误差 − 当前误差` 越大，只表示更接近该参照，
不代表检索指标一定更高；W4A16 参照自身也可能排序错误。最终方案必须依据完整候选的任务指标。

两节点方案在原 713 条的九任务等权宏平均最高，主要收益来自视频；
两整层和三整层方案的图像、文档成绩更高。未参与本轮定位/选择的 443 条上，
两节点相对原 W4A8 的图像为 −0.48pp，文档 +1.92pp、视频 +5.00pp；
图像差异相当于该部分 210 条图像查询少命中一条，不能据此宣布普遍退化。
历史基线已经存在，因此这 443 条也不是全新的外部盲测。

小样本会放大有限查询翻转带来的指标波动，但不会自动消除系统量化偏差；
3414 条扩样中的 BF16/W8A8/W4A8 已完整比较，不能把当前损失只归为样本数不足。
激活异常值、SEFP 共享指数范围或具体层内累加误差仍是待验证机制，现有位宽实验没有逐张量证明这些原因。

## 证据与完整历史

| 内容 | 原始证据 |
| --- | --- |
| 六轮输入、排序与实际图属性审计 | [integrity_audit.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/diagnosis_final_precision/integrity_audit.json) |
| 同隐藏态头部权重/激活隔离 | [head_component_summary.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/diagnosis_final_precision/head_component_summary.json) |
| 跨组件与同权重原生参考 | [component_summary.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/diagnosis_final_precision/component_summary.json)、[native_reference.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/diagnosis_final_precision/native_reference.json) |
| 早期视觉/语言对照 | [report.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/diagnosis_w8a16/report.json) |
| 激活扫描、组合选择与正式验证 | [analysis/report.json](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/hmonnx_eval/work_dirs/hmonnx_eval/activation_sensitivity_20261027/analysis/report.json) |
| 本次整理前的两份完整诊断文档 | [2026-10-23 诊断](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/ACCURACY_DIAGNOSIS_W8A16.md>)、[2026-10-24 诊断](</data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_cleanup_20261028/20261028T123756+0800/before/ACCURACY_DIAGNOSIS_FINAL.md>) |
