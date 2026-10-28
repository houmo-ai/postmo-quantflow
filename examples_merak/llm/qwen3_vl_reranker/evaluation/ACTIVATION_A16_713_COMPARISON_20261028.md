# Qwen3-VL-Reranker-2B

> 原 713 条固定子集的正式报告；统一入口见[评测说明](README.md)，原始证据保存在长期归档。

## 语言激活 A16 筛选方案与 713 条评测总览

记录日期：2026-10-28（北京时间）；正式评测完成时间：2026-10-28 02:32。

## 1. 评测口径与结果总览

本报告记录本轮三套局部 A16 方案，以及 BF16、全语言 W4A16 参照的原 713 条查询成绩。正式验证使用相同的冻结候选集，共 66,809 个候选对；三套方案均完成 713/713，输入审计均为 66,809/66,809，缺失和不一致均为 0。本轮未运行 3414 条扩展集。

9 个任务组成：图像 MSCOCO_i2t、MSCOCO_t2i、CIRR，各 100 条；文档 ViDoRe_docvqa、ViDoRe_tatdqa、VisRAG_ChartQA，分别 100、100、63 条；视频 MSR-VTT、DiDeMo、MomentSeeker，各 50 条。

图像、视频使用 Hit@1，文档使用线性 nDCG@5。各模态内 3 个任务等权平均；九任务宏平均为 9 个任务等权平均，也等于三个模态宏平均的等权平均。表内均为百分制，越大越好；不是按查询数量加权平均。

| 方案 | 语言 Linear<br>A16 数 | 图像<br>宏平均 | 文档<br>宏平均 | 视频<br>宏平均 | 九任务<br>宏平均 |
| --- | --- | --- | --- | --- | --- |
| BF16 官方浮点 | — | 75.6667 | 80.1857 | 56.0000 | 70.6175 |
| 全语言 W4A16 / head W8A16 | 196 | 74.3333 | 80.2768 | 56.0000 | 70.2034 |
| 两整层 A16（L26、L27） | 14 | 75.0000 | 79.2600 | 53.3333 | 69.1978 |
| 三整层 A16（L5、L26、L27） | 21 | 74.6667 | 79.5525 | 54.0000 | 69.4064 |
| 两节点 A16（L26 up、L27 gate） | 2 | 73.6667 | 79.1140 | 56.6667 | 69.8158 |

三套局部 A16 方案中，两节点方案的九任务宏平均最高（69.8158），主要优势来自视频；两套整层方案的图像、文档成绩更高。上述结论适用于本次 713 条评测，不能据此认定所有模态或其他数据上都由同一方案最优。

## 2. 方案范围与固定配置

| 方案 ID | 涉及层（代码编号） | A16 Linear | 其余 A8 Linear |
| --- | --- | --- | --- |
| combo_layers2 | L26、L27 | 14 | 182 |
| combo_layers3 | L5、L26、L27 | 21 | 175 |
| combo_nodes02 | L26 up、L27 gate | 2 | 194 |

层号采用代码中从 0 开始的编号：L5、L26、L27 分别是自然顺序中的第 6、27、28 个 Transformer block。语言主干共 28 层，每层 7 个 Linear，共 196 个；计数不含 lm_head。文中的“整层 A16”严格指该层 7 个 Linear 的激活为 A16，不代表该 block 全部算子均变为 A16。

三套方案固定项：语言 Linear 权重 W4、group size 64，复用同一份量化权重；评分头 lm_head 为 W8A16；视觉编码器 W8A8，五档 [96, 196, 384, 704, 1536]；FlashAttention 开启，Q/K/V/P 均为 8 bit；上下文 32768，prefill chunk 256。全语言 W4A16 参照指全部 196 个语言 Linear 激活 A16，评分头仍为 W8A16。

<!-- Word 文档在此分页 -->

## 3. 两整层方案：L26、L27 的全部 7 个 Linear

将 L26 和 L27 各自的 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj 激活设为 A16；不改变其他层的 Linear 激活精度。

方案 ID：`combo_layers2`；语言 Linear A16 共 14 个，其余 182 个保持 A8。评分头固定 W8A16，不计入本表。

| 序号 | 层号 | 设为 A16 的模块全名 |
| --- | --- | --- |
| 1 | L26 | model.language_model.layers.26.self_attn.q_proj |
| 2 | L26 | model.language_model.layers.26.self_attn.k_proj |
| 3 | L26 | model.language_model.layers.26.self_attn.v_proj |
| 4 | L26 | model.language_model.layers.26.self_attn.o_proj |
| 5 | L26 | model.language_model.layers.26.mlp.gate_proj |
| 6 | L26 | model.language_model.layers.26.mlp.up_proj |
| 7 | L26 | model.language_model.layers.26.mlp.down_proj |
| 8 | L27 | model.language_model.layers.27.self_attn.q_proj |
| 9 | L27 | model.language_model.layers.27.self_attn.k_proj |
| 10 | L27 | model.language_model.layers.27.self_attn.v_proj |
| 11 | L27 | model.language_model.layers.27.self_attn.o_proj |
| 12 | L27 | model.language_model.layers.27.mlp.gate_proj |
| 13 | L27 | model.language_model.layers.27.mlp.up_proj |
| 14 | L27 | model.language_model.layers.27.mlp.down_proj |

导出配置使用 FX 节点键，即上表模块全名中的点全部替换为下划线，并为每个准确节点设置 `act_scheme.bits: 16`；没有按父层名称递归覆盖。

完整配置：`export/configs/combo_layers2.yaml`（路径相对于本次排查目录 OUT，见第 6 节）。

### 配置键示例

模块 `model.language_model.layers.26.self_attn.q_proj` 对应配置键 `model_language_model_layers_26_self_attn_q_proj`。下列片段展示单个节点；该方案完整 YAML 中逐项列出了本页全部 14 个节点。

```yaml
model_language_model_layers_26_self_attn_q_proj:
  act_scheme:
    bits: 16
```

<!-- Word 文档在此分页 -->

## 4. 三整层方案：L5、L26、L27 的全部 7 个 Linear

在两整层方案基础上，再将 L5 的 7 个 Linear 激活设为 A16，共 21 个。所有语言 Linear 的权重仍保持 W4。

方案 ID：`combo_layers3`；语言 Linear A16 共 21 个，其余 175 个保持 A8。评分头固定 W8A16，不计入本表。

| 序号 | 层号 | 设为 A16 的模块全名 |
| --- | --- | --- |
| 1 | L5 | model.language_model.layers.5.self_attn.q_proj |
| 2 | L5 | model.language_model.layers.5.self_attn.k_proj |
| 3 | L5 | model.language_model.layers.5.self_attn.v_proj |
| 4 | L5 | model.language_model.layers.5.self_attn.o_proj |
| 5 | L5 | model.language_model.layers.5.mlp.gate_proj |
| 6 | L5 | model.language_model.layers.5.mlp.up_proj |
| 7 | L5 | model.language_model.layers.5.mlp.down_proj |
| 8 | L26 | model.language_model.layers.26.self_attn.q_proj |
| 9 | L26 | model.language_model.layers.26.self_attn.k_proj |
| 10 | L26 | model.language_model.layers.26.self_attn.v_proj |
| 11 | L26 | model.language_model.layers.26.self_attn.o_proj |
| 12 | L26 | model.language_model.layers.26.mlp.gate_proj |
| 13 | L26 | model.language_model.layers.26.mlp.up_proj |
| 14 | L26 | model.language_model.layers.26.mlp.down_proj |
| 15 | L27 | model.language_model.layers.27.self_attn.q_proj |
| 16 | L27 | model.language_model.layers.27.self_attn.k_proj |
| 17 | L27 | model.language_model.layers.27.self_attn.v_proj |
| 18 | L27 | model.language_model.layers.27.self_attn.o_proj |
| 19 | L27 | model.language_model.layers.27.mlp.gate_proj |
| 20 | L27 | model.language_model.layers.27.mlp.up_proj |
| 21 | L27 | model.language_model.layers.27.mlp.down_proj |

导出配置使用 FX 节点键，即上表模块全名中的点全部替换为下划线，并为每个准确节点设置 `act_scheme.bits: 16`；没有按父层名称递归覆盖。

完整配置：`export/configs/combo_layers3.yaml`（路径相对于本次排查目录 OUT，见第 6 节）。

<!-- Word 文档在此分页 -->

## 5. 两节点方案：L26 up_proj、L27 gate_proj

仅将 L26 的 mlp.up_proj 和 L27 的 mlp.gate_proj 激活设为 A16；同层中的其余 Linear 以及其他层的 Linear 均保持 A8。

方案 ID：`combo_nodes02`；语言 Linear A16 共 2 个，其余 194 个保持 A8。评分头固定 W8A16，不计入本表。

| 序号 | 层号 | 设为 A16 的模块全名 |
| --- | --- | --- |
| 1 | L26 | model.language_model.layers.26.mlp.up_proj |
| 2 | L27 | model.language_model.layers.27.mlp.gate_proj |

导出配置使用 FX 节点键，即上表模块全名中的点全部替换为下划线，并为每个准确节点设置 `act_scheme.bits: 16`；没有按父层名称递归覆盖。

完整配置：`export/configs/combo_nodes02.yaml`（路径相对于本次排查目录 OUT，见第 6 节）。

### 对应节点覆盖配置

以下是准确节点覆盖片段，其他字段沿用完整导出配置与原有 W4 量化权重。评分头条目作为固定项一并列出。

```yaml
export:
  model:
    quant_scheme:
      nodes:
        lm_head:
          quant_type: w8a16h1_sefp
        model_language_model_layers_26_mlp_up_proj:
          act_scheme:
            bits: 16
        model_language_model_layers_27_mlp_gate_proj:
          act_scheme:
            bits: 16
```

## 6. 数据来源与追溯

OUT 为本次排查的归档目录，路径相对于归档根目录：

```text
hmonnx_eval/work_dirs/hmonnx_eval/activation_sensitivity_20261027
```

归档根目录：/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028。提交仓库：/data01/home/jing.huang/xh2modelzoo-qwen3vl-reranker。下列路径均相对于 OUT。

| 证据 | 路径 / 字段 |
| --- | --- |
| 正式结果 | analysis/report.json → final713[*].modality_macro.full713 |
| 节点导出配置 | export/configs/combo_layers2.yaml<br>export/configs/combo_layers3.yaml<br>export/configs/combo_nodes02.yaml |
| 节点身份记录 | export/identities/{方案 ID}.json |
| 准入审计 | export/admission_{方案 ID}.json |
| 逐任务结果 | analysis/final713/{方案 ID}/REPORT713.md |

文档中的节点集合已与保存的 YAML、variant identity 交叉核对；数值直接读取最终报告，以未四舍五入的原始值计算宏平均后显示四位小数。BF16 与 W4A16 参照在三份正式结果中一致。三套产物准入记录确认仅修改指定 Linear 激活位宽，语言图 520 个 initializer payload 与原 W4A8 相同。

逐任务及四个查询分组的三方案并列表见[合并明细](reports/ACTIVATION_A16_713_DETAILS.md)，完整精确结果见[JSON](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/ACTIVATION_A16_713_COMPARISON_20261028.json)。
