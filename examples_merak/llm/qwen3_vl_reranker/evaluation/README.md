# Qwen3-VL-Reranker 评测

本目录只保留正式报告与评测说明。原生 vLLM BF16、vLLM Merak HMONNX 的评测脚本和机器可读结果保存在长期归档。
量化配置与选择原因见[模型 README](../README.md#量化配置选择与局部-a16)。

## 结果入口

| 内容 | 报告 | 精确结果 |
| --- | --- | --- |
| 五种语言精度 × head A8/A16：逐任务、模态宏平均及 BF16 差值 | [Markdown](HEAD_A8_A16_713_REPORT.md) · [Word](HEAD_A8_A16_713_REPORT.docx) | [CSV](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/HEAD_A8_A16_713_REPORT.csv) · [JSON](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/HEAD_A8_A16_713_REPORT.json) |
| 三套局部 A16：14/21/2 个节点、BF16/W4A16 参照和宏平均 | [Markdown](ACTIVATION_A16_713_COMPARISON_20261028.md) · [Word](ACTIVATION_A16_713_COMPARISON_20261028.docx) | [CSV](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/ACTIVATION_A16_713_COMPARISON_20261028.csv) · [JSON](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/ACTIVATION_A16_713_COMPARISON_20261028.json) |
| 三套 A16 的逐任务、定位集与保留集对照 | [合并明细](reports/ACTIVATION_A16_713_DETAILS.md) | 同上 A16 JSON |
| 精度损失原因、已验证因素与结论边界 | [诊断结论](ACCURACY_DIAGNOSIS_FINAL.md) | 报告内证据链接 |
| 历次运行的最终状态、扩样与停止记录 | [历史时间线](HISTORY.md) | 完整旧文及逐查询证据在归档 |

[结果清单](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/reports/RESULTS_MANIFEST.json)记录当前报告与数据的 SHA、数值一致性和归档来源。
正式报告保持原文件名，Word 为固定导出版；A16 JSON 集中保存三套方案及四个查询分组的精确结果。

## 评测口径

正式快评为 `quick-mmeb-v1`：原 713 条、9 项任务、66,809 个候选对，cohort SHA 为
`610be4c4acf3b5cecf36b882b5372f01b15829ce7440b77ee7fd8e74de9cd706`。
图像 MSCOCO_i2t、MSCOCO_t2i、CIRR 各 100 条；文档 docvqa、tatdqa、ChartQA 为 100/100/63 条；
视频 MSR-VTT、DiDeMo、MomentSeeker 各 50 条。图像/视频采用 Hit@1、文档采用线性 nDCG@5，
每个模态内三个任务等权。该固定子集及另行扩至 3414 条的结果均不能标为官方全量 MMEB 总分。

- 官方参考源码固定到 QwenLM/Qwen3-VL-Embedding 的 `393e2978d27852b0d0230d6994f37f9c15bed73c`。
  Qwen3-VL-Embedding-2B 按官方 Transformers 实现召回，冻结 Top-100 后重排；不足 100 时保留原候选数量。
- 总上下文为 32768，包含末尾五个模板 token。媒体展开后保护特殊 token、视频时间戳跨度和末尾模板再截断；保留部分自身超限时失败。
- BF16 图像保持官方动态预算上限 1800 tokens；HMONNX 采用导出五档 96/196/384/704/1536，成绩包含该视觉输入预算差异。
- 视频沿用官方最多 64 帧、总像素预算对应 7680 tokens，复用相同源、所选帧、顺序、时间戳和补帧规则。原生 vLLM 与 Merak 分别使用其已验证的视觉边界包装，避免重复套用。
- 查询、候选身份、标签与官方 RankingMetrics 保持一致；媒体读取失败、协议不符和非有限分数会报错。分数按模型与协议独立存储。

## 评测工具与精确数据归档

[支持包说明](/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/supplements/evaluation_support_20261028/README.md)记录运行环境、包路径、工具测试和复用方式。
归档保留原相对包树，20 个评测模块的源码字节不变；主仓该目录不再提供可执行模块。

| 内容 | 支持包内位置 |
| --- | --- |
| 评测脚本及原模块路径 | `source_tree/examples_merak/llm/qwen3_vl_reranker/evaluation/*.py` |
| 精确 JSON、CSV 与整理校验清单 | 上述目录的 `reports/` |
| 四套工具专用测试及 impact 声明 | `source_tree/benchmark_test/ci_test_transformer5.16.0/` |

后续执行从支持包源码根目录按其 README 设置环境，先完成实际图属性、Golden 和官方输入对齐预检，
再为新模型建立独立计划。视觉缓存需要实测检查及固定 admission 后使用。
旧计划中的绝对路径和代码指纹按原字节保存，不能改写旧指纹冒充原评测恢复；查看历史报告无需启动旧监控。

## 缓存、复用与断点

跨模型复用数据集快照、Embedding 向量、冻结候选、标签及无损原始视频帧。
视频 PNG 保存 resize 前 RGB，逐帧校验像素，记录官方采样顺序、补帧和时间戳；不补造未提供的源视频 PTS。
最终视觉特征另按以下规则复用：

- 缓存 `encode` 边界的 `image_embeds` 与三层 DeepStack，以 safetensors 无损保留有效 token 的原 FP16 值，仅去档位 padding；视频按 temporal-patch 单元保持顺序和时间信息。
- namespace 覆盖五档计算图/量化属性、graph metadata、全部外部权重内容、视觉几何、实际运行时代码/二进制、CUDA/Torch/Triton 版本、GPU 类型及精度环境；条目 key 还包含实际 FP16 patch 内容和形状、`grid_thw` 及所选 gear。
- 当前仅准入已验证的 FP16、merge=2、ALIGNED、eager 路径。运行时变化默认拒绝；兼容登记必须重新证明四特征逐元素一致、token 与分数相同。视觉图或位宽改变须使用新 namespace。
- 写入使用文件锁、临时文件、fsync 和原子 rename；读取检查四层 SHA、shape、dtype 及数值有限性。损坏明确失败，2 GiB CPU LRU 返回副本。
- 接入前以 `visual_cache_check` 覆盖五档、图像、文档及三个视频任务，清空上层缓存，验证真实保存/读取钩子、逐元素特征及最终分数。overlay 保持原计划、8 候选批次、截断、视频重放和计分，另固定 admission。
- `visual_cache_fill` 只补冻结媒体的视觉计算，保留语言分数；receipt 记录媒体、采样参数和有序 feature keys。完成以监控中的全部媒体任务 done 为准。

SQLite WAL 提供独占领取与恢复，每条查询独立持久化，HMONNX 每 8 个候选原子保存批次。
媒体前处理缓存上限 4 GiB，并单线程有界预取；长期进程以新 session、空 stdin、文件日志脱离终端，
由 watchdog/supervisor 恢复退出或超时任务。FlashAttention 使用 HMONNX CUDA Triton 路径。
不跨模型复用语言激活、KV 或重排分数；以浮点视觉特征替换量化视觉会改变被测配置。

## 证据与归档

长期归档根目录：

```text
/data01/home/jing.huang/xh2modelzoo/work_dirs/qwen3_vl_reranker_2b_eval_archive_20261028/
```

`float_eval/work_dirs/` 保存原浮点向量、候选、帧和分数；`hmonnx_eval/work_dirs/` 保存模型、Golden、
量化分数、视觉缓存和诊断。`source_tree/`、Git bundle 及归档 README 记录原源码、路径映射和完整性审计。
历史 plan/SQLite 与 SHA 绑定文件保留原字节；使用 `PATH_MAP.json` 和 `archive_paths.py` 定位旧路径或生成派生 JSON。
归档采用同盘硬链接，后续实验写入新目录，避免覆盖历史证据。

本次整理前的 47 个文件已另行独立复制并逐 SHA 核验，完整旧文位置见 [HISTORY](HISTORY.md)。
旧分模型启动日志、重复索引和迁移过程清单归入该快照；当前脚本、工具测试和机器可读结果另外集中到上方支持包，本目录仅维护报告与说明。
