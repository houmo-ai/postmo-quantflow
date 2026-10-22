# WeMM-Embedding MMEB-V3 最终精度报告

整理日期：2026-10-22。根据已完成并归档的结果整理；本次未重新运行推理或评分。

## 评测范围与配置

- 基准为各尺寸自己的浮点自测（原生 vLLM + WeMM 专用模型和 processor）；量化侧为同尺寸 W4A8，通过 vLLM Merak 执行 HMONNX。以下是 GPU 自测成绩，未执行 M50 板端评测。
- 对应独立 256K 产物，最大上下文为 262144 processed tokens；不代表模型卡基础 2048 recipe 的实测成绩。语言侧 W4A8，视觉侧 W8A8。
- 六份运行均完成 182/182 项任务，无缺失项和协议校验失败项。其中 179 项正式非音频任务计入汇总，另 3 项 VisDoc 诊断任务不计入汇总。
- 分数按 0～100 显示，表格保留四位小数；Δ = W4A8 − 自测浮点，单位为百分点。差值由未舍入值计算。
- Text、VisDoc 及 Text 子类使用 `ndcg_linear@5`；Image、Video、Tool、GUI、Memory、MCMR 使用 `hit@1`；分类分数是任务主指标的算术平均，不按样本量加权。
- Supported-Only = 179 项非音频任务之和 ÷ 179；V3-All = 同一分数之和 ÷ 190，包含 11 项未执行的音频任务零分。音频不支持，表中的总分不表示完成过音频推理。
- MMEB-v2 为 Image 36 + Video 18 + VisDoc 24，共 78 项；Agent 为 Tool 35 + GUI 8 + Memory 4，共 47 项。表格同时包含父类和子类，各行不能再次直接求平均。
- 图像保持宽高比并采用动态分辨率，像素范围 65536～1572864；量化视觉 token 档位为 `[96, 196, 384, 704, 1536]`。视频最多 64 帧、采样 fps=1。
- 浮点 `max_num_seqs=16`，W4A8 为 1；Text/GUI batch size=1，其他模态为 8；关闭 prefix caching。HMONNX prefill chunk=256、linear chunk=64，FlashAttention Q/K/V/S/P 均为 8 bit，PageAttention=true、runner_type=pooling。
- 评测代码版本为 `5f539a2f5196b08bbcf29ac71fc90a95ccad7f11`，MMEB-V3 数据版本为 `4a5560b2b64384204b6fea8a82ea986eba51f5aa`。完整配置、数据版本和复现入口见 [评测说明](EVALUATION.md)。
- 两侧使用不同推理后端，分差包含量化及导出运行时的数值影响，不能仅归因为权重位宽。

## 结果导航

- [WeMM-Embedding-2B](#wemm-embedding-2b)
- [WeMM-Embedding-4B](#wemm-embedding-4b)
- [WeMM-Embedding-9B](#wemm-embedding-9b)

## WeMM-Embedding-2B

模型卡：[wemm_embedding_2b.yaml](../model_cards/m50/wemm_embedding_2b.yaml)。

| 指标 / 分类 | 计入任务数 | 自测浮点（vLLM） | W4A8 HMONNX（vLLM Merak） | Δ（百分点） |
|---|---:|---:|---:|---:|
| V3-All（含 11 项音频零分） | 190 | 55.7517 | 55.2484 | -0.5034 |
| Supported-Only | 179 | 59.1778 | 58.6435 | -0.5343 |
| MMEB-v2 | 78 | 77.8667 | 77.3363 | -0.5303 |
| Image | 36 | 79.6417 | 78.9500 | -0.6917 |
| Video | 18 | 70.7001 | 70.3292 | -0.3709 |
| VisDoc | 24 | 80.5791 | 80.1711 | -0.4080 |
| Text | 53 | 45.3098 | 44.2730 | -1.0368 |
| Reasoning Retrieval | 20 | 26.2360 | 25.4117 | -0.8243 |
| Instruction Following | 9 | 49.3713 | 47.5058 | -1.8655 |
| Long Context | 6 | 61.7889 | 61.4163 | -0.3726 |
| Multi Condition | 5 | 59.6618 | 58.3289 | -1.3329 |
| General Retrieval | 13 | 58.7167 | 57.7340 | -0.9827 |
| Agent | 47 | 44.1538 | 44.2364 | +0.0826 |
| Tool | 35 | 46.3350 | 46.5030 | +0.1680 |
| GUI | 8 | 38.9330 | 38.9546 | +0.0216 |
| Memory | 4 | 35.5100 | 34.9671 | -0.5429 |
| MCMR | 1 | 42.5819 | 39.3795 | -3.2024 |

原始对照：`work_dirs/wemm_embedding/mmeb_v3_runs/2b/comparison/comparison.json`。

2B W4A8 为已审核的三阶段合并结果：138 项既有有效结果、36 项加速补跑、8 项修复后 GUI 重跑。保留各阶段真实来源，不声称所有任务具有同一个代码哈希。

## WeMM-Embedding-4B

模型卡：[wemm_embedding_4b.yaml](../model_cards/m50/wemm_embedding_4b.yaml)。

| 指标 / 分类 | 计入任务数 | 自测浮点（vLLM） | W4A8 HMONNX（vLLM Merak） | Δ（百分点） |
|---|---:|---:|---:|---:|
| V3-All（含 11 项音频零分） | 190 | 57.9266 | 57.8784 | -0.0482 |
| Supported-Only | 179 | 61.4863 | 61.4352 | -0.0511 |
| MMEB-v2 | 78 | 79.2425 | 78.9215 | -0.3211 |
| Image | 36 | 80.7833 | 80.5694 | -0.2139 |
| Video | 18 | 72.0850 | 72.0513 | -0.0337 |
| VisDoc | 24 | 82.2995 | 81.6021 | -0.6973 |
| Text | 53 | 47.8899 | 47.8456 | -0.0444 |
| Reasoning Retrieval | 20 | 30.0067 | 29.6075 | -0.3992 |
| Instruction Following | 9 | 52.0867 | 52.4054 | +0.3187 |
| Long Context | 6 | 62.4325 | 61.9494 | -0.4831 |
| Multi Condition | 5 | 60.1541 | 60.0376 | -0.1166 |
| General Retrieval | 13 | 61.0682 | 61.5486 | +0.4805 |
| Agent | 47 | 47.7656 | 48.1735 | +0.4078 |
| Tool | 35 | 50.3263 | 51.0217 | +0.6954 |
| GUI | 8 | 40.8153 | 40.9076 | +0.0922 |
| Memory | 4 | 39.2605 | 37.7836 | -1.4769 |
| MCMR | 1 | 41.9815 | 41.0558 | -0.9257 |

原始对照：`work_dirs/wemm_embedding/mmeb_v3_runs/4b/comparison/comparison.json`。

## WeMM-Embedding-9B

模型卡：[wemm_embedding_9b.yaml](../model_cards/m50/wemm_embedding_9b.yaml)。

| 指标 / 分类 | 计入任务数 | 自测浮点（vLLM） | W4A8 HMONNX（vLLM Merak） | Δ（百分点） |
|---|---:|---:|---:|---:|
| V3-All（含 11 项音频零分） | 190 | 59.3314 | 58.7987 | -0.5327 |
| Supported-Only | 179 | 62.9775 | 62.4120 | -0.5655 |
| MMEB-v2 | 78 | 80.4371 | 80.1167 | -0.3204 |
| Image | 36 | 81.6889 | 81.1667 | -0.5222 |
| Video | 18 | 74.2440 | 74.1364 | -0.1076 |
| VisDoc | 24 | 83.2042 | 83.0270 | -0.1772 |
| Text | 53 | 48.8504 | 48.4546 | -0.3958 |
| Reasoning Retrieval | 20 | 31.7770 | 31.6247 | -0.1523 |
| Instruction Following | 9 | 52.9553 | 52.3910 | -0.5642 |
| Long Context | 6 | 64.4917 | 63.8118 | -0.6799 |
| Multi Condition | 5 | 58.4212 | 57.6638 | -0.7574 |
| General Retrieval | 13 | 61.3755 | 60.9916 | -0.3838 |
| Agent | 47 | 50.2249 | 49.0717 | -1.1532 |
| Tool | 35 | 52.9986 | 51.5872 | -1.4114 |
| GUI | 8 | 43.8525 | 43.3057 | -0.5467 |
| Memory | 4 | 38.7001 | 38.5934 | -0.1067 |
| MCMR | 1 | 49.2369 | 48.1861 | -1.0508 |

原始对照：`work_dirs/wemm_embedding/mmeb_v3_runs/9b/comparison/comparison.json`。

## 原始证据与完整报告

本报告是可随仓库保存的分类汇总，不包含 182 项逐任务明细。逐任务分数及其他 Top-K 指标保留于本机归档：

```text
work_dirs/wemm_embedding/mmeb_v3_runs/
  {2b,4b,9b}/float/aggregate/{REPORT.md,summary.json,per_task.csv}
  {2b,4b,9b}/w4a8/aggregate/{REPORT.md,summary.json,per_task.csv}
  {2b,4b,9b}/comparison/{REPORT.md,comparison.json,per_task_delta.csv}
  {2b,4b,9b}/{float,w4a8}/results/<modality>/<dataset>/{score.json,protocol.json,config.yaml}
```

`work_dirs` 为本机运行归档，不随 Git 拉取。完整运行配置、来源与报告索引见 [EVALUATION.md](EVALUATION.md)。
