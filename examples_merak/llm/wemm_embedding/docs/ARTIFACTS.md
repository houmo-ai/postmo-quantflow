# WeMM-Embedding 文件与产物索引

2026-10-16 整理。量化/导出入口见 [README](../README.md)，实际评测配置、完整命令和报告索引见 [EVALUATION.md](EVALUATION.md)。

## 模型、配置与推理

| 位置 | 作用 |
|---|---|
| `xhmodel_merak/xh_llm/models/wemm_embedding/` | WeMM 模型、processor、HMONNX 推理、workflow 及注册 |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/{2b,4b,9b}/` | AutoRound/GPTQ 基础 recipe 与独立 256K 导出 recipe |
| `vllm_merak/vllm_merak/model_executor/model_loader/wemm_embedding.py` | 量化 pooling 模型的插件实现 |
| `vllm_merak/vllm_merak/model_executor/model_loader/wemm_embedding_native.py` | 原生浮点 WeMM 模型与多模态 processor |
| `vllm_merak/vllm_merak/native_wemm.py` | 原生 vLLM 的 WeMM 插件注册入口 |
| `wemm_embedding_workflow.py` | 量化、导出与 Golden 数据生成 |
| `hmonnx_demo.py` | 文本、图像、视频 embedding 示例 |
| `native_utils.py`、`runtime_utils.py` | 官方浮点和 HMONNX 加载辅助 |
| `model_cards/m50/wemm_embedding_{2b,4b,9b}.yaml` | M50 目标的公开模型卡；precision.comparisons 引用精简最终报告 |

## 主要评测代码

| 位置 | 作用 |
|---|---|
| `mmeb/run_mmeb_v3_vllm_merak_full.py` | 全量调度、独立任务动态领取、embedding 分片、校验与续跑 |
| `mmeb/mmeb_v3_eval.py` | 官方数据集与评分入口；模型后端适配与分片编码 |
| `mmeb/mmeb_v3_vllm_merak.py` | vLLM / Merak embedding 后端 |
| `mmeb/aggregate_mmeb_v3.py` | 正确任务集合的协议校验、汇总及官方成绩对比 |
| `mmeb/compare_mmeb_v3_aggregates.py` | 与自测浮点逐任务对齐、协议比较和报告 |
| `flickr/run_accuracy_suite.sh` | 独立 Flickr30K 图文检索及全量 cosine 套件 |
| `flickr/eval_flickr30k.py`、`flickr/eval_flickr30k_parallel.py` | Flickr30K 单进程 / 分片评测 |
| `flickr/prepare_flickr30k.py`、`flickr/flickr30k_utils.py`、`flickr/retrieval_dataset.py` | 图文检索数据准备与统一指标辅助 |
| `flickr/compare_full_cosine.py`、`flickr/summarize_accuracy.py` | Flickr30K 同序向量对比与汇总 |
| `checks/compare_native_hmonnx.py`、`checks/evaluate_wemm_embedding.py` | 小样本对齐 / 文本检索冒烟 |
| `checks/audit_dynamic_resolution.py` | 真实图像动态尺寸、视觉 token 和档位审计 |

## 本机评测留档

随仓库保存的精简最终报告为 [MMEB_V3_REPORT.md](MMEB_V3_REPORT.md)，按 2B、4B、9B 分表展示总分与分类成绩。原始逐任务分数、协议和完整报告继续保存在以下本机目录。

```text
work_dirs/wemm_embedding/
  2b_256k_context/wemm_embedding_export/   2B HMONNX 图和运行配置
  4b_256k_context/wemm_embedding_export/   4B HMONNX 图和运行配置
  9b_256k_context/wemm_embedding_export/   9B HMONNX 图和运行配置
  mmeb_v3_official/                       固定版本官方评测代码
  mmeb_v3_data/                           官方数据与媒体文件
  mmeb_v3_runs/
    README.md                            本机配置与报告索引
    2b/{float,w4a8,comparison}/           2B 最终结果
    4b/{float,w4a8,comparison}/           4B 最终结果
    9b/{float,w4a8,comparison}/           9B 最终结果
    references/                          官方 4B/9B 逐任务成绩
    sources/                             冻结模型路径源码压缩证据和哈希清单
    validation/                          前后处理及代码恢复一致性的小体积证据
    cleanup_manifest.json                清理范围、空间、旧新目录映射和验证结果
```

三种尺寸均已完成浮点及 W4A8 的 182 个执行任务；179 个进入非音频汇总，三个 VisDoc 诊断项单列。
V3-All 的 11 项音频按 WeMM 官方约定记零。2B W4A8 保留三阶段合并来源，不声称所有任务来自同一个源码哈希。

运行归档保留原始分数、协议、YAML、配置/模型/源码哈希、执行和补跑记录，可重新校验并生成报告。
embedding 缓存、逐样本预测、临时控制器、错误/中止运行及被全量覆盖的快速测试已删除。
源码和报告文档可提交；本机模型、数据与运行归档位于忽略提交的 `work_dirs`。

回归验证位于 `benchmark_test/ci_test_transformer5.16.0/` 的 WeMM 契约、支持和评测用例。
