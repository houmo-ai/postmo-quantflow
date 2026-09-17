# WeMM-Embedding 交付产物说明

本文汇总完成 WeMM-Embedding 适配所涉及的源码、配置、文档、测试和可再生产物。

状态说明：

- **源码交付**：仓库内维护的代码、配置或文档；
- **可再生**：通过 workflow 或评测命令重新生成；
- **真实验证**：已使用真实模型/导出/数据运行；
- **便携验证**：通过不依赖 GPU 和真实 checkpoint 的 CI 验证；
- **待真实验证**：接口和配置已完成，但尚未执行完整真实任务。

## 模型适配源码

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `xhmodel_merak/xh_llm/models/wemm_embedding/__init__.py` | 导出 WeMM 模型、processor、runtime 和 workflow 公共类。 | 源码交付，便携验证 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/model.py` | WeMM 专用注册、`lm_head=Identity`、hidden-state 输出、动态 embedding metadata。 | 源码交付，AutoRound 真实验证 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/processor.py` | embedding chat template 行为和视觉 patch 预处理。 | 源码交付，图像评测真实验证 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/hmonnx_inference.py` | 统一 item 接口、prefill 执行、长度检查、pooling、归一化和输出维度校验。 | 源码交付，图像评测真实验证 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/workflow.py` | 复用 Qwen3.5 量化/导出、覆盖并校验 `hf_model`、生成 AutoRound/GPTQ provenance 别名、通过 `embed()` 生成 Golden。 | 源码交付，AutoRound 真实验证 |
| `xhmodel_merak/xh_llm/models/wemm_embedding/export_checks.py` | 拒绝导出配置中残留的 `hf_model: null` 占位。 | 源码交付，便携验证 |

## 4bit recipe

### AutoRound

| 文件 | 模型 | 状态 |
| --- | --- | --- |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/2b/wemm_embedding_2b_xh2a_w4a8.yaml` | 2B，AutoRound `int_sym_xh` W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，真实验证 |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/4b/wemm_embedding_4b_xh2a_w4a8.yaml` | 4B，AutoRound `int_sym_xh` W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，真实验证 |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/9b/wemm_embedding_9b_xh2a_w4a8.yaml` | 9B，AutoRound `int_sym_xh` W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，真实验证 |

### GPTQ

| 文件 | 模型 | 状态 |
| --- | --- | --- |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/2b/wemm_embedding_2b_xh2a_w4a8_gptq.yaml` | 2B，GPTQ W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，旧产物已清理 |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/4b/wemm_embedding_4b_xh2a_w4a8_gptq.yaml` | 4B，GPTQ W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，旧产物已清理 |
| `configs_merak/workflows/xh2a/llm_models/wemm_embedding/9b/wemm_embedding_9b_xh2a_w4a8_gptq.yaml` | 9B，GPTQ W4G64，语言 W4A8、视觉 W8A8。 | 源码交付，旧产物已清理 |

## Workflow 与统一接口示例

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py` | 按 2B/4B/9B 和 AutoRound/GPTQ 选择 recipe，执行量化、导出和 Golden。 | 源码交付，AutoRound 真实验证 |
| `examples_merak/llm/wemm_embedding/hmonnx_demo.py` | 文本、图像、混合输入的统一 HMONNX embedding 示例及相似度矩阵。 | 源码交付，示例命令未单独运行 |
| `examples_merak/llm/wemm_embedding/runtime_utils.py` | HMONNX runtime 路径、metadata 和设备相关辅助。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/native_utils.py` | Native BF16 WeMM 加载和 embedding 生成辅助。 | 源码交付，真实模型可用 |

## 精度与数据集工具

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `examples_merak/llm/wemm_embedding/retrieval_dataset.py` | 通用 `RetrievalDataset`、loader 注册和加载接口。 | 源码交付，便携验证 |
| `examples_merak/llm/wemm_embedding/flickr30k_utils.py` | Flickr30K 标注、样本顺序和检索数据辅助。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/prepare_flickr30k.py` | 准备 Flickr30K 数据和标准 test split。 | 源码交付，可再生 |
| `examples_merak/llm/wemm_embedding/evaluate_wemm_embedding.py` | 通用 embedding 评测入口和指标计算。 | 源码交付，便携验证 |
| `examples_merak/llm/wemm_embedding/eval_flickr30k.py` | 单进程 Native/HMONNX Flickr30K 检索评测。 | 源码交付，Native 真实验证 |
| `examples_merak/llm/wemm_embedding/eval_flickr30k_parallel.py` | HMONNX 分片评测，合并时校验 shard 齐全且区间无重叠。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/compare_native_hmonnx.py` | Native 与 HMONNX 对齐比较入口。 | 源码交付，独立命令未单独运行 |
| `examples_merak/llm/wemm_embedding/compare_full_cosine.py` | 对全部同序文本和图像 embedding 计算逐行 cosine。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/summarize_accuracy.py` | 汇总 2B/4B/9B 检索与 cosine 结果。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/audit_dynamic_resolution.py` | 逐张审计 Flickr30K、Visual7W、ChartQA 的动态尺寸、视觉 token 和五档图命中分布。 | 源码交付，真实数据验证 |
| `examples_merak/llm/wemm_embedding/run_accuracy_suite.sh` | 一键复现三规格 Native、HMONNX、全量 cosine 和 summary。 | 源码交付，真实验证 |
| `examples_merak/llm/wemm_embedding/ACCURACY_REPORT.md` | 本轮 Flickr30K 与 MMEB-v3 两项图像任务精度报告。 | 真实验证 |

## Model Card

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `examples_merak/llm/wemm_embedding/model_cards/wemm_embedding_2b.yaml` | 2B 模型信息、接口、量化格式、精度证据和限制。 | 源码交付，schema 验证 |
| `examples_merak/llm/wemm_embedding/model_cards/wemm_embedding_4b.yaml` | 4B 模型信息、接口、量化格式、精度证据和限制。 | 源码交付，schema 验证 |
| `examples_merak/llm/wemm_embedding/model_cards/wemm_embedding_9b.yaml` | 9B 模型信息、接口、量化格式、精度证据和限制。 | 源码交付，schema 验证 |

Model Card 中的旧精度数值已清理；本轮 AutoRound 证据见 [精度复测报告](ACCURACY_REPORT.md)。

## 文档

| 文件 | 作用 |
| --- | --- |
| `examples_merak/llm/wemm_embedding/README.md` | 中文使用说明、量化配置、workflow、接口、精度结果和限制。 |
| `examples_merak/llm/wemm_embedding/IMPLEMENTATION.md` | 中文架构、量化、运行时、评测和验证边界说明。 |
| `examples_merak/llm/wemm_embedding/ARTIFACTS.md` | 本文，汇总全部交付产物及状态。 |
| `examples_merak/llm/wemm_embedding/WEMM_VS_QWEN35.md` | WeMM-Embedding 与 Qwen3.5 的差异和复用边界。 |

## CI 文件

| 文件 | 覆盖内容 | 状态 |
| --- | --- | --- |
| `benchmark_test/ci_test_transformer5.16.0/test_wemm_embedding_contract.py` | 公共 recipe、动态视觉档位、视频位置/feature、HMONNX 路由和统一推理 adapter。 | 当前 CI 便携验证 |
| `benchmark_test/ci_test_transformer5.16.0/test_wemm_embedding_contract.impact.json` | 将 WeMM 模型、integration、配置和示例映射到主契约测试。 | 当前 CI 选择声明 |
| `benchmark_test/ci_test_transformer5.16.0/wemm_embedding_support_test.py` | 模型注册、六份 recipe、AutoRound/GPTQ 路由、provenance、CLI、pooling、0～2049 长度边界、跨块 cache 和多模态 item 契约。 | 当前 CI 便携验证 |
| `benchmark_test/ci_test_transformer5.16.0/wemm_embedding_support_test.impact.json` | 将 WeMM 路径及其直接复用的 Qwen3.5 processor、runtime、量化 adapter 和 workflow 映射到 support 测试。 | 当前 CI 选择声明 |
| `benchmark_test/ci_test_transformer5.16.0/wemm_embedding_eval_test.py` | 指标、数据集接口、并行 shard、失败清理和动态分辨率审计。 | 当前 CI 便携验证 |
| `benchmark_test/ci_test_transformer5.16.0/wemm_embedding_eval_test.impact.json` | 将 WeMM 评测示例映射到 eval 测试。 | 当前 CI 选择声明 |
| `benchmark_test/ci_test_transformer5.16.0/test_model_build_registry.py` | 三个公开 model ID、variant、catalog snapshot 和动态视觉 profile。 | 共享注册表验证 |

当前项目 CI 入口为 `benchmark_test/ci_test_transformer5.16.0/run.sh`；本次提交没有交付
`tests/wemm_embedding_integration_test.py` 或其他依赖真实 HMONNX 产物的 pytest 集成测试。

## 可再生运行产物

以下目录位于 `work_dirs`，属于可再生运行产物，不应与源码文件混淆。

### AutoRound 量化和导出

```text
work_dirs/wemm_embedding/2b/wemm_embedding_quant
work_dirs/wemm_embedding/2b/wemm_embedding_export
work_dirs/wemm_embedding/4b/wemm_embedding_quant
work_dirs/wemm_embedding/4b/wemm_embedding_export
work_dirs/wemm_embedding/9b/wemm_embedding_quant
work_dirs/wemm_embedding/9b/wemm_embedding_export
```

这些目录已按当前 AutoRound 配方重新生成；旧产物及旧精度结果已清理。

### GPTQ 量化和导出

```text
work_dirs/wemm_embedding/2b/wemm_embedding_gptq_quant
work_dirs/wemm_embedding/2b/wemm_embedding_gptq_export
work_dirs/wemm_embedding/4b/wemm_embedding_gptq_quant
work_dirs/wemm_embedding/4b/wemm_embedding_gptq_export
work_dirs/wemm_embedding/9b/wemm_embedding_gptq_quant
work_dirs/wemm_embedding/9b/wemm_embedding_gptq_export
```

此前的 GPTQ 产物已按本轮清理要求删除。本轮只重新执行 AutoRound；GPTQ 需另行量化、导出和评测。

### provenance

AutoRound 量化目录应同时包含：

```text
qwen35_autoround_recipe_provenance.json
wemm_embedding_autoround_recipe_provenance.json
```

GPTQ 量化目录应同时包含：

```text
qwen35_quant_recipe_provenance.json
wemm_embedding_gptq_recipe_provenance.json
```

同一方法的两个 provenance 文件内容一致：前者记录真实共享后端，后者提供 WeMM 交付命名。

### Flickr30K-1000 精度

```text
work_dirs/wemm_embedding/datasets/flickr30k
work_dirs/wemm_embedding/eval_flickr30k_1000
work_dirs/wemm_embedding/eval_flickr30k_1000/summary.json
work_dirs/wemm_embedding/DYNAMIC_IMAGE_PREPROCESSING_AUDIT.json
```

本轮 Flickr30K 已生成三规格 Native BF16 与 AutoRound HMONNX W4A8 的检索及逐样本 cosine 结果；MMEB-v3 两项图像任务的结果位于 `work_dirs/wemm_embedding/mmeb_v3_runs`，汇总见 [精度复测报告](ACCURACY_REPORT.md)。

## 主要复现命令

AutoRound 量化、导出和 Golden：

```bash
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b --quant-method autoround --device cuda:0 --dump-golden
```

GPTQ 量化、导出和 Golden：

```bash
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b --quant-method gptq --device cuda:0 --dump-golden
```

导出步骤不自动运行 embedding 精度后验；需要时可使用下面的独立评测命令。

三规格 Flickr30K-1000：

```bash
bash examples_merak/llm/wemm_embedding/run_accuracy_suite.sh
```

便携 CI：

```bash
python -m pytest \
  -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_wemm_embedding_contract.py \
  benchmark_test/ci_test_transformer5.16.0/wemm_embedding_support_test.py \
  benchmark_test/ci_test_transformer5.16.0/wemm_embedding_eval_test.py \
  -q
```

真实产物验证未封装为随提交交付的 pytest 集成测试。复现时运行
`run_accuracy_suite.sh` 和 `run_mmeb_v3_suite.sh`；导出图、逐样本向量和原始评测结果保存在
`work_dirs/wemm_embedding/`，不进入源码提交，汇总结论见 `ACCURACY_REPORT.md`。

## 交付边界

- AutoRound：三规格真实量化、导出、Flickr30K 与 MMEB-v3 Visual7W、ChartQA 图像任务已验证；
- GPTQ：代码、recipe、共享后端注册、workflow 路由、参数翻译和 CI 已完成；旧运行产物已清理；
- 长输入：已实现单次 256、累计最多 2048 processed tokens 的多 chunk prefill；本轮未执行长文本专项精度测试；
- 数据集：已保留通用注册接口，后续可接入其他 image-caption 数据集；
- 精度：旧产物结论已撤销，本轮结果见 [精度复测报告](ACCURACY_REPORT.md)；MMEB-v3 全套任务未执行。

长输入及视频的真实产物验证也需要基于本轮导出结果重新执行。
