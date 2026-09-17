# WeMM-Embedding 适配、量化与精度验证

本目录提供 WeMM-Embedding 2B、4B、9B 在 xh2modelzoo 中的完整交付入口，包括：

- WeMM 专用模型注册、HMONNX 导出和推理接口；
- AutoRound 与 GPTQ 两套 4bit 权重量化 recipe；
- 量化、导出和 Golden 数据生成的一体化 workflow；
- 文本、图像、视频及混合输入的统一 embedding 接口；
- Flickr30K-1000 检索精度和全量 cosine 对齐测试；
- 可扩展的数据集注册接口；
- 当前 ModelZoo CI 中的便携契约测试，以及真实产物评测脚本和结果报告。

WeMM-Embedding 复用 Qwen3.5 多模态主干，但面向 embedding 任务而不是生成任务。导出时会将语言模型的 `lm_head` 替换为 `Identity`，HMONNX 运行时从最后一个有效 token 的 hidden state 取向量，并执行 FP32 L2 归一化。

| 模型 | embedding 维度 |
| --- | ---: |
| WeMM-Embedding-2B | 2048 |
| WeMM-Embedding-4B | 2560 |
| WeMM-Embedding-9B | 4096 |

## 支持状态

当前 AutoRound 配方使用 `int_sym_xh`、256 个校准样本、400 次迭代及 patch 视觉档位。2B、4B、9B 均已按此配方重新量化、导出，并完成 Flickr30K-1000 与 MMEB-v3 两项图像任务评测。

| 能力 | AutoRound | GPTQ |
| --- | --- | --- |
| 2B/4B/9B 4bit recipe | 已提供 | 已提供 |
| xh2modelzoo workflow 路由 | 已验证 | 已验证 |
| GPTQModel recipe 参数翻译 | 已验证 | 已验证 |
| provenance 双文件 | 已验证 | 已验证 |
| 真实量化产物 | 2B/4B/9B 已生成 | 旧产物已清理 |
| HMONNX 导出 | 2B/4B/9B 已生成 | 旧产物已清理 |
| 当前配方重新量化与导出 | 已验证 | 本轮未执行 |
| 当前视觉档位图导出与图像精度 | 已验证 | 本轮未执行 |
| 导出后自动校验 | 未启用 | 未启用 |
| Flickr30K-1000 精度 | 已复测 | 尚未测试 |

本轮精度结论见 [精度复测报告](ACCURACY_REPORT.md)；MMEB-v3 仅覆盖 Visual7W 和 ChartQA 两项完整图像 test 集，不是全套总分。

## 统一 embedding 接口

HMONNX 模型对文本、图像、视频以及混合输入提供同一套 item 接口。当前配方沿用 Qwen3.5 的 patch token 视觉图，导出 `[96, 196, 384, 704, 1536]` 个合并后图像 token 的档位。图片按实际 `image_grid_thw` 选择并补齐档位；视频的扁平 patch 按时间单元分组后调用同一档位图，特征按视频 token 插入语言输入，位置编码保留帧时间戳。单帧超过档位容量会明确报错，每个 item 的 processed tokens 仍受 2048 上限约束。

```python
text_embedding = model.embed({"text": "A dog playing in a park."})
image_embedding = model.embed({"image": "/path/to/image.jpg"})
video_embedding = model.embed({"video": "/path/to/video.mp4"})
mixed_embedding = model.embed(
    {
        "image": "/path/to/image.jpg",
        "text": "Represent the semantics of this image.",
    }
)

batch_embeddings = model.embed_items(
    [
        {"text": "first text"},
        {"image": "/path/to/image.jpg"},
    ]
)
text_embeddings = model.embed_texts(["first text", "second text"])
image_embeddings = model.embed_images(["/path/to/a.jpg", "/path/to/b.jpg"])
```

所有接口都返回 rank-2 FP32 tensor，每个输入 item 对应一行已经完成 L2 归一化的 embedding。

公开模型目录也注册了 `wemm-embedding-2b`、`wemm-embedding-4b`、`wemm-embedding-9b`。公开构建入口选择 AutoRound 配方；独立 workflow 仍提供 GPTQ 配方。公开接口的构建和调用方式如下，以 2B 为例：

```bash
python -m examples_merak.unified_api.catalog wemm-embedding-2b --full
python -m examples_merak.unified_api.build \
  --model-id wemm-embedding-2b \
  --model-dir /path/to/WeMM-Embedding-2B \
  --output-dir ./work_dirs/wemm_2b_api \
  --device cuda:0
python -m examples_merak.unified_api.embed \
  --artifact-dir ./work_dirs/wemm_2b_api \
  --text 'Represent this image.' \
  --image /path/to/image.jpg
```

本轮真实 HMONNX 图像评测使用动态分辨率输入并覆盖多个视觉 token 档位；视频精度仍需单独验证。Native/HF 与 HMONNX 使用相同的 65,536–1,572,864 像素动态范围并保持原图宽高比，详见后文和精度报告。

## 输入长度契约

当前 W4A8 HMONNX 导出使用 `prefill_chunk_length: 256`。公开的 embedding 运行时会先为完整 processed-token 序列生成位置编码，再按 256 token 切分并重复调用同一张 prefill 图；各段复用 full-attention KV cache、线性注意力 recurrent state 和卷积 cache，最后只对末段最后一个有效 token 的 hidden state 做 FP32 L2 归一化。

配置中的 `context_max_length: 2048` 是当前单条输入的累计 processed-token 上限。因此 257～2048 token 会执行多段 prefill，超过 2048 token 会明确报错，不会静默截断。这里的 processed tokens 包含 chat template 和视觉 token，不只是用户原始文本 token。

量化校准的 `seqlen` 与 HMONNX 单次推理长度也不是同一个概念：

- AutoRound 的 `seqlen: 2048` 用于让权重量化优化覆盖较长校准序列；
- GPTQ 的 `seqlen: 1024` 用于 Hessian/误差估计的校准样本长度；
- HMONNX 的 `prefill_chunk_length: 256` 是当前单次图执行的静态长度；
- HMONNX 的 `context_max_length: 2048` 是当前多段执行后的累计输入上限。

## 4bit 量化 recipe

### AutoRound

生产 AutoRound recipe：

```text
configs_merak/workflows/xh2a/llm_models/wemm_embedding/2b/wemm_embedding_2b_xh2a_w4a8.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/4b/wemm_embedding_4b_xh2a_w4a8.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/9b/wemm_embedding_9b_xh2a_w4a8.yaml
```

统一配置如下：

| 配置项 | 值 |
| --- | --- |
| 后端 | GPTQModel 库（`quant.algorithm: gptqmodel`），方法为 AutoRound（`quant.method: autoround`） |
| 量化范围 | 仅语言模型分支 |
| 权重 | 4bit、对称、group size 64 |
| AutoRound 数据类型 | `int_sym_xh` |
| 迭代次数 | 400 |
| 校准 | 256 个样本、2048 token、seed 42 |
| 校准数据 | `xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl`，由 `resolve_xh2modelzoo_uri()` 解析到仓库根目录 |
| 中间产物 | `auto_gptq` / GPTQModel-compatible HF checkpoint |
| 语言 HMONNX | `w4a8h0_ssfp` |
| 视觉 HMONNX | `w8a8h1_sefp` |
| 视觉输入 | patch token 档位 `[96, 196, 384, 704, 1536]`，单次最多 1536 个合并后图像 token |
| prefill/context | 256/2048 |
| 线性注意力 chunk | 64 |

AutoRound 量化接口不接收生成式校验选项，因此配方不设置无效的 `quant.validation`。嵌入向量的精度通过独立的 Flickr30K 评测与 Native/HMONNX cosine 对齐脚本验证。

### GPTQ

GPTQ recipe：

```text
configs_merak/workflows/xh2a/llm_models/wemm_embedding/2b/wemm_embedding_2b_xh2a_w4a8_gptq.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/4b/wemm_embedding_4b_xh2a_w4a8_gptq.yaml
configs_merak/workflows/xh2a/llm_models/wemm_embedding/9b/wemm_embedding_9b_xh2a_w4a8_gptq.yaml
```

统一配置如下：

| 配置项 | 值 |
| --- | --- |
| 后端 | GPTQModel 库（`quant.algorithm: gptqmodel`），方法为 GPTQ（`quant.method: gptq`）；调用 `gptqmodel.recipes.qwen35.quantize_qwen35` |
| 量化范围 | Qwen3.5/WeMM 语言层；`quant_nontext_module: false` |
| 权重 | 4bit、group size 64 |
| 误差优化 | `hessian_mse: true` |
| 校准 | 256 个样本、1024 token、seed 42 |
| 校准数据 | `gptqmodel://quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl` |
| batch size | 1 |
| GPTQ 后验 | 跳过 Qwen3.5 的 generate/PPL；导出后使用独立评测脚本检查 WeMM 检索精度和 Native cosine |
| 中间产物 | GPTQModel HF checkpoint |
| 语言 HMONNX | `w4a8h0_ssfp` |
| 视觉 HMONNX | `w8a8h1_sefp` |
| prefill/context | 256/2048 |

GPTQModel 仓库已经注册 `WeMMEmbeddingQModel`，它复用维护中的 `Qwen3_5_VLQModel` 模块树。xh2modelzoo 不重复实现 GPTQ 算法，只负责资源解析、参数翻译、recipe 调用、产物组织和后续 HMONNX 导出。共享 Qwen3.5 GPTQ 脚本默认执行文本生成、PPL 和可选视觉生成校验，这些指标不代表 embedding 质量，因此 WeMM GPTQ recipe 设置 `check_quant_vision_demo: false` 和 `skip_inference: true`。GPTQ 导出完成后可使用独立评测脚本检查检索精度和同序 Native cosine。

语言分支使用 4bit 权重，视觉分支仍按 W8A8 导出，因此这不是“所有多模态组件均为 4bit”的模型。

## 一体化 workflow

准备环境：

```bash
export PYTHONPATH=$PWD
# 可选：指定包含项目依赖的 Python
export PY=/path/to/python
```

### AutoRound 量化与导出

```bash
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b \
  --quant-method autoround \
  --device cuda:0
```

`--quant-method` 默认值是 `autoround`，因此兼容原有命令。将 `2b` 替换为 `4b` 或 `9b` 即可处理其他规格。

默认目录：

```text
work_dirs/wemm_embedding/<size>/wemm_embedding_quant
work_dirs/wemm_embedding/<size>/wemm_embedding_export
```

### GPTQ 量化与导出

```bash
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b \
  --quant-method gptq \
  --device cuda:0
```

GPTQ 使用独立目录，避免覆盖已有 AutoRound 产物：

```text
work_dirs/wemm_embedding/<size>/wemm_embedding_gptq_quant
work_dirs/wemm_embedding/<size>/wemm_embedding_gptq_export
```

导出步骤不自动执行 embedding 精度后验。需要验证质量时，运行后文的 Flickr30K 检索与 Native/HMONNX cosine 评测。

### 模型根目录与复用已有量化模型

YAML 中的 `export.model.hf_model: null` 只是占位，不会作为真实路径读取。CLI 在量化/导出前会执行 `preflight_recipe()`：确认 YAML 仍是 null 占位，`--model-dir` 必须是含 `config.json` 的目录而不是文件，校准 jsonl 通过 `resolve_xh2modelzoo_uri()` 解析 `xh2modelzoo://` / `repo://` / `gptqmodel://` / `file://`、绝对路径和相对路径。`BaseLLMWorkflow._build_export_config()` 随后把 `hf_model` 覆盖成量化产物目录；WeMM workflow 会再做一次非空校验，覆盖失败或目录不存在时直接报错，不会带着 null 进入导出器。语言与视觉 `quant_scheme.ops` 均为空，MatMul 使用与当前 Qwen3.5 9B 配方一致的量化默认规则；此前强制第二操作数为 16bit 的覆盖已移除，新精度需在重新量化、导出后验证。独立精度评测需要相应的 HMONNX 产物、Native 模型和 Flickr30K 数据。

`WEMM_MODEL_ROOT` 默认对应 `/data01/datasets`，也可以覆盖：

```bash
WEMM_MODEL_ROOT=/models \
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 4b \
  --quant-method gptq \
  --device cuda:0
```

复用默认量化目录：

```bash
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 4b \
  --quant-method autoround \
  --skip-quant \
  --device cuda:0 \
  --overwrite
```

也可以通过 `--quant-model-dir` 指定已有的 AutoRound 或 GPTQ checkpoint。此时 `--quant-method` 仍应与 checkpoint 类型一致，以便选择正确的导出配置和默认输出目录。

### 可信模型代码与快照

WeMM checkpoint 包含模型自定义代码，因此量化和 Native 对齐需要 `trust_remote_code: true`。当前 workflow 默认使用本地 `/data01/datasets/WeMM-Embedding-*` 快照，不会在运行时按模型 ID 下载任意远程仓库。执行前应确认快照来源可信，并使用不可变目录或校验清单固定版本。例如：

```bash
MODEL_DIR=/data01/datasets/WeMM-Embedding-2B
MANIFEST=/secure/manifests/WeMM-Embedding-2B.sha256

find "$MODEL_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$MANIFEST"

sha256sum --check "$MANIFEST"
```

校验清单应存放在模型目录之外的只读位置。模型快照发生变化后必须重新量化、导出并执行独立精度评测；不应对来源不明或可被其他用户修改的目录启用 `trust_remote_code`。

### Golden 输入

Golden 数据生成与 HMONNX 推理使用相同的 item 接口。每次 workflow 可选择一种代表性输入：

```bash
# 文本
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b --skip-quant --overwrite --dump-golden \
  --golden-mode text --golden-text "Represent this input."

# 图像
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b --skip-quant --overwrite --dump-golden \
  --golden-mode image --golden-image /path/to/image.jpg

# 图文混合
python examples_merak/llm/wemm_embedding/wemm_embedding_workflow.py \
  --size 2b --skip-quant --overwrite --dump-golden \
  --golden-mode mixed --golden-image /path/to/image.jpg \
  --golden-text "Represent the semantics of this image."
```

## provenance 文件

WeMM 复用 Qwen3.5 的 GPTQModel recipe，因此后端会生成带 `qwen35` 名称的原始 provenance。WeMM workflow 保留原始文件，同时复制一份内容完全相同的 WeMM 交付别名：

| 方法 | 后端原始文件 | WeMM 交付别名 |
| --- | --- | --- |
| AutoRound | `qwen35_autoround_recipe_provenance.json` | `wemm_embedding_autoround_recipe_provenance.json` |
| GPTQ | `qwen35_quant_recipe_provenance.json` | `wemm_embedding_gptq_recipe_provenance.json` |

原始文件用于准确记录实际调用的共享实现，WeMM 别名用于让交付目录语义清晰。两者并存，不隐藏也不改写真实后端来源。

## HMONNX 示例

```bash
export ENABLE_HMINFERENCE_V2=1

python examples_merak/llm/wemm_embedding/hmonnx_demo.py \
  --export-dir work_dirs/wemm_embedding/2b/wemm_embedding_export \
  --text "A dog playing in a park." "A cat sitting on a chair." \
  --image work_dirs/wemm_embedding/datasets/flickr30k/flickr30k-images/1007129816.jpg \
  --device cuda:0
```

示例会输出每组输入的 shape、dtype、L2 norm 和完整相似度矩阵。

视频示例：

```bash
python examples_merak/llm/wemm_embedding/hmonnx_demo.py \
  --export-dir work_dirs/wemm_embedding/2b/wemm_embedding_export \
  --text "Represent this video." \
  --video /path/to/video.mp4 \
  --device cuda:0
```

视频推理和精度需要基于本轮重新导出的档位图验证。

单独运行 Flickr30K 数据准备、评测、汇总和 cosine 对比：

```bash
python examples_merak/llm/wemm_embedding/prepare_flickr30k.py \
  --output-dir work_dirs/wemm_embedding/datasets/flickr30k

python examples_merak/llm/wemm_embedding/eval_flickr30k.py \
  --backend hmonnx \
  --export-dir work_dirs/wemm_embedding/2b/wemm_embedding_export \
  --dataset-dir work_dirs/wemm_embedding/datasets/flickr30k \
  --device cuda:0 \
  --image-min-pixels 65536 --image-max-pixels 1572864 \
  --save-embeddings work_dirs/wemm_embedding/eval_flickr30k_1000/2b_hmonnx_w4a8_dynamic \
  --report work_dirs/wemm_embedding/eval_flickr30k_1000/2b_hmonnx_w4a8_dynamic_flickr1000.json

python examples_merak/llm/wemm_embedding/eval_flickr30k_parallel.py \
  --export-dir work_dirs/wemm_embedding/9b/wemm_embedding_export \
  --dataset-dir work_dirs/wemm_embedding/datasets/flickr30k \
  --device cuda:0 \
  --workers 8 \
  --image-min-pixels 65536 --image-max-pixels 1572864 \
  --output-dir work_dirs/wemm_embedding/eval_flickr30k_1000/9b_hmonnx_w4a8_dynamic \
  --report work_dirs/wemm_embedding/eval_flickr30k_1000/9b_hmonnx_w4a8_dynamic_flickr1000.json
```

并行评测按 caption/image 下标切分 worker。合并时要求全部 `worker_XX.pt` 存在、区间无重叠或空洞，并且拼接后的行数等于全集。缺失 shard、区间冲突或 worker 非零退出都会直接失败，不会静默跳过或错序拼接。

```bash
python examples_merak/llm/wemm_embedding/compare_full_cosine.py \
  --native-dir work_dirs/wemm_embedding/eval_flickr30k_1000/2b_native_dynamic \
  --hmonnx-dir work_dirs/wemm_embedding/eval_flickr30k_1000/2b_hmonnx_w4a8_dynamic \
  --report work_dirs/wemm_embedding/eval_flickr30k_1000/2b_full_cosine_dynamic.json

python examples_merak/llm/wemm_embedding/compare_native_hmonnx.py \
  --model-dir /data01/datasets/WeMM-Embedding-2B \
  --export-dir work_dirs/wemm_embedding/2b/wemm_embedding_export \
  --device cuda:0

python examples_merak/llm/wemm_embedding/summarize_accuracy.py \
  --report-dir work_dirs/wemm_embedding/eval_flickr30k_1000 \
  --output work_dirs/wemm_embedding/eval_flickr30k_1000/summary.json
```

完整 2B/4B/9B Native + HMONNX 套件仍使用 `run_accuracy_suite.sh`。

## Flickr30K-1000 精度

保留的精度协议使用 Flickr30K test 全量 split：

- 1000 张图像；
- 5000 条 caption；
- Native BF16 与 HMONNX 均保持原图宽高比，使用相同的 65,536–1,572,864 像素动态范围；
- HMONNX 按有效视觉 token 数选择最小可容纳档位；
- HMONNX 语言 W4A8、视觉 W8A8；
- 全量 text-to-image 和 image-to-text 检索；
- Recall@1/5/10、MRR@10、nDCG@10；
- 5000 条文本和 1000 张图像的逐行全量 cosine 对齐。

准备数据集：

```bash
python examples_merak/llm/wemm_embedding/prepare_flickr30k.py \
  --output-dir work_dirs/wemm_embedding/datasets/flickr30k
```

复现 2B/4B/9B Native、HMONNX 和全量 cosine：

```bash
bash examples_merak/llm/wemm_embedding/run_accuracy_suite.sh

python examples_merak/llm/wemm_embedding/audit_dynamic_resolution.py \
  --flickr-dir work_dirs/wemm_embedding/datasets/flickr30k \
  --mmeb-data-root work_dirs/wemm_embedding/mmeb_v3_data \
  --output work_dirs/wemm_embedding/DYNAMIC_IMAGE_PREPROCESSING_AUDIT.json
```

可通过 `WEMM_MODEL_ROOT`、`WEMM_EXPORT_ROOT`、`DATA` 和 `OUT` 指定模型、
HMONNX 导出产物、数据集和结果的根目录。
9B 并行评测的 worker 默认最多等待 7200 秒，可用
`eval_flickr30k_parallel.py --worker-timeout` 调整。

结果目录：

```text
work_dirs/wemm_embedding/eval_flickr30k_1000
```

本轮结果已写入 `work_dirs/wemm_embedding/eval_flickr30k_1000/summary.json` 和 [精度复测报告](ACCURACY_REPORT.md)。MMEB-v3 图像任务的协议与复现方法见 [评测协议](MMEB_V3_IMAGE_PROTOCOL.md)。

`native_dynamic` 表示使用原始 BF16 模型作为参考，保持原图宽高比并使用与 HMONNX 相同的动态像素范围。“全量 cosine”表示对所有同序样本逐行计算 HMONNX embedding 与 Native embedding 的 cosine。结果 JSON 同时记录图像预处理协议，报告生成器会拒绝汇总协议不一致的结果。
动态分辨率审计脚本逐张记录数据集的原始尺寸数、预处理尺寸数、视觉 token 范围和五档视觉图命中数，输出会被精度报告强制校验。

## 数据集扩展接口

Flickr30K 通过 `retrieval_dataset.py` 中的通用检索数据集接口注册。后续数据集 loader 只需返回图像路径及每张图像的一条或多条正样本 caption：

```python
from examples_merak.llm.wemm_embedding.retrieval_dataset import (
    RetrievalDataset,
    register_retrieval_dataset,
)


def load_custom_dataset(dataset_dir):
    return RetrievalDataset(
        name="custom-dataset",
        split="test",
        rows=[
            ("/path/to/image-1.jpg", ["positive caption", "another caption"]),
        ],
        metadata={"dataset_dir": str(dataset_dir)},
    )


register_retrieval_dataset("custom", load_custom_dataset)
```

评测器消费统一的 `RetrievalDataset` 契约，因此更换 image-caption 数据集时无需重写指标、embedding 保存和 cosine 比较逻辑。若新数据集 caption 显著更长，应同时统计 processed-token 长度分布，并分别调整量化校准覆盖和运行时长度能力；不能只把校准 `seqlen` 改大就认为 HMONNX 已支持长输入。

## CI 与验证

WeMM 测试位于当前 ModelZoo CI 目录 `benchmark_test/ci_test_transformer5.16.0/`，由相邻的
`*.impact.json` 声明选择。support impact 同时声明了 WeMM 直接复用的 Qwen3.5 processor、
data preprocess、visual gears、HMONNX runtime、模型、量化 adapter 和 workflow 源文件；修改这些
共享依赖也会重新运行 WeMM 用例。

预览最终变更会选择哪些测试：

```bash
python benchmark_test/ci_test_transformer5.16.0/select_tests.py --format report
```

单独执行三份便携 WeMM 测试：

```bash
python -m pytest \
  -c benchmark_test/ci_test_transformer5.16.0/pytest.ini \
  --confcutdir=benchmark_test/ci_test_transformer5.16.0 \
  benchmark_test/ci_test_transformer5.16.0/test_wemm_embedding_contract.py \
  benchmark_test/ci_test_transformer5.16.0/wemm_embedding_support_test.py \
  benchmark_test/ci_test_transformer5.16.0/wemm_embedding_eval_test.py \
  -q
```

仓库级入口为 `./benchmark_test/ci_test_transformer5.16.0/run.sh`。本次提交没有交付依赖真实
checkpoint/HMONNX 产物的 pytest 集成测试，也没有
`tests/wemm_embedding_integration_test.py`。三规格真实 AutoRound 量化、导出、Flickr30K 和
MMEB-v3 验证由本目录的运行脚本完成，结果保存在忽略提交的 `work_dirs`，汇总结论保留在
[精度复测报告](ACCURACY_REPORT.md)。因此报告中的真实产物结果是本轮外部运行证据，不属于默认 CI 覆盖。

GPTQModel 中的 WeMM 注册可独立验证：

```bash
cd "${GPTQMODEL_ROOT:?请设置 GPTQModel 仓库路径}"
python -m pytest -q tests/test_wemm_embedding_support.py
```

## 当前限制

- 公开 HMONNX embedding 接口按 256 token 重复执行 prefill，累计最多支持 2048 processed tokens；
- decode、生成、page attention 不是 WeMM embedding runtime 的公开能力；
- 当前 AutoRound 已完成 Flickr30K 与 MMEB-v3 的 Visual7W、ChartQA 图像任务复测；MMEB-v3 全套任务未执行；
- GPTQ 旧产物已清理，本轮不重新量化 GPTQ；
- 多段长文本与 Native 的全量精度对齐和视频精度仍需单独验证。

更多实现细节见 `IMPLEMENTATION.md`，全部交付文件见 `ARTIFACTS.md`，WeMM 与 Qwen3.5 的差异见 `WEMM_VS_QWEN35.md`。
