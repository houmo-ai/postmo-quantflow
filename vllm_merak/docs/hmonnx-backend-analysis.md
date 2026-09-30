# vLLM Merak hmonnx 后端分析

> 本文基于当前仓库实现整理，目标是帮助维护者理解一次 vLLM 请求如何进入 hmonnx runtime，以及新增 hmonnx 模型架构时需要满足哪些契约。本文不假设真实模型目录、`xhmodel_merak`、`xhquant` 或目标硬件可用；涉及真实推理的命令仅作为运行形态说明，未在本次整理中执行端到端推理。

## 1. 工程目标与模型目录契约

`vllm-merak` 是一个 vLLM 插件包。它不走 Hugging Face `AutoModel` 加载权重，而是把 Merak/hmonnx 导出的模型目录包装成 vLLM V1 engine 可调度的模型，并把请求最终交给 `xhmodel_merak.xh_llm.AutoLLMHONNXModel` 执行。

一个可被当前后端识别的模型目录至少需要包含：

```text
<model>/
├── merak_config.json
├── golden_meta_info.json
└── hf_config/
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json 或同等 tokenizer 资产
    └── generation_config.json（可选）
```

关键契约如下：

| 文件 | 关键字段 | 用途 |
| --- | --- | --- |
| `merak_config.json` | `xh_model.model_type`、`xh_model.meta_info`（可选）、`enable_page_attention`（可选） | 触发 Merak config parser，并告诉 loader 这是 `hmonnx` 类型模型。 |
| `hf_config/config.json` | `architectures`、文本模型尺寸字段、tokenizer/多模态相关字段 | 提供源 Hugging Face architecture，供 Merak registry 映射到具体 loader。 |
| `golden_meta_info.json` | `kv_cache`、`model_config`、`visual_config`（多模态可选） | 提供 hmonnx runtime、KV cache 和视觉导出元信息。 |

`config_format` 和 `load_format` 应使用当前注册的 Merak 格式：

```python
LLM(
    model="/path/to/merak-model",
    config_format="merak_llm",
    load_format="merak_llm",
    trust_remote_code=True,
    enforce_eager=True,
)
```

> 注意：上面的代码片段是 API 形态说明。本次文档整理未运行真实 hmonnx 模型，因为缺少外部模型目录、runtime 包和目标硬件。

## 2. 总体架构与端到端调用链

## 2.1 关键代码索引（带行号）

以下索引用于把主文档结论快速对应到代码位置。行号基于当前工作区版本。

| 主题 | 代码位置 | 说明 |
|---|---|---|
| 插件入口 | `pyproject.toml:14-18` | 声明 `vllm.general_plugins` 与 `vllm.platform_plugins` |
| general plugin | `vllm_merak/__init__.py:5-12` | import config parser 并注册 architecture |
| platform plugin | `vllm_merak/platform.py:31-48` | 注册 vLLM ModelRegistry 与返回 platform class path |
| Merak config type | `vllm_merak/config.py:54` | `MERAK_MODEL_TYPE = "merak_llm"` |
| config 文件探测 | `vllm_merak/config.py:145-159` | 只探测 `merak_config.json`，不回退 `config.json` |
| HF/hmonnx meta 读取 | `vllm_merak/config.py:172-193` | 读取 `hf_config/config.json` 和 parser 侧 hmonnx meta |
| config merge | `vllm_merak/config.py:230-308` | 合并 HF config、Merak config、hmonnx meta，并改写 architecture |
| platform 改写 | `vllm_merak/platform.py:90-126` | world size、worker、eager、tokenizer、cache、LoRA 限制 |
| attention backend | `vllm_merak/platform.py:128-131` | 选择 `MerakAttentionBackend` |
| registry 自动导入 | `vllm_merak/model_executor/register.py:24-39` | glob 导入 `model_loader/*.py`，不是手写模块列表 |
| loader dispatch | `vllm_merak/model_executor/__init__.py:51-83` | 优先按 `source_architectures` 选择 Merak loader |
| worker 加载 | `vllm_merak/worker_v1/merak_worker_v1.py:85-99` | 加载模型并拉取 KV metadata |
| KV 内存估算 | `vllm_merak/worker_v1/merak_worker_v1.py:147-170` | `kv_cache_memory_bytes` / `VLLM_CPU_KVCACHE_SPACE` / fallback |
| runner forward | `vllm_merak/worker_v1/merak_model_runner_v1.py:87-180` | batch 拆 request、可选 page attention cache bridge、调用 `_run_request()` |
| Qwen3 hmonnx 加载 | `vllm_merak/model_executor/model_loader/qwen3.py:101-145` | 加载 runtime、解析 KV cache、注册 placeholder attention |
| Qwen3.5 hmonnx 加载 | `vllm_merak/model_executor/model_loader/qwen3_5.py:468-535` | 多模态 loader 的 runtime 与 KV meta 初始化 |
| KV copy | `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py:96-207` | paged ↔ contiguous KV cache copy |
| placeholder attention | `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py:344-377` | 创建 vLLM attention 占位层 |
| runtime forward | `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py:382-467` | prefill/decode、chunking、data_preprocessor、runtime forward |
| attention metadata | `vllm_merak/attention/backends/merak.py:18-136` | metadata 字段与 builder |
| attention KV shape | `vllm_merak/attention/backends/merak.py:139-175` | custom backend 与 KV shape `(2, num_blocks, num_kv_heads, block_size, head_size)` |
| Qwen3.5 processor | `vllm_merak/model_executor/model_loader/qwen3_5.py:193-247` | 多模态 processor 与 processing info |
| Qwen3.5 image embeddings | `vllm_merak/model_executor/model_loader/qwen3_5.py:626-695` | `visual.forward()` 生成 image embeddings |
| Qwen3.5 request image kwargs | `vllm_merak/model_executor/model_loader/qwen3_5.py:730-767` | request 级 image kwargs 抽取 |

一次请求的主链路可以概括为：

```text
vLLM entrypoints / OpenAI server / LLM.generate
  ↓
pyproject entry points
  ↓
vllm_merak.register_plugins()
  ↓
MerakConfigParser(config_format="merak_llm")
  ↓
MerakPlatform.check_and_update_config()
  ↓
MerakWorkerV1
  ↓
MerakModelRunnerV1
  ↓
get_model() + MERAK_MODEL_REGISTRY
  ↓
Qwen3MerakForCausalLM / Qwen3_5MerakForCausalLM
  ↓
AutoLLMHONNXModel.from_pretrained(golden_meta_info.json)
  ↓
hmonnx_model.forward(...)
```

这条链路分成三层：

1. **vLLM 接入层**：插件入口、config parser、platform plugin，让 vLLM 能发现 Merak 后端。
2. **调度适配层**：`MerakWorkerV1`、`MerakModelRunnerV1`、`MerakAttentionBackend`，复用 vLLM V1 scheduler、sampling 和 KV cache 管理。
3. **runtime 执行层**：具体 model loader 创建 hmonnx runtime，按 request 拆 batch，执行 prefill/decode，并在 vLLM paged KV cache 与 runtime contiguous KV cache 之间同步。

## 3. 插件注册链路

插件入口定义在 `pyproject.toml`：

```toml
[project.entry-points."vllm.general_plugins"]
merak = "vllm_merak:register_plugins"

[project.entry-points."vllm.platform_plugins"]
merak = "vllm_merak.platform:register"
```

注册分两步：

1. `vllm_merak.register_plugins()` 导入 `vllm_merak.config`，触发 `@register_config_parser(MERAK_MODEL_TYPE)`，并调用 `register_architecture()`。
2. `vllm_merak.platform.register()` 返回 `vllm_merak.platform.MerakPlatform`，使 vLLM 能选择 Merak platform。

`register_architecture()` 会读取 `MERAK_MODEL_REGISTRY` 中已经注册的 loader，把具体 Merak loader 类名注册到 vLLM `ModelRegistry`。这里注册的是 Merak 类名，例如 `Qwen3MerakForCausalLM`、`Qwen3_5MerakForCausalLM`，不是源 HF architecture 名称。

## 4. 配置解析链路

`vllm_merak/config.py` 定义：

```python
MERAK_MODEL_TYPE = "merak_llm"
```

`MerakConfigParser` 被注册到该格式。当 vLLM 以 `config_format="merak_llm"` 调用 config parser 时，解析行为如下：

1. `_locate_merak_config_file(model)` 只接受目录下存在 `merak_config.json` 的模型路径。
2. 如果没有 `merak_config.json`，parser 委托给 vLLM 原生 `HFConfigParser`。这用于多模态 processor 或 generation config 读取 `hf_config/` 目录时避免误走 Merak schema。
3. `_merge_merak_config()` 合并三类信息：
   - `hf_config/config.json` 的源 HF 配置；
   - `merak_config.json` 的 Merak 后端配置；
   - `golden_meta_info.json` 的导出元信息。
4. 合并后构造 `MerakConfig`，并设置 `_name_or_path` 为模型目录。

### architecture 映射

当前实现保留源 HF architecture，并把 `architectures` 改写为已注册的 Merak loader 类名：

```text
hf_config/config.json.architectures
  ↓
source_architectures
  ↓
MERAK_MODEL_REGISTRY.get(source_architecture)
  ↓
Merak loader class name
```

已覆盖的映射包括：

| 源 HF architecture | Merak loader |
| --- | --- |
| `Qwen3ForCausalLM` | `Qwen3MerakForCausalLM` |
| `Qwen2ForCausalLM` | 代码中部分 loader 校验允许该值，但当前没有独立 registry 映射；按现状不能只靠该 architecture 自动选到 loader。 |
| `Qwen3_5ForConditionalGeneration` | `Qwen3_5MerakForCausalLM` |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MerakForCausalLM` |

如果源 architecture 找不到注册项，config parser 会回退到 `MerakForCausalLM`。但 `model_executor.get_model()` 运行时不接受只有泛型 `MerakForCausalLM` 的配置；它会要求 `source_architectures` 或具体 Merak loader 存在。因此新模型不能只依赖默认值，必须注册明确映射。

## 5. Platform 配置改写

`MerakPlatform` 以 CPU platform 的形式接入 vLLM：

```python
_enum = PlatformEnum.CPU
device_type = "cpu"
dist_backend = "gloo"
```

这不表示 hmonnx runtime 只能在 CPU 上运行。它的含义是：vLLM 调度层、worker 设备声明和 KV cache 管理按 CPU backend 接入；具体 hmonnx runtime 在 loader 内部根据 `torch.cuda.is_available()`、runtime device map 和环境变量移动到 `cuda:0` 或 runtime 自己选择的主设备。

`MerakPlatform.check_and_update_config()` 负责关键改写：

- 限制 `world_size == 1`，当前只支持单进程/单 socket 形态。
- 当 `parallel_config.worker_cls == "auto"` 时替换为 `vllm_merak.worker_v1.merak_worker_v1.MerakWorkerV1`。
- 强制 eager：如果 `enforce_eager=False`，记录 warning 并改为 `True`。
- 如果模型目录有 `hf_config/`，把 `model_config.hf_config_path` 和 tokenizer 路径指向该目录。
- 强制 KV cache dtype 为 `float16`，block size 为 `MERAK_PREFERRED_BLOCK_SIZE = 128`。
- 禁用 LoRA。
- `get_attn_backend_cls()` 固定返回 `MerakAttentionBackend`。

## 6. Worker / ModelRunner 执行链路

### MerakWorkerV1

`MerakWorkerV1` 继承 vLLM V1 `WorkerBase`，负责：

- 初始化 gloo 分布式环境和 model parallel 状态。
- 创建 `MerakModelRunnerV1`。
- 调用 runner 加载模型。
- 从加载后的模型读取 `key_cache_config`、`value_cache_config` 和 `cache_dtype`。
- 估算或读取 KV cache 可用空间。
- 把 vLLM 的 `KVCacheConfig` 下发给 runner。

KV cache 空间优先级为：

1. `cache_config.kv_cache_memory_bytes` 显式配置；
2. `VLLM_CPU_KVCACHE_SPACE`，单位 GiB；
3. fallback：按模型 cache block 大小和最大上下文估算，但至少 4 GiB。

### MerakModelRunnerV1

`MerakModelRunnerV1` 继承 `CPUModelRunner`，但重写了 Merak 相关入口：

- `load_model()` 调用 `vllm_merak.model_executor.get_model()`。
- `initialize_kv_cache()` 先让父类初始化 vLLM KV cache，再调用 hmonnx runtime mixin 的 `prepare_kv_cache()` 和 `prepare_other_cache()`。
- `_model_forward()` 是每步推理的核心：
  1. 从 vLLM forward context 读取 `MerakAttentionMetadata`。
  2. 拉平成 token 或 embedding 序列。
  3. 预分配 `[num_tokens, vocab_size]` 的 logits。
  4. 根据 `subsequence_begins`、`past_lens`、`block_indices_begins`、`block_indices` 按 request 拆分 batch。
  5. 对每个 request 单独调用 loader 的 `_run_request()`。
  6. 只把每个 request 最后一个 token 的 logits 写回 `flat_logits[query_end - 1]`，交给 vLLM sampling。

当前实现本质上是“vLLM 负责 batch 调度，Merak runtime 按 request 串行执行”。这降低了接入复杂度，但也意味着高并发性能会受 Python 循环、cache copy 和 runtime 单请求调用影响。

## 7. hmonnx runtime 加载与执行

文本和多模态 loader 都通过同一类 runtime API 加载：

```python
AutoLLMHONNXModel.from_pretrained(
    hmonnx_meta_file,
    **get_hmonnx_runtime_kwargs(),
)
```

`hmonnx_meta_file` 当前固定解析为模型目录下的 `golden_meta_info.json`。`merak_config.json` 中的 `xh_model.meta_info` 会被 config parser 用于读取 meta 信息，但 loader 当前仍直接拼接 `golden_meta_info.json`；如果未来需要支持自定义 meta 文件名，应同步修改 loader。

runtime 环境变量由 `vllm_merak/utils.py` 处理：

| 环境变量 | 行为 |
| --- | --- |
| `VLLM_MERAK_USE_V2=1` | 设置 `ENABLE_HMINFERENCE_V2=1`。 |
| `VLLM_MERAK_AUTO_OFFLOAD=1` | 向 runtime 传 `enable_auto_offload=True`，或在非 V2 场景调用 `enable_auto_offload()`。 |
| `VLLM_MERAK_CUDA_GRAPH=1` | 向 runtime 传 `enable_cuda_graph=True`。 |
| 三者同时开启 | 显式传入可见 GPU 的 `device_map`。 |

加载后，loader 会：

1. 根据是否保留 runtime device map，决定是否调用 `hmonnx_model.to(self.inference_device)`。
2. 通过 `resolve_hmonnx_primary_device()` 更新 `self.inference_device`。
3. 移动 embeddings、KV cache mixin、data processor 等辅助 tensor。
4. 读取 `golden_meta_info.json.kv_cache`，生成 vLLM placeholder attention 需要的 cache shape。
5. 注册 placeholder attention layers、logits processor 和 sampler。

### prefill / decode 选择

`MerakForCausalLM._run_request()` 根据当前 request token 数和历史长度选择 runtime 模式：

```text
num_tokens > 1 或 past_len == 0 → runtime_model.set_prefill()
否则 → runtime_model.set_decode()
```

随后按 runtime 支持的最大输入长度切 chunk：

```text
max_chunk_len = runtime_model.get_input_sequence_length()
```

每个 chunk 都带 `past_seq_length = past_len + chunk_start`。如果使用 raw `input_ids`，传入 `input_ids`；否则先通过 embedding 层得到 `inputs_embeds`。多模态首个 prefill chunk 还会携带 `image_embeds`、`image_grid_thw` 和 `video_grid_thw`。

### 长 prompt 风险

长 prompt 会被按 `max_chunk_len` 切成多个 runtime forward。文本场景可以依赖 `past_seq_length` 和 runtime cache 逐段推进；多模态场景额外要求 image placeholder token 出现在第一个 runtime chunk 内，否则会抛出错误。导出模型时如果 prefill 序列长度太小，图片 token 可能越界。

## 8. KV cache 适配机制

vLLM V1 使用 paged KV cache，形状由 attention backend 定义。`MerakAttentionBackend.get_kv_cache_shape()` 返回：

```text
(2, num_blocks, num_kv_heads, block_size, head_size)
```

其中第一维 `2` 分别表示 key/value。hmonnx runtime mixin 维护的是每层一份 contiguous cache，例如 `[batch, heads, seq, dim]` 形态。当前工程通过三件事桥接两套 cache：

1. **placeholder attention layers**：`_register_placeholder_attention_layers()` 按源 HF 文本配置创建 vLLM `Attention` 层，但这些层不执行真实 attention，只用于让 vLLM 分配 paged KV cache 并生成 metadata。
2. **MerakAttentionMetadata**：`MerakAttentionMetadataBuilder` 从 vLLM `CommonAttentionMetadata` 生成 Merak 需要的字段：
   - `past_lens`：每个 request 已经计算的 token 数；
   - `subsequence_begins`：当前 batch 内每个 request token slice 的起止；
   - `block_indices`：物理 block id；
   - `block_indices_begins`：每个 request 在 `block_indices` 中的起止；
   - `max_context_len` 和 `sampled_token_indices`。
3. **paged ↔ contiguous copy**：
   - `_load_runtime_request_cache_from_paged(block_ids, past_len)` 在 decode 或续写前，把 vLLM paged cache 中 `[0, past_len)` 拷到 runtime contiguous cache。
   - `_scatter_runtime_request_cache_to_paged(block_ids, start_pos, end_pos)` 在 runtime forward 后，把新增 `[past_len, past_len + query_len)` 拷回 vLLM paged cache。

复制逻辑要求：

- `golden_meta_info.json.kv_cache.cache_axis == 2`，即序列维在第 2 轴；其他轴当前直接 `NotImplementedError`。
- block table 至少覆盖 `end_pos` 所需 block 数，否则抛 `ValueError`。
- `MERAK_PREFERRED_BLOCK_SIZE` 当前固定为 `128`，platform 会把 vLLM cache block size 强制改成该值。

如果 `enable_page_attention` 为真，`MerakModelRunnerV1._model_forward()` 会显式执行 paged/cache 互拷；基类 `MerakForCausalLM.forward()` 也包含同类逻辑。维护时需要避免两条路径行为分叉。

## 9. 多模态适配机制

Qwen3.5 多模态 loader (`Qwen3_5MerakForCausalLM`) 比文本 loader 多做四类工作。

### 9.1 vLLM 多模态 processor 注册

类上同时注册：

```python
@MULTIMODAL_REGISTRY.register_processor(...)
@MERAK_MODEL_REGISTRY.register_module([...])
class Qwen3_5MerakForCausalLM(...):
    ...
```

这让 vLLM 在处理 chat/image 输入时使用 `MerakQwen3_5MultiModalProcessor`、`MerakQwen3_5ProcessingInfo` 和 `Qwen3VLDummyInputsBuilder`。

### 9.2 图片预处理对齐 hmonnx 导出尺寸

`MerakQwen3_5MultiModalProcessor._call_hf_processor()` 会读取 `golden_meta_info.json.visual_config` 或 `model_config.visual_config`，把输入图片 resize 到导出尺寸，然后额外生成 `hm_pixel_values`。`hm_pixel_values` 通过 hmonnx 的 `Qwen3_5ImageProcessor._hm_preprocess()` 生成，用于匹配 runtime 视觉塔输入。

### 9.3 视觉 embedding 生成与合并

`embed_multimodal()` 支持两条路径：

- 优先使用 `hm_pixel_values`，逐图调用 `self.inference_model.visual.forward(...)`，得到 image embeddings。
- 如果没有 `hm_pixel_values`，则使用 vLLM/HF processor 的 `pixel_values` 和 `image_grid_thw`，按 patch 数拆分后调用 visual tower。

随后 `embed_input_ids()` 使用 vLLM 的 `_merge_multimodal_embeddings()` 把 image embeddings 合并到文本 embeddings 中。

### 9.4 raw input_ids runtime 路径

多模态 runtime 配置会把 `_prefer_input_ids_runtime` 置为 `True`。因此 `_run_request()` 会优先把 raw `input_ids` 交给 hmonnx data preprocessor，并在首个 prefill chunk 附带 `image_embeds` 和 `image_grid_thw`。当前支持 image，不支持 video；如果检测到 video token，会抛 `NotImplementedError`。

## 10. 新模型架构接入方法

新增模型时先判断它是文本模型还是多模态模型。两者共享基础 contract，但多模态还要补 processor、视觉预处理和 placeholder token 逻辑。

### 10.1 新文本模型 checklist

- [ ] 模型目录包含 `merak_config.json`，且 `xh_model.model_type` 为 `hmonnx`。
- [ ] 模型目录包含 `hf_config/config.json`，其中 `architectures[0]` 是源 HF architecture。
- [ ] 模型目录包含 `golden_meta_info.json`，其中 `kv_cache` 至少包含：
  - [ ] `kv_cache_shape`
  - [ ] `num_layers`
  - [ ] `cache_dtype`（可选，默认 `float16`）
  - [ ] `cache_axis`（可选，但当前必须等于 `2`）
- [ ] 新建或复用 `vllm_merak/model_executor/model_loader/*.py` loader。
- [ ] 使用 `@MERAK_MODEL_REGISTRY.register_module("SourceHFArchitecture")` 注册源 architecture 到 Merak loader。
- [ ] loader 初始化时校验 `model_config.hf_config` 是 `MerakConfig`。
- [ ] loader 调用 `AutoLLMHONNXModel.from_pretrained(golden_meta_info.json, **get_hmonnx_runtime_kwargs())`。
- [ ] loader 读取 `kv_cache` 元信息，设置：
  - [ ] `self.key_cache_config`
  - [ ] `self.value_cache_config`
  - [ ] `self.cache_dtype`
  - [ ] `self.cache_axis`
  - [ ] `self.kv_cache_shape`
- [ ] loader 调用 `_register_placeholder_attention_layers(hf_model_config)`。
- [ ] loader 设置 `self.input_embeddings`、`self.logits_processor`、`self.sampler`。
- [ ] 如果模型 attention 层不是每层 full attention，确认 `hf_model_config.layer_types` 能正确筛选 placeholder attention layers。

### 10.2 新多模态模型 checklist

在文本 checklist 基础上增加：

- [ ] 在 `MERAK_MODEL_REGISTRY` 注册多模态源 HF architecture。
- [ ] 在 `MULTIMODAL_REGISTRY` 注册 processor、processing info 和 dummy input builder。
- [ ] 明确 placeholder token：`image_token_id`、`video_token_id`、`vision_start_token_id`、`vision_end_token_id`。
- [ ] 从 `golden_meta_info.json.model_config` 或 runtime data preprocessor 同步视觉 token 和导出参数。
- [ ] 明确 `visual_config` 的来源和字段名，例如 `image_size_w`、`image_size_h`、`patch_size`、`temporal_patch_size`、`spatial_merge_size`。
- [ ] 在 processor 中把输入图片 resize 到 hmonnx 导出尺寸。
- [ ] 生成 runtime 需要的额外视觉输入，例如当前 Qwen3.5 的 `hm_pixel_values`。
- [ ] 实现 `embed_multimodal()`，调用 runtime visual tower 并返回 vLLM 可合并的 embeddings。
- [ ] 实现或复用 `_extract_request_multimodal_kwargs()`，按 request 从 batch 级 `image_grid_thw` 中切出当前 request 的图像信息。
- [ ] 如果 runtime 必须接收 raw token，设置 `_prefer_input_ids_runtime=True`。
- [ ] 明确是否支持 video；如果不支持，应像当前实现一样显式抛错。

### 10.3 最小 loader 模板

```python
@MERAK_MODEL_REGISTRY.register_module("NewHFArchitecture")
class NewMerakForCausalLM(MerakForCausalLM):
    def __init__(self, *, vllm_config, kv_cache_dtype="auto", prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError("NewMerakForCausalLM expects MerakConfig")

        # 1. 定位模型目录和 hf_config。
        # 2. 加载 AutoLLMHONNXModel.from_pretrained(golden_meta_info.json, ...)。
        # 3. 读取 golden_meta_info.json.kv_cache。
        # 4. 设置 cache 元信息、placeholder attention、embedding、logits_processor、sampler。
```

> 该模板只描述接入骨架，不能直接运行。新增模型时应按具体 hmonnx runtime 输入/输出、KV cache shape 和 tokenizer contract 补全。

## 11. 验证清单

由于真实推理依赖外部模型和硬件，建议分层验证。

### 11.1 静态配置验证

- [ ] `pyproject.toml` entry points 被安装环境识别。
- [ ] `config_format="merak_llm"` 能解析包含 `merak_config.json` 的目录。
- [ ] `hf_config/config.json.architectures` 能映射到具体 Merak loader。
- [ ] `MerakConfig.source_architectures` 保留源 HF architecture。
- [ ] `max_position_embeddings` 被 `golden_meta_info.json.model_config.context_max_length` 正确截断。

### 11.2 模型加载验证

- [ ] `MerakPlatform.check_and_update_config()` 把 worker 改为 `MerakWorkerV1`。
- [ ] loader 能找到 `hf_config/config.json` 和 `golden_meta_info.json`。
- [ ] `AutoLLMHONNXModel.from_pretrained(...)` 成功。
- [ ] `kv_cache.num_layers` 与 placeholder attention layer 数匹配。
- [ ] `cache_axis == 2`。
- [ ] `input_embeddings`、`logits_processor`、`sampler` 均初始化完成。

### 11.3 运行时验证

- [ ] 单 prompt prefill：`past_len == 0`，runtime 进入 `set_prefill()`。
- [ ] 单 token decode：`num_tokens == 1` 且 `past_len > 0`，runtime 进入 `set_decode()`。
- [ ] 长 prompt：超过 `runtime_model.get_input_sequence_length()` 时能分 chunk 推进。
- [ ] batch：多个 request 的 `subsequence_begins` 和 `block_indices_begins` 切片正确。
- [ ] KV cache：decode 前能从 paged cache 拷到 runtime cache，forward 后能拷回 paged cache。
- [ ] sampling：只有每个 request 最后一个 token logits 被用于采样。

### 11.4 多模态验证

- [ ] processor 能把图片 resize 到导出尺寸。
- [ ] `hm_pixel_values` shape 满足 runtime visual tower。
- [ ] `image_grid_thw` 按 request 正确切分。
- [ ] image placeholder token 全部出现在第一个 runtime chunk 内。
- [ ] video 输入触发明确错误，而不是静默走错路径。

### 11.5 OpenAI server 验证

示例入口在 `examples/vllm_chat_server.py` 和 `examples/vllm_server_start.sh`。真实环境中建议验证：

- [ ] server 能用 `--config-format merak_llm --load-format merak_llm` 启动。
- [ ] `/v1/chat/completions` 文本请求成功。
- [ ] 多模态模型的 image chat 请求成功。
- [ ] tool calling 参数不会破坏 Merak 后端加载。
- [ ] 进程退出时无残留 distributed process group。

## 12. 当前限制、风险与重构建议

### 12.1 当前限制

- 仅支持 vLLM V1 generation 路径；pooling 未实现。
- `MerakPlatform` 限制 `world_size == 1`，LoRA 不支持。
- KV cache 只支持 `cache_axis == 2`。
- cache block size 固定为 `128`，与导出/runtime 假设耦合。
- 多模态当前支持 image，不支持 video。
- `golden_meta_info.json` 在 loader 中被硬编码；`merak_config.json.xh_model.meta_info` 只在 config merge 阶段生效。
- batch 在 runtime 层按 request 串行执行，不是真正的 hmonnx batch forward。
- CPU platform 形态与 runtime CUDA/hmonnx 设备并存，排障时容易误判设备路径。

### 12.2 性能风险

- 每个 request 都可能执行 paged ↔ contiguous KV cache copy，decode 场景会放大拷贝开销。
- `_model_forward()` 中按 request 串行调用 runtime，batch 越大 Python 循环开销越明显。
- 长 prompt chunking 依赖 runtime cache 正确维护 `past_seq_length`；如果 runtime preprocessor 对 chunk 语义有额外限制，需要单独验证。
- 多模态视觉 embedding 在 Python 层逐图处理，图片数量增加时可能成为瓶颈。

### 12.3 排障关注点

- 如果 vLLM 报找不到 architecture，先检查 `hf_config/config.json.architectures` 和 `MERAK_MODEL_REGISTRY.register_module(...)`。
- 如果 tokenizer 或 processor 读取失败，检查 `MerakPlatform` 是否把 tokenizer/hf_config path 指向 `<model>/hf_config`。
- 如果 KV cache shape 报错，检查 `golden_meta_info.json.kv_cache`、`cache_axis` 和 block size。
- 如果 decode 输出异常，重点检查 `_load_runtime_request_cache_from_paged()` 与 `_scatter_runtime_request_cache_to_paged()` 的区间 `[start_pos, end_pos)`。
- 如果多模态首轮失败，检查 image placeholder 是否落在第一个 runtime chunk，以及 `hm_pixel_values` 是否按导出尺寸生成。

### 12.4 重构建议

- 统一 `MerakModelRunnerV1._model_forward()` 和 `MerakForCausalLM.forward()` 中重复的 request loop，减少两条执行路径分叉。
- 让 loader 复用 `merak_config.json.xh_model.meta_info`，不要硬编码 `golden_meta_info.json`。
- 抽象通用 hmonnx loader 基类，把 Qwen3 与 Qwen3.5 共享的 runtime 加载、KV 元信息读取、device 迁移逻辑收敛到一处。
- 把多模态导出元信息解析封装成独立 helper，并为字段缺失提供更清晰的错误信息。
- 为 KV cache copy 增加小规模单元测试，覆盖 block 边界、跨 block prompt、decode 单 token 和 block table 不足。
- 为 architecture registry 增加静态测试，确保每个支持的 HF architecture 都能映射到 concrete Merak loader。

## 17. 已知边界澄清

为了避免误读，当前版本需要特别澄清以下边界：

- `config_format` 的名字是 `merak_llm`，不是 `merak`。
- `MerakConfigParser` 只用 `merak_config.json` 判断是否进入 Merak 解析；它不会把普通 `config.json` 当作 Merak 配置入口。
- parser 侧 `_load_hmonnx_meta_info()` 会读取 `xh_model.meta_info` 指定的文件名；但 Qwen3/Qwen3.5 loader 真正调用 `AutoLLMHONNXModel.from_pretrained()` 时硬编码 `config_dir / "golden_meta_info.json"`。
- paged ↔ runtime KV cache copy 不是无条件执行；`MerakModelRunnerV1._model_forward()` 中由 `enable_page_attention` 控制。
- 新 loader 文件由 `register.py` glob 导入 `model_loader/*.py`，当前没有需要维护的 `_MERAK_MODEL_LOADER_MODULES` 列表。
- 代码里 Qwen3 loader 的 architecture allowlist 出现了 `Qwen2ForCausalLM`，但当前 registry 没有看到 `@MERAK_MODEL_REGISTRY.register_module("Qwen2ForCausalLM")`，因此不能把 Qwen2 描述为已完整注册支持。
- video placeholder 和字段存在，但 Qwen3.5 runtime 路径对 video 显式抛 `NotImplementedError`。
- vLLM 层 CUDA graph 不支持，platform 会强制 eager；`VLLM_MERAK_CUDA_GRAPH` 只表示传给 hmonnx runtime 的选项。
- 该后端不是 CPU-only 执行。它在 vLLM platform 形态上声明为 CPU，但 loader 会在可用时把 hmonnx runtime 放到 CUDA 或保留 runtime 自己的 device map。
- 运行真实推理验证依赖外部 hmonnx 模型目录、`xhmodel_merak`、`xhquant` 以及目标硬件/运行时环境，本文档只基于当前仓库静态分析。
