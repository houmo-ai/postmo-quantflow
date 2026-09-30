# vLLM Merak Qwen3.5 推理接入与源码阅读笔记

本文说明 `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py` 和 `vllm_merak/model_executor/model_loader/qwen3_5.py` 的职责、推理链路、cache 管理方式，以及它们如何接入 vLLM。

适合按源码阅读时对照使用。

---

## 1. 总体架构

`vllm-merak` 并不是让 vLLM 直接执行 PyTorch/HuggingFace Qwen 模型，而是让 vLLM 负责服务框架、调度和采样，真正的模型计算交给 HMONNX runtime。

整体链路如下：

```text
OpenAI API / LLM.generate
        ↓
vLLM scheduler / block manager / sampler
        ↓
MerakPlatform + MerakWorkerV1
        ↓
Qwen3_5MerakForCausalLM.forward()
        ↓
HMONNX runtime: AutoLLMHONNXModel
        ↓
返回 logits 给 vLLM sampler
```

核心分工：

| 模块 | 职责 |
|---|---|
| vLLM | 请求调度、batching、block table、OpenAI API、采样、tool call parser |
| Merak backend | 注册平台、注册模型类、构造 attention metadata |
| `qwen3_5.py` | 加载 Qwen3.5 HMONNX 模型、读取导出 metadata、处理多模态配置 |
| `merak_for_causal_lm.py` | 通用推理逻辑：拆 vLLM batch、搬 KV cache、调用 HMONNX runtime、回填 logits |

一个关键点：

> Merak 在 vLLM 侧表现为 CPU 风格 backend，但 HMONNX runtime 实际仍然可以跑在 GPU 上。

也就是说，`MerakPlatform.device_type = "cpu"` 主要是为了让 vLLM 调度走 CPU backend 路径，避免走 vLLM 自带 CUDA attention kernel；真正的 HMONNX runtime 通过 `.to(cuda)`、auto-offload、cuda graph 等机制在 GPU 上执行。

---

## 2. vLLM 如何接入 Merak

### 2.1 插件入口：`vllm_merak/__init__.py`

```python
def register_plugins() -> None:
    import vllm_merak.config
    from vllm_merak.platform import register_architecture
    register_architecture()
```

这里做两件事：

1. import `vllm_merak.config`，注册自定义 config parser。
2. 调用 `register_architecture()`，把 Merak 模型类注册到 vLLM 的 `ModelRegistry`。

---

### 2.2 自定义配置：`vllm_merak/config.py`

Merak 定义了自定义 config format：

```python
MERAK_MODEL_TYPE = "merak_llm"
```

因此启动服务时需要：

```bash
--config-format merak_llm
--load-format merak_llm
```

vLLM 解析模型配置时会走：

```python
@register_config_parser(MERAK_MODEL_TYPE)
class MerakConfigParser(ConfigParserBase):
```

这个 parser 负责读取并合并：

```text
<model_dir>/merak_config.json
<model_dir>/hf_config/config.json
<model_dir>/golden_meta_info.json
```

最关键的逻辑是：

```python
merged_config["architectures"] = [
    _resolve_registered_architecture(
        source_architectures,
        hmonnx_meta_info,
    )
]
merged_config["model_type"] = MERAK_MODEL_TYPE
```

原始 HF config 中可能是：

```json
"architectures": ["Qwen3_5ForConditionalGeneration"]
```

`MerakConfigParser` 会根据这个 architecture 找到对应的 Merak loader class，最后告诉 vLLM：

```text
真正要实例化的是 Qwen3_5MerakForCausalLM
```

---

### 2.3 平台注册：`vllm_merak/platform.py`

`MerakPlatform` 伪装成 CPU backend：

```python
class MerakPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name = "xh2"
    device_type = "cpu"
    dispatch_key = "CPU"
    dist_backend = "gloo"
```

在 `check_and_update_config()` 中，Merak 会修改 vLLM config：

```python
parallel_config.worker_cls = "vllm_merak.worker_v1.merak_worker_v1.MerakWorkerV1"
```

这表示 vLLM worker 不用默认 worker，而使用 Merak 自己的 worker。

同时它会设置 cache 配置：

```python
cache_config.cache_dtype = "float16"
cache_config.block_size = MERAK_PREFERRED_BLOCK_SIZE
cache_config.user_specified_block_size = True
```

注意：

```python
logger.warning("CUDA graph is not supported on Merak backend, fallback to the eager mode.")
```

这里说的是 vLLM 侧 CPU platform 不支持 vLLM 自己的 CUDA graph，不代表 HMONNX runtime 不能用 cuda graph。你启动时传的：

```bash
--cuda-graph
```

是给 HMONNX runtime 用的。

---

## 3. `qwen3_5.py` 的职责

`qwen3_5.py` 是 Qwen3.5 的具体模型 loader。

核心类：

```python
@MERAK_MODEL_REGISTRY.register_module([
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration"
])
class Qwen3_5MerakForCausalLM(MerakForCausalLM):
```

含义是：

> 如果导出模型的 HF architecture 是 `Qwen3_5ForConditionalGeneration` 或 `Qwen3_5MoeForConditionalGeneration`，则使用 `Qwen3_5MerakForCausalLM` 加载。

---

## 4. `Qwen3_5MerakForCausalLM.__init__` 加载流程

### 4.1 获取 vLLM model config

```python
if vllm_config is not None:
    model_config = vllm_config.model_config
```

vLLM 实例化模型时会传入 `VllmConfig`，其中包含：

- model path
- hf_config
- cache_config
- parallel_config
- tokenizer config
- max_model_len
- enforce_eager 等

---

### 4.2 要求 `hf_config` 必须是 `MerakConfig`

```python
config = model_config.hf_config
if not isinstance(config, MerakConfig):
    raise TypeError(...)
```

这就是为什么启动时必须使用：

```bash
--config-format merak_llm
```

否则 vLLM 会按普通 HF config 解析，后续不会进入 Merak 的加载路径。

---

### 4.3 定位模型目录

```python
model_path = getattr(model_config, "model", None)
name_or_path = getattr(config, "_name_or_path", None)
```

然后找到：

```text
<model_dir>/hf_config/config.json
<model_dir>/golden_meta_info.json
```

---

### 4.4 读取 HF architecture

```python
hf_config_file = hf_config_dir / "config.json"
hf_config = json.load(hf_config_file.open("r", encoding="utf-8"))
architectures = hf_config.get("architectures") or []
architecture = str(architectures[0])
self.architecture_spec = _resolve_merak_architecture_spec(architecture)
```

对于 Qwen3.5，多模态 spec 是：

```python
_MERAK_QWEN3_5_MULTIMODAL_SPEC = _MerakArchitectureSpec(
    source_architectures=frozenset({
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }),
    supports_multimodal=True,
    supports_multimodal_raw_input_only=True,
    requires_raw_input_tokens=True,
    supports_mrope=True,
)
```

这会影响后续是否走 raw input ids、多模态处理和 M-RoPE。

---

### 4.5 加载 HMONNX runtime

最核心代码：

```python
hmonnx_meta_file = str(config_dir / "golden_meta_info.json")
hmonnx_model = AutoLLMHONNXModel.from_pretrained(
    hmonnx_meta_file,
    **get_hmonnx_runtime_kwargs(),
)
apply_hmonnx_runtime_options(hmonnx_model)
```

也就是说，真正模型不是 HF 权重，而是从导出的：

```text
golden_meta_info.json
```

加载 HMONNX runtime。

`get_hmonnx_runtime_kwargs()` 和 `apply_hmonnx_runtime_options()` 会读取启动参数或环境配置，例如：

```bash
--use-v2
--auto-offload
--cuda-graph
```

这些参数影响的是 HMONNX runtime。

然后：

```python
if not preserve_runtime_device_map:
    hmonnx_model.to(self.inference_device)
self.inference_device = resolve_hmonnx_primary_device(hmonnx_model, self.inference_device)
self.inference_model = hmonnx_model
```

通常 `self.inference_device` 是：

```text
cuda:0
```

如果启动时设置：

```bash
CUDA_VISIBLE_DEVICES=6
```

则这里的 `cuda:0` 实际对应物理 GPU 6。

---

### 4.6 读取 KV cache metadata

```python
kv_cache_meta = hmonnx_meta_info["kv_cache"]
kv_cache_shape = list(kv_cache_meta["kv_cache_shape"])
num_layers = int(kv_cache_meta["num_layers"])
cache_dtype = kv_cache_meta.get("cache_dtype", "float16")
```

然后保存：

```python
self.num_layers = num_layers
self.kv_cache_shape = tuple(kv_cache_shape)
self.cache_axis = int(kv_cache_meta.get("cache_axis", 2))
self.max_cache_tokens = int(self.kv_cache_shape[self.cache_axis])
```

Merak 当前假设 cache 的 sequence axis 是：

```python
cache_axis == 2
```

如果不是 2，会报错：

```python
if self.cache_axis != 2:
    raise NotImplementedError(...)
```

---

### 4.7 注册 placeholder attention layers

```python
hf_model_config = AutoConfig.from_pretrained(hf_config_dir)
if hasattr(hf_model_config, "text_config"):
    hf_model_config = hf_model_config.text_config

self._register_placeholder_attention_layers(hf_model_config)
```

这里非常重要。

Merak 不让 vLLM 的 Attention 层真正计算 attention。真正 attention 在 HMONNX runtime 里算。

但 vLLM 的 block manager、cache engine、attention metadata builder 需要看到 attention layer，才能：

- 分配 KV cache；
- 构建 block table；
- 构建 `MerakAttentionMetadata`。

因此这里创建了一组“占位 Attention 层”：

```python
Attention(
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_size=head_size,
    scale=scale,
    cache_config=self.vllm_config.cache_config,
    prefix=f"model.layers.{layer_idx}.attn",
    attn_backend=MerakAttentionBackend,
)
```

它们的作用不是算 attention，而是让 vLLM 分配和维护 KV cache。

对于 Qwen3.5，代码会优先只给 `full_attention` 层注册 placeholder：

```python
attn_layer_indices = [
    layer_idx
    for layer_idx, layer_type in enumerate(layer_types)
    if str(layer_type) == "full_attention"
]
```

这也是 linear attention cache 需要额外处理的原因：linear attention 不在 vLLM 标准 paged KV cache 中。

---

### 4.8 初始化 logits processor / embedding / sampler

```python
self.logits_processor = LogitsProcessor(hf_model_config.vocab_size, logits_as_input=True)
self.input_embeddings = cast(nn.Embedding, self.inference_model.get_input_embeddings())
self.sampler = Sampler()
```

注意：

```python
logits_as_input=True
```

因为 Merak 的 `forward()` 直接返回 logits，而不是 hidden states。

普通 vLLM 模型通常是：

```text
forward() -> hidden_states
compute_logits() -> logits
sample() -> token
```

Merak 是：

```text
forward() -> logits
compute_logits() -> 基本原样返回 logits
sample() -> vLLM sampler
```

---

## 5. `qwen3_5.py` 的多模态逻辑

虽然当前文本测试不一定用到，但 Qwen3.5 loader 支持多模态。

### 5.1 注册多模态 processor

```python
@MULTIMODAL_REGISTRY.register_processor(
    MerakQwen3_5MultiModalProcessor,
    info=MerakQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
```

这让 vLLM 知道 Qwen3.5 Merak 模型如何处理 image 输入。

---

### 5.2 图片 resize 到导出尺寸

```python
visual_config = _load_merak_visual_config(self.info.model_id)
target_width = int(visual_config.get("image_size_w", 448))
target_height = int(visual_config.get("image_size_h", 448))
```

导出模型如果固定为：

```text
448x448
```

则服务侧收到图片后也会 resize 到这个尺寸。

---

### 5.3 构造 `hm_pixel_values`

```python
processed_outputs["hm_pixel_values"] = _build_merak_hm_pixel_values(
    resized_images,
    self.info.model_id,
)
```

这是 Merak/HMONNX visual runtime 需要的图片输入格式，不完全等价于普通 HF Qwen-VL 的 `pixel_values`。

---

### 5.4 visual embedding

```python
visual = cast(Any, getattr(self.inference_model, "visual"))
visual_output = visual.forward(
    image_pixels.to(device=visual.device, dtype=visual.dtype),
)
```

然后将 visual output 归一化成 image embeddings，再和文本 embedding merge。

---

## 6. `merak_for_causal_lm.py` 的职责

这个文件是 Merak causal LM 的通用基类。

它的职责是：

```text
把 vLLM 的 batch/paged-cache 调度，翻译成 HMONNX runtime 的单请求/连续 cache 调用。
```

最重要的方法：

| 方法 | 作用 |
|---|---|
| `forward()` | vLLM 调用的模型入口 |
| `_run_request()` | 真正调用 HMONNX runtime |
| `_load_runtime_request_cache_from_paged()` | vLLM paged KV cache -> HMONNX contiguous KV cache |
| `_scatter_runtime_request_cache_to_paged()` | HMONNX contiguous KV cache -> vLLM paged KV cache |
| `_register_placeholder_attention_layers()` | 创建占位 attention，让 vLLM 分配 cache |
| `_ensure_runtime_cache_tensors()` | 确保 HMONNX runtime 的 cache tensor 已初始化 |

---

## 7. `MerakForCausalLM.forward()` 主流程

这是 vLLM 真正调用的入口。

简化逻辑：

```python
def forward(input_ids, positions, inputs_embeds, **kwargs):
    attn_metadata = self._get_attn_metadata()

    flat_input_ids = flatten(input_ids)
    flat_inputs_embeds = prepare embeddings

    flat_logits = zeros([num_tokens, vocab_size])

    for each request in vLLM batch:
        past_len = ...
        query_start, query_end = ...
        block_ids = ...

        load KV cache from vLLM paged cache to HMONNX contiguous cache

        last_token_logits = self._run_request(...)

        scatter HMONNX KV cache back to vLLM paged cache

        flat_logits[query_end - 1] = last_token_logits

    return flat_logits
```

---

### 7.1 获取 attention metadata

```python
attn_metadata = self._get_attn_metadata()
```

这个 metadata 来自 `MerakAttentionBackend`。

里面包含：

| 字段 | 含义 |
|---|---|
| `past_lens` | 每个 request 已经有多少历史 token |
| `subsequence_begins` | 当前 batch 中每个 request 的 token 起止位置 |
| `block_indices` | 每个 request 使用的 physical KV block id |
| `block_indices_begins` | 每个 request 在 `block_indices` 里的起止位置 |
| `max_context_len` | 当前 batch 最大上下文长度 |

例子：

```text
req0: 当前输入 5 tokens，past_len = 0
req1: 当前输入 1 token，past_len = 128
```

`forward()` 会按 request 拆开，分别调用 HMONNX runtime。

---

### 7.2 准备 embedding

```python
flat_inputs_embeds = self._prepare_request_embeds(
    input_ids=input_ids,
    inputs_embeds=inputs_embeds,
)
```

文本场景会走：

```python
self.embed_input_ids(flat_input_ids)
```

Qwen3.5 子类中实际使用的是 HMONNX runtime 提供的 embedding：

```python
self.input_embeddings = self.inference_model.get_input_embeddings()
```

---

### 7.3 初始化输出 logits

```python
flat_logits = torch.zeros(
    flat_inputs_embeds.shape[0],
    self.logits_processor.vocab_size,
    device=device,
    dtype=self.input_embeddings.weight.dtype,
)
```

形状是：

```text
[num_tokens, vocab_size]
```

但 Merak 只填每个 request 的最后一个 token：

```python
flat_logits[query_end - 1].copy_(last_token_logits)
```

其它 token 位置保持 0。

这是因为 vLLM sampler 只关心每个 request 的最后 token logits。

---

### 7.4 遍历 request

```python
num_reqs = int(attn_metadata.past_lens.shape[0])

for req_idx in range(num_reqs):
    past_len = int(past_lens_list[req_idx])
    query_start = int(subsequence_begins_list[req_idx])
    query_end = int(subsequence_begins_list[req_idx + 1])
    query_len = query_end - query_start
```

每个 request 都会被单独运行一次 HMONNX runtime。

---

### 7.5 取 block table

```python
block_begin = int(block_indices_begins_list[req_idx])
block_end = int(block_indices_begins_list[req_idx + 1])
block_ids = tuple(
    int(block_id)
    for block_id in block_indices_list[block_begin:block_end]
)
```

`block_ids` 是 vLLM 给当前 request 分配的 physical KV block。

例如：

```text
logical token 0-15   -> physical block 13
logical token 16-31  -> physical block 8
logical token 32-47  -> physical block 27
```

则：

```python
block_ids = (13, 8, 27)
```

---

## 8. vLLM paged cache 与 HMONNX contiguous cache

这是理解 `merak_for_causal_lm.py` 最关键的部分。

vLLM 的 paged KV cache 形状大致是：

```text
paged_cache:
[
  2,              # key/value
  num_blocks,
  num_kv_heads,
  block_size,
  head_dim,
]
```

代码中：

```python
key_blocks, value_blocks = paged_cache.unbind(0)
```

因此：

```text
key_blocks:   [num_blocks, num_kv_heads, block_size, head_dim]
value_blocks: [num_blocks, num_kv_heads, block_size, head_dim]
```

HMONNX runtime 的 cache 是连续序列形式：

```text
past_key_cache:
[1, num_kv_heads, max_seq_len, head_dim]
```

即：

```python
dst_key_cache[0, :, seq_start:seq_end, :]
```

---

## 9. `_copy_paged_to_contiguous()`

作用：

```text
vLLM paged KV cache -> HMONNX contiguous KV cache
```

核心逻辑：

```python
for logical_block_idx in range(start_block, required_blocks):
    physical_block_id = block_ids[logical_block_idx]

    dst_key_cache[0, :, copy_start:copy_end, :].copy_(
        key_blocks[physical_block_id, :, src_start:src_end, :]
    )
```

也就是根据 block table，把 vLLM 离散 block 里的 KV 还原成 HMONNX 需要的连续 `[0:past_len]`。

decode 前必须做这一步，否则 HMONNX runtime 看不到历史 KV。

---

## 10. `_copy_contiguous_to_paged()`

作用相反：

```text
HMONNX contiguous KV cache -> vLLM paged KV cache
```

HMONNX runtime 跑完当前 query 后，新的 KV 已经写入自己的 contiguous cache。

然后代码：

```python
self._scatter_runtime_request_cache_to_paged(
    block_ids,
    past_len,
    past_len + query_len,
)
```

只把新增区间写回 vLLM paged cache。

例如 decode 一个 token：

```text
past_len = 128
query_len = 1
```

只回写：

```text
[128:129]
```

---

## 11. `_run_request()` 真正调用 HMONNX

`forward()` 是 vLLM 侧 wrapper，`_run_request()` 才是真正执行模型。

核心变量：

```python
runtime_model = self.inference_model
runtime_dtype = self.input_embeddings.weight.dtype
runtime_device = self.inference_device
```

---

### 11.1 判断传 `input_ids` 还是 `inputs_embeds`

```python
use_raw_input_ids = flat_input_ids is not None and (
    self._prefer_input_ids_runtime
    or image_embeds is not None
    or image_grid_thw is not None
    or video_grid_thw is not None
    or inputs_embeds is None
)
```

文本模型可以传 embedding。

多模态模型通常需要传 raw input ids，因为 HMONNX data_preprocessor 需要根据 image token、grid 信息等做处理。

Qwen3.5 多模态中：

```python
self._prefer_input_ids_runtime = True
```

所以通常会走 raw input ids 路径。

---

### 11.2 判断 prefill / decode

```python
use_prefill = num_tokens > 1 or past_len == 0
if use_prefill:
    runtime_model.set_prefill()
else:
    runtime_model.set_decode()
```

| 场景 | 条件 | runtime mode |
|---|---|---|
| prompt 首次输入 | `past_len == 0` | prefill |
| 多 token chunk | `num_tokens > 1` | prefill |
| 单 token decode | `past_len > 0 and num_tokens == 1` | decode |

---

### 11.3 chunk 执行

```python
max_chunk_len = max(1, int(runtime_model.get_input_sequence_length()))
for chunk_start in range(0, num_tokens, max_chunk_len):
```

如果 prompt 长度超过 HMONNX 导出的单次输入长度，就分 chunk 执行。

每个 chunk 构造：

```python
data_batch = {
    "past_seq_length": past_len + chunk_start,
}
```

`past_seq_length` 很关键，HMONNX data_preprocessor 和位置编码/cache 都依赖它。

如果传 raw input ids：

```python
data_batch["input_ids"] = chunk_input_ids.unsqueeze(0)
```

如果传 embedding：

```python
data_batch["inputs_embeds"] = self._restore_runtime_batch_shape(chunk_embeds)
```

多模态首 chunk 还会带：

```python
image_embeds
image_grid_thw
video_grid_thw
```

注意，图片相关信息只在：

```python
past_len + chunk_start == 0
```

时加入。

---

### 11.4 data preprocessor

```python
data_preprocessor = runtime_model.get_data_preprocessor()
net_inputs = data_preprocessor(data_batch)
```

这个是 xhmodel_merak/HMONNX 自己的预处理器。

它把：

```python
{
  "input_ids": ...,
  "past_seq_length": ...,
}
```

转换成 HMONNX graph 真正需要的输入列表：

```python
net_inputs
```

---

### 11.5 调用 runtime forward

```python
model_output = runtime_model.forward(*net_inputs)
last_token_logits = self._extract_last_token_logits(model_output)
```

HMONNX runtime 输出可能是多种形式：

- Tensor
- dict
- tuple/list
- object with `.logits`

因此 `_extract_last_token_logits()` 做了兼容：

```python
if isinstance(model_output, dict):
    logits = model_output["logits"]
elif hasattr(model_output, "logits"):
    logits = model_output.logits
elif isinstance(model_output, (list, tuple)):
    logits = model_output[0]
```

最后只取最后一个 token：

```python
return logits.reshape(-1, logits.shape[-1])[-1, :vocab_size]
```

---

## 12. `compute_logits()` 与 `sample()`

Merak 的 `forward()` 已经返回 logits，所以：

```python
def compute_logits(self, hidden_states, sampling_metadata=None):
    if sampling_metadata is None:
        return hidden_states
    return self.logits_processor(None, hidden_states, sampling_metadata)
```

这里参数名叫 `hidden_states`，但实际已经是 logits。

然后：

```python
def sample(self, logits, sampling_metadata):
    return self.sampler(logits, sampling_metadata)
```

采样仍然使用 vLLM sampler。

完整链路：

```text
MerakForCausalLM.forward()
        ↓
返回 flat_logits
        ↓
compute_logits()
        ↓
vLLM sampler
        ↓
next token
```

---

## 13. cache 问题为什么容易发生

当前有两类 cache。

---

### 13.1 vLLM 管理的 full attention KV cache

这部分代码已有明确管理：

```python
past_key_caches
past_value_caches
```

新请求：

```python
_clear_runtime_request_caches()
```

已有历史的 decode 请求：

```python
_load_runtime_request_cache_from_paged()
```

跑完后：

```python
_scatter_runtime_request_cache_to_paged()
```

这部分是标准 vLLM paged cache 与 HMONNX contiguous cache 的桥接。

---

### 13.2 HMONNX runtime 内部的 linear attention cache

Qwen3.5 还有 linear attention cache，例如：

```python
past_conv_caches
past_recurrent_states
```

这些不是 vLLM 标准 paged KV cache 的一部分。

如果只清理：

```python
past_key_caches
past_value_caches
```

而没有清理：

```python
past_conv_caches
past_recurrent_states
```

就可能出现：

```text
请求 A 跑完后，runtime 内部 linear cache 仍然保留。
请求 B 从 past_len=0 开始，但继承了请求 A 的 linear cache。
```

这会导致 logits 异常，例如：

- 输出前几 token 正常；
- 后面开始空行或重复；
- 一直生成到 `finish_reason = length`。

因此新请求开始时必须同时 reset full KV cache 和 linear attention cache。

---

## 14. 当前代码的心智模型

可以把这套逻辑理解为：

```text
vLLM:
  我负责调度请求，给你 input_ids、past_len、block table。
  你这个模型类只要按我的格式返回 logits。

MerakForCausalLM:
  我先把你的 paged KV cache 搬成 HMONNX 能用的 contiguous cache。
  然后调用 HMONNX runtime。
  跑完我再把新增 KV 搬回你的 paged cache。
  最后把 logits 放到你要求的位置。

Qwen3_5MerakForCausalLM:
  我负责告诉基类：
    - HMONNX 模型在哪里；
    - KV cache shape 是什么；
    - vocab size 是什么；
    - embedding 怎么拿；
    - image token / vision config 是什么；
    - 是否需要多模态 raw input ids。
```

---

## 15. 推荐源码阅读顺序

建议按下面顺序看：

1. `vllm_merak/config.py`
   - 看 `MerakConfigParser`
   - 理解 `merak_config.json` 怎么变成 vLLM config

2. `vllm_merak/platform.py`
   - 看 `MerakPlatform.check_and_update_config`
   - 理解为什么 worker 是 `MerakWorkerV1`
   - 理解为什么 `device_type` 是 `cpu`

3. `vllm_merak/model_executor/register.py`
   - 看 `MERAK_MODEL_REGISTRY`
   - 理解 source HF architecture 到 Merak model class 的映射

4. `vllm_merak/model_executor/model_loader/qwen3_5.py`
   - 看 `Qwen3_5MerakForCausalLM.__init__`
   - 理解 HMONNX runtime 如何加载

5. `vllm_merak/model_executor/model_loader/merak_for_causal_lm.py`
   - 看 `forward()`
   - 看 `_run_request()`
   - 看 `_load_runtime_request_cache_from_paged()`
   - 看 `_scatter_runtime_request_cache_to_paged()`

6. `vllm_merak/attention/backends/merak.py`
   - 看 `MerakAttentionMetadataBuilder`
   - 理解 `past_lens`、`subsequence_begins`、`block_indices` 是怎么来的

---

## 16. 一句话总结

`qwen3_5.py` 负责把导出的 Qwen3.5 HMONNX 模型加载进 vLLM；`merak_for_causal_lm.py` 负责把 vLLM 的 batch 调度和 paged KV cache 翻译成 HMONNX runtime 能执行的单请求、连续 cache 推理。
