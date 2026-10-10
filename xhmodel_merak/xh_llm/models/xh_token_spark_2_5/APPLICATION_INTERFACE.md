# 模型应用接口实现规则

本文档定义 Hugging Face/Transformers 模型接入 Merak/XH 应用层时应遵循的实现规则，并给出 Spark2_5 1.7B 的当前状态和待实现映射。规则主要参考 `ifly/spark_xh/spark_moe_model.py`。

图构建、量化和 KV Cache 的内部 Wrapper 规则见 [WRAP_RULES.md](WRAP_RULES.md)。

## 接口分层

模型接入分为三层，每层只暴露本层职责：

| 层级 | 面向对象 | 主要入口 | 职责 |
| --- | --- | --- | --- |
| 原生 Transformers 层 | 原生 PyTorch/Transformers 调用方 | `AutoModelForCausalLM.from_pretrained()`、`generate()` | 模型注册、权重加载和标准生成 |
| Merak/XH 应用层 | 业务应用和推理脚本 | `register_custom_model()`、`AutoLLMConfig`、`AutoLLMModel` | 配置解析、模型状态、设备管理、生成和 HMONNX 接入 |
| XHQuant Wrapper 层 | 图构建和量化框架 | `register_wrap_modules()`、`_setup()`、`graph_forward()` | 张量图、算子替换、KV Cache 和权重映射 |

应用代码不得直接依赖以下内部实现：

- 以下划线开头的 Wrapper 类。
- `_setup()` 和 `graph_forward()`。
- 每层显式传递的 K/V Cache Tensor。
- XHQuant 注册表中的内部模块名称。

应用层对外只保证配置加载、模型加载、processor、状态设置和 `generate()` 等稳定入口。

## 应用层类定义

### 配置类

每个模型应提供一个继承 `TextLLMModelConfig` 的配置类：

```python
class SparkTokensModelConfig(TextLLMModelConfig):
    pass
```

只有模型确实需要新增且无法由基础配置表达的字段时，才扩展配置类。新增字段必须给出默认值、类型和约束，避免在模型方法中读取未声明的临时属性。

### 模型类

模型类必须继承 `TextLLMModel`，并通过 `register_llm_model()` 注册：

```python
@register_llm_model("Spark2_5ForCausalLM")
class SparkTokensModel(TextLLMModel):
    transformers_min_version = "<最低验证版本>"
    transformers_max_version = "<最高验证版本>"
    HF_MODEL_CLS = Spark2_5ForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = SparkTokensHMONNXModel
    CONFIG_CLS = SparkTokensModelConfig
```

实现时遵循以下规则：

- 注册键必须与 Merak 配置中的 `model_type` 一致，并能唯一标识模型架构。
- Transformers 最低和最高版本必须来自实际验证结果，不能直接照抄其他模型。
- `HF_MODEL_CLS` 指向具体模型类，`HF_AUTO_MODEL_CLS` 指向对应的 Transformers AutoModel 类。
- 支持 HMONNX 时声明 `HMONNXINFERENCE_CLS`；未实现时不得设置无效占位类。
- `CONFIG_CLS` 必须指向该模型的应用层配置类。
- 构造函数只负责将配置交给基类，不应重复加载权重或创建图算子。

### 包级导出

对外类必须从包的 `__init__.py` 导出，并通过 `__all__` 固定公共接口。内部 Wrapper 类和 `_model.py` 中的实现不得加入 `__all__`。

## 注册与加载流程

应用层应按以下顺序创建模型：

1. 调用 `register_custom_model("./ifly/xh_token_spark_2_5")` 导入并注册自定义模型。
2. 使用 `AutoLLMConfig.from_pretrained(cfg.model)` 构造应用层配置。
3. 校验返回配置类型与目标模型一致。
4. 使用 `AutoLLMModel.from_pretrained(config=model_cfg)` 加载模型。
5. 根据推理目标调用 `set_state()`。
6. 获取 processor，构造模型输入。
7. 设置设备、dtype 和 `eval()` 状态后执行生成。

应用代码不应直接实例化应用层模型类，也不应绕过 `AutoLLMConfig` 手工拼装不完整配置。

## Wrap 生命周期

### `init_wrap_model`

应用层模型必须在基类开始转换前导入并调用本模型的 Wrapper 注册入口：

```python
def init_wrap_model(self, hf_model):
    from ._model import register_wrap_modules

    register_wrap_modules()
    super().init_wrap_model(hf_model)
```

`register_wrap_modules()` 是稳定的内部注册入口。即使装饰器在模块导入时已经完成注册，该函数仍应保留，以确保应用层无需了解导入副作用。

### `_wraped_post`

基础转换完成后的模型结构适配放在 `_wraped_post()` 中，包括：

- 复制输入 embedding，供生成流程构造 `inputs_embeds`。
- 设置 `pad_token_id`。
- 根据实际 Decoder 层数设置 KV Cache 层数。
- 根据 Attention 结构设置 K/V Cache shape。
- 处理 `max_layers` 等调试或裁剪配置。

Cache shape 必须从当前模型配置和 Attention 实现推导，不能照抄其他架构。设置前应校验最大层数不超过模型实际层数。

### `get_hf_model`

模型权重加载后、Wrapper 转换前所需的结构调整放在 `get_hf_model()` 中。例如 MoE 模型在此进行专家或 MLP 解融合。

该方法不得改变应用层输入输出语义。若模型无需预处理，应直接复用基类实现。

### `_get_hf_config_files`

导出或复制模型产物时，必须包含运行时加载 tokenizer、processor 和 generation config 所需的文件。通常包括：

- `*.json`
- `*.jinja`
- tokenizer 使用的 `*.model`
- 当前模型明确需要的其他配置资产

不得无条件复制权重文件或与运行时无关的大文件。

## 配置规则

Merak 模型配置至少应明确以下字段：

| 字段 | 规则 |
| --- | --- |
| `model_type` | 必须与 `register_llm_model()` 的注册键一致 |
| `hf_model` | 指向可由目标 Transformers 类加载的模型目录 |
| `context_max_length` | 同时约束输入上下文和 KV Cache 容量 |
| `prefill_chunk_length` | 启用分块 prefill 时必须为正数且不超过上下文长度 |
| `use_cache` | decode 或分块 prefill 使用 Cache 时必须开启 |
| `num_logits_to_keep` | 由 Wrapper 明确校验支持值；Spark2_5 当前只支持 `0` 和 `1` |
| `quant_scheme` | 明确默认量化方案以及需要覆盖的节点或算子 |
| `only_first_block` | 仅用于调试或图验证，不作为正式完整模型推理配置 |
| `max_layers` | 可选；正数时不得超过模型实际 Decoder 层数 |
| `enable_auto_offload` | 仅在目标运行环境经过验证后启用 |

配置不满足模型约束时，应在加载、wrap 或 `_setup()` 阶段尽早抛出带上下文的 `ValueError` 或框架配置异常，不应等到图执行阶段出现模糊的 shape 错误。

## 推理调用契约

### 输入

推荐通过 `get_tf_processor()` 获取 processor，再调用 `apply_chat_template()` 生成张量输入。应用层应：

- 按 tokenizer 定义传入 `messages`。
- 显式选择是否启用 thinking 模式。
- 将输入移动到模型所在设备。
- 使用 `pop("token_type_ids", None)` 移除模型不支持的字段。
- 在生成前检查输入长度不超过 `context_max_length`；截断策略必须由调用方明确选择。

### 生成

生成前必须执行：

```python
model.to(device=device, dtype=dtype)
model.eval()
```

生成调用应处于以下上下文中：

- `LLMInferenceContextManager(model)`
- `torch.no_grad()` 或 `torch.inference_mode()`
- 可选的时间和显存统计上下文

`generate()` 参数沿用 Transformers/Merak 的生成语义。采样参数、最大生成长度以及 EOS/PAD token 应由调用方显式配置。

### 输出

`generate()` 返回的 token Tensor 包含输入 prompt。应用层应使用输入 token 数切掉 prompt，再交给 tokenizer decode：

```python
output_ids = generated_ids[0, input_length:]
text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

thinking 内容与最终回答的拆分属于产品协议，不应写入模型 Wrapper。

### 流式和批处理

- `TextStreamer` 只是在同步 `generate()` 中接收 token 的输出回调，不代表模型提供异步 iterator、SSE 或服务端流式协议。
- 当前 Spark XH 示例证明了单样本生成，Spark2_5 原生示例证明了多条记录的逐条生成。
- 在多样本 padded batch、不同 prompt 长度和批量 KV Cache 均完成测试前，不得声明正式支持张量级批处理。

## 异常和资源管理

模型接入应在初始化阶段明确拒绝未支持能力，例如：

- sliding-window attention。
- 非完整 attention 层。
- Wrapper 无法表达的多套 RoPE 参数。
- 缺少位置编码、输入长度或 Cache。
- 不支持的 `num_logits_to_keep`。
- Cache shape 与 Attention 结构不匹配。

当前应用接口没有公共 `close()` 或 `shutdown()` 方法。资源按以下原则管理：

- 推理状态由 `LLMInferenceContextManager` 管理。
- Autograd 状态由 PyTorch 推理上下文管理。
- 文件、streamer 和服务连接由创建它们的调用方关闭。
- CUDA 临时缓存清理由内部实现按需处理，不作为公共应用 API。

## Spark2_5 1.7B 当前状态

### 已完成

| 能力 | 实现位置 | 状态 |
| --- | --- | --- |
| Transformers 配置和模型注册 | `spark_2_5/register_spark2_5()` | 已完成 |
| 原生 `AutoModelForCausalLM` 加载和 `generate()` | `native_spark_tokens_generate.py` | 已完成 |
| XHQuant Wrapper 注册入口 | `_model.py/register_wrap_modules()` | 已完成 |
| CausalLM 图入口 | `_Spark2_5ForCausalLM.graph_forward()` | 已完成 |
| tied embedding logits | `_Spark2_5ForCausalLM.graph_forward()` | 已完成 |
| `num_logits_to_keep` 校验 | `_Spark2_5Model._setup()` | 已完成 |
| 最小无 Cache 图执行 | 1 batch、3 tokens、2 layers 的本地探针 | 已验证 |

### 待实现

| 能力 | Spark2_5 目标位置 | 参考实现 |
| --- | --- | --- |
| `TextLLMModelConfig` 子类 | `spark_tokens_1_7b_model.py` | `XHSparkModeModelConfig` |
| 注册后的 `TextLLMModel` 子类 | `spark_tokens_1_7b_model.py` | `XHSparkModeModel` |
| Transformers 版本范围 | 应用层模型类属性 | `XHSparkModeModel` 类属性 |
| `init_wrap_model()` | `spark_tokens_1_7b_model.py` | `XHSparkModeModel.init_wrap_model()` |
| embedding 和 KV Cache 后处理 | `spark_tokens_1_7b_model.py` | `XHSparkModeModel._wraped_post()` |
| HMONNX 推理类 | 新增专用模块 | `XHSparkMoeHMONNXModel` |
| 公共类导出 | `xh_token_spark_2_5/__init__.py` | `spark_xh/__init__.py` |
| Merak 模型配置 | `configs_merak` 下新增配置 | Spark XH 模型配置 |
| Merak 端到端生成示例 | `examples_merak` 下新增示例 | `spark_xh_generate.py` |

`spark_tokens_1_7b_model.py` 当前为空，因此不能把 Wrapper 已完成等同于 Merak/XH 应用接口已经可用。

## 实现检查清单

新增模型应用接口完成前，应逐项确认：

- [ ] `model_type` 与应用层注册键一致。
- [ ] Transformers 版本范围经过目标模型验证。
- [ ] HF 具体模型类和 AutoModel 类正确。
- [ ] 应用层配置和模型类可通过 Auto 类加载。
- [ ] `init_wrap_model()` 在基类转换前完成 Wrapper 注册。
- [ ] embedding、层数和 KV Cache shape 来自模型实际结构。
- [ ] 不支持配置在初始化阶段给出明确异常。
- [ ] tokenizer、chat template 和 generation config 等资产可随模型导出。
- [ ] 包级 `__all__` 只暴露稳定应用接口。
- [ ] `generate()` 输出按输入长度切片后再 decode。
- [ ] 至少验证无 Cache prefill、带 Cache decode 和目标量化配置。
- [ ] 批处理、流式和 CPU 能力只在真实测试后声明支持。

## 最小验证范围

应用接口实现后，至少应执行以下验证：

1. 注册包后，`AutoLLMConfig` 返回目标配置类。
2. `AutoLLMModel` 返回目标模型类。
3. 小配置模型可以完成 Wrapper 转换和一次 `graph_forward()`。
4. `num_logits_to_keep=0` 返回全序列 logits，`1` 返回最后位置 logits。
5. 开启 Cache 后，prefill 和单 token decode 的 Cache shape 正确。
6. processor 生成的输入可完成一次 `generate()` 并正确切除 prompt。
7. HMONNX 若被声明支持，可加载导出产物并完成同一输入的推理。
