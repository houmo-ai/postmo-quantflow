# 可复用模型 Wrap 规则

本文档定义将 Hugging Face/Transformers 模型封装为 XHQuant `DynamicModule` 时应遵循的通用规则，并给出 Spark2_5 1.7B 模型的具体映射。后续新增模型时，应优先复用本规则，而不是直接复制某个模型的完整 `_model.py`。

本文档只描述图构建和量化使用的内部 Wrapper。模型注册、配置加载、生成调用和公共导出等应用层规则见 [APPLICATION_INTERFACE.md](APPLICATION_INTERFACE.md)。

## 通用规则

### 目标

Wrap 层负责将原生 PyTorch 模型转换为适合 XHQuant 图构建、量化和 KV Cache 管理的模块，同时满足以下约束：

- 不修改原始模型类及其权重布局。
- Wrapper 与原始模块一一注册，转换后可通过 `DynamicModule` 检查。
- `_setup` 只创建图执行所需的算子、状态和权重映射。
- `graph_forward` 只使用可追踪、可导出的张量运算。
- 原生模型和 Wrap 模型在相同输入下保持数值语义一致。
- 模型专用逻辑与通用规则分离，便于其他 Decoder-only 模型复用。

### 规则优先级与等价性定义

规则生成代码时，原始模型的 `forward`、模块属性和算子调用顺序是唯一语义来源；本文件只规定如何把这些语义改写为可追踪的 Wrapper。等价性按以下顺序判断：

1. 输入/输出 Tensor 的形状、dtype、设备和结构一致。
2. 参数切分、权重归属、Cache 更新、位置编码和 mask 语义一致。
3. 算子顺序、残差顺序、激活函数、缩放系数和异常条件一致。
4. 允许将原生动态容器或输出对象改为固定 Tensor/tuple，但不得改变数值语义。

Wrapper 不得为了“看起来更通用”而添加原模型没有的 `to`、reshape、mask、Cache 或精度转换；也不得删除原模型已有的转换。只有 XHQuant 算子本身的输入契约要求时，才允许在契约规定的位置转换。

### 生成流程

对任意新模型，生成器必须按以下顺序工作：

1. 从原始 `modeling_*.py` 提取参与图执行的模块、构造属性、forward 参数和静态配置分支。
2. 为每个必须转换的原始类建立一一对应的注册 Wrapper；先确定公开图入口和参数传播，再编写 `_setup`。
3. 将原始 forward 中的动态 Cache、输出对象和输入处理逻辑转换为显式 Tensor/固定列表；不改变核心算子顺序。
4. 将模型专用名称和能力映射写入“模型映射”小节，通用规则只描述能力，不把专用字段泄漏到通用模板。
5. 生成后按“结构检查 -> 数值检查 -> 图构建检查”验收。若与参考 Wrapper 不一致，先定位为参数、状态、算子、dtype、异常或配置分支哪一类差异，再补充规则并重新生成；不得直接修改参考 Wrapper 来掩盖规则缺口。

### 通用 Wrapper 类模板

每个需要参与图构建的原始模块都应声明一个对应 Wrapper。为兼顾静态类型检查和运行时动态替换，统一使用以下模板：

```python
from typing import TYPE_CHECKING, Any

from xhquant.utils.registry import DynamicModule


if TYPE_CHECKING:

    class _OriginalModuleBase(DynamicModule, OriginalModule):  # type: ignore[misc]
        ...

else:
    _OriginalModuleBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({OriginalModule: "OriginalModule"})
class _WrappedModule(_OriginalModuleBase):
    def _setup(self, cfg: dict[str, Any] | None = None):
        return self

    def graph_forward(self, *args, **kwargs):
        raise NotImplementedError
```

`graph_forward` 不是所有叶子模块都必须单独声明：没有图专用签名的简单算子（例如仅替换实现的 RMSNorm）可以保留 `forward`，但其调用必须仍然是可追踪的。开始 FX/Dynamo trace 前，必须使用框架提供的 graph-forward guard，将根模块和已转换子模块的 `forward` 临时绑定到 `graph_forward`；不能假设直接调用已转换模型会自动走 `graph_forward`。

#### 类命名规则

- 原始类：`OriginalAttention`
- 运行时基类：`_OriginalAttentionBase`
- Wrapper 类：`_OriginalAttention`
- 注册键：使用原始类名，例如 `"OriginalAttention"`
- 注册入口：统一命名为 `register_wrap_modules`

Wrapper 类保持模块内部实现，不通过包的 `__all__` 暴露。外部代码只调用注册入口。

### Wrapper 生命周期

#### 注册

导入 `_model.py` 时，装饰器将原始类与 Wrapper 类写入 `XHLLM_TRACEABLE_MODULES`。模型开始 wrap 前必须调用：

```python
from ._model import register_wrap_modules

register_wrap_modules()
```

`register_wrap_modules` 可以为空操作，但必须存在，作用是提供稳定、显式的注册入口，并确保 `_model.py` 已被导入。

```python
def register_wrap_modules(hf_model=None):
    return None
```

#### `_setup`

`_setup` 在原始模块被替换为 `DynamicModule` 后执行，负责：

- 读取模型属性和 wrap 配置。
- 创建 XHQuant 算子，例如 `RMSNorm`、`MaskedSoftmax`、`LLMCache`、`DynamicSlice`。
- 拆分或重组融合权重。
- 创建只用于图执行的别名和状态。
- 返回 `self`，除非框架明确不要求返回值。

Wrap 框架可能先转换父模块、再转换子模块；父模块 `_setup` 不得假设子模块已经是 `DynamicModule`，也不得在 setup 阶段调用子模块的推理路径。

`_setup` 不应：

- 加载模型文件。
- 修改原始配置文件。
- 执行实际推理。
- 创建依赖当前输入张量值的常量。

依赖配置的静态表、Slice/Gather 元数据、Cache 算子和别名必须在 `_setup` 创建；依赖当前输入长度、past 长度或 batch 内容的值只能作为 `graph_forward` 的 Tensor 输入传入。配置更新回调只能更新已有算子的元数据，不能重新注册子模块或替换参数。

#### `graph_forward`

`graph_forward` 是图模式的执行入口，负责：

- 使用固定、明确的张量参数。
- 调用已 wrap 的子模块。
- 返回下一级 wrapper 所需的张量或固定结构。
- 避免 Transformers 输出数据类、动态字典和依赖输入值的 Python 分支；只允许基于模型配置的静态分支。

推荐将输入收敛为以下通用参数：

```python
def graph_forward(
    self,
    hidden_states,
    past_seq_length=None,
    current_input_length=None,
    past_k_cache=None,
    past_v_cache=None,
    position_embeddings=None,
    window_attention_mask=None,
):
    ...
```

不同层只保留实际需要的参数。对于具有窗口/局部注意力的模型，窗口 mask 是 CausalLM/Model/DecoderLayer/Attention 之间的可选运行时 Tensor 参数；Model 只向需要窗口约束的层传递实际 mask，其余层传入 `None`。全局因果注意力不应凭空增加模型没有定义的额外 mask 参数。具体参数名放在模型特定映射中定义。参数名、参数顺序和 `None` 传播规则必须在模型映射中固定，不能在生成时自行重排。

### 模块分类规则

| 模块类别                    | 是否需要 Wrap | 主要职责                                    |
| --------------------------- | ------------- | ------------------------------------------- |
| `nn.Linear`、`nn.Embedding` | 通常不需要    | 由框架内置转换和量化处理                    |
| RMSNorm                     | 需要          | 替换为 XHQuant RMSNorm，复制权重和 epsilon  |
| MLP                         | 需要          | 固化激活与投影顺序，避免动态函数分派        |
| Attention                   | 必须          | 拆分 QKV、RoPE、Mask、KV Cache 和输出门控   |
| DecoderLayer                | 必须          | 固化残差、Norm、Attention、MLP 的执行顺序   |
| Decoder Model               | 必须          | 遍历层、管理位置编码、Cache 列表和最终 Norm |
| CausalLM                    | 必须          | 调用基础模型并计算 logits                   |
| Transformers 输出类         | 不直接 Wrap   | 图模式返回 Tensor 或固定 tuple              |

### 权重处理规则

#### 保持参数所有权

优先复用原始子模块。必须拆分融合权重时，新建模块后使用 `copy_` 写入参数，避免通过切片创建共享视图。

```python
with torch.no_grad():
    self.q_proj.weight.copy_(fused_weight[:q_size])
```

#### 数据类型

- 不在 `_setup` 中强制转换全模型 dtype。
- 不要为了适配图构建而删除或增加模型原本的 dtype 边界。原模型在 Norm、激活、门控、RoPE、残差或输出头前后的显式转换，必须在相同位置保留。
- `past_seq_length`、`current_input_length` 等索引张量不得无条件转换；只有被所用 Cache/索引算子契约要求时才转换，并在规则中记录该契约。
- 窗口 mask 应使用与 attention logits 兼容的 dtype，并直接交给 mask 算子；不要在 Attention wrapper 中重复构造、复制或隐式改变 mask 语义。
- 只有原始模型或所使用的 XHQuant 算子明确要求时，才进行精度转换，并保持转换的位置和输出 dtype 与原始语义一致。
- Norm、Softmax 和门控计算的精度策略应与原始模型及 XHQuant 算子契约一致。

#### Bias

拆分投影时必须同时处理 bias。原始模块无 bias 时，新模块也必须设置 `bias=False`。

### Attention Wrap 规则

Attention wrapper 应按原模型的投影布局实现以下步骤；下列名称是抽象角色，不要求所有模型使用同名模块：

1. 通过独立的 Q/K/V 投影，或通过一个融合投影后按模型声明的维度/顺序拆分 Q、K、V；拆分只在 `_setup` 或图中执行一次。
2. 保持原模型的 Q/K/V 维度、头数、head dimension 和 bias；不得根据通用默认值猜测布局。
3. reshape/transpose 的顺序必须与原模型一致，通常为 `[batch, sequence, heads, head_dim] -> [batch, heads, sequence, head_dim]`。
4. 对 Q、K 应用 RoPE；RoPE 的输入表形状、旋转通道数和输出 dtype 必须以原模型/算子契约为准。
5. 在启用 Cache 时，通过每层 Cache 算子更新 K、V；Cache 的 axis、最大长度、past/current 索引和返回形状必须保持一致。
6. 按原模型的 MQA/GQA/repeat 规则将 KV heads 扩展到 query heads；若模型没有分组 KV，不应强行 repeat。
7. 按原模型顺序计算缩放点积注意力。缩放可以施加在 Q、logits 或指定的算子参数上，但不得同时重复施加。
8. 全局因果约束使用模型映射指定的 causal softmax；窗口/局部约束使用模型映射指定的 mask 算子和外部 mask，不得将二者静默互换。
9. 计算 attention output；如果原模型有 head-wise 或 channel-wise gate，保留其激活函数、计算精度、广播形状和输出 cast。
10. 按原模型的 transpose/reshape 顺序恢复 `[batch, sequence, hidden]`，再通过输出投影。

图模式不直接使用 Transformers `Cache` 对象。Cache 应转换为显式输入：

- `past_seq_length`
- `current_input_length`
- `past_k_cache`
- `past_v_cache`

每层 wrapper 只接收本层对应的 K/V cache Tensor。外层 Model 可以接收按层排列的 Tensor 列表，但不得把 Transformers `Cache` 对象传入图。

#### 全局与窗口注意力

- 全局注意力使用模型原生的 causal 约束，不应因为 Wrapper 而新增一个未定义的显式 mask。
- 窗口/局部注意力必须接收非空的运行时 mask（或模型明确规定的等价输入），限制 query 可访问的 key；不得静默降级为全局注意力。
- 该 mask 必须沿实际调用链显式传递，并在不需要窗口约束的层传入 `None`。参数名和层类型由具体模型映射定义。
- Mask 必须由输入处理器或上游图输入根据窗口配置、past 长度和 query/key 长度构造，并与 logits 的 batch/head/query/key 维度对齐；Attention 不应隐式重复构造。
- 如果模型映射指定 `MaskedAdd`、`MaskedSoftmax` 或其他 XHQuant mask 算子，必须直接调用该算子，不得用看似等价的手写 `masked_fill`/额外加法替换；算子的饱和、重复加法和 dtype 行为属于其契约。
- 可以保留用于调试或参考的 mask 辅助函数，但除非原模型明确要求，`graph_forward` 不得调用它来重新生成运行时 mask。
- 窗口大小配置可同时影响 Cache 的最大长度，但配置值本身不是 `graph_forward` 的 Tensor 输入。
- Cache 的无限长度/窗口长度取值必须与原模型的缓存语义一致，不能把所有层统一截断。

### RoPE 规则

- cos/sin 应在 Model wrapper 或专用 RoPE wrapper 中统一生成或缓存。
- Attention wrapper 只消费 `position_embeddings=(cos, sin)`。
- `partial_rotary_factor` 小于 1 时，只旋转模型定义的 rotary width，其余通道直接拼接。rotary width 的来源必须与原模型一致：如果原模型以 cos/sin 表宽度为准，就使用表宽度；如果原模型以配置计算值为准，就使用该计算值；不得自行修正、取整或扩大旋转范围。
- 不在每个 DecoderLayer 中重复构建相同的 cos/sin。
- 多种 attention/layer 类型使用不同 RoPE 参数时，应按类型分别缓存。
- Model wrapper 遍历层时，必须按当前层声明的类型选择对应的 `position_embeddings`，不能要求不同配置共用一套 RoPE 表。

### 配置契约

对于需要固定长度图和 KV Cache 的 Decoder-only 模型，Wrapper 配置通常需要以下字段；没有对应能力的模型可以省略，并应在模型映射中说明：

| 字段                    | 用途                                    |
| ----------------------- | --------------------------------------- |
| `use_cache`             | 是否启用 KV Cache                       |
| `input_sequence_length` | 当前图的输入长度                        |
| `context_max_length`    | 最大上下文长度                          |
| `num_logits_to_keep`    | 输出全部 logits 或最后一个 token logits |
| `only_first_block`      | 调试时仅运行第一层                      |
| `max_layers`            | 限制实际执行层数                        |
| `kv_cache.cache_axis`   | KV Cache 的序列轴                       |

这些字段是 XHQuant 图配置，不是 Hugging Face 配置的强制命名。生成通用 Wrapper 时应通过一个小的配置适配层读取它们，避免把任何单一模型的字段名、层类型或窗口参数写死到通用基类中。

窗口大小等模型配置应从原始 Hugging Face 配置读取，不属于 XHQuant wrap 配置；窗口 mask 应由输入处理器生成并作为运行时 Tensor 传递，也不应转换为配置字段或在 Wrapper 内部缓存。具体字段名由模型特定映射定义。

读取配置时应兼容 `ConfigDict` 和普通 `dict`，但内部使用方式要统一。

模型适配器应通过一个统一 helper 将 `ConfigDict` 和普通 `dict` 归一化；所有 `_setup` 都在 setup 阶段读取配置，不能在 `graph_forward` 中读取或修改配置。`cfg.get(name, default)` 只用于确有默认值的字段，必需字段缺失时应尽早报错。

### 错误处理规则

遇到不支持的模型特性时，应在 `_setup` 阶段尽早失败，并给出配置名和解决方式。支持集合必须由模型适配器声明，不能把某一个模型的层类型集合写入通用 Wrapper。例如：

```python
# attention_features 由具体模型适配器从原始 config 映射得到
supported_features = {"global_causal_attention", "window_attention", "gqa", "partial_rope"}
declared_features = set(self.attention_features)
unsupported_features = declared_features - supported_features
if unsupported_features:
    raise NotImplementedError(f"Unsupported attention features: {sorted(unsupported_features)}")
```

运行时如果模型声明需要窗口约束但没有提供对应 mask，应显式报错，不能退化为全局注意力。

禁止静默降级以下行为：

- 窗口/局部 attention 变成全局 attention。
- head-wise gate 被忽略。
- KV Cache 被关闭。
- partial RoPE 被当作完整 RoPE。
- tied embedding 被当作独立 `lm_head`。

### 通用验收标准

每个 wrapper 实现完成后至少执行以下通用检查；模型专用参数名和层类型通过适配器映射后再检查。

对于每个模型，直接验收基准应是该模型的参考 Wrapper：生成器可以改变类名、注解和局部变量名，但不得改变其公开图入口、模块注册关系、setup 状态、算子顺序、异常条件和 Tensor 语义。任何差异都必须先判断是等价重写还是规则遗漏；若属于规则遗漏，应先补充本文件再重新生成。

#### 注册检查

- 所有规则表中标记“必须”的类都存在于 `XHLLM_TRACEABLE_MODULES`。
- Wrap 后这些原始模块都变为 `DynamicModule`。
- `register_wrap_modules()` 可重复调用且不报错。

#### 结构检查

- 参数名和权重形状符合预期。
- Decoder 层数未变化。
- Q、K、V heads 和维度正确。
- KV Cache 每层数量与 `num_hidden_layers` 一致。
- 全局 attention 层的 Cache 不截断，窗口 attention 层的 Cache 最大长度等于模型配置指定的窗口长度。
- Model 按每层声明的 attention/RoPE 类型选择位置编码，且只有需要窗口约束的层收到非空 mask。

#### 数值检查

使用小配置、固定随机种子和相同权重，对比原始模型与 wrapper：

- RMSNorm 输出。
- 单层 MLP 输出。
- 无 Cache 的 Attention 输出。
- 全局 attention 层与原始标准 causal attention 输出。
- 窗口 attention 层在相同外部 mask 下的输出。
- prefill 输出。
- 单 token decode 输出。
- 最终 logits。

建议误差阈值：

- `float32`: `atol=1e-5`, `rtol=1e-4`
- `float16`/`bfloat16`: 根据算子实现设置更宽松阈值，并记录实际最大误差

#### 图构建检查

- prefill 图构建成功。
- decode 图构建成功。
- 包含全局和窗口 attention 的混合层模型图构建成功（若模型支持两者）。
- 窗口 mask 能从模型输入传递到目标 attention；全局 attention 层收到 `None`，并使用其 causal 逻辑。
- Cache/位置索引和 mask 在传递过程中不会出现规则之外的 dtype 转换。
- 计算图中不出现由 Wrapper 新增的无语义 `to`、mask 构造、重复缩放或重复 reshape；所有 Tensor 转换都能在原模型或 XHQuant 算子契约中找到对应来源。
- 图中不存在 Transformers `Cache`、输出数据类或无法追踪的 Python 容器。
- 导出模型可完成至少一次 prefill 和一次 decode。

## 模型特定规则：Spark2_5 1.7B

以下内容只适用于 `ifly/xh_token_spark_2_5/spark_2_5/modeling_spark.py`，不应上升为通用 Wrapper 规则。

Spark2_5 的直接验收基准是 `xhmodel_merak/xh_llm/models/xh_token_spark_2_5/_model.py`；生成代码可以改变量名和注解，但必须保持该文件的公开入口、注册关系、setup 状态、算子顺序、异常条件和 Tensor 语义。

### 类映射

目标文件：`spark_2_5/modeling_spark.py`

- `Spark2_5RMSNorm` -> `_Spark2_5RMSNorm`（必须）：使用 `xhquant.nn.RMSNorm`，复制 `weight` 和 `variance_epsilon`。
- `Spark2_5MLP` -> `_Spark2_5MLP`（必须）：保持 `down_proj(gelu(gate_proj(x)) * up_proj(x))`。
- `Spark2_5Attention` -> `_Spark2_5Attention`（必须）：处理融合 QKV、GQA、RoPE、按类型处理 Mask/Cache 和可选 head gate。
- `Spark2_5DecoderLayer` -> `_Spark2_5DecoderLayer`（必须）：保持 pre-norm、双残差结构，并沿统一签名向 Attention 传递 mask（full 层收到 `None`）。
- `Spark2_5Model` -> `_Spark2_5Model`（必须）：遍历 `layers`，按层选择 RoPE、Mask 和 Cache，最终执行 `norm`。
- `Spark2_5ForCausalLM` -> `_Spark2_5ForCausalLM`（必须）：接收 sliding mask，调用 `model` 后执行 `lm_head`。
- `Spark2_5PreTrainedModel` -> 无（不需要）：只提供 Transformers 基类能力，不直接参与图执行。

### Spark2_5 特殊规则

- 原始 Attention 使用融合 `q_k_v_proj`，Wrapper 可在图中切片，也可在 `_setup` 中拆为三个 Linear；选择必须与量化流程兼容。
- `num_key_value_heads` 可能小于 `num_attention_heads`，必须保留 GQA repeat 语义。
- `headwise_attn_output_gate=True` 时必须保留 `g_proj` 和 `sigmoid`/`silu` 门控。
- `layer_types` 同时包含 `full_attention` 与 `sliding_attention` 时，Model wrapper 必须按层选择对应的 RoPE，并且只向 sliding attention 层传递实际的 `sliding_attention_mask`。
- 通用规则中的 `window_attention_mask` 在 Spark2_5 中具体命名为 `sliding_attention_mask`。
- `_Spark2_5ForCausalLM.graph_forward`、`_Spark2_5Model.graph_forward`、`_Spark2_5DecoderLayer.graph_forward` 和 `_Spark2_5Attention.graph_forward` 应显式保留 `sliding_attention_mask` 参数；只有当前层为 sliding attention 时才消费非空 mask，full attention 必须拒绝非空 mask并继续使用 `MaskedSoftmax`。
- `past_seq_length` 和 `current_input_length` 应沿调用链直接传递，不得在 Attention、DecoderLayer 或 Model 中额外 `.to()`、reshape 或复制；输入处理器提供的 dtype 必须满足所用 XHQuant Cache/softmax 算子契约。
- Spark2_5 DecoderLayer 的执行顺序必须严格为：`input_layernorm -> self_attn -> residual add -> post_attention_layernorm -> mlp -> residual add`，不得插入额外的 Norm、激活或 dtype 转换。
- Spark2_5 head gate 直接使用配置指定的 `sigmoid` 或 `silu`，保持 gate 的 reshape、广播和与 attention output 相乘的顺序；不得擅自增加 float32 cast。
- Spark2_5 的 `rope_head_dim` 使用 `int(head_dim * partial_rotary_factor)`，cos/sin 表由 `compute_rope_cos_sin` 生成并按 layer type 缓存；不得改用另一套 rotary width 推导。
- Spark2_5 不定义 `full_attention_mask` wrap 参数。full attention 使用标准 causal attention，不能照搬 Gemma4 将额外 mask 传入 full attention 的接口。
- sliding attention 的 `LLMCache.attention_max_length` 必须设为 `config.sliding_window`；full attention 保持 `-1`。
- 当前 Spark2_5 1.7B Wrapper 不支持 `tie_word_embeddings=True` 的 CausalLM 输出路径；应在运行时显式抛出 `NotImplementedError`，不得静默改用未验证的 embedding weight 或独立 `lm_head`。
- `num_logits_to_keep` 只能接受 Wrapper 明确支持的值。用于生成时通常支持 `0` 或 `1`。

### Spark2_5 图执行顺序

以下是 Spark2_5 1.7B 当前 Wrapper 的行为契约，是通用 Attention/Decoder 规则在本模型上的具体实例：

1. `_Spark2_5RMSNorm.forward` 只调用已创建的 `xhquant.nn.RMSNorm`；`_setup` 创建该算子，以独立 `Parameter` 深拷贝原 `weight`，并保留 `variance_epsilon`。
2. `_Spark2_5MLP.forward` 固定执行 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`；`_setup` 不添加其他状态。
3. `_Spark2_5Attention._setup` 先校验 gate activation 只允许 `sigmoid`/`silu`，再创建 `Rope`、`MaskedSoftmax(dim=-1)`、`MaskedAdd` 和 `SoftmaxPlus(dim=-1)`；按 `use_cache` 创建 K/V `LLMCache`。full attention 的 `attention_max_length=-1`，sliding attention 使用 `config.sliding_window`。
4. Attention 图执行固定为：独立 Q/K/V 投影（由融合 QKV 按 `q_dim`、`kv_dim` 切分）-> reshape/transpose -> Q/K RoPE -> 可选 K/V Cache -> Q 乘 `scaling` -> `repeat_interleave` 扩展 KV heads -> QK matmul。
5. full attention 只调用 `MaskedSoftmax(attention_weights, past_seq_length)`；收到非空 `sliding_attention_mask` 必须报错。sliding attention 必须收到非空 mask，按 `MaskedAdd` 后 `SoftmaxPlus` 执行。
6. Attention output 经 V matmul 后，按配置执行可选 gate，再 transpose/reshape，最后调用 `out_proj`。
7. `_Spark2_5DecoderLayer.graph_forward` 只执行双残差顺序：保存 residual，调用 `input_layernorm` 和 self-attention 后相加；再次保存 residual，调用 `post_attention_layernorm` 和 MLP 后相加。
8. `_Spark2_5Model.graph_forward` 只接收 `inputs_embeds`，要求 `past_seq_length` 和 `current_input_length` 非空；按每种 layer type 生成一次 cos/sin 缓存和 `DynamicSlice`，逐层传入对应 RoPE、Cache 和 mask。
9. `use_cache=True` 时，`past_key_cache[layer_index]` 与 `past_value_cache[layer_index]` 只传给对应层；`use_cache=False` 时每层显式传入 `None`。
10. `only_first_block=True` 时执行层数优先固定为 1；否则 `max_layers` 大于 0 时在达到上限后停止，`max_layers` 为 0 或负数表示不限制层数。`num_logits_to_keep=1` 时在最终 Norm 前按 `current_input_length - 1` 做 `BatchGather`；只允许 `0` 和 `1`。
11. `_Spark2_5ForCausalLM.graph_forward` 只调用 `self.model` 后执行 `lm_head`；`tie_word_embeddings=True` 必须抛出 `NotImplementedError`，不能静默走另一条输出路径。

`_Spark2_5DecoderLayer._setup` 必须拒绝未知的 `layer_type`，并把已验证的 layer type 同步到 `self_attn.layer_type`；`_Spark2_5Model._setup` 必须在创建位置表前拒绝未知 layer type。`_Spark2_5ForCausalLM._setup` 可以保持空操作并返回 `self`。

Spark2_5 四个图入口的参数名和顺序固定如下：

```text
Attention:    hidden_states, past_seq_length, current_input_length,
              position_embeddings, sliding_attention_mask,
              past_k_cache, past_v_cache
DecoderLayer: hidden_states, past_seq_length, current_input_length,
              position_embeddings, sliding_attention_mask,
              past_k_cache, past_v_cache
Model:        inputs_embeds, past_seq_length, current_input_length,
              sliding_attention_mask, past_key_cache, past_value_cache
CausalLM:     inputs_embeds, past_seq_length, current_input_length,
              sliding_attention_mask, past_key_cache, past_value_cache
```

调用时允许使用关键字传参，但不得把 `past_k_cache/past_v_cache` 与外层的 `past_key_cache/past_value_cache` 混用，也不得把 `position_embeddings` 放到 Model 的公共输入中。

`_Spark2_5Model._setup` 的静态构建细节也属于本模型契约：位置长度取 `max(cfg.max_position_embeddings（缺省 32768）, cfg.context_max_length)`；位置索引为 embedding 所在设备上的 `torch.long` 序列；每个 layer type 的 cos/sin 增加 batch/head 维后，在 axis=2 使用 `DynamicSlice`。`DynamicSlice` 和 `BatchGather` 必须提供 `_update_cfg`，以便 batch size 和 input sequence length 变化时只更新算子元数据，不在 `graph_forward` 中创建新模块。

除上述静态配置分支外，Spark2_5 生成代码不得引入依赖输入值的 Python 控制流、Transformers Cache、输出数据类或隐式 mask 构造。

### Spark2_5 配置与验收补充

- `sliding_window` 从 Hugging Face 配置读取，不属于 XHQuant wrap 配置。
- `sliding_attention_mask` 由输入处理器生成，作为运行时 Tensor 依次传入 CausalLM、Model、DecoderLayer 和 Attention；不得转换成配置字段或在 Wrapper 内部缓存。
- Spark2_5 的验收必须额外覆盖 full/sliding 混合层、GQA、head gate、partial RoPE、prefill、单 token decode 和 `tie_word_embeddings=True` 的异常路径。
- 额外检查 `q_k_v_proj` 切分后原融合模块不再被重复调用，切分维度、bias 存在性和参数名称符合映射。
- 额外检查 `only_first_block`、`max_layers`、`num_logits_to_keep` 的优先级和执行位置符合 Spark2_5 图执行顺序。

### Spark2_5 文件结构

```text
ifly/xh_token_spark_2_5/
├── __init__.py
├── _model.py                 # XHQuant wrapper 实现
├── WRAP_RULES.md             # 本规则文档
└── spark_2_5/
    ├── __init__.py
    ├── configuration_spark.py
    └── modeling_spark.py
```

`modeling_spark.py` 保持 Transformers 原生实现，所有 XHQuant 适配逻辑放在 `_model.py` 中。

### FlashAttention score 精度

FlashAttention 使用 `q_bits/k_bits/v_bits/p_bits`，QK score 固定为 FP16，
不再接受 `s_bits`。旧 workflow 需移除该字段；`p_bits` 缺省为 8，
不从 Q 或旧 score 位宽继承。显式 `p_bits` 保持有效。
