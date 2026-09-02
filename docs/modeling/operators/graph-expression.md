# 模型语义如何进入图

## 两种注册各有职责

[LLM registry](../../../xhmodel_merak/xh_llm/register.py) 包含不同层次的注册表：

| 入口 | 解决的问题 |
| --- | --- |
| [`register_llm_model()`](../../../xhmodel_merak/xh_llm/builder.py) / `XH_LLM_MODELS` | 从公开 model type 找到模型适配入口，供 family 加载/分流使用 |
| `XHLLM_TRACEABLE_MODULES` | 为具体上游模块类选择可追踪的 wrapper，保留模型语义并表达图操作 |

注册 wrapper 不会自动完成公开模型接入。参考当前
[Qwen3.5 wrappers](../../../xhmodel_merak/xh_llm/models/qwen3_5/_llm_model_impl.py)
中注册的 attention/norm 类，以及 [模型入口](../../../xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py)。
这是当前 `xh_llm` 的机制，不要求 other-model 强行引入另一套 LLM registry。

## `_setup`、`forward` 与 `graph_forward`

wrapper 的 `_setup` 常用于构造 SDK 算子、变换权重、预计算 RoPE 常量等图准备；
与输入有效长度相关的切片、mask 和状态变化仍遵循显式的 tensor/metadata 契约。
如果把一次 dummy input 的长度缓存成固定事实，后续 decode 或另一档位会使用错误的图语义。

原生 `forward` 能表达所需图时可直接使用。依赖 HF Cache 对象或动态 Python 控制流的路径，
可通过 `graph_forward` 暴露前端需要的 tensor 输入与状态；不要求每个 wrapper 都新增它，
也不把“与 forward 签名相同”本身判成错误。关键是所用 frontend 能追踪，且保留数值与状态语义。
特定 API 是否支持、何时选择 `graph_forward` 以当前 SDK 为准。

前端选择也是实际图路径的一部分。旧视觉流程可能先导 ONNX 再转 frontend，
当前 Qwen3.5 patch-token 视觉路径已有直接 TorchFX 入口；不能把旧 skill 中的
“视觉必须先转 ONNX”写成全仓库约束。证据见
[视觉实现](../../../xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_vision_model.py)
与历史 [frontend routing test](../../../tests/qwen3_5/test_visual_token_gears.py)。
该专项断言尚未进入当前 CI；后续回归在活动 CI 内构造最小真实模型，
同时检查 eager 语义与实际 frontend 路径，不能只检查 wrapper 类名或函数是否存在。
