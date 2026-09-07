# Attention 的图表达

适用范围：当前 Merak 的 Qwen3.5/Next hybrid attention、Gemma4 视觉和 Kokoro ALBERT 路径。
这些是有具体使用方的契约，不是所有 attention 或所有芯片的统一限制。

## FlashAttention：布局与五路精度

[HybridGatedAttentionMixin](../../../xhmodel_merak/xh_llm/models/qwen3_5/_hybrid_gated_delta_net.py)
在 `_setup()` 中构造 `xhquant.nn.FlashAttention`，将
`num_heads`、`num_kv_heads`、`scale`、`is_causal` 和精度参数按关键字传入。

| 接口 | 此路径的语义 |
| --- | --- |
| Q / K / V | `[B, Hq, Sq, D]` / `[B, Hkv, Skv, D]` / `[B, Hkv, Skv, D]`，GQA 的 head 数分别传递；cache 路径中 `Sq` 与 `Skv` 可不同 |
| FA 输出 | `[B, Hq, Sq, D]`，先 transpose 为 `[B, Sq, Hq, D]`，再合并 head 轴 |
| `q_bits`, `k_bits`, `v_bits` | 三路输入精度，wrapper 分别接受 8 或 16 |
| `s_bits`, `p_bits` | score/probability 的独立精度，不能从 Q 的精度推断两者相同 |
| enable 开关 | `flash_attention.enable`；旧 `bfp_flash_attention` 不再启用该路径 |

普通 matmul 分支手工使用缩放；FA 构造传入同一语义的 scale，避免重复缩放。
只检查 reshape 后的最终 shape 看不出 head/token 轴交换，需用非平凡输入比较输出。
配置传入合法 bit 数仅证明 wrapper 接受参数，SDK/目标支持和数值容差仍需单独验证。

参考配置：[Qwen3.6 MoE 全 FA16](../../../configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_fa_all16.yaml)。
历史细粒度断言在 [precision wrapper test](../../../tests/qwen3_5/test_flash_attention_precision_wrapper.py)，
不属于当前 CI。当前 [Qwen block 导出](../../../benchmark_test/ci_test_transformer5.16.0/test_qwen35_35b_a3b_block_export.py)
检查其固定配置的导出链路，并不穷举五路精度组合；新增这类回归应在当前 CI 内补用例与 impact。

## MLA：权重融合不等于共享量化 cache

Ling 的 XH2a 路径允许 Q/K 与 V/output 权重吸收，但保持 Q/K 的 D=C+R、V 的 D=C。
K/V 的 SEFP 指数对齐轴不同，不能把相同浮点 latent 当成可共享的量化张量。
融合必须在 GPTQ/AutoRound 之前完成，导出时保留融合后 Linear 的整数载荷。
具体公式、gate 限制、softmax scale 与验证见 [Ling MLA](../structures/ling-mla-quantization.md)。

## MaskAdd：数学式相似不代表图契约相同

[Gemma4 视觉适配](../../../xhmodel_merak/xh_llm/models/gemma4_series/gemma4_series_vision_model.py)
使用 `xhquant.nn.MaskedAdd` 表达 Softmax 前的 mask；不能直接“简化”为一个普通 Add。
[Kokoro ALBERT](../../../xhmodel_merak/xh_other_model/models/kokoro/graph.py) 的
`rewrite_albert_attention_double_mask_add()` 则保留两个连续的 mask Add，
对应它的 ONNX 导出路径。两者应按各自 frontend 验证，不能无条件互换。

Kokoro 的 Host mask 使用 FP16 可表示的有限值 `0 / -65504`，
避免先构造 FP32 极小值再转 FP16 得到 `-inf`。这里描述的是送入图的 mask；
不承诺两次 FP16 加法的中间结果仍有限，也不能对任意 additive bias 套用 double-add。

历史证据：[Gemma4 MaskAdd](../../../tests/gemma4/test_visual_masked_add.py)、
[Kokoro attention/ONNX 断言](../../../tests/kokoro/test_kokoro_support.py)。
这些专门断言尚不在当前 CI，不能据此标为 CI 已覆盖。
移除改写前应在目标 SDK 上同时比较 eager 数值、导出节点及实际执行结果；
仅看到普通 ONNX 可运行不足以证明目标 lowering 仍正确。
