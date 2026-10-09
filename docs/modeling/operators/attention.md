# Attention 的图表达

适用范围：当前 Merak 的 Qwen3.5/Next hybrid attention、GLM-5.3、Gemma4 视觉和 Kokoro ALBERT 路径。
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

## GLM-5.3：KDA scan 与 C4 SelectKV

GLM 的 prefill 默认使用单输出 `GDRChunkScan`，decode 默认使用单输出
`GDRRecurrentScan`。两者接受每个 key channel 的 decay，Q/K 由 GLM 的
`rsqrt(sum(x*x)+1e-6)` 归一化，不能再套一次 Ling 的 `F.normalize`。
ChunkScan 的输入 `initial_state` 是原始 CacheTensor，算子就地写回最终状态；
Host 在调用前备份该状态，失败时回滚，成功后用于后续 prefill/decode。
ChunkScan 的 `matmul_input_bit=16` 配置内部 TE MatMul 为 A16/W16；
默认 8 保留其他调用者的旧语义。`GDRBlockTriInverse` 仍按 VP 实现，不增加 bit 属性。

`SparseAttentionSelectKV` 沿用 VP 的 Mul → feature ReduceSum → ReLU LUT →
head Mul/ReduceSum → scale → stable TopK → Gather，仅返回 selected latent。
默认 `pool_size=1` 保持旧 decode 契约；GLM 使用 `pool_size=4`：选中
pool p 后，依次取原始 latent 的 `4p、4p+1、4p+2、4p+3` 行。
压缩 index key 只参与选择，attention 仍消费原始 latent。

| 输入 | GLM 的布局和单位 |
| --- | --- |
| query | `[1,H,Tq,Dindex]` |
| index_key_cache | `[1,1,Npool,Dindex]`，完整 pool 的压缩 index key |
| payload_cache | `[1,1,Ntoken,Dlatent]`，原始 token latent |
| valid_length | int32 scalar，已写入的完整 pool 数 |
| head_weights | `[1,H,Tq,1]`，可选 |
| query_valid_length | int32 `[1,Tq]`，每个 query 可见的完整 pool 数 |
| selected_kv（唯一输出） | `[1,Tq,Kpool*4,Dlatent]` |

第六个输入限制每个 query 的候选前缀。无效候选在 TopK 前设为 `-inf`，
stable TopK 使有效候选排在前面；不足 Kpool 时，剩余输出槽位置零。
因此第 t 个 query 的有效 latent 数为 `min(query_valid_length[t],Kpool)*4`，
Host 根据该有效前缀和最多三个尾部 token 的有效性，预先生成
`dsa_attention_mask`，形状为 `[1,Tq,1,Kpool*4+3]`，有效位置为 `0`，
无效位置为 FP16 有限值 `-65504`，广播到各个 attention head。
图中使用 `xhquant.nn.MaskedAdd(saturate=True)` 加到 attention scores 后再做 Softmax，
不在图中比较 slot index 或乘分数清零，不需要 SelectKV 返回第二个输出。
未形成完整 pool 的尾部 token 仍单独 gather；右侧 padding query 的有效 pool 数为零、
mask 全为 `-65504`，最终 attention 输出按 query validity 清零。

DSA 的 latent/pool KV cache 从图入口、`KVcache(axis=2)` 写回到 SelectKV
和图出口统一保持 `[1,1,N,D]`，不对整个 cache 做 squeeze、unsqueeze 或 reshape。
尾部直接在四维 latent cache 的 axis 2 上 Gather，只有选中的行重排为 query 槽位。
KDA recurrent state 保持 `[1,H,Dk,Dv]`；卷积状态和四个 token 的压缩边界状态
属于辅助状态，沿用各自的三维布局。

Host control 由旧的 `[1,Tq,Npool]` 浮点 `pool_validity` 改为 `[1,Tq]`
int32 `pool_valid_length`；ABI 3 将 `tail_validity` 替换为 Host 准备的四维
`dsa_attention_mask`。旧开发图必须重新导出，加载端拒绝旧的 state ABI。
原 `hmfp_pool` 双输出分支已移除。部署前同时更新 ModelZoo 和 xhquanttool。
数值、原始 token 展开和续接验证见
[当前 GLM 静态图测试](../../../benchmark_test/ci_test_transformer5.16.0/test_glm53_static.py)。
