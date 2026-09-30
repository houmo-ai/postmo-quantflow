# Cache 与状态契约

## 长度、布局和所有权

`past_seq_length` 是已经提交的有效历史长度，`current_input_length` 是本次有效输入长度，
不能用静态图容量或 padding 后长度替代。cache 的轴、输入顺序和更新方式由具体图 ABI 定义。
例如 [HybridGatedAttentionMixin](../../../xhmodel_merak/xh_llm/models/qwen3_5/_hybrid_gated_delta_net.py)
的 K/V 按 `[B, Hkv, S, D]` 送入 `LLMCacheV2`，更新轴来自配置。
这不能推广成所有模型、所有 cache 都固定 axis=2。

当前 [MiniCPM-V-4.6 CPU 契约](../../../benchmark_test/ci_test_transformer5.16.0/test_minicpm_v_4_6_preprocess_contract.py)
调用真实 export/runtime preprocessor factory，用小 tensor 检查 padding、连续位置、
首个 decode 的有效长度、flat/grouped conv cache 和 page-attention 的输入顺序。
其 [impact 声明](../../../benchmark_test/ci_test_transformer5.16.0/test_minicpm_v_4_6_preprocess_contract.impact.json)
同时覆盖两个调用方及共享 preprocessing/cache helper。
它验证文本输入接口，不验证完整 attention 数值或 NPU cache 写回。

## 相同浮点值不代表同一量化状态

Ling XH2a 的 K 存 latent＋RoPE，V 单独存 latent。两者指数对齐轴不同，
必须各自维护数据与指数，不能按 GPU 单 latent cache 的做法合并物理存储。
显式 attention 与 FlashAttention 使用相同 cache ABI，详见 [Ling MLA](ling-mla-quantization.md)。

## 混合注意力与投机验证

[hybrid_cache_runtime](../../../xhmodel_merak/xh_llm/models/qwen3_5/hybrid_cache_runtime.py)
维护 Qwen3.5 Dense/MoE 与 Qwen3-Next 共用的 cache ABI：除了 K/V，
还有 convolution cache 和 recurrent state。K/V 长度回退并不能恢复 recurrent state。

- verify 的每个候选 step 有对应状态快照，提交接受前缀时选中对应快照；拒绝分支不能污染正式状态。
- split conv cache 有完整 q/k/v 分组，输入扁平化、输出名称和 layer/step 的顺序必须一致。
- prefill recurrent state 是否原地更新由导出 metadata 的
  `prefill_recurrent_state_uses_cache` 表达；不能只根据 `fuse_gdr_ops` 猜测。
- `normalize_hybrid_hmonnx_args()` 的整数 tensor 转 int32 是这条 HMONNX 边界的行为，
  不意味着 Host tokenizer 或所有 RoPE 输入都应改成 int32。

历史 [transaction test](../../../tests/qwen3_5/test_hybrid_cache_transaction.py)
有 accepted-prefix、零 draft 接受、abort 和模型状态切换断言，尚未纳入当前 CI。
新增状态回归应在当前 CI 重现“预填充 → verify → 部分接受/拒绝 → 下一步 decode”，
比较状态与结果；不要仅以输出个数或最终文本可读判断 ABI 正确。
新设备若改变 cache 所有权或写回形式，应由 adapter/metadata 显式转换，保留相同的提交语义。

## DFlash2：滑窗上下文与临时查询 KV

`qwen_dflash2_hybrid_v1` 使用独立的 draft 缓存，布局为 `[B, Hkv, S, D]`。
主模型的 context 长度和 FlashAttention 设置不决定 draft 的缓存容量。
设滑窗大小为 `W`、prefill chunk 为 `C`、draft block 为 `B`（含 anchor），则
容量为 `align16(W + max(C, B) + max(B - 1, 16))`；当前 Qwen3.8-27B DFlash2 的
`W=2048, C=256, B=8`，容量为 2320。

- context prefill/context decode 通过 `LLMCacheV2(attention_max_length=W)` 更新
  `HybridCacheTensor`：保留已提交前缀末尾最多 `W-1` 个 KV，移到开头，再追加本次有效
  context KV。更新后的有效长度 `H` 是保留长度加本次有效长度，可能大于 `W`。
- spec decode 的 `past_seq_length` 仍是绝对位置，用于 RoPE；独立输入
  `cache_position=H` 表示紧凑存储中的写入位置，两者不能互换。
- spec decode 为 anchor 和 mask token 计算临时 K/V，写在 `[H,H+B)`。
  运行时传入共享 `.data` 的普通 `CacheTensor`，使写入不推进 HybridCache 的已提交长度。
  下一次 context 更新只读取已提交前缀，并覆盖临时区域。
- draft attention 使用 dense attention 和 `MaskedAdd`，mask 同时限制有效 KV 和每个
  query 的滑窗范围；此 ABI 不使用 draft FlashAttention 或 `query_kv_range_abs`。
- 新请求必须清零 HybridCache 的数据和有效长度。旧 DFlash 沿用原 cache/FA 契约，
  不采用 DFlash2 的滑窗容量和额外输入。

[DFlash2 cache 回归](../../../benchmark_test/ci_test_transformer5.16.0/test_dflash2_hybrid_cache.py)
用真实 attention 比较紧凑 cache 与完整 cache 在部分提交、重复滑窗移动后的数值，
并通过真实 HMONNX context 图验证 HybridCache 更新。该小模型证据不代替完整模型的
生成、接受率或目标芯片写回验证。
