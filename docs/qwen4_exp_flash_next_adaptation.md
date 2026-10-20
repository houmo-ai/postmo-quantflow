# Qwen3.8-Flash-Next / Qwen4-Exp adaptation

For the current unified build/inference entry points, visual token gears, MTP
and SDK Golden capture, see the [model README](../examples_merak/llm/qwen4_exp/README.md).
The measurements below record the earlier text-decode adaptation; they do not
constitute validation of the current unified multimodal pipeline.

This document records the architecture, precision policy, and Merak/XH2
contract used for `/data01/datasets/Qwen3.8-Flash-Next`. The comparison target
is `/data01/datasets/Qwen3.5-35B-A3B`.

## Architecture delta from Qwen3.5-35B-A3B

| Area | Qwen3.5-35B-A3B | Qwen3.8-Flash-Next | Adaptation consequence |
| --- | --- | --- | --- |
| Text width / depth | 2048 / 40 | 2560 / 48 | New shapes; no Qwen3.5 graph reuse |
| Hybrid layers | 30 DeltaNet + 10 full attention | 36 DeltaNet + 12 QSA attention | Same 3:1 cadence, different full-attention implementation |
| Attention | 16 Q, 2 KV, head dim 256 | 24 Q, 2 KV, head dim 256 | New grouped-query shapes |
| DeltaNet | 16 K heads, 32 V heads, dim 128 | 16 K heads, 48 V heads, dim 128 | Larger recurrent and convolution cache ABI |
| Routed MoE | 256 experts, top-8, width 512 | 512 experts, top-10, width 640 | Expert packing and routing shapes differ |
| Shared expert | Width 512 | Width 640 | Kept separate from routed W4 experts |
| Hyper-connection | Absent | 4 streams, low rank 320 | Per-channel sigmoid mixing requires a new final mixer and cache-independent high-precision path |
| PLE/ngram | Absent | Layer 2, ngram 3, 8 heads/ngram, base vocabulary 20M | Adds embedding, dilated-convolution state, and two-token ID history |
| QSA indexer | Absent | 4 Q heads, 1 KV head, dim 128, budget 2048, compression 4 | Adds indexer-key cache and sparse block selection |
| Context | 262144 | 262144 | Position range is unchanged, but QSA needs full cached positions |
| Vision | Qwen3.5-VL family | 27 blocks, width 1152, FFN 4304, 16 heads, output 2560 | Vision-only mean-preserving QuaRot is supported |

The real Qwen4-Exp text cache is heterogeneous:

- DeltaNet layers carry convolution and recurrent state.
- PLE layers additionally carry dilated-convolution state and raw ngram token history.
- QSA layers carry KV tensors plus indexer-key state.
- Full M-RoPE position history is an explicit model-level cache input/output.

These tensors are defined by `Qwen4ExpCacheContract`; the exporter does not
pretend that Qwen4-Exp has the Qwen3.5 cache layout.

## Precision policy

| Module | Weight / activation | Rationale |
| --- | --- | --- |
| Routed expert gate/up/down | W4A8, group 64 | Dominant checkpoint size; requested target |
| Shared expert gate/up/down | W8A8, group 64 | Runs for every token and is more sensitive than routed experts |
| QSA q/k/v/o | W8A8, group 64 | Token-critical attention projections |
| DeltaNet qkv/z/out | W8A8, group 64 | Main attention-equivalent projections |
| LM head and token/vision bridge | W8A16 | Preserve output and cross-modal sensitivity |
| QSA indexer and q/k norms | BF16 / W16A16 | Determines which blocks are visible |
| DeltaNet a/b, convolution, state, norm | BF16 / W16A16 | Recurrence and decay are numerically sensitive |
| Hyper-connections and final mixer | BF16 / W16A16 | Sigmoid-gated elementwise mixing |
| PLE/ngram embedding and convolution path | BF16 / W16A16 | Hash lookup and lexical injection affect every residual stream |
| MoE router and shared-expert gate | BF16 / W16A16 | Small routing errors change expert selection |

Language-side dense QuaRot is rejected deliberately. Qwen4-Exp's
channel-wise hyper-connection gates and PLE injection do not commute with a
single hidden-space rotation. The supported QuaRot mode is confined to the
conventional vision transformer and uses an orthogonal transform that fixes
the all-ones direction, so LayerNorm mean subtraction remains invariant.

## Implementation and artifacts

- GPTQModel definition and AutoRound recipe live in the sibling `gptqmodel`
  workspace under `gptqmodel/models/definitions/qwen4_exp.py` and
  `gptqmodel/recipes/qwen4_exp.py`.
- Merak workflow configuration:
  `configs_merak/workflows/xh2a/llm_models/qwen4_exp/flash_next/qwen3_8_flash_next_hf_autoround_xh2a_256k.yaml`.
- Requested quantized checkpoint location:
  `/data01/home/xuzk/datasets/Qwen3.8-Flash-Next-AutoRound-W4A8-G64`.
- Completed XH2/HMONNX artifact location:
  `/data01/home/xuzk/datasets/Qwen3.8-Flash-Next-AutoRound-W4A8-G64-XH2`.
- The workflow loads either the raw HF model or a GPTQModel-HF artifact,
  converts packed linears/MoE blocks, exports a cached text-decode HMONNX,
  executes one HMONNX decode turn, and writes all graph inputs/outputs as
  Golden tensors.

Qwen4-specific frontend mappings and the PLE operator registration are scoped
in `xhmodel_merak/xh_llm/models/qwen4_exp/xhquant_extensions.py`. Standard ONNX
parsers, Conv1d runtime ABI and HMONNX serialization belong to
`xhquanttool`; the model no longer replaces those parsers. The exporter captures
the complete output/state program once and passes it to the existing public
`to_export_hmonnx_v2(..., save_as_external_data=True)` API with the default
optimization level of `0`.
There is no separate serialization mode. SDK fixes are tracked in
[47747](http://gerrit.houmo.ai/c/houmoquantization/xhquanttool/+/47747).

The shared Qwen3.5 DeltaNet length helper includes a two-line plain-`int`
fast path. Qwen4 reuses that mixin, and
TorchDynamo otherwise attempts to inspect `int.__class__.__name__` and aborts
the export. Packed-MoE handling remains entirely inside the Qwen4 adapter.

The Qwen4 exporter enables the default Torch/ONNX optimizers and requests ONNX
external-data serialization. It requires SDK support for saving external weights
before protobuf conversion and optimizing the graph with lazy external-data
references. Explicit state outputs remain part of the export contract. The SDK
lifts static constants for runtime parsers and applies its existing HMONNX transforms.
Host PLE data remains outside the serving graph. Imported quantized linears
carry their checkpoint's declared `logical_bits`, including W4 and W8; their
width is not inferred from the observed range of integer codes.

The model-owned RMSNorm wrapper maps native `Qwen4ExpTextRMSNorm` to the SDK
`RMSNorm` before tracing, for target prefill/decode and MTP alike. It retains
the checkpoint's original zero-centered weight and applies `(1 + weight)`
after normalization in FP32, before restoring the input dtype. Grouped HC
normalization reshapes into hidden-size groups (2560 for Flash Next), rather
than reducing across all four streams (10240). The SDK norm uses a one-dimensional
unit scale stored as a nonpersistent buffer; checkpoint keys and parameter
identity remain intact. GDR `RMSNormGated` also uses SDK `RMSNorm`, with its
original head-size weight directly (without adding one), followed by the
checkpoint-selected sigmoid or SiLU gate and multiplication. Its weight keeps
the original parameter/key; a nonpersistent SDK buffer aliases that parameter.
Eager
wrapper parity and deployed SDK precision are checked separately: XH2 lowers
floating Casts to FP16 and the fused norm uses `force_fp32` accumulation,
so SDK replay checks fused-operator execution and grouping separately from
native wrapper parity.

Before PTQ, the Qwen4 frontend removes the private GDR input-projection
`Split → ordered Q/K/V reshapes → Concat` identity. The convolution consumes
the original combined projection; the temporal cache concatenation and the
necessary post-convolution Q/K/V split remain. The rewrite requires complete
sections, matching shapes/dtypes/axes and no shared intermediate users. Packed
projections retain their original owner path through Dynamo restoration, so
the same rule applies after their temporary module names change. Shared Qwen3.5
and Qwen3-Next projection implementations are unchanged.

The formal export path also removes static inverse/no-op reshapes, inverse
transposes and duplicated boolean-to-FP16 mask views. Shape constants, tensor
shapes/dtypes and known operator attributes must prove the rewrite; public
outputs and cache aliases are preserved. Cleanup happens on the complete graph
before serving components or inspection blocks are extracted, retaining the
external weight files and the required cache/convolution layouts.

## Completed real-checkpoint evidence

The real reduced-calibration run completed on 2026-09-02:

- GPTQModel-HF checkpoint: `169 GiB`, `20` indexed SafeTensor files, with all
  `297247` indexed tensor keys present.
- GPTQModel reload: `73728` W4 group-64 routed-expert linears and `300` W8
  group-64 attention/shared linears.
- HMONNX graph: `48 MiB` protobuf plus `216 GiB` external data; all `2006`
  external initializer ranges are non-empty and within the external-data file.
- Graph stages: `11153` frontend nodes, `12155` quant nodes, and `16603` export
  nodes; the serialized HMONNX contains `11262` nodes and passed ONNX checker.
- HMONNX Golden: `114` inputs, `112` outputs, and all `112` persisted output
  tensors are finite.
- Conversation validation: the native seed token followed by one HMONNX decode
  token produced `你好！` for the configured Chinese introduction prompt.

The complete XH2 directory occupies about `432 GiB` because Golden execution
keeps its own copy of the `216 GiB` external-data file alongside the canonical
text-decode artifact.

## Verification contract

Completion requires all of the following on the real checkpoint:

1. Quantized checkpoint save finishes and its tensor/index/config manifest is complete.
2. GPTQModel reload succeeds and module precision matches the policy above.
3. `AutoLLMWorkflow` resolves the Qwen4 workflow from the YAML config.
4. The real packed checkpoint exports to checker-valid HMONNX.
5. HMONNX cached decode returns finite hidden state and all expected cache outputs.
6. The native-seeded/HMONNX-decoded conversation report is readable and Golden files are present.

The fast functional quantization run uses reduced calibration (`8` samples,
sequence length `512`, `2` AutoRound iterations). The committed workflow keeps
the quality-oriented defaults (`128`, `2048`, `200`) for production reruns.
