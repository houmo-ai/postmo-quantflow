# Qwen3.8-Flash-Next / Qwen4-Exp adaptation

For the current unified build/inference entry points, visual token gears, MTP
and SDK Golden capture, see the [model README](../examples_merak/llm/qwen4_exp/README.md).
The dated evidence below separates historical full-checkpoint exports from
current wrap/TorchFX graph validation and tiny multimodal SDK validation.

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
| Context | 262144 | 262144 | Position range is unchanged; the host supplies current and C4 group positions |
| Vision | Qwen3.5-VL family | 27 blocks, width 1152, FFN 4304, 16 heads, output 2560 | Vision-only mean-preserving QuaRot is supported |

The real Qwen4-Exp text cache is heterogeneous:

- DeltaNet layers carry convolution and recurrent state.
- PLE layers additionally carry dilated-convolution state and raw ngram token history.
- QSA layers carry KV tensors plus indexer-key state.
- The host supplies current T/H/W and C4 group-start M-RoPE positions.

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
| PLE/ngram path | Host SEFP4 lookup; BF16 / W16A16 convolution | Hash lookup and lexical injection affect every residual stream |
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
- Historical XH2/HMONNX artifact location (see the 2026-09-02 evidence):
  `/data01/home/xuzk/datasets/Qwen3.8-Flash-Next-AutoRound-W4A8-G64-XH2`.
- The workflow loads either the raw HF model or a GPTQModel-HF artifact and
  converts packed linears/MoE blocks. The current static lifecycle exports
  prefill/decode; serving configuration can additionally export target verify
  and MTP draft graphs. SDK Golden capture is a separate validation step.

Qwen4-specific frontend mappings and the PLE operator registration are scoped
in `xhmodel_merak/xh_llm/models/qwen4_exp/xhquant_extensions.py`. Standard ONNX
parsers, Conv1d runtime ABI and HMONNX serialization belong to
`xhquanttool`; the model no longer replaces those parsers. Ordinary, streamed
and MTP exports use static wrappers through the standard
wrap → TorchFX frontend → quant → HMONNX lifecycle, matching GLM5.3. The final
state argument is a nested tuple indexed by the fixed cache contract.
Text serialization uses the existing public
`to_export_hmonnx_v2(..., save_as_external_data=True)` API and SDK defaults.
Vision exports the stem, each block and merger directly from QuantGraph via
`to_export_hmonnx_from_quanted_graph`, with explicit PTQ device selection.
Each component's weights are released before loading the next component.
There is no independent Dynamo exporter or module-restoration path.
SDK fixes are tracked in
[47747](http://gerrit.houmo.ai/c/houmoquantization/xhquanttool/+/47747).

Packed-MoE handling remains inside the Qwen4 adapter; delta-rule kernels remain
in the existing shared implementation. The wrapper expresses static shapes,
cache updates and numerical operations directly, without global capture patches.

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

The Qwen4 GDR source keeps the combined QKV projection merged through convolution;
the temporal cache concatenation and necessary post-convolution split remain.
Grouped RMSNorm preserves batch/sequence axes and uses native flattening to
restore features. PLE normalization consumes its already-grouped streams, QSA
constructs the required grouped query and final output layout directly, and MTP
passes grouped fusion into its first HC norm while retaining a flat residual.
The mask is cast once before its shared views, and EOS is a scalar tensor.
These expressions replace the former frontend peephole and global layout cleanup.

M-RoPE tables are built in FP32, then converted once to the native model's floating
dtype before frontend capture. For the validated FP16/CPU PTQ route, all T/H/W
Gather registrations retain one cosine and one sine table. FP32 native eager
outputs retain their original precision.

The model-local `publish_qwen4_exp_qsa_layout_resources` publication step preserves
QSA layout constants and degenerate views required by the old artifact contract.
It derives them from current graph/configuration geometry, checks tensor shapes,
and does not read the old baseline or change active connections, precision,
weights or SDK diagnostics. This bounded compatibility step is distinct from
the historical whole-graph cleanup.

## Historical real-checkpoint evidence (2026-09-02)

The earlier adapter's real reduced-calibration run completed on 2026-09-02;
these full-network results have not been rerun through the current wrap path:

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

A complete real-checkpoint validation requires all of the following:

1. Quantized checkpoint save finishes and its tensor/index/config manifest is complete.
2. GPTQModel reload succeeds and module precision matches the policy above.
3. `AutoLLMWorkflow` resolves the Qwen4 workflow from the YAML config.
4. The real packed checkpoint exports to checker-valid HMONNX.
5. HMONNX cached decode returns finite hidden state and all expected cache outputs.
6. The native-seeded/HMONNX-decoded conversation report is readable and Golden files are present.

The fast functional quantization run uses reduced calibration (`8` samples,
sequence length `512`, `2` AutoRound iterations). The committed workflow keeps
the quality-oriented defaults (`128`, `2048`, `200`) for production reruns.

## Historical static lifecycle validation (2026-11-12)

This evidence used the previous Dynamo capture/restoration and layout-cleanup
path. That implementation has been superseded by the wrap/TorchFX path above;
the artifacts and measurements below remain historical records.

[Gerrit 48561](http://gerrit.houmo.ai/c/houmoquantization/xh2modelzoo/+/48561)
adds the ordinary model-owned export lifecycle. The ordinary unified YAML drops
`export_mode: serving`; the default static entry exports prefill/decode, while
MTP remains an explicit serving configuration. The README describes this entry
and the focused impact declarations cover its configuration and visual hooks.
The model configuration rejects unsupported batch/sequence geometry before
loading checkpoint weights.

The local evidence directory is
`work_dirs/qwen4_static_lifecycle_20261112/`; its formal package is
`first4_package/`. Fresh text capture used the existing RTN checkpoint, its first
four blocks, context 262144 and prefill chunk 256. The comparison baseline was
`work_dirs/qwen38_first4_hmonnx_latest_three_repos_20261112`.
The dependency revisions were GPTQModel `1c04f7fd40047e79a762316c4e67fcd21111c1c6`
and xhquanttool `cceedf68033c197546b885898b08eca14bb445f0`.

| Component | Ordered nodes | Initializers | Inputs | Outputs |
| --- | ---: | ---: | ---: | ---: |
| prefill | 667 | 381 | 30 | 9 |
| decode | 600 | 353 | 30 | 12 |

Both fresh graphs pass ONNX checker. Ordered operator attributes, connections,
IO shapes/types, opsets and local functions match the baseline. A bounded-memory
comparison read 181915468263 bytes across 45 unique files: all 734 initializer
byte ranges, both complete text sidecars including offsets/padding, all four PLE
packed/exponent files, token embedding and HF assets match their baseline bytes.
Five actual visual gears (96/196/384/704/1536) also pass ONNX checker. The formal
package passes the Qwen4 serving contract and public `export_meta_info.json`
validation. This is not a full `build_model` transaction or build Manifest test.

**Strict protobuf/file-name parity did not pass.** Fresh SDK graphs retained new
process IDs in initializer names, generated stack traces and PLE UUID paths.
Removing capture-only outputs also removes the former `USER_OUTPUT` diagnostic
annotation on the PLE input. Each graph additionally retains the three required
FP16 internal cache ValueInfo entries. SDK metadata was preserved; no graph was
anonymized or rewritten to make the exact comparison pass. The exhaustive local
report is `first4_package_parity.json`, with ordered-computation supplements in
`prefill_semantic_parity.json` and `decode_semantic_parity.json`.

The matched Golden capture contains one prefill and two decode steps: 207 listed
arrays have the declared shape/dtype, logits are finite, both 14-state handoffs
match exactly, and three HostPLE sample replays match exactly. This is a four-block
smoke with 59 prompt tokens and three generated tokens. It does not establish
full-network quality, full-context occupancy, rejection/rollback, verify/MTP,
HMCC or TCIM behavior. See `golden_manifest_validation.json` in the evidence root.

Focused checks include the real streamed prefill/decode replay tests and
`test_static_model_publishes_two_streamed_phases_without_capture_files`, which
runs the standard model entry and checks that no capture-only graphs or child
weight copies are published. Lifecycle/runtime, legacy/disabled visual geometry,
relative output paths and declared precision snapshots have separate regressions.
The 41-case runtime suite also loads a complete tiny conditional-generation
checkpoint and checks every state-dict key, shape, dtype and tensor value before
wrapping. Positional slots, padding and nonzero past lengths are checked for
prefill/decode; cache rejection tests require the specific ValueError messages.
The local XH2 Python 3.10 runs used a process-only `typing_extensions.Self` alias;
publication checks also used the installed `aenum.StrEnum` alias. Repository
Python requirements and shared SDK sources were unchanged. A broader local run
was interrupted with SIGTERM (143), so it is not recorded as a full-suite pass.

## Wrap/TorchFX strict graph validation (2026-11-13)

The same real RTN four-layer checkpoint/configuration was fully re-exported via
`XHQwen4ExpModel.export_hmonnx()`: three GDR blocks and one QSA block, W4/W8 G64,
Host SEFP4 PLE, context capacity 262144, prefill 256 and decode 1. The source
model was FP16 and PTQ ran on CPU. The export did not reuse baseline graphs or
apply an M-RoPE publication patch.

| Graph | Old / new nodes | Old / new initializers | Complete DAG |
| --- | ---: | ---: | --- |
| prefill | 667 / 667 | 381 / 381 | PASS |
| decode | 600 / 600 | 353 / 353 | PASS |

The comparison preserves operator/domain/attributes, every port and shared
connection, all initializer dtype/shape/payloads including unused resources,
ordered public IO and opset/IR contracts. Both graphs pass ONNX checker. PLE
packed/exponent resources total 65,280,313,344 bytes (65.28 GB) across the two
phases; full SHA256 reads match, as do PLE metadata semantics apart from resource
file names. Internal names, debug metadata, storage paths/offsets and ValueInfo
differences are audited separately; protobuf byte equality is not claimed.

Local evidence is under `work_dirs/qwen4_structure_alignment_20261113/`:

- `VALIDATION.md`, `first4_final_comparison.json`, `ple_resources_comparison.json`
  and `first4_serialization_differences.json` record the four-layer comparison.
- `five_graphs_source_final_comparison.json` and
  `five_graphs_source_final_summary.md` record final-source target prefill,
  decode, verify4 and MTP prefill/decode: all five DAGs match the old exports,
  and all four checkpoint source files have identical SHA256 values. All four
  integration cases pass, including actual CPU SDK smoke execution of all five graphs.
- `structure_all_final.log` records 18 passing structural regressions, including
  real ordinary two-phase export and negative port/attribute/sharing/constant checks.
- `ordinary_source_final_comparison.json` and
  `runtime_source_final/replay_ab_report.json` record a controlled small model's
  strict two-phase match and nonzero-input prefill plus four decode steps;
  every output/state and the independent same-state decode are bitwise equal.

The real-checkpoint conclusion is limited to this four-layer FP16/CPU
configuration. It does not establish 48-layer, GPU, visual, HMCC/TCIM or new
four-layer Golden validation. The numerical A/B replay is from the small model.
FP32 eager parity is tested; FP32-source large-table sharing through FP16 PTQ
is outside this strict export result. These focused checks are not a full CI run.

## GLM5.3 vision lifecycle alignment (2026-11-13)

Qwen4 now follows GLM5.3's vision export and inference ownership:

- Export a meta-shell stem, one block at a time and the merger, releasing
  each component's loaded weights before loading the next one. TorchFX feeds
  QuantGraph and its direct HMONNX serializer; PTQ receives the requested
  CPU or CUDA device explicitly.
- Publish component ABI 2 under `qwen4_exp_vision`, with a root
  `visual_gears.json` and `visual_m<N>/{stem,block_XX,merger}.onnx`.
  Build and loading validate ordered components, native geometry, graph IO,
  confined paths and checksums. Existing text, source, PLE and MTP validation
  remains active. Historical whole-gear artifacts have a loading path only.
- The loaded model owns persistent vision sessions and a cached native
  processor. `model.generate` accepts native processor outputs for images,
  videos and mixed media, and performs fusion and three-axis M-RoPE setup.
  Unified CLI accepts one image or video; multiple media use the model API.
- Real tiny conditional-model regressions save a checkpoint, export all
  components, move the package, delete the source checkpoint and load through
  the public HMONNX entry. They compare actual SDK visual features and
  conditional language logits with the same native model, and exercise
  generation, repeated requests, error recovery and actual component Golden.
  MTP additionally checks native assistant prefill/decode, target verify
  acceptance/rejection, every live state after rollback and speculative
  tokens against target-only greedy generation. Repeated requests retain
  model-owned graph templates while sharing pooled/lazy weight storage.

The final case union is five native media references, five actual SDK media
cases and one actual SDK MTP case. Visual features require cosine above 0.99;
language and assistant logits/hidden states require above 0.98, with elementwise
tolerances as well. Component export/runtime has 19 passing cases and the
publication/load contract has 33. The 18-case strict text graph regression
also passes. The evidence report records separate runs and corrected attempts;
these counts are not a single combined CI result.

Qwen's five post-merge token gears, default W8A8 vision precision, learned
position interpolation, head-72 padding and native LayerNorm/merger are retained.
The component lifecycle bounds loaded model parameters; the regression does
not measure process RSS. Tiny-model native references, vision PTQ and SDK
execution use CPU. Ordinary text PTQ retains its inherited CUDA selection;
MTP serving text/draft PTQ uses CPU. The 27-layer visual tower and full-network
quality/HMCC/TCIM are outside this validation. Evidence is recorded under
`work_dirs/qwen4_glm53_vision_alignment_20261113/` and
`work_dirs/qwen4_multimodal_e2e_20261113/`.

## Qwen3.5 MTP scheduling alignment (2026-11-13)

New MTP exports use runtime contract v4 and default to four draft candidates
with a five-token target verifier. Candidate count and physical verify width
are validated together before loading weights. Existing v2 single-step and
v3 single-candidate packages retain their loading paths. The model keeps its
dedicated target decode1 graph for recurrent replay, so the language package
still contains target prefill/decode/verify and draft prefill/decode graphs.
Qwen4's HC projections, grouped normalization, QSA, PLE and M-RoPE remain native;
the Qwen3.5 concatenating projection and cache layout are not substituted.

The target warms up the shifted assistant inputs after each prompt block,
retaining only the final target HC for the next block. Draft graphs share the
target's fixed quantized output head, using the existing shared-head binding
before PTQ. All five graphs reuse its quantized weight and scale payloads.
Resume binds the fixed head to the source head/assistant tensor payloads and
records each draft only after successful publication. It checks the recorded
graph and external-weight digests before reuse, rejecting changed sources,
changed outputs and unrecorded partial drafts. The fixed-head checkpoint is
an export intermediate and is removed from the published runtime package.
Model-owned draft SDK sessions persist between requests, their state resets
at request boundaries, and `model.close()` releases them. Per-request statistics
include round count, proposed/accepted/rejected candidates and acceptance rate.

Each speculative round snapshots the target's physical verifier write range
and the draft's candidate writes plus its possible final shifted pair. Partial
rejection restores both models, replays the committed target prefix with
decode1 and repairs draft inputs with the corresponding actual target HC.
Full acceptance retains recursively predicted draft HC and completes the last
shifted pair with target HC. Exceptions restore bounded K/V and QSA regions,
tails, recurrent/PLE state and C4 group positions in place. Golden records
group positions before execution and after a successful state commit.

Real tiny-model validation exports, relocates and loads an N4 package after
deleting source weights. It compares SDK/native target and recursively fed
assistant logits/HC, all five shared-head payloads and actual MTP Golden.
Controlled production rounds accept zero, two and four proposals: only proposal
selection is injected after actual SDK draft execution, while HC and caches
remain real. Every target and draft state is compared with an independent SDK
oracle. Natural speculative output is checked against target-only greedy
generation, with persistent sessions, repeated requests, failure recovery and
close/reopen. Separate real N1/v3 and N2/verify3 cases cover compatibility and
non-default geometry. The previous pipeline's five ordinary SDK media cases
also pass with the native reference explicitly using its PyTorch CPU kernels.

Evidence and exact source/validation boundaries are recorded under
`work_dirs/qwen4_mtp_qwen35_alignment_20261113/`,
`work_dirs/qwen4_mtp_alignment_20261113/` and
`work_dirs/qwen4_glm53_vision_compare_20261113/n4_generation_validation.json`.
The random tiny assistant naturally rejected all 26 proposals in the diagnostic
ten-token run; controlled acceptance tests establish state correctness, not
acceptance quality or speedup. Full-checkpoint MTP quality, GPU inference,
HMCC and TCIM remain outside this validation.
