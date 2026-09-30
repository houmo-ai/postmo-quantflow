# Qwen3.5 W4 speculative decoding validation

## Runtime contract

The packed-W4 narrow-token contract is:

- `M=1`: normal autoregressive decode;
- `M=2..16`: speculative target verification (`M=K+1`);
- CUDA execution must consume the load-time packed W4 weight directly;
- `XHQUANT_FORBID_CUDA_W4_UNPACK=1` turns any CUDA unpack fallback into a
  failure.

The contract is enforced in three places:

1. xhquant dispatch covers W4A8/W4A16, CUDA/Triton activation packing, and
   every `M=1..16`;
2. speculative artifact loading rejects a verify width outside `2..16`;
3. the Merak model runner requires `M=1` for decode and `M=2..16` for verify.

Validation results:

```text
xhquant packed-W4 focused matrix: 140 passed
vLLM artifact/runner/backend matrix: 100 passed
```

## DFlash W4 artifact

Artifact:

```text
/data01/home/yujy/work/xh2modelzoo/work_dirs/
qwen3_5_9b_dflash_flashattention_fuse_gdr_export/
hmquant_xh2_qwen3_5_9b_dflash_w4a8_256_256k_mpe256k_448x448_20260727
```

The vLLM loader audit reports exactly one W4 draft head:

```text
nodes=1
initializers=1
source_nbytes=1017118720
packed_nbytes=508559360
saved_nbytes=508559360
```

The DFlash context and context-decode graphs contain no LM head and therefore
contain no W4 Linear. Target prefill and decode each contain 176 packed W4
Linear nodes.

Qwen3.5 workflow finalization now writes `merak_config.json` next to every
normal/MTP/DFlash HMONNX export. The page-attention value is derived from the
final exported metadata, including CLI overrides; deployment no longer relies
on a manual file copy.

## vLLM A/B

Hardware: one NVIDIA A100-SXM4-80GB. Request: 32 prompt tokens, deterministic
64-token completion. The reference completion SHA256 is:

```text
ecce25c2547c4a1452867bd49b21cd3766a1513f3c678b99f3aff3ac10ee03eb
```

| Mode | Median latency | Steady GPU memory | Exact target output |
| --- | ---: | ---: | --- |
| target only | 13.21 s | - | yes |
| historical DFlash W8 draft | 32.28 s | 34624 MiB after load | yes |
| DFlash W4 eager | 14.82 s | 23150 MiB | yes, 3/3 |
| DFlash W4 decode CUDA Graph | 14.93 s | 23248 MiB | yes, 3/3 |
| DFlash W4 eager, controlled rerun | 14.828 s | 23172 MiB | yes, 10/10 |
| DFlash W4 `context_decode` CUDA Graph | **14.698 s** | 23214 MiB | yes, 10/10 |
| DFlash W4 decode + `context_decode` graphs | 15.187 s | 23278 MiB | yes, 3/3 |
| historical MTP eager | 54.09 s | - | yes |
| MTP W4 decode CUDA Graph | 19.83 s | 30713 MiB | yes, 3/3 |

DFlash W4 eager is `2.18x` faster than the historical W8-draft deployment and
is only `12.2%` slower than target-only generation for this prompt. Its measured
draft acceptance was `177/279 = 63.4%`, which is expected for this DFlash
checkpoint.

The DFlash block decode already executes one fixed `M=10` graph per proposal.
Capturing that graph alone measured no latency gain (`0.993x`). MTP executes
repeated `M=1` draft steps, so decode CUDA Graph remains valuable there
(`2.73x` over its eager measurement).

The initial `context_decode` capture failure was not a cache-sharing failure.
`context`, `context_decode`, and draft decode receive the same ten persistent
full-capacity `CacheTensor` objects. Their addresses are checked after every
run. Terminal context outputs alias those input objects and are discarded, so
no full cache is copied or rebound; only the newly accepted K/V rows are
written.

The failure came from using `.item()` solely to calculate the dynamic write
offset on the host. The in-place cache update now has a capture-safe,
device-side offset path. It replays dynamic current lengths `1..16` while
retaining the original cache addresses. With that fix:

- isolated `context_decode` median latency fell from `8.163 ms` to `2.345 ms`
  (`3.48x`, 100 iterations);
- controlled end-to-end median latency fell from `14.828 s` to `14.698 s`
  (`1.0089x`, ten deterministic 64-token requests);
- eager and graph runs had identical `568/909` accepted draft tokens;
- all ten completions matched the target SHA256;
- startup memory was `15026 MiB` eager versus `15066 MiB` with only
  `context_decode` captured, and post-request memory differed by `42 MiB`.

The safe deployment setting is therefore to capture `context_decode` without
capturing DFlash block decode:

```bash
VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH=0
VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH=1
```

The initial `M=256` context graph remains eager. Combining both DFlash graphs
was slower in the end-to-end test, so the two capture controls are independent.

## Memory and evidence

The monitored load-time pack window was flat:

```text
pre_pack_last_mib=9630
pack_min_mib=9630
pack_max_mib=9630
gpu_growth_mib=0
```

Detailed evidence:

```text
tmp/spec_decode_validation/qwen35_dflash_w4_eager_server_20260727.log
tmp/spec_decode_validation/qwen35_dflash_w4_eager_exact_prompt64_repeats.json
tmp/spec_decode_validation/qwen35_dflash_w4_draft_cudagraph_server_20260727.log
tmp/spec_decode_validation/qwen35_dflash_w4_draft_cudagraph_exact_prompt64_repeats.json
tmp/spec_decode_validation/qwen35_dflash_w4_context_cudagraph_server_20260727.log
tmp/spec_decode_validation/qwen35_dflash_w4_context_cudagraph_exact_prompt64_repeats_20260727.json
tmp/spec_decode_validation/qwen35_dflash_w4_eager_fresh_exact_prompt64_10_repeats_20260727.json
tmp/spec_decode_validation/qwen35_dflash_context_decode_session_eager_bench_20260727.log
tmp/spec_decode_validation/qwen35_dflash_context_decode_session_cudagraph_bench_20260727.log
tmp/spec_decode_validation/qwen35_spec_decode_w4_performance_summary_20260727.json
```

## Paged draft-cache result

The final Qwen3.5 MTP and DFlash ABI is no longer the contiguous-cache ABI
measured above.  The exported HMONNX files remain unchanged, while Merak
creates a small adjacent `*.merak_page_attention.onnx` view at load time.  The
derived graph keeps the original external-data references, so it neither
duplicates draft weights nor changes the offline export format.

The runtime follows the same design used by upstream vLLM speculators:

- DFlash context graphs only project target hidden states to K/V and write
  those rows into five private HMFP page caches.  Block decode reads and
  updates the same pages.
- MTP prefill and decode share one private HMFP page cache.  The four
  autoregressive token-feedback steps stay on the device and perform one D2H
  transfer only after the complete proposal.
- Private cache capacity follows the static HMONNX cache shape exactly.  A
  262144-token export allocates 262144-token MTP/DFlash draft caches regardless
  of the serving process' current `max_model_len`.
- The W4 draft head is packed once and shared by its prefill/decode graph.

Real single-request vLLM Merak measurements on an A100-80GB, with a 64-token
completion, are:

| Prompt | Target M=1 | MTP K=4 | MTP speedup | DFlash K=9 | DFlash speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.619 tok/s | 46.874 tok/s | **1.904x** | 71.247 tok/s | **2.894x** |
| 1024 | 24.751 tok/s | 40.795 tok/s | **1.648x** | 68.158 tok/s | **2.754x** |

The MTP compiler prompt accepted `65/76 = 85.5%` of draft tokens.  The
128-token throughput workload accepted `208/240 = 86.7%`.  DFlash's mixed
post-restart validation accepted `347/414 = 83.8%`; its dedicated throughput
run reached about 88%.  Both paged modes reproduced the deterministic target
completion SHA256 exactly:

```text
ecce25c2547c4a1452867bd49b21cd3766a1513f3c678b99f3aff3ac10ee03eb
```

They also matched the valid M=1 target exactly on three additional greedy
chat prompts covering factual QA, arithmetic, and sequence generation.

Steady process memory was 8770 MiB for target-only, 10620 MiB for MTP, and
13592 MiB for DFlash.  This replaces the earlier 30.7 GiB MTP and 23.2 GiB
DFlash contiguous-cache deployments.

The target-only control must use the normal M=1 artifact.  A speculative
artifact intentionally exports its target decode graph at verify width M=5
(MTP) or M=10 (DFlash); running such a verify graph as an autoregressive M=1
server is not a valid baseline.

Final evidence:

```text
tmp/spec_decode_validation/live_ab_20260727/target_m1_bench.jsonl
tmp/spec_decode_validation/live_ab_20260727/target_m1_1024.jsonl
tmp/spec_decode_validation/live_ab_20260727/mtp_export_capacity_bench.jsonl
tmp/spec_decode_validation/live_ab_20260727/mtp_export_capacity_1024.jsonl
tmp/spec_decode_validation/live_ab_20260727/dflash_export_capacity_bench.jsonl
tmp/spec_decode_validation/live_ab_20260727/dflash_export_capacity_1024.jsonl
tmp/spec_decode_validation/live_ab_20260727/paged_cross_prompt_parity.json
tmp/spec_decode_validation/live_ab_20260727/mtp_paged_server.log
tmp/spec_decode_validation/live_ab_20260727/dflash_paged_warm_server.log
```
