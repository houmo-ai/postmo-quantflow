# HMONNX speculative decoding design

Status: design baseline for Qwen3.5 MTP, Qwen3.5 DFlash, and Gemma4 MTP.

This document defines the cross-repository contract between:

- `xh2modelzoo`: target/draft graph export and artifact metadata.
- `xhquanttool`: HMONNX cache and PageAttention runtime semantics.
- `vllm-merak`: scheduling, target verification, rejection sampling, and
  transactional cache commit.

The first implementation is intentionally limited to one live request
(`max_num_seqs=1`). Qwen3.5 hybrid linear state and Gemma4's fixed sliding
ring already require that limit. The design must nevertheless keep request
ownership explicit so batching can be added later without changing state
semantics.

## 1. Correctness boundary

vLLM remains authoritative for:

1. which target tokens are scheduled;
2. the draft token list attached to a request;
3. target-logit rejection sampling;
4. the accepted output token count; and
5. the accepted logical sequence length.

HMONNX remains authoritative for:

1. target and draft graph execution;
2. physical target PageAttention storage;
3. Qwen linear-attention state tensors; and
4. MTP/DFlash private draft state.

The Merak plugin is the transaction coordinator. It must not infer acceptance
from argmax comparisons when vLLM is configured for stochastic sampling. It
uses vLLM's `RejectionSampler` result, where non-`-1` output entries are the
accepted drafts plus the recovered/bonus token.

## 1.1 Unified cache transaction model

Standard inference and speculative inference must not have two independent
cache implementations. Every request owns one cache descriptor:

```text
RequestCache
  owner/request_id
  epoch
  committed_length
  target PageKV views
  target LinearState slot
  private draft-cache slot
  inflight transaction | none
```

Every model execution is one transaction with `base_length`, `staged_width`
and `accepted_width`:

| Execution | `staged_width` | `accepted_width` |
| --- | ---: | ---: |
| target prompt/chunk prefill | `Q` | `Q` |
| draft prompt/context prefill | `Q` | `Q` |
| ordinary target decode | `1` | `1` |
| speculative target verify | `M = K + 1` | `A`, where `1 <= A <= M` |

Thus prefill and decode are simply transactions which commit their complete
staged range. Verify uses the same begin/stage/commit protocol but selects an
accepted prefix. `committed_length` is the only logical visibility cursor:

```text
new_committed_length = base_length + accepted_width
```

Physical storage and logical visibility are deliberately separate. A rejected
suffix is never cleared or copied backwards. It stays physically present,
remains outside `committed_length`, and is overwritten by a later transaction.
The transaction carries an `epoch`; commit/abort must reject a stale epoch so
an output lease from an earlier CUDA Graph replay cannot update a reused
request slot.

Cache participants implement different physical commit operations under this
single logical transaction:

| Participant | Stage | Commit accepted prefix |
| --- | --- | --- |
| full PageKV | append by target `slot_mapping` | advance logical length only |
| sliding PageKV | append into reserved ring/page slots | advance length/window only |
| Qwen LinearState | retain per-step state bank `S[0:M]` | select `S[A-1]` |
| DFlash private KV | mutate fixed K/V owner in place | advance private valid length |
| Qwen MTP private state | write shifted pairs into fixed owner | advance by accepted shifted pairs |
| Gemma4 MTP | read target PageKV | no physical commit |

The target transaction commits before the dependent draft-cache update. The
draft update consumes exactly the accepted target hidden-state prefix. This
ordering prevents MTP/DFlash private state from becoming newer than the target
cache when a request is rejected, cancelled, or fails.

For CUDA Graph, all physical addresses and capacity tensors are stable.
Lengths and the selected state-bank index are device tensors; graph-captured
code must not call `.item()`. The current serial implementation uses the
HMONNX runtime state as a one-entry execution cache in front of the persistent
Linear checkpoint slot: consecutive calls for the same request skip redundant
checkpoint reloads. The final ABI should let the verify graph write snapshots
directly into a cache-manager-owned state bank and use one fused device-side
selection/commit, rather than materializing outputs and performing
load/commit/scatter copies.

Memory is provisioned once at load:

- full PageKV reserves scheduler pages for the maximum staged suffix;
- sliding PageKV has at least `window + max_staged_width` physical capacity, so
  rejected writes cannot overwrite the oldest still-committed window entry;
- LinearState reuses graph output snapshots as its transaction bank and must
  not clone them;
- DFlash/MTP private caches are fixed request-owned allocations shared by all
  of their graph variants.

## 2. Why the normal Merak forward contract is insufficient

The current Merak target path returns only one row of logits per request:

```text
HMONNX output -> select last token -> copy to query_end - 1
```

Target verification requires one logit row for every input position:

```text
[current token, draft_0, ..., draft_K-1]
   -> [prediction_0, ..., prediction_K]
```

The current Qwen HMONNX wrapper also commits the final conv/recurrent state
immediately. During verification, the final state belongs to the fully
speculative suffix and is wrong whenever a draft is rejected. Therefore:

- phase selection must be explicit; `num_tokens > 1` is not a valid proxy for
  prefill;
- target execution must return all verification logits;
- the target hidden output must survive until draft/context update;
- Qwen per-step linear-state outputs must remain uncommitted until rejection
  sampling finishes.

## 3. Runtime phases and result type

The plugin uses an explicit phase enum:

```text
PREFILL  - target prompt/chunk execution
DECODE   - ordinary one-token target execution
VERIFY   - current token plus one or more scheduled draft tokens
```

One request execution returns a structured value:

```text
MerakRequestResult
  logits: Tensor[Q, vocab]
  target_hidden: Tensor[Q, hidden] | None
  qwen_linear_steps:
    conv: mapping[layer/branch, tuple[Tensor, ...]]
    recurrent: mapping[layer, tuple[Tensor, ...]]
  phase: PREFILL | DECODE | VERIFY
  initial_sequence_length: int
  query_length: int
```

Ordinary vLLM sampling still receives a flat tensor whose row order matches
`logits_indices`. The structured side data is retained by request ID until
sampling decides how many outputs are valid.

## 4. Target PageAttention transaction

Target verification may write KV for all `K + 1` input tokens. Rejected slots
do not need a physical rollback:

1. vLLM advances the request by the accepted output count only;
2. subsequent block tables and slot mappings describe that accepted logical
   length;
3. stale suffix slots are not in the readable range; and
4. a later target execution overwrites them before they become visible.

This is valid only if every attention kernel receives an exact logical read
range. No code may use physical cache fill level as sequence length.

For Gemma4 sliding attention, the physical cache is a fixed ring. The plugin
must continue to provide:

```text
kv_window_start_abs = max(0, query_absolute_position - sliding_window + 1)
```

and absolute logical block IDs projected onto the ring. Rejected suffix writes
may remain stale, but neither the accepted sequence length nor
`kv_window_start_abs` may advance past the accepted point.

The exported `accepted_count` input is not a rollback mechanism. The current
xhquant eager/custom-op/parser implementations ignore it. Merak therefore
must never rely on `accepted_count` to repair target KV.

## 5. Qwen3.5 hybrid linear-state transaction

Qwen target verification exports a conv/recurrent snapshot per verify step.
For `N` valid output tokens from vLLM, the committed target-input step is:

```text
commit_step = N - 1
```

Examples:

| Verification outcome | vLLM valid outputs | State to commit |
| --- | ---: | ---: |
| first draft rejected | 1 | step 0 (`current`) |
| two drafts accepted, third rejected | 3 | step 2 |
| all K drafts accepted plus bonus | K + 1 | step K |

The transaction has three stages:

1. `begin`: bind the current accepted conv/recurrent state as graph inputs;
2. `execute`: retain per-step outputs without mutating the persistent state;
3. `commit`: alias/copy only `commit_step` into the persistent state after
   vLLM rejection sampling.

The implementation must fail closed if:

- a verify graph lacks a per-step output;
- output count exceeds the exported verify length;
- another request owns the serial runtime state; or
- preemption occurs before a persistent checkpoint protocol exists.

No full pre-verify snapshot is required. The accepted input state remains
untouched until commit.

## 6. Qwen3.5 MTP draft state

MTP has a private draft conv/recurrent state. It is not the target linear
state.

### Prompt initialization

For target hidden states `h[0:P]` and prompt tokens `t[0:P]`, MTP prefill
consumes shifted pairs:

```text
(h[0], t[1]), (h[1], t[2]), ..., (h[P-2], t[P-1])
```

The final target hidden `h[P-1]` is retained for the first generation token.
Chunk boundaries carry one pending hidden state so no shifted pair is lost.

### Draft round

MTP generates `K` tokens sequentially. Each step consumes:

- the previous token embedding;
- the previous target/draft hidden state;
- the MTP logical past length; and
- the same persistent private MTP cache allocation.

The cache may contain a rejected draft suffix. Its logical length advances by
the accepted target-input step count only, and later writes overwrite the
stale suffix. No draft-cache-to-target-cache copy is permitted.

## 7. Qwen3.5 DFlash state

DFlash owns two persistent cache tensors, K and V, shared by its context and
decode graphs.

### Pointer invariant

For the lifetime of a request:

```text
context_input_K.data_ptr == decode_input_K.data_ptr
context_input_V.data_ptr == decode_input_V.data_ptr
```

Context graph outputs are not copied back to decode inputs. `CacheTensor`
mutation is in place. A changed pointer is a correctness error, not a
performance warning.

### Prompt and round updates

1. Target prefill hidden states are projected into the DFlash context cache.
2. DFlash decode reads that cache and predicts its draft block in parallel.
3. Target verifies `[current] + drafts`.
4. Only the accepted target hidden prefix is appended to DFlash context.
5. DFlash logical context length advances by the appended accepted prefix.

The context-decode graph may have a different fixed input shape from prompt
context. Both sessions still bind the same persistent cache objects.

## 8. Gemma4 MTP shared target KV

Gemma4's assistant is Q-only: it projects Q but deliberately ignores local
assistant K/V weights. Each assistant layer reads the target cache belonging
to the last non-shared target layer of the same attention type.

The existing standalone draft graph accepts four dense FP16 tensors:

```text
sliding K/V, full K/V
```

Merak target caches are quantized and paged. Dequantizing or gathering these
four tensors for each draft step defeats KV sharing and is not an acceptable
deployment path.

### Required HMONNX ABI

The deployment graph uses read-only PageAttention:

```text
PageAttentionReadOnly(
  q,
  block_ids,
  seq_lens,
  query_kv_range_abs,
  kv_window_start_abs,
  cache_handle
) -> attention_output
```

It has no K/V write inputs and no slot mapping. The target
`HMFPPagedKVCache` handle is supplied out of band by
`PageAttentionContext`.

Read ranges:

- full attention: `[0, accepted_target_len)`;
- sliding attention:
  `[max(window_start, draft_position - window + 1), accepted_target_len)`.

The xhquant SEFP paged kernel already has an explicit query-range execution
path. The parser currently rejects this combination and must be enabled with
focused correctness tests before the Gemma graph is switched.

The existing dense-cache graph remains a standalone reference oracle. The
Merak artifact must identify which draft ABI it contains; the plugin must
reject a dense-cache Gemma draft graph rather than silently copying cache.

### Position rule

All assistant draft layers use the same draft position for a round, matching
the official Gemma4 proposer. Full and sliding groups have separate block
tables/ranges because their physical cache geometry differs.

## 9. Artifact metadata contract

`golden_meta_info.json` gains a versioned deployment contract:

```json
{
  "spec_decode": {
    "runtime_contract_version": 2,
    "mode": "mtp",
    "num_draft_tokens": 4,
    "verify_length": 5,
    "target": {
      "hidden_output_name": "post_norm_hidden",
      "linear_state_outputs": "per_step"
    },
    "draft": {
      "abi": "qwen_mtp_v1",
      "prefill_hmonnx": "...",
      "decode_hmonnx": "..."
    }
  }
}
```

DFlash uses:

```json
{
  "draft": {
    "abi": "qwen_dflash_v1",
    "context_hmonnx": "...",
    "context_decode_hmonnx": "...",
    "decode_hmonnx": "...",
    "cache_mutation": "in_place"
  }
}
```

Gemma4 uses:

```json
{
  "draft": {
    "abi": "gemma4_mtp_readonly_page_attention_v1",
    "decode_hmonnx": "...",
    "cache_binding": "target_attention_type_owner"
  }
}
```

Every referenced path is relative to the artifact directory and must exist.
The parser validates graph input/output names and fixed sequence lengths
before allocating target KV. Legacy metadata is accepted only by an explicit
normalizer with all required fields derivable; otherwise loading fails.

Target and draft graphs share target weights only through their declared
cache/hidden contracts. Loading a speculative artifact must not instantiate a
second target HMONNX runtime.

## 10. vLLM integration

The public vLLM configuration continues to request `mtp` or `dflash`. During
Merak platform configuration, an HMONNX speculative artifact is translated to
the plugin custom proposer backend while preserving the original Merak mode.
This avoids vLLM trying to load the Merak target directory as a Transformers
draft model.

The plugin uses:

- vLLM input preparation and scheduled draft tokens;
- vLLM target `logits_indices`;
- vLLM `RejectionSampler`; and
- vLLM output bookkeeping.

It overrides only the Merak-specific boundaries:

1. structured target execution and explicit phase selection;
2. deferred target linear-state commit;
3. HMONNX draft proposal using target hidden state;
4. accepted-count postprocessing for Qwen/DFlash state; and
5. draft session lifecycle.

The initial implementation may propose after synchronous output bookkeeping
because vLLM's generic `custom_class` path supplies CPU token lists. This is a
correctness milestone, not the final performance design. The optimized path
must consume device sampled IDs and target hidden states before CPU
bookkeeping, like the official DFlash/Gemma proposers.

## 11. Memory invariants

The following are release blockers:

1. target weights are loaded exactly once;
2. draft weights are loaded once;
3. DFlash context/decode caches have stable pointers;
4. Gemma draft reads target paged cache directly;
5. target verify does not clone the entire KV cache;
6. Qwen verify retains only per-step linear outputs needed until sampling;
7. graph load/pack does not hold both unpacked and packed target weights at
   steady state; and
8. finishing a request releases private MTP/DFlash state ownership.

## 12. Failure and lifecycle rules

Phase one rejects:

- `max_num_seqs != 1`;
- request preemption;
- prefix-cache sharing for Gemma's fixed sliding ring;
- a verify query longer than the exported `verify_length`;
- missing hidden/per-step state outputs;
- draft/target vocab mismatch;
- a Gemma dense shared-KV draft ABI;
- DFlash cache pointer replacement; and
- CUDA graph replay whose captured output lifetime cannot cover rejection
  sampling.

EOS and max-model-length handling are still owned by vLLM. If there is not
enough target or draft context capacity for a complete round, the runner emits
zero draft tokens for that round rather than partially executing an
incompatible graph.

## 13. Verification matrix

Each model/mode must pass all rows before performance claims:

| Test | Qwen MTP | Qwen DFlash | Gemma MTP |
| --- | --- | --- | --- |
| no-draft target parity | required | required | required |
| all drafts accepted | required | required | required |
| rejection at every draft position | required | required | required |
| bonus token path | required | required | required |
| EOS inside accepted prefix | required | required | required |
| prompt split across export chunks | required | required | required |
| long-context cache wrap | linear + paged | context + paged | sliding ring |
| pointer/no-copy assertions | MTP private | context/decode | target paged |
| greedy text parity | required | required | required |
| stochastic rejection parity | required | required | required |
| vLLM service deployment | required | required | required |

For Qwen hybrid state, every rejection-position test also compares committed
conv/recurrent state against a non-speculative target run at the same accepted
logical length. For Gemma, tests straddle the sliding-window boundary and the
physical ring wrap.

## 14. Delivery order

1. Add artifact parser and pure transaction/unit tests.
2. Enable explicit PageAttention query ranges in xhquant.
3. Add `PageAttentionReadOnly` and its no-write tests.
4. Export Qwen3.5-9B MTP and validate target per-step state outputs.
5. Integrate Qwen MTP with vLLM rejection sampling.
6. Export/integrate Qwen3.5-9B DFlash with pointer assertions.
7. Export Gemma4-12B MTP using read-only paged attention.
8. Run vLLM Merak end-to-end correctness, acceptance, latency, and memory
   measurements.

This order makes target verification and state commit correct before adding
the harder shared-cache optimization. It also prevents the dense Gemma
standalone implementation from becoming an accidental deployment ABI.
