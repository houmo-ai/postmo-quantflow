# MiniCPM-V-4.7 Merak Examples

MiniCPM-V-4.7 reuses MiniCPM Vision token-capacity graphs (`4x` / `16x`) and the
Qwen3.5 Dense / Qwen3.5 MoE text Prefill/Decode stacks.

## Dynamic vision inputs

Both modes share five input patch capacities `[384, 784, 1536, 2816, 6144]`.
For each processor-resized image or slice, runtime selects the smallest gear
that can hold its valid output tokens. Each HMONNX graph has fixed
shapes: the host orders patches into hierarchical 2x2 groups, pads to capacity,
builds positions from the actual grid, and masks the padded tail. After ViT
and projection, the host keeps only the valid output prefix.

| Mode | Compression | Required patch-grid alignment | Valid output tokens |
| --- | --- | --- | --- |
| `4x` | Final projector merges each 2x2 group | Both dimensions divisible by 2 | `H_patch * W_patch / 4` |
| `16x` | In-ViT 2x2 merger followed by final 2x2 projector | Both dimensions divisible by 4 | `H_patch * W_patch / 16` |

ViT and the projector are exported together, with five graphs per mode:

| Input patch capacity (both modes) | 384 | 784 | 1536 | 2816 | 6144 |
| --- | --- | --- | --- | --- | --- |
| `4x` output token capacity | 96 | 196 | 384 | 704 | 1536 |
| `16x` output token capacity | 24 | 49 | 96 | 176 | 384 |

`export.vision.patch_token_gears` and CLI `--patch-token-gears` use **input
patch tokens**. Export derives each mode's output capacities automatically.
Per-graph `token_capacity` uses input patches; `image_token_capacity` uses
post-merge outputs. The legacy `image_token_gears` override applies the same
output capacities to both modes and is mutually exclusive with patch gears.
Runtime chooses by capacity, not nearest width/height; it never picks a smaller
graph that would lose valid patches. Above the largest gear it raises an error.
The native processor still determines image resizing and slicing. The legacy
single-graph CLI `--token-capacity` uses input patches and overrides gear export.

The padding/mask formula follows the Qwen3.5 patch-token path. For each
processor-resized image or slice, `N = H_patch * W_patch`, where
`H_patch = H_resized / 14` and `W_patch = W_resized / 14`. With exported capacity
`C`, append `C - N` zero patches and generate a key-padding bias
`mask[j] = 0 if j < N else -65504`. The mask has shape `[1,1,1,C]`.
For `16x`, the in-ViT merger also needs a second mask of shape `[1,1,1,C/4]`,
with its first `N/4` entries valid. Both masks enter `xhquant.nn.MaskedAdd`
followed by `Softmax`; HMONNX conversion uses TorchFX to retain that quantized
mask contract. FP32 reference masks are clamped before FP16 conversion to keep
the runtime inputs finite. Native MiniCPM resizing, crop geometry and position
encoding determine the valid tokens. Re-export existing vision graphs to use
this mask-preserving conversion path.
Custom attention sequences above 6144 patches use fixed query chunks of 1024 to retain
all keys and the same Softmax/mask semantics while limiting score tensor memory.
The independent local 2x2 windows use batches of at most 2048 windows, including
the largest `16x` gear, to stay within the aligned CUDA MatMul batch-grid limit.
`keep_onnx: false` exports HMONNX directly and validates it against native
features; `keep_onnx: true` additionally exports and validates float ONNX.

## Variants

| Variant | HF weights | Merak `model_type` | Workflow YAML |
| --- | --- | --- | --- |
| Dense 1B W8A8 | `/path/to/minicpm_4_7/MiniCPM-V-4.7-1B` | `MiniCPMV47ForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml` |
| Dense 1B W4A8 GPTQ | `/path/to/minicpm_4_7/MiniCPM-V-4.7-1B` | `MiniCPMV47ForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w4a8_gptq.yaml` |
| MoE 35B-A3B W8A8 | `/path/to/minicpm_4_7/MiniCPM-V-4.7-35B-A3B` | `MiniCPMV47MoeForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w8a8.yaml` |
| MoE 35B-A3B W4A8 AutoRound | `/path/to/minicpm_4_7/MiniCPM-V-4.7-35B-A3B` | `MiniCPMV47MoeForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w4a8_autoround.yaml` |
| MoE 35B-A3B W4A8 AutoRound + DSpark | Same target, paired `MiniCPM-V-4.7-35B-A3B-DSpark` | `MiniCPMV47MoeForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w4a8_autoround_dflash.yaml` |

## Optional lookahead expert scheduling (LES)

For the 35B MoE ordinary workflow, set `export.model.les_prediction_k: 8`
or pass `--les-prediction-k 8` to the workflow example. The default is `0`
(disabled). Existing W4A8/W5-head configs and dynamic vision gears can be reused;
LES does not change their precision settings. The unified `build_model` API
currently has no LES build parameter; use the family workflow for this option.

The predictor follows the [reference formula](https://github.com/tc-mb/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L455-L465):
`predicted_experts[l+1] = TopK(W_gate[l+1] @ post_attention_norm(h[l]))`.
The normalized input is the current layer's **MoE input**, before its experts
and before the next layer's attention. No next-layer hidden state is used to
make the prediction. `prediction_k` is the number of predicted experts, not
the number of layers ahead. For this checkpoint the real router selects 8 of
256 experts; predicting 8 enables exact Top-8 set comparisons.

Only the single-token decode frontend gets the extra projections. The SDK
quantizes these branches with their own activation observations, sharing the
source floating router weights. The decode graph appends
`les_router_probs_0..39` (actual probabilities entering MoeBlock) and
`les_prediction_logits_1..39`, interleaved by layer; metadata records names,
roles, layer IDs and Top-K. Prefill and all existing cache ports are preserved.
There is no external gate input: predictions are scheduling hints, and the
original gate still determines every expert and its weight. Feeding a
predicted gate back as the real router would change the model.

`MiniCPMV47HMONNXRuntime(..., les_prefetch=..., les_require=...)` delivers
signals during eager SDK V2 execution. `les_prefetch(layer, ids)` can start
loading the next layer's weights; `les_require(layer, actual_ids)` must wait
for hits and load **all misses** before that layer computes. Both receive
device tensors. The family adapter owns signal timing; a device backend owns
expert storage, DMA, residency and cache eviction. With fully resident SDK
weights no I/O is necessary. This integration does not claim accelerator
prefetch speedups without such a backend. Python callbacks do not support
CUDA graph capture or SDK auto-offload execution. Dense 1B, ACEP pruning and
multi-token DSpark verification are rejected when LES is enabled; their
existing LES-disabled paths remain available.

The evaluation example uses the native 35B BF16 checkpoint and compares
predictions against the next layer's **actual** selected experts. It tests
50 original questions, 10 each in mathematics, programming, science, Chinese
writing and image understanding. Images are generated diagrams with visible
shapes, text and charts. Each question is generated twice (greedy, thinking
disabled, at most 64 new tokens), comparing all output token IDs with LES
observation on/off. Statistics exclude prefill, first-layer warmup, final-layer
predictions and multi-token verify. Reports contain per-question/per-layer
counts, raw expert IDs, expert recall, prediction precision, complete actual
Top-K coverage and exact Top-K set equality. These are routing measurements,
not an answer-quality benchmark or W4A8 HMONNX accuracy results.

```bash
python examples_merak/llm/minicpm_v_4_7/evaluate_les.py \
  --model-dir /path/to/MiniCPM-V-4.7-35B-A3B \
  --output-dir work_dirs/minicpm35b_les_eval --device cuda:0
```

## Calibrate a W5 head for an existing 35B AutoRound backbone

The W5 export configs consume a completed head checkpoint through
`--quanted-model-dir`. They set `quant: null` because the backbone and head
have already been calibrated. Both retain dynamic `4x` / `16x` vision gears.
The ordinary and DSpark exports use the same calibrated target head.
The W5 presets enable causal `xhquant.nn.FlashAttention` in the target with
Q/K/V/score/probability precision explicitly set to 8 bits; the target MatMul
operator override is also 8/8. The DSpark draft disables fused FlashAttention
and uses Q×K → explicit mask → Softmax → P×V. Its mask exposes the valid
target prefix and complete 15-query block non-causally, excluding unused cache
slots. This avoids requiring valid-length support in a non-causal fused
operator. An explicit draft attention setting overrides target inheritance.
Dynamic vision profiles retain their own precision.

Use GPTQModel with `gptqmodel.recipes.minicpm_v47_qdq_head` and
`gptqmodel.recipes.qwen35_head` available. The QDQ adapter converts the existing
`float_causal_bin` representation to safetensors without rounding or casting
the backbone. Its metadata excludes every backbone module from quantization.
The existing head-only recipe replays that backbone with its normal MoE routing
and final norm, calibrates the floating head, and saves a new checkpoint.
`algorithm=inherit` uses AutoRound for the existing AutoRound backbone.

```bash
python -m gptqmodel.recipes.minicpm_v47_qdq_head \
  --model-dir /path/to/autoround_llm_dequant \
  --tokenizer-dir /path/to/MiniCPM-V-4.7-35B-A3B \
  --output-dir work_dirs/minicpm35b_head_w5 \
  --calibration-jsonl data/calib_data/NeelNanda-pile-10k.jsonl \
  --head '{"bits":5,"algorithm":"inherit","iters":400}' \
  --nsamples 256 --seqlen 2048 --device cuda:0

python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /path/to/MiniCPM-V-4.7-35B-A3B \
  --quanted-model-dir work_dirs/minicpm35b_head_w5 \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w4a8_autoround_head_w5.yaml \
  --export-output-dir work_dirs/minicpm35b_head_w5_export
```

For DSpark, select `minicpm_v_4_7_moe_xh2a_w4a8_autoround_head_w5_dflash.yaml`
and add `--dflash-model-dir /path/to/MiniCPM-V-4.7-35B-A3B-DSpark`.
The export validates the calibrated head's actual packed layout and precision;
a floating head or mismatched precision is rejected before export.
The target head is quantized once for deployment and its fixed integer codes
and scale/exponent tensors are bound into the DSpark draft graph.
W5 uses a 5-bit numerical range in 8-bit storage slots, packed into int32 words;
the checkpoint records both logical and storage precision. The producer is
GPTQModel, the inherited solver is AutoRound, and the saved layout is GPTQ V2.
These are separate properties.

## Latest checkpoints and DFlash / DSpark

Use Python >=3.11 and Transformers >=5.18.0 with native `minicpmv4_7`
support. If using an unreleased Transformers checkout, put its `src/` directory
on `PYTHONPATH` in a dedicated environment. The shared CI profiles for other
families pin Transformers 5.16.1 and must run in their own environment.
MiniCPM loaders register `AutoConfig` and `AutoModelForImageTextToText` before
loading, including workflow export, text/vision export and processor loading.
They prefer the native Transformers classes and also support importable local
`configuration_minicpmv4_7.py` / `modeling_minicpmv4_7.py` modules when the native
model module is absent. No manual registration before `AutoWorkflow` is needed.
The model implementation and the required Transformers version must still be
installed; registration only supplies the Auto mappings.

The new checkpoints use canvas MRoPE. The runtime computes positions for the
complete prompt with the native MiniCPM position utilities, then slices them
for Prefill and uses the same RoPE delta during Decode/Verify. Both `4x` and
`16x` vision profiles use the selected mode's crop geometry.

DSpark is exposed through the `dflash` workflow/recipe variant. Its five target
hidden-state layers come from the assistant config. DSpark predicts at query
position zero and applies its vanilla Markov transition to each proposal.
The draft keeps its complete 15-query backbone, then applies the logits and
Markov heads only to the first `num_draft_tokens` positions. The default is
3 proposals and a 4-token verify graph; unused confidence outputs are omitted.
The existing `dflash` model, recipe, source and runtime interfaces are retained.

Optional `export.model.dflash_config.enable_confidence_truncation: true` enables
the confidence head; `confidence_threshold` defaults to `0.3`. The graph exports
raw confidence logits. The host applies sigmoid and a cumulative prefix product,
then verifies only the retained proposals plus the anchor. Export includes exact
target verify lengths `1..num_draft_tokens+1`; confidence never replaces target
token matching. The CLI shorthand is `--confidence-threshold 0.3`. This option
requires regular export and currently excludes LoRA. Older packages with
`confidence` probabilities remain loadable with their original fixed verify graph.
The 1B and 35B-A3B assistants must be paired with their own target/tokenizer.

The non-FlashAttention target Decode/Verify graphs zero invisible V-cache
entries separately for each query before the quantized P×V product. Causal
probabilities alone cannot isolate the HMFP exponent shared across 64 V tokens;
future proposals or rejected cache entries would otherwise affect the visible
prefix. Only P×V is split by query; projections, experts and Q×K remain batched.

```bash
MODEL_ROOT=/path/to/minicpm_4_7
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir "$MODEL_ROOT/MiniCPM-V-4.7-1B" \
  --dflash-model-dir "$MODEL_ROOT/MiniCPM-V-4.7-1B-DSpark" \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8_dflash.yaml \
  --num-draft-tokens 3 \
  --export-output-dir work_dirs/minicpm_v_4_7_1b_dflash
```

For MoE, use `MiniCPM-V-4.7-35B-A3B`, its `-DSpark` directory, and
`configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w8a8_dflash.yaml`.
The existing `hmonnx_demo.py` loads either variant from its root metadata.
DSpark export defaults to the SDK's non-strict tracing path to avoid excessive
Torch 2.8 guard construction. Standard Torch ONNX optimization stays enabled:
it normalizes tuple access and slice boundaries required by the runtime.
Explicit SDK environment settings are honored and process environment values
are restored afterward.
The complete 15-proposal target verify graph is not full-model validated: this
SDK/ONNXScript combination spends over ten minutes optimizing that graph.
The default 3-proposal profile avoids that large verification graph while
keeping the assistant's trained 15-query backbone intact.
Leave one draft block of context space beyond prompt plus requested output.

The unified API accepts the same pairing:

```python
from xhmodel_merak.model_build import build_model

root = "/path/to/minicpm_4_7"
result = build_model(
    model_id="minicpm-v-4.7-1b",
    model_dir=f"{root}/MiniCPM-V-4.7-1B",
    sources={"dflash": f"{root}/MiniCPM-V-4.7-1B-DSpark"},
    speculative={"mode": "dflash", "num_draft_tokens": 3},
    output_dir="work_dirs/minicpm_v_4_7_1b_dflash_api",
    device="cuda:0",
)
```

Exports include `hf_config/` tokenizer/processor resources and draft graphs,
so inference does not need the source weight directories. The root metadata
records draft paths relative to the export root.

Regression tests live in `benchmark_test/ci_test_transformer5.16.0/`:
`test_minicpm_dynamic_vision.py` and `test_minicpm_vision_gears.py` cover native
features, padding masks, gear routing and executable HMONNX;
`test_minicpm_v_4_7_dspark.py` covers proposal alignment, confidence scheduling
and target verification; `test_minicpm_v47_head_w5.py` checks checkpoint
consumption and shared integer head payloads. Run these with the MiniCPM
Transformers environment and the current CI `pytest.ini` / `--confcutdir`.

## Float baseline

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/float_demo.py \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-1B \
  --image examples_merak/llm/minicpm_v_4_7/pipeline-cat-chonk.jpeg \
  --downsample-mode 4x \
  --max-new-tokens 64
```

## Export workflow

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-1B \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm_v_4_7_1b_export \
  --overwrite
```

Dense W4A8 GPTQ (extract Qwen3.5 text backbone → GPTQ → dequant `quant_weight`; Vision stays W8A8):

```bash
CUDA_VISIBLE_DEVICES=2 GPTQMODEL_SOURCE=/path/to/gptqmodel \
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-1B \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w4a8_gptq.yaml \
  --quant-output-dir work_dirs/minicpm_v_4_7_1b_w4a8_gptq \
  --export-output-dir work_dirs/minicpm_v_4_7_1b_w4a8_export \
  --device cuda:0 \
  --overwrite
```

MoE W8A8:

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm_v_4_7_35b_a3b_export \
  --overwrite
```

MoE W4A8 AutoRound (in-memory `Qwen3_5MoeForCausalLM` → AutoRound mixed bits → restore packed experts → float `quant_weight`; Vision stays W8A8):

```bash
CUDA_VISIBLE_DEVICES=2 GPTQMODEL_SOURCE=/path/to/gptqmodel \
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-35B-A3B \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w4a8_autoround.yaml \
  --quant-output-dir work_dirs/minicpm_v_4_7_35b_a3b_w4a8_autoround \
  --export-output-dir work_dirs/minicpm_v_4_7_35b_a3b_w4a8_export \
  --device cuda:0 \
  --overwrite
```

## HMONNX demo

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/hmonnx_demo.py \
  --export-meta work_dirs/minicpm_v_4_7_1b_export/export_meta_info.json \
  --image examples_merak/llm/minicpm_v_4_7/pipeline-cat-chonk.jpeg \
  --downsample-mode 4x
```

## Unified recipes

Registered unified capability is **W8A8 / full and dflash**. The W4A8 GPTQ/AutoRound
YAMLs above remain workflow examples and are **not** registered as unified
recipes.

| Recipe ID | Variant | Artifact format | Unified scope |
| --- | --- | --- | --- |
| `minicpm-v-4.7-1b` | Dense 1B | `minicpm_v_4_7` | W8A8 / full, dflash |
| `minicpm-v-4.7-35b-a3b` | MoE 35B-A3B | `minicpm_v_4_7` | W8A8 / full, dflash |

### Unified `build_model` → `inference_model`

```bash
# Dense 1B
python -m examples_merak.unified_api.build \
  --model-id minicpm-v-4.7-1b \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-1B \
  --output-dir ./work_dirs/minicpm_v_4_7_1b_api \
  --device cuda:0

python -m examples_merak.unified_api.generate \
  --artifact-dir ./work_dirs/minicpm_v_4_7_1b_api \
  --prompt "Describe the image." \
  --image examples_merak/llm/minicpm_v_4_7/pipeline-cat-chonk.jpeg \
  --max-new-tokens 64 \
  --device cuda:0

# MoE 35B-A3B
python -m examples_merak.unified_api.build \
  --model-id minicpm-v-4.7-35b-a3b \
  --model-dir /path/to/minicpm_4_7/MiniCPM-V-4.7-35B-A3B \
  --output-dir ./work_dirs/minicpm_v_4_7_35b_a3b_api \
  --device cuda:0

python -m examples_merak.unified_api.generate \
  --artifact-dir ./work_dirs/minicpm_v_4_7_35b_a3b_api \
  --prompt "Describe the image." \
  --image examples_merak/llm/minicpm_v_4_7/pipeline-cat-chonk.jpeg \
  --max-new-tokens 64 \
  --device cuda:0
```

- Dense W4A8 GPTQ mirrors MiniCPM-V-4.5: GPTQModel cannot load the MiniCPM remote-code host, so the GPTQModel recipe extracts `Qwen3_5ForCausalLM`, quantizes with `minicpm_v_4_5_qwen_mixed_295.jsonl`, dequantizes to float `pytorch_model.bin`, then ModelZoo applies it via `quant_weight` while Vision still loads from the raw host.
- MoE W4A8 AutoRound uses the GPTQModel recipe to assemble an in-memory `Qwen3_5MoeForCausalLM` from the MiniCPM host (`language_model` + `lm_head`), run AutoRound with `data_type=int_sym_xh`, attn/shared 8-bit and expert 4-bit (`group_size=64`), restore packed experts after `.quantize()`, and emit float CausalLM `pytorch_model.bin` for `quant_weight`. Default `save_quantized=false` avoids writing a ~70GB packed checkpoint.
  The export keeps routed `MoeBlock` experts at W4A8, while `Linear` projections
  (attention, shared experts, router and head) and GDN `Conv2d` use W8A8. These
  exceptions are explicit because float QDQ weights do not retain the producer's
  per-module bit metadata. Both ordinary and DSpark configurations use this
  scheme; Vision remains W8A8.

Both recipes keep graphs at the workflow root (`visual_4x_m96`, etc., plus `llm`)
and validate through `export_meta_info.json`. Unified-API golden dump is not
enabled on this adapter; use the workflow `dump_golden` path instead.

## Notes

- Float and HMONNX paths both require `downsample_mode` to stay consistent.
- Processor loading uses a Transformers compatibility patch for MiniCPM remote video code.
- Export writes root `export_meta_info.json`, two gear JSON files, ten `visual_<mode>_m<capacity>/` graph directories, and `llm/` when enabled.

