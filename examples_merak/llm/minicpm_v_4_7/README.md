# MiniCPM-V-4.7 Merak Examples

MiniCPM-V-4.7 reuses MiniCPM Vision token-capacity graphs (`4x` / `16x`) and the
Qwen3.5 Dense / Qwen3.5 MoE text Prefill/Decode stacks.

## Variants

| Variant | HF weights | Merak `model_type` | Workflow YAML |
| --- | --- | --- | --- |
| Dense 1B W8A8 | `/data01/datasets/Minicpm-v4.7/weights/1b` | `MiniCPMV47ForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml` |
| Dense 1B W4A8 GPTQ | `/data01/datasets/Minicpm-v4.7/weights/1b` | `MiniCPMV47ForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w4a8_gptq.yaml` |
| MoE 35B-A3B W8A8 | `/data01/datasets/Minicpm-v4.7/weights/35b-a3b` | `MiniCPMV47MoeForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w8a8.yaml` |
| MoE 35B-A3B W4A8 AutoRound | `/data01/datasets/Minicpm-v4.7/weights/35b-a3b` | `MiniCPMV47MoeForConditionalGeneration` | `configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w4a8_autoround.yaml` |

## Float baseline

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/float_demo.py \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/1b \
  --image examples_merak/llm/minicpm_v_4_7/pipeline-cat-chonk.jpeg \
  --downsample-mode 4x \
  --max-new-tokens 64
```

## Export workflow

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/1b \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm_v_4_7_1b_export \
  --overwrite
```

Dense W4A8 GPTQ (extract Qwen3.5 text backbone → GPTQ → dequant `quant_weight`; Vision stays W8A8):

```bash
CUDA_VISIBLE_DEVICES=2 GPTQMODEL_SOURCE=/path/to/gptqmodel \
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/1b \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w4a8_gptq.yaml \
  --quant-output-dir work_dirs/minicpm_v_4_7_1b_w4a8_gptq \
  --export-output-dir work_dirs/minicpm_v_4_7_1b_w4a8_export \
  --device cuda:0 \
  --overwrite
```

MoE W8A8:

```bash
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/35b-a3b \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/minicpm_v_4_7_moe_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm_v_4_7_35b_a3b_export \
  --overwrite
```

MoE W4A8 AutoRound (in-memory `Qwen3_5MoeForCausalLM` → AutoRound mixed bits → restore packed experts → float `quant_weight`; Vision stays W8A8):

```bash
CUDA_VISIBLE_DEVICES=2 GPTQMODEL_SOURCE=/path/to/gptqmodel \
PYTHONPATH=. python examples_merak/llm/minicpm_v_4_7/minicpm_v_4_7_workflow.py \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/35b-a3b \
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

Registered unified capability is **W8A8 / full only**. The W4A8 GPTQ/AutoRound
YAMLs above remain workflow examples and are **not** registered as unified
recipes.

| Recipe ID | Variant | Artifact format | Unified scope |
| --- | --- | --- | --- |
| `minicpm-v-4.7-1b` | Dense 1B | `minicpm_v_4_7` | W8A8 / full |
| `minicpm-v-4.7-35b-a3b` | MoE 35B-A3B | `minicpm_v_4_7` | W8A8 / full |

### Unified `build_model` → `inference_model`

```bash
# Dense 1B
python -m examples_merak.unified_api.build \
  --model-id minicpm-v-4.7-1b \
  --model-dir /data01/datasets/Minicpm-v4.7/weights/1b \
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
  --model-dir /data01/datasets/Minicpm-v4.7/weights/35b-a3b \
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

Both recipes keep the workflow root layout (`vision_4x` / `vision_16x` / `llm`)
and validate through `export_meta_info.json`. Unified-API golden dump is not
enabled on this adapter; use the workflow `dump_golden` path instead.

## Notes

- Float and HMONNX paths both require `downsample_mode` to stay consistent.
- Processor loading uses a Transformers compatibility patch for MiniCPM remote video code.
- Export writes root `export_meta_info.json` plus `vision_4x/`, `vision_16x/`, and `llm/`.

