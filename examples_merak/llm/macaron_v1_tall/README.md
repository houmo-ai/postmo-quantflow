# Macaron-V1-Tall

Macaron-V1-Tall reuses the Qwen3.5-MoE quantization and base-export
workflow. `MacaronV1TallWorkflow` injects one LoRA bundle containing four
adapter ONNX files after the Qwen graph export:

- `L0`–`L3` have one shared initializer schema.
- Prefill and Decode bind canonical `L0` initializers.
- The compiler selects `L1`–`L3` by initializer-name replacement before a
  request is built.
- Routed experts use the 27-input `MoeBlock` ABI.

Two Macaron workflow YAMLs are available under
`configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/`:

- `macaron_v1_tall_w8a8.yaml`: W8A8 backbone; all LoRA weights inherit
  their target projection format.
- `macaron_v1_tall_w4a8.yaml`: W4A8 backbone with W8A8 non-routed and
  routed-expert LoRA weights.

Each YAML retains its corresponding Qwen quantization and calibration
settings. The `macaron_lora` section configures only LoRA extraction and
quantization.

## Supported entry points

- `macaron_workflow.py`: standard quantize/export/Golden workflow.
- `hmonnx_demo.py`: HMONNX generation demo for L0--L3.
- `hf_demo.py`: Hugging Face + PEFT reference demo for L0–L3.

The workflow wrapper uses `AutoLLMWorkflow.from_config(...)`; the YAML model
type selects `MacaronV1TallWorkflow` without a Macaron-specific registry.

Golden generation reuses `Qwen35Workflow.dump_golden()` instead of carrying
a second implementation. Like the Qwen backbone, a full-model run executes
every visual token gear before language Prefill/Decode.

The public workflow keeps the same `export(...)` interface as Qwen. It creates
a derived sibling Qwen export only as an internal intermediate and removes it
after the final Macaron export has been assembled. Before changing a graph, it
builds one immutable plan and requires Prefill and Decode to contain the same
LoRA targets and quantization formats. It then writes the complete L0–L3
bundle, appends L0 to each normal Qwen stage external-data file, and binds
non-routed and routed LoRA in one graph load/save per stage.

Like the Qwen3.5/Qwen3.6 example, `--export-from-quanted-model` applies
`config_overrides={"quant": None}` to skip quantization. Without the flag, the
workflow follows the YAML `quant` section. An already quantized Macaron
checkpoint must still contain `loras/L0` through `loras/L3` for injection.

The final directory is first assembled in a sibling temporary directory and
is renamed to the requested path only after bundle, initializer-order, and
27-input `MoeBlock` validation succeeds. A failed injection therefore does
not leave a deliverable-looking partial directory. The workflow does not
write extracted adapter `.pt` files, per-layer packs, or a non-routed
intermediate export. Do not run individual injection stages by hand.

Both LoRA groups accept `inherit` or a standard XH2a quantization type such
as `w8a8h1_sefp`. `inherit` uses the target Linear or MoeBlock projection
weight format. An explicit routed-expert scheme overrides only LoRA A/B
weights; its activation and partial-sum formats remain those of MoeBlock.
All L0--L3 adapters in one export use the same quantization schema.

The final export retains the normal Qwen layout plus:

```text
runtime_lora/
  manifest.json
  L0.onnx, L0.onnx.data
  L1.onnx, L1.onnx.data
  L2.onnx, L2.onnx.data
  L3.onnx, L3.onnx.data
```

The compiler/runtime selects L1–L3 by replacing matching initializer values
from the selected bundle. For Python HMONNX validation, the generation demo
creates small, test-only L0–L3 overlay graphs. They symlink immutable base
weights and point LoRA initializer references at the selected quantized pack;
the deliverable export remains unchanged.
