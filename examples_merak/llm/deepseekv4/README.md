# DeepSeek-V4 Flash

Use the workflow YAML as the primary export interface. The default production
profile is W4A8:

```bash
python examples_merak/llm/deepseekv4/deepseek_v4_workflow.py \
  --model-dir /path/to/deepseek-v4-flash-W8E4G64 \
  --config-path configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731/deepseek_v4_flash_0731_w4a8.yaml \
  --export-output-dir work_dirs/deepseek_v4_flash_w4a8
```

For the higher-precision activation profile, select
`deepseek_v4_flash_0731_w4a16.yaml`. Both profiles consume the same W8E4
GPTQModel/AutoRound checkpoint. The graph quant type is `w8a8h1_sefp` or
`w8a16h1_sefp`; routed expert weights remain W4 because they come from the E4
source tensors.

The workflow defaults to streamed low-memory export, packed W4 weights, all 43
layers, a 256-token prefill chunk, and a 262144-token context. Use
`--dump-golden --golden-device-map 0 1 2 3` to emit formal prefill/decode golden
data after export.

Add `--max-layers 6` to export the diagnostic first-six prefix with the same
prefill/decode shapes and context limit.

CSA uses one `SparseAttentionSelectKV` VP node per layer in both graphs,
with `pool_size=1`, TopK-512 and shared FP16 compressed payloads. The Host
provides causal int32 `csa_candidate_lengths` of shape `[1,256]` for prefill
and `[1,1]` for decode, including zero lengths for padding. This replaces the
full-cache validity matrix and the expanded score/TopK/Gather graph. The
attention mask still covers selected rows and SWA independently. State ABI 2
requires re-exporting older graphs and the SDK update from Gerrit 46158.

`export_hmonnx.py` remains available as the low-level diagnostic CLI. Its
default quant type is W4A8-compatible `w8a8h1_sefp`.
