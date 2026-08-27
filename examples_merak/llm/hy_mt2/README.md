# Hy-MT2-1.8B on XH2A

This workflow adapts the official Apache-2.0
[`tencent/Hy-MT2-1.8B`](https://huggingface.co/tencent/Hy-MT2-1.8B)
checkpoint (`HunYuanDenseV1ForCausalLM`) to XH2A. It exports W8A8 prefill
and decode HMONNX graphs and can generate golden data for both graphs.

The workflow config is
`configs_merak/workflows/xh2a/other_models/hy_mt2/hy_mt2.yaml`. The explicit
`quant()` stage is skipped because this legacy adapter performs XH2A PTQ while
building the export graph; `export()` therefore produces quantized HMONNX.

## Download

```bash
hf download tencent/Hy-MT2-1.8B --local-dir <model_dir>
hf download yash9439/flores200 --repo-type dataset --local-dir <flores_dir>
```

The verified `model.safetensors` is 4,077,072,784 bytes with SHA-256
`29e9117a44c79f81857613601968ff482d8a23c2d6736a1710bba9e5ca4762e5`.
The logical tied-weight model has 1,791,080,448 parameters; the checkpoint
stores a duplicate `lm_head.weight`, so its tensor entries sum to
2,038,515,712 parameters.

The evaluation mirror contains the original FLORES-200 dev/devtest aligned
sentences in Parquet form. The default evaluation uses 100 devtest rows in
each of `zho_Hans→eng_Latn` and `eng_Latn→zho_Hans`. chrF++ is the primary,
language-neutral metric; BLEU is also reported. Greedy decoding is used by
default so FP/HMONNX differences are attributable to quantization rather than
sampling. The evaluated `devtest.parquet` SHA-256 is
`37adade6f0174c18b1169721196ae22d6294157c46d9d77f4b2152afa0d5d2b5`.
Evaluation requires `pyarrow` and `sacrebleu` (2.5.1 was used below).

## Export and golden

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD python \
  examples_merak/llm/hy_mt2/hy_mt2_workflow.py \
  --model-dir <model_dir> \
  --export-output-dir work_dirs/hy_mt2_1_8b_w8a8 \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

The output root contains `export_meta_info.json`, `meta.json`, tokenizer and
embedding assets, prefill/decode HMONNX, and per-graph golden directories. The
golden workflow uses the 25-token official prompt for prefill, then feeds the
prefill argmax embedding and the mutated prefill KV cache into decode. The
generated shapes are `[1, 256, 2048]` for prefill input and `[1, 1, 2048]` for
decode input; both graphs produce `[1, 1, 120818]` logits.

## HMONNX conversation smoke test

```bash
USE_TRITON_MATMUL=0 CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD python \
  examples_merak/llm/hy_mt2/chat_hmonnx.py \
  --model-dir <model_dir> \
  --export-path work_dirs/hy_mt2_1_8b_w8a8 \
  --source-text "今天天气真好。" \
  --source-code zho_Hans \
  --target-code eng_Latn
```

`USE_TRITON_MATMUL=0` selects the compiler-free simulator path. For the target
aligned CUDA extension path, omit it and provide a working C++17 host compiler
through `CC`/`CXX`; the extension is compiled once and then reused.

Verified outputs from the generated HMONNX are:

| Direction | Source | HMONNX response |
| --- | --- | --- |
| Chinese→English | `今天天气真好。` | `The weather is really nice today.` |
| English→Chinese | `Machine translation helps people communicate across languages.` | `机器翻译有助于人们跨越语言进行交流。` |

The compiler-free path may emit an xhquant `wb_layout` macro-order warning in
this development environment; both smoke processes still exit successfully.
The accuracy results below use the aligned CUDA extension path instead.

## FP and HMONNX accuracy

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD python \
  examples_merak/llm/hy_mt2/eval_flores.py \
  --backend fp \
  --model-dir <model_dir> \
  --dataset-dir <flores_dir> \
  --output-dir work_dirs/hy_mt2_1_8b_eval/fp \
  --limit-per-direction 100 \
  --overwrite

CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD python \
  examples_merak/llm/hy_mt2/eval_flores.py \
  --backend hmonnx \
  --model-dir <model_dir> \
  --export-path work_dirs/hy_mt2_1_8b_w8a8 \
  --dataset-dir <flores_dir> \
  --output-dir work_dirs/hy_mt2_1_8b_eval/hmonnx \
  --limit-per-direction 100 \
  --retain-sessions \
  --baseline-summary work_dirs/hy_mt2_1_8b_eval/fp/fp_summary.json \
  --overwrite
```

Prediction JSONL is flushed after each sample and reused on restart unless
`--overwrite` is passed. This makes the slower HMONNX evaluation resumable.
`--retain-sessions` avoids reparsing the prefill graph for every sample at the
cost of keeping both HMONNX graphs in GPU memory.

## Verified accuracy

The following results use FLORES-200 devtest rows 0–99 in both directions,
greedy decoding, `max_new_tokens=128`, and `repetition_penalty=1.05`. BF16 and
W8A8 records were matched by all 200 sample IDs and prompt/reference content.
HMONNX used `fast_mode=False`, retained sessions, and the aligned CUDA
extension on A100 simulator GPUs. Sharding was used only to reduce evaluation
wall time, so the recorded per-row latency is not a performance benchmark.

| Direction | Metric | BF16 | W8A8 HMONNX | Delta |
| --- | ---: | ---: | ---: | ---: |
| Chinese→English | chrF++ | 54.7512 | 54.1775 | -0.5737 |
| Chinese→English | BLEU | 22.8875 | 22.5156 | -0.3719 |
| English→Chinese | chrF++ | 29.4626 | 29.2464 | -0.2162 |
| English→Chinese | BLEU | 44.7296 | 44.4714 | -0.2582 |
| Macro average | chrF++ | 42.1069 | 41.7119 | -0.3949 |
| Macro average | BLEU | 33.8085 | 33.4935 | -0.3150 |

Exact generated-string agreement with BF16 is 46/100 for Chinese→English and
48/100 for English→Chinese; the reference-based metric deltas above remain
below 0.6 point. Reproducible outputs are in
`work_dirs/hy_mt2_1_8b_eval/fp/fp_summary.json` and
`work_dirs/hy_mt2_1_8b_eval/hmonnx/hmonnx_summary.json`.
