# Calibration data

AutoRound quantization for Qwen3.5/Qwen3.6 and Gemma4 reads the local
`NeelNanda/pile-10k` JSONL from this path:

```text
data/calib_data/NeelNanda-pile-10k.jsonl
```

The JSONL is intentionally committed because customer and CI environments may
not have public network access. CI should use this checked-in file directly and
should not download calibration data at runtime.

MiniCPM-V-4.5 W4A8 GPTQ uses the QwenVL-style mixed text pool:

```text
data/calib_data/minicpm_v_4_5_qwen_mixed_295.jsonl
```

The workflow selects 64 records without replacement with a fixed seed.

Qwen3.5/Qwen3.6 automatic mixed precision can additionally use
`realworldqa_search_64.jsonl`. This manifest identifies the same 64 RealWorldQA
records used in the earlier candidate experiments. They are search samples,
not an independent acceptance set. Each row records the prompt, reference
answer, original dataset index, relative image path and image SHA-256.
The local `realworldqa_search_images/` directory contains the original decoded
image bytes from `RealWorldQA.tsv`; image binaries are not bundled in Git.
On another machine, supply these images with the manifest or configure your own
`image_dataset` JSONL containing local `image`, `text` and `answer` fields.
The recipe does not download images implicitly. Image and text screening use
separate KL/PPL gates; final solver calibration has its own dataset setting.
