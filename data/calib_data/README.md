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
`realworldqa_search_64.jsonl`. This self-contained JSONL includes the same 64 RealWorldQA
records used in the earlier candidate experiments. They are search samples,
not an independent acceptance set. Each row records the prompt, reference
answer, original dataset index, original image filename (`image_name`), image
SHA256 and `image_base64`. The Base64 preserves the exact image file bytes from
the existing `RealWorldQA.tsv` search pool; it does not resize or recompress them.
The JSONL is about 13.3 MiB and works from a fresh checkout without the original
image directory or an Artifactory download. Existing YAML resource URIs remain
unchanged. GPTQModel must include the `image_base64` reader support.

Custom JSONL accepts either `image_base64` (raw Base64, no data-URI prefix) or
`image` (a local absolute path or a path relative to the JSONL), together with
`text` and `answer`. Do not set both image fields. Optional `image_sha256` is
verified against the encoded file bytes before decoding; malformed data fails.
Image and text screening use
separate KL/PPL gates; final solver calibration has its own dataset setting.

Source: [xAI RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA),
licensed under CC BY-ND 4.0. The selected image bytes are unchanged; the local
records format question/answer fields for the search interface. Dataset assets
retain their upstream license; see `licenses/RealWorldQA/LICENSE` and
`THIRD_PARTY_NOTICES` at the repository root for attribution.
