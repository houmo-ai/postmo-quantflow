# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Map Sol-H3's feature-conditioning request to GPTQModel's public recipe."""

from pathlib import Path


def quantize_conditioner(assets, output_dir, options, *, layers=50, device="cuda:0"):
    import torch
    from gptqmodel.recipes import quantize_qwen3_vl

    options = dict(options)
    supported = {"method", "calibration_file", "bits", "group_size", "seed", "iters"}
    if options.keys() - supported:
        raise ValueError(f"unsupported Sol-H3 conditioner quantization options: {sorted(options.keys() - supported)}")
    if "calibration_file" not in options:
        raise ValueError("conditioner quantization requires native pre-tokenized calibration_file")
    calibration = torch.load(Path(options.pop("calibration_file")).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(calibration, list):
        raise ValueError("calibration_file must contain a list of native processor tensor dictionaries")
    if options.get("bits", 4) != 4:
        raise ValueError("Sol-H3 quantized conditioner deployment currently requires W4")
    return quantize_qwen3_vl(
        model_dir=str(assets["h3_model"] / "text_encoder"),
        tokenizer_dir=str(assets["h3_model"] / "tokenizer"),
        processor_dir=str(assets["h3_model"] / "processor"),
        output_dir=str(output_dir),
        calibration_data=calibration,
        feature_layer=layers,
        device=str(device),
        **options,
    )
