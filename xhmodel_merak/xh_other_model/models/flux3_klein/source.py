# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Source identity and strict native checkpoint loading."""

import json
import math
from pathlib import Path


MODEL_ID = "flux-3-image-klein"


def read_source(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    text_config = json.loads((root / "text_encoder/config.json").read_text())
    if config.get("model") != MODEL_ID or config.get("format_version") != 1 or config.get("quantization") is not None:
        raise ValueError("source must be the floating-point flux-3-image-klein checkpoint")
    dit, text = config["dit"], config["text_encoder"]
    heads, hidden = dit.get("heads"), dit.get("hidden")
    if (
        dit.get("flow_map") is not True
        or dit.get("axes_dim") != [32, 32, 32, 32]
        or type(heads) is not int
        or heads <= 0
        or type(hidden) is not int
        or hidden != 128 * heads
        or dit.get("theta") != 10000
        or dit.get("image_rope_scale") != 0.5
    ):
        raise ValueError("source requires the distilled four-axis flow-map DiT")
    if (
        dit.get("out_modes") != ["image"]
        or dit["in_channels"]["image"] != 128
        or dit["in_channels"]["image_cond"] != 128
    ):
        raise ValueError("source must use 128-channel image tokens")
    if text.get("layers") != [4, 8, 12, 16, 20, 24, 28, 32] or text.get("pad_multiple") != 80:
        raise ValueError("source requires the checkpoint's eight-layer text feature contract")
    if text_config.get("model_type") != "qwen3_vl" or text_config["text_config"]["num_hidden_layers"] < 32:
        raise ValueError("source text encoder must provide the selected Qwen3-VL layers")
    if dit["in_channels"]["text"] != 8 * text_config["text_config"]["hidden_size"]:
        raise ValueError("DiT text width disagrees with the text encoder")
    if text.get("path") != "text_encoder" or config["vae"].get("path") != "ae.safetensors":
        raise ValueError("source component paths must remain inside the checkpoint package")
    sampler = config["sampler"]
    if sampler.get("guidance") is not None or type(sampler["steps"]) is not int or sampler["steps"] <= 0:
        raise ValueError("distilled sampling requires positive steps without CFG")
    if not math.isfinite(sampler["alpha"]) or sampler["alpha"] <= 0:
        raise ValueError("sampler alpha must be finite and positive")
    for name in ("model.safetensors", "ae.safetensors", "text_encoder/model.safetensors.index.json"):
        if not (root / name).is_file():
            raise ValueError(f"checkpoint component missing: {name}")
    index = json.loads((root / "text_encoder/model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if Path(name).name != name or not (root / "text_encoder" / name).is_file():
            raise ValueError("text encoder shard is missing or outside the checkpoint")
    return config, text_config


def load_native(root, role, *, dtype=None, device="cpu"):
    import torch
    from safetensors.torch import load_file

    from .native_autoencoder import AutoEncoder
    from .native_dit import DiT

    config, _ = read_source(root)
    with torch.device("meta"):
        module = DiT(True, config["dit"]) if role == "dit" else AutoEncoder(config["vae"].get("architecture"))
    state = load_file(str(Path(root) / ("model.safetensors" if role == "dit" else "ae.safetensors")), device="cpu")
    module.load_state_dict(state, strict=True, assign=True)
    return module.to(device=device, dtype=dtype or torch.float32).eval()
