# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Reuse native Qwen3-VL geometry and the existing H3 vision graph contract."""

import json
from pathlib import Path

import torch

from .feature_graph import load_component_weights


def load_vision_source(model_dir):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding

    root = Path(model_dir) / "text_encoder"
    data = json.loads((root / "config.json").read_text())["vision_config"]
    config = Qwen3VLVisionConfig.from_dict(data)
    with torch.device("meta"):
        source = Qwen3VLVisionModel(config)
    load_component_weights(source, root, "model.visual.")
    source.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
    return source.eval().requires_grad_(False), data


class VisionFeatureAdapter(torch.nn.Module):
    """Native Qwen vision protocol over exported per-frame packed sessions.

    Native vision attention is independent between temporal patch groups. The
    original grid and media order are retained when their outputs are joined.
    This adapter owns no floating vision backbone or fallback weights.
    """

    def __init__(self, invoke, config, *, gears, device="cpu", dtype=torch.float16):
        super().__init__()
        from ..qwen_image21.vision_graph import PackedVisionSession

        capacities = [gear["capacity"] for gear in gears]
        if (
            not capacities
            or capacities != sorted(set(capacities))
            or any(type(value) is not int or value <= 0 or value % 4 for value in capacities)
            or len({gear["graph"] for gear in gears}) != len(gears)
        ):
            raise ValueError("vision gears require distinct graphs and increasing patch capacities")
        self.sessions = [
            (
                gear["capacity"],
                PackedVisionSession(
                    lambda *values, name=gear["graph"]: invoke(name, *values),
                    gear["capacity"],
                    int(config["num_position_embeddings"] ** 0.5),
                    config["hidden_size"] // config["num_heads"],
                    device=device,
                    dtype=dtype,
                ),
            )
            for gear in gears
        ]
        self.dtype = dtype
        self.spatial_merge_size = config["spatial_merge_size"]

    def forward(self, pixel_values, grid_thw, **kwargs):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.is_floating_point():
            raise ValueError("vision grids must be an integer [media, 3] tensor")
        grids = grid_thw.detach().cpu().tolist()
        if not grids or any(t <= 0 or h <= 0 or w <= 0 or h % 2 or w % 2 for t, h, w in grids):
            raise ValueError("vision grids require positive temporal groups and even spatial dimensions")
        if pixel_values.shape != (sum(t * h * w for t, h, w in grids), 1536):
            raise ValueError("vision pixels do not match the complete image/video grids")
        outputs, offset = [[], [], [], []], 0
        for temporal, height, width in grids:
            count = height * width
            session = next((session for capacity, session in self.sessions if count <= capacity), None)
            if session is None:
                raise ValueError(f"vision patch count {count} exceeds exported maximum {self.sessions[-1][0]}")
            for _ in range(temporal):
                result = session(pixel_values[offset : offset + count], (1, height, width))
                for rows, value in zip(outputs, result, strict=True):
                    rows.append(value[0])
                offset += count
        joined = [torch.cat(values, dim=0) for values in outputs]
        return BaseModelOutputWithDeepstackFeatures(pooler_output=joined[0], deepstack_features=joined[1:])
