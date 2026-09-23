from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .export_common import build_component_model, quantize_and_export
from .minicpmo_vision_model import build_resampler_pos_embed


def capture_vision_inputs(host: Any, model: Any) -> tuple[torch.Tensor, ...]:
    patch_size = host.vpm.embeddings.patch_size
    height, width = model.wrap_cfg.image_slice_max_size
    patch_count = height * width
    values = torch.zeros((1, 3, patch_size, patch_count * patch_size), dtype=torch.float16)
    position_ids = torch.arange(patch_count, dtype=torch.int32).unsqueeze(0)
    attention_mask = torch.zeros((1, 1, patch_count, patch_count), dtype=torch.float16)
    resampler_pos_embed = build_resampler_pos_embed(
        (height, width), patch_count, host.resampler.pos_embed, torch.float16
    ).permute(1, 0, 2)
    resampler_key_padding_mask = torch.zeros((1, patch_count), dtype=torch.float16)
    return values, position_ids, attention_mask, resampler_pos_embed, resampler_key_padding_mask


def export_minicpm_o_4_5_vision(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    model = build_component_model(model_dir, component_cfg)
    host = model.get_hf_model(device_map="cpu")
    model.init_wrap_model(host)
    graph = Path(
        quantize_and_export(
            model,
            capture_vision_inputs(host, model),
            work_dir / "visual",
            f"{model_name or 'minicpm_o_4_5'}_vision_offline_{target_device}_{component_cfg['quant_type']}",
            device,
        )
    )
    return {"quant_type": component_cfg["quant_type"], "graphs": {"main": str(graph.relative_to(work_dir))}}


__all__ = ["capture_vision_inputs", "export_minicpm_o_4_5_vision"]
