"""ONNX/HMONNX export for MiniCPM-V-4.7 token-capacity Vision graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from xhmodel_merak.xh_llm.minicpm_v_series.vision_gears import (
    factor_patch_grid,
    normalize_image_token_gears,
    output_token_divisor,
)

from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision_export import (
    export_vision_profile as _export_vision_profile,
    vision_input_names,
)

from .auto_classes import register_minicpmv47_auto_classes


def export_vision_profile(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    token_capacity: int,
    downsample_mode: str,
    validation_target: tuple[int, int],
    device: str,
    keep_onnx: bool = False,
) -> dict[str, Any]:
    register_minicpmv47_auto_classes()
    return _export_vision_profile(
        model_dir=model_dir,
        work_dir=work_dir,
        target_device=target_device,
        quant_type=quant_type,
        token_capacity=token_capacity,
        downsample_mode=downsample_mode,
        validation_target=validation_target,
        device=device,
        keep_onnx=keep_onnx,
        model_label="MiniCPM-V-4.7",
        artifact_family="MiniCPM-V-4.7",
        trust_remote_code=True,
        place_full_model_on_device=False,
    )


def export_vision_gears(
    *,
    image_token_gears,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    downsample_mode: str,
    validation_target: tuple[int, int],
    device: str,
    keep_onnx: bool = False,
) -> dict[str, Any]:
    """Export one graph per post-merge capacity, with portable routing metadata."""
    register_minicpmv47_auto_classes()
    gears = normalize_image_token_gears(image_token_gears)
    divisor = output_token_divisor(downsample_mode)
    manifest = f"visual_{downsample_mode}_gears.json"
    result = {
        "schema_version": 2,
        "downsample_mode": downsample_mode,
        "token_unit": "post_spatial_merge",
        "image_token_gears": list(gears),
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
        "gear_manifest": manifest,
        "gears": [],
    }
    for gear in gears:
        capacity = gear * divisor
        target = validation_target
        if target[0] * target[1] > capacity:
            target = factor_patch_grid(capacity, downsample_mode)
        entry = _export_vision_profile(
            model_dir=model_dir, work_dir=work_dir, target_device=target_device,
            quant_type=quant_type, token_capacity=capacity,
            downsample_mode=downsample_mode, validation_target=target,
            device=device, keep_onnx=keep_onnx, model_label="MiniCPM-V-4.7",
            artifact_family="MiniCPM-V-4.7", trust_remote_code=True,
            profile_name=f"visual_{downsample_mode}_m{gear}",
            # HMONNX is traced directly from TorchFX to preserve MaskedAdd.
            # Native and executable HMONNX comparisons cover every gear.
            validate_onnx=keep_onnx,
        )
        entry["image_token_capacity"] = gear
        result["gears"].append(entry)
        (work_dir / manifest).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["export_vision_profile", "export_vision_gears", "vision_input_names"]
