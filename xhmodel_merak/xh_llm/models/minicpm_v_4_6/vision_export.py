"""ONNX/HMONNX export for MiniCPM-V-4.6 token-capacity Vision graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision_export import (
    export_vision_profile as _export_vision_profile,
    vision_input_names,
)


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
        model_label="MiniCPM-V-4.6",
        artifact_family="MiniCPM-V-4.6",
        trust_remote_code=False,
        place_full_model_on_device=True,
    )


__all__ = ["export_vision_profile", "vision_input_names"]
