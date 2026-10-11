"""Token-capacity Vision helpers for MiniCPM-V-4.7."""

from __future__ import annotations

from pathlib import Path

from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision import (
    DEFAULT_TOKEN_CAPACITY,
    PATCH_SIZE,
    MiniCPMTokenCapacityVisionExportModel,
    build_token_capacity_attention_biases,
    build_token_capacity_position_ids,
    build_token_capacity_vision_export_model as _build_token_capacity_vision_export_model,
    crop_token_capacity_image_embeds,
    hierarchical_merge_permutation,
    image_token_count,
    merge_factor,
    pad_packed_patch_tokens,
    prepare_token_capacity_vision_inputs,
    split_packed_pixel_values,
    token_capacity_artifact_name as _token_capacity_artifact_name,
    token_capacity_hmonnx_path as _token_capacity_hmonnx_path,
    validate_token_capacity,
)

MiniCPMV47TokenCapacityVisionExportModel = MiniCPMTokenCapacityVisionExportModel


def build_token_capacity_vision_export_model(
    full_model,
    token_capacity: int,
    downsample_mode: str = "16x",
) -> MiniCPMTokenCapacityVisionExportModel:
    return _build_token_capacity_vision_export_model(
        full_model,
        token_capacity,
        downsample_mode,
        model_label="MiniCPM-V-4.7",
    )


def token_capacity_artifact_name(
    token_capacity: int,
    downsample_mode: str,
) -> str:
    return _token_capacity_artifact_name(
        token_capacity,
        downsample_mode,
        artifact_family="minicpm_v_4_7",
    )


def token_capacity_hmonnx_path(
    capacity_root: Path,
    token_capacity: int,
    downsample_mode: str,
) -> Path:
    return _token_capacity_hmonnx_path(
        capacity_root,
        token_capacity,
        downsample_mode,
        artifact_family="minicpm_v_4_7",
    )


__all__ = [
    "DEFAULT_TOKEN_CAPACITY",
    "PATCH_SIZE",
    "MiniCPMV47TokenCapacityVisionExportModel",
    "build_token_capacity_attention_biases",
    "build_token_capacity_position_ids",
    "build_token_capacity_vision_export_model",
    "crop_token_capacity_image_embeds",
    "hierarchical_merge_permutation",
    "image_token_count",
    "merge_factor",
    "pad_packed_patch_tokens",
    "prepare_token_capacity_vision_inputs",
    "split_packed_pixel_values",
    "token_capacity_artifact_name",
    "token_capacity_hmonnx_path",
    "validate_token_capacity",
]
