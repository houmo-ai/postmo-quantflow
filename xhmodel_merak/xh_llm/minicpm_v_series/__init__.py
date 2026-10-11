"""Shared MiniCPM-V helpers reused by 4.6 / 4.7 adapters."""

from .text_export import export_minicpm_qwen35_text_hmonnx
from .token_capacity_vision import (
    DEFAULT_TOKEN_CAPACITY,
    PATCH_SIZE,
    MiniCPMTokenCapacityVisionExportModel,
    build_token_capacity_attention_biases,
    build_token_capacity_position_ids,
    build_token_capacity_vision_export_model,
    crop_token_capacity_image_embeds,
    hierarchical_merge_permutation,
    image_token_count,
    merge_factor,
    pad_packed_patch_tokens,
    prepare_token_capacity_vision_inputs,
    split_packed_pixel_values,
    token_capacity_artifact_name,
    token_capacity_hmonnx_path,
    validate_token_capacity,
)
from .token_capacity_vision_export import export_vision_profile, vision_input_names

__all__ = [
    "DEFAULT_TOKEN_CAPACITY",
    "PATCH_SIZE",
    "MiniCPMTokenCapacityVisionExportModel",
    "build_token_capacity_attention_biases",
    "build_token_capacity_position_ids",
    "build_token_capacity_vision_export_model",
    "crop_token_capacity_image_embeds",
    "export_minicpm_qwen35_text_hmonnx",
    "export_vision_profile",
    "hierarchical_merge_permutation",
    "image_token_count",
    "merge_factor",
    "pad_packed_patch_tokens",
    "prepare_token_capacity_vision_inputs",
    "split_packed_pixel_values",
    "token_capacity_artifact_name",
    "token_capacity_hmonnx_path",
    "validate_token_capacity",
    "vision_input_names",
]
