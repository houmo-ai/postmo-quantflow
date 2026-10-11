"""CPU-only contracts for shared MiniCPM-V 4.6/4.7 helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xhmodel_merak.xh_llm import minicpm_v_series as series
from xhmodel_merak.xh_llm.minicpm_v_series import (
    DEFAULT_TOKEN_CAPACITY,
    PATCH_SIZE,
    build_token_capacity_attention_biases,
    build_token_capacity_position_ids,
    crop_token_capacity_image_embeds,
    export_minicpm_qwen35_text_hmonnx,
    hierarchical_merge_permutation,
    image_token_count,
    merge_factor,
    pad_packed_patch_tokens,
    token_capacity_artifact_name,
    token_capacity_hmonnx_path,
    validate_token_capacity,
    vision_input_names,
)


def test_minicpm_v_series_package_exports_shared_helpers() -> None:
    assert series.__doc__
    assert "4.6" in series.__doc__ and "4.7" in series.__doc__
    assert PATCH_SIZE == 14
    assert DEFAULT_TOKEN_CAPACITY == 1536
    assert callable(export_minicpm_qwen35_text_hmonnx)
    for name in (
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
    ):
        assert name in series.__all__
        assert hasattr(series, name)


@pytest.mark.parametrize(
    ("downsample_mode", "factor", "names"),
    (
        ("4x", 2, ("pixel_values", "position_ids", "pre_attention_bias")),
        (
            "16x",
            4,
            (
                "pixel_values",
                "position_ids",
                "pre_attention_bias",
                "post_attention_bias",
            ),
        ),
    ),
)
def test_minicpm_v_series_mode_helpers(
    downsample_mode: str,
    factor: int,
    names: tuple[str, ...],
) -> None:
    assert merge_factor(downsample_mode) == factor
    assert vision_input_names(downsample_mode) == names
    assert token_capacity_artifact_name(16, downsample_mode) == (
        f"minicpm_v_vision_tokens_16_{downsample_mode}.onnx"
    )
    assert token_capacity_hmonnx_path(Path("/tmp/capacity"), 16, downsample_mode) == Path(
        f"/tmp/capacity/{downsample_mode}/hmonnx/minicpm_v_vision_tokens_16_{downsample_mode}.onnx"
    )


def test_minicpm_v_series_token_capacity_geometry() -> None:
    validate_token_capacity((4, 4), 16, "4x")
    with pytest.raises(ValueError, match="must be divisible by 2"):
        validate_token_capacity((3, 4), 16, "4x")
    with pytest.raises(ValueError, match="exceeding capacity"):
        validate_token_capacity((8, 8), 16, "4x")

    assert hierarchical_merge_permutation((4, 4), "4x").tolist() == [
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        10,
        11,
        14,
        15,
    ]
    assert image_token_count(torch.tensor([[4, 4], [2, 2]]), "4x") == 5


def test_minicpm_v_series_fixed_capacity_abi_helpers() -> None:
    target = (4, 4)
    capacity = 16
    pixel_values = torch.arange(
        1 * 3 * PATCH_SIZE * target[0] * target[1] * PATCH_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, PATCH_SIZE, target[0] * target[1] * PATCH_SIZE)

    padded = pad_packed_patch_tokens(pixel_values, target, capacity, "4x")
    assert padded.shape == (1, 3, PATCH_SIZE, capacity * PATCH_SIZE)

    position_ids = build_token_capacity_position_ids(
        target,
        capacity,
        "4x",
        positions_per_side=8,
    )
    assert position_ids.shape == (1, capacity)
    assert position_ids.dtype == torch.int32

    pre_bias, post_bias = build_token_capacity_attention_biases(
        target,
        capacity,
        "4x",
        dtype=torch.float16,
    )
    assert pre_bias.shape == (1, 1, 1, capacity)
    assert post_bias.shape == (1, 1, 1, capacity // 4)
    assert torch.count_nonzero(pre_bias == 0) == target[0] * target[1]
    assert torch.count_nonzero(post_bias == 0) == (target[0] * target[1]) // 4

    embeds = torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    cropped = crop_token_capacity_image_embeds(embeds, target, capacity, "4x")
    torch.testing.assert_close(cropped, embeds)
