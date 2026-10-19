"""CPU contracts for fixed-length padded MiniMax-H3 text conditioning."""

import json
from pathlib import Path

import pytest
import torch

from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import MiniMaxH3Converter
from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
    load_fixed_point_calibration,
)


def _make_model() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(7)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
        enable_ffn_scaling=False,
    ).eval()


def _run_model(
    model: MiniMaxH3Transformer3DModel,
    video: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_length = text.shape[1]
    video_indices = torch.arange(text_length, text_length + video.shape[1])
    audio_indices = torch.arange(text_length + video.shape[1], text_length + video.shape[1] + audio.shape[1])
    text_indices = torch.arange(text_length)
    sequence_length = text_length + video.shape[1] + audio.shape[1]
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:text_length, 0] = torch.arange(text_length)
    token_tags = torch.cat(
        (
            torch.ones(text_length, dtype=torch.long),
            torch.zeros(video.shape[1], dtype=torch.long),
            torch.full((audio.shape[1],), 2, dtype=torch.long),
        )
    )
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    with torch.no_grad():
        return model(
            video,
            audio,
            text,
            torch.tensor([0.5]),
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            encoder_attention_mask,
            return_dict=False,
        )


def test_padded_text_mask_preserves_video_and_audio_outputs():
    model = _make_model()
    generator = torch.Generator().manual_seed(11)
    video = torch.randn(1, 2, 2, generator=generator)
    audio = torch.randn(1, 1, 4, generator=generator)
    text = torch.randn(1, 3, 12, generator=generator)

    expected_video, expected_audio = _run_model(model, video, audio, text, None)

    padded_text = torch.cat((text, torch.randn(1, 2, 12, generator=generator)), dim=1)
    attention_mask = torch.zeros(1, 1, 1, padded_text.shape[1])
    attention_mask[..., text.shape[1] :] = torch.finfo(attention_mask.dtype).min
    actual_video, actual_audio = _run_model(model, video, audio, padded_text, attention_mask)

    torch.testing.assert_close(actual_video, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(actual_audio, expected_audio, rtol=0, atol=0)


def test_padded_text_mask_rejects_incompatible_shape():
    model = _make_model()
    video = torch.randn(1, 2, 2)
    audio = torch.randn(1, 1, 4)
    text = torch.randn(1, 5, 12)

    with pytest.raises(ValueError, match="encoder_attention_mask"):
        _run_model(model, video, audio, text, torch.zeros(1, 1, 5, 5))


def test_converter_builds_additive_encoder_attention_mask():
    converter = MiniMaxH3Converter.__new__(MiniMaxH3Converter)
    converter.device = torch.device("cpu")
    converter.seed = 0
    converter.cfg = {
        "minimax_h3": {
            "input_shapes": {
                "transformer": [
                    {"name": "hidden_states", "shape": [1, 2, 2], "dtype": "float16"},
                    {"name": "audio_hidden_states", "shape": [1, 1, 4], "dtype": "float16"},
                    {"name": "encoder_hidden_states", "shape": [1, 5, 12], "dtype": "float16"},
                    {"name": "step_index", "shape": [1], "dtype": "int64"},
                    {"name": "timestep_indices", "shape": [8], "dtype": "int64"},
                    {"name": "adaln_indices", "shape": [8], "dtype": "int64"},
                    {"name": "rotary_cos", "shape": [8, 6], "dtype": "float16"},
                    {"name": "rotary_sin", "shape": [8, 6], "dtype": "float16"},
                    {"name": "video_indices", "shape": [2], "dtype": "int64"},
                    {"name": "audio_indices", "shape": [1], "dtype": "int64"},
                    {"name": "pack_gather_indices", "shape": [8], "dtype": "int64"},
                    {
                        "name": "encoder_attention_mask",
                        "shape": [1, 1, 1, 5],
                        "dtype": "float16",
                        "valid_length": 3,
                    },
                    {
                        "name": "packed_attention_mask",
                        "shape": [1, 1, 1, 8],
                        "dtype": "float16",
                    },
                ]
            }
        }
    }

    inputs = converter._build_inputs("transformer", pipe=None)
    attention_mask = inputs[-2]
    packed_attention_mask = inputs[-1]

    torch.testing.assert_close(attention_mask[..., :3], torch.zeros(1, 1, 1, 3, dtype=torch.float16))
    assert torch.all(attention_mask[..., 3:] == torch.finfo(torch.float16).min)
    torch.testing.assert_close(packed_attention_mask[..., :3], torch.zeros(1, 1, 1, 3, dtype=torch.float16))
    assert torch.all(packed_attention_mask[..., 3:5] == torch.finfo(torch.float16).min)
    torch.testing.assert_close(packed_attention_mask[..., 5:], torch.zeros(1, 1, 1, 3, dtype=torch.float16))


def test_converter_resolves_fixed_point_calibration_path(tmp_path: Path):
    configured = tmp_path / "custom_calibration.json"
    configured.write_text("{}", encoding="utf-8")

    assert MiniMaxH3Converter._resolve_fixed_point_calibration_path({}) is not None
    assert MiniMaxH3Converter._resolve_fixed_point_calibration_path({"fixed_point_calibration_path": None}) is None
    assert (
        MiniMaxH3Converter._resolve_fixed_point_calibration_path({"fixed_point_calibration_path": str(configured)})
        == configured.resolve()
    )


def test_load_fixed_point_calibration_applies_v3_configs(tmp_path: Path):
    model = _make_model()
    calibration_path = tmp_path / "minimax_h3_fixed_point_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "format": "minimax_h3_fixed_point_calibration_v3",
                "layers": [
                    {
                        "layer": 0,
                        "points": {
                            "attention_residual": {
                                "fixed_point": {
                                    "lhs": 8,
                                    "rhs": 8,
                                    "accumulator": 16,
                                    "bias": 22,
                                    "output_shift": 16,
                                }
                            },
                            "ffn_residual": {
                                "fixed_point": {
                                    "lhs": 5,
                                    "rhs": 5,
                                    "accumulator": 10,
                                    "bias": 18,
                                    "output_shift": 15,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    configs = load_fixed_point_calibration(model, calibration_path)

    assert configs[0][1] == (8, 8, 16, 22, 16)
    assert configs[0][3] == (5, 5, 10, 18, 15)
    assert model.transformer_blocks[0].fixed_point_configs[1] == (8, 8, 16, 22, 16)
    assert model.transformer_blocks[0].fixed_point_configs[3] == (5, 5, 10, 18, 15)


def test_converter_applies_fixed_point_calibration(tmp_path: Path):
    model = _make_model()
    calibration_path = tmp_path / "minimax_h3_fixed_point_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "format": "minimax_h3_fixed_point_calibration_v3",
                "layers": [
                    {
                        "layer": 0,
                        "points": {
                            "attention_residual": {
                                "fixed_point": {
                                    "lhs": 7,
                                    "rhs": 7,
                                    "accumulator": 14,
                                    "bias": 20,
                                    "output_shift": 14,
                                }
                            },
                            "ffn_residual": {
                                "fixed_point": {
                                    "lhs": 4,
                                    "rhs": 4,
                                    "accumulator": 8,
                                    "bias": 16,
                                    "output_shift": 12,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    converter = MiniMaxH3Converter.__new__(MiniMaxH3Converter)
    converter._apply_fixed_point_calibration(
        model,
        {"fixed_point_calibration_path": str(calibration_path)},
    )

    assert model.transformer_blocks[0].fixed_point_configs[1] == (7, 7, 14, 20, 14)
    assert model.transformer_blocks[0].fixed_point_configs[3] == (4, 4, 8, 16, 12)
