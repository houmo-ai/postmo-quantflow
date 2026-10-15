# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Independent offset/normalization formulas and constructed codec integration."""

import copy
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from xhmodel_merak.xh_other_model.models.sol_h3.conv_graph import prepare_static_shapes
from xhmodel_merak.xh_other_model.models.sol_h3.vae_graph import (
    CausalPadConv,
    ChannelPixelNorm,
    DepthExpansion,
    DepthReduction,
    MaskedVideoDecodeGraph,
    SpatialUnpatchify,
    VideoDecodeGraph,
    decoder_mask_inputs,
)


def test_zero_and_large_half_pixelnorm_preserve_epsilon_without_overflow():
    values = torch.tensor([0, 0.0001, 20000.0, -30000.0], dtype=torch.float16).reshape(1, 4, 1, 1, 1)
    norm = ChannelPixelNorm(1e-8)
    norm.set_static_shape(values.shape)
    norm.half()
    expected = values.float() * torch.rsqrt(values.float().square().mean(1, keepdim=True) + 1e-8)
    torch.testing.assert_close(norm(values).float(), expected, rtol=0.001, atol=0.001)
    assert torch.count_nonzero(norm(torch.zeros_like(values))) == 0
    assert norm.norm.variance_epsilon == 1e-8


def test_scaled_encoder_pixelnorm_bounds_half_reciprocal_for_zero_and_small_inputs():
    eps = 1e-8 / 32**2
    norm = ChannelPixelNorm(eps)
    norm.set_static_shape((1, 4, 1, 1, 2))
    norm.half()
    values = torch.tensor([0, 1e-6, 0, -2e-6, 0, 3e-6, 0, -1e-6], dtype=torch.float16).reshape(1, 4, 1, 1, 2)
    expected = values.float() * torch.rsqrt(values.float().square().mean(1, keepdim=True) + eps)
    actual = norm(values)
    assert norm.bounded and bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, rtol=0.004, atol=0.001)
    assert not bool(norm(torch.zeros_like(values)).any())


@pytest.mark.parametrize("causal", [False, True])
def test_causal_temporal_padding_matches_replicated_boundaries(causal):
    convolution = torch.nn.Conv3d(2, 4, 3, padding=(0, 1, 1))
    graph = CausalPadConv(SimpleNamespace(conv=convolution, time_kernel_size=3))
    value = torch.randn(1, 2, 3, 3, 4)
    padding = (0, 0, 0, 0, 2, 0) if causal else (0, 0, 0, 0, 1, 1)
    expected = convolution(torch.nn.functional.pad(value, padding, mode="replicate"))
    torch.testing.assert_close(graph(value, causal=causal), expected)


def test_final_unpatchify_uses_checkpoint_width_then_height_offset_order():
    values = torch.arange(3 * 16 * 2 * 3 * 4).reshape(1, 48, 2, 3, 4)
    expected = rearrange(values, "b (c r q) f h w -> b c f (h q) (w r)", q=4, r=4)
    torch.testing.assert_close(SpatialUnpatchify(4)(values), expected, rtol=0, atol=0)


@torch.no_grad()
def test_shared_decoder_masks_preserve_four_constructed_tile_boundaries(monkeypatch, tmp_path):
    from _sol_h3_tiny_components import video_decoder_source

    from xhmodel_merak.xh_other_model.models.sol_h3 import source_io

    monkeypatch.setattr(source_io, "prepare_video_decoder", lambda source: source)

    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(74)
    source = video_decoder_source()
    source.per_channel_statistics.get_buffer("mean-of-means").fill_(0.3)
    source.per_channel_statistics.get_buffer("std-of-means").fill_(1.2)
    graph = MaskedVideoDecodeGraph(copy.deepcopy(source))
    values = torch.randn(1, 128, 2, 3, 3)
    args = decoder_mask_inputs(values, (2, 3, 3), graph.mask_scales)
    names = ["latent", *[f"spatial_mask_{i}" for i in range(len(args) - 1)]]
    lowered = prepare_static_shapes(copy.deepcopy(graph), args)
    path = export_graph(
        graph,
        args,
        tmp_path / "decode.onnx",
        names,
        ["pixels"],
        quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp"),
        dynamo=True,
    )
    runner = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        value = values[..., :h, :w]
        expected = VideoDecodeGraph(copy.deepcopy(source))(value)
        args = decoder_mask_inputs(value, (2, 3, 3), graph.mask_scales)
        actual = lowered(*args)[..., : h * 32, : w * 32]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        result = runner.run(dict(zip(names, [arg.half().cuda() for arg in args], strict=True)))
        result = result[0] if isinstance(result, (tuple, list)) else result
        result = torch.as_tensor(result).float().cpu()[..., : h * 32, : w * 32]
        assert torch.isfinite(result).all()
        assert (result - expected).norm() / expected.norm() < 0.12


def _expand_offsets_by_index(value, factors):
    dt, dh, dw = factors
    batch, channels, frames, height, width = value.shape
    result = value.new_empty(batch, channels // (dt * dh * dw), frames * dt, height * dh, width * dw)
    for channel in range(channels):
        output_channel, offset = divmod(channel, dt * dh * dw)
        time_offset, offset = divmod(offset, dh * dw)
        height_offset, width_offset = divmod(offset, dw)
        result[:, output_channel, time_offset::dt, height_offset::dh, width_offset::dw] = value[:, channel]
    return result[:, :, 1:] if dt == 2 else result


@pytest.mark.parametrize("stride", [(1, 2, 2), (2, 1, 1), (2, 2, 2)])
@torch.no_grad()
def test_depth_expansion_matches_explicit_offset_indices_and_residual(stride):
    from _sol_h3_tiny_components import causal_conv

    torch.manual_seed(44)
    factor = stride[0] * stride[1] * stride[2]
    convolution = causal_conv(8, 8 * factor)
    source = SimpleNamespace(conv=convolution, stride=stride, residual=True, out_channels_reduction_factor=1)
    value = torch.randn(1, 8, 2, 3, 4)
    padded = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 1), mode="replicate")
    conv = convolution.convolution
    raw = torch.nn.functional.conv3d(padded, conv.weight, conv.bias, padding=(0, 1, 1))
    expected = _expand_offsets_by_index(raw, stride) + _expand_offsets_by_index(value, stride).repeat(
        1, factor, 1, 1, 1
    )
    torch.testing.assert_close(DepthExpansion(source)(value, causal=False), expected, rtol=0, atol=0)


@pytest.mark.parametrize("stride,frames", [((1, 2, 2), 1), ((2, 1, 1), 3), ((2, 2, 2), 1)])
@torch.no_grad()
def test_residual_downsample_matches_explicit_offset_stack(stride, frames):
    from _sol_h3_tiny_components import causal_conv

    torch.manual_seed(44)
    factor = stride[0] * stride[1] * stride[2]
    convolution = causal_conv(8, 16 // factor, causal=True)
    value = torch.randn(1, 8, frames, 4, 6)
    extended = torch.cat((value[:, :, :1], value), 2) if stride[0] == 2 else value

    def pack(tensor):
        return torch.stack(
            [
                tensor[:, :, t :: stride[0], h :: stride[1], w :: stride[2]]
                for t in range(stride[0])
                for h in range(stride[1])
                for w in range(stride[2])
            ],
            2,
        ).flatten(1, 2)

    padded = torch.nn.functional.pad(extended, (0, 0, 0, 0, 2, 0), mode="replicate")
    conv = convolution.convolution
    raw = torch.nn.functional.conv3d(padded, conv.weight, conv.bias, padding=(0, 1, 1))
    skip = pack(extended)
    group = 8 * factor // 16
    expected = pack(raw) + skip.unflatten(1, (16, group)).mean(2)
    module = DepthReduction(SimpleNamespace(conv=convolution, stride=stride, group_size=group))
    torch.testing.assert_close(module(value), expected, rtol=0, atol=0)
