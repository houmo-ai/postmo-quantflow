# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Channel-first VAE normalization: zero, outlier and low-precision inputs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture
def vae_graph():
    path = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21/vae_graph.py"
    spec = importlib.util.spec_from_file_location("qwen_image21_vae_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [4, 5])
def test_vae_normalization_uses_fp32_norm(vae_graph, dtype, rank):
    source = nn.Module()
    source.channel_first = True
    source.scale = 2.0
    source.gamma = nn.Parameter(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=dtype).reshape(4, *([1] * (rank - 2))))
    source.bias = 0.0
    shape = (1, 4, 3, 5) if rank == 4 else (1, 4, 1, 3, 5)
    value = torch.zeros(shape, dtype=dtype)
    value[:, 0] = 14000
    value[..., 0, 0] = 0
    expected = (value.float() / torch.linalg.vector_norm(value.float(), dim=1, keepdim=True).clamp_min(1e-12)).to(dtype)
    expected = expected * source.scale * source.gamma
    actual = vae_graph.ChannelNormalize(source)(value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[..., 0, 0]) == 0


def test_vae_rejects_patchified_variant(vae_graph):
    source = SimpleNamespace(config=SimpleNamespace(patch_size=2))
    for role in (vae_graph.VaeEncode, vae_graph.VaeDecode):
        with pytest.raises(ValueError, match="patch_size=None"):
            role(source)


def test_frame_attention_matches_torch_sdpa(vae_graph):
    torch.manual_seed(8)
    source = nn.Module()
    source.dim = 8
    source.norm = nn.Module()
    source.norm.gamma = nn.Parameter(torch.rand(8, 1, 1))
    source.norm.bias = 0.0
    source.norm.scale = 8**0.5
    source.norm.channel_first = True
    source.to_qkv = nn.Conv2d(8, 24, 1)
    source.proj = nn.Conv2d(8, 8, 1)
    value = torch.randn(1, 8, 1, 3, 5)
    normalized = torch.nn.functional.normalize(value[:, :, 0], dim=1) * source.norm.scale * source.norm.gamma
    q, k, v = source.to_qkv(normalized).reshape(1, 1, 24, 15).transpose(-1, -2).chunk(3, -1)
    attended = torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).reshape(1, 8, 3, 5)
    expected = value + source.proj(attended).unsqueeze(2)
    module = vae_graph.FrameAttention(source)
    torch.testing.assert_close(module(value), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.fx.symbolic_trace(module)(value), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kernel,pad,stride", [(3, 1, 2), (1, 0, 1)])
def test_frame_conv_preserves_explicit_padding(vae_graph, kernel, pad, stride):
    source = nn.Conv2d(3, 5, kernel, stride=stride)
    source._padding = (pad, pad, pad, pad)
    value = torch.randn(1, 3, 1, 7, 9)
    expected = source(torch.nn.functional.pad(value[:, :, 0], source._padding)).unsqueeze(2)
    module = vae_graph.FrameConv2d(source)
    torch.testing.assert_close(module(value), expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="temporal cache"):
        module(value, cache_x=value)
