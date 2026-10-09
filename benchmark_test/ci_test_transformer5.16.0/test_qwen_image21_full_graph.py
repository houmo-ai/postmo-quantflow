"""Complete DiT, immutable prefix and request-local target replacement contracts."""

import torch
from test_qwen_image21_graph import source_block
from torch import nn

from xhmodel_merak.xh_other_model.models.qwen_image21.full_graph import CachedGraph, ExtractGraph, FullDenoiser
from xhmodel_merak.xh_other_model.models.qwen_image21.full_runtime import FullDenoiseRequest
from xhmodel_merak.xh_other_model.models.qwen_image21.graph import (
    BlockCore,
    CachedStack,
    ExtractStack,
    ImageProjection,
    OutputHead,
    TimeConditioning,
)
from xhmodel_merak.xh_other_model.models.qwen_image21.layout import prepare_layout
from xhmodel_merak.xh_other_model.models.qwen_image21.runtime import DenoiseRequest


def source():
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([source_block(), source_block()])
    model.img_in = nn.Linear(4, 128)
    model.time_text_embed = nn.Module()
    model.time_text_embed.timestep_embedder = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    model.modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, 512))
    model.norm_out = nn.Module()
    model.norm_out.linear = nn.Linear(128, 128)
    model.norm_out.norm = nn.LayerNorm(128, elementwise_affine=False)
    model.proj_out = nn.Linear(128, 4)
    return model.eval()


def test_complete_graph_matches_layerwise_and_replaces_targets_without_request_leaks():
    torch.manual_seed(13)
    model, text = source(), nn.Linear(96, 128)
    layout = prepare_layout(torch.zeros(1, 3, dtype=torch.bool), [(1, 2, 2)], prefix_capacity=8)
    rope = torch.polar(torch.ones(7, 32), torch.randn(7, 32))
    features, refs = torch.randn(1, 3, 96), torch.empty(1, 0, 4)
    extract = ExtractGraph(FullDenoiser(model, text, 8, 4))
    cached = CachedGraph(FullDenoiser(model, text, 8, 4, cached=True))

    def request(value):
        return FullDenoiseRequest(
            layout, rope, value, refs, extract=extract, cached=cached, layers=2, heads=2, head_dim=64
        )

    full = request(features)
    cores = [BlockCore(block) for block in model.transformer_blocks]
    legacy = DenoiseRequest(
        layout,
        rope,
        features,
        refs,
        text_projection=text,
        image_projection=ImageProjection(model),
        time_conditioning=TimeConditioning(model),
        output_head=OutputHead(model),
        extract=ExtractStack(cores, 8, 4),
        cached=CachedStack(cores),
        layers=2,
        heads=2,
        head_dim=64,
    )
    targets = [torch.randn(1, 4, 4) for _ in range(3)]
    with torch.no_grad():
        first = full(targets[0], 1.0)
        torch.testing.assert_close(first, legacy(targets[0], 1.0), atol=2e-6, rtol=2e-5)
        prefixes = [cache[:, :, :8].clone() for cache in full.caches]
        old_targets = [cache[:, :, 8:].clone() for cache in full.caches]
        pointers = [cache.data_ptr() for cache in full.caches]
        other = request(features + 1)
        other(targets[1], 1.0)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(full.caches, other.caches, strict=True))
        for step in (1, 2):
            output = full(targets[step], 1.0 - step / 3)
            torch.testing.assert_close(output, legacy(targets[step], 1.0 - step / 3), atol=2e-6, rtol=2e-5)
            fresh = request(features)
            torch.testing.assert_close(output, fresh(targets[step], 1.0 - step / 3), atol=2e-6, rtol=2e-5)
            for index, cache in enumerate(full.caches):
                assert cache.data_ptr() == pointers[index]
                torch.testing.assert_close(cache[:, :, :8], prefixes[index], rtol=0, atol=0)
                assert not torch.equal(cache[:, :, 8:], old_targets[index])
                torch.testing.assert_close(cache, fresh.caches[index], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(request(features)(targets[0], 1.0), first, rtol=0, atol=0)


def test_one_graph_pair_supports_actual_lengths_and_masks_capacity_tail():
    torch.manual_seed(31)
    model, text = source(), nn.Linear(96, 128)
    extract = ExtractGraph(FullDenoiser(model, text, 16, 8))
    cached = CachedGraph(FullDenoiser(model, text, 16, 8, cached=True))
    features = torch.randn(1, 3, 96)
    refs = torch.empty(1, 0, 4)

    def request(target_hw):
        layout = prepare_layout(
            torch.zeros(1, 3, dtype=torch.bool), [(1, *target_hw)], prefix_capacity=16, target_capacity=8
        )
        angle = torch.arange((3 + layout.target_tokens) * 32).reshape(-1, 32).float() / 113
        rope = torch.polar(torch.ones_like(angle), angle)
        return FullDenoiseRequest(
            layout, rope, features, refs, extract=extract, cached=cached, layers=2, heads=2, head_dim=64
        )

    small, large = request((2, 2)), request((2, 4))
    target = torch.randn(1, 4, 4)
    with torch.inference_mode():
        small(target, 1.0)
        prefix = [value[:, :, :16].clone() for value in small.caches]
        output = small(target + 0.25, 0.5)
        assert output.shape == target.shape
        assert large(torch.randn(1, 8, 4), 0.75).shape == (1, 8, 4)
        torch.testing.assert_close(small(target + 0.25, 0.5), output, rtol=0, atol=0)
        for value, original in zip(small.caches, prefix, strict=True):
            torch.testing.assert_close(value[:, :, :16], original, rtol=0, atol=0)
            assert torch.count_nonzero(value[:, :, 20:]) == 0
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(small.caches, large.caches, strict=True))
        padded = torch.nn.functional.pad(target + 0.25, (0, 0, 0, 4), value=1000.0)
        from xhmodel_merak.xh_other_model.models.qwen_image21.runtime import timestep_features

        poisoned = cached(
            padded,
            timestep_features(0.5, device="cpu"),
            small.cos[:, :, 16:],
            small.sin[:, :, 16:],
            small.mask[:, 16:],
            small.caches,
        )[0]
        torch.testing.assert_close(poisoned[:, :4], output, atol=2e-6, rtol=2e-5)
