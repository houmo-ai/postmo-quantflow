# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real H3 blocks: hybrid folding, padded attention and compensated FP16 range."""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import H3Schedule, fold_h3_conditioning, modality_tables
from xhmodel_merak.xh_other_model.models.sol_h3.h3_graph import H3DenoiserGraph, H3TextGraph
from xhmodel_merak.xh_other_model.models.sol_h3.precision import H3PrecisionPolicy


def tiny_source():
    torch.manual_seed(314)
    return (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=2,
            ffn_dim=64,
            in_channels=2,
            audio_in_channels=4,
            patch_size=(1, 2, 2),
            text_dim=24,
            freq_dim=8,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=2,
            norm_eps=0.1,
            final_norm_eps=0.1,
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("step", range(4))
@torch.no_grad()
def test_four_folded_steps_match_native_with_padding_and_no_projection(step):
    source = tiny_source()
    folded = fold_h3_conditioning(source)
    tables, output = modality_tables(folded)
    v, a, text = torch.randn(1, 6, 8), torch.randn(1, 4, 4), torch.randn(1, 3, 24)
    positions = torch.randn(13, 3)
    tags = torch.tensor([0] * 6 + [2] * 4 + [1] * 3)
    timeidx, _ = H3Schedule().indices(step, tags)
    expected = source(
        v,
        a,
        text,
        folded.row_times[step],
        timeidx.long(),
        tags,
        positions,
        torch.arange(6),
        torch.arange(6, 10),
        torch.arange(10, 13),
        return_dict=False,
    )
    text_graph = H3TextGraph(source)
    refined = text_graph(torch.cat((text, torch.zeros(1, 2, 24)), 1), torch.tensor([3]))
    graph = H3DenoiserGraph(source, tables, output, 6, 4, H3PrecisionPolicy(rms_norm_mode="stable"))
    cos, sin = source.rope(torch.cat((positions, torch.zeros(2, 3))))
    actual = graph(v, a, refined, torch.tensor([step]), cos[None, None], sin[None, None], torch.tensor([13]))
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-6)
    assert not any("adaln_proj" in name or "time_embedder" in name for name, _ in graph.named_parameters())
    assert all(block.modulation.table.shape == (4, 6, 3, 32) for block in graph.blocks)
    assert sum(p.numel() for p in graph.parameters()) + sum(p.numel() for p in text_graph.parameters()) == (
        sum(p.numel() for p in source.parameters())
        - folded.original_projection_parameters
        - sum(p.numel() for p in source.time_embedder.parameters())
    )


@torch.no_grad()
def test_fp16_graph_survives_large_text_projection_and_residual_without_clips():
    source = tiny_source()
    source.context_embedder.weight.fill_(0.5)
    for block in source.transformer_blocks:
        block.ff.net[2].weight.mul_(5000)
    folded = fold_h3_conditioning(source)
    tables, output = modality_tables(folded)
    text_graph = H3TextGraph(source).half()
    graph = H3DenoiserGraph(source, tables, output, 6, 4).half()
    refined = text_graph(torch.full((1, 3, 24), 10000.0, dtype=torch.float16), torch.tensor([3]))
    cos, sin = source.rope(torch.zeros(13, 3))
    actual = graph(
        torch.ones(1, 6, 8, dtype=torch.float16),
        torch.ones(1, 4, 4, dtype=torch.float16),
        refined,
        torch.tensor([0]),
        cos[None, None].half(),
        sin[None, None].half(),
        torch.tensor([13]),
    )
    assert all(bool(torch.isfinite(t).all()) for t in (refined, *actual))


@pytest.mark.parametrize("value", [3])
def test_invalid_static_scale_is_rejected(value):
    with pytest.raises(ValueError, match="power of two"):
        H3PrecisionPolicy(residual_scale=value)


@pytest.mark.parametrize("anchors", [("first", "last")])
def test_keyframe_layout_roles_and_positions(anchors):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    from xhmodel_merak.xh_other_model.models.sol_h3.host import conditioned_h3_layout

    text_tags = torch.tensor([1, 0, 1])
    positions, roles, cv, ca = conditioned_h3_layout(text_tags, size=(32, 64), anchors=anchors)
    native = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags,
        37,
        2,
        4,
        207,
        (1, 2, 2),
        2,
        2,
        0,
        keyframe_anchors=anchors,
    )
    p, _, vi, ai, ti, ncv, nca = native
    torch.testing.assert_close(positions, p[torch.cat((vi, ai, ti))].float())
    assert cv == ncv == len(anchors) * 2 and ca == nca == 0
    assert roles[:cv].tolist() == [3] * cv
    assert roles[-3:].tolist() == [1, 0, 1]


def test_conditioned_solver_preserves_references_at_every_step():
    from xhmodel_merak.xh_other_model.models.sol_h3.host import conditioned_h3_updates

    video, audio = torch.randn(1, 6, 8), torch.randn(1, 4, 4)
    seen = []

    def invoke(v, a, *args):
        torch.testing.assert_close(v[:, :2], video[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(a[:, :1], audio[:, :1], rtol=0, atol=0)
        seen.append(args[1].item())
        return torch.ones_like(v), torch.ones_like(a)

    result = conditioned_h3_updates(
        invoke,
        video,
        audio,
        torch.zeros(1, 3, 24),
        None,
        None,
        3,
        torch.zeros(13, dtype=torch.int32),
        condition_video_rows=2,
        condition_audio_rows=1,
    )
    assert seen == [0, 1, 2, 3]
    torch.testing.assert_close(result[0], video[:, 2:] + 1)
    torch.testing.assert_close(result[1], audio[:, 1:] + 1)
