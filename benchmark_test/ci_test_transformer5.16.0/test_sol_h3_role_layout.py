# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed AdaLN slots: native semantics, explicit holes and static broadcasts."""

from copy import deepcopy

import pytest
import torch
from test_sol_h3_static_graph import text_key_mask

from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import (
    ConditionedH3Schedule,
    conditioned_tables,
    fold_h3_conditioning,
)
from xhmodel_merak.xh_other_model.models.sol_h3.h3_graph import H3TextGraph, RolePartitionedH3DenoiserGraph
from xhmodel_merak.xh_other_model.models.sol_h3.precision import H3PrecisionPolicy
from xhmodel_merak.xh_other_model.models.sol_h3.role_layout import (
    make_role_layout,
    role_capacities_from_plan,
    role_spans,
)


def mixed_layout(*, capacities=(64, 64, 64, 64, 64), block_size=64):
    roles = torch.tensor([3, 3, 0, 0, 0, 0, 4, 4, 2, 2, 1, 0, 1], dtype=torch.int32)
    positions = torch.arange(39).reshape(13, 3).float() / 40
    return make_role_layout(
        (6, 4, 3), (8, 6, 6), roles, positions, capacities, attention_heads=2, block_size=block_size
    )


def test_role_slots_preserve_native_coordinates_and_head_origins():
    layout = mixed_layout()
    assert layout.counts == (5, 2, 2, 2, 2)
    assert layout.pack.shape == (320,) and layout.head_unpack.shape == (14,)
    assert int(layout.valid_mask.sum()) == 13
    # Role zero is target video first, then the Qwen vision-context row.
    assert layout.pack[:5].tolist() == [2, 3, 4, 5, 15]
    assert layout.pack[64:66].tolist() == [14, 16]
    assert layout.pack[128:130].tolist() == [10, 11]
    assert layout.pack[192:194].tolist() == [0, 1]
    assert layout.pack[256:258].tolist() == [8, 9]
    assert layout.head_unpack.tolist() == [192, 193, 0, 1, 2, 3, 320, 320, 256, 257, 128, 129, 320, 320]
    positions = torch.arange(39).reshape(13, 3).float() / 40
    assert torch.equal(layout.positions[:4], positions[2:6])
    assert torch.equal(layout.positions[4], positions[11])
    assert torch.equal(layout.positions[192:194], positions[:2])
    assert bool((layout.positions[~layout.valid_mask[0]] == 0).all())
    assert bool((layout.pack[~layout.valid_mask[0]] == 20).all())  # Appended zero row, not a live token.


def test_key_mask_selects_all_and_only_live_rows_across_role_holes():
    layout = mixed_layout(capacities=(128,) * 5)
    assert layout.block_sizes.tolist() == [[5, 0, 2, 0, 2, 0, 2, 0, 2, 0]]
    assert layout.key_mask.shape == (1, 1, 1, 640)
    assert layout.key_mask.dtype == torch.float16
    assert torch.equal(layout.key_mask[0, 0, 0] == 0, layout.valid_mask[0])
    assert (layout.key_mask[0, 0, 0, ~layout.valid_mask[0]] == -65504).all()


def test_role_capacity_plan_covers_five_modes_without_using_calibration_lengths():
    common = dict(
        prompt_max_tokens=1024,
        token_alignment=256,
        target_video_tokens=9324,
        target_audio_tokens=414,
        reference_audio_tokens=0,
    )
    plans = [
        dict(common, text_tokens=n, presentation_overhead=overhead, reference_video_tokens=references)
        for n, overhead, references in ((1024, 0, 0), (1284, 8, 252), (1284, 8, 252), (1544, 16, 504))
    ]
    base = tuple(max(capacities[role] for capacities in map(role_capacities_from_plan, plans)) for role in range(5))
    assert base == (9984, 1280, 512, 512, 0) and sum(base) == 12288
    reference = dict(
        common, text_tokens=18822, presentation_overhead=64, reference_video_tokens=25024, reference_audio_tokens=800
    )
    assert role_capacities_from_plan(reference) == (27136, 1280, 512, 25088, 1024)
    assert sum(role_capacities_from_plan(reference)) == 55040


@pytest.mark.parametrize("step", range(4))
@torch.no_grad()
def test_fixed_roles_match_native_h3_with_qwen_vision_and_both_reference_types(step):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from test_sol_h3_static_graph import tiny_source

    source = tiny_source()
    schedule = ConditionedH3Schedule()
    folded = fold_h3_conditioning(source, schedule)
    tables, output = conditioned_tables(folded, schedule)
    video, audio, text = torch.randn(1, 6, 8), torch.randn(1, 4, 4), torch.randn(1, 3, 24)
    layout = mixed_layout()
    tags = torch.tensor([0] * 6 + [2] * 4 + [1, 0, 1])
    positions = torch.arange(39).reshape(13, 3).float() / 40
    vs, aus = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    vs.set_timesteps(sigmas=schedule.video_sigmas)
    aus.set_timesteps(sigmas=schedule.audio_sigmas)
    times, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        torch.arange(6),
        torch.arange(6, 10),
        2,
        2,
        3,
        float(vs.timesteps[step]),
        float(aus.timesteps[step]),
        schedule.condition_video_time,
        1.0,
    )
    expected = source(
        video,
        audio,
        text,
        times,
        indices,
        tags,
        positions,
        torch.arange(6),
        torch.arange(6, 10),
        torch.arange(10, 13),
        return_dict=False,
    )
    refined = H3TextGraph(deepcopy(source))(torch.nn.functional.pad(text, (0, 0, 0, 3)), text_key_mask(6, 3))
    graph = RolePartitionedH3DenoiserGraph(
        deepcopy(source),
        tables,
        output,
        8,
        6,
        H3PrecisionPolicy(rms_norm_mode="stable"),
        role_capacities=layout.capacities,
        attention_block_size=64,
    )
    cos, sin = source.rope(layout.positions)
    inputs = (
        torch.nn.functional.pad(video, (0, 0, 0, 2)),
        torch.nn.functional.pad(audio, (0, 0, 0, 2)),
        refined,
        torch.tensor([step]),
        cos[None, None],
        sin[None, None],
        layout.pack,
        layout.head_unpack,
        layout.key_mask,
    )
    actual = graph(*inputs)
    for value, reference, length in zip(actual, expected, (6, 4), strict=True):
        torch.testing.assert_close(value[:, :length], reference, rtol=3e-5, atol=3e-6)
    # Source padding never enters a role's live rows; poison every unused source
    # slot to ensure a mask hole cannot silently become an attention key.
    changed = [value.clone() for value in inputs]
    changed[0][:, 6:] = 100
    changed[1][:, 4:] = -100
    changed[2][:, 3:] = 100
    for value, reference in zip(graph(*changed), actual, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert all(block.role_spans == role_spans(layout.capacities) for block in graph.blocks)
    assert not any("role_gather" in name for name, _ in graph.named_modules())


def test_host_packet_has_fixed_roles_and_block_validity_without_runtime_role_gathers():
    from xhmodel_merak.xh_other_model.models.sol_h3.host import RolePartitionedDraft

    draft = RolePartitionedDraft.from_text(
        seed=42,
        size=(32, 32),
        text_length=3,
        capacities=dict(video_capacity=64, audio_capacity=512, text_capacity=64),
        rope_config=dict(rope_freq_dim=2, rope_theta=10000),
        role_capacities=(64, 64, 512, 0, 0),
        attention_heads=2,
        attention_block_size=64,
    )
    packet = draft.inputs(torch.zeros(1, 3, 32))
    assert len(packet) == 9
    assert packet[6] is draft.layout.pack and packet[7] is draft.layout.head_unpack
    assert packet[8] is draft.key_mask
    assert draft.layout.counts == (37, 3, 414, 0, 0)
    assert draft.layout.valid_mask.shape == (1, 640)
    assert int(draft.layout.valid_mask.sum()) == 454
    assert draft.layout.metadata()["attention_mode"] == "broadcast_key_mask"


@torch.no_grad()
def test_t2va_native_oracle_with_zero_reference_capacities():
    from test_sol_h3_static_graph import tiny_source

    source, schedule = tiny_source(), ConditionedH3Schedule()
    folded = fold_h3_conditioning(source, schedule)
    tables, output = conditioned_tables(folded, schedule)
    video, audio, text = torch.randn(1, 6, 8), torch.randn(1, 4, 4), torch.randn(1, 3, 24)
    tags = torch.tensor([0] * 6 + [2] * 4 + [1] * 3)
    positions = torch.arange(39).reshape(13, 3).float() / 40
    indices, _ = schedule.indices(3, tags)
    expected = source(
        video,
        audio,
        text,
        schedule.row_times()[3],
        indices.long(),
        tags,
        positions,
        torch.arange(6),
        torch.arange(6, 10),
        torch.arange(10, 13),
        return_dict=False,
    )
    layout = make_role_layout(
        (6, 4, 3),
        (8, 6, 6),
        tags.int(),
        positions,
        (64, 64, 64, 0, 0),
        attention_heads=2,
        block_size=64,
    )
    graph = RolePartitionedH3DenoiserGraph(
        source,
        tables,
        output,
        8,
        6,
        H3PrecisionPolicy(rms_norm_mode="stable"),
        role_capacities=layout.capacities,
        attention_block_size=64,
    )
    refined = H3TextGraph(deepcopy(source))(torch.nn.functional.pad(text, (0, 0, 0, 3)), text_key_mask(6, 3))
    cos, sin = source.rope(layout.positions)
    actual = graph(
        torch.nn.functional.pad(video, (0, 0, 0, 2)),
        torch.nn.functional.pad(audio, (0, 0, 0, 2)),
        refined,
        torch.tensor([3]),
        cos[None, None],
        sin[None, None],
        layout.pack,
        layout.head_unpack,
        layout.key_mask,
    )
    for value, reference, length in zip(actual, expected, (6, 4), strict=True):
        torch.testing.assert_close(value[:, :length], reference, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("bad", [(64, 64, -1, 64, 64)])
def test_bad_role_capacities_fail(bad):
    with pytest.raises(ValueError, match="capacities"):
        mixed_layout(capacities=bad)


def test_role_overflow_and_unaligned_ranges_fail_closed():
    with pytest.raises(ValueError, match="exceed"):
        mixed_layout(capacities=(64, 64, 64, 64, 0))
    with pytest.raises(ValueError, match="align"):
        mixed_layout(capacities=(65, 64, 64, 64, 64))


@torch.no_grad()
def test_fixed_roles_export_only_boundary_gathers_and_execute_masked_hmonnx(tmp_path):
    """Exercise the real frontend, ONNX ABI and quantized CUDA interpreter."""
    import onnx
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(314)
    source = (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=1,
            attention_head_dim=64,
            hidden_size=64,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=128,
            in_channels=2,
            audio_in_channels=4,
            patch_size=(1, 2, 2),
            text_dim=24,
            freq_dim=8,
            time_embed_hidden_dim=64,
            time_embed_dim=32,
            rope_freq_dim=2,
            norm_eps=0.1,
            final_norm_eps=0.1,
        )
        .eval()
        .requires_grad_(False)
    )
    layout = make_role_layout(
        (6, 4, 3),
        (8, 6, 6),
        torch.tensor([3, 3, 0, 0, 0, 0, 4, 4, 2, 2, 1, 0, 1], dtype=torch.int32),
        torch.arange(39).reshape(13, 3).float() / 40,
        (128,) * 5,
        attention_heads=1,
        block_size=64,
    )
    folded = fold_h3_conditioning(source, ConditionedH3Schedule())
    tables, output = conditioned_tables(folded)
    policy = H3PrecisionPolicy(rms_norm_mode="stable")
    graph = RolePartitionedH3DenoiserGraph(
        source,
        tables,
        output,
        8,
        6,
        policy,
        role_capacities=layout.capacities,
        attention_block_size=64,
        sdk_ops=True,
    )
    refined = H3TextGraph(deepcopy(source))(torch.randn(1, 6, 24), text_key_mask(6, 3))
    cos, sin = source.rope(layout.positions)
    inputs = (
        torch.randn(1, 8, 8),
        torch.randn(1, 6, 4),
        refined,
        torch.tensor([3], dtype=torch.int32),
        cos[None, None],
        sin[None, None],
        layout.pack,
        layout.head_unpack,
        layout.key_mask,
    )
    names = [
        "video_tokens",
        "audio_tokens",
        "text_features",
        "step",
        "cos",
        "sin",
        "role_pack",
        "head_unpack",
        "key_mask",
    ]
    path = tmp_path / "h3_roles.hmonnx"
    export_graph(
        graph, inputs, path, names, ["video", "audio"], quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp")
    )
    model = onnx.load(path, load_external_data=False)
    gathers = [node for node in model.graph.node if node.op_type == "Gather"]
    # One per-block time selector, one output time selector, and the two
    # boundary permutations. No token-shaped AdaLN vector is materialized.
    assert len(gathers) == 4
    attentions = [node for node in model.graph.node if node.op_type == "FlashAttention"]
    assert len(attentions) == 1 and len(attentions[0].input) == 5
    assert attentions[0].input[4] == "key_mask"
    assert not any(node.op_type.startswith("SparseFlash") for node in model.graph.node)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))

    def execute(packet):
        feed = {
            name: value.cuda().half() if value.is_floating_point() else value.cuda()
            for name, value in zip(names, packet, strict=True)
        }
        return session.run(feed)

    actual = execute(inputs)
    assert all(bool(torch.isfinite(value).all()) for value in actual)
    poison = [value.clone() for value in inputs]
    poison[0][:, 6:] = 100
    poison[1][:, 4:] = -100
    poison[2][:, 3:] = 100
    for changed, original in zip(execute(poison), actual, strict=True):
        torch.testing.assert_close(changed, original, rtol=0, atol=0)
