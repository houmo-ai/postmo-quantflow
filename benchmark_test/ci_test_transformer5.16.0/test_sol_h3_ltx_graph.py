# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Self-contained LTX graph contracts, not author-checkpoint equivalence."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
from xhmodel_merak.xh_other_model.models.sol_h3.ltx_graph import (
    LTXAttention,
    LTXDenoiserGraph,
    LTXInputRoPEGraph,
    StepTable,
    fold_ltx_block,
    video_modulation,
)


def test_time_tables_reject_nonfinite_before_fp16_narrowing():
    with pytest.raises(FloatingPointError):
        StepTable(torch.tensor([70000.0]))
    with pytest.raises(FloatingPointError):
        StepTable(torch.tensor([float("nan")]))


def test_video_time_roles_match_expanded_per_token_modulation():
    torch.manual_seed(640)
    x = torch.randn(1, 7, 8)
    params = torch.randn(3, 2, 8)
    roles = torch.tensor([1, 1, 0, 0, 0, 0, 1])
    ranges = ((0, 2, 1), (2, 6, 0), (6, 7, 1))
    expanded = params[:, roles]
    expected = x * (1 + expanded[1]) + expanded[0]
    torch.testing.assert_close(video_modulation(x, params, ranges, shift=0, scale=1), expected, rtol=0, atol=0)
    torch.testing.assert_close(video_modulation(x, params, ranges, gate=2), x * expanded[2], rtol=0, atol=0)


def test_static_solver_keeps_audio_clean_across_three_steps():
    from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import REFINER_SIGMAS
    from xhmodel_merak.xh_other_model.models.sol_h3.ltx_keyframes import (
        prepare_static_keyframe_state,
        static_keyframe_step,
    )

    generator = torch.Generator().manual_seed(1001)
    video = torch.randn(1, 128, 16, 1, 2, generator=generator)
    audio = torch.randn(1, 8, 126, 16, generator=generator)
    endpoints = [(index, torch.randn(1, 128, 1, 1, 2, generator=generator)) for index in (0, 120)]
    options = dict(seed=42, size=(32, 64), refiner_frames=121, keyframe_latents=endpoints)
    legacy = prepare_static_keyframe_state(video, audio, **options)
    frozen = prepare_static_keyframe_state(video, audio, freeze_audio=True, **options)
    clean = audio.permute(0, 2, 1, 3).reshape(1, 126, 128)
    assert torch.equal(frozen.video, legacy.video) and torch.equal(frozen.audio, clean)
    assert not torch.equal(legacy.audio, clean)
    for sigma, next_sigma in zip(REFINER_SIGMAS[:-1], REFINER_SIGMAS[1:], strict=True):
        expected = frozen.video[:, 2:-2] + (next_sigma - sigma)
        frozen = static_keyframe_step(
            frozen, torch.ones_like(frozen.video), torch.full_like(frozen.audio, 100), sigma, next_sigma
        )
        torch.testing.assert_close(frozen.video[:, 2:-2], expected, rtol=1e-6, atol=1e-6)
        assert torch.equal(frozen.audio, clean)


class _LargeValueCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        source = SimpleNamespace(
            heads=2,
            dim_head=64,
            to_q=nn.Linear(128, 128),
            to_k=nn.Linear(128, 128),
            to_v=nn.Linear(128, 128),
            to_out=nn.ModuleList([nn.Linear(128, 128)]),
            q_norm=nn.RMSNorm(128, eps=1e-6),
            k_norm=nn.RMSNorm(128, eps=1e-6),
            to_gate_logits=None,
        )
        with torch.no_grad():
            for layer in (source.to_q, source.to_k):
                layer.weight.zero_()
                layer.bias.zero_()
            source.to_v.weight.copy_(torch.eye(128))
            source.to_v.bias.fill_(20)
            source.to_out[0].weight.copy_(torch.eye(128))
            source.to_out[0].bias.zero_()
        self.core = LTXAttention(source, sdk_ops=True)
        self.register_buffer("step", torch.zeros(1, dtype=torch.int32))

    def forward(self, query, context):
        return self.core(query, context, self.step, None, None, None, None)


@torch.no_grad()
def test_cross_attention_sefp8_keeps_large_value_mean_finite(tmp_path):
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(183)
    inputs = (torch.randn(1, 126, 128) * 0.01, torch.randn(1, 16128, 128) * 0.01)
    path = tmp_path / "attention.hmonnx"
    export_graph(
        _LargeValueCrossAttention(),
        inputs,
        path,
        ["query", "context"],
        ["output"],
        quant_scheme={"target_device": "XH2a", "quant_type": "w8a8_sefp"},
    )
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    outputs = session.run(
        {name: value.cuda().half() for name, value in zip(session.get_input_names(), inputs, strict=True)}
    )
    actual = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    # Zero Q/K produces uniform attention. The unnormalized numerator is
    # ~322560 and exceeds FP16; the correct normalized output is ~20.
    expected = (inputs[1] + 20).mean(1, keepdim=True).expand(1, 126, 128).cuda()
    assert bool(torch.isfinite(actual).all())
    assert float(actual.abs().max()) < 21
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.05)


@torch.no_grad()
@pytest.mark.parametrize("freeze_audio", [False, True])
def test_constructed_time_tables_keep_noisy_clean_roles_and_prompt_projections(freeze_audio):
    from _sol_h3_tiny_components import refiner_conditioning, refiner_source

    torch.manual_seed(274)
    model = refiner_source()
    cache = {"contexts": {key: torch.randn(1, 16, 128).bfloat16() for key in ("video", "audio")}}
    videos, audios, _, _ = refiner_conditioning(
        model, cache, size=(32, 32), keyframe=True, freeze_audio=freeze_audio
    )
    block = model.transformer_blocks[0]
    folded = fold_ltx_block(block, videos, audios)
    for step, (video, audio) in enumerate(zip(videos, audios, strict=True)):
        if freeze_audio:
            zero_time = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float16)
            torch.testing.assert_close(audio.timesteps, model.time_projection(zero_time)[None], rtol=0, atol=0)
            torch.testing.assert_close(
                video.cross_gate_timestep, model.gate_projection(zero_time)[None], rtol=0, atol=0
            )
            assert not torch.equal(video.cross_gate_timestep, audio.cross_gate_timestep)
        for role, args, table, cross_table in (
            ("video", video, block.scale_shift_table, block.scale_shift_table_a2v_ca_video),
            ("audio", audio, block.audio_scale_shift_table, block.scale_shift_table_a2v_ca_audio),
        ):
            # Expand raw learned tables and time features independently of the
            # block's get_ada_values helper used by the production folder.
            expected = table[None, None] + args.timesteps.reshape(1, -1, 9, 128)
            torch.testing.assert_close(folded[role][step], expected[0].transpose(0, 1), rtol=0, atol=0)
            affine = cross_table[:4][None, None] + args.cross_scale_shift_timestep.reshape(1, -1, 4, 128)
            gate = cross_table[4:][None, None] + args.cross_gate_timestep.reshape(1, 1, 1, 128)
            expected_cross = torch.cat((affine, gate.expand(-1, affine.shape[1], -1, -1)), dim=2)
            torch.testing.assert_close(folded["av_" + role][step], expected_cross[0].transpose(0, 1), rtol=0, atol=0)
            attention = block.attn2 if role == "video" else block.audio_attn2
            prompt = block.prompt_scale_shift_table if role == "video" else block.audio_prompt_scale_shift_table
            affine = prompt[None, None] + args.prompt_timestep.reshape(1, 1, 2, 128)
            context = args.context * (1 + affine[:, :, 1]) + affine[:, :, 0]
            key = torch.nn.functional.linear(context, attention.to_k.weight, attention.to_k.bias)
            wide = key.float()
            key = (
                wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-6) * attention.k_norm.weight.float()
            ).half()
            value = torch.nn.functional.linear(context, attention.to_v.weight, attention.to_v.bias)
            key = key.reshape(1, -1, 2, 64).transpose(1, 2)
            value = value.reshape(1, -1, 2, 64).transpose(1, 2)
            torch.testing.assert_close(folded[role + "_k"][step], key, rtol=0.002, atol=0.002)
            torch.testing.assert_close(folded[role + "_v"][step], value, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("static_guide", [False, True])
def test_dynamic_audio_mode_export_matches_single_policy_graphs(tmp_path, static_guide):
    import onnx
    from _sol_h3_tiny_components import refiner_conditioning, refiner_source

    from xhquant.nn import RMSNorm
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(274)
    source = refiner_source()
    # Match the normalization adapter used by the full Sol-H3 CI fixture.
    for name in ("norm_out", "audio_norm_out"):
        native = getattr(source, name)
        adapted = RMSNorm(native.normalized_shape[-1], native.eps)
        adapted.weight = nn.Parameter(native.weight.detach().clone(), requires_grad=False)
        source.set_submodule(name, adapted)
    cache = {"contexts": {key: torch.randn(1, 16, 128).bfloat16() for key in ("video", "audio")}}
    frozen_video, frozen_audio, rope, mask = refiner_conditioning(
        source, cache, size=(32, 32), keyframe=static_guide, freeze_audio=True
    )
    joint_video, joint_audio, _, _ = refiner_conditioning(
        source, cache, size=(32, 32), keyframe=static_guide, freeze_audio=False
    )
    block = source.transformer_blocks[0]
    frozen = fold_ltx_block(block, frozen_video, frozen_audio)
    joint = fold_ltx_block(block, joint_video, joint_audio)
    both = fold_ltx_block(
        block, frozen_video, frozen_audio, joint_video_args=joint_video, joint_audio_args=joint_audio
    )
    for name in ("video", "video_k", "video_v"):
        torch.testing.assert_close(both[name], frozen[name], rtol=0, atol=0)
        torch.testing.assert_close(joint[name], frozen[name], rtol=0, atol=0)
    for name in ("audio", "av_video", "av_audio", "audio_k", "audio_v"):
        torch.testing.assert_close(both[name][:, 0], joint[name], rtol=0, atol=0)
        torch.testing.assert_close(both[name][:, 1], frozen[name], rtol=0, atol=0)
    # A/V gates depend on the other modality's global time, not their own.
    assert not torch.equal(joint["av_video"][:, 4], frozen["av_video"][:, 4])
    assert torch.equal(joint["av_audio"][:, 4], frozen["av_audio"][:, 4])

    options = dict(keyframes_mask=mask, first_frame_tokens=1 if static_guide else None)
    references = [
        LTXDenoiserGraph(source, [table], video_args, audio_args, **options).float().eval()
        for table, video_args, audio_args in ((joint, joint_video, joint_audio), (frozen, frozen_video, frozen_audio))
    ]
    graph = LTXDenoiserGraph(
        source, [both], frozen_video, frozen_audio, joint_audio_args=joint_audio, sdk_ops=True, **options
    ).float().eval()
    eager = LTXDenoiserGraph(
        source, [both], frozen_video, frozen_audio, joint_audio_args=joint_audio, **options
    ).float().eval()
    if not static_guide:
        graph, eager = LTXInputRoPEGraph(graph), LTXInputRoPEGraph(eager)
        references = [LTXInputRoPEGraph(reference) for reference in references]
    names = ["video", "audio", "step", "vcos", "vsin", "acos", "asin", "vccos", "vcsin", "accos", "acsin"]
    inputs = [
        torch.randn(1, frozen_video[0].x.shape[1], 128),
        torch.randn(1, frozen_audio[0].x.shape[1], 128),
        torch.tensor([1], dtype=torch.int32),
        *(value.float() for value in rope),
    ]
    if static_guide:
        names += ["first_role", "video_valid_length"]
        inputs += [torch.tensor([1], dtype=torch.int32), torch.tensor([17], dtype=torch.int32)]
    mode = torch.tensor([1], dtype=torch.int32)
    expected_by_step = []
    for step in range(3):
        inputs[2] = torch.tensor([step], dtype=torch.int32)
        expected = [reference(*inputs) for reference in references]
        for mode_value in (0, 1):
            for actual, oracle in zip(
                eager(*inputs, torch.tensor([mode_value], dtype=torch.int32)), expected[mode_value], strict=True
            ):
                torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
        for actual, oracle in zip(eager(*inputs), expected[1], strict=True):
            torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
        expected_by_step.append(expected)
    inputs[2] = torch.tensor([1], dtype=torch.int32)

    names += ["audio_mode"]
    path = tmp_path / "ltx.hmonnx"
    export_graph(
        graph,
        (*inputs, mode),
        path,
        names,
        ["video_velocity", "audio_velocity"],
        quant_scheme={"target_device": "XH2a", "quant_type": "w8a8_sefp"},
    )
    onnx_graph = onnx.load(path, load_external_data=False).graph
    assert [value.name for value in onnx_graph.input] == names
    mode_input = onnx_graph.input[-1].type.tensor_type
    assert mode_input.elem_type == onnx.TensorProto.INT32
    assert [dim.dim_value for dim in mode_input.shape.dim] == [1]
    consumers = [node for node in onnx_graph.node if "audio_mode" in node.input]
    assert consumers and all(node.op_type == "Gather" for node in consumers)

    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    for step, expected in enumerate(expected_by_step):
        inputs[2] = torch.tensor([step], dtype=torch.int32)
        outputs = []
        for mode_value in (0, 1, 0):
            values = (*inputs, torch.tensor([mode_value], dtype=torch.int32))
            feed = {
                name: value.to(device="cuda:0", dtype=session.get_input(name).dtype)
                for name, value in zip(names, values, strict=True)
            }
            result = session.run(feed)
            result = result if isinstance(result, (tuple, list)) else (result,)
            current = tuple(value.float().cpu().clone() for value in result)
            for actual, oracle, other in zip(current, expected[mode_value], expected[1 - mode_value], strict=True):
                error = (actual - oracle).norm()
                assert torch.isfinite(actual).all()
                assert error < 0.08 * oracle.norm()
                assert error < (actual - other).norm()
            outputs.append(current)
        for first, repeated in zip(outputs[0], outputs[2], strict=True):
            torch.testing.assert_close(first, repeated, rtol=0, atol=0)
