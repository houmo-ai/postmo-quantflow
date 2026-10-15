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
def test_constructed_time_tables_keep_noisy_clean_roles_and_prompt_projections():
    from _sol_h3_tiny_components import refiner_conditioning, refiner_source

    torch.manual_seed(274)
    model = refiner_source()
    cache = {"contexts": {key: torch.randn(1, 16, 128).bfloat16() for key in ("video", "audio")}}
    videos, audios, _, _ = refiner_conditioning(model, cache, size=(32, 32), keyframe=True)
    block = model.transformer_blocks[0]
    folded = fold_ltx_block(block, videos, audios)
    for step, (video, audio) in enumerate(zip(videos, audios, strict=True)):
        for role, args, table in (
            ("video", video, block.scale_shift_table),
            ("audio", audio, block.audio_scale_shift_table),
        ):
            # Expand raw learned tables and time features independently of the
            # block's get_ada_values helper used by the production folder.
            expected = table[None, None] + args.timesteps.reshape(1, -1, 9, 128)
            torch.testing.assert_close(folded[role][step], expected[0].transpose(0, 1), rtol=0, atol=0)
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
