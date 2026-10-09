# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Small real tensor blocks: cache lifecycle, shared modulation, and graph traceability."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture
def graph():
    path = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21/graph.py"
    spec = importlib.util.spec_from_file_location("qwen_image21_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_block():
    # Real modules with the native weight interface; no SDK/import side effects or mocked arithmetic.
    model = nn.Module()
    model.img_norm1 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    model.img_norm2 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    model.attn = nn.Module()
    model.attn.heads = 2
    for name in ("to_q", "to_k", "to_v"):
        setattr(model.attn, name, nn.Linear(128, 128, bias=False))
    model.attn.to_out = nn.ModuleList([nn.Linear(128, 128, bias=False)])
    model.attn.norm_q = nn.RMSNorm(64, eps=1e-6)
    model.attn.norm_k = nn.RMSNorm(64, eps=1e-6)
    model.img_mlp = nn.Module()
    model.img_mlp.gate_layer = nn.Linear(128, 384, bias=False)
    model.img_mlp.proj = nn.Linear(128, 384, bias=False)
    model.img_mlp.out = nn.Linear(384, 128, bias=False)
    return model.eval()


def test_two_layer_cache_uses_frozen_prefix_across_steps(graph):
    torch.manual_seed(17)
    cores = [graph.BlockCore(source_block()) for _ in range(2)]
    extract = [graph.ExtractBlock(core, 8) for core in cores]
    cached = [graph.CachedBlock(core) for core in cores]
    modulation = graph.SharedModulation(nn.Sequential(nn.SiLU(), nn.Linear(128, 512, bias=False)))
    zero = torch.randn(1, 128)
    prefix = torch.randn(1, 8, 128)
    angles = torch.randn(16, 32)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    # Causal text, separate adjacent image blocks, then a bidirectional target.
    ids = torch.tensor([-1] * 4 + [0] * 2 + [1] * 2 + [2] * 8)
    pos = torch.arange(16)
    allowed = (pos[:, None] >= pos[None]) | ((ids[:, None] == ids[None]) & (ids[:, None] >= 0))
    mask = torch.where(allowed, 0.0, -65504.0)[None, None]
    saved = []
    with torch.inference_mode():
        for step in range(3):
            target = torch.randn(1, 8, 128)
            values = modulation(torch.cat((zero, torch.randn(1, 128))))
            full_params = [torch.cat((p[:1, None].expand(1, 8, 128), p[1:, None].expand(1, 8, 128)), 1) for p in values]
            full = torch.cat((prefix, target), 1)
            decoded = target
            for layer in range(2):
                full, key, value = extract[layer](full, *full_params, cos, sin, mask)
                if step == 0:
                    saved.append((key.clone(), value.clone()))
                else:
                    torch.testing.assert_close(key, saved[layer][0], rtol=0, atol=0)
                    torch.testing.assert_close(value, saved[layer][1], rtol=0, atol=0)
                before = tuple(t.clone() for t in saved[layer])
                decoded = cached[layer](
                    decoded,
                    *[p[:, 8:] for p in full_params],
                    cos[:, :, 8:],
                    sin[:, :, 8:],
                    mask[:, :, 8:],
                    *saved[layer],
                )
                for actual, expected in zip(saved[layer], before, strict=False):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(decoded, full[:, 8:], atol=2e-6, rtol=2e-5)


def test_shared_modulation_is_computed_once_per_call(graph):
    torch.manual_seed(4)
    linear = nn.Linear(128, 512, bias=False)
    calls = []
    linear.register_forward_hook(lambda module, args, result: calls.append(1))
    module = graph.SharedModulation(nn.Sequential(nn.SiLU(), linear))
    result = module(torch.randn(2, 128))
    assert len(calls) == 1
    assert all(t.shape == (2, 128) for t in result)
    assert result[1].abs().max() <= 1 and result[3].abs().max() <= 1
    assert len(list(torch.fx.symbolic_trace(module).graph.nodes)) > 1


@pytest.mark.parametrize("length", [-1, 1.5, True])
def test_invalid_prefix_length_rejected(graph, length):
    with pytest.raises(ValueError, match="prefix_length"):
        graph.ExtractBlock(graph.BlockCore(source_block()), length)


def test_both_graph_signatures_trace(graph):
    core = graph.BlockCore(source_block())
    for module, expected in [(graph.ExtractBlock(core, 8), 8), (graph.CachedBlock(core), 10)]:
        traced = torch.fx.symbolic_trace(module)
        assert sum(n.op == "placeholder" for n in traced.graph.nodes) == expected


@pytest.fixture
def runtime():
    import sys

    path = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21/runtime.py"
    spec = importlib.util.spec_from_file_location("qwen_image21_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stack_runner_matches_reextract_and_owns_compact_cache(graph, runtime):
    torch.manual_seed(38)
    cores = [graph.BlockCore(source_block()) for _ in range(2)]
    extract = graph.ExtractStack(cores, 8, 8)
    cached = graph.CachedStack(cores)
    runner = runtime.DenoiseStackRunner(
        extract, cached, layers=2, heads=2, head_dim=64, prefix_length=8, target_length=8
    )
    shared = graph.SharedModulation(nn.Sequential(nn.SiLU(), nn.Linear(128, 512, bias=False)))
    hidden = torch.randn(1, 16, 128)
    rows = torch.randn(2, 128)
    angles = torch.randn(1, 1, 16, 32)
    cos, sin = angles.cos(), angles.sin()
    allowed = torch.arange(16)[:, None] >= torch.arange(16)[None]
    allowed[8:, 8:] = True
    mask = torch.where(allowed, 0.0, -65504.0)[None, None]
    values = shared(rows)
    first, state = runner.extract(hidden, *values, cos, sin, mask)
    assert first.shape == (1, 8, 128)
    assert len(state.tensors) == 4
    for tensor in state.tensors:
        assert tensor.is_contiguous()
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    snapshots = [tensor.clone() for tensor in state.tensors]
    for _ in range(2):
        hidden[:, 8:] = torch.randn(1, 8, 128)
        rows[0] = torch.randn(128)
        values = shared(rows)
        reference, regenerated = runner.extract(hidden, *values, cos, sin, mask)
        actual = runner.cached(state, hidden[:, 8:], *values, cos[:, :, 8:], sin[:, :, 8:], mask[:, :, 8:])
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        for original, stored, repeated in zip(snapshots, state.tensors, regenerated.tensors, strict=False):
            torch.testing.assert_close(stored, original, atol=0, rtol=0)
            torch.testing.assert_close(repeated, original, atol=0, rtol=0)
    other = runtime.DenoiseStackRunner(
        extract, cached, layers=2, heads=2, head_dim=64, prefix_length=8, target_length=8
    )
    with pytest.raises(ValueError, match="another request"):
        other.cached(state, hidden[:, 8:], *values, cos[:, :, 8:], sin[:, :, 8:], mask[:, :, 8:])
    assert len(list(torch.fx.symbolic_trace(extract).graph.nodes)) > 90
    assert len(list(torch.fx.symbolic_trace(cached).graph.nodes)) > 90


def test_runner_rejects_missing_or_wrong_cache_ports(runtime):
    hidden = torch.ones(1, 16, 128)
    bad = runtime.DenoiseStackRunner(
        lambda *args: (hidden,), None, layers=2, heads=2, head_dim=64, prefix_length=8, target_length=8
    )
    with pytest.raises(ValueError, match="one K/V pair"):
        bad.extract(hidden)
    bad.extract_graph = lambda *args: (hidden, *[torch.zeros(1, 8, 2, 64)] * 4)
    with pytest.raises(ValueError, match="cache tensor"):
        bad.extract(hidden)


def test_shared_conditioning_and_target_only_output_head(graph):
    torch.manual_seed(7)
    source = nn.Module()
    source.modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, 512, bias=False))
    source.norm_out = nn.Module()
    source.norm_out.norm = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    source.norm_out.linear = nn.Linear(128, 128, bias=False)
    source.proj_out = nn.Linear(128, 64, bias=False)
    shared = graph.SharedConditioning(source)
    head = graph.OutputHead(source)
    calls = []
    handle = source.modulation[0].register_forward_hook(lambda *args: calls.append(1))
    embedding = torch.randn(2, 128)
    parameters = shared(embedding)
    assert calls == [1]
    handle.remove()
    expected = graph.SharedModulation(source.modulation)(embedding)
    for actual, reference in zip(parameters[:4], expected, strict=False):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    hidden = torch.randn(1, 4, 128)
    output = head(hidden, parameters[4])
    changed_prefix_scale = parameters[4].detach().clone()
    changed_prefix_scale[1] += 100
    torch.testing.assert_close(head(hidden, changed_prefix_scale), output, rtol=0, atol=0)
    assert output.shape == (1, 4, 64)
    for module in (shared, head):
        torch.fx.symbolic_trace(module)


@pytest.fixture
def layout_module():
    import sys

    path = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21/layout.py"
    spec = importlib.util.spec_from_file_location("qwen_image21_layout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_time_features_keep_fp32_phase_and_prefix_row(runtime, graph):
    wide = runtime.timestep_features(0.987654)
    half = runtime.timestep_features(0.987654, dtype=torch.float16)
    assert wide.shape == (2, 256)
    torch.testing.assert_close(half, wide.half(), rtol=0, atol=0)
    torch.testing.assert_close(wide[1, :128], torch.ones(128), rtol=0, atol=0)
    torch.testing.assert_close(wide[1, 128:], torch.zeros(128), rtol=0, atol=0)
    for invalid in (float("nan"), float("inf"), [0.2, 0.3]):
        with pytest.raises(ValueError, match="finite scalar"):
            runtime.timestep_features(invalid)
    source = nn.Module()
    source.time_text_embed = nn.Module()
    source.time_text_embed.timestep_embedder = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    source.modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, 512))
    source.norm_out = nn.Module()
    source.norm_out.linear = nn.Linear(128, 128)
    module = graph.TimeConditioning(source)
    current = module(wide)
    later = module(runtime.timestep_features(0.2))
    for a, b in zip(current, later, strict=False):
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    prepared = runtime.prepare_conditioning(*current)
    expected = graph.SharedConditioning(source)(source.time_text_embed.timestep_embedder(wide))
    for actual, reference in zip(prepared, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.fx.symbolic_trace(module)


def test_layout_distinguishes_adjacent_images_and_preserves_rope(layout_module):
    layout = layout_module.prepare_layout(
        torch.tensor([[False, True, True, False]]), [(1, 2, 2)] * 3, prefix_capacity=16
    )
    assert layout.prefix_tokens == 10 and layout.target_tokens == 4
    bias = layout.attention_bias(device="cpu", dtype=torch.float32)[0, 0]
    assert bias[7, 10] == 0  # Same reference is bidirectional.
    assert bias[7, 11] < 0  # Adjacent references remain separate blocks.
    assert bias[11, 7] == 0  # Second reference sees previous reference.
    assert torch.all(bias[:, :6] < 0)  # Right-aligned prefix padding never contributes keys.
    assert torch.all(bias[:16, 16:] < 0)  # Prefix cannot see target.
    assert torch.all(bias[16:, 16:] == 0)
    text = torch.arange(4.0).view(1, 4, 1)
    images = (100 + torch.arange(12.0)).view(1, 12, 1)
    hidden = layout.assemble(text, images)
    assert hidden[0, 6, 0] == 0 and hidden[0, 15, 0] == 3
    torch.testing.assert_close(hidden[:, 7:15], images[:, :8])
    torch.testing.assert_close(hidden[:, 16:], images[:, 8:])
    assert torch.count_nonzero(hidden[:, :6]) == 0
    angle = torch.arange(14 * 4).reshape(14, 4).float() / 17
    rope = torch.polar(torch.ones_like(angle), angle)
    cos, sin = layout.pad_rope(rope)
    torch.testing.assert_close(cos[0, 0, 16:], rope[10:].real, rtol=0, atol=0)
    torch.testing.assert_close(sin[0, 0, 16:], rope[10:].imag, rtol=0, atol=0)


def test_prefix_gear_padding_preserves_two_layer_outputs_and_cache(graph, layout_module):
    torch.manual_seed(44)
    slots = torch.tensor([[False, True, True, False]])
    layouts = [layout_module.prepare_layout(slots, [(1, 2, 2)] * 3, prefix_capacity=capacity) for capacity in (10, 16)]
    text = torch.randn(1, 4, 128)
    images = torch.randn(1, 12, 128)
    angle = torch.randn(14, 32)
    rope = torch.polar(torch.ones_like(angle), angle)
    cores = [graph.BlockCore(source_block()) for _ in range(2)]
    params = graph.SharedModulation(nn.Sequential(nn.SiLU(), nn.Linear(128, 512, bias=False)))(torch.randn(2, 128))
    outputs = []
    with torch.inference_mode():
        for layout in layouts:
            cos, sin = layout.pad_rope(rope)
            bias = layout.attention_bias(device="cpu", dtype=torch.float32)
            extracted = graph.ExtractStack(cores, layout.prefix_capacity, 4)(
                layout.assemble(text, images), *params, cos, sin, bias
            )
            target = graph.CachedStack(cores)(
                images[:, -4:], *params, cos[:, :, -4:], sin[:, :, -4:], bias[:, :, -4:], extracted[1:]
            )
            torch.testing.assert_close(target, extracted[0][:, -4:], atol=2e-6, rtol=2e-5)
            outputs.append(extracted)
        torch.testing.assert_close(outputs[0][0][:, -4:], outputs[1][0][:, -4:], atol=2e-6, rtol=2e-5)
        for compact, padded in zip(outputs[0][1:], outputs[1][1:], strict=False):
            torch.testing.assert_close(compact, padded[:, :, 6:16], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("capacity,shapes", [(9, [(1, 2, 2)] * 3), (16, [(1, 2, 2)] * 2), (16, [(1, 3, 2)] * 3)])
def test_invalid_layout_rejected(layout_module, capacity, shapes):
    with pytest.raises(ValueError):
        layout_module.prepare_layout(torch.tensor([[False, True, True, False]]), shapes, prefix_capacity=capacity)


@pytest.mark.parametrize("layerwise", [False, True])
def test_denoise_request_matches_full_recomputation(graph, runtime, layout_module, layerwise):
    torch.manual_seed(91)
    source = nn.Module()
    source.img_in = nn.Linear(64, 128, bias=False)
    source.txt_in = nn.Module()
    source.txt_in.text_norm = nn.RMSNorm(96, eps=1e-6)
    source.txt_in.in_layer = nn.Linear(96, 128, bias=False)
    source.txt_in.act = nn.GELU(approximate="tanh")
    source.txt_in.out_layer = nn.Linear(128, 128, bias=False)
    source.time_text_embed = nn.Module()
    source.time_text_embed.timestep_embedder = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    source.modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, 512, bias=False))
    source.norm_out = nn.Module()
    source.norm_out.linear = nn.Linear(128, 128, bias=False)
    source.norm_out.norm = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    source.proj_out = nn.Linear(128, 64, bias=False)
    layout = layout_module.prepare_layout(
        torch.tensor([[False, True, True, False]]), [(1, 2, 2)] * 3, prefix_capacity=16
    )
    rope = torch.polar(torch.ones(14, 32), torch.randn(14, 32))
    text, references = torch.randn(1, 4, 96), torch.randn(1, 8, 64)
    cores = [graph.BlockCore(source_block()) for _ in range(2)]
    extract, cached = graph.ExtractStack(cores, 16, 4), graph.CachedStack(cores)
    instrumented = extract
    if layerwise:
        adapter = runtime.LayerwiseDenoiser(
            [graph.ExtractBlock(core, 16) for core in cores],
            [graph.CachedBlock(core) for core in cores],
            prefix_length=16,
            target_length=4,
        )
        instrumented = adapter.extract_blocks[0]
        extract, cached = adapter.extract, adapter.cached
    image_projection, text_projection = graph.ImageProjection(source), graph.TextProjection(source)
    time_conditioning, output_head = graph.TimeConditioning(source), graph.OutputHead(source)
    request = runtime.DenoiseRequest(
        layout,
        rope,
        text,
        references,
        text_projection=text_projection,
        image_projection=image_projection,
        time_conditioning=time_conditioning,
        output_head=output_head,
        extract=extract,
        cached=cached,
        layers=2,
        heads=2,
        head_dim=64,
    )
    calls = []
    handle = instrumented.register_forward_hook(lambda *args: calls.append(1))
    snapshots = None
    with torch.no_grad():
        for time in (0.9, 0.5, 0.1):
            target = torch.randn(1, 4, 64)
            actual = request(target, time)
            assert calls == [1]
            params = runtime.prepare_conditioning(*time_conditioning(runtime.timestep_features(time)))
            joint = layout.assemble(text_projection(text), image_projection(torch.cat((references, target), dim=1)))
            # Bypass the instrumented extract wrapper for a fresh full-sequence reference.
            expected_hidden = graph.ExtractStack(cores, 16, 4)(
                joint, *params[:4], *layout.pad_rope(rope), layout.attention_bias(device="cpu", dtype=torch.float32)
            )[0]
            expected = output_head(expected_hidden[:, -4:], params[4])
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            if snapshots is None:
                snapshots = [value.clone() for value in request.state.tensors]
            else:
                for a, b in zip(snapshots, request.state.tensors, strict=True):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
    handle.remove()
    with pytest.raises(ValueError, match="target latents"):
        request(torch.randn(1, 5, 64), 0.1)


def test_token_projection_preserves_short_inputs(runtime):
    linear = nn.Linear(8, 16)
    projection = runtime.TokenProjection(linear, capacity=12, input_channels=8)
    for tokens in (1, 7, 12):
        value = torch.randn(1, tokens, 8)
        torch.testing.assert_close(projection(value), linear(value))
    for shape in ((1, 13, 8), (1, 0, 8), (2, 7, 8), (1, 7, 9)):
        with pytest.raises(ValueError, match="projection"):
            projection(torch.zeros(shape))


@pytest.mark.parametrize("tokens", [3, 8])
def test_sdk_norm_and_rope_preserve_adjacent_pairs_and_cache_layout(graph, tokens):
    from xhquant.nn import RMSNorm, Rope

    torch.manual_seed(71)
    source = source_block()
    with torch.no_grad():
        source.attn.norm_q.weight.copy_(torch.randn(64))
        source.attn.norm_k.weight.copy_(torch.randn(64))
    explicit = graph.BlockCore(source)
    fused = graph.BlockCore(source, sdk_ops=True)
    assert isinstance(fused.q_norm, RMSNorm) and isinstance(fused.k_norm, RMSNorm)
    assert isinstance(fused.rotary, Rope)
    hidden = torch.randn(1, tokens, 128) * 19
    scale = torch.randn(1, tokens, 128)
    angles = torch.randn(1, 1, tokens, 32)
    with torch.no_grad():
        expected = explicit.project(hidden, scale, angles.cos(), angles.sin())
        actual = fused.project(hidden, scale, angles.cos(), angles.sin())
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=2e-6)
    # A coordinate basis exposes accidental half-rotation or failure to restore
    # interleaved K cache layout even when QK dot products happen to agree.
    basis = torch.arange(128 * tokens, dtype=torch.float32).reshape(1, 2, tokens, 64)
    torch.testing.assert_close(
        fused.rope(basis, angles.cos(), angles.sin()), explicit.rope(basis, angles.cos(), angles.sin()), atol=0, rtol=0
    )
