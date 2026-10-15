# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native text prefix before final norm and complete FastVideo hybrid semantics."""

import copy

import pytest
import torch
from safetensors.torch import save_file
from test_qwen_image21_feature_encoder import tiny_config
from transformers import Qwen3VLForConditionalGeneration

from xhmodel_merak.xh_other_model.models.sol_h3.feature_graph import (
    TEXT_FEATURE_DIVISOR,
    TextFeatureStack,
    load_text_source,
)
from xhmodel_merak.xh_other_model.models.sol_h3.hybrid import merge_dense_adapter


@torch.no_grad()
def test_native_used_text_prefix_keeps_prenorm_features_and_compensated_scale(tmp_path):
    divisor = TEXT_FEATURE_DIVISOR
    torch.manual_seed(214)
    native = Qwen3VLForConditionalGeneration(tiny_config()).eval()
    native.save_pretrained(tmp_path / "text_encoder", max_shard_size="1MB")
    source, config = load_text_source(tmp_path, layers=2)
    assert len(source.layers) == 2 and config["num_hidden_layers"] == 3
    values = torch.randn(1, 5, 256)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5)
    captured = []
    hook = native.model.language_model.layers[1].register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    native.model.language_model(inputs_embeds=values, position_ids=positions, use_cache=False)
    hook.remove()
    frequency = torch.arange(5)[:, None] * source.rotary_emb.inv_freq[None]
    phase = torch.cat((frequency, frequency), -1)[None]
    graph = TextFeatureStack(source.layers, divisor=divisor)
    actual = graph(values / divisor, phase.cos(), phase.sin(), torch.tensor([5]))
    torch.testing.assert_close(actual * divisor, captured[0], rtol=2e-5, atol=2e-6)
    assert not any("embed_tokens" in key or key.startswith("norm") for key in graph.state_dict())


def write_hybrid(path, *, bad=None):
    tensors = {
        "adaln_proj.lora_A.weight": torch.randn(2, 8).bfloat16() * 0.02,
        "adaln_proj.lora_B.weight": torch.randn(6, 2).bfloat16() * 0.02,
        "context.diff_b": torch.randn(6).bfloat16() * 0.01,
        "output.diff": torch.randn(4, 6).bfloat16() * 0.01,
    }
    if bad == "replacement":
        tensors["gate.set_weight"] = torch.zeros(6, 8)
    elif bad == "shape":
        tensors["output.diff"] = torch.zeros(4, 5)
    save_file(
        tensors,
        path,
        metadata=dict(
            format="fastvideo-lora-v2", rank="2", low_rank_tensors="2", diff_tensors="2", set_weight_tensors="0"
        ),
    )
    return tensors


def hybrid_source():
    source = torch.nn.Module()
    source.adaln_proj = torch.nn.Linear(8, 6)
    source.context = torch.nn.Linear(8, 6)
    source.output = torch.nn.Linear(6, 4)
    return source.bfloat16().eval()


@torch.no_grad()
def test_hybrid_merges_adaln_and_full_parameter_diffs_in_native_order(tmp_path):
    path = tmp_path / "adapter.safetensors"
    tensors = write_hybrid(path)
    source = hybrid_source()
    expected = copy.deepcopy(source)
    expected.adaln_proj.weight.addmm_(tensors["adaln_proj.lora_B.weight"], tensors["adaln_proj.lora_A.weight"])
    expected.context.bias.copy_((expected.context.bias.float() + tensors["context.diff_b"].float()).bfloat16())
    expected.output.weight.copy_((expected.output.weight.float() + tensors["output.diff"].float()).bfloat16())
    receipt = merge_dense_adapter(source, path, compute_device="cpu")
    assert receipt["pairs"] == 1 and receipt["diffs"] == 2 and receipt["effective_scale"] == 1
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["replacement", "shape"])
def test_bad_hybrid_checkpoint_is_rejected_before_any_weight_mutates(tmp_path, bad):
    path = tmp_path / "adapter.safetensors"
    write_hybrid(path, bad=bad)
    source = hybrid_source()
    original = copy.deepcopy(source.state_dict())
    with pytest.raises(ValueError):
        merge_dense_adapter(source, path, compute_device="cpu")
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@torch.no_grad()
def test_scaled_gptq_projection_preserves_integer_payload_and_exact_units():
    from xhmodel_merak.xh_other_model.models.sol_h3.precision import scaled_linear
    from xhquant.nn import GPTQPackedLinear, GPTQPackedWeight

    torch.manual_seed(51)
    packed = GPTQPackedWeight(
        torch.randint(-(2**31), 2**31 - 1, (16, 64), dtype=torch.int32),
        torch.full((2, 8), -2004318072, dtype=torch.int32),
        torch.rand(2, 64) * 0.02,
        torch.arange(128, dtype=torch.int32) // 64,
        bits=4,
        group_size=64,
        in_features=128,
        out_features=64,
    )
    source = GPTQPackedLinear(packed, torch.randn(64))
    graph = scaled_linear(source, input_multiplier=16, output_divisor=256)
    x = torch.randn(1, 5, 128)
    torch.testing.assert_close(graph(x / 16) * 256, source(x), rtol=0, atol=0)
    assert graph.weight is None
    for name in ("qweight", "qzeros", "g_idx"):
        assert getattr(graph.packed_weight, name).data_ptr() == getattr(packed, name).data_ptr()


@torch.no_grad()
@pytest.mark.parametrize("chunk,length", [(4, 11)])
def test_chunk_prefill_matches_native_all_features_with_deepstack(chunk, length):
    from xhmodel_merak.exec_backends import CacheTensor
    from xhmodel_merak.xh_other_model.models.sol_h3.feature_graph import CachedTextFeatureBlock

    torch.manual_seed(215)
    native = Qwen3VLForConditionalGeneration(tiny_config()).eval()
    source = native.model.language_model
    hidden = torch.randn(1, length, 256)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[:, chunk - 2 : chunk + 2] = True  # visual tokens straddle a chunk boundary
    increments = [torch.randn(mask.sum(), 256) * 0.1 for _ in range(3)]
    expected = []
    # Last layer output is before the final DeepStack addition and final norm.
    hook = source.layers[-1].register_forward_hook(lambda _, args, output: expected.append(output.clone()))
    positions = torch.arange(length).reshape(1, 1, length).expand(3, 1, length)
    source(
        inputs_embeds=hidden,
        position_ids=positions,
        visual_pos_masks=mask,
        deepstack_visual_embeds=increments,
        use_cache=False,
    )
    hook.remove()
    expected = expected[0]
    expected[mask] += increments[-1]
    phase = torch.arange(length)[:, None] * source.rotary_emb.inv_freq[None]
    phase = torch.cat((phase, phase), -1)[None]
    dense = []
    for increment in increments:
        value = torch.zeros_like(hidden)
        value[mask] = increment / 16
        dense.append(value)
    blocks = [CachedTextFeatureBlock(layer, divisor=16) for layer in source.layers]
    capacity = ((length + chunk - 1) // chunk) * chunk
    caches = [
        (
            CacheTensor(torch.zeros(1, block.kv_heads, capacity, block.head_dim)),
            CacheTensor(torch.zeros(1, block.kv_heads, capacity, block.head_dim)),
        )
        for block in blocks
    ]
    results = []
    for start in range(0, length, chunk):
        valid = min(chunk, length - start)

        def padded(value, start=start, valid=valid):
            return torch.nn.functional.pad(value[:, start : start + valid], (0, 0, 0, chunk - valid))

        value = padded(hidden / 16)
        cos, sin = padded(phase.cos()), padded(phase.sin())
        for block, (key, val), increment in zip(blocks, caches, dense, strict=True):
            value = block(
                value,
                cos,
                sin,
                torch.tensor([valid], dtype=torch.int32),
                torch.tensor([start], dtype=torch.int32),
                key,
                val,
                padded(increment),
            )
        results.append(value[:, :valid])
    torch.testing.assert_close(torch.cat(results, 1) * 16, expected, rtol=3e-5, atol=4e-6)


def test_chunk_prefill_layer_group_exports_separate_caches(tmp_path):
    from xhmodel_merak.exec_backends import CacheTensor
    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhmodel_merak.xh_other_model.models.sol_h3.feature_graph import CachedMultimodalTextFeatureStack
    from xhmodel_merak.xh_other_model.models.sol_h3.graph_audit import audit_graph

    source = Qwen3VLForConditionalGeneration(tiny_config()).eval().model.language_model
    graph = CachedMultimodalTextFeatureStack(source.layers, divisor=16)
    hidden = torch.randn(1, 32, 256) / 16
    phase = torch.arange(32)[:, None] * source.rotary_emb.inv_freq[None]
    phase = torch.cat((phase, phase), -1)[None]
    caches = [
        [CacheTensor(torch.zeros(1, layer.kv_heads, 96, layer.head_dim)) for layer in graph.layers] for _ in range(2)
    ]
    inputs = (
        hidden,
        phase.cos(),
        phase.sin(),
        torch.tensor([32], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        *[torch.zeros_like(hidden) for _ in range(3)],
        *caches,
    )
    names = [
        "hidden",
        "cos",
        "sin",
        "valid_length",
        "past_length",
        "deepstack_0",
        "deepstack_1",
        "deepstack_2",
        *[f"key_cache_{i}" for i in range(3)],
        *[f"value_cache_{i}" for i in range(3)],
    ]
    path = export_graph(
        graph,
        inputs,
        tmp_path / "cached_group.onnx",
        names,
        ["features"],
        quant_scheme=dict(target_device="XH2a", quant_type="w16a16_sefp"),
    )
    audit = audit_graph(path, quant_type="w16a16_sefp", inputs=names, outputs=["features"])
    assert audit["operators"]["KVcache"] == 6
    assert audit["operators"]["FlashAttention"] == 3
    from xhmodel_merak.xh_other_model.models.sol_h3.feature_graph import MultimodalTextFeatureStack
    from xhmodel_merak.xh_other_model.models.sol_h3.prefill import prefill_feature_group
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))

    def invoke(*values):
        feed = {
            name: value.to(device="cuda:0", dtype=session.get_input(name).dtype)
            for name, value in zip(names, values, strict=True)
        }
        output = session.run(feed)
        return output[0] if isinstance(output, (tuple, list)) else output

    # Reuse the graph for a shorter second request; no previous request's cache survives.
    for length in (59, 17):
        sequence = torch.randn(1, length, 256) / 16
        phase = torch.arange(length)[:, None] * source.rotary_emb.inv_freq[None]
        phase = torch.cat((phase, phase), -1)[None]
        additions = [torch.zeros_like(sequence) for _ in range(3)]
        for value in additions:
            value[:, max(0, length // 2 - 1) : length // 2 + 2] = torch.randn(1, 3, 256) / 16
        with torch.no_grad():
            reference = MultimodalTextFeatureStack(source.layers, divisor=16)(
                sequence, phase.cos(), phase.sin(), torch.tensor([length]), *additions
            )
        result = prefill_feature_group(
            invoke,
            sequence,
            phase.cos(),
            phase.sin(),
            additions,
            chunk_length=32,
            cache_shape=[1, graph.layers[0].kv_heads, 96, graph.layers[0].head_dim],
            num_layers=3,
            device="cuda:0",
        )
        assert result.shape == sequence.shape
        torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=5e-4)


@torch.no_grad()
def test_reference_adapter_reuses_native_peft_fusion_and_alpha_over_rank(tmp_path):
    from safetensors.torch import save_file
    from test_sol_h3_static_graph import tiny_source

    from xhmodel_merak.xh_other_model.models.minimax_h3.lora import LORA_TARGET_MODULES
    from xhmodel_merak.xh_other_model.models.sol_h3.hybrid import merge_reference_adapter

    torch.manual_seed(61)
    source = tiny_source()
    pairs, expected = {}, {}
    for name, module in source.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(LORA_TARGET_MODULES):
            a, b = torch.randn(2, module.in_features) * 0.01, torch.randn(module.out_features, 2) * 0.01
            pairs[name + ".lora_A.default.weight"] = a
            pairs[name + ".lora_B.default.weight"] = b
            expected[name] = module.weight.detach().clone() + (b @ a) * (8 / 2)
    path = tmp_path / "reference.safetensors"
    save_file(pairs, path, metadata={"alpha": "8", "format": "pt"})
    report = merge_reference_adapter(source, path)
    assert report == dict(format="peft", pairs=len(expected), rank=2, alpha=8, effective_scale=4.0)
    for name, weight in expected.items():
        torch.testing.assert_close(source.get_submodule(name).weight, weight)
    assert not any("lora_" in name for name, _ in source.named_parameters())
