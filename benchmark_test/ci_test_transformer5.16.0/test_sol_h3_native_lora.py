# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 keep-LoRA preserves base weights, compensation and both components."""

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file
from test_sol_h3_static_graph import text_key_mask, tiny_source

from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import H3Schedule, fold_h3_conditioning, modality_tables
from xhmodel_merak.xh_other_model.models.sol_h3.h3_graph import H3DenoiserGraph, H3TextGraph
from xhmodel_merak.xh_other_model.models.sol_h3.native_lora import bind_native_lora, inspect_native_lora
from xhmodel_merak.xh_other_model.models.sol_h3.precision import H3PrecisionPolicy


PROJECTIONS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def adapter_fixture(path, source, *, rank=4, alpha=8):
    torch.manual_seed(721)
    values = {}
    for stack in ("transformer_blocks", "token_refiner.refiner_blocks"):
        for index in range(len(source.get_submodule(stack))):
            for projection in PROJECTIONS:
                target = f"{stack}.{index}.{projection}"
                module = source.get_submodule(target)
                values[f"{target}.lora_A.default.weight"] = torch.randn(rank, module.in_features) * 0.1
                values[f"{target}.lora_B.default.weight"] = torch.randn(module.out_features, rank) * 0.1
    save_file(values, path, metadata={"alpha": str(alpha), "key_format": "minimax-h3-diffusers"})
    return values


def explicit_merge(source, values, *, alpha=8, rank=4):
    result = deepcopy(source)
    for key, a in values.items():
        if ".lora_A." not in key:
            continue
        name = key.split(".lora_A.")[0]
        b = values[key.replace(".lora_A.", ".lora_B.")]
        result.get_submodule(name).weight.addmm_(b, a, alpha=alpha / rank)
    return result


def graph_inputs(source, *, text_length=3):
    torch.manual_seed(722)
    video, audio, text = torch.randn(1, 6, 8), torch.randn(1, 4, 4), torch.randn(1, text_length, 24)
    positions = torch.randn(10 + text_length, 3)
    cos, sin = source.rope(positions)
    return video, audio, text, positions, cos[None, None], sin[None, None]


@torch.no_grad()
def test_native_lora_preserves_base_and_matches_explicit_ba_native_model(tmp_path):
    source = tiny_source()
    original = {name: value.clone() for name, value in source.state_dict().items()}
    values = adapter_fixture(tmp_path / "turbo.safetensors", source)
    spec = inspect_native_lora(tmp_path / "turbo.safetensors")
    oracle = explicit_merge(source, values)
    policy = H3PrecisionPolicy(residual_scale=1024.0, rms_norm_mode="stable")
    folded = fold_h3_conditioning(source)
    tables, output = modality_tables(folded)
    text_graph = H3TextGraph(source, policy, input_divisor=16).float()
    graph = H3DenoiserGraph(source, tables, output, 6, 4, policy).float()
    base_text_weights = {name: p.clone() for name, p in text_graph.named_parameters()}
    base_dit_weights = {name: p.clone() for name, p in graph.named_parameters()}
    text_bound = bind_native_lora(text_graph, spec, component="text", policy=policy)
    dit_bound = bind_native_lora(graph, spec, component="dit", policy=policy)
    assert text_bound["target_count"] == dit_bound["target_count"] == 12
    assert text_bound["target_count"] + dit_bound["target_count"] == spec.to_dict()["target_count"] == 24
    assert set(text_bound["targets"] + dit_bound["targets"]) == {pair.target for pair in spec.pairs}
    from xhmodel_merak.xh_llm.lora_layer import LoRAStaticLayer

    for component, snapshot in ((text_graph, base_text_weights), (graph, base_dit_weights)):
        for name, module in component.named_modules():
            if isinstance(module, LoRAStaticLayer):
                torch.testing.assert_close(module.linear.weight, snapshot[name + ".weight"], rtol=0, atol=0)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    video, audio, text, positions, cos, sin = graph_inputs(source)
    refined = text_graph(text / 16, text_key_mask(3, 3))
    expected_refined = H3TextGraph(oracle, policy, input_divisor=16)(text / 16, text_key_mask(3, 3))
    torch.testing.assert_close(refined, expected_refined, rtol=2e-5, atol=2e-6)
    tags = torch.tensor([0] * 6 + [2] * 4 + [1] * 3)
    for step in (0, 3):
        indices, _ = H3Schedule().indices(step, tags)
        expected = oracle(
            video,
            audio,
            text,
            folded.row_times[step],
            indices.long(),
            tags,
            positions,
            torch.arange(6),
            torch.arange(6, 10),
            torch.arange(10, 13),
            return_dict=False,
        )
        actual = graph(video, audio, refined, torch.tensor([step]), cos, sin, torch.tensor([13]))
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=5e-5, atol=5e-6)
    with pytest.raises(ValueError, match="type mismatch"):
        bind_native_lora(graph, spec, component="dit", policy=policy)


@pytest.mark.parametrize("fault", ["adaln", "missing_b", "rank", "alpha", "bias"])
def test_native_lora_rejects_unsupported_or_incomplete_weights(tmp_path, fault):
    a, b = (
        "transformer_blocks.0.attn.to_q.lora_A.default.weight",
        "transformer_blocks.0.attn.to_q.lora_B.default.weight",
    )
    values = {a: torch.ones(4, 32), b: torch.ones(32, 4)}
    metadata = {"alpha": "8"}
    if fault == "adaln":
        replacement = "adaln_proj.linear"
        values = {key.replace("attn.to_q", replacement): value for key, value in values.items()}
    elif fault == "missing_b":
        del values[b]
    elif fault == "rank":
        values[b] = torch.ones(32, 3)
    elif fault == "alpha":
        metadata["alpha"] = "nan"
    else:
        values["transformer_blocks.0.attn.to_q.bias"] = torch.zeros(32)
    save_file(values, tmp_path / "bad.safetensors", metadata=metadata)
    with pytest.raises(ValueError):
        inspect_native_lora(tmp_path / "bad.safetensors")


@torch.no_grad()
def test_native_lora_target_validation_does_not_partially_bind(tmp_path):
    source = tiny_source()
    values = adapter_fixture(tmp_path / "good.safetensors", source)
    key = "transformer_blocks.1.attn.to_q.lora_A.default.weight"
    values[key] = torch.zeros(4, 33)
    save_file(values, tmp_path / "bad.safetensors", metadata={"alpha": "8"})
    spec = inspect_native_lora(tmp_path / "bad.safetensors")
    tables, output = modality_tables(fold_h3_conditioning(source))
    graph = H3DenoiserGraph(source, tables, output, 6, 4)
    before = tuple(graph.named_modules())
    with pytest.raises(ValueError, match="shape/type"):
        bind_native_lora(graph, spec, component="dit", policy=H3PrecisionPolicy())
    assert tuple(graph.named_modules()) == before




def test_native_lora_peft_directory_preserves_alpha_and_rank(tmp_path):
    import hashlib
    import json

    source = tiny_source()
    adapter_fixture(tmp_path / "adapter_model.safetensors", source)
    config = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "bias": "none"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    spec = inspect_native_lora(tmp_path, name="turbo8")
    assert spec.name == "turbo8" and spec.rank == 4 and spec.alpha == 8 and spec.scale == 2
    assert spec.to_dict()["checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    assert str(tmp_path) not in json.dumps(spec.to_dict())
    config["r"] = 8
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="rank disagrees"):
        inspect_native_lora(tmp_path)
