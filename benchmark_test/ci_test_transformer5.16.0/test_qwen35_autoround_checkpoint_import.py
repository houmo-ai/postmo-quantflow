from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from _qwen35_moe_text_checkpoint import write_native_moe_text_checkpoint as _write_native_checkpoint
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5MoeForCausalLM

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.checkpoint import inspect_quant_artifact
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import create_workflow, materialize_workflow_config, resolve_quant_result
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.xh_llm.models.qwen3_5_moe import XHQwen3_5MoeCausalLMModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe.autoround_checkpoint import (
    build_safetensors_weight_map,
    checkpoint_overlay_with_untied_lm_head,
    convert_autoround_to_xh_ssfp,
    import_autoround_gptq_v2_weights,
    restore_untied_lm_head,
    unpack_gptq_v2_quant_weight,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


def _pack(values: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    pack_factor = 32 // bits
    values = values.to(torch.int32)
    if dim == 0:
        assert values.shape[0] % pack_factor == 0
        chunks = values.reshape(-1, pack_factor, values.shape[1])
        shifts = torch.arange(0, 32, bits, dtype=torch.int32).view(1, pack_factor, 1)
        return torch.sum(chunks << shifts, dim=1).to(torch.int32)
    assert dim == 1
    assert values.shape[1] % pack_factor == 0
    chunks = values.reshape(values.shape[0], -1, pack_factor)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32).view(1, 1, pack_factor)
    return torch.sum(chunks << shifts, dim=2).to(torch.int32)


@pytest.mark.parametrize("bits", [4, 8])
def test_unpack_gptq_v2_uses_direct_zero_points(bits: int):
    out_features = 32 // bits
    in_features = 2 * (32 // bits)
    groups = 2
    zeros = torch.full((groups, out_features), 1 << (bits - 1), dtype=torch.int16)
    signed = torch.arange(out_features * in_features, dtype=torch.int16).reshape(out_features, in_features)
    signed = signed % (1 << bits) - (1 << (bits - 1))
    g_idx = torch.arange(in_features) // (in_features // groups)
    unsigned = signed.transpose(0, 1) + zeros[g_idx]
    assert unsigned.min() >= 0 and unsigned.max() < (1 << bits)

    unpacked = unpack_gptq_v2_quant_weight(
        _pack(unsigned, bits, dim=0),
        _pack(zeros, bits, dim=1),
        g_idx,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
    )

    torch.testing.assert_close(unpacked, signed.to(torch.int8), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_negative_scale_cannot_saturate_the_signed_minimum(bits):
    codes = torch.tensor([[-(1 << (bits - 1)), 1]], dtype=torch.int8)
    with pytest.raises(ValueError, match="negative scale.*range"):
        convert_autoround_to_xh_ssfp(
            codes, torch.tensor([[-1.0]]), torch.zeros(2, dtype=torch.int32), bits=bits, group_size=2
        )


@pytest.mark.parametrize("bits", [4, 8])
def test_representable_negative_scale_preserves_each_weight(bits):
    codes = torch.tensor([[-3, 1, 0, 2]], dtype=torch.int8)
    scales = torch.tensor([[-0.5]], dtype=torch.float16)
    converted, positive = convert_autoround_to_xh_ssfp(
        codes, scales, torch.zeros(4, dtype=torch.int32), bits=bits, group_size=4
    )
    torch.testing.assert_close(converted.float() * positive.T, codes.float() * scales.T, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["get_hf_model", "get_empty_hf_model"])
def test_remote_config_respects_explicit_untrusted_load(tmp_path, method):
    marker = tmp_path / "config_executed"
    (tmp_path / "remote_config.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PretrainedConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class RemoteConfig(PretrainedConfig):\n"
        "    model_type = 'qwen_moe_untrusted_test'\n"
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen_moe_untrusted_test", "auto_map": {"AutoConfig": "remote_config.RemoteConfig"},
    }))
    with pytest.raises(ValueError):
        getattr(XHQwen3_5MoeCausalLMModel, method)(str(tmp_path), trust_remote_code=False)
    assert not marker.exists(), "trust_remote_code=False must prevent config module execution"


@pytest.mark.parametrize("method", ["get_hf_model", "get_empty_hf_model"])
@pytest.mark.parametrize("trusted", [False, True])
def test_native_config_passes_remote_model_trust_through(native_checkpoint, method, trusted):
    source, _ = native_checkpoint
    marker = source / "model_executed"
    (source / "remote_model.py").write_text(
        "from pathlib import Path\n"
        "from transformers import Qwen3_5MoeForCausalLM\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class RemoteModel(Qwen3_5MoeForCausalLM):\n"
        "    pass\n"
    )
    config = json.loads((source / "config.json").read_text())
    config["auto_map"] = {"AutoModelForCausalLM": "remote_model.RemoteModel"}
    (source / "config.json").write_text(json.dumps(config))
    loaded = getattr(XHQwen3_5MoeCausalLMModel, method)(str(source), trust_remote_code=trusted)
    assert marker.exists() is trusted
    assert (type(loaded).__name__ == "RemoteModel") is trusted


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 6, bias=False)
        self.config = type("Config", (), {"tie_word_embeddings": False})()


class _PackedExpertsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        experts = nn.Module()
        experts.gate_up_proj = nn.Parameter(torch.zeros(2, 16, 8))
        experts.down_proj = nn.Parameter(torch.zeros(2, 8, 8))
        self.model.layers[0].mlp.experts = experts


def _write_checkpoint(root: Path, tensors: dict[str, torch.Tensor], *, group_size: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "checkpoint_format": "gptq_v2",
                    "group_size": group_size,
                },
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )


def test_import_autoround_weights_attaches_signed_codes(tmp_path: Path):
    model = _TinyModel()
    signed = (torch.arange(64, dtype=torch.int16).reshape(8, 8) % 16) - 8
    g_idx = torch.arange(8) // 4
    zeros = torch.full((2, 8), 8, dtype=torch.int16)
    unsigned = signed.transpose(0, 1) + zeros[g_idx]
    tensors = {
        "layers.0.proj.qweight": _pack(unsigned, 4, dim=0),
        "layers.0.proj.qzeros": _pack(zeros, 4, dim=1),
        "layers.0.proj.scales": torch.ones(2, 8, dtype=torch.float16),
        "layers.0.proj.g_idx": g_idx.to(torch.int32),
    }
    _write_checkpoint(tmp_path, tensors)

    stats = import_autoround_gptq_v2_weights(model, tmp_path)

    assert stats.packed_modules == 1
    assert stats.w4_modules == 1
    assert stats.source_group_size == 4
    torch.testing.assert_close(model.model.layers[0].proj.quant_weight, signed.to(torch.int8), rtol=0, atol=0)
    assert model.model.layers[0].proj._autoround_bits == 4


def test_inconsistent_float_and_packed_weights_are_rejected(tmp_path):
    codes = torch.ones(8, 8, dtype=torch.int8)
    _write_checkpoint(tmp_path, {
        "layers.0.proj.weight": codes.float() * 2,
        "layers.0.proj.qweight": _pack(codes.T + 8, 4, dim=0),
        "layers.0.proj.qzeros": _pack(torch.full((2, 8), 8), 4, dim=1),
        "layers.0.proj.scales": torch.ones(2, 8, dtype=torch.float16),
        "layers.0.proj.g_idx": torch.arange(8, dtype=torch.int32) // 4,
    })
    with pytest.raises(ValueError, match="floating.*packed"):
        import_autoround_gptq_v2_weights(_TinyModel(), tmp_path)


def test_scales_outside_the_exact_sdk_fp16_contract_are_rejected(tmp_path):
    codes = torch.ones(8, 8, dtype=torch.int8)
    _write_checkpoint(tmp_path, {
        "layers.0.proj.qweight": _pack(codes.T + 8, 4, dim=0),
        "layers.0.proj.qzeros": _pack(torch.full((2, 8), 8), 4, dim=1),
        "layers.0.proj.scales": torch.full((2, 8), 0.1, dtype=torch.float32),
        "layers.0.proj.g_idx": torch.arange(8, dtype=torch.int32) // 4,
    })
    with pytest.raises(ValueError, match="scales cannot be recovered exactly"):
        import_autoround_gptq_v2_weights(_TinyModel(), tmp_path)


@pytest.mark.parametrize(
    ("bits", "has_float_weight", "code_pattern"),
    [(bits, floating, "full") for bits in (4, 8) for floating in (False, True)]
    + [(bits, bool(index % 2), pattern) for bits in (4, 8)
       for index, pattern in enumerate(("zero", "plus_one", "minus_one", "small_mixed"))],
)
def test_imported_checkpoint_scale_reaches_real_sdk_linear(
    tmp_path, monkeypatch, bits, has_float_weight, code_pattern
):
    from torch.fx import Graph, GraphModule

    from xhquant.api import ConfigDict
    from xhquant.core.graph import FrontendGraph
    from xhquant.quantization.chain_api.to_quant_graph import to_quant_graph

    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    model = _TinyModel()
    model.model.layers[0].proj = nn.Linear(64, 8, bias=False)
    codes = (torch.arange(512).reshape(8, 64) % ((1 << bits) - 1) - ((1 << (bits - 1)) - 1)).to(torch.int8)
    if code_pattern != "full":
        pattern = torch.tensor({
            "zero": [0], "plus_one": [1], "minus_one": [-1], "small_mixed": [-1, 0, 1],
        }[code_pattern], dtype=torch.int8)
        codes = pattern[torch.arange(512) % len(pattern)].reshape(8, 64)
    scales = torch.tensor([[0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0]], dtype=torch.float16)
    zeros = torch.full((1, 8), 1 << (bits - 1), dtype=torch.int32)
    prefix = "model.layers.0.proj"
    tensors = {
        prefix + ".qweight": _pack(codes.T.to(torch.int32) + zeros, bits, 0),
        prefix + ".qzeros": _pack(zeros, bits, 1),
        prefix + ".scales": scales,
        prefix + ".g_idx": torch.zeros(64, dtype=torch.int32),
    }
    expected = codes.float() * scales.T
    if has_float_weight:
        tensors[prefix + ".weight"] = expected
    _write_checkpoint(tmp_path, tensors, group_size=64)
    import_autoround_gptq_v2_weights(model, tmp_path)
    graph = Graph()
    hidden = graph.placeholder("hidden")
    graph.output(graph.call_module(prefix, (hidden,)))
    frontend = FrontendGraph.convert(GraphModule(model, graph))
    exporter = _model(tmp_path)
    exporter._apply_autoround_linear_w_schemas(frontend)
    quanted = to_quant_graph(frontend, "XH2a", ConfigDict(exporter.get_quant_cfg()))
    consumed = quanted.get_submodule(prefix)
    assert consumed.logical_bits == bits
    consumed.weight_static_quant()
    assert consumed.w_cfg.qspec.man_bit == bits
    active = codes.ne(0).any(dim=-1)
    expected_scales = scales.clone()
    expected_scales[:, ~active] = 1
    torch.testing.assert_close(consumed.scale_or_exp.squeeze(1), expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(consumed.qweight.flatten(0, 1).T, codes, rtol=0, atol=0)
    observed = (consumed.qweight.float() * consumed.scale_or_exp.float()).flatten(0, 1).T
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_import_autoround_weights_packs_defused_checkpoint_experts(tmp_path: Path):
    model = _PackedExpertsModel()
    tensors: dict[str, torch.Tensor] = {}
    expected: dict[tuple[int, str], torch.Tensor] = {}
    for expert_idx in range(2):
        for projection_index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            signed = ((torch.arange(64, dtype=torch.int16).reshape(8, 8) + expert_idx + 3 * projection_index) % 16) - 8
            zeros = torch.full((2, 8), 8, dtype=torch.int16)
            g_idx = torch.arange(8) // 4
            unsigned = signed.transpose(0, 1) + zeros[g_idx]
            base = f"layers.0.mlp.experts.{expert_idx}.{projection}"
            tensors[f"{base}.qweight"] = _pack(unsigned, 4, dim=0)
            tensors[f"{base}.qzeros"] = _pack(zeros, 4, dim=1)
            tensors[f"{base}.scales"] = torch.ones(2, 8, dtype=torch.float16)
            tensors[f"{base}.g_idx"] = g_idx.to(torch.int32)
            expected[(expert_idx, projection)] = signed.to(torch.int8)
    _write_checkpoint(tmp_path, tensors)

    stats = import_autoround_gptq_v2_weights(model, tmp_path)

    assert stats.packed_modules == 6
    experts = model.model.layers[0].mlp.experts
    for expert_idx in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            torch.testing.assert_close(
                getattr(experts, f"{projection}_quant_weight")[expert_idx],
                expected[(expert_idx, projection)],
                rtol=0,
                atol=0,
            )
            if projection == "down_proj":
                floating = experts.down_proj[expert_idx]
            else:
                offset = 0 if projection == "gate_proj" else 8
                floating = experts.gate_up_proj[expert_idx, offset:offset + 8]
            torch.testing.assert_close(floating, expected[(expert_idx, projection)].float(), rtol=0, atol=0)


def test_missing_untied_lm_head_fails_without_explicit_source(tmp_path: Path):
    quantized = tmp_path / "quantized"
    _write_checkpoint(quantized, {"embed_tokens.weight": torch.zeros(2, 2)})

    with pytest.raises(RuntimeError, match="lm_head_source"):
        restore_untied_lm_head(_TinyModel(), quantized, None)


def test_missing_untied_lm_head_is_restored_from_explicit_source(tmp_path: Path):
    quantized = tmp_path / "quantized"
    source = tmp_path / "source"
    _write_checkpoint(quantized, {"embed_tokens.weight": torch.zeros(2, 2)})
    expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    _write_checkpoint(source, {"lm_head.weight": expected})
    model = _TinyModel()

    assert restore_untied_lm_head(model, quantized, source) is True
    torch.testing.assert_close(model.lm_head.weight, expected, rtol=0, atol=0)


def test_lm_head_overlay_makes_weight_present_during_checkpoint_load(tmp_path: Path):
    quantized = tmp_path / "quantized"
    source = tmp_path / "source"
    _write_checkpoint(quantized, {"embed_tokens.weight": torch.zeros(2, 2)})
    index = {
        "metadata": {"total_size": 16},
        "weight_map": {"embed_tokens.weight": "model.safetensors"},
    }
    (quantized / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    _write_checkpoint(source, {"lm_head.weight": expected})
    source_index = {
        "metadata": {"total_size": expected.numel() * expected.element_size()},
        "weight_map": {"lm_head.weight": "model.safetensors"},
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(source_index), encoding="utf-8")

    with checkpoint_overlay_with_untied_lm_head(quantized, source) as overlay:
        overlay_map = build_safetensors_weight_map(overlay)
        assert "lm_head.weight" in overlay_map
        assert overlay_map["lm_head.weight"].is_symlink()
        assert overlay_map["lm_head.weight"].resolve() == (source / "model.safetensors").resolve()

    assert not overlay.exists()


MODEL_ID = "qwen3.5-moe-text"


@pytest.fixture
def native_checkpoint(tmp_path):
    return _write_native_checkpoint(tmp_path)


def _plan(source, output, **kwargs):
    return resolve_build_plan(
        model_id=MODEL_ID,
        model_dir=str(source),
        output_dir=str(output),
        device="cpu",
        context_length=64,
        prefill_chunk_length=8,
        **kwargs,
    )


def _model(source):
    return XHQwen3_5MoeCausalLMModel(
        XHQwen3_5MoeModelConfig(
            model_type="Qwen3_5MoeForCausalLM",
            model_name="qwen3_5_moe_text",
            hf_model=str(source),
            visual_config=None,
        )
    )


@pytest.mark.parametrize("scheme", ["w4a8h0_ssfp", "w8a8h1_sefp"])
def test_discovered_text_recipe_plans_both_existing_workflows(native_checkpoint, tmp_path, scheme):
    source, _ = native_checkpoint
    recipe = get_recipe(MODEL_ID)
    assert recipe.task == "text-generation"
    assert recipe.modalities == ("text",)
    assert recipe.quant_methods == ()
    plan = _plan(source, tmp_path / "artifact", deployment_precision={"prefill": scheme})
    config_path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, config_path)
    effective = yaml.safe_load(Path(config_path).read_text())
    assert effective["export"]["model"]["model_type"] == "Qwen3_5MoeForCausalLM"
    assert effective["export"]["model"]["model_name"] == "qwen3_5_moe_text"
    assert effective["export"]["model"]["quant_scheme"]["quant_type"] == scheme
    assert plan.components == ("prefill", "decode")
    assert plan.quant_method is None
    workflow = create_workflow(str(source), str(config_path))
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.workflow import Qwen35MoeTextWorkflow

    assert type(workflow) is Qwen35MoeTextWorkflow
    assert isinstance(workflow, Qwen35Workflow)
    assert workflow.expected_model_cls_names == {"XHQwen3_5MoeCausalLMModel"}
    assert "XHQwen3_5MoeCausalLMModel" not in Qwen35Workflow.expected_model_cls_names
    config_dir = Path("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe")
    workflow_name = (
        "122b_a10b/qwen3_6_122b_a10b_reap90_autoround.yaml"
        if scheme == "w4a8h0_ssfp"
        else "35b_a3b/qwen_agentworld_35b_a3b_autoround_attn8_shared4_expert4.yaml"
    )
    original = yaml.safe_load((config_dir / workflow_name).read_text())["export"]["model"]
    assert original["model_type"] == effective["export"]["model"]["model_type"]
    assert original["quant_scheme"] == effective["export"]["model"]["quant_scheme"]
    with pytest.raises(ModelError):
        _plan(source, tmp_path / "invalid", quant_method="autoround")
    with pytest.raises(ModelError):
        _plan(source, tmp_path / "invalid", vision={"width": 224, "height": 224})


def test_native_saved_text_checkpoint_loads_with_unchanged_weights(native_checkpoint):
    source, expected = native_checkpoint
    actual = _model(source).get_native_model()
    assert isinstance(actual, Qwen3_5MoeForCausalLM)
    for name, weight in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], weight, rtol=0, atol=0)
    # FLA may be installed in CI and routes GDR through CUDA-only Triton
    # kernels. Keep checkpoint/weight inspection above on CPU; run both real
    # models on the same GPU for the functional comparison.
    device = torch.device("cuda:0")
    actual.to(device)
    expected.to(device)
    tokens = torch.tensor([[3, 6, 8]], device=device)
    with torch.inference_mode():
        reference = expected(tokens, use_cache=False).logits
        observed = actual(tokens, use_cache=False).logits
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize(("bits", "precision"), [(4, "w4"), (8, "w8"), (4, "wmix")])
def test_saved_text_checkpoint_packed_weights_pass_public_inspection_and_load(
    native_checkpoint, tmp_path, bits, precision
):
    source, reference = native_checkpoint
    checkpoint = tmp_path / "packed"
    checkpoint.mkdir()
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "gptq",
        "provider": "auto-round",
        "bits": bits,
        "group_size": 64,
        "checkpoint_format": "gptq_v2",
        "sym": True,
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    weights = load_file(source / "model.safetensors")
    name = "model.layers.0.mlp.shared_expert.down_proj"
    shape = weights.pop(name + ".weight").shape
    assert tuple(shape) == (64, 64)
    zero = 1 << (bits - 1)
    factor = 32 // bits
    signed = (torch.arange(4096, dtype=torch.int16).reshape(64, 64) % (1 << bits) - zero).to(torch.int8)
    unsigned = signed.T.to(torch.int32) + zero
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    weights[name + ".qweight"] = (
        (unsigned.reshape(64 // factor, factor, 64) << shifts.view(1, factor, 1)).sum(1).to(torch.int32)
    )
    zeros = (torch.full((factor,), zero, dtype=torch.int32) << shifts).sum().to(torch.int32)
    weights[name + ".qzeros"] = zeros.expand(1, 64 // factor).contiguous()
    weights[name + ".scales"] = torch.ones(1, 64, dtype=torch.float16)
    weights[name + ".g_idx"] = torch.zeros(64, dtype=torch.int32)
    save_file(weights, checkpoint / "model.safetensors")
    plan = _plan(
        source,
        tmp_path / "artifact",
        quant_checkpoint=str(checkpoint),
        precision={"weight": precision, "activation": "a8"},
    )
    inspection = inspect_quant_artifact(str(checkpoint), plan)
    assert inspection["verification"] == "checkpoint_verified"
    assert not any(item.startswith(("gptqmodel", "auto-round")) for item in plan.worker_requirements)
    config_path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, config_path)
    workflow = create_workflow(str(source), str(config_path))
    result, _, _, _ = resolve_quant_result(workflow, plan, tmp_path / "artifact")
    assert result.quanted_model_dir == str(checkpoint)
    actual = _model(checkpoint).get_native_model()
    loaded = actual.model.layers[0].mlp.shared_expert.down_proj
    torch.testing.assert_close(loaded.quant_weight, signed, rtol=0, atol=0)
    torch.testing.assert_close(loaded.quant_scale, torch.ones(1, 64, dtype=torch.float16))
    assert loaded._xhquant_weight_origin == "autoround-gptq-v2"
    for key, value in weights.items():
        if key in actual.state_dict():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    expected = signed.to(torch.float16)
    torch.testing.assert_close(loaded.weight, expected, rtol=0, atol=0)
    with torch.no_grad():
        reference.model.layers[0].mlp.shared_expert.down_proj.weight.copy_(expected)
    for key, weight in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], weight, rtol=0, atol=0)
    direct = XHQwen3_5MoeCausalLMModel.get_hf_model(str(checkpoint), device_map="cpu")
    torch.testing.assert_close(direct.model.layers[0].mlp.shared_expert.down_proj.weight, expected, rtol=0, atol=0)
    # Match the optional FLA backend's CUDA execution contract without
    # replacing native forwards or weakening the exact output comparison.
    device = torch.device("cuda:0")
    actual.to(device)
    reference.to(device)
    tokens = torch.tensor([[3, 6, 8]], device=device)
    with torch.inference_mode():
        torch.testing.assert_close(
            actual(tokens, use_cache=False).logits, reference(tokens, use_cache=False).logits, rtol=0, atol=0
        )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("has_unused_base", [False, True])
def test_native_packed_untied_head_passes_inspection_and_loads_without_base(
    native_checkpoint, tmp_path, bits, has_unused_base
):
    source, reference = native_checkpoint
    checkpoint = tmp_path / "packed_head"
    checkpoint.mkdir()
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "gptq", "provider": "auto-round", "bits": bits,
        "group_size": 64, "checkpoint_format": "gptq_v2", "sym": True,
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    weights = load_file(source / "model.safetensors")
    shape = weights.pop("lm_head.weight").shape
    codes = (torch.arange(shape.numel()).reshape(shape) % 7 - 3).to(torch.int8)
    zeros = torch.full((1, shape[0]), 1 << (bits - 1), dtype=torch.int32)
    weights.update({
        "lm_head.qweight": _pack(codes.T.to(torch.int32) + zeros, bits, 0),
        "lm_head.qzeros": _pack(zeros, bits, 1),
        "lm_head.scales": torch.full((1, shape[0]), 0.5, dtype=torch.float16),
        "lm_head.g_idx": torch.zeros(shape[1], dtype=torch.int32),
    })
    save_file(weights, checkpoint / "model.safetensors")
    plan = _plan(source, tmp_path / "artifact", quant_checkpoint=str(checkpoint),
                 precision={"weight": f"w{bits}", "activation": "a8"})
    assert inspect_quant_artifact(str(checkpoint), plan)["verification"] == "checkpoint_verified"
    exporter = _model(checkpoint)
    if has_unused_base:
        exporter.config.lm_head_source = str(source)
    actual = exporter.get_native_model()
    assert actual.lm_head._xhquant_weight_origin == "autoround-gptq-v2"
    expected = codes.to(torch.float16) * 0.5
    torch.testing.assert_close(actual.lm_head.weight, expected, rtol=0, atol=0)
    hidden = torch.arange(shape[1], dtype=torch.float16).view(1, 1, -1) / shape[1]
    with torch.no_grad():
        reference.lm_head.weight.copy_(expected)
        torch.testing.assert_close(actual.lm_head(hidden), reference.lm_head(hidden), rtol=0, atol=0)


def test_partial_packed_head_is_not_replaced_by_an_external_head(tmp_path):
    _write_checkpoint(tmp_path, {"lm_head.qweight": torch.zeros(1, 8, dtype=torch.int32)})
    with pytest.raises(ValueError, match="Incomplete packed lm_head"):
        with checkpoint_overlay_with_untied_lm_head(tmp_path, tmp_path / "missing_source"):
            raise AssertionError("A partial packed head must not be replaced")


def test_text_model_owns_preprocessor_defaults_without_visual_config():
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess

    embedding = nn.Embedding(16, 8)
    source = SimpleNamespace(
        embed_tokens=embedding, wrap_cfg=SimpleNamespace(input_sequence_length=8),
        past_key_caches=[], past_value_caches=[], past_conv_caches=[], past_recurrent_states=[],
    )
    processor = XHQwen3_5MoeCausalLMModel._get_data_preprocessor(source)
    assert isinstance(processor, Qwen3_5_DataPreprocess)
    assert processor.embed_tokens is embedding
    assert processor.past_conv_caches is source.past_conv_caches
    assert processor.past_recurrent_states is source.past_recurrent_states
    assert processor.patch_size == processor.spatial_merge_size == 1
    assert (processor.image_token_id, processor.video_token_id,
            processor.vision_start_token_id, processor.vision_end_token_id) == (-1, -1, -1, -1)


@pytest.mark.parametrize("text_only", [False, True])
def test_moe_model_selects_its_own_device_aligned_processor(text_only):
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5_moe.data_preprocess import Qwen3_5_DataPreprocess
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel

    source = SimpleNamespace(
        embed_tokens=nn.Embedding(16, 8), wrap_cfg=SimpleNamespace(input_sequence_length=4),
        past_key_caches=[], past_value_caches=[], past_conv_caches=[], past_recurrent_states=[],
        config=SimpleNamespace(
            visual_config=SimpleNamespace(patch_size=14), image_token_id=11, video_token_id=12,
            vision_start_token_id=13, vision_end_token_id=14, spatial_merge_size=2,
        ),
    )
    cls = XHQwen3_5MoeCausalLMModel if text_only else XHQwen3_5MoeModel
    processor = cls._get_data_preprocessor(source)
    assert type(processor) is Qwen3_5_DataPreprocess
    assert processor.embed_tokens is source.embed_tokens
    assert processor.image_token_id == (-1 if text_only else 11)
    assert processor.patch_size == (1 if text_only else 14)


@pytest.mark.parametrize("embedding_device,input_device", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_moe_processor_aligns_embedding_inputs_and_preserves_shared_math(embedding_device, input_device):
    from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess as SharedProcessor
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.data_preprocess import Qwen3_5_DataPreprocess as MoeProcessor
    from xhmodel_merak.xh_llm.types import CacheList

    embedding = nn.Embedding(16, 8, device=embedding_device).half()
    kwargs = dict(
        token_embedding=embedding, input_sequence_length=4, spatial_merge_size=1,
        past_key_caches=CacheList(), past_value_caches=CacheList(),
        past_conv_caches=CacheList(), past_recurrent_states=CacheList(),
    )
    actual, reference = MoeProcessor(**kwargs), SharedProcessor(**kwargs)
    actual._device = reference._device = torch.device("cuda:0")
    # Exercise both prefill and decode, retaining the real shared RoPE state.
    for past_length, ids in [(0, [2, 3, 4]), (3, [5])]:
        tokens = torch.tensor([ids], device=input_device)
        data = {"input_ids": tokens, "past_seq_length": past_length}
        result = actual.forward(data)
        expected = reference.forward({**data, "input_ids": tokens.to(embedding_device)})
        assert data["input_ids"] is tokens and tokens.device == torch.device(input_device)
        for lhs, rhs in zip(result, expected, strict=True):
            if torch.is_tensor(lhs):
                torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
            else:
                assert lhs == rhs
