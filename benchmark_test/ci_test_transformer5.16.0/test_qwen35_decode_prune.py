"""Real small MoE contracts for decode ACEP, score provenance and graph isolation."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_llm.models.qwen3_5_moe.dynamic_prune import (
    configure_decode_graph,
    normalize_dynamic_prune,
    prepare_scalars,
    validate_scalars,
)
from xhquant.nn.modules.moeblock import MoeBlock


def make_block(experts=4):
    torch.manual_seed(71)
    return MoeBlock(
        "silu",
        experts,
        expert_gate_proj_weight=torch.randn(experts, 6, 4),
        expert_up_proj_weight=torch.randn(experts, 6, 4),
        expert_down_proj_weight=torch.randn(experts, 4, 6),
    )


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.moeblock = make_block()
            layer.post_attention_layernorm = nn.Module()
            # Qwen stores zero-centered HF weights; the wrapped norm uses 1+w.
            layer.post_attention_layernorm.weight = nn.Parameter(torch.zeros(4))
            layer.post_attention_layernorm.norm = nn.LayerNorm(4)
            self.layers.append(layer)


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "all"},
        {"mode": "prefill"},
        {"threshold": 0},
        {"threshold": -1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"s_scalar_eps": 0},
        {"s_scalar_eps": float("inf")},
        {"typo": True},
    ],
)
def test_invalid_prune_configuration(value):
    with pytest.raises(ValueError):
        normalize_dynamic_prune(value)


def test_prune_is_opt_in_and_decode_only():
    assert normalize_dynamic_prune(None) == {"mode": "off"}
    assert normalize_dynamic_prune({"mode": "off"})["mode"] == "off"
    assert normalize_dynamic_prune({})["mode"] == "decode"
    assert normalize_dynamic_prune({})["threshold"] == 0.85


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"0": [1, 1]},
        {"0": [1, 1], "1": [1, 1], "mtp.0": [1, 1]},
        {"0": [0, 1], "1": [1, 1]},
        {"0": [-1, 1], "1": [1, 1]},
        {"0": [float("nan"), 1], "1": [1, 1]},
        {"0": [[1, 1]], "1": [1, 1]},
    ],
)
def test_scalar_file_must_cover_exact_body_layers_with_valid_values(payload):
    with pytest.raises(ValueError):
        validate_scalars(payload, {"0": 2, "1": 2})


@pytest.mark.parametrize("suffix", [".pt", ".json"])
def test_scores_use_materialized_body_and_effective_norm_and_cache_source(tmp_path, suffix):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"quantization_config":{"bits":4}}')
    body = Body()
    # Include an MTP with the same numeric layer ID; it must never be traversed.
    body.mtp = Body()
    for layer in body.layers:
        block = layer.mlp.moeblock
        for name in ("expert_gate_proj_weight", "expert_up_proj_weight", "expert_down_proj_weight"):
            weight = getattr(block, name)
            weight[1].copy_(weight[0] * 2)
    cfg = normalize_dynamic_prune(
        {"s_scalar_compute_device": "cpu", "s_scalar_output": str(tmp_path / "export" / ("scores" + suffix))}
    )
    scores, path = prepare_scalars(body, cfg, model_dir, tmp_path / "export")
    assert set(scores) == {"0", "1"}
    for value in scores.values():
        assert value[1] / value[0] == pytest.approx(8, rel=1e-5)
    assert not hasattr(body.mtp.layers[0].mlp.moeblock, "_acep_layer_idx")
    payload = json.loads(Path(path).read_text()) if suffix == ".json" else torch.load(path, weights_only=True)
    assert payload["method"] == "acep_method1_effective_norm_v1"
    assert payload["eps"] == cfg["s_scalar_eps"]
    # A valid automatic cache is reused even if computing is disabled.
    restored, _ = prepare_scalars(body, {**cfg, "auto_s_scalar": False}, model_dir, tmp_path / "export")
    for key in scores:
        torch.testing.assert_close(restored[key], scores[key])
    with pytest.raises(ValueError, match="s_scalar_eps"):
        prepare_scalars(body, {**cfg, "s_scalar_eps": 1e-4}, model_dir, tmp_path / "export")
    (model_dir / "config.json").write_text('{"quantization_config":{"bits":8}}')
    with pytest.raises(ValueError, match="does not match"):
        prepare_scalars(body, cfg, model_dir, tmp_path / "export")


def test_explicit_partial_scalar_fails_instead_of_defaulting_missing_layers(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"0": [1, 1, 1, 1]}))
    with pytest.raises(ValueError, match="layer IDs"):
        prepare_scalars(Body(), {"s_scalar_path": str(path)}, tmp_path, tmp_path / "export")


class RoutedLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.moeblock = block

    def forward(self, hidden, routes):
        return self.moeblock(hidden, routes)


def test_frontend_decode_attributes_are_independent_but_expert_storage_is_shared():
    from xhquant.api import to_frontend_graph

    block = make_block()
    block._acep_layer_idx = "0"
    hidden = torch.tensor([[[0.4, -0.2, 0.7, 0.1]]])
    routes = torch.tensor([[[0.5, 0.25, 0.15, 0.1]]])
    prefill = to_frontend_graph(RoutedLayer(block), "TorchFX", (hidden, routes))
    decode = to_frontend_graph(RoutedLayer(block), "TorchFX", (hidden, routes))
    # Reproduce the canonical model's MoE instance sharing optimization.
    before = dict(prefill.named_modules())["moeblock"]
    decode.set_submodule("moeblock", before)
    baseline = before(hidden, routes)
    graphs = {"prefill": prefill, "decode": decode}
    configure_decode_graph(graphs, {"0": torch.ones(4)}, 0.85)
    p = dict(prefill.named_modules())["moeblock"]
    d = dict(decode.named_modules())["moeblock"]
    assert p is not d
    assert p.prune_threshold is None and p.s_scalar is None
    assert d.prune_threshold == 0.85
    assert p.expert_gate_proj_weight.data_ptr() == d.expert_gate_proj_weight.data_ptr()
    torch.testing.assert_close(p(hidden, routes), baseline)
    assert not torch.allclose(d(hidden, routes), baseline)
    weights, experts, k = d._prune_route(routes.reshape(1, 4))
    assert k == 3 and experts.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(weights, routes.reshape(1, 4)[:, :3])
    assert p.prune_threshold is None  # decode forward cannot enable prefill pruning


def test_frontend_missing_layer_tag_fails():
    block = make_block()
    with pytest.raises(ValueError, match="Invalid body MoE mapping"):
        configure_decode_graph(
            {"prefill": RoutedLayer(block), "decode": RoutedLayer(block)}, {"0": torch.ones(4)}, 0.85
        )


def test_w4_decode_prune_yaml_keeps_head_agreement_and_flashattention():
    import yaml

    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import _head_recipe_kwargs
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.xh_qwen3_5_moe_dynamic_prune_config import (
        XHQwen3_5MoeDynamicPruneModelConfig,
    )

    root = Path(__file__).resolve().parents[2]
    directory = root / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
    cfg = yaml.safe_load((directory / "qwen3_6_35b_a3b_full_head_w4_decode_prune_256k_fa.yaml").read_text())
    model = cfg["export"]["model"]
    assert model["dynamic_prune"]["mode"] == "decode"
    assert model["context_max_length"] == 262144 and model["flash_attention"]["enable"]
    assert cfg["quant"]["head"]["bits"] == 4
    assert _head_recipe_kwargs(cfg["quant"], model)
    config = XHQwen3_5MoeModelConfig(**{k: v for k, v in model.items() if k != "visual_config"})
    assert config.dynamic_prune["threshold"] == 0.85
    legacy = XHQwen3_5MoeDynamicPruneModelConfig(model_name="tiny", dynamic_prune_threshold=0.9)
    assert legacy.dynamic_prune["mode"] == "decode" and legacy.dynamic_prune["threshold"] == 0.9
