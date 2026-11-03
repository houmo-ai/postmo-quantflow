"""Real CPU export and W16A16/W8A16/W8A8 graph and cache regressions."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from test_qwen_image21_full_graph import source
from torch import nn

from xhmodel_merak.xh_other_model.models.qwen_image21.artifacts import DenoiseBundle, DenoiseGear
from xhmodel_merak.xh_other_model.models.qwen_image21.export import export_cpu_graph, export_full_dit_bundle
from xhmodel_merak.xh_other_model.models.qwen_image21.layout import prepare_layout


def quant_config(quant_type):
    from xhquant.api import ConfigDict, DeviceType, QuantScheme, create_quant_config

    return DeviceType.XH2a, ConfigDict(
        create_quant_config(
            QuantScheme(
                target_device=DeviceType.XH2a,
                quant_type=quant_type,
            )
        )
    )


def assert_graph_precision(graph, weight_bits, activation_bits):
    weighted_nodes = [node for node in graph.node if node.op_type in {"Linear", "Conv2d"}]
    assert weighted_nodes
    for node in weighted_nodes:
        attributes = {entry.name: onnx.helper.get_attribute_value(entry) for entry in node.attribute}
        assert attributes["hmfp_weight_man_bit"] == weight_bits
        assert attributes["hmfp_act_man_bit"] == activation_bits


@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("quant_type,activation_bits", [("w8a8_sefp", 8), ("w8a16_sefp", 16)])
def test_both_frontends_export_on_cpu_without_mutating_source(tmp_path, dynamo, quant_type, activation_bits):
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8)).eval()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    values = torch.randn(1, 4, 8)
    target, config = quant_config(quant_type)
    path = tmp_path / "linear.onnx"
    export_cpu_graph(
        model, (values,), path, ["hidden"], ["output"], target_device=target, quant_config=config, dynamo=dynamo
    )
    graph = onnx.load(path).graph
    assert_graph_precision(graph, 8, activation_bits)
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_export_rejects_device_residency_before_conversion(tmp_path):
    target, config = quant_config("w8a8_sefp")
    for model, values, message in (
        (nn.Linear(8, 8, device="meta"), torch.zeros(1, 8), "source weights"),
        (nn.Linear(8, 8), torch.empty(1, 8, device="meta"), "CPU calibration"),
    ):
        path = tmp_path / "invalid.onnx"
        with pytest.raises(ValueError, match=message):
            export_cpu_graph(model, (values,), path, ["hidden"], ["output"], target_device=target, quant_config=config)
        assert not path.exists()


@pytest.mark.parametrize(
    "quant_type,weight_bits,activation_bits",
    [
        ("w16a16_sefp", 16, 16),
        ("w8a16_sefp", 8, 16),
        ("w8a8_sefp", 8, 8),
    ],
)
def test_two_cpu_exported_graphs_preserve_cache_abi_and_request_isolation(
    tmp_path,
    quant_type,
    weight_bits,
    activation_bits,
):
    torch.manual_seed(21)
    model = source()
    model.config = SimpleNamespace(causal_condition=True, patch_size=1)
    model.txt_in = nn.Module()
    model.txt_in.text_norm = nn.RMSNorm(96, eps=1e-6)
    model.txt_in.in_layer = nn.Linear(96, 128)
    model.txt_in.act = nn.GELU(approximate="tanh")
    model.txt_in.out_layer = nn.Linear(128, 128)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    layout = prepare_layout(
        torch.tensor([[False, True, False, False]]), [(1, 2, 2), (1, 2, 2)], prefix_capacity=8, target_capacity=8
    )
    rope = torch.polar(torch.ones(11, 32), torch.randn(11, 32))
    features, latents = torch.randn(1, 4, 96), torch.randn(1, 8, 4)
    target, config = quant_config(quant_type)
    root = tmp_path / "dit"
    meta = export_full_dit_bundle(
        model,
        root,
        gear=DenoiseGear(8, 8, 8, 8),
        layout=layout,
        compact_rope=rope,
        text_features=features,
        image_latents=latents,
        timestep=1.0,
        target_device=target,
        quant_config=config,
    )
    assert meta["cache_policy"] == "inplace_replace_target"
    for role, input_count in (("text", 10), ("image", 10)):
        from _qwen_image21_split_probe import assert_split_probe

        exported = onnx.load(root / meta["graphs"][role])
        graph = exported.graph
        assert len(graph.input) == input_count and len(graph.output) == 1
        assert_graph_precision(graph, weight_bits, activation_bits)
        assert_split_probe(exported, [128] * 4, [1, 512], tmp_path / f"{role}_split.onnx", split_output="split")
        assert sum(node.op_type == "RMSNorm" for node in graph.node) == (5 if role == "text" else 4)
        assert sum(node.op_type == "Rope" for node in graph.node) == 4
        if role == "text":
            assert sum(node.op_type == "FlashAttention" for node in graph.node) == 2
            assert not any(node.op_type in {"Softmax", "MaskedSoftmax", "MatMul"} for node in graph.node)
        else:
            assert not any(node.op_type == "FlashAttention" for node in graph.node)
            assert sum(node.op_type == "MatMul" for node in graph.node) == 4
            assert sum(node.op_type == "Softmax" for node in graph.node) == 2
        assert sum(node.op_type == "KVcache" for node in graph.node) == 4
        assert not any(value.name == "query_kv_range_abs" for value in graph.input)
        lengths = [value for value in graph.input if value.name in {"past_seq_length", "current_input_length"}]
        assert len(lengths) == 2 and all(
            value.type.tensor_type.elem_type == onnx.TensorProto.INT32 for value in lengths
        )
        assert not any(node.op_type in {"ReduceMean", "Sqrt", "Div"} for node in graph.node)
    for name, value in model.state_dict().items():
        assert value.device.type == "cpu" and value.dtype == torch.float32
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    moved = tmp_path / "moved"
    root.rename(moved)
    bundle = DenoiseBundle(moved, device="cpu")
    assert bundle.loader.initializer_pool.summary()["shared_saved_nbytes"] > 0

    def request():
        return bundle.new_request(layout, rope, features, latents[:, :4])

    current = request()
    first = current(latents[:, 4:].half(), 1.0)
    prefixes = [value[:, :, : layout.prefix_tokens].clone() for value in current.caches]
    targets = [value[:, :, layout.prefix_tokens : layout.prefix_tokens + 4].clone() for value in current.caches]
    second = current((latents[:, 4:] + 0.2).half(), 0.8)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    for index, value in enumerate(current.caches):
        torch.testing.assert_close(value[:, :, : layout.prefix_tokens], prefixes[index], rtol=0, atol=0)
        assert not torch.equal(value[:, :, layout.prefix_tokens : layout.prefix_tokens + 4], targets[index])
    other = request()
    torch.testing.assert_close(other(latents[:, 4:].half(), 1.0), first, rtol=0, atol=0)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(current.caches, other.caches, strict=True))
    larger_layout = prepare_layout(
        torch.tensor([[False, True, False, False]]), [(1, 2, 2), (1, 2, 4)], prefix_capacity=8, target_capacity=8
    )
    larger_rope = torch.cat((rope[:7], torch.polar(torch.ones(8, 32), torch.randn(8, 32))))
    larger = bundle.new_request(larger_layout, larger_rope, features, latents[:, :4])
    assert larger(torch.randn(1, 8, 4).half(), 1.0).shape == (1, 8, 4)
    torch.testing.assert_close(request()(latents[:, 4:].half(), 1.0), first, rtol=0, atol=0)


# Keep the expensive public six-graph flow to one deployment configuration per
# model. Component-level tests retain the precision and attention-mode matrices.
@pytest.mark.parametrize("quant_type,weight_bits,activation_bits", [("w8a8_sefp", 8, 8)])
@pytest.mark.parametrize("model_id", ["qwen-image-2.1", "qwen-image-2.1-turbo"])
@pytest.mark.parametrize("native_attention", [True])
def test_public_six_graph_build_relocate_generate_edit_and_repeat(
    tmp_path,
    monkeypatch,
    restore_model_initialization_state,
    quant_type,
    weight_bits,
    activation_bits,
    model_id,
    native_attention,
):
    import numpy as np
    from _qwen_image21_small_fixture import prepare_qwen_image21

    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_catalog.registry import get_catalog
    from xhmodel_merak.model_inference import inference_model

    catalog = get_catalog()
    source_dir = tmp_path / "source"
    with restore_model_initialization_state():
        inputs = prepare_qwen_image21(source_dir, tmp_path, catalog, catalog._recipes[model_id], monkeypatch)
        if native_attention:
            from dataclasses import replace

            recipe = catalog._recipes[model_id]
            monkeypatch.setitem(
                catalog._recipes,
                model_id,
                replace(
                    recipe,
                    config_replacements=(*recipe.config_replacements, ("export.use_noncausal_flash_attention", True)),
                ),
            )
        result = build_model(
            model_id=model_id,
            model_dir=str(source_dir),
            output_dir=str(tmp_path / "artifact"),
            device="cuda:0",
            deployment_precision={"image": quant_type},
        )
    assert result.loadable and result.deployment_precision == {"image": quant_type}
    moved = tmp_path / "relocated"
    Path(result.output_dir).rename(moved)
    shutil.rmtree(source_dir)
    graphs = list(moved.rglob("*.onnx"))
    assert len(graphs) == 6
    for path in graphs:
        graph = onnx.load(path).graph
        assert_graph_precision(graph, weight_bits, activation_bits)
        causal = path.parent.name.startswith(("text_encode_", "denoise_text_"))
        if causal:
            nodes = [node for node in graph.node if node.op_type == "FlashAttention"]
            assert nodes and all(len(node.input) >= 6 and node.input[4] and node.input[5] for node in nodes)
            assert not any(node.op_type in {"Softmax", "MaskedSoftmax", "BfpFlashAttention"} for node in graph.node)
        elif native_attention:
            nodes = [node for node in graph.node if node.op_type == "FlashAttention"]
            assert nodes and all(len(node.input) >= 10 and node.input[5] and node.input[9] for node in nodes)
            assert not any(node.op_type in {"Softmax", "MatMul"} for node in graph.node)
        else:
            assert not any(node.op_type == "FlashAttention" for node in graph.node)
            assert any(node.op_type == "Softmax" for node in graph.node)
            assert any(node.op_type == "MatMul" for node in graph.node)
        if path.parent.name.startswith("vae_"):
            assert all(len(value.type.tensor_type.shape.dim) == 4 for value in graph.input[:6])
            assert len(graph.output[0].type.tensor_type.shape.dim) == 4
            assert not any(node.op_type == "Normalize" for node in graph.node)
            norms = [node for node in graph.node if node.op_type == "RMSNorm"]
            assert norms
            for node in norms:
                attrs = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
                assert "epsilon_mode" not in attrs
                assert attrs["eps"] == pytest.approx(1e-8)
        if path.parent.name.startswith("denoise_"):
            assert len(graph.output) == 1
            assert sum(node.op_type == "KVcache" for node in graph.node) == 4
        initializers = {value.name: value for value in graph.initializer}
        for node in graph.node:
            if node.op_type in {"Linear", "Conv2d"}:
                expected_type = onnx.TensorProto.INT8 if weight_bits == 8 else onnx.TensorProto.INT16
                assert initializers[node.input[1]].data_type == expected_type
    arguments = dict(artifact_dir=str(moved), expected_model_id=model_id, device="cuda:0")
    generated = inference_model(**arguments, inputs=inputs)
    assert generated.image.shape == (64, 32, 4) and generated.image.dtype == np.uint8
    assert generated.stats["steps"] == (8 if model_id.endswith("-turbo") else inputs["steps"])
    if model_id.endswith("-turbo"):
        overridden = inference_model(**arguments, inputs={**inputs, "sigmas": [1.0, 0.5]})
        assert overridden.stats["steps"] == 2
        assert not np.array_equal(overridden.image, generated.image)
    edited = inference_model(
        **arguments,
        inputs={
            **inputs,
            "prompt": "Make it blue.",
            "images": [str(tmp_path / "reference.png")],
        },
    )
    assert edited.image.shape == generated.image.shape and edited.stats["reference_count"] == 1
    other_size = inference_model(**arguments, inputs={**inputs, "height": 32, "width": 96})
    assert other_size.image.shape == (32, 96, 4)
    resized_edit = inference_model(
        **arguments,
        inputs={
            **inputs,
            "height": 96,
            "width": 32,
            "images": [str(tmp_path / "reference.png")],
            "negative_prompt": "",
            "guidance_scale": 2.0,
        },
    )
    assert resized_edit.image.shape == (96, 32, 4)
    repeated = inference_model(**arguments, inputs=inputs)
    assert np.array_equal(generated.image, repeated.image)
    from PIL import Image

    from examples_merak.aigc.qwen_image21 import unified
    from examples_merak.unified_api import generate_image

    cli_generated = tmp_path / "cli" / "generated.png"
    unified.main([
        "--model-id", model_id,
        "--artifact-dir", str(moved), "--prompt", inputs["prompt"], "--steps", str(inputs["steps"]),
        "--seed", str(inputs["seed"]), "--device", "cuda:0", "--output", str(cli_generated),
    ])
    with Image.open(cli_generated) as image:
        assert image.mode == "RGBA"
        np.testing.assert_array_equal(image, generated.image)
    if model_id.endswith("-turbo"):
        cli_override = tmp_path / "cli" / "sigma_override.png"
        unified.main([
            "--model-id", model_id, "--artifact-dir", str(moved), "--prompt", inputs["prompt"],
            "--sigmas", "1", "0.5", "--seed", str(inputs["seed"]), "--device", "cuda:0",
            "--output", str(cli_override),
        ])
        with Image.open(cli_override) as image:
            np.testing.assert_array_equal(image, overridden.image)
    cli_edit = tmp_path / "cli" / "edited.png"
    generate_image.main([
        "--artifact-dir", str(moved), "--prompt", inputs["prompt"], "--steps", str(inputs["steps"]),
        "--seed", str(inputs["seed"]), "--device", "cuda:0", "--output", str(cli_edit),
        "--image", str(tmp_path / "reference.png"), "--height", "96", "--width", "32",
        "--negative-prompt", "", "--guidance-scale", "2",
    ])
    with Image.open(cli_edit) as image:
        assert image.mode == "RGBA" and image.size == (32, 96)
        np.testing.assert_array_equal(image, resized_edit.image)
