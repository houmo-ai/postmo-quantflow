# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""H3's existing vision wrapper through Sol-H3 loading and real W8A8 export."""

import copy

import pytest
import torch
from test_qwen_image21_feature_encoder import tiny_config
from transformers import Qwen3VLForConditionalGeneration

from xhmodel_merak.xh_other_model.models.qwen_image21.vision_graph import (
    KEY_MASK_VISION_INPUT_NAMES,
    KeyMaskedVision,
)
from xhmodel_merak.xh_other_model.models.sol_h3.vision import VisionFeatureAdapter, load_vision_source


@torch.no_grad()
def test_vision_loads_only_its_component_and_exports_all_deepstack_outputs(tmp_path):
    from pathlib import Path

    import onnx
    import yaml

    from xhmodel_merak.xh_other_model.models.sol_h3.build import BundleExporter
    from xhmodel_merak.xh_other_model.models.sol_h3.graph_audit import audit_graph

    torch.manual_seed(42)
    native = Qwen3VLForConditionalGeneration(tiny_config()).eval()
    native.save_pretrained(tmp_path / "text_encoder", max_shard_size="1MB")
    vision, config = load_vision_source(tmp_path)
    for name, value in vision.state_dict().items():
        torch.testing.assert_close(value, native.model.visual.state_dict()[name], rtol=0, atol=0)
    assert config["depth"] == 3
    capacity = 1280
    pixels = torch.randn(8, 1536)
    grid = torch.tensor([[1, 2, 4]])
    expected = vision(pixels, grid_thw=grid)
    wrapper = KeyMaskedVision(copy.deepcopy(vision), capacity).eval()
    adapter = VisionFeatureAdapter(
        lambda name, *args: wrapper(*args), config, gears=[dict(graph="tiny", capacity=capacity)], dtype=torch.float32
    )
    actual = adapter(pixels, grid)
    for value, reference in zip(
        (actual.pooler_output, *actual.deepstack_features),
        (expected.pooler_output, *expected.deepstack_features),
        strict=True,
    ):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-6)
    names = KEY_MASK_VISION_INPUT_NAMES
    outputs = ["image_features", "deepstack_0", "deepstack_1", "deepstack_2"]
    workflow = (
        Path(__file__).resolve().parents[2]
        / "configs_merak/workflows/xh2a/other_models/sol_h3/sol_h3_all_384p_768p_w4.yaml"
    )
    build_config = yaml.safe_load(workflow.read_text())["export"]
    build_config["text_capacity"] = 512
    exporter = BundleExporter({"h3_model": tmp_path}, tmp_path / "bundle", build_config, "cuda:0")
    exporter.meta["capacity_plan"] = dict(qwen_visual_max_patches=capacity, qwen_visual_patch_gears=[capacity])
    with pytest.raises(ValueError, match="native"):
        exporter.vision(pixels, grid.float())
    with pytest.raises(ValueError, match="patch capacity"):
        exporter.vision(torch.randn(capacity * 2, 1536), torch.tensor([[1, 4, capacity // 2]]))
    exporter.vision(pixels, grid, additional_calibrations=[(pixels[:4], torch.tensor([[1, 2, 2]]))])
    gears = exporter.meta["visual_graphs"]
    assert [gear["capacity"] for gear in gears] == [capacity]
    path = exporter.root / exporter.meta["graphs"][f"h3_visual_p{capacity}"]["path"]
    audit = audit_graph(path, quant_type="w8a8_sefp", inputs=names, outputs=outputs)
    assert audit["input_shapes"]["key_mask"] == [1, 1, 1, capacity]
    assert all(shape == [1, capacity // 4, 256] for shape in audit["output_shapes"].values())
    nodes = onnx.load(path, load_external_data=False).graph.node
    attention = [node for node in nodes if node.op_type == "FlashAttention"]
    assert len(attention) == config["depth"]
    assert all(len(node.input) == 5 and node.input[4] == "key_mask" for node in attention)
    assert not any(node.op_type in {"Less", "LessOrEqual"} for node in nodes)
    assert not any(node.op_type in {"MatMul", "Softmax"} for node in nodes)
    import shutil

    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    shutil.rmtree(tmp_path / "text_encoder")
    sessions = {
        gear["graph"]: HMONNXInferenceV2(
            str(exporter.root / exporter.meta["graphs"][gear["graph"]]["path"]),
            session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]),
        )
        for gear in gears
    }
    used = []

    def invoke_named(name, *values):
        used.append(name)
        session = sessions[name]
        return session.run(
            {
                name: value.to(device="cuda:0", dtype=session.get_input(name).dtype)
                for name, value in zip(names, values, strict=True)
            }
        )

    def invoke(*values):
        return invoke_named(f"h3_visual_p{capacity}", *values)

    exported = VisionFeatureAdapter(invoke_named, config, gears=gears, device="cuda:0")
    assert list(exported.parameters()) == []
    actual = exported(pixels, grid)
    assert used[-1] == f"h3_visual_p{capacity}"
    for value, reference in zip(
        (actual.pooler_output, *actual.deepstack_features),
        (expected.pooler_output, *expected.deepstack_features),
        strict=True,
    ):
        relative_l2 = (value.float().cpu() - reference).norm() / reference.norm()
        assert torch.isfinite(value).all()
        assert relative_l2 < 0.08, float(relative_l2)

    from xhmodel_merak.xh_other_model.models.qwen_image21.vision_graph import geometry_inputs

    # Reuse this same exported graph at another partial prefix and full capacity.
    # Native Qwen vision is the independent oracle, including all DeepStack taps.
    for height, width in ((2, 2), (4, 4)):
        count = height * width
        sample = torch.randn(count, 1536)
        routed = exported(sample, torch.tensor([[1, height, width]]))
        assert used[-1] == f"h3_visual_p{capacity}"
        expected = vision(sample, grid_thw=torch.tensor([[1, height, width]]))
        for value, reference in zip(
            (routed.pooler_output, *routed.deepstack_features),
            (expected.pooler_output, *expected.deepstack_features), strict=True
        ):
            assert (value.float().cpu() - reference).norm() / reference.norm() < 0.08
        geometry = geometry_inputs(
            (height, width),
            capacity,
            vision.num_grid_per_side,
            config["hidden_size"] // config["num_heads"],
            device="cpu",
            dtype=torch.float32,
            key_mask_only=True,
        )
        padded = torch.nn.functional.pad(sample, (0, 0, 0, capacity - count))[None]
        clean = invoke(padded, *geometry)
        dirty = padded.clone()
        dirty[:, count:] = torch.randn_like(dirty[:, count:])
        floating = wrapper(dirty, *geometry)
        for value, changed, reference in zip(
            clean,
            floating,
            (expected.pooler_output, *expected.deepstack_features),
            strict=True,
        ):
            assert torch.isfinite(value).all()
            torch.testing.assert_close(changed[:, : count // 4], reference[None], rtol=3e-5, atol=3e-6)
            relative_l2 = (value[:, : count // 4].float().cpu() - reference[None]).norm() / reference.norm()
            assert relative_l2 < 0.08, float(relative_l2)


@torch.no_grad()
def test_visual_adapter_preserves_mixed_grids_and_video_temporal_groups():
    capacities = (4, 8, 16)
    native = Qwen3VLForConditionalGeneration(tiny_config()).eval().model.visual
    wrappers = {str(n): KeyMaskedVision(copy.deepcopy(native), n) for n in capacities}
    used = []

    def invoke(name, *args):
        used.append(name)
        return wrappers[name](*args)

    adapter = VisionFeatureAdapter(
        invoke, native.config.to_dict(),
        gears=[dict(graph=str(n), capacity=n) for n in capacities], dtype=torch.float32
    )
    pixels, grids = torch.randn(36, 1536), torch.tensor([[1, 2, 4], [3, 2, 2], [1, 4, 4]])
    expected, actual = native(pixels, grid_thw=grids), adapter(pixels, grids)
    assert used == ["8", "4", "4", "4", "16"]
    with pytest.raises(ValueError, match="capacity|patch|gear"):
        adapter(torch.randn(24, 1536), torch.tensor([[1, 4, 6]]))
    for value, reference in zip(
        (actual.pooler_output, *actual.deepstack_features),
        (expected.pooler_output, *expected.deepstack_features),
        strict=True,
    ):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-6)


@torch.no_grad()
@pytest.mark.parametrize("feature_layer", [2, 3, 4])
def test_native_multimodal_presentation_reuses_mrope_and_chunk_feature_protocol(feature_layer):
    from xhmodel_merak.xh_other_model.models.sol_h3.conditioner import H3FeatureEncoder
    from xhmodel_merak.xh_other_model.models.sol_h3.feature_graph import CachedMultimodalTextFeatureStack
    from xhmodel_merak.xh_other_model.models.sol_h3.prefill import prefill_feature_group

    config = tiny_config()
    config.text_config.num_hidden_layers = 5
    native = Qwen3VLForConditionalGeneration(config).eval()
    wrapper = KeyMaskedVision(copy.deepcopy(native.model.visual), 16)
    vision = VisionFeatureAdapter(
        lambda name, *args: wrapper(*args),
        config.vision_config.to_dict(),
        gears=[dict(graph="tiny", capacity=16)],
        dtype=torch.float32,
    )
    graph = CachedMultimodalTextFeatureStack(native.model.language_model.layers[:feature_layer], divisor=16)

    def run_features(hidden, cos, sin, additions):
        def invoke(*values):
            return graph(*values[:8], list(values[8 : 8 + feature_layer]), list(values[8 + feature_layer :]))

        return prefill_feature_group(
            invoke,
            hidden,
            cos,
            sin,
            additions,
            chunk_length=4,
            cache_shape=[1, 1, 32, 128],
            num_layers=feature_layer,
            device="cpu",
            cache_dtype=torch.float32,
        )

    encoder = H3FeatureEncoder(
        config.to_dict(),
        native.model.language_model.embed_tokens.weight,
        vision,
        run_features,
        feature_layer=feature_layer,
    )
    ids = torch.tensor([[1, 62, 60, 60, 63, 2, 62, 61, 63, 3, 62, 61, 63, 4, 62, 61, 63, 5]])
    types = torch.zeros_like(ids)
    types[ids == 60], types[ids == 61] = 1, 2
    inputs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=types,
        pixel_values=torch.randn(8, 1536),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        pixel_values_videos=torch.randn(12, 1536),
        video_grid_thw=torch.tensor([[3, 2, 2]]),
        output_hidden_states=True,
        use_cache=False,
    )
    expected = native.model(**inputs).hidden_states[feature_layer]
    actual = encoder(**inputs).hidden_states[feature_layer]
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=5e-6)
    assert all("embed_tokens" in name for name, _ in encoder.named_parameters())
