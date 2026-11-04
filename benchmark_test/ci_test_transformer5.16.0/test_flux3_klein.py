# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 preprocessing, capacity and real neural graph contracts."""

import numpy as np
import pytest
import torch
from PIL import Image

from _small_model_threads import small_model_cpu_threads  # noqa: F401

from xhmodel_merak.xh_other_model.models.flux3_klein.capacity import validate_capacity, validate_request
from xhmodel_merak.xh_other_model.models.flux3_klein.processor import prepare_references


pytestmark = pytest.mark.usefixtures("small_model_cpu_threads")


@pytest.mark.parametrize("invalid_part", [None, "q", "k", "v", "p"])
def test_bundle_attention_precision_uses_current_qkvp_contract(tmp_path, invalid_part):
    """The current SDK exports Q/K/V/P attributes, without score precision."""
    import json

    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import read_metadata
    from xhmodel_merak.xh_other_model.models.flux3_klein.source import MODEL_ID

    capacity = validate_capacity({})
    ch, cw = capacity["canvas_size"]
    ports = {
        "text_encode": {"embeddings": [1, capacity["text_capacity"], 32]},
        "vae_encode": {"vae_input": [1, 3, ch, cw]},
        "vae_decode": {"vae_input": [1, 128, ch // 16, cw // 16]},
        "denoise": {
            "target": [1, capacity["target_capacity"], 128],
            "references": [1, capacity["reference_total_capacity"], 128],
            "text_features": [1, capacity["text_capacity"], 256],
        },
    }
    graphs = {}
    for role, shapes in ports.items():
        attributes = {f"{part}_bits": 8 for part in ("q", "k", "v", "p")}
        if role == "denoise" and invalid_part:
            attributes[f"{invalid_part}_bits"] = 16
        first = next(iter(shapes))
        node = helper.make_node("FlashAttention", [first] * 3, ["output"], domain="com.houmo", **attributes)
        graph = helper.make_graph(
            [node],
            role,
            [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name, shape in shapes.items()],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, shapes[first])],
        )
        graphs[role] = f"{role}.onnx"
        onnx.save(helper.make_model(graph), tmp_path / graphs[role])
    source = {"model": MODEL_ID, "dit": {"flow_map": True, "in_channels": {"text": 256}}, "sampler": {"guidance": None}}
    for name, data in (("source_config", source), ("text_config", {}), ("tokenizer", {})):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    meta = {
        "schema_version": 1,
        "kind": "flux3_klein",
        "model_id": MODEL_ID,
        "quant_type": "w8a8_sefp",
        "capacity": capacity,
        "graphs": graphs,
        "resources": {name: f"{name}.json" for name in ("source_config", "text_config")},
        "processor_files": ["tokenizer.json"],
    }
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    if invalid_part:
        with pytest.raises(ValueError, match="physical precision"):
            read_metadata(tmp_path)
    else:
        assert read_metadata(tmp_path) == meta


def test_single_reference_reflect_padding_and_crop():
    array = np.arange(700 * 1000 * 3, dtype=np.uint8).reshape(700, 1000, 3)
    prepared, crop = prepare_references([Image.fromarray(array)])
    assert prepared[0].size == (1008, 704)
    assert crop == (4, 2, 1004, 702)
    np.testing.assert_array_equal(np.asarray(prepared[0].crop(crop)), array)
    np.testing.assert_array_equal(np.asarray(prepared[0]), np.pad(array, ((2, 2), (4, 4), (0, 0)), mode="reflect"))


def test_multi_reference_uses_centre_crop_and_preserves_order():
    images = [Image.new("RGB", (1000, 700), color) for color in ("red", "blue")]
    prepared, crop = prepare_references(images)
    assert crop is None and [image.size for image in prepared] == [(992, 688)] * 2
    assert prepared[0].getpixel((0, 0)) == (255, 0, 0)
    assert prepared[1].getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_reference_total_budget_fits_large_images_and_preserves_two(count):
    images = [Image.new("RGB", (2048, 2048), (20 * i, 40, 200)) for i in range(count)]
    prepared, _ = prepare_references(images, single_max_pixels=1024**2, max_hw=(1536, 1536), total_max_tokens=8704)
    assert len(prepared) == count
    assert sum(image.width * image.height // 256 for image in prepared) <= 8704
    for i, image in enumerate(prepared):
        assert image.width % 16 == image.height % 16 == 0
        assert image.width * image.height <= 1024**2
        assert image.getpixel((image.width // 2, image.height // 2)) == (20 * i, 40, 200)
    if count <= 2:
        assert [image.size for image in prepared] == [(1024, 1024)] * count


def test_reference_total_budget_keeps_fitting_pixels_and_redistributes_small_images():
    rng = np.random.default_rng(42)
    images = [Image.fromarray(rng.integers(0, 256, (700, 1000, 3), dtype=np.uint8)) for _ in range(3)]
    expected, _ = prepare_references(images)
    fitted, crop = prepare_references(images, total_max_tokens=8704)
    assert crop is None
    for actual, original in zip(fitted, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))
    small = Image.new("RGB", (128, 128), "red")
    mixed, _ = prepare_references([small, *[Image.new("RGB", (2048, 2048), "blue")] * 3], total_max_tokens=8704)
    assert mixed[0].size == small.size
    assert all(image.width * image.height // 256 > 2176 for image in mixed[1:])
    assert sum(image.width * image.height // 256 for image in mixed) <= 8704


def test_reference_total_budget_preserves_aspect_exif_and_resamples_original_once():
    # Only the final centre crop may remove up to 15 pixels from each dimension.
    rng = np.random.default_rng(7)
    raw = Image.fromarray(rng.integers(0, 256, (1024, 2048, 3), dtype=np.uint8))
    rotated = raw.copy()
    rotated.getexif()[274] = 6
    prepared, _ = prepare_references([raw, rotated, raw, rotated], total_max_tokens=8704)
    assert prepared[0].size == prepared[2].size
    assert prepared[1].size == prepared[0].size[::-1]
    budget = 8704 // 4
    scale = (budget * 256 / (raw.width * raw.height)) ** 0.5
    resized = raw.resize((int(raw.width * scale), int(raw.height * scale)), Image.Resampling.LANCZOS)
    width, height = resized.width // 16 * 16, resized.height // 16 * 16
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    expected = resized.crop((left, top, left + width, top + height))
    np.testing.assert_array_equal(np.asarray(prepared[0]), np.asarray(expected))
    with pytest.raises(ValueError, match="budget"):
        prepare_references([raw] * 4, total_max_tokens=3)


def test_capacity_checks_joint_tokens_geometry_and_count():
    capacity = validate_capacity({})
    assert capacity["max_references"] == 8
    assert capacity["reference_total_capacity"] == 8704
    assert capacity["joint_capacity"] == 14336
    validate_request(capacity, (1024, 1088), [(1024, 1024)], 1280)
    assert validate_request(capacity, (1024, 1024), [(1024, 1024)] * 2, 1280) == [4096] * 3
    for target, references, text in [
        ((1536, 1536), [], 80),
        ((1024, 1024), [(16, 16)] * 9, 80),
        ((1024, 1024), [], 1360),
        ((1023, 1024), [], 80),
    ]:
        with pytest.raises(ValueError):
            validate_request(capacity, target, references, text)


def test_dual_reference_default_profiles_and_individual_pixel_budget():
    from pathlib import Path

    import yaml

    from xhmodel_merak.xh_other_model.models.flux3_klein.build import validate_build_config

    root = Path(__file__).resolve().parents[2]
    profiles = root / "configs_merak/workflows/xh2a/other_models/flux3_klein"
    capacity = validate_build_config(
        yaml.safe_load((profiles / "flux3_klein_1mp_w8.yaml").read_text())["export"]
    )
    assert capacity == validate_capacity({})
    originals = [Image.new("RGB", size, color) for size, color in [((2048, 2048), "red"), ((2400, 1600), "blue")]]
    prepared, crop = prepare_references(
        originals,
        multi_max_pixels=capacity["multi_reference_max_pixels"],
        max_hw=capacity["canvas_size"],
    )
    assert crop is None
    assert prepared[0].size == (1024, 1024)
    assert all(image.width * image.height <= 1024**2 for image in prepared)
    assert [image.getpixel((0, 0)) for image in prepared] == [(255, 0, 0), (0, 0, 255)]
    validate_request(capacity, (1024, 1024), [(im.height, im.width) for im in prepared], 1280)
    with pytest.raises(ValueError, match="image token count"):
        validate_request(capacity, (1024, 1024), [(1024, 1104), (1024, 1024)], 80)
    limited = validate_capacity({"reference_total_capacity": 4352})
    with pytest.raises(ValueError, match="total reference tokens"):
        validate_request(limited, (1024, 1024), [(1024, 1024)] * 2, 80)


def test_masked_groupnorm_matches_native_and_ignores_dirty_padding():
    from xhmodel_merak.xh_other_model.models.flux3_klein.vae_graph import MaskedGroupNorm

    torch.manual_seed(3)
    source = torch.nn.GroupNorm(4, 16, eps=1e-6)
    graph = MaskedGroupNorm(source)
    value = torch.randn(1, 16, 4, 6)
    padded = torch.nn.functional.pad(value, (0, 4, 0, 4))
    mask = torch.zeros(1, 1, 8, 10)
    mask[:, :, :4, :6] = 1
    with torch.no_grad():
        actual = graph(padded, mask, torch.tensor([80 / 24]))
        torch.testing.assert_close(actual[:, :, :4, :6], source(value), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(graph(padded + 1000 * (1 - mask), mask, torch.tensor([80 / 24])), actual)
        assert torch.count_nonzero(actual * (1 - mask)) == 0


@pytest.mark.parametrize("references", [0, 1, 2])
def test_complete_flow_map_dit_matches_native_and_hides_padding(monkeypatch, references):
    from xhmodel_merak.xh_other_model.models.flux3_klein import native_dit
    from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import Denoiser, denoise_inputs, image_ids

    for name, value in dict(
        HIDDEN=128, HEADS=1, DEPTH=1, DEPTH_SINGLE=2, IN_CHANNELS={"image": 128, "image_cond": 128, "text": 256}
    ).items():
        monkeypatch.setattr(native_dit, name, value)
    torch.manual_seed(23)
    source = native_dit.DiT(flow_map=True).eval()
    graph = Denoiser(source, (8, 12, 8)).eval()
    target, ref, text = torch.randn(1, 4, 128), torch.randn(1, 4 * references, 128), torch.randn(1, 5, 256)
    shapes = [(32, 32)] * references
    features = dict(image=target[0], image_cond=ref[0], text=text[0])
    p = torch.arange(5)
    ids = dict(
        image=image_ids(32, 32) * torch.tensor([1, 2, 2, 1]),
        image_cond=torch.cat([image_ids(32, 32, reference=i) for i in range(references)])
        if references
        else torch.empty(0, 4),
        text=torch.stack((p * 0, p * 0, p * 0, p), -1).float(),
    )
    # image_ids already applies the official spatial scale.
    ids["image"] = image_ids(32, 32)
    times = {name: torch.zeros(value.shape[0]) for name, value in features.items()}
    times["image"].fill_(0.8)
    packed = native_dit.Packed(features, ids, times, {name: [(0, value.shape[0])] for name, value in features.items()})
    with torch.no_grad():
        expected = source(packed, timesteps_ref=torch.tensor([0.6]))
        inputs = denoise_inputs(target, ref, text, (32, 32), shapes, (8, 12, 8), 0.8, 0.6)
        actual = graph(*inputs)[:, :4]
        torch.testing.assert_close(actual[0], expected, atol=3e-6, rtol=3e-5)
        dirty = list(inputs)
        for index, length in enumerate((4, 4 * references, 5)):
            dirty[index] = inputs[index].clone()
            dirty[index][:, length:] = 1000
        torch.testing.assert_close(graph(*dirty)[:, :4], actual, atol=0, rtol=0)
        changed = denoise_inputs(target, ref, text, (32, 32), shapes, (8, 12, 8), 0.8, 0.4)
        assert not torch.allclose(graph(*changed)[:, :4], actual)


@pytest.mark.parametrize("size", [(32, 32), (32, 64), (64, 32)])
def test_canvas_vae_matches_native_across_sizes(monkeypatch, size):
    from xhmodel_merak.xh_other_model.models.flux3_klein import native_autoencoder as native
    from xhmodel_merak.xh_other_model.models.flux3_klein.vae_graph import CanvasVAE, spatial_inputs

    monkeypatch.setattr(native, "CH", 32)
    monkeypatch.setattr(native, "CH_MULT", (1, 1, 1, 1))
    monkeypatch.setattr(native, "NUM_RES_BLOCKS", 1)
    torch.manual_seed(7)
    source = native.AutoEncoder().eval()
    height, width = size
    value = torch.randn(1, 3, height, width)
    canvas = (64, 64)
    geometry = spatial_inputs(size, canvas)
    with torch.no_grad():
        tokens, ids = source.encode(value[0])
        padded = torch.nn.functional.pad(value, (0, 64 - width, 0, 64 - height))
        actual = CanvasVAE(source, encode=True)(padded, *geometry)
        actual_tokens = actual[:, :, : height // 16, : width // 16].flatten(2).transpose(1, 2)
        torch.testing.assert_close(actual_tokens[0], tokens, atol=4e-6, rtol=4e-5)
        latent = torch.nn.functional.pad(
            tokens.reshape(height // 16, width // 16, 128).permute(2, 0, 1)[None],
            (0, (64 - width) // 16, 0, (64 - height) // 16),
        )
        actual_image = CanvasVAE(source, encode=False)(latent, *geometry)
        torch.testing.assert_close(
            actual_image[0, :, :height, :width], source.decode(tokens, ids), atol=2e-5, rtol=2e-4
        )


def test_text_features_keep_official_padding_queries():
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    from xhmodel_merak.xh_other_model.models.flux3_klein.text_graph import TextFeatures, feature_inputs

    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [12, 10, 10]},
        ),
        vision_config=dict(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=128,
            deepstack_visual_indexes=[0, 1, 2],
        ),
    )
    torch.manual_seed(4)
    source = Qwen3VLForConditionalGeneration(config).eval().model.language_model
    source.config._attn_implementation = "sdpa"
    embeddings = torch.randn(1, 8, 128)
    positions = torch.tensor([[0, 1, 2, 3, 4, 1, 1, 1]])
    with torch.no_grad():
        native = source(
            inputs_embeds=embeddings,
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        expected = torch.cat([native.hidden_states[k] for k in (1, 2, 3)], -1)
        graph = TextFeatures(source, output_layers=(1, 2, 3), max_position=128)
        inputs = feature_inputs(
            torch.nn.functional.pad(embeddings, (0, 0, 0, 4)), torch.nn.functional.pad(positions, (0, 4), value=1), 5, 8
        )
        actual = graph(*inputs)
        torch.testing.assert_close(actual[:, :8], expected, atol=2e-6, rtol=3e-5)
        assert torch.count_nonzero(actual[:, 8:]) == 0
        assert torch.count_nonzero(actual[:, 5:8]) > 0


def test_masked_norm_real_hmonnx_export(tmp_path):
    from xhmodel_merak.xh_other_model.models.flux3_klein.export import export_graph
    from xhmodel_merak.xh_other_model.models.flux3_klein.vae_graph import MaskedGroupNorm
    from xhquant.api import QuantScheme, create_quant_config
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    graph = MaskedGroupNorm(torch.nn.GroupNorm(4, 16)).eval()
    value = torch.randn(1, 16, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :6] = 1
    inputs = (value, mask, torch.tensor([64 / 24]))
    output = tmp_path / "norm.onnx"
    export_graph(
        graph,
        inputs,
        output,
        ["value", "mask", "fraction"],
        create_quant_config(QuantScheme(target_device="XH2a", quant_type="w8a8_sefp")),
        device="cuda",
    )
    session = HMONNXInferenceV2(str(output), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    actual = session(*(x.half().cuda() for x in inputs))
    torch.testing.assert_close(actual.float().cpu(), graph(*inputs), atol=0.02, rtol=0.02)


def test_recipe_declares_real_workflow_and_sampling_identity():
    from xhmodel_merak.model_build import describe_model, list_models

    assert "flux-3-image-klein" in {x["model_id"] for x in list_models()}
    meta = describe_model("flux-3-image-klein", full=True)
    assert meta["components"] == ["image"]


def test_native_bf16_timestep_embedding_rounding():
    from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import time_features
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_dit import timestep_embedding

    values = torch.tensor([1.0, 0.978453, 0.414568, 0.0], dtype=torch.bfloat16)
    torch.testing.assert_close(time_features(values, dtype=torch.bfloat16), timestep_embedding(values), atol=0, rtol=0)


def _assert_exported_precision(path):
    import onnx

    nodes = onnx.load(str(path), load_external_data=False).graph.node
    assert any(node.op_type in {"Linear", "Conv2d"} for node in nodes)
    assert any(node.op_type == "FlashAttention" for node in nodes)
    for node in nodes:
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type in {"Linear", "Conv2d"}:
            assert attributes["hmfp_weight_man_bit"] == attributes["hmfp_act_man_bit"] == 8
            assert attributes["hmfp_weight_nshare"] == attributes["hmfp_act_nshare"] == 64
            assert attributes["mode"] == b"sefp"
        elif node.op_type == "FlashAttention":
            assert all(attributes[f"{part}_bits"] == 8 for part in ("q", "k", "v", "p"))


def test_vae_512_head_padding_real_hmonnx(tmp_path):
    from xhmodel_merak.xh_other_model.models.flux3_klein.export import export_graph
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_autoencoder import AttnBlock
    from xhmodel_merak.xh_other_model.models.flux3_klein.vae_graph import Attention, spatial_inputs
    from xhquant.api import QuantScheme, create_quant_config
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    precision = "w8a8_sefp"
    torch.manual_seed(41)
    source = AttnBlock(512).eval()
    graph = Attention(source).eval()
    value = torch.randn(1, 512, 4, 4)
    geometry = spatial_inputs((32, 32), (64, 64))
    mask, fraction, order, restore, length = geometry[3], *geometry[5:]
    inputs = (torch.nn.functional.pad(value, (0, 4, 0, 4)), mask, fraction, order, restore, length)
    with torch.no_grad():
        expected = source(value)
        torch.testing.assert_close(graph(*inputs)[:, :, :4, :4], expected, atol=3e-6, rtol=3e-5)
    path = tmp_path / "attention.onnx"
    export_graph(
        graph,
        inputs,
        path,
        ["value", "mask", "fraction", "order", "restore", "length"],
        create_quant_config(QuantScheme(target_device="XH2a", quant_type=precision)),
        device="cuda:0",
        dynamo=False,
    )
    _assert_exported_precision(path)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    actual = session(*(x.half().cuda() if x.is_floating_point() else x.cuda() for x in inputs))
    torch.testing.assert_close(actual[:, :, :4, :4].float().cpu(), expected, atol=0.03, rtol=0.03)


def test_flow_map_real_hmonnx_export(tmp_path):
    from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import DIT_INPUT_NAMES, Denoiser, denoise_inputs
    from xhmodel_merak.xh_other_model.models.flux3_klein.export import export_graph
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_dit import DiT
    from xhquant.api import QuantScheme, create_quant_config
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    config = dict(hidden=128, heads=1, depth=1, depth_single=1, in_channels=dict(image=128, image_cond=128, text=256))
    torch.manual_seed(31)
    source = DiT(True, config).eval()
    with torch.no_grad():
        # Exercise large residuals without conflating this range regression
        # with FP16 subnormal weights from random tiny initialization.
        for name, module in source.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                name.startswith("emb_in.") or name == "txt_in" or name.endswith(("attn_out", "mlp_out"))
            ):
                module.weight.mul_(16)
        source.txt_in.weight[0, 0] = 8
    graph = Denoiser(source, (8, 8, 8)).eval()
    target, reference, text = torch.randn(1, 4, 128), torch.randn(1, 4, 128), torch.randn(1, 5, 256)
    # Legal FP16 graph input with a native projection above 65504, as observed
    # in the real checkpoint. A direct unscaled residual projection saturates.
    text[:, 0, 0] = 20000
    inputs = denoise_inputs(target, reference, text, (32, 32), [(32, 32)], (8, 8, 8), 0.8, 0.6)
    cases = [inputs, denoise_inputs(target, reference[:, :0], text, (32, 32), [], (8, 8, 8), 0.8, 0.4)]
    path = tmp_path / "dit.onnx"
    export_graph(
        graph,
        inputs,
        path,
        DIT_INPUT_NAMES,
        create_quant_config(QuantScheme(target_device="XH2a", quant_type="w8a8_sefp")),
        device="cuda:0",
    )
    _assert_exported_precision(path)
    # The legacy interpreter is an independent deployment reference for the
    # FX runtime. Native graph equivalence is checked by the CPU cases above;
    # comparing W8 outputs directly to floats would test quantization accuracy.
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference as LegacyHMONNXInference

    reference_session = LegacyHMONNXInference(str(path), exec_device="cuda:0").to("cuda:0")
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    for current in cases:
        values = [x.cuda().half() if x.is_floating_point() else x.cuda() for x in current]
        with torch.no_grad():
            assert torch.isfinite(graph(*current)).all()
            expected = reference_session(*values)
            actual = session(*values)
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        torch.testing.assert_close(actual[:, :4].float().cpu(), expected[:, :4].float().cpu(), atol=0.03, rtol=0.03)


def test_public_four_graph_build_relocate_generate_edit_and_repeat(
    tmp_path, monkeypatch, restore_model_initialization_state
):
    # One W8A8 run owns publication, geometry/reuse, Golden and repeatability.
    precision = "w8a8_sefp"
    import json
    import shutil
    from dataclasses import replace

    import onnx
    from _flux3_klein_fixture import prepare_source

    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_catalog.registry import get_catalog
    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime, read_metadata

    source = tmp_path / "source"
    with restore_model_initialization_state():
        prepare_source(source, monkeypatch)
        catalog = get_catalog()
        dual_recipe = catalog._recipes["flux-3-image-klein"]
        single_replacements = dict(dual_recipe.config_replacements)
        single_replacements.update({"export.max_references": 1, "export.reference_total_capacity": 16})
        monkeypatch.setitem(
            catalog._recipes,
            dual_recipe.model_id,
            replace(dual_recipe, config_replacements=tuple(single_replacements.items())),
        )
        previous = build_model(
            model_id="flux-3-image-klein",
            model_dir=str(source),
            output_dir=str(tmp_path / "single"),
            device="cuda:0",
            deployment_precision={"image": precision},
        )
        monkeypatch.setitem(catalog._recipes, dual_recipe.model_id, dual_recipe)
        from xhmodel_merak.xh_other_model.models.flux3_klein import export

        exported = []
        original_export = export.export_graph

        def observe_export(module, inputs, path, *args, **kwargs):
            exported.append(path.name)
            return original_export(module, inputs, path, *args, **kwargs)

        monkeypatch.setattr(export, "export_graph", observe_export)
        result = build_model(
            model_id="flux-3-image-klein",
            model_dir=str(source),
            output_dir=str(tmp_path / "built"),
            device="cuda:0",
            deployment_precision={"image": precision},
            sources={"image_bundle": previous.output_dir},
        )
        assert exported == ["flux3_klein_denoise.onnx"]
        old_meta, new_meta = read_metadata(previous.output_dir), read_metadata(result.output_dir)
        assert old_meta["capacity"]["max_references"] == 1
        assert new_meta["capacity"]["max_references"] == 2
        from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import reuse_image_bundle

        incompatible = {
            **new_meta["capacity"],
            "target_device": "XH2a",
            "model": {"quant_scheme": {"target_device": "XH2a", "quant_type": precision}},
            "text_capacity": new_meta["capacity"]["text_capacity"] + 80,
        }
        with pytest.raises(ValueError, match="capacity/precision"):
            reuse_image_bundle(previous.output_dir, tmp_path / "incompatible", incompatible, model_dir=source)
        assert not (tmp_path / "incompatible").exists()
        from pathlib import Path

        for role in ("text_encode", "vae_encode", "vae_decode"):
            old_path = Path(previous.output_dir) / old_meta["graphs"][role]
            new_path = Path(result.output_dir) / new_meta["graphs"][role]
            assert old_path.read_bytes() == new_path.read_bytes()
        # Raising only the count keeps all four physical graphs unchanged.
        replacements = dict(dual_recipe.config_replacements)
        replacements["export.max_references"] = 8
        monkeypatch.setitem(
            catalog._recipes,
            dual_recipe.model_id,
            replace(dual_recipe, config_replacements=tuple(replacements.items())),
        )
        exported.clear()
        expanded = build_model(
            model_id="flux-3-image-klein",
            model_dir=str(source),
            output_dir=str(tmp_path / "adaptive"),
            device="cuda:0",
            deployment_precision={"image": precision},
            sources={"image_bundle": result.output_dir},
        )
        assert expanded.loadable and exported == []
        expanded_meta = read_metadata(expanded.output_dir)
        assert expanded_meta["capacity"]["max_references"] == 8
        assert expanded_meta["capacity"]["joint_capacity"] == new_meta["capacity"]["joint_capacity"]
        for role, relative in new_meta["graphs"].items():
            assert (Path(result.output_dir) / relative).read_bytes() == (
                Path(expanded.output_dir) / expanded_meta["graphs"][role]
            ).read_bytes()
        result = expanded
    assert result.loadable
    moved = tmp_path / "relocated"
    from pathlib import Path

    Path(result.output_dir).rename(moved)
    shutil.rmtree(source)
    graphs = list(moved.rglob("*.onnx"))
    assert len(graphs) == 4
    for path in graphs:
        model = onnx.load(str(path), load_external_data=False)
        assert any(node.op_type == "FlashAttention" for node in model.graph.node)
        assert not any(node.op_type in {"Softmax", "BfpFlashAttention", "KVcache"} for node in model.graph.node)
    meta = json.loads((moved / "runtime_meta_info.json").read_text())
    assert meta["quant_type"] == precision
    metadata_path = moved / "runtime_meta_info.json"
    original = metadata_path.read_text()
    try:
        for field, value in [
            ("quant_type", "w16a16_sefp"),
            (
                "capacity",
                {
                    **meta["capacity"],
                    "text_capacity": meta["capacity"]["text_capacity"] + 80,
                    "joint_capacity": meta["capacity"]["joint_capacity"] + 80,
                },
            ),
            (
                "capacity",
                {
                    **meta["capacity"],
                    "reference_total_capacity": meta["capacity"]["reference_total_capacity"] + 16,
                    "joint_capacity": meta["capacity"]["joint_capacity"] + 16,
                },
            ),
        ]:
            metadata_path.write_text(json.dumps({**meta, field: value}))
            with pytest.raises(ValueError):
                read_metadata(moved)
    finally:
        metadata_path.write_text(original)
    inputs = dict(prompt="A red teapot rests on a table beside a window.", steps=2, seed=9)
    output = inference_model(
        artifact_dir=str(moved), expected_model_id="flux-3-image-klein", device="cuda:0", inputs=inputs
    )
    assert output.image.shape == (32, 64, 3) and output.stats["steps"] == 2
    runtime = load_runtime(moved)
    image, initial = runtime.generate(**inputs)
    assert image.size == (64, 32)
    edited, _ = runtime.generate(**inputs, images=[Image.new("RGB", (59, 43), "blue")])
    assert edited.size == (59, 43)
    multi, _ = runtime.generate(
        **inputs, images=[Image.new("RGB", (48, 32), "red"), Image.new("RGB", (32, 48), "blue")]
    )
    assert multi.size == (48, 32)
    runtime.generate(**inputs, images=[Image.new("RGB", (64, 64), (i * 20, 30, 200)) for i in range(8)])
    geometry = runtime.last_geometry
    assert len(geometry["references_hw"]) == 8
    assert sum(h * w // 256 for h, w in geometry["references_hw"]) <= meta["capacity"]["reference_total_capacity"]
    multi_inputs = dict(inputs, images=[str(tmp_path / "multi.png")] * 8)
    Image.new("RGB", (64, 64), "blue").save(tmp_path / "multi.png")
    public_multi = inference_model(artifact_dir=str(moved), device="cuda:0", inputs=multi_inputs)
    assert public_multi.stats["reference_count"] == 8
    assert len(public_multi.stats["geometry"]["references_hw"]) == 8
    with pytest.raises(ValueError, match="reference count"):
        runtime.generate(**inputs, images=[Image.new("RGB", (32, 32))] * 9)
    runtime.generate(**inputs, height=64, width=32)
    again, repeated = runtime.generate(**inputs)
    torch.testing.assert_close(initial, repeated, atol=0, rtol=0)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(image))
    from xhmodel_merak.xh_other_model.models.flux3_klein.golden import dump_golden

    reference_path = tmp_path / "reference.png"
    edited.save(reference_path)
    report_path = dump_golden(
        moved,
        inputs={**inputs, "steps": 1, "height": 32, "width": 64, "images": [str(reference_path)] * 8},
    )
    report = json.loads(Path(report_path).read_text())
    assert report["complete"] and report["real_reference"]
    assert len(report["cases"]) == 2
    assert len(report["cases"][1]["geometry"]["references_hw"]) == 8
    for path in graphs:
        for step in (0, 1):
            dumps = list((path.parent / f"step_{step}").rglob("*.npy"))
            assert len(dumps) > 10
    assert not any(path.is_symlink() for path in moved.rglob("*"))
    after_golden = inference_model(artifact_dir=str(moved), device="cuda:0", inputs=inputs)
    np.testing.assert_array_equal(after_golden.image, output.image)
