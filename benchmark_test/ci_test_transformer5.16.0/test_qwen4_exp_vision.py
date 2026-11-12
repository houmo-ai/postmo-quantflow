"""Small real Qwen4 visual rewrites, native positions and HMONNX export."""

from __future__ import annotations

import copy
import json

import onnx
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import (
    build_visual_token_gear_inputs,
    pad_flattened_patches,
)
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig
from xhmodel_merak.xh_llm.models.qwen4_exp.multimodal import (
    image_rope_positions,
    merge_image_embeddings,
    prepare_image_prompt,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.vision import (
    XHQwen4ExpVisionModel,
    export_qwen4_exp_visual,
    load_qwen4_exp_vision,
)


def _tiny_checkpoint(path, *, head_dim=8, depth=1):
    torch.manual_seed(37)
    config = Qwen4ExpVisionConfig(
        depth=depth,
        hidden_size=2 * head_dim,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    native = Qwen4ExpVisionModel(config).eval()
    path.mkdir()
    Qwen4ExpConfig(vision_config=config.to_dict()).save_pretrained(path)
    save_file(
        {"model.visual." + name: value for name, value in native.state_dict().items()}, path / "model.safetensors"
    )
    return native


@pytest.mark.parametrize("head_dim", [8, 72])
@pytest.mark.parametrize("height,width,capacity", [(2, 4, 4), (4, 4, 4), (2, 8, 8)])
def test_visual_wrapper_preserves_native_patch_bias_positions_and_padded_attention(
    tmp_path, head_dim, height, width, capacity
):
    native = _tiny_checkpoint(tmp_path / "source", head_dim=head_dim)
    model = XHQwen4ExpVisionModel(
        XHQwen3_5_VisualConfig(
            model_name="tiny_visual",
            hf_model=str(tmp_path / "source"),
            dtype="float32",
            visual_input_mode="patches",
            image_token_gears=[capacity],
            image_token_capacity=capacity,
            spatial_merge_size=2,
            patch_size=2,
            temporal_patch_size=2,
        )
    )
    # Rewriting the native Qwen4 classes exercises the actual export wrappers,
    # including 72->128 head padding used by Flash Next's real visual tower.
    wrapped = model.init_wrap_model(copy.deepcopy(native))
    from xhmodel_merak.xh_llm.models.qwen4_exp.vision_export import (
        Qwen4VisionBlock,
        Qwen4VisionMerger,
        Qwen4VisionStem,
    )

    source = copy.deepcopy(native)
    stem = Qwen4VisionStem(source, model.config)
    blocks = [Qwen4VisionBlock(block, model.config) for block in source.blocks]
    merger = Qwen4VisionMerger(source.merger, model.config)
    pixels = torch.randn(height * width, 24)
    grid = torch.tensor([[1, height, width]])
    geometry = build_visual_token_gear_inputs(
        grid,
        patch_capacity=capacity * 4,
        num_position_embeddings=16,
        spatial_merge_size=2,
        dtype=torch.float32,
    )
    padded, valid = pad_flattened_patches(pixels, capacity * 4)
    with torch.inference_mode():
        reference = native(pixels, grid).pooler_output
        actual = wrapped(
            padded,
            geometry["position_ids"],
            geometry["position_weights"],
            geometry["rotary_position_ids"],
            geometry["attention_mask"],
        )
        component_hidden, cos, sin = stem(
            padded,
            geometry["position_ids"].int(),
            geometry["position_weights"],
            geometry["rotary_position_ids"].int(),
        )
        for block in blocks:
            component_hidden = block(component_hidden, cos, sin, geometry["attention_mask"])
        component_output = merger(component_hidden)
    assert valid == height * width
    torch.testing.assert_close(actual[0, : reference.shape[0]], reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(component_output, actual, atol=0, rtol=0)


def test_multimodal_positions_keep_image_height_width_and_continuation_offset():
    config = Qwen4ExpConfig(image_token_id=90, video_token_id=91, vision_config={"spatial_merge_size": 2})
    # Text prefix, a 2x3 merged grid, text suffix. Temporal/height/width
    # coordinates differ; flattening them to sequential IDs changes RoPE.
    ids = torch.tensor([[3, 4, 90, 90, 90, 90, 90, 90, 5]])
    positions, delta = image_rope_positions(config, ids, torch.tensor([[1, 4, 6]]))
    assert positions.tolist() == [
        [[0, 1, 2, 2, 2, 2, 2, 2, 5]],
        [[0, 1, 2, 2, 2, 3, 3, 3, 5]],
        [[0, 1, 2, 3, 4, 2, 3, 4, 5]],
    ]
    assert delta.tolist() == [[-3]]
    with pytest.raises(ValueError, match="placeholder"):
        image_rope_positions(config, ids, torch.tensor([[1, 2, 2]]))


def test_image_embedding_insertion_preserves_text_and_rejects_mismatched_features():
    embedding = torch.nn.Embedding(10, 4)
    ids = torch.tensor([[1, 8, 8, 2]])
    visual = torch.arange(8).float().reshape(1, 2, 4)
    actual = merge_image_embeddings(ids, embedding, [visual], 8)
    torch.testing.assert_close(actual[:, 1:3], visual)
    torch.testing.assert_close(actual[:, [0, 3]], embedding(ids[:, [0, 3]]))
    assert ids.tolist() == [[1, 8, 8, 2]]
    with pytest.raises(ValueError, match="do not match"):
        merge_image_embeddings(ids, embedding, [visual[:, :1]], 8)


def test_native_image_processor_preserves_normalization_and_respects_exported_capacity(tmp_path):
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = {
        "<unk>": 0,
        "<|image_pad|>": 1,
        "<|video_pad|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "describe": 5,
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        additional_special_tokens=list(vocab)[1:5],
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=256,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        tokenizer=tokenizer,
        chat_template="<|vision_start|><|image_pad|><|vision_end|>describe",
    )
    processor.save_pretrained(tmp_path)
    Qwen4ExpConfig(
        image_token_id=1,
        video_token_id=2,
        vision_config={"patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2},
    ).save_pretrained(tmp_path)
    inputs = prepare_image_prompt(tmp_path, "describe", Image.new("RGB", (8, 16), "white"), max_image_tokens=4)
    assert inputs["pixel_values"].shape[1] == 24
    assert inputs["pixel_values"].shape[0] <= 16
    torch.testing.assert_close(inputs["pixel_values"], torch.ones_like(inputs["pixel_values"]))
    assert inputs["position_ids"].shape == (3, 1, inputs["input_ids"].shape[1])
    assert (inputs["input_ids"] == 1).sum() == inputs["image_grid_thw"].prod() // 4


def test_real_visual_component_export_cpu_matches_native_and_flat_gears(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp import vision_export

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    native = _tiny_checkpoint(tmp_path / "source")
    devices = []
    original_ptq = vision_export.ptq_quantize

    def tracked_ptq(graph, data, precision, execution_devices, **kwargs):
        devices.append([torch.device(device).type for device in execution_devices])
        return original_ptq(graph, data, precision, execution_devices, **kwargs)

    monkeypatch.setattr(vision_export, "ptq_quantize", tracked_ptq)
    result = export_qwen4_exp_visual(
        str(tmp_path / "source"),
        tmp_path / "export",
        image_token_gears=[2, 4],
        quant_scheme={"quant_type": "w16a16h1_sefp"},
        dtype="float32",
    )
    assert result["image_token_gears"] == [2, 4]
    assert result["patch_size"] == 2
    assert result["num_heads"] == 2
    assert result["hidden_size"] == 16
    assert result["abi_version"] == 2
    assert result["layout"] == "components"
    assert result["hmonnx"] is None
    assert devices == [["cpu"]] * 6
    assert result["input_names"] == [
        "pixel_values",
        "position_ids",
        "position_weights",
        "rotary_position_ids",
        "attention_mask",
    ]
    manifest = json.loads((tmp_path / "export" / "visual_gears.json").read_text())
    assert manifest["routing_policy"] == "smallest_fit"
    assert manifest["gears"] == result["gears"]
    for gear in result["gears"]:
        assert [part["name"] for part in gear["components"]] == ["stem", "block_00", "merger"]
        for part in gear["components"]:
            graph_path = tmp_path / "export" / part["hmonnx"]
            assert graph_path.parent.name == f"visual_m{gear['image_token_capacity']}"
            graph = onnx.load(graph_path)
            assert [value.name for value in graph.graph.input] == part["input_names"]
            assert [value.name for value in graph.graph.output] == part["output_names"]
            onnx.checker.check_model(str(graph_path))
            if part["name"].startswith("block_"):
                assert sum("attention_mask" in node.input for node in graph.graph.node) == 2
            assert not any(node.op_type in {"Conv", "Conv3D"} for node in graph.graph.node)

    encoder = load_qwen4_exp_vision(result, tmp_path / "export", device="cpu")
    assert encoder.max_image_tokens == 4
    pixels = [torch.randn(8, 24), torch.randn(16, 24)]
    grids = torch.tensor([[1, 2, 4], [1, 4, 4]])
    with torch.inference_mode():
        actual = encoder.encode_many(pixels, grids)
        expected = [native(value, grid.unsqueeze(0)).pooler_output for value, grid in zip(pixels, grids, strict=True)]
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output.squeeze(0).float(), reference, atol=2e-3, rtol=2e-2)
    video = encoder.encode_video(pixels[0].repeat(2, 1), torch.tensor([[2, 2, 4]]))
    torch.testing.assert_close(video, actual[0].squeeze(0).repeat(2, 1), atol=0, rtol=0)
    assert not list((tmp_path / "export").rglob("step_*"))

    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture

    capture = GoldenCapture(tmp_path / "capture", ["visual"])
    encoder.capture_all(capture)
    assert encoder.capture is None
    assert len(capture.finish()["visual"]) == 6
    assert {step["graph"] for step in capture.steps} == {
        f"visual_m{gear}_{name}" for gear in [2, 4] for name in ["stem", "block_00", "merger"]
    }


def test_visual_component_loading_is_bounded_and_failure_releases_native_parameters(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp import vision_export

    native = _tiny_checkpoint(tmp_path / "source", depth=3)
    largest = max(
        sum(value.numel() for value in module.parameters())
        for module in [native.patch_embed, native.pos_embed, *native.blocks, native.merger]
    )
    stem_size = sum(value.numel() for module in [native.patch_embed, native.pos_embed] for value in module.parameters())
    shells, resident_counts, component_names = [], [], []
    original_class = vision_export.Qwen4ExpVisionModel
    original_load = vision_export.Qwen4VisionCheckpoint.load_module

    def record_shell(config):
        shell = original_class(config)
        shells.append(shell)
        return shell

    def record_load(reader, module, prefix):
        original_load(reader, module, prefix)
        resident_counts.append(sum(value.numel() for value in shells[-1].parameters() if not value.is_meta))

    def fail_after_loading(wrapped, inputs, path, **kwargs):
        component_names.append(path.stem)
        if path.stem == "merger":
            raise RuntimeError("injected component export failure")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"component boundary placeholder")
        return path

    monkeypatch.setattr(vision_export, "Qwen4ExpVisionModel", record_shell)
    monkeypatch.setattr(vision_export.Qwen4VisionCheckpoint, "load_module", record_load)
    monkeypatch.setattr(vision_export, "_export_component", fail_after_loading)
    with pytest.raises(RuntimeError, match="injected component export failure"):
        export_qwen4_exp_visual(
            tmp_path / "source",
            tmp_path / "export",
            image_token_gears=[2, 4],
            quant_scheme={"quant_type": "w16a16h1_sefp"},
            dtype="float32",
        )
    assert component_names == ["stem", "block_00", "block_01", "block_02", "merger"]
    assert max(resident_counts) == max(largest, stem_size)
    assert all(value.is_meta for shell in shells for value in shell.parameters())
    assert not (tmp_path / "export" / "visual_gears.json").exists()


def test_legacy_whole_gear_loader_preserves_sdk_execution_and_request_capture(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture
    from xhmodel_merak.xh_llm.models.qwen4_exp.vision_export import _export_component

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    native = _tiny_checkpoint(tmp_path / "source")
    model = XHQwen4ExpVisionModel(
        XHQwen3_5_VisualConfig(
            hf_model=str(tmp_path / "source"),
            model_name="legacy_visual",
            dtype="float32",
            visual_input_mode="patches",
            image_token_gears=[2],
            image_token_capacity=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            quant_scheme={"quant_type": "w16a16h1_sefp"},
        )
    )
    grid = torch.tensor([[1, 2, 4]])
    geometry = build_visual_token_gear_inputs(
        grid,
        patch_capacity=8,
        num_position_embeddings=16,
        spatial_merge_size=2,
        dtype=torch.float32,
        rotary_cache_length=model.config.visual_rope_cache_length,
    )
    pixels = torch.randn(8, 24)
    names = ["pixel_values", "position_ids", "position_weights", "rotary_position_ids", "attention_mask"]
    inputs = [pixels.unsqueeze(0), *[geometry[name] for name in names[1:]]]
    path = tmp_path / "legacy" / "visual_m2" / "legacy.onnx"
    _export_component(
        model.init_wrap_model(copy.deepcopy(native)),
        inputs,
        path,
        input_names=names,
        output_names=["image_embeds"],
        quant_scheme=model.get_quant_cfg(),
        chip_arch="XH2a",
        device="cpu",
    )
    metadata = model.create_export_metadata(str(tmp_path / "legacy")).to_dict()
    metadata["gears"] = [{"image_token_capacity": 2, "patch_token_capacity": 8, "hmonnx": "visual_m2/legacy.onnx"}]
    encoder = load_qwen4_exp_vision(metadata, tmp_path / "legacy", device="cpu")
    with torch.inference_mode():
        actual = encoder.encode(pixels, grid)
        expected = native(pixels, grid).pooler_output
    torch.testing.assert_close(actual.squeeze(0).float(), expected, atol=2e-3, rtol=2e-2)
    assert not list((tmp_path / "legacy").rglob("step_*"))
    capture = GoldenCapture(tmp_path / "capture", ["visual"])
    encoder.capture_all(capture)
    assert len(capture.finish()["visual"]) == 1
    assert capture.steps[0]["graph"] == "visual_m2"


def test_standard_model_visual_export_translates_legacy_geometry_and_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel

    source = tmp_path / "source"
    _tiny_checkpoint(source)
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    checkpoint_config = json.loads((source / "config.json").read_text())
    checkpoint_config["text_config"] = _tiny_stateful_model().config.to_dict()
    (source / "config.json").write_text(json.dumps(checkpoint_config))
    config = XHQwen4ExpModelConfig(
        hf_model=str(source),
        model_name="tiny_qwen4",
        dtype="float32",
        visual_config={
            "enable": True,
            "max_size_w": 8,
            "max_size_h": 8,
            "quant_scheme": {"quant_type": "w16a16h1_sefp"},
            "dtype": "float32",
        },
    )
    assert config.visual_config.image_token_gears == [4]
    assert config.visual_config.patch_size == 2
    assert config.visual_config.spatial_merge_size == 2
    model = XHQwen4ExpModel(config)
    output = tmp_path / "export"
    monkeypatch.chdir(tmp_path)
    info = SimpleNamespace(exported_dir="export", meta=SimpleNamespace())
    model._export_visual_hmonnx_impl(info)
    visual = info.meta.qwen4_exp_vision
    assert visual.gear_manifest == "visual_gears.json"
    assert visual.hmonnx is None
    assert not (output / "visual_meta_info.json").exists()
    assert (output / visual.gear_manifest).is_file()
    for gear in visual.gears:
        for part in gear["components"]:
            graph_path = output / part["hmonnx"]
            assert graph_path.is_file()
            onnx.checker.check_model(str(graph_path))


@pytest.mark.parametrize(
    "geometry",
    [
        {"max_size_w": 8},
        {"max_size_w": 7, "max_size_h": 8},
        {"max_size_w": 0, "max_size_h": 8},
    ],
)
def test_legacy_visual_geometry_rejects_incomplete_or_unaligned_dimensions(tmp_path, geometry):
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    source = tmp_path / "source"
    _tiny_checkpoint(source)
    with pytest.raises(ValueError, match="legacy Qwen4 visual"):
        XHQwen4ExpModelConfig(hf_model=str(source), model_name="tiny_qwen4", visual_config={"enable": True, **geometry})


def test_disabled_visual_config_does_not_require_gears_or_checkpoint():
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    config = XHQwen4ExpModelConfig(
        hf_model="/unused/checkpoint",
        model_name="text_only_qwen4",
        visual_config={"enable": False},
    )
    assert config.visual_config is None


def test_declared_export_quant_config_isolated_from_shared_scheme_defaults_and_mutation():
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    declared = {"quant_type": "w8a8h1_sefp", "ops": {"projection": {"quant_type": "w16a16h0_sefp"}}}
    expected = copy.deepcopy(declared)
    config = XHQwen4ExpModelConfig(
        hf_model="/unused/checkpoint",
        model_name="qwen4_quant",
        quant_scheme=declared,
        visual_config={"enable": False},
    )
    assert config._export_quant_config == expected
    assert "target_device" not in config._export_quant_config
    declared["ops"]["projection"]["quant_type"] = "w4a8h1_sefp"
    assert config._export_quant_config == expected
    config.quant_scheme.target_device = "another_device"
    assert config._export_quant_config == expected
