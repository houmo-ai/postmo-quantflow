"""Optional source selection and official prompt-upsample input contracts."""

import pytest
from PIL import Image


def test_optional_sources_are_described_and_validated():
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build import describe_model
    from xhmodel_merak.model_build.resolver import _validate_sources
    from xhmodel_merak.model_catalog import get_recipe

    recipe = get_recipe("flux-3-image-klein")
    variant = recipe.default_variant
    description = describe_model(recipe.model_id)["sources"]
    assert "prompt_upsampler" in description["optional_by_variant"]["full"]
    _validate_sources(recipe.model_id, variant, {})
    _validate_sources(recipe.model_id, variant, {"prompt_upsampler": "unused"})
    with pytest.raises(ModelError, match="unknown sources"):
        _validate_sources(recipe.model_id, variant, {"unknown": "unused"})


def test_reference_policy_and_task_messages():
    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler import policy_reference, prepare_messages

    source = Image.new("RGBA", (2048, 1024), (0, 0, 0, 0))
    image = policy_reference(source, 1)
    assert image.mode == "RGB" and image.width % 32 == image.height % 32 == 0
    assert image.width * image.height <= 1024**2
    assert image.getpixel((0, 0)) == (255, 255, 255)
    prompts = {"t2i": "generation", "i2i": "editing"}
    messages = prepare_messages("a dog", [], prompts, (768, 1376))
    assert messages[0]["content"] == "generation"
    assert messages[1]["content"].startswith("TARGET CANVAS: 1376:768\n")
    messages = prepare_messages("change the collar", [source], prompts, (768, 1376))
    assert messages[0]["content"] == "editing"
    assert messages[1]["content"][0]["type"] == "image"
    assert "User request: change the collar" in messages[1]["content"][-1]["text"]
    assert source.mode == "RGBA" and source.size == (2048, 1024)


def test_presence_penalty_only_penalizes_generated_tokens_once():
    import torch

    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler import PresencePenalty

    processor = PresencePenalty(3, 1.5)
    scores = torch.zeros(1, 8)
    actual = processor(torch.tensor([[1, 2, 3, 4, 4, 5]]), scores)
    assert actual.tolist() == [[0, 0, 0, 0, -1.5, -1.5, 0, 0]]


def test_explicit_expansion_requires_packaged_component():
    from xhmodel_merak.xh_other_model.models.flux3_klein.pipeline import ImageRuntime

    runtime = ImageRuntime({}, None, None, {}, {}, device="cpu")
    with pytest.raises(ValueError, match="upsampler"):
        runtime.generate("a dog", prompt_upsampling=True)
    assert runtime.upsampler is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optional_sources": ("pup", "pup")},
        {"optional_sources": ("pup",), "required_sources": ("pup",)},
        {"optional_sources": ("",)},
    ],
)
def test_optional_source_contract_rejects_ambiguous_names(kwargs):
    from xhmodel_merak.model_catalog.contracts import VariantRecipe

    with pytest.raises(ValueError, match="optional sources"):
        VariantRecipe("full", "unused", **kwargs)


def test_source_selection_binds_exact_paths_and_rejects_conflicts(monkeypatch):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build.contracts import SourceRef
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.model_integrations.flux3_klein.source import resolve_config
    from xhmodel_merak.xh_other_model.models.flux3_klein import source

    # This test targets selection/binding, independently of expensive weight validation.
    monkeypatch.setattr(source, "read_source", lambda _: None)
    recipe = get_recipe("flux-3-image-klein")
    sources = {"upsampler_bundle": SourceRef(path="/a/component"), "image_bundle": SourceRef(path="/b/core")}
    result = resolve_config(recipe=recipe, model_dir="unused", sources=sources)
    assert result == {"export.upsampler_bundle": "/a/component", "export.image_bundle": "/b/core"}
    with pytest.raises(ModelError, match="choose raw"):
        resolve_config(
            recipe=recipe,
            model_dir="unused",
            sources={"prompt_upsampler": SourceRef(path="/raw"), "upsampler_bundle": SourceRef(path="/component")},
        )


def test_reference_policy_rejects_unsupported_geometry():
    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler import policy_reference

    with pytest.raises(ValueError, match="aspect ratio"):
        policy_reference(Image.new("RGB", (201, 1)), 1)
    with pytest.raises(ValueError, match="one to ten"):
        policy_reference(Image.new("RGB", (32, 32)), 0)


@pytest.mark.parametrize(
    "context,input_length,maximum,expected",
    [(8192, 1000, 3584, 3584), (8192, 5000, 3584, 3192), (4096, 4000, 3584, 96), (16384, 9000, 4096, 4096)],
)
def test_generation_budget_uses_exported_context(context, input_length, maximum, expected):
    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler import generation_budget

    assert generation_budget(input_length, dict(context_length=context, max_new_tokens=maximum)) == expected
    with pytest.raises(ValueError, match="context"):
        generation_budget(context, dict(context_length=context, max_new_tokens=maximum))


def _prompt_contract_bundle(root, context=4096, prefill=128, gears=(256, 512, 1024)):
    """ONNX port/metadata fixture; never used as neural backend execution proof."""
    import json

    import onnx
    from onnx import TensorProto, helper

    hf = root / "hf_config"
    hf.mkdir()
    (hf / "config.json").write_text(
        json.dumps(
            dict(
                model_type="qwen3_5",
                text_config=dict(
                    hidden_size=4096,
                    num_key_value_heads=4,
                    head_dim=256,
                    layer_types=["linear_attention"] * 24 + ["full_attention"] * 8,
                ),
                vision_config=dict(patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2),
            )
        )
    )
    for name in ("processor_config.json", "system_prompts.json"):
        (hf / name).write_text("{}")
    (root / "quant_embedding.pt").write_bytes(b"contract embedding inventory")
    meta = dict(
        model_config=dict(context_max_length=context, prefill_chunk_length=prefill),
        hf_config="hf_config",
        quant_embedding="quant_embedding.pt",
    )
    for role, length in (("prefill", prefill), ("decode", 1)):
        folder = root / role
        folder.mkdir()
        ports = [helper.make_tensor_value_info("input_1", TensorProto.FLOAT16, [1, length, 4096])]
        ports += [
            helper.make_tensor_value_info(f"{i}_{kv}cache_input", TensorProto.FLOAT16, [1, 4, context, 256])
            for i in range(8)
            for kv in ("k", "v")
        ]
        op = "GDRChunkScan" if role == "prefill" else "GDRRecurrentScan"
        nodes = [helper.make_node(op, ["input_1"], [f"state_{i}"], domain="contract") for i in range(24)]
        nodes.append(helper.make_node("Identity", ["input_1"], ["output"]))
        graph = helper.make_graph(
            nodes, role, ports, [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, length, 4096])]
        )
        path = folder / "model.onnx"
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("contract", 1)]),
            path,
        )
        meta[f"{role}_hmonnx"] = str(path.relative_to(root))
    visual = []
    for gear in gears:
        folder = root / f"visual_m{gear}"
        folder.mkdir()
        port = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT16, [1, gear * 4, 1536])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["pixel_values"], ["output"])],
            "visual",
            [port],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, gear * 4, 1536])],
        )
        path = folder / "model.onnx"
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("contract", 1)]),
            path,
        )
        visual.append(
            dict(image_token_capacity=gear, patch_token_capacity=gear * 4, hmonnx=str(path.relative_to(root)))
        )
    meta["visual_config"] = dict(gears=visual, spatial_merge_size=2)
    (root / "upsampler_meta_info.json").write_text(json.dumps(meta))
    return meta


@pytest.mark.parametrize(
    "context,prefill,gears",
    [(4096, 128, (256, 512, 1024)), (16384, 512, (512, 1024)), (8192, 256, (64, 128, 256, 512, 1024))],
)
def test_component_derives_gears_and_lengths_from_export(tmp_path, context, prefill, gears):
    import json

    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import describe_component, read_component

    _prompt_contract_bundle(tmp_path, context, prefill, gears)
    component = describe_component(tmp_path, max_new_tokens=2048)
    assert component["context_length"] == context
    assert component["prefill_chunk_length"] == prefill
    assert set(component["graphs"]) == {"prefill", "decode", *(f"visual_m{x}" for x in gears)}
    path = tmp_path / "upsampler_component.json"
    path.write_text(json.dumps(component))
    assert read_component(tmp_path) == component
    # Internally consistent metadata forgery must still fail against physical cache ports.
    meta_file = tmp_path / "upsampler_meta_info.json"
    meta = json.loads(meta_file.read_text())
    component["context_length"] = context * 2
    meta["model_config"]["context_max_length"] = context * 2
    path.write_text(json.dumps(component))
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="KV"):
        read_component(tmp_path)


def test_multireference_messages_preserve_order_and_official_area_policy():
    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler import prepare_messages

    sources = [Image.new("RGB", (1024, 1024), color) for color in ("red", "blue", "green", "yellow")]
    messages = prepare_messages("Combine these objects.", sources, dict(t2i="t2i", i2i="i2i"), (1024, 1024))
    images = [entry["image"] for entry in messages[1]["content"] if entry["type"] == "image"]
    assert [image.size for image in images] == [(512, 512)] * 4
    assert images[0].getpixel((0, 0))[0] > 240
    assert images[1].getpixel((0, 0))[2] > 240
    assert messages[0]["content"] == "i2i"


def test_legacy_component_is_readable_and_prefill_tampering_is_rejected(tmp_path):
    import json

    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import describe_component, read_component

    _prompt_contract_bundle(tmp_path, 8192, 256, (1024,))
    component = describe_component(tmp_path)
    component["schema_version"] = 1
    component.pop("prefill_chunk_length")
    component["graphs"]["visual"] = component["graphs"].pop("visual_m1024")
    (tmp_path / "upsampler_component.json").write_text(json.dumps(component))
    assert read_component(tmp_path) == component
    p = tmp_path / "upsampler_meta_info.json"
    meta = json.loads(p.read_text())
    meta["model_config"]["prefill_chunk_length"] = 512
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="language input shape"):
        read_component(tmp_path)


def test_visual_gear_metadata_forgery_is_rejected_by_physical_input(tmp_path):
    import json

    from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import describe_component, read_component

    _prompt_contract_bundle(tmp_path)
    p = tmp_path / "upsampler_meta_info.json"
    meta = json.loads(p.read_text())
    gear = meta["visual_config"]["gears"][0]
    gear["image_token_capacity"], gear["patch_token_capacity"] = 128, 512
    p.write_text(json.dumps(meta))
    (tmp_path / "upsampler_component.json").write_text(json.dumps(describe_component(tmp_path)))
    with pytest.raises(ValueError, match="visual capacity"):
        read_component(tmp_path)


def test_public_missing_component_error_precedes_session_loading(monkeypatch, tmp_path):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.model_integrations.flux3_klein import inference
    from xhmodel_merak.xh_other_model.models.flux3_klein import bundle

    # Isolate input routing from the independently tested artifact validator.
    monkeypatch.setattr(inference, "validate_runtime", lambda *args, **kwargs: ({}, {}))

    def unexpected_load(*args, **kwargs):
        raise AssertionError("sessions must not load for a missing component")

    monkeypatch.setattr(bundle, "load_runtime", unexpected_load)
    with pytest.raises(ModelError, match="does not include a prompt upsampler"):
        inference.adapter.run(
            recipe=get_recipe("flux-3-image-klein"),
            root=tmp_path,
            manifest={},
            meta_file=tmp_path / "runtime_meta_info.json",
            inputs={"prompt": "a dog", "prompt_upsampling": True},
            device="cpu",
            components=("image",),
            selected_golden=(),
            golden_artifacts={},
        )
