# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public native H3 registration and non-mutating adapter view selection."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_sol_h3_capacity import components as components

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.xh_other_model.models.sol_h3.capacity import plan_capacities
from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import mode_metadata
from xhmodel_merak.xh_other_model.models.sol_h3.native_build import validate_native_config
from xhmodel_merak.xh_other_model.models.sol_h3.native_schedule import native_schedule


def test_public_native_h3_plan_needs_no_ltx_sources(tmp_path, components):
    source = tmp_path / "source"
    for component in ("transformer", "transformer_ref", "vae", "audio_vae", "text_encoder", "processor", "tokenizer"):
        directory = source / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
    (source / "config.json").write_text((source / "text_encoder/config.json").read_text())
    recipe = get_recipe("minimax-h3")
    assert recipe.family == "h3"
    assert recipe.artifact_format == "native_h3"
    assert recipe.get_variant("lora").requires_lora
    assert recipe.profiles == ("all-native-768p-5s",)
    assert [variant.name for variant in recipe.variants] == ["full", "lora"]
    plan = resolve_build_plan(model_id="minimax-h3", model_dir=str(source), output_dir=str(tmp_path / "output"))
    assert plan.variant == "full"
    assert plan.profile == "all-native-768p-5s"
    config = yaml.safe_load(open(plan.config_path))
    assert config["export"]["model"]["type"] == "XHNativeH3Model"
    assert config["export"]["native_sampling"]["steps"] == 50
    assert config["export"]["mode"] == "all"
    assert config["export"]["draft_size"] == [768, 1344]
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "h3_adapter.json").write_text("{}")
    adapted = resolve_build_plan(
        model_id="minimax-h3",
        model_dir=str(source),
        output_dir=str(tmp_path / "adapted"),
        lora={"paths": [str(adapter)]},
    )
    assert adapted.variant == "lora"
    assert adapted.config_overrides["export.model.lora.path"] == [str(adapter)]
    assert adapted.profile == plan.profile and adapted.config_path == plan.config_path
    with pytest.raises(ModelError, match="profile"):
        resolve_build_plan(
            model_id="minimax-h3", model_dir=str(source),
            output_dir=str(tmp_path / "rejected"), profile="i2va-native-384p-5s",
        )
    assert get_recipe("minimax-h3-fl2va").family == "minimax-h3"

    # Exercise the three distinct media budgets retained as direct workflows.
    directory = Path(plan.config_path).parent
    for filename, kind in (
        ("h3_native_t2va_384p_w4.yaml", "text"),
        ("h3_native_ref2va_384p_image384_w4.yaml", "image"),
        ("h3_native_ref2va_768p_video2s_w4.yaml", "video"),
    ):
        cfg = validate_native_config(yaml.safe_load((directory / filename).read_text())["export"])
        budget = plan_capacities(cfg, **components)
        assert budget["dit_capacity"] % 256 == 0
        if kind == "text":
            assert cfg["mode"] == "t2va" and cfg["draft_size"] == [384, 672]
            assert budget["dit_capacity"] == 11008 and budget["text_capacity"] == 1024
            assert budget["reference_video_tokens"] == budget["reference_audio_tokens"] == 0
        elif kind == "image":
            assert cfg["references"]["image"]["vae_short_edge"] == 384
            assert budget["reference_audio_tokens"] == budget["video_qwen_tokens"] == 0
            assert budget["image_qwen_tokens"] > 0
        else:
            assert cfg["draft_size"] == [768, 1344]
            assert budget["reference_audio_tokens"] == 2 * 2 * 40
            assert budget["reference_video_tokens"] > 0 and budget["image_qwen_tokens"] == 0


def metadata():
    root = dict(native_variants={}, mode_profiles={}, graphs={}, resources={})
    for name, steps, shift in (("standard", 50, 12), ("lora8", 8, 6)):
        schedule = native_schedule(steps=steps, video_shift=shift)
        variant = dict(steps=steps, h3_sigmas=schedule.video_sigmas, audio_sigmas=schedule.audio_sigmas, branches={})
        for branch in ("base", "ref"):
            mapping = {role: f"{role}_{name}_{branch}" for role in ("h3_denoiser", "h3_text_refiner")}
            variant["branches"][branch] = dict(graphs=mapping, h3_rope_config={})
            root["graphs"].update({physical: {"path": physical + ".onnx"} for physical in mapping.values()})
        root["native_variants"][name] = variant
    for mode in ("t2va", "i2va", "l2va", "fl2va", "ref2va"):
        root["mode_profiles"][mode] = dict(
            branch="ref" if mode == "ref2va" else "base", input_config=mode, capacity_plan={"text_capacity": 256}
        )
        root["resources"][mode] = mode + ".json"
    return root


@pytest.mark.parametrize("mode", ("t2va", "ref2va"))
def test_switch_base_lora_base_keeps_each_text_and_dit_binding(mode):
    meta = metadata()
    original = deepcopy(meta)
    base = mode_metadata(meta, mode)
    adapted = mode_metadata(meta, mode, "lora8")
    restored = mode_metadata(meta, mode)
    assert base == restored
    assert meta == original
    assert base["draft_steps"] == 50 and adapted["draft_steps"] == 8
    for role in ("h3_denoiser", "h3_text_refiner"):
        assert base["graphs"][role] != adapted["graphs"][role]
    adapted["capacity_plan"]["text_capacity"] = 0
    assert meta == original
    del meta["native_variants"]["lora8"]["branches"]["ref"]
    with pytest.raises(ValueError, match="does not support"):
        mode_metadata(meta, "ref2va", "lora8")
    with pytest.raises(ValueError, match="Unknown"):
        mode_metadata(meta, mode, "missing")


def encoder_metadata():
    plan = dict(
        text_capacity=256,
        video_capacity=256,
        audio_capacity=256,
        kv_capacity=512,
        qwen_visual_max_patches=256,
        qwen_visual_patch_gears=[256],
        prompt_max_tokens=128,
        reference_audio_tokens=0,
    )
    return dict(
        resources={
            k: k
            for k in (
                "text_embedding",
                "text_rope_inv_freq",
                "text_config",
                "qwen_config",
                "qwen_vision_config",
                "reference_vae_config",
                "reference_audio_config",
                "h3_audio_config",
            )
        },
        tokenizer_files=["tokenizer/config.json"],
        processor_files=["processor/config.json"],
        text_feature_divisor=16,
        text_refiner_input_divisor=16,
        capacity_plan=plan,
        mode_profiles={m: {"capacity_plan": deepcopy(plan)} for m in ("t2va", "i2va", "l2va", "fl2va", "ref2va")},
        text_capacity=256,
        prompt_max_tokens=128,
        text_prefill_chunk_length=256,
        text_cache_shape=[1, 2, 512, 64],
        text_layers=2,
        feature_graphs=["feature"],
        visual_graphs=[dict(graph="h3_visual_256", capacity=256)],
        graphs={"feature": {"text_layers": [0, 2]}, "h3_visual_256": {}, "base": {}, "ref": {}},
        draft_size=[32, 64],
        reference_vae_base=dict(shape=[1, 32, 64], graph="base"),
        reference_vae_tiles={"1x64x64": "ref"},
        reference_vae_masks={"ref": {"shape": [1, 64, 64], "levels": [[1, 1], [4, 16]]}},
        reference_vae_policy=dict(tile_size=[64, 64], overlap=[16, 16]),
        reference_audio_frames=10,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(text_feature_divisor=256),
        lambda m: m["resources"].pop("text_embedding"),
        lambda m: m["text_cache_shape"].__setitem__(2, 256),
        lambda m: m["graphs"]["feature"].update(text_layers=[1, 2]),
        lambda m: m["mode_profiles"]["ref2va"]["capacity_plan"].update(kv_capacity=256),
        lambda m: m["visual_graphs"][0].update(capacity=512),
        lambda m: m["reference_vae_masks"].clear(),
    ],
)
def test_native_encoder_contract_rejects_invalid_shared_views(mutate):
    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import _validate_encoders

    meta = encoder_metadata()
    _validate_encoders(meta)
    mutate(meta)
    with pytest.raises(ValueError, match="Native H3"):
        _validate_encoders(meta)


def test_native_adapter_resource_and_graph_ownership(tmp_path):
    import json

    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import _validate_variant

    meta = metadata()
    variant = meta["native_variants"]["lora8"]
    variant.update(sampling={"steps": 8}, adapter_resource="adapter")
    spec = dict(checkpoint_sha256="a" * 64, target_count=2, component_target_counts={"text": 1, "dit": 1})
    bindings = [
        dict(component=c, target_count=1, targets=[target])
        for c, target in (
            ("text", "token_refiner.refiner_blocks.0.attn.to_q"),
            ("dit", "transformer_blocks.0.attn.to_q"),
        )
    ]
    for branch in variant["branches"].values():
        branch.update(draft_steps=8, h3_packed_inputs=True, lora=dict(**spec, bindings=deepcopy(bindings)))
    config = dict(name="lora8", sampling=variant["sampling"], branches={b: spec for b in ("base", "ref")})
    meta["resources"]["adapter"] = "adapter.json"
    (tmp_path / "adapter.json").write_text(json.dumps(config))
    _validate_variant(tmp_path, meta, "lora8", variant)
    original = deepcopy(variant)
    variant["branches"]["ref"]["graphs"] = variant["branches"]["base"]["graphs"]
    with pytest.raises(ValueError, match="wrong adapter or weight branch"):
        _validate_variant(tmp_path, meta, "lora8", variant)
    variant = deepcopy(original)
    variant["branches"]["ref"]["lora"]["bindings"][0]["target_count"] = 2
    with pytest.raises(ValueError, match="binding counts"):
        _validate_variant(tmp_path, meta, "lora8", variant)
    variant = deepcopy(original)
    variant["branches"]["base"]["lora"]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="bound checkpoint disagree"):
        _validate_variant(tmp_path, meta, "lora8", variant)
