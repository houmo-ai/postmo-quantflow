# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public recipe, complete inventory, source validation and independent release layout."""

import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.xh_other_model.models.sol_h3.artifacts import export_entries, read_metadata
from xhmodel_merak.xh_other_model.models.sol_h3.build import validate_build_config


@pytest.fixture
def unified_mode_metadata():
    from xhmodel_merak.xh_other_model.models.sol_h3.capacity import MODES

    metadata = dict(
        mode="all", mode_profiles={}, branches={}, graphs={"h3_feature": {"path": "text/shared.onnx"}}, resources={}
    )
    for branch in ("base", "ref"):
        aliases = {name: f"{name}_{branch}" for name in ("h3_denoiser", "h3_text_refiner")}
        metadata["branches"][branch] = dict(
            graphs=aliases,
            h3_adapter={"identity": branch},
            h3_rope_config={"theta": 10000},
            h3_role_layout={"format": "fixed_adaln_role_slots_v1", "role_capacities": [256] * 5},
            folded_h3_projection_parameters=128,
        )
        metadata["graphs"].update({name: {"path": f"{name}/graph.onnx"} for name in aliases.values()})
    for mode in MODES:
        resource = "input_config_" + mode
        metadata["resources"][resource] = f"hf_config/{resource}.json"
        metadata["mode_profiles"][mode] = dict(
            profile=mode,
            branch="ref" if mode == "ref2va" else "base",
            input_config=resource,
            capacity_plan={"text_capacity": 512 if mode == "ref2va" else 256},
        )
    return metadata


@pytest.mark.parametrize("mode", ["t2va", "ref2va"])
def test_unified_mode_views_bind_native_weights_without_mutating_shared_inventory(unified_mode_metadata, mode):
    from xhmodel_merak.xh_other_model.models.sol_h3.bundle import mode_metadata

    before = json.dumps(unified_mode_metadata, sort_keys=True)
    view = mode_metadata(unified_mode_metadata, mode)
    branch = "ref" if mode == "ref2va" else "base"
    assert view["h3_adapter"]["identity"] == branch
    assert view["h3_role_layout"] == unified_mode_metadata["branches"][branch]["h3_role_layout"]
    assert view["graphs"]["h3_denoiser"]["path"] == f"h3_denoiser_{branch}/graph.onnx"
    assert view["graphs"]["h3_text_refiner"]["path"] == f"h3_text_refiner_{branch}/graph.onnx"
    assert view["graphs"]["h3_feature"] == unified_mode_metadata["graphs"]["h3_feature"]
    assert view["resources"]["input_config"] == f"hf_config/input_config_{mode}.json"
    view["h3_adapter"]["identity"] = "changed"
    view["graphs"]["h3_feature"]["path"] = "changed.onnx"
    view["capacity_plan"]["text_capacity"] = 0
    view["h3_role_layout"]["role_capacities"][0] = 0
    assert json.dumps(unified_mode_metadata, sort_keys=True) == before


@pytest.mark.parametrize("field,size", [("image", (63, 32)), ("last_image", (64, 31))])
def test_low_resolution_keyframes_fail_before_model_loading(tmp_path, unified_mode_metadata, field, size):
    import torch
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.sol_h3.ltx_keyframes import validate_keyframe_resolution
    from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime

    rotated = Image.new("RGB", (32, 64))
    rotated.getexif()[274] = 6
    rotated.save(tmp_path / "rotated.png", exif=rotated.getexif())
    validate_keyframe_resolution(tmp_path / "rotated.png", (32, 64))
    Image.new("RGB", size).save(tmp_path / "low_resolution.png")
    runtime = object.__new__(SolH3Runtime)
    runtime.device, runtime.sessions = torch.device("cuda:0"), {}
    runtime.bundle_meta = unified_mode_metadata
    runtime.bundle_meta.update(
        schema_version=8, ltx_audio_frozen=True, refiner_size=[32, 64],
        ltx_reference_conditioning={"enabled": True, "require_native_resolution": True},
    )
    runtime.meta = runtime.bundle_meta
    runtime.close = runtime.sessions.clear
    runtime.condition = lambda *args, **kwargs: pytest.fail("conditioning loaded before keyframe validation")
    runtime.invoke = lambda *args, **kwargs: pytest.fail("model loaded before keyframe validation")
    with pytest.raises(ValueError, match="requires at least 64x32"):
        runtime.generate("A person in a cafe", **{field: str(tmp_path / "low_resolution.png")})
    assert runtime.sessions == {}


@pytest.mark.parametrize("fault", ["wrong_branch", "shared_path", "missing_graph"])
def test_unified_mode_views_reject_incomplete_or_cross_bound_branches(unified_mode_metadata, fault):
    from xhmodel_merak.xh_other_model.models.sol_h3.bundle import mode_metadata

    meta = unified_mode_metadata
    if fault == "wrong_branch":
        meta["mode_profiles"]["ref2va"]["branch"] = "base"
    elif fault == "shared_weights":
        meta["branches"]["ref"]["graphs"]["h3_denoiser"] = "h3_denoiser_base"
    elif fault == "shared_path":
        meta["graphs"]["h3_denoiser_ref"]["path"] = meta["graphs"]["h3_denoiser_base"]["path"]
    elif fault == "missing_graph":
        del meta["graphs"]["h3_denoiser_ref"]
    else:
        del meta["resources"]["input_config_ref2va"]
    with pytest.raises(ValueError):
        mode_metadata(meta, "ref2va")


@pytest.mark.parametrize("field", ["text_capacity", "dit_capacity", "root_text"])
def test_unified_metadata_rejects_root_capacity_drift(tmp_path, unified_mode_metadata, field):
    meta = unified_mode_metadata
    meta.update(
        schema_version=5, draft_size=[384, 672], draft_frames=124, refiner_frames=121, fps=24,
        profile="all-dense-384p-768p-5s", text_capacity=512,
    )
    for profile in meta["mode_profiles"].values():
        profile["capacity_plan"].update(
            video_capacity=1024, audio_capacity=256, kv_capacity=768, qwen_visual_max_patches=256
        )
    meta["capacity_plan"] = dict(
        text_capacity=512,
        video_capacity=1024,
        audio_capacity=256,
        kv_capacity=768,
        qwen_visual_max_patches=256,
        dit_capacity=1792,
    )
    if field == "root_text":
        meta["text_capacity"] += 256
    else:
        meta["capacity_plan"][field] += 256
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unified Sol-H3 root"):
        read_metadata(tmp_path)


@pytest.mark.parametrize("phase", ("build", "runtime"))
def test_fp16_input_overflow_is_rejected_before_sdk_can_saturate_it(tmp_path, phase):
    import torch

    from xhmodel_merak.xh_other_model.models.sol_h3.build import BundleExporter
    from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime

    calls = []

    def saturating_session(feed):
        calls.append(feed)
        return torch.full((1,), 65504, dtype=torch.float16)

    # SDK kernels may return a finite saturated result after receiving Inf.
    # The finite FP32 value must therefore be checked after narrowing to FP16.
    session = SimpleNamespace(
        get_input_names=lambda: ("x",),
        get_input=lambda _: SimpleNamespace(dtype=torch.float16),
        run=saturating_session,
    )
    kind = BundleExporter if phase == "build" else SolH3Runtime
    runner = object.__new__(kind)
    runner.root, runner.device = tmp_path, torch.device("cpu")
    runner.branch = None
    runner.meta = {"graphs": {"fixture": {"path": "unused.onnx", "input_names": ["x"]}}}
    runner.sessions = {"fixture": session}
    large = torch.tensor([70000.0])
    with pytest.raises(FloatingPointError, match="FP16"):
        if phase == "build":
            runner.infer("fixture", (large,))
        else:
            runner.invoke("fixture", large)
    assert not calls


def source_fixture(root):
    paths = {}
    for name in (
        "dense_lora",
        "transformer",
        "refiner_lora",
        "output_video_vae",
        "audio_vae",
        "h3_upscaler_checkpoint",
        "upscaler_source",
        "refiner_prompt_cache",
    ):
        target = root / name
        target.write_bytes(b"source-inventory-contract")
        paths[name] = str(target)
    for name in ("h3_model", "adapter_dir", "sana_source", "ltx_source"):
        target = root / name
        target.mkdir()
        paths[name] = str(target)
    for name in (
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "audio_vae/config.json",
    ):
        target = Path(paths["h3_model"]) / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("{}")
    (Path(paths["adapter_dir"]) / "model.safetensors").touch()
    (root / "config.json").write_bytes((Path(paths["h3_model"]) / "text_encoder/config.json").read_bytes())
    (root / "checkpoint-paths.json").write_text(json.dumps(paths))


def bundle_fixture(root):
    from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import AUDIO_SIGMAS, REFINER_SIGMAS, VIDEO_SIGMAS

    names = [
        "h3_text_refiner",
        "h3_denoiser",
        "h3_to_ltx",
        "h3_audio_decode",
        "ltx_audio_encode",
        "ltx_denoiser",
        "h3_feature_00_50",
        "h3_upscale_t42",
        "ltx_video_decode_16x14x24",
    ]
    graphs = {}
    for name in names:
        directory = "text_encode_h3_s512" if name.startswith("h3_feature") else "denoise_" + name
        path = root / directory / f"{name}.onnx"
        path.parent.mkdir(exist_ok=True)
        x, y = (helper.make_tensor_value_info(value, TensorProto.FLOAT16, [1, 8]) for value in ("x", "y"))
        onnx.save(
            helper.make_model(helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], name, [x], [y])), path
        )
        graphs[name] = dict(path=path.relative_to(root).as_posix(), input_names=["x"], output_names=["y"])
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer_config.json").write_text("{}")
    resource_file = root / "fixture_resource.json"
    resource_file.write_text("{}")
    resources = {
        name: resource_file.name
        for name in (
            "text_config",
            "text_embedding",
            "text_rope_inv_freq",
            "h3_audio_config",
            "ltx_audio_frontend",
            "upscaler_statistics",
            "refiner_prompt_recipe",
            "ltx_rope",
            "decoder_masks",
        )
    }
    import numpy as np

    for name in ("text_embedding", "text_rope_inv_freq"):
        np.save(root / f"{name}.npy", np.zeros((1, 8), dtype=np.float16))
        resources[name] = f"{name}.npy"
    for name in ("ltx_rope", "decoder_masks"):
        np.savez(root / f"{name}.npz", value=np.zeros(1, dtype=np.float16))
        resources[name] = f"{name}.npz"
    meta = dict(
        schema_version=3,
        kind="sol_h3",
        complete=True,
        target_device="XH2a",
        input_dtype="float16",
        attention_value_divisor=512.0,
        scaled_rms_policy="plain_wide_reduction",
        h3_precision={"residual_scale": 1024.0, "rms_norm_mode": "plain", "rms_norm_eps": 2.0**-24},
        quant_type="w8a8_sefp",
        profile="dense-384p-768p-5s",
        draft_steps=4,
        refiner_steps=3,
        draft_size=[384, 672],
        refiner_size=[768, 1344],
        draft_frames=124,
        refiner_frames=121,
        fps=24,
        graphs=graphs,
        resources=resources,
        text_capacity=512,
        h3_sigmas=VIDEO_SIGMAS,
        audio_sigmas=AUDIO_SIGMAS,
        refiner_sigmas=REFINER_SIGMAS,
        feature_graphs=["h3_feature_00_50"],
        upscaler_graphs=["h3_upscale_t42"],
        video_decoder_graphs=["ltx_video_decode_16x14x24"],
        tokenizer_files=["tokenizer/tokenizer_config.json"],
        refiner_cache_policy="per_sigma_exact_fixed_prompt",
        refiner_prompt_fingerprint="test-recipe-fingerprint",
    )
    (root / "runtime_meta_info.json").write_text(json.dumps(meta))
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(meta)))
    return meta


def test_independent_public_plan_materializes_sefp8_static_workflow(tmp_path):
    from xhmodel_merak.model_build import describe_model
    from xhmodel_merak.model_inference.inputs import validate_inputs

    source = tmp_path / "source"
    source.mkdir()
    source_fixture(source)
    plan = resolve_build_plan(
        model_id="sol-h3", model_dir=str(source), output_dir=str(tmp_path / "artifact"), device="cuda:0"
    )
    assert plan.components == ("video",)
    assert plan.deployment_precision == {"video": "w8a8_sefp"}
    assert (plan.precision.weight, plan.precision.activation) == ("w4", "a8")
    assert plan.quant_method == "autoround"
    assert {
        "gptqmodel.recipes:quantize_qwen3_vl",
        "auto_round:AutoRound",
        "auto_round.data_type.int:quant_tensor_sym_xh",
    } <= set(plan.worker_capabilities)
    assert "gptqmodel==5.8.0" in plan.worker_requirements
    assert "torchao==0.17.0" in plan.worker_requirements
    assert "auto-round==0.12.0" not in plan.worker_requirements
    assert plan.artifact_name == "hmquant_xh2_sol_h3_w8a8_384p_768p"
    materialize_workflow_config(plan, tmp_path / "workflow.yaml")
    data = yaml.safe_load((tmp_path / "workflow.yaml").read_text())
    assert data["export"]["model"]["type"] == "XHSolH3Model"
    assert data["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8_sefp"
    config = validate_build_config(data["export"])
    assert config["refiner_frames"] == 121
    assert config["ltx_reference_conditioning"]["require_native_resolution"] is True
    assert config["h3_precision"]["residual_scale"] == 1024
    assert config["h3_precision"]["rms_norm_mode"] == "plain"
    assert config["h3_precision"]["rms_norm_eps"] == 2.0**-24
    assert get_recipe("sol-h3").inference_adapter != get_recipe("minimax-h3-fl2va").inference_adapter
    recipe = get_recipe("sol-h3")
    schema = describe_model("sol-h3", full=True)["input_schema"]["properties"]
    assert schema["ltx_guide_policy"]["type"] == "string"
    assert schema["ltx_guide_policy"]["enum"] == ["midguide", "none"]
    assert schema["ltx_audio_mode"]["type"] == "string"
    assert schema["ltx_audio_mode"]["enum"] == ["joint", "frozen"]
    automatic = validate_inputs(recipe, {"prompt": "A scene"})
    assert "ltx_guide_policy" not in automatic and "ltx_audio_mode" not in automatic
    for policy in ("midguide", "none"):
        values = validate_inputs(recipe, {"prompt": "A scene", "ltx_guide_policy": policy})
        assert values["ltx_guide_policy"] == policy
    for mode in ("joint", "frozen"):
        assert validate_inputs(recipe, {"prompt": "A scene", "ltx_audio_mode": mode})["ltx_audio_mode"] == mode
    for field, value in (("ltx_guide_policy", "auto"), ("ltx_guide_policy", 1), ("ltx_guide_policy", None),
                         ("ltx_audio_mode", "auto"), ("ltx_audio_mode", None)):
        with pytest.raises(ModelError, match=field):
            validate_inputs(recipe, {"prompt": "A scene", field: value})


def test_cli_forwards_policies_and_omits_automatic_defaults(monkeypatch, capsys):
    import sys

    from examples_merak.aigc.sol_h3 import generate
    from xhmodel_merak.model_inference.inputs import validate_inputs

    captured = {}

    def inference_model(**kwargs):
        captured.update(kwargs)
        validate_inputs(get_recipe("sol-h3"), kwargs["inputs"])
        return SimpleNamespace(to_dict=lambda: {"output_path": "out.mp4"})

    monkeypatch.setattr(generate, "inference_model", inference_model)
    for flags, expected in (
        ([], {}),
        (["--ltx-guide-policy", "midguide"], {"ltx_guide_policy": "midguide"}),
        (["--ltx-guide-policy", "none", "--ltx-audio-mode", "frozen"],
         {"ltx_guide_policy": "none", "ltx_audio_mode": "frozen"}),
    ):
        monkeypatch.setattr(
            sys, "argv", ["generate", "--artifact-dir", "bundle", "--prompt", "A scene", "--output", "out.mp4", *flags]
        )
        generate.main()
        assert captured["inputs"] == {"prompt": "A scene", "seed": 42, "output_path": "out.mp4", **expected}
        assert '"output_path": "out.mp4"' in capsys.readouterr().out


def test_unsafe_rms_is_rejected_even_when_its_interface_is_fp16(tmp_path):
    meta = bundle_fixture(tmp_path)
    path = tmp_path / meta["graphs"]["h3_feature_00_50"]["path"]
    graph = onnx.load(path)
    graph.graph.node[0].op_type = "RMSNorm"
    graph.graph.node[0].attribute.extend(
        [
            helper.make_attribute("force_fp32", 1),
            helper.make_attribute("sum_dtype", "float32"),
            helper.make_attribute("eps", 1e-6 / 256**2),
        ]
    )
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="reciprocal"):
        read_metadata(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("h3_precision", {"rms_norm_mode": "plain", "rms_norm_eps": 1e-10}),
        ("input_dtype", "bfloat16"),
        ("refiner_cache_policy", "shared_across_sigmas"),
    ],
)
def test_partial_or_different_runtime_contract_is_rejected(tmp_path, field, value):
    meta = bundle_fixture(tmp_path)
    meta[field] = value
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        read_metadata(tmp_path)


def test_missing_refiner_or_path_escape_is_rejected(tmp_path):
    meta = bundle_fixture(tmp_path)
    del meta["graphs"]["ltx_denoiser"]
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="missing learned"):
        read_metadata(tmp_path)
    meta = json.loads((tmp_path / "runtime_meta_info.json").read_text())
    meta["graphs"]["ltx_denoiser"] = dict(path="../outside.onnx", input_names=["x"], output_names=["y"])
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="escapes"):
        read_metadata(tmp_path)


def test_public_duration_profiles_select_unified_workflows(tmp_path):
    recipe = get_recipe("sol-h3")
    assert recipe.profiles == ("all-dense-384p-768p-5s", "all-dense-384p-768p-15s")
    assert recipe.default_profile == recipe.profiles[0]
    source = tmp_path / "source"
    source.mkdir()
    source_fixture(source)
    profile = recipe.profiles[1]
    plan = resolve_build_plan(
        model_id="sol-h3",
        model_dir=str(source),
        profile=profile,
        output_dir=str(tmp_path / "artifact"),
        device="cuda:0",
    )
    destination = tmp_path / "workflow.yaml"
    materialize_workflow_config(plan, destination)
    cfg = validate_build_config(yaml.safe_load(destination.read_text())["export"])
    assert plan.variant == "full"
    assert cfg["mode"] == "all" and cfg["profile"] == profile
    assert (cfg["draft_frames"], cfg["refiner_frames"]) == (362, 361)
    assert cfg["ltx_reference_conditioning"]["require_native_resolution"] is True
    assert cfg["conditioner_quant_type"] == "w4a8h0_ssfp"
    assert cfg["prompt"]["max_tokens"] == 1024
    assert cfg["qwen3_vl"]["prefill_chunk_length"] == 256
    assert "image_max_pixels" not in cfg["qwen3_vl"]
    assert "video_max_pixels" not in cfg["qwen3_vl"]
    assert cfg["references"]["image"]["max_count"] == 3
    assert cfg["references"]["image"]["vae_gears"] == [[384, 672], [448, 672], [672, 448], [512, 512]]
    assert cfg["references"]["video"]["vae_size"] == [384, 672]
    with pytest.raises(ModelError, match="profile"):
        resolve_build_plan(model_id="sol-h3", model_dir=str(source), profile="all-dense-384p-768p-2s")


def test_calibration_inventory_resolves_relative_media_and_yaml_takes_precedence(tmp_path):
    from xhmodel_merak.xh_other_model.models.sol_h3.build import calibration_inputs

    config = {"mode": "ref2va"}
    with pytest.raises(ValueError, match="calibration_inputs"):
        calibration_inputs(tmp_path, config)
    (tmp_path / "calibration_inputs.json").write_text(
        json.dumps(
            {
                "ref2va": {"references": [{"kind": "image", "path": "image.png"}]},
                "fl2va": {"image": "first.png", "last_image": "/absolute/last.png"},
            }
        )
    )
    assert calibration_inputs(tmp_path, config)["references"][0]["path"] == str(tmp_path / "image.png")
    assert calibration_inputs(tmp_path, {"mode": "fl2va"}) == {
        "image": str(tmp_path / "first.png"),
        "last_image": "/absolute/last.png",
    }
    assert calibration_inputs(tmp_path, {**config, "calibration_inputs": {"references": []}}) == {"references": []}


def test_schema4_rejects_legacy_unaligned_graph_contract(tmp_path):
    meta = bundle_fixture(tmp_path)
    meta["schema_version"] = 4
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema-4.*aligned capacities"):
        read_metadata(tmp_path)
