"""Public planning and publication contracts; learned graphs have separate SDK tests."""

import json
from types import SimpleNamespace

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import describe_model, list_models
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_inference.inputs import validate_inputs


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for path, payload in {
        "transformer/config.json": {"_class_name": "QwenImage21Transformer2DModel"},
        "vae/config.json": {"_class_name": "AutoencoderKLQwenImage21"},
        "text_encoder/config.json": {"model_type": "qwen3_vl"},
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
    }.items():
        file = root / path
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps(payload))
    return root


@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_public_plan_and_discovery(source, tmp_path, quant_type):
    import yaml

    from xhmodel_merak.model_build.workflow import materialize_workflow_config

    assert "qwen-image-2.1" in {model["model_id"] for model in list_models()}
    description = describe_model("qwen-image-2.1", full=True)
    assert description["components"] == ["image"]
    assert description["golden_components"] == []
    assert description["deployment_precision_options"]["image"]["choices"] == [
        "w16a16_sefp",
        "w8a16_sefp",
        "w8a8_sefp",
    ]
    plan = resolve_build_plan(
        model_id="qwen-image-2.1",
        model_dir=str(source),
        output_dir=str(tmp_path / "release"),
        device="cuda:0",
        deployment_precision={"image": quant_type},
    )
    assert plan.components == ("image",)
    assert plan.config_overrides == {"quant": None}
    assert plan.build_adapter == "qwen_image21_build"
    assert plan.deployment_precision == {"image": quant_type}
    token = quant_type.removesuffix("_sefp")
    assert plan.artifact_name == f"hmquant_xh2_qwen_image21_{token}_1mp_c4352"
    effective = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, effective)
    config = yaml.safe_load(effective.read_text())
    assert config["export"]["model"]["quant_scheme"] == dict(target_device="XH2a", quant_type=quant_type)
    assert "quant_type" not in config["export"]
    assert "diffusers==0.41.0" in plan.worker_requirements
    assert "huggingface-hub==1.33.0" in plan.worker_requirements


def test_public_plan_defaults_to_w16(source, tmp_path):
    plan = resolve_build_plan(
        model_id="qwen-image-2.1", model_dir=str(source), output_dir=str(tmp_path / "release"), device="cuda:0"
    )
    assert plan.deployment_precision == {"image": "w16a16_sefp"}
    assert plan.artifact_name == "hmquant_xh2_qwen_image21_w16a16_1mp_c4352"


@pytest.mark.parametrize(
    "options",
    [
        {"precision": {"weight": "w4", "activation": "a16"}},
        {"quant_method": "gptq"},
        {"components": ["denoise"]},
        {"profile": "dynamic"},
        {"low_memory": True},
        {"deployment_precision": {"image": "w4a16_sefp"}},
    ],
)
def test_public_plan_rejects_unimplemented_modes(source, tmp_path, options):
    with pytest.raises(ModelError):
        resolve_build_plan(
            model_id="qwen-image-2.1",
            model_dir=str(source),
            output_dir=str(tmp_path / "release"),
            device="cuda:0",
            **options,
        )


def test_source_identity_is_checked_without_weights(source, tmp_path):
    (source / "transformer/config.json").write_text('{"_class_name": "QwenImageTransformer2DModel"}')
    with pytest.raises(ModelError, match="QwenImage21Transformer2DModel"):
        resolve_build_plan(
            model_id="qwen-image-2.1", model_dir=str(source), output_dir=str(tmp_path / "release"), device="cuda:0"
        )


@pytest.mark.parametrize(
    "inputs",
    [
        {"prompt": ""},
        {"prompt": "x", "steps": True},
        {"prompt": "x", "steps": 0},
        {"prompt": "x", "guidance_scale": float("nan")},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "images": "a"},
        {"prompt": "x", "height": 1000},
    ],
)
def test_public_inputs_reject_invalid_values(inputs):
    with pytest.raises(ModelError):
        validate_inputs(get_recipe("qwen-image-2.1"), inputs)


@pytest.fixture(params=["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def bundle(tmp_path, request):
    # These graph files stand only for publication paths, not learned-model accuracy.
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from xhmodel_merak.xh_other_model.models.qwen_image21.build import export_entries

    root = tmp_path / "stage" / "qwen_image21_export"
    root.mkdir(parents=True)
    from _qwen_image21_bundle_fixture import bundle_metadata

    meta, dit, paths = bundle_metadata(request.param)
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".onnx":
            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "weight"], ["y"])],
                "inventory",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                [numpy_helper.from_array(np.ones(1, dtype=np.float32), "weight")],
            )
            onnx.save_model(
                helper.make_model(graph),
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
        else:
            path.write_text("{}")
    (root / "dit_meta_info.json").write_text(json.dumps(dit))
    (root / "runtime_meta_info.json").write_text(json.dumps(meta))
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(root, meta)))
    return root


def test_publication_handoff_relocation_and_manifest_inventory(bundle):
    from xhmodel_merak.model_artifact.export_info import publish_export_info
    from xhmodel_merak.model_integrations.qwen_image21.build import normalize_export, validate_artifact
    from xhmodel_merak.model_integrations.qwen_image21.inference import adapter

    stage = bundle.parent
    quant_type = json.loads((bundle / "runtime_meta_info.json").read_text())["quant_type"]
    plan = SimpleNamespace(components=("image",), deployment_precision={"image": quant_type})
    assert normalize_export(stage, SimpleNamespace(work_dir=str(bundle)), plan) is None
    metadata = validate_artifact(stage, plan)
    assert not (stage / "export_meta_info.json").exists()
    publish_export_info(stage, metadata)
    moved = stage.with_name("moved")
    stage.rename(moved)
    files = [{"path": path.relative_to(moved).as_posix()} for path in moved.rglob("*") if path.is_file()]
    manifest = {"artifact": {"files": files}, "parameters": {"precision": {"deployment": {"image": quant_type}}}}
    args = dict(
        recipe=get_recipe("qwen-image-2.1"), root=moved, manifest=manifest, meta_file=moved / metadata["metadata_file"]
    )
    adapter.validate_artifact(**args)
    manifest["artifact"]["files"] = [file for file in files if file["path"] != "denoise_image/model.onnx"]
    with pytest.raises(ModelError, match="inventory"):
        adapter.validate_artifact(**args)
    manifest["artifact"]["files"] = [file for file in files if file["path"] != "denoise_image/weights.bin"]
    with pytest.raises(ModelError, match="inventory"):
        adapter.validate_artifact(**args)


def test_publication_rejects_inconsistent_export_inventory(bundle):
    from xhmodel_merak.model_integrations.qwen_image21.build import validate_artifact

    (bundle / "export_meta_info.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        quant_type = json.loads((bundle / "runtime_meta_info.json").read_text())["quant_type"]
        validate_artifact(bundle, SimpleNamespace(components=("image",), deployment_precision={"image": quant_type}))


def test_publication_rejects_precision_mismatch(bundle):
    from xhmodel_merak.model_integrations.qwen_image21.build import validate_artifact
    from xhmodel_merak.model_integrations.qwen_image21.inference import adapter

    quant_type = json.loads((bundle / "runtime_meta_info.json").read_text())["quant_type"]
    wrong_type = "w8a8_sefp" if quant_type == "w16a16_sefp" else "w16a16_sefp"
    with pytest.raises(ValueError, match="planned deployment precision"):
        validate_artifact(bundle, SimpleNamespace(components=("image",), deployment_precision={"image": wrong_type}))
    manifest = {
        "artifact": {
            "files": [{"path": path.relative_to(bundle).as_posix()} for path in bundle.rglob("*") if path.is_file()]
        },
        "parameters": {"precision": {"deployment": {"image": wrong_type}}},
    }
    with pytest.raises(ValueError, match="Manifest deployment precision"):
        adapter.validate_artifact(
            recipe=get_recipe("qwen-image-2.1"),
            root=bundle,
            manifest=manifest,
            meta_file=bundle / "runtime_meta_info.json",
        )


def test_publication_accepts_a_different_self_consistent_profile(bundle):
    from xhmodel_merak.model_integrations.qwen_image21.artifact import validate_runtime

    path = bundle / "runtime_meta_info.json"
    meta = json.loads(path.read_text())
    meta["deployment_config"].update(output_size=[768, 1376], reference_size=[1376, 768])
    path.write_text(json.dumps(meta))
    validated, _ = validate_runtime(bundle, ("image",))
    assert validated["deployment_config"]["output_size"] == [768, 1376]


def test_configured_build_example_exposes_workflow_path():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "examples_merak.aigc.qwen_image21.configured", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout and "--model-dir" in result.stdout


def test_metadata_configuration_must_match_all_component_gears(bundle):
    from xhmodel_merak.model_integrations.qwen_image21.artifact import validate_runtime

    path = bundle / "runtime_meta_info.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps(meta))
    validate_runtime(bundle, ("image",))
    for field, value in [
        ("target_capacity", 4608),
        ("reference_capacity", 4096),
        ("visual_patch_capacity", 4608),
        ("canvas_size", [1568, 1568]),
        ("max_position", 65536),
    ]:
        original = meta["deployment_config"][field]
        meta["deployment_config"][field] = value
        path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="deployment_config"):
            validate_runtime(bundle, ("image",))
        meta["deployment_config"][field] = original


def test_processor_supports_legacy_and_combined_config_without_losing_image_config(bundle):
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    path = bundle / "runtime_meta_info.json"
    meta = json.loads(path.read_text())
    legacy = next(value for value in meta["processor_files"] if value.endswith("preprocessor_config.json"))
    combined = legacy.replace("preprocessor_config.json", "processor_config.json")
    (bundle / legacy).rename(bundle / combined)
    meta["processor_files"] = [combined if value == legacy else value for value in meta["processor_files"]]
    path.write_text(json.dumps(meta))
    assert read_bundle_metadata(bundle) == meta
    meta["processor_files"].remove(combined)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="processor resources"):
        read_bundle_metadata(bundle)
