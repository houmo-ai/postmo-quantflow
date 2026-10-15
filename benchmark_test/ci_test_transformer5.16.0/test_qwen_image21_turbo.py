"""Turbo source identity, real scheduler behavior, and relocatable package contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_qwen_image21_integration import bundle as bundle

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import describe_model, list_models
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_catalog import get_recipe


MODEL_ID = "qwen-image-2.1-turbo"
SAMPLE_SIGMAS = [1.0, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568]
SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "shift": 1.0,
    "use_dynamic_shifting": False,
    "shift_terminal": None,
}


@pytest.fixture
def turbo_source(tmp_path):
    root = tmp_path / "source"
    for relative, value in {
        "model_index.json": {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SAMPLE_SIGMAS},
        "transformer/config.json": {"_class_name": "QwenImage21Transformer2DModel"},
        "vae/config.json": {"_class_name": "AutoencoderKLQwenImage21"},
        "text_encoder/config.json": {"model_type": "qwen3_vl"},
        "scheduler/scheduler_config.json": SCHEDULER_CONFIG,
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return root


@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_turbo_public_recipe_and_build_plan(turbo_source, tmp_path, quant_type):
    import yaml

    from xhmodel_merak.model_build.workflow import materialize_workflow_config

    assert MODEL_ID in {entry["model_id"] for entry in list_models()}
    recipe = get_recipe(MODEL_ID)
    assert recipe.family == "qwen_image21" and recipe.components == ("image",)
    assert recipe.input_schema["properties"]["steps"]["default"] == 8
    assert recipe.build_adapter == "qwen_image21_build"
    assert describe_model(MODEL_ID, full=True)["components"] == ["image"]
    plan = resolve_build_plan(
        model_id=MODEL_ID,
        model_dir=str(turbo_source),
        output_dir=str(tmp_path / "artifact"),
        device="cuda:0",
        deployment_precision={"image": quant_type},
    )
    assert plan.artifact_name == f"hmquant_xh2_qwen_image21_turbo_{quant_type.removesuffix('_sefp')}_1mp_c4352"
    effective = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, effective)
    config = yaml.safe_load(effective.read_text())
    assert config["export"]["model_id"] == MODEL_ID
    assert config["export"]["model"]["type"] == "XHQwenImage21Model"
    assert config["export"]["model"]["quant_scheme"]["quant_type"] == quant_type


def test_turbo_requires_checkpoint_sampling_grid(turbo_source, tmp_path):
    (turbo_source / "model_index.json").write_text('{"_class_name": "QwenImage21Pipeline"}')
    with pytest.raises(ModelError, match="sample_sigmas"):
        resolve_build_plan(model_id=MODEL_ID, model_dir=str(turbo_source), output_dir=str(tmp_path / "artifact"))


def test_base_recipe_rejects_turbo_checkpoint(turbo_source, tmp_path):
    with pytest.raises(ModelError, match="Turbo|turbo|sample_sigmas"):
        resolve_build_plan(
            model_id="qwen-image-2.1", model_dir=str(turbo_source), output_dir=str(tmp_path / "artifact")
        )


@pytest.mark.parametrize("change", [{"use_dynamic_shifting": True}, {"shift_terminal": 0.02}, {"shift": 2.0}])
def test_turbo_rejects_incompatible_scheduler(turbo_source, tmp_path, change):
    (turbo_source / "scheduler/scheduler_config.json").write_text(json.dumps({**SCHEDULER_CONFIG, **change}))
    with pytest.raises(ModelError, match="scheduler"):
        resolve_build_plan(model_id=MODEL_ID, model_dir=str(turbo_source), output_dir=str(tmp_path / "artifact"))


@pytest.mark.parametrize("steps,override", [(1, None), (20, None), (20, [1.0, 0.6, 0.2])])
def test_turbo_runtime_matches_real_scheduler_and_reports_effective_steps(steps, override):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from test_qwen_image21_pipeline import runtime_fixture

    runtime, requests, _ = runtime_fixture(sample_sigmas=SAMPLE_SIGMAS)
    seen = []
    _, actual = runtime.generate(
        "positive",
        steps=steps,
        seed=23,
        sigmas=override,
        callback=lambda index, timestep, latents: seen.append(timestep.clone()),
    )
    expected_scheduler = FlowMatchEulerDiscreteScheduler.from_config(runtime.scheduler_config)
    selected = SAMPLE_SIGMAS if override is None else override
    expected_scheduler.set_timesteps(sigmas=selected, device="cpu")
    expected_scheduler.set_begin_index(0)
    expected = torch.randn((1, 1, 4, 2, 2), generator=torch.Generator().manual_seed(23))
    expected = expected.reshape(1, 4, 4).transpose(1, 2)
    for timestep in expected_scheduler.timesteps:
        expected = expected_scheduler.step(torch.ones_like(expected).half(), timestep, expected, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(seen), expected_scheduler.timesteps, rtol=0, atol=0)
    assert runtime.num_inference_steps == len(selected)
    assert len(requests) == 1


@pytest.mark.parametrize("sigmas", [[], [True], [float("nan")], [1.0, 0.0], [0.5, 0.6], [1.0, 1.0], "eight"])
def test_invalid_sigma_grid_is_rejected_before_graph_execution(sigmas):
    from test_qwen_image21_pipeline import runtime_fixture

    runtime, requests, _ = runtime_fixture(sample_sigmas=SAMPLE_SIGMAS)
    with pytest.raises(ValueError, match="sigmas"):
        runtime.generate("positive", sigmas=sigmas)
    assert not requests


@pytest.fixture
def turbo_bundle(bundle):
    root = bundle
    meta = json.loads((root / "runtime_meta_info.json").read_text())
    meta["model_id"] = MODEL_ID
    meta["resources"]["pipeline_config"] = "resources/model_index.json"
    path = root / meta["resources"]["pipeline_config"]
    path.parent.mkdir(parents=True, exist_ok=True)
    (root / meta["resources"]["pipeline_config"]).write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "sample_sigmas": SAMPLE_SIGMAS,
            }
        )
    )
    (root / meta["resources"]["scheduler_config"]).write_text(json.dumps(SCHEDULER_CONFIG))
    (root / "runtime_meta_info.json").write_text(json.dumps(meta))
    return root


def test_turbo_bundle_keeps_pipeline_config_after_relocation(turbo_bundle):
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import export_entries
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    moved = turbo_bundle.with_name("relocated")
    turbo_bundle.rename(moved)
    meta = read_bundle_metadata(moved)
    assert meta["model_id"] == MODEL_ID
    entries = export_entries(moved, meta)
    assert entries["pipeline_config"] == {"path": "resources/model_index.json", "type": "json"}
    pipeline = json.loads((moved / entries["pipeline_config"]["path"]).read_text())
    assert pipeline["sample_sigmas"] == SAMPLE_SIGMAS


def test_turbo_bundle_rejects_missing_sampling_resource(turbo_bundle):
    meta_path = turbo_bundle / "runtime_meta_info.json"
    meta = json.loads(meta_path.read_text())
    del meta["resources"]["pipeline_config"]
    meta_path.write_text(json.dumps(meta))
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    with pytest.raises(ValueError, match="pipeline_config|sample_sigmas"):
        read_bundle_metadata(turbo_bundle)


def test_publication_rejects_turbo_bundle_under_base_identity(turbo_bundle):
    from xhmodel_merak.model_integrations.qwen_image21.build import validate_artifact
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import export_entries
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    meta = read_bundle_metadata(turbo_bundle)
    (turbo_bundle / "export_meta_info.json").write_text(json.dumps(export_entries(turbo_bundle, meta)))
    plan = SimpleNamespace(
        model_id="qwen-image-2.1", components=("image",), deployment_precision={"image": meta["quant_type"]}
    )
    with pytest.raises(ValueError, match="model_id|identity"):
        validate_artifact(turbo_bundle, plan)
