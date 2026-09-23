import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest
import yaml

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build import UNSET, SpeculativeRequest, describe_model
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe, list_recipes, resolve_workflow_config
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "hidden_size": 2048,
                "num_hidden_layers": 48,
                "num_experts": 512,
                "router_aux_loss_coef": 0.001,
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("quant_method", [UNSET, "autoround"])
def test_paddleocr_existing_checkpoint_skips_quantizer_requirements(tmp_path, quant_method):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "paddleocr_vl",
        "architectures": ["PaddleOCRVLForConditionalGeneration"],
    }))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "quantization_config.json").write_text(
        json.dumps({"bits": 8, "provider": "auto-round"})
    )
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{}\n')

    plan = resolve_build_plan(
        model_id="paddleocr-vl-1.6",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        quant_checkpoint=str(checkpoint),
        quant_method=quant_method,
        resources={"calibration": str(calibration)},
    )

    assert plan.quant_method == "autoround"
    assert plan.worker_capabilities == ()
    assert "gptqmodel>=5.8,<5.9" in plan.worker_requirements
    assert not any(requirement.startswith("auto-round") for requirement in plan.worker_requirements)
    assert plan.deployment_precision == {
        "visual": "w8a16h1_sefp", "prefill": "w8a8_sefp", "decode": "w8a8_sefp",
    }


def test_default_plan_uses_full_autoround_and_dynamic_vit(model_dir, tmp_path):
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
    )

    assert plan.variant == "full"
    assert plan.quant_method == "autoround"
    assert plan.precision.weight == "w4"
    assert plan.precision.activation == "a8"
    assert plan.vision_profile == "visualm96-196-384-704-1536"
    assert plan.vision_gears == (96, 196, 384, 704, 1536)
    assert plan.workflow_config == get_recipe("qwen3.5-9b").default_variant.workflow_config
    assert plan.source_config_sha256 == hashlib.sha256(Path(plan.config_path).read_bytes()).hexdigest()
    assert (
        plan.source_bindings["model"]["config_sha256"]
        == hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()
    )
    assert isinstance(plan.source_bindings["model"]["device"], int)
    assert isinstance(plan.source_bindings["model"]["inode"], int)
    assert plan.artifact_name == ("hmquant_xh2_qwen3_5_9b_w4a8_256_2k_mpe256k_visualm96_196_384_704_1536")
    assert re.search(r"(?:^|_)20\d{6}(?:_|$)", plan.artifact_name) is None


@pytest.mark.parametrize("weight, quant_method", [("w4", "autoround"), ("w16", None)])
def test_described_build_choices_are_accepted_by_the_real_resolver(model_dir, tmp_path, weight, quant_method):
    parameters = describe_model("qwen3.5-9b")
    values = {key: item["default"] for key, item in parameters.items() if "default" in item}
    values["precision"] = next(choice for choice in parameters["precision"]["choices"] if choice["weight"] == weight)
    values["quant_method"] = quant_method
    values["context_length"] = parameters["context_length"]["maximum"]
    values["output_dir"] = str(tmp_path / "artifact")
    assert quant_method in parameters["quant_method"]["choices"]

    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        **values,
    )

    assert plan.precision.weight == weight
    assert plan.quant_method == quant_method
    assert plan.context_length == parameters["context_length"]["maximum"]
    assert plan.prefill_chunk_length == parameters["prefill_chunk_length"]["default"]


def test_qwen3_next_mtp_selects_registered_gptq_variant(model_dir, tmp_path):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config["model_type"] = "qwen3_next"
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    plan = resolve_build_plan(
        model_id="qwen3-next-80b-a3b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        speculative={"mode": "mtp", "num_draft_tokens": 6},
    )

    assert plan.variant == "mtp"
    assert plan.config_path.endswith("qwen3_next_80b_a3b_full_mtp_gptq.yaml")
    assert plan.quant_method == "gptq"
    assert "gptqmodel>=5.8,<5.9" in plan.worker_requirements
    assert not any(requirement.startswith("auto-round") for requirement in plan.worker_requirements)
    assert plan.worker_capabilities == ("gptqmodel.recipes.qwen3_next:quantize_qwen3_next",)
    assert plan.config_overrides["export.model.num_draft_tokens"] == 6
    assert plan.config_replacements["export.model.max_pe_length"] == 262144
    assert plan.vision_profile is None

    effective = tmp_path / "qwen3_next_effective.yaml"
    materialize_workflow_config(plan, effective)
    config = WorkflowConfig.from_file(str(effective)).with_overrides(plan.config_overrides)
    assert config.export["model"]["max_pe_length"] == 262144


@pytest.mark.parametrize(
    "num_draft_tokens",
    [True, False, 0, -1, 1.0, "1"],
)
def test_speculative_num_draft_tokens_must_be_a_positive_non_bool_int(
    model_dir,
    tmp_path,
    num_draft_tokens,
):
    with pytest.raises(ModelError, match="positive integer") as exc_info:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            speculative=SpeculativeRequest(mode="mtp", num_draft_tokens=num_draft_tokens),
        )

    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_dflash_requires_an_explicit_assistant_source(model_dir, tmp_path):
    with pytest.raises(ModelError) as exc_info:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            speculative={"mode": "dflash"},
        )

    assert exc_info.value.code == ErrorCode.RESOURCE_NOT_FOUND

    assistant = tmp_path / "assistant"
    assistant.mkdir()
    with pytest.raises(ModelError) as empty_error:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            speculative={"mode": "dflash"},
            sources={"dflash": str(assistant)},
        )
    assert empty_error.value.code == ErrorCode.INVALID_PARAMETER

    (assistant / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "num_target_layers": 32,
                "dflash_config": {
                    "mask_token_id": 0,
                    "target_layer_ids": [1, 9, 17, 25],
                },
            }
        ),
        encoding="utf-8",
    )
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        speculative={"mode": "dflash"},
        sources={"dflash": str(assistant)},
    )
    assert plan.config_replacements["export.model.dflash_config.hf_model"] == str(assistant.resolve())
    assert plan.config_replacements["export.model.dflash_config.noise_token_id"] == 0

    effective = tmp_path / "dflash_effective.yaml"
    materialize_workflow_config(plan, effective)
    config = WorkflowConfig.from_file(str(effective)).with_overrides(plan.config_overrides)
    assert config.export["model"]["dflash_config"]["noise_token_id"] == 0

    # Exercise the real plan/validator boundary, including token ID zero.
    from xhmodel_merak.model_integrations.autoregressive.build import _validate_speculative_assets

    for name in ("context.hmonnx", "draft.hmonnx"):
        (tmp_path / name).write_bytes(b"graph-existence-fixture")
    draft_tokens = config.export["model"]["num_draft_tokens"]
    metadata = {
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": "dflash",
            "num_draft_tokens": draft_tokens,
            "verify_length": draft_tokens + 1,
            "draft_context_onnx": "context.hmonnx",
            "draft_decode_onnx": "draft.hmonnx",
            "draft": {"noise_token_id": 0, "context_hmonnx": "context.hmonnx", "decode_hmonnx": "draft.hmonnx"},
        }
    }
    _validate_speculative_assets(tmp_path, metadata, plan)
    metadata["spec_decode"]["draft"]["noise_token_id"] = 1
    with pytest.raises(ModelError, match="incomplete draft graph set"):
        _validate_speculative_assets(tmp_path, metadata, plan)


def test_gemma_max_pe_is_materialized_before_strict_workflow_overrides(tmp_path):
    source = tmp_path / "gemma"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "enable_moe_block": False,
                },
            }
        ),
        encoding="utf-8",
    )
    plan = resolve_build_plan(
        model_id="gemma4-31b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
    )

    effective = tmp_path / "gemma_effective.yaml"
    materialize_workflow_config(plan, effective)
    config = WorkflowConfig.from_file(str(effective)).with_overrides(plan.config_overrides)

    assert plan.config_replacements["export.model.max_pe_length"] == 262144
    assert config.export["model"]["max_pe_length"] == 262144


def test_gemma_e2b_binds_packaged_calibration_resource(tmp_path):
    source = tmp_path / "gemma"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "enable_moe_block": False,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = resolve_build_plan(
        model_id="gemma4-e2b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
    )

    calibration = plan.resources["calibration"]
    expected = Path(__file__).resolve().parents[2] / "data/calib_data/NeelNanda-pile-10k.jsonl"
    assert Path(calibration.path).resolve() == expected.resolve()
    assert calibration.sha256 == hashlib.sha256(expected.read_bytes()).hexdigest()

    effective = tmp_path / "gemma_e2b_effective.yaml"
    materialize_workflow_config(plan, effective)
    config = WorkflowConfig.from_file(str(effective)).with_overrides(plan.config_overrides)
    assert Path(config.quant["calibration"]["jsonl"]).resolve() == expected.resolve()


def test_qwen_autoround_binds_packaged_calibration_resource(model_dir, tmp_path):
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
    )

    calibration = plan.resources["calibration"]
    expected = Path(__file__).resolve().parents[2] / "data/calib_data/NeelNanda-pile-10k.jsonl"
    assert Path(calibration.path).resolve() == expected.resolve()
    assert calibration.sha256 == hashlib.sha256(expected.read_bytes()).hexdigest()

    effective = tmp_path / "qwen_effective.yaml"
    materialize_workflow_config(plan, effective)
    config = WorkflowConfig.from_file(str(effective)).with_overrides(plan.config_overrides)
    assert Path(config.quant["calibration"]["dataset"]).resolve() == expected.resolve()


@pytest.mark.parametrize(
    "assistant_config",
    [
        {},
        {"dflash_config": {}},
        {"dflash_config": {"mask_token_id": -1}},
        {"dflash_config": {"mask_token_id": 1, "noise_token_id": 2}},
        {"hidden_size": 1, "dflash_config": {"mask_token_id": 1}},
        {"num_target_layers": 64, "dflash_config": {"mask_token_id": 1}},
        {"dflash_config": {"mask_token_id": 1, "target_layer_ids": [1, 32]}},
        {"dflash_config": {"mask_token_id": 1, "target_model_id": "qwen3.8-27b"}},
    ],
)
def test_dflash_source_requires_semantic_assistant_contract(
    model_dir,
    tmp_path,
    assistant_config,
):
    assistant = tmp_path / "assistant"
    assistant.mkdir()
    (assistant / "config.json").write_text(json.dumps(assistant_config), encoding="utf-8")

    with pytest.raises(ModelError) as exc_info:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            speculative={"mode": "dflash"},
            sources={"dflash": str(assistant)},
        )

    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_qwen38_selects_registered_dflash2_variant(model_dir, tmp_path):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config["text_config"] = {"hidden_size": 5120, "num_hidden_layers": 64}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assistant = tmp_path / "assistant"
    assistant.mkdir()
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 5120,
                "num_target_layers": 64,
                "dflash_config": {
                    "mask_token_id": 1,
                    "target_layer_ids": [1, 17, 33, 49],
                },
            }
        ),
        encoding="utf-8",
    )

    plan = resolve_build_plan(
        model_id="qwen3.8-27b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        speculative={"mode": "dflash"},
        sources={"dflash": str(assistant)},
    )

    assert plan.variant == "dflash"
    assert plan.config_path.endswith("qwen3_8_27b_full_dflash2_8k.yaml")
    assert plan.config_replacements["export.model.dflash_config.hf_model"] == str(assistant.resolve())
    assert plan.config_replacements["export.model.model_name"] == "qwen3_8_27b"

    long_plan = resolve_build_plan(
        model_id="qwen3.8-27b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "long-artifact"),
        context_length=262144,
        speculative={"mode": "dflash"},
        sources={"dflash": str(assistant)},
    )
    assert long_plan.config_path.endswith("qwen3_8_27b_full_dflash2_256k_fa.yaml")
    assert long_plan.workflow_config.endswith("qwen3_8_27b_full_dflash2_256k_fa.yaml")


def test_lora_selects_the_only_validated_lora_workflow(model_dir, tmp_path):
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "router_aux_loss_coef": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    plan = resolve_build_plan(
        model_id="qwen3.6-35b-a3b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        lora={"paths": [str(adapter)]},
    )

    assert plan.variant == "lora"
    assert plan.config_overrides["export.model.lora.path"] == [str(adapter.resolve())]
    assert plan.components[-1] == "lora"
    assert plan.lora_paths == (str(adapter.resolve()),)
    assert plan.lora_sha256 == (_directory_sha256(adapter),)


def test_w16_derives_null_quant_method_and_explicit_low_bit_null_is_rejected(model_dir, tmp_path):
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        precision={"weight": "w16", "activation": "a8"},
        quant_method=UNSET,
    )

    assert plan.quant_method is None
    assert plan.field_sources["quant_method"] == "derived"
    assert plan.config_overrides["quant"] is None

    with pytest.raises(ModelError) as exc_info:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact-2"),
            quant_method=None,
            precision={"weight": "w4", "activation": "a8"},
        )
    assert exc_info.value.code == ErrorCode.UNSUPPORTED_COMBINATION


def test_build_plans_deep_copy_registry_default_precision_components(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "router_aux_loss_coef": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    first = resolve_build_plan(
        model_id="qwen3.6-35b-a3b",
        model_dir=str(source),
        output_dir=str(tmp_path / "first"),
    )
    second = resolve_build_plan(
        model_id="qwen3.6-35b-a3b",
        model_dir=str(source),
        output_dir=str(tmp_path / "second"),
    )
    registry_default = get_recipe("qwen3.6-35b-a3b").default_precision

    first.precision.components["attention"]["weight"] = "changed"

    assert second.precision.components["attention"]["weight"] == "w8"
    assert registry_default.components["attention"]["weight"] == "w8"


def test_static_vit_requests_are_rejected(model_dir, tmp_path):
    with pytest.raises(ModelError, match="static width/height profiles are deprecated"):
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            vision={"width": 448, "height": 448},
        )

    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact-dynamic"),
        vision={
            "profile": "visualm96-196-384-704-1536",
            "token_gears": [96, 196, 384, 704, 1536],
        },
    )
    assert plan.vision_gears == (96, 196, 384, 704, 1536)


def test_materialized_workflow_preserves_upstream_visual_contract(model_dir, tmp_path):
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
    )
    effective = tmp_path / "effective_workflow.yaml"

    materialize_workflow_config(plan, effective)

    visual = yaml.safe_load(effective.read_text(encoding="utf-8"))["export"]["model"]["visual_config"]
    model = yaml.safe_load(effective.read_text(encoding="utf-8"))["export"]["model"]
    assert visual["visual_input_mode"] == "patches"
    assert visual["image_token_gears"] == [96, 196, 384, 704, 1536]
    assert visual["image_token_capacity"] == 1536
    assert model["fuse_gdr_ops"] is True
    assert model["fuse_gdr_block_recurrent_ops"] is True
    assert not any(key.startswith("export.model.visual_config.") for key in plan.config_replacements)


def test_every_public_vlm_variant_declares_the_registered_visual_contract():
    for recipe in list_recipes():
        if recipe.vision_profile is None:
            continue
        assert recipe.config_resolver is not None
        planning_module, _, _ = recipe.config_resolver.partition(":")
        validate_visual_workflow = importlib.import_module(planning_module)._validate_visual_workflow
        for variant in recipe.variants:
            for workflow in {variant.workflow_config, *(path for _, path in variant.context_workflows)}:
                data = yaml.safe_load(resolve_workflow_config(workflow).read_text(encoding="utf-8"))
                validate_visual_workflow(recipe, data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_token_gears", [96, 196]),
        ("image_token_capacity", 1024),
        ("patch_size", 16),
        ("spatial_merge_size", 4),
        ("max_size_w", 448),
        ("max_size_h", 448),
    ],
)
def test_paddleocr_plan_rejects_visual_workflow_drift(field, value):
    from xhmodel_merak.model_integrations.paddleocr_vl.planning import _validate_visual_workflow

    recipe = get_recipe("paddleocr-vl-1.6")
    workflow = resolve_workflow_config(recipe.default_variant.workflow_config)
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    data["export"]["model"]["visual_config"][field] = value

    with pytest.raises(ModelError, match="PaddleOCR-VL Workflow visual contract") as error:
        _validate_visual_workflow(recipe, data)
    assert field in error.value.details["mismatches"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visual_input_mode", "images"),
        ("image_token_gears", [96, 196]),
        ("image_token_capacity", 196),
        ("spatial_merge_size", 4),
        ("spatial_merge_size", 2.0),
        ("max_size_w", 448),
        ("max_size_h", 896),
    ],
)
def test_qwen_plan_rejects_visual_workflow_drift_before_export(model_dir, tmp_path, monkeypatch, field, value):
    import xhmodel_merak.model_build.resolver as resolver

    recipe = get_recipe("qwen3.5-9b")
    original = resolve_workflow_config(recipe.default_variant.workflow_config)
    data = yaml.safe_load(original.read_text(encoding="utf-8"))
    data["export"]["model"]["visual_config"][field] = value
    changed = tmp_path / "workflow.yaml"
    changed.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(resolver, "resolve_workflow_config", lambda _path: changed)

    with pytest.raises(ModelError, match="Workflow visual contract") as error:
        resolve_build_plan(model_id=recipe.model_id, model_dir=str(model_dir), output_dir=str(tmp_path / "output"))
    assert field in error.value.details["mismatches"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("size", ["0.8b", "2b", "4b"])
def test_small_qwen_standalone_visual_workflow_uses_current_config_abi(size):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig

    name = size.replace(".", "_")
    path = resolve_workflow_config(
        f"configs_merak/workflows/xh2a/llm_models/qwen3_5/{size}/"
        f"qwen3_5_{name}_visual_only_token_gears_99p_candidate.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["export"]["model"]
    config = XHQwen3_5_VisualConfig(**data)
    assert config.image_token_gears == [96, 196, 384, 704, 1536]
    assert config.image_token_capacity == 1536
    assert config.visual_rope_cache_length >= 3072


@pytest.mark.parametrize("hook", [None, 42, lambda **kwargs: None, lambda **kwargs: {"": 1}])
def test_registered_config_resolver_must_be_callable_and_return_replacements(tmp_path, monkeypatch, hook):
    from dataclasses import replace
    from types import SimpleNamespace

    from xhmodel_merak.model_build import recipe_config

    recipe = replace(get_recipe("qwen3.5-9b"), config_resolver="xhmodel_merak.model_integrations.qwen.planning:hook")
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("export: {}\n", encoding="utf-8")
    monkeypatch.setattr(recipe_config.importlib, "import_module", lambda _name: SimpleNamespace(hook=hook))
    with pytest.raises(ModelError, match="config resolver"):
        recipe_config.resolve_recipe_config(recipe, recipe.default_variant, workflow, model_dir=tmp_path, sources={})


def test_qwen_planning_does_not_import_model_runtime_dependencies(model_dir, tmp_path):
    import subprocess
    import sys

    code = (
        "import sys; from xhmodel_merak.model_build.resolver import resolve_build_plan; "
        f"resolve_build_plan(model_id='qwen3.5-9b', model_dir={str(model_dir)!r}, "
        f"output_dir={str(tmp_path / 'output')!r}); "
        "assert not {'torch', 'transformers', 'xhmodel_merak.xh_llm', 'xhquant'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_qwen_calibration_binding_cannot_be_shadowed_by_cwd(model_dir, tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_autoround_kwargs

    plan = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir), output_dir=str(tmp_path / "output"))
    effective = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, effective)
    data = yaml.safe_load(effective.read_text(encoding="utf-8"))
    shadow = tmp_path / "data/calib_data/NeelNanda-pile-10k.jsonl"
    shadow.parent.mkdir(parents=True)
    shadow.write_text('{"text":"shadow"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir=str(model_dir),
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        quant_cfg=data["quant"],
        export_model_cfg=data["export"]["model"],
        workflow_seed=42,
    )
    assert kwargs["dataset"] == plan.resources["calibration"].path
    assert Path(kwargs["dataset"]) != shadow


def test_qwen_workflow_gdr_flags_apply_only_to_text_models():
    config_root = resolve_workflow_config(
        "configs_merak/workflows/xh2a/llm_models/qwen3_5/4b/qwen3_5_4b_full.yaml"
    ).parents[2]
    gdr_flags = {"fuse_gdr_ops", "fuse_gdr_block_recurrent_ops"}
    for family in ("qwen3_5", "qwen3_5_moe", "qwen3_next"):
        for path in (config_root / family).rglob("*.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            model = data.get("export", {}).get("model", {})
            if str(model.get("model_type", "")).endswith("_visual"):
                assert gdr_flags.isdisjoint(model), path
                continue
            assert gdr_flags.isdisjoint(model.get("visual_config") or {}), path
            if "fuse_gdr_ops" in model:
                assert model["fuse_gdr_ops"] is True, path
            if "fuse_gdr_block_recurrent_ops" in model:
                assert model["fuse_gdr_block_recurrent_ops"] is True, path


def test_quant_output_dir_must_be_unused_disjoint_and_applicable(model_dir, tmp_path):
    output = tmp_path / "artifact"
    quant_output = tmp_path / "quant"
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(output),
        quant_output_dir=str(quant_output),
    )
    assert plan.quant_output_dir == str(quant_output.resolve())

    quant_output.mkdir()
    with pytest.raises(ModelError) as existing_error:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(output),
            quant_output_dir=str(quant_output),
        )
    assert existing_error.value.code == ErrorCode.INVALID_PARAMETER

    nested_output = tmp_path / "nested-artifact"
    with pytest.raises(ModelError, match="must not equal, contain, or be contained"):
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(nested_output),
            quant_output_dir=str(nested_output / "quant"),
        )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "quantization_config.json").write_text(json.dumps({"bits": 4, "provider": "auto-round"}))
    with pytest.raises(ModelError) as checkpoint_error:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(output),
            quant_output_dir=str(tmp_path / "quant-2"),
            quant_checkpoint=str(checkpoint),
        )
    assert checkpoint_error.value.code == ErrorCode.UNSUPPORTED_COMBINATION

    (checkpoint / "config.json").write_text((model_dir / "config.json").read_text())
    checkpoint_plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "checkpoint-artifact"),
        quant_checkpoint=str(checkpoint),
    )
    assert "gptqmodel>=5.8,<5.9" in checkpoint_plan.worker_requirements
    assert not any(requirement.startswith("auto-round") for requirement in checkpoint_plan.worker_requirements)

    with pytest.raises(ModelError) as skipped_error:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(output),
            quant_output_dir=str(tmp_path / "quant-3"),
            precision={"weight": "w16", "activation": "a8"},
            quant_method=None,
        )
    assert skipped_error.value.code == ErrorCode.UNSUPPORTED_COMBINATION


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        file_digest = hashlib.sha256(child.read_bytes()).digest()
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
        digest.update(b"\0")
    return digest.hexdigest()


def test_resource_sha256_is_verified_before_build(tmp_path):
    source = tmp_path / "pi05"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 50,
                "output_features": {"action": {"shape": [7]}},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"tokenizer")
    expected = _directory_sha256(tokenizer)

    plan = resolve_build_plan(
        model_id="pi0.5-libero",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        resources={"tokenizer": {"path": str(tokenizer), "sha256": expected.upper()}},
    )
    assert plan.resources["tokenizer"].sha256 == expected

    inferred_plan = resolve_build_plan(
        model_id="pi0.5-libero",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact-inferred"),
        resources={"tokenizer": str(tokenizer)},
    )
    assert inferred_plan.resources["tokenizer"].sha256 == expected

    with pytest.raises(ModelError) as mismatch_error:
        resolve_build_plan(
            model_id="pi0.5-libero",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact"),
            resources={"tokenizer": {"path": str(tokenizer), "sha256": "0" * 64}},
        )
    assert mismatch_error.value.code == ErrorCode.INVALID_PARAMETER

    with pytest.raises(ModelError) as malformed_error:
        resolve_build_plan(
            model_id="pi0.5-libero",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact"),
            resources={"tokenizer": {"path": str(tokenizer), "sha256": "not-a-digest"}},
        )
    assert malformed_error.value.code == ErrorCode.INVALID_PARAMETER


def test_output_paths_are_new_and_disjoint_from_every_input(model_dir, tmp_path, monkeypatch):
    with pytest.raises(ModelError, match="contained by model_dir"):
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(model_dir / "artifact"),
        )

    with pytest.raises(ModelError, match="contained by model_dir"):
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            quant_output_dir=str(model_dir / "quant"),
        )

    with pytest.raises(ModelError, match="non-empty"):
        resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir), output_dir="  ")

    monkeypatch.chdir(tmp_path)
    from datetime import datetime

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 18, tzinfo=tz)

    monkeypatch.setattr("datetime.datetime", FixedDateTime)
    default = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir))
    assert Path(default.output_dir) == tmp_path / f"{default.artifact_name}_20260918"
    assert not Path(default.output_dir).exists()  # Resolving a plan does not create an output.
    Path(default.output_dir).mkdir()
    with pytest.raises(ModelError, match="must not already exist"):
        resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir))
    with pytest.raises(ModelError, match="must not already exist"):
        resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir), output_dir=".")
    destination = tmp_path / "chosen-release"
    plan = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir), output_dir=str(destination))
    assert Path(plan.output_dir) == destination


def test_source_constraints_reject_cross_size_qwen_sources(model_dir, tmp_path):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config["text_config"] = {"hidden_size": 2560, "num_hidden_layers": 32}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ModelError, match="does not match Registry identity") as exc_info:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
        )

    assert exc_info.value.details["mismatches"][0]["field"] == "text_config.hidden_size"


def test_source_constraints_reject_wrong_family_gemma_sources(tmp_path):
    source = tmp_path / "wrong-gemma"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "unrelated",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "enable_moe_block": False,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="does not match Registry identity") as exc_info:
        resolve_build_plan(
            model_id="gemma4-31b",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact"),
        )

    assert exc_info.value.details["mismatches"][0]["field"] == "model_type"


def test_source_constraints_distinguish_ornith_from_qwen35_moe(tmp_path):
    source = tmp_path / "moe"
    source.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "router_aux_loss_coef": 0.001,
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ModelError, match="does not match Registry identity"):
        resolve_build_plan(
            model_id="ornith-1.0-35b-a3b",
            model_dir=str(source),
            output_dir=str(tmp_path / "ornith-artifact"),
        )

    config["text_config"]["router_aux_loss_coef"] = 0.0
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelError, match="does not match Registry identity"):
        resolve_build_plan(
            model_id="qwen3.6-35b-a3b",
            model_dir=str(source),
            output_dir=str(tmp_path / "qwen-artifact"),
        )


def test_fixed_name_recipes_reject_custom_model_name(tmp_path):
    source = tmp_path / "pi05"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"type": "pi05", "chunk_size": 50, "output_features": {"action": {"shape": [7]}}}),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()

    with pytest.raises(TypeError, match="model_name"):
        resolve_build_plan(
            model_id="pi0.5-libero",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact"),
            model_name="custom_pi",
            resources={"tokenizer": str(tokenizer)},
        )


@pytest.mark.parametrize(
    "field", ["model", "model_name", "artifact_name", "component_names", "input_shapes", "options"]
)
def test_removed_public_parameters_are_rejected_even_when_none(model_dir, tmp_path, field):
    with pytest.raises(TypeError, match=field):
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "artifact"),
            **{field: None},
        )


def test_plan_uses_recipe_export_name_and_explicit_device(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("XHMODEL_BUILD_DEVICE", "cuda:0")
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        device="cpu",
    )
    assert plan.export_name == "qwen3_5_9b"
    assert plan.model_id == "qwen3.5-9b"
    assert plan.device == "cpu"
    assert "model_name" not in plan.field_sources
    assert "artifact_name" not in plan.field_sources


def test_resource_roots_and_children_must_not_be_symlinks(tmp_path):
    source = tmp_path / "pi05"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"type": "pi05", "chunk_size": 50, "output_features": {"action": {"shape": [7]}}}),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"tokenizer")
    linked_root = tmp_path / "linked-tokenizer"
    linked_root.symlink_to(tokenizer, target_is_directory=True)

    with pytest.raises(ModelError, match="symlink"):
        resolve_build_plan(
            model_id="pi0.5-libero",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact-root-link"),
            resources={"tokenizer": str(linked_root)},
        )

    (tokenizer / "linked.model").symlink_to(tokenizer / "tokenizer.model")
    with pytest.raises(ModelError, match="symlink"):
        resolve_build_plan(
            model_id="pi0.5-libero",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact-child-link"),
            resources={"tokenizer": str(tokenizer)},
        )


def test_build_plan_owns_its_nested_input_snapshot(model_dir, tmp_path):
    from dataclasses import replace

    plan = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(model_dir), output_dir=str(tmp_path / "artifact"))
    overrides = {"export.model.test": {"nested": [1]}}
    environment = {"SOME_BUILD_SETTING": "original"}
    clone = replace(plan, config_overrides=overrides, environment=environment)
    overrides["export.model.test"]["nested"].append(2)
    environment["SOME_BUILD_SETTING"] = "changed"
    assert clone.config_overrides == {"export.model.test": {"nested": [1]}}
    assert clone.environment["SOME_BUILD_SETTING"] == "original"
    clone.deployment_precision["prefill"] = "modified"
    assert plan.deployment_precision["prefill"] != "modified"
