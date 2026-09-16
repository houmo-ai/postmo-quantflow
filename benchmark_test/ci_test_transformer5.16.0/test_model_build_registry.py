import hashlib
import inspect
import json
import subprocess
import sys

import pytest
import yaml

from xhmodel_merak._requirements import validate_worker_requirements
from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import build_model, describe_model, list_models
from xhmodel_merak.model_catalog import list_recipes, resolve_workflow_config
from xhmodel_merak.model_catalog.registry import get_adapter_spec, get_artifact_format_spec


EXPECTED_VARIANTS = {
    "macaron-v1-tall": ["full"],
    "qwen3.5-0.8b": ["full"],
    "qwen3.5-2b": ["full"],
    "qwen3.5-4b": ["full"],
    "qwen3.5-9b": ["full", "mtp", "dflash"],
    "qwen3.6-27b": ["full", "mtp", "dflash"],
    "qwen3.8-27b": ["full", "mtp", "dflash"],
    "qwen3.6-35b-a3b": ["full", "mtp", "dflash", "lora"],
    "ornith-1.0-35b-a3b": ["full"],
    "qwen3.5-122b-a10b": ["full", "mtp", "dflash"],
    "qwen3-next-80b-a3b": ["full", "mtp"],
    "gemma4-12b-unified": ["full"],
    "gemma4-e2b": ["full"],
    "gemma4-e4b": ["full"],
    "gemma4-31b": ["full"],
    "gemma4-26b-a4b": ["full"],
    "ling-3.0-flash": ["full"],
    "pi0.5-libero": ["lerobot-h50"],
    "kokoro-82m-v1.1-zh": ["full"],
}


def test_registry_exposes_the_stable_cross_task_matrix():
    models = list_models()

    assert {item["model_id"]: item["variants"] for item in models} == EXPECTED_VARIANTS
    assert [item["model_id"] for item in models] == sorted(EXPECTED_VARIANTS)


def test_distributed_catalog_preserves_the_full_public_description_snapshot():
    descriptions = {model: describe_model(model, full=True) for model in EXPECTED_VARIANTS}
    # First-release API: one public model_id; export naming stays internal.
    payload = json.dumps(descriptions, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(payload).hexdigest() == "56847513a2af0d3c74c4b60da2fe4249ceeb4eeeb7c334780455f767619e0045"


@pytest.mark.parametrize("model_id", EXPECTED_VARIANTS)
def test_default_description_only_lists_editable_build_arguments(model_id):
    parameters = describe_model(model_id)

    assert parameters == describe_model(model_id, full=False)
    json.dumps(parameters)
    assert set(parameters) <= set(inspect.signature(build_model).parameters)
    assert parameters["model_dir"]["required"] is True
    assert parameters["output_dir"]["required"] is True
    assert parameters["device"]["default"] is None
    assert not set(parameters) & {
        "model_id",
        "components",
        "vision",
        "attention",
        "profile",
        "max_pe_length",
        "deployment_precision",
    }
    assert all("required" in item or "default" in item for item in parameters.values())


def test_qwen_description_has_editable_defaults_and_correlated_choices():
    parameters = describe_model("qwen3.5-4b")

    assert set(parameters) == {
        "model_dir",
        "output_dir",
        "device",
        "precision",
        "quant_method",
        "quant_checkpoint",
        "quant_output_dir",
        "context_length",
        "prefill_chunk_length",
    }
    assert parameters["precision"]["default"] == {"weight": "w4", "activation": "a8"}
    assert parameters["precision"]["choices"] == [
        {"weight": "w4", "activation": "a8"},
        {"weight": "w16", "activation": "a8"},
    ]
    assert parameters["quant_method"]["default"] == "autoround"
    assert parameters["quant_method"]["choices"] == ["autoround", None]
    assert parameters["context_length"]["default"] == 2048
    assert parameters["context_length"]["minimum"] == 1
    assert parameters["context_length"]["maximum"] == 262144
    assert parameters["prefill_chunk_length"]["default"] == 256
    assert "context_length" in parameters["prefill_chunk_length"]["description"]


def test_editable_description_omits_fixed_and_inapplicable_quantization():
    kokoro = describe_model("kokoro-82m-v1.1-zh")
    assert set(kokoro) == {"model_dir", "output_dir", "device"}

    ling = describe_model("ling-3.0-flash")
    assert "precision" not in ling
    assert "quant_method" not in ling
    assert {"quant_checkpoint", "quant_output_dir", "context_length", "prefill_chunk_length"} <= ling.keys()


def test_editable_description_exposes_registered_variants_and_required_inputs():
    qwen = describe_model("qwen3.5-9b")
    assert qwen["speculative"]["mode"]["choices"] == ["mtp", "dflash"]
    assert qwen["speculative"]["num_draft_tokens"] == {"default": None, "minimum": 1}
    assert qwen["sources"]["required_by_variant"] == {"dflash": ["dflash"]}
    assert "lora" not in qwen
    assert "lora" in describe_model("qwen3.6-35b-a3b")
    assert describe_model("qwen3-next-80b-a3b")["speculative"]["mode"]["choices"] == ["mtp"]

    pi05 = describe_model("pi0.5-libero")
    assert pi05["resources"]["required"] is True
    assert pi05["resources"]["keys"] == ["tokenizer"]
    assert set(pi05) == {"model_dir", "output_dir", "device", "resources"}
    assert "resources" not in describe_model("qwen3.5-4b")


def test_editable_description_returns_independent_nested_copies():
    parameters = describe_model("qwen3.5-9b")
    parameters["precision"]["default"]["weight"] = "changed"
    parameters["precision"]["choices"][0]["weight"] = "changed"
    parameters["speculative"]["mode"]["choices"].clear()
    parameters["sources"]["required_by_variant"]["dflash"].clear()

    fresh = describe_model("qwen3.5-9b")
    assert fresh["precision"]["default"]["weight"] == "w4"
    assert fresh["precision"]["choices"][0]["weight"] == "w4"
    assert fresh["speculative"]["mode"]["choices"] == ["mtp", "dflash"]
    assert fresh["sources"]["required_by_variant"] == {"dflash": ["dflash"]}


@pytest.mark.parametrize("full", [None, 0, 1, "false", [], {}])
def test_describe_model_requires_a_boolean_full_flag(full):
    with pytest.raises(TypeError, match="full must be a bool"):
        describe_model("qwen3.5-4b", full=full)


@pytest.mark.parametrize("full", [False, True])
def test_both_description_modes_reject_unknown_models(full):
    with pytest.raises(ModelError):
        describe_model("not-a-registered-model", full=full)


def test_catalog_is_the_single_registration_entry_point():
    import importlib.util

    for path in (
        "xhmodel_merak.model_build.registry",
        "xhmodel_merak.model_build.adapters",
        "xhmodel_merak.model_inference.adapters",
        "xhmodel_merak.model_inference.formats",
    ):
        spec = importlib.util.find_spec(path)
        assert spec is None or spec.origin is None  # Local __pycache__ may leave an empty namespace.


def test_adapter_refs_are_resolved_through_registered_targets():
    assert get_adapter_spec("build", "autoregressive.build").target == (
        "xhmodel_merak.model_integrations.autoregressive.build"
    )
    assert get_adapter_spec("build", "pi05.build").target == ("xhmodel_merak.model_integrations.pi05.build")
    assert get_adapter_spec("inference", "qwen.inference").target == (
        "xhmodel_merak.model_integrations.qwen.inference:adapter"
    )
    assert get_artifact_format_spec("autoregressive").validator == (
        "xhmodel_merak.model_integrations.autoregressive.artifact:validate_artifact"
    )


def test_descriptions_are_json_serializable_and_qwen_static_vit_is_not_public():
    for model in EXPECTED_VARIANTS:
        description = describe_model(model, full=True)
        json.dumps(description)
        assert description["input_schema"]["type"] == "object"
        assert description["output_schema"]["type"] == "object"
        if not model.startswith(("macaron", "qwen", "ornith")) or model == "qwen3-next-80b-a3b":
            assert description["vision"] is None
        else:
            assert description["vision"] == {
                "mode": "dynamic_token_gears",
                "profile": "visualm96-196-384-704-1536",
                "image_token_gears": [96, 196, 384, 704, 1536],
                "static_profiles": [],
            }
            assert "448x448" not in json.dumps(description)
            assert "896x896" not in json.dumps(description)


def test_describe_model_returns_independent_schema_copies():
    description = describe_model("qwen3.5-4b", full=True)
    description["input_schema"]["required"].clear()
    description["output_schema"]["properties"].clear()

    fresh = describe_model("qwen3.5-4b", full=True)
    assert fresh["input_schema"]["required"] == ["prompt"]
    assert "output_text" in fresh["output_schema"]["properties"]


def test_text_output_schema_matches_typed_inference_result_fields():
    for model in EXPECTED_VARIANTS:
        description = describe_model(model, full=True)
        if description["task"] not in {
            "multimodal-generation",
            "text-generation",
            "vision-language-generation",
        }:
            continue
        assert description["output_schema"]["required"] == ["output_text", "output_tokens"]


def test_all_public_workflow_configs_exist_and_match_registered_quantization():
    for recipe in list_recipes():
        bundled_resource_names = {name for name, _ in recipe.bundled_resources}
        assert {name for name, _ in recipe.resource_config_bindings} <= bundled_resource_names | set(
            recipe.required_resources
        )
        for variant in recipe.variants:
            workflow_configs = {variant.workflow_config}
            workflow_configs.update(path for _, path in variant.context_workflows)
            for workflow_config in workflow_configs:
                config_path = resolve_workflow_config(workflow_config)
                data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                quant = data.get("quant")
                if recipe.quant_methods:
                    assert quant["method"] in recipe.quant_methods, config_path
                else:
                    assert quant is None, config_path
                if recipe.artifact_format == "autoregressive":
                    quant_type = data["export"]["model"]["quant_scheme"]["quant_type"]
                    assert set(recipe.deployment_precision.values()) == {quant_type}, config_path
                for resource_name, dotted_path in recipe.resource_config_bindings:
                    if resource_name in recipe.required_resources:
                        continue
                    value = data
                    for part in dotted_path.split("."):
                        value = value[part]
                    assert isinstance(value, str) and value, (config_path, dotted_path)


def test_registry_covers_each_public_artifact_kind():
    recipes = list_recipes()

    assert {recipe.artifact_format for recipe in recipes} == {"autoregressive", "vla", "tts"}
    assert {recipe.task for recipe in recipes} >= {
        "text-generation",
        "vision-language-generation",
        "action-prediction",
        "text-to-speech",
    }
    assert all(recipe.build_adapter and recipe.inference_adapter for recipe in recipes)


def test_qwen3_next_public_variants_use_the_pinned_gptq_recipe():
    description = describe_model("qwen3-next-80b-a3b", full=True)
    full = next(item for item in description["workflow_variants"] if item["name"] == "full")
    mtp = next(item for item in description["workflow_variants"] if item["name"] == "mtp")

    assert full["workflow_config"].endswith("qwen3_next_80b_a3b_full_gptq.yaml")
    assert mtp["workflow_config"].endswith("qwen3_next_80b_a3b_full_mtp_gptq.yaml")
    assert description["quant_methods"] == ["gptq"]
    assert description["quant_requirements"] == {"gptq": ["gptqmodel>=5.8,<5.9"]}
    assert description["quant_capabilities"] == {"gptq": ["gptqmodel.recipes.qwen3_next:quantize_qwen3_next"]}

    for model in ("qwen3.6-35b-a3b", "ornith-1.0-35b-a3b", "qwen3.5-122b-a10b"):
        assert describe_model(model, full=True)["quant_methods"] == ["autoround"]


def test_registry_listing_does_not_import_model_runtime_dependencies():
    script = (
        "import sys; "
        "from xhmodel_merak.model_build import describe_model, list_models; "
        "list_models(); "
        "describe_model('qwen3.5-9b'); "
        "describe_model('qwen3.5-9b', full=True); "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'xhmodel_merak.xh_llm' not in sys.modules; "
        "assert 'xhmodel_merak.xh_other_model' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_registry_exposes_executable_worker_dependency_profiles(monkeypatch):
    digest_cache_capability = (
        "python-predicate:xhquant.xhonnxruntime.llm_hmonnx_loader:initializer_digest_cache_disable_available"
    )
    assert "transformers>=5.13.0,<5.17.0" in describe_model("gemma4-12b-unified", full=True)["worker_requirements"]
    assert "transformers>=5.5.0,<5.17.0" in describe_model("qwen3.5-4b", full=True)["worker_requirements"]
    assert describe_model("qwen3.5-4b", full=True)["quant_requirements"]["autoround"] == [
        "auto-round>=0.12.0.dev0,<0.13.0",
        "gptqmodel>=5.8,<5.9",
    ]
    assert describe_model("qwen3.5-4b", full=True)["quant_capabilities"]["autoround"] == [
        "gptqmodel.recipes.qwen35_autoround:quantize_qwen35_autoround"
    ]
    assert describe_model("qwen3.5-4b", full=True)["quant_checkpoint_requirements"] == ["gptqmodel>=5.8,<5.9"]
    assert describe_model("gemma4-31b", full=True)["quant_requirements"]["gptq"] == [
        "gptqmodel>=5.8,<5.9",
    ]
    assert describe_model("gemma4-31b", full=True)["quant_capabilities"]["gptq"] == [
        "gptqmodel.recipes.gemma4:quantize_gemma4"
    ]
    assert describe_model("kokoro-82m-v1.1-zh", full=True)["worker_requirements"] == [
        "xhquant",
        "misaki==0.9.4",
        "cn2an",
        "jieba",
        "ordered-set",
        "pypinyin",
        "pypinyin-dict",
    ]
    assert describe_model("pi0.5-libero", full=True)["required_resources"] == ["tokenizer"]
    assert describe_model("pi0.5-libero", full=True)["worker_requirements"] == [
        "xhquant",
        "transformers>=5.3.0,<5.4.0",
        "lerobot==0.5.1",
    ]
    assert describe_model("qwen3.5-4b", full=True)["worker_capabilities"] == [digest_cache_capability]
    assert describe_model("gemma4-e2b", full=True)["worker_capabilities"] == [digest_cache_capability]
    assert describe_model("ling-3.0-flash", full=True)["worker_capabilities"] == [digest_cache_capability]
    assert describe_model("pi0.5-libero", full=True)["worker_capabilities"] == []
    assert describe_model("gemma4-e2b", full=True)["bundled_resources"] == {
        "calibration": "data/calib_data/NeelNanda-pile-10k.jsonl"
    }
    assert describe_model("qwen3.5-4b", full=True)["bundled_resources"] == {
        "calibration": "data/calib_data/NeelNanda-pile-10k.jsonl"
    }
    assert describe_model("qwen3.6-35b-a3b", full=True)["bundled_resources"] == {
        "calibration": "data/calib_data/NeelNanda-pile-10k.jsonl"
    }
    assert describe_model("ling-3.0-flash", full=True)["bundled_resources"] == {
        "calibration": "data/calib_data/NeelNanda-pile-10k.jsonl"
    }
    assert describe_model("gemma4-e2b", full=True)["quant_capabilities"]["autoround"] == [
        "python-predicate:xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter:autoround_mode1_available"
    ]

    monkeypatch.setattr(
        "xhmodel_merak._requirements._installed_distribution_version",
        lambda name: "4.57.1" if name == "transformers" else None,
    )
    monkeypatch.setattr("xhmodel_merak._requirements._module_available", lambda _name: True)
    with pytest.raises(ModelError, match="dependency profile"):
        validate_worker_requirements(
            ("xhquant", "transformers>=5.5.0,<5.17.0"),
            model="qwen3.5-4b",
            operation="build",
        )
