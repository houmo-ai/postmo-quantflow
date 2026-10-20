import inspect
import json
import subprocess
import sys

import pytest
import yaml
from _unified_cases import MODELS

from xhmodel_merak._requirements import validate_worker_requirements
from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import build_model, describe_model, list_models
from xhmodel_merak.model_catalog import list_recipes, resolve_workflow_config
from xhmodel_merak.model_catalog.registry import get_adapter_spec, get_artifact_format_spec, get_catalog


EXPECTED_VARIANTS = {model: list(variants) for model, (_, variants) in MODELS.items()}


def test_registry_exposes_the_stable_cross_task_matrix():
    models = list_models()

    assert {item["model_id"]: item["variants"] for item in models} == EXPECTED_VARIANTS
    assert [item["model_id"] for item in models] == sorted(EXPECTED_VARIANTS)


@pytest.mark.parametrize("model_id,variants", EXPECTED_VARIANTS.items())
def test_full_description_matches_public_model_and_variants(model_id, variants):
    description = describe_model(model_id, full=True)

    assert description["model_id"] == model_id
    assert description["variants"] == variants
    assert description["default_variant"] == variants[0]
    assert [item["name"] for item in description["workflow_variants"]] == variants


@pytest.mark.parametrize("model_id", EXPECTED_VARIANTS)
def test_default_description_only_lists_editable_build_arguments(model_id):
    parameters = describe_model(model_id)

    assert parameters == describe_model(model_id, full=False)
    json.dumps(parameters)
    assert set(parameters) <= set(inspect.signature(build_model).parameters)
    assert parameters["model_dir"]["required"] is True
    assert parameters["output_dir"]["default"] is None
    assert parameters["device"]["default"] is None
    assert not set(parameters) & {
        "model_id",
        "components",
        "vision",
        "attention",
        "max_pe_length",
    }
    assert all("required" in item or "default" in item for item in parameters.values())


def test_qwen_description_has_editable_defaults_and_correlated_choices():
    parameters = describe_model("qwen3.5-4b")

    assert set(parameters) == {
        "model_dir",
        "output_dir",
        "device",
        "low_memory",
        "precision",
        "deployment_precision",
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
    parameters["deployment_precision"]["components"]["visual"]["choices"].clear()

    fresh = describe_model("qwen3.5-9b")
    assert fresh["precision"]["default"]["weight"] == "w4"
    assert fresh["precision"]["choices"][0]["weight"] == "w4"
    assert fresh["speculative"]["mode"]["choices"] == ["mtp", "dflash"]
    assert fresh["sources"]["required_by_variant"] == {"dflash": ["dflash"]}
    assert fresh["deployment_precision"]["components"]["visual"]["choices"] == ["w8a8h1_sefp", "w8a16h1_sefp"]


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
    assert get_adapter_spec("inference", "paddleocr-vl.inference").target == (
        "xhmodel_merak.model_integrations.paddleocr_vl.inference:adapter"
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
        if model == "paddleocr-vl-1.6":
            assert description["vision"] == {
                "mode": "dynamic_token_gears",
                "profile": "visualm96-196-384-704-1024-1280",
                "image_token_gears": [96, 196, 384, 704, 1024, 1280],
                "static_profiles": [],
            }
        elif model == "ocr-rm-2b-value":
            assert description["vision"] == {
                "mode": "dynamic_token_gears",
                "profile": "visualm96-384-1536",
                "image_token_gears": [96, 384, 1536],
                "static_profiles": [],
            }
        elif model.startswith("wemm-embedding"):
            assert description["vision"] == {
                "mode": "dynamic_token_gears",
                "profile": "visualm96-196-384-704-1536",
                "image_token_gears": [96, 196, 384, 704, 1536],
                "static_profiles": [],
            }
        elif not model.startswith(("macaron", "qwen", "ornith")) or model in {
            "qwen3-next-80b-a3b",
            "qwen3.5-moe-text",
            "qwen-image-2.1",
            "qwen-image-2.1-turbo",
        }:
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


def test_paddleocr_vl_description_exposes_document_runtime_contract():
    description = describe_model("paddleocr-vl-1.6", full=True)

    assert description["workflow_variants"] == [
        {
            "dependencies": description["dependencies"]["profiles"]["full"],
            "name": "full",
            "workflow_config": (
                "configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6_r1_autoround_w8_token_gears.yaml"
            ),
            "speculative_mode": None,
            "requires_lora": False,
            "required_sources": [],
            "extra_components": [],
            "context_workflows": [],
            "export_modes": ["standard"],
        }
    ]
    assert description["input_schema"]["required"] == ["prompt", "image"]
    assert description["input_schema"]["properties"]["max_new_tokens"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 4096,
        "default": 1024,
    }
    assert description["defaults"]["vision_profile"] == "visualm96-196-384-704-1024-1280"
    assert description["vision"]["image_token_gears"] == [96, 196, 384, 704, 1024, 1280]
    assert description["required_resources"] == ["calibration"]
    assert description["quant_checkpoint_methods"] == ["autoround"]
    assert description["worker_capabilities"] == []


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
            "document-understanding",
            "text-generation",
            "vision-language-generation",
        }:
            continue
        schema = description["output_schema"]
        if model == "minicpm-o-4.5":
            assert [item["required"] for item in schema["oneOf"]] == [
                ["output_text", "output_tokens"],
                ["waveform", "sample_rate", "stats"],
            ]
        else:
            assert schema["required"] == ["output_text", "output_tokens"]


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
                if quant is not None:
                    method = quant["method"] if "method" in quant else {"gptqmodel": "gptq"}[quant["algorithm"]]
                    assert method in recipe.quant_methods, config_path
                elif recipe.quant_methods:
                    assert recipe.default_precision.weight == "w16" and recipe.config_resolver is not None, config_path
                else:
                    assert quant is None, config_path
                if recipe.artifact_format == "autoregressive":
                    model = data["export"]["model"]
                    assert set(recipe.deployment_precision) == set(recipe.components), config_path
                    for component, precision in recipe.deployment_precision.items():
                        component_config = model.get(f"{component}_config") or {}
                        scheme = component_config.get("quant_scheme")
                        if scheme is None and component == "video_visual":
                            scheme = (model.get("visual_config") or {}).get("quant_scheme")
                        if scheme is None:
                            scheme = model["quant_scheme"]
                        assert precision == scheme["quant_type"], (config_path, component)
                for resource_name, dotted_path in recipe.resource_config_bindings:
                    if resource_name in recipe.required_resources:
                        continue
                    value = data
                    for part in dotted_path.split("."):
                        value = value[part]
                    assert isinstance(value, str) and value, (config_path, dotted_path)


def test_registry_covers_each_public_artifact_kind():
    recipes = list_recipes()

    artifact_formats = {recipe.artifact_format for recipe in recipes}
    assert artifact_formats == set(get_catalog()._artifact_formats)
    assert artifact_formats >= {
        "aigc",
        "gaussian-scene",
        "autoregressive",
        "asr",
        "tts",
        "vla",
        "glm53",
        "qwen_image21",
        "flux3_klein",
        "wall_oss",
        "qwen4-exp",
        "minicpm_v_4_7",
        "minicpm_o_4_5",
        "sol_h3",
        "qwen3-vl-reranker",
    }
    assert {recipe.task for recipe in recipes} >= {
        "text-generation",
        "vision-language-generation",
        "action-prediction",
        "text-to-speech",
        "speech-transcription",
        "image-to-video",
        "image-generation",
        "multimodal-reranking",
    }
    assert all(recipe.build_adapter and recipe.inference_adapter for recipe in recipes)


def test_moss_transcribe_diarize_public_io_is_audio_prompt_text():
    description = describe_model("moss-transcribe-diarize-0.9b", full=True)

    assert description["task"] == "speech-transcription"
    assert description["vision"] is None
    assert description["input_schema"]["required"] == ["audio"]
    assert "prompt" in description["input_schema"]["properties"]
    generation_limit = description["input_schema"]["properties"]["max_new_tokens"]
    assert "default" not in generation_limit
    assert generation_limit["maximum"] == 65536
    assert description["output_schema"]["required"] == ["output_text"]
    parameters = describe_model("moss-transcribe-diarize-0.9b")
    assert parameters["context_length"]["default"] == 2048
    assert parameters["context_length"]["maximum"] == 262144
    assert parameters["prefill_chunk_length"]["default"] == 512


def _write_moss_bundle(model_dir):
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"moss_transcribe_diarize","text_config":{"model_type":"qwen3"},'
        '"audio_config":{"model_type":"whisper"},"audio_merge_size":4,"adaptor_input_dim":4096}'
    )
    (model_dir / "processor_config.json").write_text(
        '{"audio_tokens_per_second":12.5,"audio_merge_size":4,"time_marker_every_seconds":5,"enable_time_marker":true}'
    )
    (model_dir / "preprocessor_config.json").write_text(
        '{"sampling_rate":16000,"feature_size":80,"hop_length":160,"n_fft":400,"n_samples":480000,"nb_max_frames":3000}'
    )
    (model_dir / "generation_config.json").write_text('{"max_new_tokens":5120,"eos_token_id":151645}')
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model_dir / name).write_text("{}")
    for name in (
        "configuration_moss_transcribe_diarize.py",
        "modeling_moss_transcribe_diarize.py",
        "processing_moss_transcribe_diarize.py",
    ):
        (model_dir / name).write_text("# remote-code fixture\n")
    (model_dir / "model.safetensors.index.json").write_text('{"weight_map":{"model":"model.safetensors"}}')
    (model_dir / "model.safetensors").write_bytes(b"fixture")


@pytest.mark.parametrize(
    ("context_length", "artifact_name", "prefill", "max_pe", "workflow_suffix"),
    [
        (
            2048,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_2k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize.yaml",
        ),
        (
            4096,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_4k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize.yaml",
        ),
        (
            8192,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_8k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize_16k.yaml",
        ),
        (
            16384,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_16k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize_16k.yaml",
        ),
        (
            65536,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_64k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize_64k.yaml",
        ),
        (
            131072,
            "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_512_128k_mpe256k",
            512,
            262144,
            "moss_transcribe_diarize_128k.yaml",
        ),
    ],
)
def test_moss_transcribe_diarize_has_a_stable_artifact_name_contract(
    tmp_path, context_length, artifact_name, prefill, max_pe, workflow_suffix
):
    model_dir = tmp_path / "model"
    _write_moss_bundle(model_dir)

    from xhmodel_merak.model_build.resolver import resolve_build_plan

    plan = resolve_build_plan(
        model_id="moss-transcribe-diarize-0.9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        device="cpu",
        context_length=context_length,
    )

    assert plan.artifact_name == artifact_name
    assert plan.context_length == context_length
    assert plan.config_overrides["export.model.context_max_length"] == context_length
    assert plan.config_overrides["export.model.prefill_chunk_length"] == prefill
    assert plan.config_replacements["export.model.max_pe_length"] == max_pe
    assert plan.workflow_config.endswith(workflow_suffix)


def test_moss_transcribe_diarize_keeps_explicit_short_prefill_on_16k(tmp_path):
    model_dir = tmp_path / "model"
    _write_moss_bundle(model_dir)
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    plan = resolve_build_plan(
        model_id="moss-transcribe-diarize-0.9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        device="cpu",
        context_length=16384,
        prefill_chunk_length=256,
    )

    assert plan.config_overrides["export.model.prefill_chunk_length"] == 256
    assert plan.artifact_name == "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_256_16k_mpe256k"


def test_moss_transcribe_diarize_thirty_minute_build_passes_explicit_prefill(tmp_path):
    model_dir = tmp_path / "model"
    _write_moss_bundle(model_dir)
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    plan = resolve_build_plan(
        model_id="moss-transcribe-diarize-0.9b",
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "artifact"),
        device="cpu",
        context_length=65536,
        prefill_chunk_length=4096,
    )

    assert plan.config_overrides["export.model.prefill_chunk_length"] == 4096
    assert plan.workflow_config.endswith("moss_transcribe_diarize_64k.yaml")
    assert plan.artifact_name == "hmquant_xh2_moss_transcribe_diarize_09b_w8a8_4k_64k_mpe256k"


def test_qwen3_next_public_variants_use_the_pinned_gptq_recipe():
    description = describe_model("qwen3-next-80b-a3b", full=True)
    full = next(item for item in description["workflow_variants"] if item["name"] == "full")
    mtp = next(item for item in description["workflow_variants"] if item["name"] == "mtp")

    assert full["workflow_config"].endswith("qwen3_next_80b_a3b_full_gptq.yaml")
    assert mtp["workflow_config"].endswith("qwen3_next_80b_a3b_full_mtp_gptq.yaml")
    assert description["quant_methods"] == ["gptq"]
    assert "gptqmodel==5.8.0" in description["dependencies"]["profiles"]["full"]["quantization"]["gptq"]["pip"]
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
    def profile(model):
        description = describe_model(model, full=True)
        return next(iter(description["dependencies"]["profiles"].values()))

    assert "transformers==5.16.1" in profile("gemma4-12b-unified")["inference"]["pip"]
    assert "transformers==5.16.1" in profile("qwen3.5-4b")["inference"]["pip"]
    assert "qwen-vl-utils==0.0.14" in profile("qwen3.5-4b")["inference"]["pip"]
    assert "qwen-vl-utils==0.0.14" in profile("wemm-embedding-2b")["inference"]["pip"]
    assert "torchvision==0.23.0" in profile("wemm-embedding-2b")["inference"]["pip"]
    assert "qwen-vl-utils==0.0.14" not in profile("qwen3-next-80b-a3b")["inference"]["pip"]
    assert {"auto-round==0.12.0", "gptqmodel==5.8.0"} <= set(profile("qwen3.5-4b")["quantization"]["autoround"]["pip"])
    assert describe_model("qwen3.5-4b", full=True)["quant_capabilities"]["autoround"] == [
        "gptqmodel.recipes.qwen35_autoround:quantize_qwen35_autoround"
    ]
    assert "gptqmodel==5.8.0" in profile("qwen3.5-4b")["checkpoint_build"]["pip"]
    assert "gptqmodel==5.8.0" in profile("gemma4-31b")["quantization"]["gptq"]["pip"]
    assert describe_model("gemma4-31b", full=True)["quant_capabilities"]["gptq"] == [
        "gptqmodel.recipes.gemma4:quantize_gemma4"
    ]
    assert "misaki[en,zh]==0.9.4" in profile("kokoro-82m-v1.1-zh")["inference"]["pip"]
    assert describe_model("pi0.5-libero", full=True)["required_resources"] == ["tokenizer"]
    assert {"transformers==5.16.1", "lerobot==0.6.0"} <= set(
        profile("pi0.5-libero")["inference"]["pip"]
    )
    assert describe_model("qwen3.5-4b", full=True)["worker_capabilities"] == []
    assert describe_model("gemma4-e2b", full=True)["worker_capabilities"] == []
    assert describe_model("ling-3.0-flash", full=True)["worker_capabilities"] == []
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
            ("xhquant", "transformers==5.16.1"),
            model="qwen3.5-4b",
            operation="build",
        )
