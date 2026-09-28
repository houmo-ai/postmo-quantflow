"""Lightweight public recipes for the WeMM embedding checkpoints."""

from __future__ import annotations

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision, VariantRecipe
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    WEMM_MEDIA,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
)


PROVIDER_ID = "wemm-embedding"
INFERENCE_ADAPTER = "wemm-embedding.inference"
_CONFIG_ROOT = "configs_merak/workflows/xh2a/llm_models/wemm_embedding"
_GEARS = (96, 196, 384, 704, 1536)


def _input_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "image": {"type": "string", "minLength": 1, "description": "Local image path"},
            "video": {"type": "string", "minLength": 1, "description": "Local video path"},
        },
        "required": [],
        "additionalProperties": False,
        "maxMediaInputs": 2,
    }


def _output_schema() -> dict:
    return {
        "type": "object",
        "properties": {"embedding": {"type": "array", "items": {"type": "number"}}},
        "required": ["embedding"],
    }


def _recipe(size: str, hidden_size: int, num_layers: int) -> ModelRecipe:
    name = f"wemm_embedding_{size}"
    root = f"{_CONFIG_ROOT}/{size}"
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=WEMM_MEDIA,
            system=("FFmpeg shared libraries",),
            quantization=(("autoround", AUTOROUND_REQUIREMENTS),),
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id=f"wemm-embedding-{size}",
        display_name=f"WeMM-Embedding {size.upper()}",
        family="wemm-embedding",
        export_name=name,
        task="multimodal-embedding",
        modalities=("text", "image", "video"),
        variants=(VariantRecipe("full", f"{root}/{name}_xh2a_w4a8.yaml"),),
        default_precision=ResolvedPrecision(weight="w4", activation="a8"),
        supported_precisions=(("w4", "a8"),),
        quant_methods=("autoround",),
        quant_checkpoint_methods=("gptq", "autoround"),
        deployment_precision={"visual": "w8a8h1_sefp", "prefill": "w4a8h0_ssfp"},
        components=("visual", "prefill"),
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=INFERENCE_ADAPTER,
        artifact_format="autoregressive",
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        default_profile="visualm96-196-384-704-1536",
        profiles=("visualm96-196-384-704-1536",),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        vision_profile="visualm96-196-384-704-1536",
        vision_gears=_GEARS,
        attention_profiles=("standard",),
        config_resolver="xhmodel_merak.model_integrations.qwen.planning:resolve_config",
        config_type_aliases=(("vision_config.model_type", "qwen3_5", "qwen3_5_vision"),),
        source_constraints=(
            ("model_type", "qwen3_5"),
            ("architectures.0", "Qwen3_5ForConditionalGeneration"),
            ("text_config.hidden_size", hidden_size),
            ("text_config.num_hidden_layers", num_layers),
        ),
        quant_capabilities=(("autoround", ("gptqmodel.recipes.qwen35_autoround:quantize_qwen35_autoround",)),),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=INFERENCE_ADAPTER,
            kind="inference",
            target="xhmodel_merak.model_integrations.wemm_embedding.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    for size, hidden_size, num_layers in (("2b", 2048, 24), ("4b", 2560, 32), ("9b", 4096, 32)):
        catalog.register_recipe(_recipe(size, hidden_size, num_layers), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
