from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    AUTOROUND_REQUIREMENTS,
    GEMMA4_INFERENCE,
    GPTQ_REQUIREMENTS,
    XHQUANT_INITIALIZER_CACHE_CAPABILITY,
    llm_deployment_precision,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "gemma4"

_GEMMA4 = "configs_merak/workflows/xh2a/llm_models/gemma4_series"
_GEMMA4_RECIPE_CAPABILITY = "gptqmodel.recipes.gemma4:quantize_gemma4"
_GEMMA4_AUTOROUND_CAPABILITY = (
    "python-predicate:xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter:autoround_mode1_available"
)


def _recipe(
    model_id: str,
    display_name: str,
    export_name: str,
    config: str,
    *,
    method: str,
    max_pe_length: int,
    hidden_size: int,
    num_hidden_layers: int,
    audio: bool,
    moe: bool = False,
    bundled_resources: tuple[tuple[str, str], ...] = (),
) -> ModelRecipe:
    modalities = ("text", "image", "video", "audio") if audio else ("text", "image", "video")
    media_components = ("visual", "video_visual", "audio") if audio else ("visual", "video_visual")
    components = (*media_components, "prefill", "decode")
    unified = model_id == "gemma4-12b-unified"
    constraints: tuple[tuple[str, Any], ...] = (
        ("model_type", "gemma4_unified" if unified else "gemma4"),
        (
            "architectures.0",
            "Gemma4UnifiedForConditionalGeneration" if unified else "Gemma4ForConditionalGeneration",
        ),
        ("text_config.hidden_size", hidden_size),
        ("text_config.num_hidden_layers", num_hidden_layers),
        ("text_config.enable_moe_block", moe),
    )
    return ModelRecipe(
        model_id=model_id,
        display_name=display_name,
        family="gemma4-series",
        export_name=export_name,
        task="multimodal-generation",
        modalities=modalities,
        variants=(variant("full", config),),
        default_precision=ResolvedPrecision(weight="w4", activation="a8"),
        supported_precisions=(("w4", "a8"), ("w16", "a8")),
        quant_methods=(method,),
        deployment_precision=llm_deployment_precision(components),
        components=components,
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=GEMMA4_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(tuple(modality for modality in modalities if modality != "text")),
        output_schema=text_output_schema(),
        default_profile="padded-i280-v70",
        profiles=("padded-i280-v70",),
        default_context_length=2048,
        default_prefill_chunk_length=320,
        max_pe_length=max_pe_length,
        attention_profiles=("standard",),
        bundled_resources=bundled_resources,
        resource_config_bindings=((("calibration", "quant.calibration.jsonl"),) if bundled_resources else ()),
        source_constraints=constraints,
        worker_requirements=(
            "xhquant",
            "transformers>=5.13.0,<5.17.0" if unified else "transformers>=5.5.0,<5.17.0",
        ),
        worker_capabilities=(XHQUANT_INITIALIZER_CACHE_CAPABILITY,),
        quant_requirements=(
            (
                method,
                AUTOROUND_REQUIREMENTS if method == "autoround" else GPTQ_REQUIREMENTS,
            ),
        ),
        quant_capabilities=(
            (
                method,
                (_GEMMA4_AUTOROUND_CAPABILITY,) if method == "autoround" else (_GEMMA4_RECIPE_CAPABILITY,),
            ),
        ),
        quant_checkpoint_requirements=GPTQ_REQUIREMENTS,
        required_runtime_files=("per_layer_input_embedding",) if model_id in {"gemma4-e2b", "gemma4-e4b"} else (),
    )


def _recipes() -> tuple[ModelRecipe, ...]:
    return (
        _recipe(
            "gemma4-12b-unified",
            "Gemma4 12B Unified",
            "gemma_4_12b_unified",
            f"{_GEMMA4}/12b_unified/gemma4_12b_unified_full.yaml",
            method="gptq",
            max_pe_length=262144,
            hidden_size=3840,
            num_hidden_layers=48,
            audio=True,
        ),
        _recipe(
            "gemma4-e2b",
            "Gemma4 E2B",
            "gemma_4_e2b",
            f"{_GEMMA4}/e2b/gemma4_e2b_autoround.yaml",
            method="autoround",
            max_pe_length=131072,
            hidden_size=1536,
            num_hidden_layers=35,
            audio=True,
            bundled_resources=(("calibration", "data/calib_data/NeelNanda-pile-10k.jsonl"),),
        ),
        _recipe(
            "gemma4-e4b",
            "Gemma4 E4B",
            "gemma_4_e4b",
            f"{_GEMMA4}/e4b/gemma4_e4b_full.yaml",
            method="gptq",
            max_pe_length=131072,
            hidden_size=2560,
            num_hidden_layers=42,
            audio=True,
        ),
        _recipe(
            "gemma4-31b",
            "Gemma4 31B",
            "gemma_4_31b",
            f"{_GEMMA4}/31b/gemma4_31b_full.yaml",
            method="gptq",
            max_pe_length=262144,
            hidden_size=5376,
            num_hidden_layers=60,
            audio=False,
        ),
        _recipe(
            "gemma4-26b-a4b",
            "Gemma4 26B-A4B",
            "gemma_4_26b_a4b",
            f"{_GEMMA4}/26b_a4b/gemma4_26b_a4b_full.yaml",
            method="gptq",
            max_pe_length=262144,
            hidden_size=2816,
            num_hidden_layers=30,
            audio=False,
            moe=True,
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=GEMMA4_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.gemma4.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    for recipe in _recipes():
        catalog.register_recipe(recipe, provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
