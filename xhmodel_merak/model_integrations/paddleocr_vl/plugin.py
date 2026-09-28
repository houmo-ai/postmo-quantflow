from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    IMAGE,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    text_output_schema,
    variant,
)


PROVIDER_ID = "paddleocr-vl"
PADDLEOCR_VL_INFERENCE = "paddleocr-vl.inference"

_WORKFLOW = (
    "configs_merak/workflows/xh2a/llm_models/paddleocr_vl/"
    "paddleocr_vl_1_6_r1_autoround_w8_token_gears.yaml"
)
_VISION_GEARS = (96, 196, 384, 704, 1024, 1280)


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "image": {"type": "string", "minLength": 1, "description": "Local document image path"},
            "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 1024},
        },
        "required": ["prompt", "image"],
        "additionalProperties": False,
        "maxMediaInputs": 1,
    }


def _recipe() -> ModelRecipe:
    components = ("visual", "prefill", "decode")
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=IMAGE,
            quantization=(("autoround", AUTOROUND_REQUIREMENTS),),
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id="paddleocr-vl-1.6",
        display_name="PaddleOCR-VL 1.6",
        family="paddleocr-vl",
        config_resolver="xhmodel_merak.model_integrations.paddleocr_vl.planning:resolve_config",
        export_name="paddleocr_vl_1_6",
        task="document-understanding",
        modalities=("text", "image"),
        variants=(variant("full", _WORKFLOW),),
        default_precision=ResolvedPrecision(
            weight="w8",
            activation="a8",
            components={
                "language_model": {"weight": "w8", "activation": "a8"},
                "vision_encoder": {"weight": "w8", "activation": "a16"},
            },
        ),
        supported_precisions=(("w8", "a8"),),
        quant_methods=("autoround",),
        quant_checkpoint_methods=("autoround",),
        deployment_precision={
            "visual": "w8a16h1_sefp",
            "prefill": "w8a8_sefp",
            "decode": "w8a8_sefp",
        },
        components=components,
        golden_components=components,
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=PADDLEOCR_VL_INFERENCE,
        artifact_format="autoregressive",
        input_schema=_input_schema(),
        output_schema=text_output_schema(),
        default_profile="visualm96-196-384-704-1024-1280",
        profiles=("visualm96-196-384-704-1024-1280",),
        default_context_length=16384,
        default_prefill_chunk_length=256,
        max_pe_length=16384,
        vision_profile="visualm96-196-384-704-1024-1280",
        vision_gears=_VISION_GEARS,
        attention_profiles=("standard",),
        required_resources=("calibration",),
        resource_config_bindings=(("calibration", "quant.calibration_jsonl"),),
        source_constraints=(
            ("model_type", "paddleocr_vl"),
            ("architectures.0", "PaddleOCRVLForConditionalGeneration"),
        ),
        quant_capabilities=(("autoround", ("recipes.paddleocr_vl:quantize_paddleocr_vl",)),),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=PADDLEOCR_VL_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.paddleocr_vl.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
