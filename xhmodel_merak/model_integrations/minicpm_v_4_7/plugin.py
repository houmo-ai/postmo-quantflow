from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    IMAGE,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    MINICPM_V_4_7_BUILD,
    MINICPM_V_4_7_INFERENCE,
    text_output_schema,
    variant,
)


PROVIDER_ID = "minicpm-v-4-7"

_COMPONENTS = ("vision_4x", "vision_16x", "llm")
_DENSE_WORKFLOW = (
    "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/"
    "minicpm_v_4_7_xh2a_w8a8.yaml"
)
_MOE_WORKFLOW = (
    "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/"
    "minicpm_v_4_7_moe_xh2a_w8a8.yaml"
)


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "image": {
                "type": "string",
                "minLength": 1,
                "description": "Local image path for MiniCPM-V-4.7 multimodal generation",
            },
            "max_new_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1024,
                "default": 64,
            },
            "downsample_mode": {
                "type": "string",
                "enum": ["4x", "16x"],
                "default": "4x",
                "description": "Vision token-capacity profile exported by the workflow",
            },
        },
        "required": ["prompt", "image"],
        "additionalProperties": False,
        "maxMediaInputs": 1,
    }


def _dense_recipe() -> ModelRecipe:
    return ModelRecipe(
        # Default public workflows are W8A8 with quant: null. Keep W4A8 YAML
        # examples available outside the registered default/context matrix.
        dependencies=runtime_dependencies(common=IMAGE, transformers="5.18.0"),
        model_id="minicpm-v-4.7-1b",
        display_name="MiniCPM-V 4.7 1B",
        family="minicpm-v-4.7",
        config_resolver="xhmodel_merak.model_integrations.minicpm_v_4_7.planning:resolve_config",
        export_name="minicpm_v_4_7_1b",
        task="vision-language-generation",
        modalities=("text", "image"),
        variants=(
            variant("full", _DENSE_WORKFLOW),
            VariantRecipe(
                name="dspark",
                workflow_config=_DENSE_WORKFLOW.replace("w8a8.yaml", "w8a8_dspark.yaml"),
                speculative_mode="dspark",
                required_sources=("dspark",),
                extra_components=("dspark",),
            ),
        ),
        default_precision=ResolvedPrecision(weight="w8", activation="a8"),
        supported_precisions=(("w8", "a8"),),
        quant_methods=(),
        quant_checkpoint_methods=(),
        deployment_precision={component: "w8a8h1_sefp" for component in _COMPONENTS},
        components=_COMPONENTS,
        golden_components=(),
        build_adapter=MINICPM_V_4_7_BUILD,
        inference_adapter=MINICPM_V_4_7_INFERENCE,
        artifact_format="minicpm_v_4_7",
        input_schema=_input_schema(),
        output_schema=text_output_schema(),
        default_profile="full",
        profiles=("full",),
        default_context_length=8192,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        attention_profiles=("standard",),
        source_constraints=(
            ("model_type", "minicpmv4_7"),
            ("architectures.0", "MiniCPMV4_7ForConditionalGeneration"),
            ("text_config.model_type", "qwen3_5_text"),
            ("text_config.hidden_size", 1024),
            ("text_config.num_hidden_layers", 24),
        ),
    )


def _moe_recipe() -> ModelRecipe:
    return ModelRecipe(
        # Default public workflows are W8A8 with quant: null. Keep W4A8 YAML
        # examples available outside the registered default/context matrix.
        dependencies=runtime_dependencies(common=IMAGE, transformers="5.18.0"),
        model_id="minicpm-v-4.7-35b-a3b",
        display_name="MiniCPM-V 4.7 35B-A3B",
        family="minicpm-v-4.7",
        config_resolver="xhmodel_merak.model_integrations.minicpm_v_4_7.planning:resolve_config",
        export_name="minicpm_v_4_7_35b_a3b",
        task="vision-language-generation",
        modalities=("text", "image"),
        variants=(
            variant("full", _MOE_WORKFLOW),
            VariantRecipe(
                name="dspark",
                workflow_config=_MOE_WORKFLOW.replace("w8a8.yaml", "w8a8_dspark.yaml"),
                speculative_mode="dspark",
                required_sources=("dspark",),
                extra_components=("dspark",),
            ),
        ),
        default_precision=ResolvedPrecision(weight="w8", activation="a8"),
        supported_precisions=(("w8", "a8"),),
        quant_methods=(),
        quant_checkpoint_methods=(),
        deployment_precision={component: "w8a8h1_sefp" for component in _COMPONENTS},
        components=_COMPONENTS,
        golden_components=(),
        build_adapter=MINICPM_V_4_7_BUILD,
        inference_adapter=MINICPM_V_4_7_INFERENCE,
        artifact_format="minicpm_v_4_7",
        input_schema=_input_schema(),
        output_schema=text_output_schema(),
        default_profile="full",
        profiles=("full",),
        default_context_length=8192,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        attention_profiles=("standard",),
        source_constraints=(
            ("model_type", "minicpmv4_7"),
            ("architectures.0", "MiniCPMV4_7ForConditionalGeneration"),
            ("text_config.model_type", "qwen3_5_moe_text"),
            ("text_config.hidden_size", 2048),
            ("text_config.num_hidden_layers", 40),
            ("text_config.num_experts", 256),
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(
        ArtifactFormatSpec(name="minicpm_v_4_7"),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=MINICPM_V_4_7_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.minicpm_v_4_7.build",
            artifact_formats=("minicpm_v_4_7",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=MINICPM_V_4_7_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.minicpm_v_4_7.inference:adapter",
            artifact_formats=("minicpm_v_4_7",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_dense_recipe(), provider=PROVIDER_ID)
    catalog.register_recipe(_moe_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
