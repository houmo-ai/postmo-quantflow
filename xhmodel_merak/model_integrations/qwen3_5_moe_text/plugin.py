from __future__ import annotations

from dataclasses import replace

from xhmodel_merak.model_catalog.contracts import ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import runtime_dependencies
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    QWEN_INFERENCE,
    autoregressive_precision_bindings,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "qwen3.5-moe-text"
_QWEN35_MOE = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe"


def _recipe() -> ModelRecipe:
    """Native text architecture; weight products share this execution contract."""
    components = ("prefill", "decode")
    return ModelRecipe(
        model_id="qwen3.5-moe-text",
        recipe_version="3",
        display_name="Qwen3.5 MoE Text",
        family="qwen3.5-moe",
        export_name="qwen3_5_moe_text",
        task="text-generation",
        modalities=("text",),
        variants=(variant(
            "full", f"{_QWEN35_MOE}/122b_a10b/qwen3_6_122b_a10b_reap90_autoround.yaml",
        ),),
        default_precision=ResolvedPrecision(weight="w4", activation="a8"),
        supported_precisions=(("w4", "a8"), ("w8", "a8"), ("wmix", "a8"), ("w16", "a8")),
        quant_methods=(),
        quant_checkpoint_methods=("autoround",),
        dependencies=runtime_dependencies(),
        deployment_precision={component: "w4a8h0_ssfp" for component in components},
        deployment_precision_bindings=tuple(
            replace(binding, choices=("w4a8h0_ssfp", "w8a8h1_sefp"))
            for binding in autoregressive_precision_bindings(components)
        ),
        components=components,
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=QWEN_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(()),
        output_schema=text_output_schema(),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        attention_profiles=("standard",),
        config_replacements=(("export.model.model_name", "qwen3_5_moe_text"),),
        source_constraints=(
            ("model_type", "qwen3_5_moe_text"),
            ("architectures.0", "Qwen3_5MoeForCausalLM"),
        ),
    )



def register(catalog: ModelCatalog) -> None:
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
