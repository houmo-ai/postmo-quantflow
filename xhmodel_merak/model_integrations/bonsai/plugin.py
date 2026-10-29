from xhmodel_merak.model_catalog.contracts import ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_integrations._dependencies import runtime_dependencies
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD, QWEN_INFERENCE, text_media_input_schema, text_output_schema, variant,
)


PROVIDER_ID = "bonsai"
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_recipe(ModelRecipe(
        model_id="bonsai-27b-phase-a",
        display_name="Ternary Bonsai 27B (SSFP4 Phase A)",
        family="bonsai",
        export_name="bonsai_27b_phase_a",
        task="text-generation",
        modalities=("text",),
        source_constraints=(("text_config.hidden_size", 5120), ("text_config.num_hidden_layers", 64)),
        variants=(variant("full", "configs_merak/workflows/xh2a/llm_models/bonsai/bonsai_27b_phase_a.yaml"),),
        default_precision=ResolvedPrecision(weight="wmix", activation="a8"),
        supported_precisions=(("wmix", "a8"),),
        quant_methods=(),
        deployment_precision={"prefill": "w8a8h1_sefp", "decode": "w8a8h1_sefp"},
        components=("prefill", "decode"),
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=QWEN_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(()),
        output_schema=text_output_schema(),
        default_context_length=2048,
        default_prefill_chunk_length=64,
        max_pe_length=262144,
        dependencies=runtime_dependencies(),
    ), provider=PROVIDER_ID)
