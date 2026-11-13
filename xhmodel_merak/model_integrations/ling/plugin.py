from __future__ import annotations

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    LING_INFERENCE,
    PILE10K_RESOURCE,
    llm_deployment_precision,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "ling"

_LING3 = "configs_merak/workflows/xh2a/llm_models/ling_3_flash"


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=("fla-core==0.5.2",),
            quantization=(("autoround", AUTOROUND_REQUIREMENTS),),
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id="ling-3.0-flash",
        display_name="Ling 3.0 Flash",
        family="ling-3.0",
        export_name="ling_3_flash",
        task="text-generation",
        modalities=("text",),
        variants=(
            variant(
                "full",
                f"{_LING3}/ling_3_flash_autoround.yaml",
                export_modes=("low_memory", "standard"),
            ),
        ),
        default_precision=ResolvedPrecision(
            weight="wmix",
            activation="amix",
            components={
                "attention": {"weight": "w8", "activation": "a8"},
                "dense_mlp": {"weight": "w8", "activation": "a8"},
                "shared_expert": {"weight": "w8", "activation": "a8"},
                "routed_expert": {"weight": "w4", "activation": "a8"},
            },
        ),
        supported_precisions=(("wmix", "amix"),),
        quant_methods=("autoround",),
        quant_checkpoint_methods=("gptq", "autoround"),
        checkpoint_embedding_suffixes=("word_embeddings.weight",),
        checkpoint_auxiliary_layers_field="num_nextn_predict_layers",
        deployment_precision=llm_deployment_precision(("prefill", "decode")),
        components=("prefill", "decode"),
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=LING_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(()),
        output_schema=text_output_schema(),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        attention_profiles=("standard",),
        bundled_resources=(PILE10K_RESOURCE,),
        resource_config_bindings=(("calibration", "quant.calibration.jsonl"),),
        config_replacements=(
            ("export.model.fuse_gdr_ops", True),
            ("export.model.fuse_gdr_block_recurrent_ops", True),
        ),
        source_constraints=(
            ("architectures.0", "BailingMoeV3ForCausalLM"),
            ("hidden_size", 2560),
            ("num_hidden_layers", 42),
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=LING_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.ling.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
