"""Lightweight catalog metadata for the existing Xing4.0 workflow."""

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import GPTQ_REQUIREMENTS, runtime_dependencies
from xhmodel_merak.model_integrations._shared import (
    PILE10K_RESOURCE,
    text_media_input_schema,
    text_output_schema,
    variant,
)

PROVIDER_ID = "xing4_0"


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name="xing4_0.build", kind="build",
            target="xhmodel_merak.model_integrations.xing4_0.build",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="xing4_0.inference", kind="inference",
            target="xhmodel_merak.model_integrations.xing4_0.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(
        ModelRecipe(
            model_id="xing4.0-29b-a4b", display_name="Xing4.0-29B-A4B",
            family="xing4.0", export_name="xing4_0_29b_a4b", task="text-generation",
            modalities=("text",),
            variants=(variant(
                "full", "configs_merak/workflows/xh2a/llm_models/xing4_0/29b_a4b/xing4_0_29b_a4b_xh2a_w4a8.yaml",
            ),),
            # Xing validates its fixed region layout before export and publication.
            default_precision=ResolvedPrecision(weight="wmix", activation="amix"),
            supported_precisions=(("wmix", "amix"),),
            quant_methods=("gptq",),
            quant_checkpoint_methods=("gptq", "autoround"),
            checkpoint_auxiliary_layers_field="num_nextn_predict_layers",
            config_resolver="xhmodel_merak.model_integrations.xing4_0.planning:resolve_config",
            deployment_precision={"prefill": "w4a8h0_ssfp", "decode": "w4a8h0_ssfp"},
            components=("prefill", "decode"), golden_components=(),
            build_adapter="xing4_0.build", inference_adapter="xing4_0.inference",
            artifact_format="autoregressive",
            input_schema=text_media_input_schema(()), output_schema=text_output_schema(),
            default_context_length=2048, default_prefill_chunk_length=256, max_pe_length=262144,
            attention_profiles=("standard",),
            bundled_resources=(PILE10K_RESOURCE,),
            resource_config_bindings=(("calibration", "quant.calibration.jsonl"),),
            source_constraints=(
                ("architectures.0", "Xing4_0ForCausalLM"), ("model_type", "xing4_0"),
                ("hidden_size", 3584), ("num_hidden_layers", 40),
                ("num_attention_heads", 32), ("n_routed_experts", 64),
                ("num_experts_per_tok", 4), ("q_lora_rank", 768), ("kv_lora_rank", 512),
                ("qk_rope_head_dim", 64),
                ("hc_mult", 4),
            ),
            dependencies=runtime_dependencies(
                quantization=(("gptq", GPTQ_REQUIREMENTS),),
                checkpoint_build=GPTQ_REQUIREMENTS,
            ),
        ),
        provider=PROVIDER_ID,
    )


__all__ = ["PROVIDER_ID", "register"]
