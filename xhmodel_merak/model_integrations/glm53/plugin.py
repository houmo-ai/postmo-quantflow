from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_integrations._shared import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    PILE10K_RESOURCE,
    text_media_input_schema,
    text_output_schema,
    variant,
)

PROVIDER_ID = "glm53"
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_artifact_format(
        ArtifactFormatSpec(name="glm53", validator="xhmodel_merak.model_integrations.glm53.artifact:validate_artifact"),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="glm53.build",
            kind="build",
            target="xhmodel_merak.model_integrations.glm53.build",
            artifact_formats=("glm53",),
            source_config_for_embeddings=True,
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="glm53.inference",
            kind="inference",
            target="xhmodel_merak.model_integrations.glm53.inference:adapter",
            artifact_formats=("glm53",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(
        ModelRecipe(
            model_id="glm-5.3-flash",
            display_name="GLM 5.3 Flash",
            family="glm-5.3",
            export_name="glm5_3_flash_w4g64",
            task="text-generation",
            modalities=("text", "image", "video"),
            variants=(
                variant(
                    "full",
                    "configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_autoround_w4g64.yaml",
                    export_modes=("low_memory",),
                ),
            ),
            default_precision=ResolvedPrecision(
                weight="wmix",
                activation="a16",
                components={
                    "attention": {"weight": "w8", "activation": "a16"},
                    "routed_expert": {"weight": "w4", "activation": "a16"},
                    "shared_expert": {"weight": "w8", "activation": "a16"},
                    "dense_mlp": {"weight": "w8", "activation": "a16"},
                    "vision": {"weight": "w8", "activation": "a16"},
                    "linear_decay_beta": {"weight": "fp16", "activation": "a16"},
                },
            ),
            supported_precisions=(("wmix", "a16"),),
            quant_methods=("autoround",),
            quant_checkpoint_methods=("autoround",),
            checkpoint_auxiliary_layers_field="num_nextn_predict_layers",
            deployment_precision={"prefill": "w8a16h0_sefp", "decode": "w8a16h0_sefp", "vision": "w8a16h0_sefp"},
            components=("prefill", "decode", "vision"),
            golden_components=(),
            build_adapter="glm53.build",
            inference_adapter="glm53.inference",
            artifact_format="glm53",
            input_schema=text_media_input_schema(("image", "video")),
            output_schema=text_output_schema(),
            default_context_length=4096,
            default_prefill_chunk_length=256,
            max_pe_length=1048576,
            attention_profiles=("standard",),
            bundled_resources=(PILE10K_RESOURCE,),
            resource_config_bindings=(("calibration", "quant.calibration.jsonl"),),
            source_constraints=(
                ("model_type", "glm5_next"),
                ("architectures.0", "Glm5NextForConditionalGeneration"),
                ("text_config.hidden_size", 4096),
                ("text_config.num_hidden_layers", 45),
            ),
            worker_requirements=("xhquant", "transformers>=5.16.1,<5.17.0"),
            quant_requirements=(("autoround", AUTOROUND_REQUIREMENTS),),
            quant_capabilities=(("autoround", (
                "gptqmodel.recipes.glm5_3_flash_autoround:quantize_glm5_3_flash_autoround",
            )),),
            quant_checkpoint_requirements=GPTQ_REQUIREMENTS,
            environment={"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"},
        ),
        provider=PROVIDER_ID,
    )
