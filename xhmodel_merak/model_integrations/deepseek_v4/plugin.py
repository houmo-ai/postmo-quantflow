from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision, VariantRecipe
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import GPTQ_REQUIREMENTS, runtime_dependencies
from xhmodel_merak.model_integrations._shared import (autoregressive_precision_bindings,
                                                    llm_deployment_precision, text_media_input_schema,
                                                    text_output_schema)

PROVIDER_ID = "deepseek_v4"
_CONFIG = "configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731"
_DRAFT = ("dspark_context", "dspark_context_decode", "dspark_draft_decode", "dspark_markov")


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(AdapterSpec(name="deepseek-v4.build", kind="build",
        target="xhmodel_merak.model_integrations.deepseek_v4.build", artifact_formats=("autoregressive",),
        source_config_for_embeddings=True), provider=PROVIDER_ID)
    catalog.register_adapter(AdapterSpec(name="deepseek-v4.inference", kind="inference",
        target="xhmodel_merak.model_integrations.deepseek_v4.inference:adapter",
        artifact_formats=("autoregressive",)), provider=PROVIDER_ID)
    schema = text_media_input_schema(())
    schema["properties"]["max_new_tokens"].update(maximum=32768)
    schema["properties"]["temperature"] = {"type": "number", "minimum": 0, "default": 1.0}
    schema["properties"]["devices"] = {"type": "array", "minItems": 1,
                                          "items": {"type": "string"},
                                          "description": "GPUs holding the target and draft weights"}
    catalog.register_recipe(ModelRecipe(
        model_id="deepseek-v4-flash-0731", display_name="DeepSeek-V4 Flash 0731",
        family="deepseek-v4", export_name="deepseek_v4_flash_0731", task="text-generation",
        modalities=("text",),
        variants=(
            VariantRecipe(name="full", workflow_config=f"{_CONFIG}/deepseek_v4_flash_0731_w4a8.yaml",
                          export_modes=("low_memory",)),
            VariantRecipe(name="dspark", workflow_config=f"{_CONFIG}/deepseek_v4_flash_0731_dspark.yaml",
                          speculative_mode="dspark", extra_components=_DRAFT, export_modes=("low_memory",)),
        ),
        default_precision=ResolvedPrecision(weight="wmix", activation="amix"),
        supported_precisions=(("wmix", "amix"),), quant_methods=(),
        quant_checkpoint_methods=("gptq", "autoround"),
        deployment_precision=llm_deployment_precision(("prefill", "decode")),
        deployment_precision_bindings=autoregressive_precision_bindings(("prefill", "decode")),
        components=("prefill", "decode"), golden_components=(),
        build_adapter="deepseek-v4.build", inference_adapter="deepseek-v4.inference",
        artifact_format="autoregressive", input_schema=schema, output_schema=text_output_schema(),
        default_context_length=262144, default_prefill_chunk_length=256, max_pe_length=262144,
        attention_profiles=("standard",),
        source_constraints=(("model_type", "deepseek_v4"), ("architectures.0", "DeepseekV4ForCausalLM"),
                            ("hidden_size", 4096), ("num_hidden_layers", 43)),
        config_resolver="xhmodel_merak.model_integrations.deepseek_v4.planning:resolve_config",
        dependencies=runtime_dependencies(build=GPTQ_REQUIREMENTS, checkpoint_build=GPTQ_REQUIREMENTS),
    ), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
