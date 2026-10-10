"""Discoverable Spark X2.5 recipes; registration does not import model code."""

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_integrations._dependencies import (
    GPTQ_REQUIREMENTS,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "spark2-5"
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_adapter(
        AdapterSpec(
            name="spark2-5.inference",
            kind="inference",
            target="xhmodel_merak.model_integrations.spark2_5.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    for size, folder, hidden, layers, bits in (("1.7b", "1_7b", 2048, 28, "w8"), ("4b", "4b", 2560, 36, "w4")):
        catalog.register_recipe(
            ModelRecipe(
                model_id=f"spark-x2.5-{size}",
                display_name=f"Spark X2.5 {size}",
                family=PROVIDER_ID,
                export_name=f"xh_token_spark_{folder}",
                task="text-generation",
                modalities=("text",),
                variants=(
                    variant(
                        "full",
                        f"configs_merak/workflows/xh2a/llm_models/xh_token_spark/{folder}/xh_token_spark_{folder}_xh2a_{bits}a8.yaml",
                    ),
                ),
                default_precision=ResolvedPrecision(weight=bits, activation="a8"),
                supported_precisions=((bits, "a8"),),
                quant_methods=("autoround",) if bits == "w4" else (),
                quant_checkpoint_methods=("autoround",),
                deployment_precision={"prefill": "w8a8h1_sefp", "decode": "w8a8h1_sefp"},
                components=("prefill", "decode"),
                golden_components=(),
                build_adapter=AUTOREGRESSIVE_BUILD,
                inference_adapter="spark2-5.inference",
                artifact_format="autoregressive",
                input_schema=text_media_input_schema(()),
                output_schema=text_output_schema(),
                default_context_length=2048,
                default_prefill_chunk_length=256,
                max_pe_length=1048576,
                attention_profiles=("standard",),
                checkpoint_embedding_suffixes=("embedding.weight",),
                source_constraints=(
                    ("model_type", "spark2_5"),
                    ("architectures.0", "Spark2_5ForCausalLM"),
                    ("hidden_size", hidden),
                    ("num_hidden_layers", layers),
                ),
                dependencies=runtime_dependencies(
                    checkpoint_build=GPTQ_REQUIREMENTS,
                    quantization=(("autoround", GPTQ_REQUIREMENTS),) if bits == "w4" else (),
                ),
            ),
            provider=PROVIDER_ID,
        )
