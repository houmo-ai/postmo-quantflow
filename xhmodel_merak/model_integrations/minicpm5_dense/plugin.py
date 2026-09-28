from __future__ import annotations

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
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


PROVIDER_ID = "minicpm5-dense"
INFERENCE_ADAPTER = "minicpm5-dense.inference"

_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/minicpm5/2_6b/"
    "minicpm5_2_6b_xh2a_w8a8.yaml"
)


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        dependencies=runtime_dependencies(
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id="minicpm5-2.6b",
        display_name="MiniCPM5 2.6B",
        family="minicpm5-dense",
        export_name="minicpm5_2_6b",
        task="text-generation",
        modalities=("text",),
        variants=(variant("full", _CONFIG),),
        default_precision=ResolvedPrecision(weight="w8", activation="a8"),
        supported_precisions=(("w8", "a8"),),
        quant_methods=("gptq",),
        quant_checkpoint_methods=("gptq",),
        deployment_precision={
            "prefill": "w8a8_sefp",
            "decode": "w8a8_sefp",
        },
        components=("prefill", "decode"),
        golden_components=(),
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=INFERENCE_ADAPTER,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(()),
        output_schema=text_output_schema(),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=131072,
        attention_profiles=("standard",),
        source_constraints=(
            ("architectures.0", "LlamaForCausalLM"),
            ("hidden_size", 2048),
            ("num_hidden_layers", 42),
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=INFERENCE_ADAPTER,
            kind="inference",
            target="xhmodel_merak.model_integrations.minicpm5_dense.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
