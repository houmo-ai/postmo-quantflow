"""Declare the real Prefill-only reranking contract without loading weights."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    QWEN_MEDIA,
    runtime_dependencies,
)


PROVIDER_ID = "qwen3-vl-reranker"
_PREFIX = "xhmodel_merak.model_integrations.qwen3_vl_reranker"


def register(catalog):
    catalog.register_artifact_format(
        ArtifactFormatSpec(name=PROVIDER_ID, validator=f"{_PREFIX}.artifact:validate_artifact"), provider=PROVIDER_ID
    )
    for kind in ("build", "inference"):
        catalog.register_adapter(
            AdapterSpec(
                name=f"{PROVIDER_ID}.{kind}",
                kind=kind,
                target=f"{_PREFIX}.{kind}" + (":adapter" if kind == "inference" else ""),
                artifact_formats=(PROVIDER_ID,),
                source_config_for_embeddings=kind == "build",
            ),
            provider=PROVIDER_ID,
        )
    item = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "image": {"description": "Local image path or list of paths"},
        },
        "additionalProperties": False,
    }
    catalog.register_recipe(
        ModelRecipe(
            model_id="qwen3-vl-reranker-2b",
            display_name="Qwen3-VL-Reranker-2B",
            family=PROVIDER_ID,
            export_name="qwen3_vl_reranker_2b",
            artifact_name_template="hmquant_xh2_qwen3_vl_reranker_2b_{deployment_precision}_dynamic",
            task="multimodal-reranking",
            modalities=("text", "image"),
            variants=(
                VariantRecipe(
                    "full",
                    "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml",
                ),
            ),
            default_precision=ResolvedPrecision(weight="w16", activation="a8"),
            supported_precisions=(("w16", "a8"), ("w4", "a8"), ("w4", "a16")),
            quant_methods=("autoround",),
            quant_checkpoint_methods=("autoround",),
            quant_capabilities=(("autoround", ("gptqmodel.recipes.qwen35_autoround:quantize_qwen35_autoround",)),),
            config_resolver=f"{_PREFIX}.config:resolve_config",
            deployment_precision={"visual": "w8a8h1_sefp", "prefill": "w8a8h1_sefp"},
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    ("prefill",),
                    "export.model.quant_scheme",
                    ("w8a8h1_sefp", "w4a8h0_ssfp", "w8a16h1_sefp", "w4a16h0_ssfp"),
                ),
                DeploymentPrecisionBinding(("visual",), "export.model.visual_config.quant_scheme", ("w8a8h1_sefp",)),
            ),
            components=("visual", "prefill"),
            golden_components=(),
            build_adapter=f"{PROVIDER_ID}.build",
            inference_adapter=f"{PROVIDER_ID}.inference",
            artifact_format=PROVIDER_ID,
            input_schema={
                "type": "object",
                "properties": {
                    "query": item,
                    "documents": {"type": "array", "items": item, "minItems": 1},
                    "instruction": {"type": "string", "minLength": 1},
                },
                "required": ["query", "documents"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scores": {"type": "array", "items": {"type": "number"}},
                    "ranking": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["scores", "ranking"],
            },
            default_context_length=32768,
            default_prefill_chunk_length=256,
            max_pe_length=262144,
            vision_profile="visualm96-196-384-704-1536",
            vision_gears=(96, 196, 384, 704, 1536),
            dependencies=runtime_dependencies(
                common=QWEN_MEDIA,
                quantization=(("autoround", AUTOROUND_REQUIREMENTS),),
                checkpoint_build=GPTQ_REQUIREMENTS,
            ),
            source_constraints=(
                ("model_type", "qwen3_vl"),
                ("architectures.0", "Qwen3VLForConditionalGeneration"),
                ("text_config.hidden_size", 2048),
                ("text_config.num_hidden_layers", 28),
                ("vision_config.out_hidden_size", 2048),
            ),
            config_type_aliases=(("vision_config.model_type", "qwen3_vl", "qwen3_vl_vision"),),
        ),
        provider=PROVIDER_ID,
    )


__all__ = ["PROVIDER_ID", "register"]
