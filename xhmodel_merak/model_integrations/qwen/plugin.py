from __future__ import annotations

from dataclasses import replace
from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ModelRecipe, ResolvedPrecision, VariantRecipe
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    QWEN_MEDIA,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    AUTOREGRESSIVE_BUILD,
    PILE10K_RESOURCE,
    QWEN_INFERENCE,
    autoregressive_precision_bindings,
    llm_deployment_precision,
    moe_precision,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "qwen"

_QWEN35 = "configs_merak/workflows/xh2a/llm_models/qwen3_5"
_QWEN35_MOE = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe"
_QWEN3_NEXT = "configs_merak/workflows/xh2a/llm_models/qwen3_next"
_QWEN35_AUTOROUND_CAPABILITY = "gptqmodel.recipes.qwen35_autoround:quantize_qwen35_autoround"
_QWEN3_NEXT_GPTQ_CAPABILITY = "gptqmodel.recipes.qwen3_next:quantize_qwen3_next"


def _dense_recipe(
    model_id: str,
    display_name: str,
    export_name: str,
    variants: tuple[VariantRecipe, ...],
    *,
    hidden_size: int,
    num_hidden_layers: int,
    weight: str = "w4",
    activation: str = "a8",
    deployment_quant_type: str = "w8a8h1_sefp",
) -> ModelRecipe:
    components = ("visual", "prefill", "decode")
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=QWEN_MEDIA,
            quantization=(("autoround", AUTOROUND_REQUIREMENTS),),
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id=model_id,
        display_name=display_name,
        family="qwen3.5",
        config_type_aliases=(("vision_config.model_type", "qwen3_5", "qwen3_5_vision"),),
        config_resolver="xhmodel_merak.model_integrations.qwen.planning:resolve_config",
        export_name=export_name,
        task="vision-language-generation",
        modalities=("text", "image"),
        variants=_export_modes(variants),
        default_precision=ResolvedPrecision(weight=weight, activation=activation),
        supported_precisions=((weight, activation), ("w16", activation)),
        quant_methods=("autoround",),
        quant_checkpoint_methods=("gptq", "autoround"),
        deployment_precision={component: deployment_quant_type for component in components},
        deployment_precision_bindings=autoregressive_precision_bindings(components),
        components=components,
        golden_components=components,
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=QWEN_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(("image",)),
        output_schema=text_output_schema(),
        default_profile="visualm96-196-384-704-1536",
        profiles=("visualm96-196-384-704-1536",),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        vision_profile="visualm96-196-384-704-1536",
        vision_gears=(96, 196, 384, 704, 1536),
        attention_profiles=("standard",),
        bundled_resources=(PILE10K_RESOURCE,),
        resource_config_bindings=(("calibration", "quant.calibration.dataset"),),
        config_replacements=(
            ("export.model.fuse_gdr_ops", True),
            ("export.model.fuse_gdr_block_recurrent_ops", True),
        ),
        source_constraints=(
            ("model_type", "qwen3_5"),
            ("text_config.hidden_size", hidden_size),
            ("text_config.num_hidden_layers", num_hidden_layers),
        ),
        quant_capabilities=(("autoround", (_QWEN35_AUTOROUND_CAPABILITY,)),),
    )


def _moe_recipe(
    model_id: str,
    display_name: str,
    export_name: str,
    variants: tuple[VariantRecipe, ...],
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_experts: int,
    vision: bool = True,
    model_type: str | None = "qwen3_5_moe",
    router_aux_loss_coef: float | None = None,
    quant_method: str = "autoround",
) -> ModelRecipe:
    components = ("visual", "prefill", "decode") if vision else ("prefill", "decode")
    profile = "visualm96-196-384-704-1536" if vision else None
    replacements: tuple[tuple[str, Any], ...] = (
        ("export.model.fuse_gdr_ops", True),
        ("export.model.fuse_gdr_block_recurrent_ops", True),
    )
    prefix = "text_config." if vision else ""
    constraints: list[tuple[str, Any]] = [
        (f"{prefix}hidden_size", hidden_size),
        (f"{prefix}num_hidden_layers", num_hidden_layers),
        (f"{prefix}num_experts", num_experts),
    ]
    if model_type is not None:
        constraints.insert(0, ("model_type", model_type))
    if router_aux_loss_coef is not None:
        constraints.append((f"{prefix}router_aux_loss_coef", router_aux_loss_coef))
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=QWEN_MEDIA if vision else (),
            build=("Defuser==0.0.25",) if quant_method == "gptq" else (),
            quantization=(
                (quant_method, AUTOROUND_REQUIREMENTS if quant_method == "autoround" else GPTQ_REQUIREMENTS),
            ),
            checkpoint_build=GPTQ_REQUIREMENTS,
        ),
        model_id=model_id,
        display_name=display_name,
        family="qwen3-next" if not vision else "qwen3.5-moe",
        config_type_aliases=((("vision_config.model_type", "qwen3_5_moe", "qwen3_5_moe_vision"),) if vision else ()),
        config_resolver="xhmodel_merak.model_integrations.qwen.planning:resolve_config",
        export_name=export_name,
        task="vision-language-generation" if vision else "text-generation",
        modalities=("text", "image") if vision else ("text",),
        variants=_export_modes(variants),
        default_precision=moe_precision(),
        supported_precisions=(("w4", "a8"), ("w16", "a8")),
        quant_methods=(quant_method,),
        quant_checkpoint_methods=("gptq", "autoround"),
        deployment_precision=llm_deployment_precision(components),
        deployment_precision_bindings=autoregressive_precision_bindings(components),
        components=components,
        golden_components=components,
        build_adapter=AUTOREGRESSIVE_BUILD,
        inference_adapter=QWEN_INFERENCE,
        artifact_format="autoregressive",
        input_schema=text_media_input_schema(("image",) if vision else ()),
        output_schema=text_output_schema(),
        default_profile=profile,
        profiles=(profile,) if profile else (),
        default_context_length=2048,
        default_prefill_chunk_length=256,
        max_pe_length=262144,
        vision_profile=profile,
        vision_gears=(96, 196, 384, 704, 1536) if vision else (),
        attention_profiles=("standard",),
        bundled_resources=(PILE10K_RESOURCE,) if quant_method == "autoround" else (),
        resource_config_bindings=((("calibration", "quant.calibration.jsonl"),) if quant_method == "autoround" else ()),
        config_replacements=replacements,
        source_constraints=tuple(constraints),
        quant_capabilities=(
            (
                quant_method,
                (_QWEN35_AUTOROUND_CAPABILITY,) if quant_method == "autoround" else (_QWEN3_NEXT_GPTQ_CAPABILITY,),
            ),
        ),
    )


def _macaron_recipe() -> ModelRecipe:
    recipe = _moe_recipe(
        "macaron-v1-tall",
        "Macaron V1 Tall",
        "macaron_v1_tall",
        (
            variant(
                "full",
                f"{_QWEN35_MOE}/35b_a3b/macaron_v1_tall_w4a8.yaml",
            ),
            variant("lora", f"{_QWEN35_MOE}/35b_a3b/macaron_v1_tall_lora.yaml", requires_lora=True),
        ),
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        router_aux_loss_coef=0.001,
    )
    return replace(
        recipe,
        default_precision=ResolvedPrecision(weight="w4", activation="a8"),
        supported_precisions=(("w4", "a8"),),
        deployment_precision={"visual": "w8a8h1_sefp", "prefill": "w4a8h0_ssfp", "decode": "w4a8h0_ssfp"},
        deployment_precision_bindings=tuple(
            replace(binding, choices=("w4a8h0_ssfp",))
            if binding.config_path == "export.model.quant_scheme"
            else binding
            for binding in recipe.deployment_precision_bindings
        ),
        source_constraints=(
            ("architectures.0", "Qwen3_5MoeForConditionalGeneration"),
            *recipe.source_constraints,
        ),
    )


def _export_modes(variants: tuple[VariantRecipe, ...]) -> tuple[VariantRecipe, ...]:
    # Non-speculative LoRA uses the streamed frontend binding; LoRA + MTP
    # retains its existing standard-only contract.
    return tuple(
        replace(item, export_modes=("standard",) if item.requires_lora and item.speculative_mode else ("standard", "low_memory"))
        for item in variants
    )


def _recipes() -> tuple[ModelRecipe, ...]:
    return (
        _macaron_recipe(),
        _dense_recipe(
            "qwen3.5-0.8b",
            "Qwen3.5 0.8B",
            "qwen3_5_0_8b",
            (variant("full", f"{_QWEN35}/0.8b/qwen3_5_0_8b_full.yaml"),),
            hidden_size=1024,
            num_hidden_layers=24,
            weight="w8",
            activation="a16",
            deployment_quant_type="w8a16h1_sefp",
        ),
        _dense_recipe(
            "qwen3.5-2b",
            "Qwen3.5 2B",
            "qwen3_5_2b",
            (variant("full", f"{_QWEN35}/2b/qwen3_5_2b_full.yaml"),),
            hidden_size=2048,
            num_hidden_layers=24,
            weight="w8",
            activation="a16",
            deployment_quant_type="w8a16h1_sefp",
        ),
        _dense_recipe(
            "qwen3.5-4b",
            "Qwen3.5 4B",
            "qwen3_5_4b",
            (variant("full", f"{_QWEN35}/4b/qwen3_5_4b_full.yaml"),),
            hidden_size=2560,
            num_hidden_layers=32,
        ),
        _dense_recipe(
            "qwen3.5-9b",
            "Qwen3.5 9B",
            "qwen3_5_9b",
            (
                variant("full", f"{_QWEN35}/9b/qwen3_5_9b_full.yaml"),
                variant("mtp", f"{_QWEN35}/9b/qwen3_5_9b_full_mtp.yaml", speculative_mode="mtp"),
                variant("dflash", f"{_QWEN35}/9b/qwen3_5_9b_full_dflash.yaml", speculative_mode="dflash"),
            ),
            hidden_size=4096,
            num_hidden_layers=32,
        ),
        _dense_recipe(
            "qwen3.6-27b",
            "Qwen3.6 27B",
            "qwen3_6_27b",
            (
                variant("full", f"{_QWEN35}/27b/qwen3_6_27b_full.yaml"),
                variant("mtp", f"{_QWEN35}/27b/qwen3_6_27b_full_mtp.yaml", speculative_mode="mtp"),
                variant("dflash", f"{_QWEN35}/27b/qwen3_6_27b_full_dflash.yaml", speculative_mode="dflash"),
            ),
            hidden_size=5120,
            num_hidden_layers=64,
        ),
        _dense_recipe(
            "qwen3.8-27b",
            "Qwen3.8 27B",
            "qwen3_8_27b",
            (
                variant("full", f"{_QWEN35}/27b/qwen3_8_27b_full.yaml"),
                variant("mtp", f"{_QWEN35}/27b/qwen3_8_27b_full_mtp.yaml", speculative_mode="mtp"),
                variant(
                    "dflash2",
                    f"{_QWEN35}/27b/qwen3_8_27b_full_dflash2_8k.yaml",
                    speculative_mode="dflash2",
                    context_workflows=(
                        (8192, f"{_QWEN35}/27b/qwen3_8_27b_full_dflash2_8k.yaml"),
                        (262144, f"{_QWEN35}/27b/qwen3_8_27b_full_dflash2_256k_fa.yaml"),
                    ),
                ),
            ),
            hidden_size=5120,
            num_hidden_layers=64,
        ),
        _moe_recipe(
            "qwen3.6-35b-a3b",
            "Qwen3.6 35B-A3B",
            "qwen3_6_35b_a3b",
            (
                variant("full", f"{_QWEN35_MOE}/35b_a3b/qwen3_6_35b_a3b_full.yaml"),
                variant("mtp", f"{_QWEN35_MOE}/35b_a3b/qwen3_6_35b_a3b_full_head_w5_mtp.yaml", speculative_mode="mtp"),
                variant(
                    "dflash",
                    f"{_QWEN35_MOE}/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml",
                    speculative_mode="dflash",
                ),
                variant("lora", f"{_QWEN35_MOE}/35b_a3b/qwen3_6_35b_a3b_lora.yaml", requires_lora=True),
                variant("lora_mtp", f"{_QWEN35_MOE}/35b_a3b/qwen3_6_35b_a3b_full_head_w5_auto_mixed_mtp.yaml",
                        speculative_mode="mtp", requires_lora=True),
            ),
            hidden_size=2048,
            num_hidden_layers=40,
            num_experts=256,
            router_aux_loss_coef=0.001,
        ),
        _moe_recipe(
            "ornith-1.0-35b-a3b",
            "Ornith 1.0 35B-A3B",
            "ornith_1_0_35b_a3b",
            (variant("full", f"{_QWEN35_MOE}/35b_a3b/ornith-1.0-35b_full.yaml"),),
            hidden_size=2048,
            num_hidden_layers=40,
            num_experts=256,
            model_type="qwen3_5_moe",
            router_aux_loss_coef=0.0,
        ),
        _moe_recipe(
            "qwen3.5-122b-a10b",
            "Qwen3.5 122B-A10B",
            "qwen3_5_122b_a10b",
            (
                variant("full", f"{_QWEN35_MOE}/122b_a10b/qwen3_5_122b_a10b_full.yaml"),
                variant(
                    "mtp",
                    f"{_QWEN35_MOE}/122b_a10b/qwen3_5_122b_a10b_full_mtp.yaml",
                    speculative_mode="mtp",
                ),
                variant(
                    "dflash",
                    f"{_QWEN35_MOE}/122b_a10b/qwen3_5_122b_a10b_full_dflash.yaml",
                    speculative_mode="dflash",
                ),
            ),
            hidden_size=3072,
            num_hidden_layers=48,
            num_experts=256,
            router_aux_loss_coef=0.001,
        ),
        _moe_recipe(
            "qwen3-next-80b-a3b",
            "Qwen3-Next 80B-A3B",
            "qwen3_next_80b_a3b_instruct",
            (
                variant("full", f"{_QWEN3_NEXT}/80b_a3b/qwen3_next_80b_a3b_full_gptq.yaml"),
                variant(
                    "mtp",
                    f"{_QWEN3_NEXT}/80b_a3b/qwen3_next_80b_a3b_full_mtp_gptq.yaml",
                    speculative_mode="mtp",
                ),
            ),
            hidden_size=2048,
            num_hidden_layers=48,
            num_experts=512,
            vision=False,
            model_type="qwen3_next",
            quant_method="gptq",
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_adapter(
        AdapterSpec(
            name=QWEN_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.qwen.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    for recipe in _recipes():
        if recipe.model_id.startswith(("qwen3.5-", "qwen3.6-", "qwen3.8-")):
            recipe = replace(recipe, recipe_version="3")
        if recipe.model_id == "qwen3.6-35b-a3b":
            recipe = replace(recipe, supported_precisions=(*recipe.supported_precisions, ("wmix", "a8")))
        catalog.register_recipe(recipe, provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
