from __future__ import annotations

import re
from dataclasses import dataclass, replace

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import (
    AUTOROUND_REQUIREMENTS,
    GPTQ_REQUIREMENTS,
    QWEN_MEDIA,
    runtime_dependencies,
)
from xhmodel_merak.model_integrations._shared import (
    PILE10K_RESOURCE,
    text_media_input_schema,
    text_output_schema,
    variant,
)


PROVIDER_ID = "qwen4-exp"


@dataclass(frozen=True)
class Qwen4ExpRecipe(ModelRecipe):
    """Qwen4's native terminal layout, without changing other model recipes."""

    checkpoint_terminal_tensors: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        methods = self.quant_checkpoint_methods
        if len(set(methods)) != len(methods) or any(
            method not in {"rtn", "autoround", "gptq", "qat"} for method in methods
        ):
            raise ValueError("quant_checkpoint_methods must contain unique supported producer names")
        # Extend only this recipe's producer vocabulary; all other validation
        # still belongs to the shared contract. The instance is not published
        # until construction has completed.
        object.__setattr__(self, "quant_checkpoint_methods", tuple(m for m in methods if m != "rtn"))
        try:
            super().__post_init__()
        finally:
            object.__setattr__(self, "quant_checkpoint_methods", methods)
        if not self.checkpoint_terminal_tensors or len({
            suffix for suffix, _ in self.checkpoint_terminal_tensors
        }) != len(self.checkpoint_terminal_tensors):
            raise ValueError("checkpoint terminal tensors must be non-empty and unique")
        for suffix, shape in self.checkpoint_terminal_tensors:
            if (
                not suffix.endswith(".weight")
                or any(not part.isidentifier() for part in suffix.split("."))
                or not shape
                or any(re.fullmatch(r"[A-Za-z_]\w*(?:\*[A-Za-z_]\w*)*", axis) is None for axis in shape)
            ):
                raise ValueError("checkpoint terminal tensors must name weights and config-field shape products")


def register(catalog: ModelCatalog) -> None:
    dependencies = runtime_dependencies(
        common=QWEN_MEDIA,
        quantization=(("rtn", GPTQ_REQUIREMENTS), ("autoround", AUTOROUND_REQUIREMENTS)),
        checkpoint_build=GPTQ_REQUIREMENTS,
    )
    # The full visual Dynamo export requires 2.11 on the tested host. SDK
    # inference uses its separately built 2.8 hsum extension; keep phase pins
    # independent rather than declaring the crashing 2.8 visual build tested.
    dependencies = replace(
        dependencies,
        common=tuple(
            item for item in dependencies.common
            if not item.startswith(("torch==", "torchvision==", "triton=="))
        ),
        build=(*dependencies.build, "torch==2.11.0", "torchvision==0.26.0", "triton==3.6.0"),
        inference=(*dependencies.inference, "torch==2.8.0", "torchvision==0.23.0", "triton==3.4.0"),
        unresolved=(*dependencies.unresolved, "Validate matching Python/PyTorch hsum wheels for both execution phases"),
    )
    catalog.register_artifact_format(
        ArtifactFormatSpec(
            name="qwen4-exp",
            validator="xhmodel_merak.model_integrations.qwen4_exp.artifact:validate_artifact",
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="qwen4-exp.build",
            kind="build",
            target="xhmodel_merak.model_integrations.qwen4_exp.build",
            artifact_formats=("qwen4-exp",),
            source_config_for_embeddings=True,
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="qwen4-exp.inference",
            kind="inference",
            target="xhmodel_merak.model_integrations.qwen4_exp.inference:adapter",
            artifact_formats=("qwen4-exp",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(
        Qwen4ExpRecipe(
            model_id="qwen3.8-flash-next",
            display_name="Qwen3.8 Flash Next",
            family=PROVIDER_ID,
            export_name="qwen3_8_flash_next",
            task="vision-language-generation",
            modalities=("text", "image"),
            variants=(
                variant(
                    "full",
                    "configs_merak/workflows/xh2a/llm_models/qwen4_exp/flash_next/qwen3_8_flash_next_unified.yaml",
                    export_modes=("low_memory", "standard"),
                ),
                variant(
                    "mtp",
                    "configs_merak/workflows/xh2a/llm_models/qwen4_exp/flash_next/qwen3_8_flash_next_unified_mtp.yaml",
                    speculative_mode="mtp",
                    export_modes=("low_memory", "standard"),
                ),
            ),
            default_precision=ResolvedPrecision(
                weight="wmix",
                activation="amix",
                components={
                    "attention": {"weight": "w8", "activation": "a8"},
                    "shared_expert": {"weight": "w8", "activation": "a8"},
                    "routed_expert": {"weight": "w4", "activation": "a8"},
                },
            ),
            supported_precisions=(("wmix", "amix"), ("w16", "amix")),
            quant_methods=("rtn", "autoround"),
            quant_checkpoint_methods=("rtn", "autoround"),
            deployment_precision={"visual": "w8a8h1_sefp", "prefill": "w8a8h1_sefp", "decode": "w8a8h1_sefp"},
            components=("visual", "prefill", "decode"),
            golden_components=("visual", "prefill", "decode", "mtp"),
            build_adapter="qwen4-exp.build",
            inference_adapter="qwen4-exp.inference",
            artifact_format="qwen4-exp",
            input_schema=text_media_input_schema(("image",)),
            output_schema=text_output_schema(),
            default_context_length=262144,
            default_prefill_chunk_length=256,
            max_pe_length=262144,
            vision_profile="visualm96-196-384-704-1536",
            vision_gears=(96, 196, 384, 704, 1536),
            attention_profiles=("standard",),
            bundled_resources=(PILE10K_RESOURCE,),
            resource_config_bindings=(("calibration", "quant.calibration.jsonl"),),
            source_constraints=(
                ("model_type", "qwen4_exp"),
                ("architectures.0", "Qwen4ExpForConditionalGeneration"),
                ("text_config.hidden_size", 2560),
                ("text_config.num_hidden_layers", 48),
            ),
            dependencies=dependencies,
            quant_capabilities=(
                ("rtn", ("gptqmodel.recipes.qwen4_exp:quantize_qwen4_exp",)),
                ("autoround", ("gptqmodel.recipes.qwen4_exp:quantize_qwen4_exp",)),
            ),
            checkpoint_auxiliary_layers_field="mtp_num_hidden_layers",
            checkpoint_config_projection="xhmodel_merak.model_integrations.qwen4_exp.build:project_checkpoint_config",
            checkpoint_terminal_tensors=(
                ("hyper_connection_mixer.hc_norm.weight", ("hc_count*hidden_size",)),
                ("hyper_connection_mixer.input_mix_weight_down.weight", ("hc_lowrank", "hc_count*hidden_size")),
                ("hyper_connection_mixer.input_mix_weight_up.weight", ("hc_count*hidden_size", "hc_lowrank")),
            ),
        ),
        provider=PROVIDER_ID,
    )


__all__ = ["PROVIDER_ID", "register"]
