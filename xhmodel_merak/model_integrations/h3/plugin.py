# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 and selectable LoRA recipes using the Sol-H3 implementation."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import GPTQ_REQUIREMENTS, IMAGE, runtime_dependencies


__all__ = ("PROVIDER_ID", "register")

PROVIDER_ID = "h3"
_DEFAULT_PROFILE = "all-native-768p-5s"
_WORKFLOW = "configs_merak/workflows/xh2a/other_models/h3/h3_native_768p_w4.yaml"
_QUANT = (*GPTQ_REQUIREMENTS, "torchao==0.17.0")


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name="native_h3"), provider=PROVIDER_ID)
    for kind in ("build", "inference"):
        catalog.register_adapter(
            AdapterSpec(
                name=f"h3_{kind}",
                kind=kind,
                target=f"xhmodel_merak.model_integrations.h3.{kind}" + (":adapter" if kind == "inference" else ""),
                artifact_formats=("native_h3",),
            ),
            provider=PROVIDER_ID,
        )
    catalog.register_recipe(
        ModelRecipe(
            model_id="minimax-h3",
            display_name="MiniMax-H3 native with selectable LoRA",
            family=PROVIDER_ID,
            export_name="minimax_h3",
            task="text-to-video",
            modalities=("text", "image", "video", "audio"),
            variants=(
                VariantRecipe("full", _WORKFLOW),
                VariantRecipe("lora", _WORKFLOW, requires_lora=True),
            ),
            default_precision=ResolvedPrecision(weight="w4", activation="a8"),
            supported_precisions=(("w4", "a8"), ("w16", "a8"), ("w16", "a16")),
            quant_methods=("autoround",),
            quant_checkpoint_methods=("autoround", "gptq"),
            deployment_precision={"video": "w8a8_sefp"},
            components=("video",),
            golden_components=(),
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    components=("video",), config_path="export.model.quant_scheme", choices=("w8a8_sefp", "w16a16_sefp")
                ),
            ),
            build_adapter="h3_build",
            inference_adapter="h3_inference",
            artifact_format="native_h3",
            default_profile=_DEFAULT_PROFILE,
            profiles=(_DEFAULT_PROFILE,),
            artifact_name_template="hmquant_xh2_minimax_h3_{deployment_precision}",
            config_resolver="xhmodel_merak.model_integrations.h3.source:resolve_config",
            checkpoint_config_projection="xhmodel_merak.model_integrations.sol_h3.source:normalize_checkpoint_config",
            dependencies=runtime_dependencies(
                common=(*IMAGE, "diffusers==0.41.0", "av==15.1.0", "torchaudio==2.8.0"),
                quantization=(("autoround", _QUANT),),
                checkpoint_build=_QUANT,
            ),
            worker_capabilities=("diffusers.models.transformers.transformer_minimax_h3:MiniMaxH3Transformer3DModel",),
            quant_capabilities=(
                (
                    "autoround",
                    (
                        "gptqmodel.recipes:quantize_qwen3_vl",
                        "auto_round:AutoRound",
                        "auto_round.data_type.int:quant_tensor_sym_xh",
                    ),
                ),
            ),
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "image": {"type": "string", "minLength": 1},
                    "last_image": {"type": "string", "minLength": 1},
                    "references": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {
                            "type": "object",
                            "required": ["kind", "path"],
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string", "enum": ["image", "video", "audio"]},
                                "path": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                    "lora_adapter": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Packaged adapter name; omit to use unmodified H3 weights.",
                    },
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": 42},
                    "output_path": {"type": "string", "minLength": 1, "default": "h3.mp4"},
                },
            },
            output_schema={
                "type": "object",
                "required": ["output_path", "width", "height", "frames", "fps"],
                "properties": {
                    "output_path": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "frames": {"type": "integer"},
                    "fps": {"type": "number"},
                },
            },
        ),
        provider=PROVIDER_ID,
    )
