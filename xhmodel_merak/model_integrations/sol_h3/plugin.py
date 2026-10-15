# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""The complete independently published two-stage AV pipeline."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import (
    GPTQ_REQUIREMENTS,
    IMAGE,
    runtime_dependencies,
)

from .recipe import H3Recipe


PROVIDER_ID = "sol_h3"
COMPONENTS = ("video",)
_WORKFLOWS = "configs_merak/workflows/xh2a/other_models/sol_h3/"
# TorchAO 0.18 imports ScalingType, which the pinned Torch 2.8 does not provide.
_QUANT_REQUIREMENTS = (*GPTQ_REQUIREMENTS, "torchao==0.17.0")
_PROFILE_WORKFLOWS = (
    ("all-dense-384p-768p-5s", _WORKFLOWS + "sol_h3_all_384p_768p_w4.yaml"),
    ("dense-384p-768p-5s", _WORKFLOWS + "sol_h3_dense_384p_768p_w4.yaml"),
) + tuple(
    (f"{mode}-dense-384p-768p-5s", _WORKFLOWS + f"sol_h3_{mode}_384p_768p_w4.yaml")
    for mode in ("i2va", "l2va", "fl2va", "ref2va")
)
__all__ = ("PROVIDER_ID", "COMPONENTS", "register")


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name="sol_h3"), provider=PROVIDER_ID)
    for kind in ("build", "inference"):
        target = f"xhmodel_merak.model_integrations.sol_h3.{kind}"
        catalog.register_adapter(
            AdapterSpec(
                name=f"sol_h3_{kind}",
                kind=kind,
                target=target + (":adapter" if kind == "inference" else ""),
                artifact_formats=("sol_h3",),
            ),
            provider=PROVIDER_ID,
        )
    catalog.register_recipe(
        H3Recipe(
            model_id="sol-h3",
            display_name="Sol-H3 (H3 4-step + LTX-2.5 3-step)",
            family=PROVIDER_ID,
            export_name="sol_h3",
            task="text-to-video",
            modalities=("text", "image", "video", "audio"),
            variants=(VariantRecipe("full", _PROFILE_WORKFLOWS[0][1]),),
            workflows_by_profile=_PROFILE_WORKFLOWS,
            default_precision=ResolvedPrecision(weight="w4", activation="a8"),
            supported_precisions=(("w4", "a8"), ("w16", "a8"), ("w16", "a16")),
            quant_methods=("autoround",),
            quant_checkpoint_methods=("autoround", "gptq"),
            deployment_precision={"video": "w8a8_sefp"},
            components=COMPONENTS,
            golden_components=(),
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    components=COMPONENTS, config_path="export.model.quant_scheme", choices=("w8a8_sefp", "w16a16_sefp")
                ),
            ),
            build_adapter="sol_h3_build",
            inference_adapter="sol_h3_inference",
            artifact_format="sol_h3",
            default_profile="all-dense-384p-768p-5s",
            profiles=tuple(profile for profile, _ in _PROFILE_WORKFLOWS),
            artifact_name_template="hmquant_xh2_sol_h3_{deployment_precision}_384p_768p_5s",
            config_resolver="xhmodel_merak.model_integrations.sol_h3.source:resolve_config",
            checkpoint_config_projection="xhmodel_merak.model_integrations.sol_h3.source:normalize_checkpoint_config",
            dependencies=runtime_dependencies(
                common=(*IMAGE, "diffusers==0.41.0", "av==15.1.0", "torchaudio==2.8.0"),
                build=("openimageio==3.1.18.1", "cloudpickle==3.1.2", "scipy==1.17.1"),
                quantization=(("autoround", _QUANT_REQUIREMENTS),),
                checkpoint_build=_QUANT_REQUIREMENTS,
                unresolved=(
                    "Pinned external Sana/LTX/upscaler sources and the native fixed-prompt cache "
                    "are local build inputs",
                ),
            ),
            worker_capabilities=("diffusers.models.transformers.transformer_minimax_h3:MiniMaxH3Transformer3DModel",),
            quant_capabilities=(("autoround", (
                "gptqmodel.recipes:quantize_qwen3_vl",
                "auto_round:AutoRound",
                "auto_round.data_type.int:quant_tensor_sym_xh",
            )),),
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
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": 42},
                    "output_path": {"type": "string", "minLength": 1, "default": "sol_h3.mp4"},
                },
            },
            output_schema={
                "type": "object",
                "required": ["output_path", "frames", "width", "height", "fps"],
                "properties": {
                    "output_path": {"type": "string"},
                    "frames": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "fps": {"type": "number"},
                },
            },
        ),
        provider=PROVIDER_ID,
    )
