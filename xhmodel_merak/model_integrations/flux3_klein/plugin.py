"""Lightweight catalog declarations for the complete distilled image pipeline."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import QWEN_MEDIA, runtime_dependencies


PROVIDER_ID = "flux3_klein"
COMPONENTS = ("image",)
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name=PROVIDER_ID), provider=PROVIDER_ID)
    for kind in ("build", "inference"):
        catalog.register_adapter(
            AdapterSpec(
                name=f"flux3_klein_{kind}",
                kind=kind,
                target=f"xhmodel_merak.model_integrations.flux3_klein.{kind}"
                + (":adapter" if kind == "inference" else ""),
                artifact_formats=(PROVIDER_ID,),
            ),
            provider=PROVIDER_ID,
        )
    catalog.register_recipe(
        ModelRecipe(
            model_id="flux-3-image-klein",
            display_name="FLUX 3 Image Klein",
            family=PROVIDER_ID,
            export_name=PROVIDER_ID,
            task="image-generation",
            modalities=("text", "image"),
            variants=(
                VariantRecipe(
                    "full",
                    "configs_merak/workflows/xh2a/other_models/flux3_klein/flux3_klein_1mp_w16.yaml",
                    optional_sources=("prompt_upsampler", "image_bundle", "upsampler_bundle"),
                ),
            ),
            default_precision=ResolvedPrecision(weight="w16", activation="a16"),
            supported_precisions=(("w16", "a16"),),
            quant_methods=(),
            deployment_precision={"image": "w16a16_sefp"},
            components=COMPONENTS,
            golden_components=(),
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    components=COMPONENTS,
                    config_path="export.model.quant_scheme",
                    choices=("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"),
                ),
            ),
            build_adapter="flux3_klein_build",
            inference_adapter="flux3_klein_inference",
            artifact_format=PROVIDER_ID,
            default_profile="1mp-c4352",
            profiles=("1mp-c4352",),
            artifact_name_template="hmquant_xh2_flux3_klein_{deployment_precision}_1mp_adaptive_c4352",
            config_resolver="xhmodel_merak.model_integrations.flux3_klein.source:resolve_config",
            dependencies=runtime_dependencies(common=QWEN_MEDIA),
            worker_capabilities=("transformers.models.qwen3_vl.modeling_qwen3_vl:Qwen3VLForConditionalGeneration",),
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Full scene caption, or a short request when prompt_upsampling is enabled.",
                    },
                    "prompt_upsampling": {"type": "boolean", "default": False},
                    "images": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "default": [],
                        "description": "Up to eight ordered references sharing a token budget; at most 1 MP each.",
                    },
                    "height": {"type": "integer", "minimum": 16, "multipleOf": 16},
                    "width": {"type": "integer", "minimum": 16, "multipleOf": 16},
                    "steps": {"type": "integer", "minimum": 1, "default": 16},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": 42},
                },
            },
            output_schema={
                "type": "object",
                "required": ["image", "width", "height"],
                "properties": {
                    "image": {"type": "array", "description": "H x W x 3 uint8 RGB"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
        ),
        provider=PROVIDER_ID,
    )
