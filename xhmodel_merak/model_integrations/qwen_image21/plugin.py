"""Declare the complete image generation/editing pipeline."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import IMAGE, runtime_dependencies


PROVIDER_ID = "qwen_image21"
COMPONENTS = ("image",)
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name="qwen_image21"), provider=PROVIDER_ID)
    for kind in ("build", "inference"):
        target = f"xhmodel_merak.model_integrations.qwen_image21.{kind}"
        catalog.register_adapter(
            AdapterSpec(
                name=f"qwen_image21_{kind}",
                kind=kind,
                target=target + (":adapter" if kind == "inference" else ""),
                artifact_formats=("qwen_image21",),
            ),
            provider=PROVIDER_ID,
        )
    catalog.register_recipe(
        ModelRecipe(
            model_id="qwen-image-2.1",
            display_name="Qwen-Image-2.1",
            family=PROVIDER_ID,
            export_name="qwen_image21",
            task="image-generation",
            modalities=("text", "image"),
            variants=(
                VariantRecipe(
                    "full", "configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_1mp_w16.yaml"
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
            build_adapter="qwen_image21_build",
            inference_adapter="qwen_image21_inference",
            artifact_format="qwen_image21",
            default_profile="1mp-c4352",
            artifact_name_template="hmquant_xh2_qwen_image21_{deployment_precision}_1mp_c4352",
            profiles=("1mp-c4352",),
            config_resolver="xhmodel_merak.model_integrations.qwen_image21.source:resolve_config",
            dependencies=runtime_dependencies(
                common=(
                    *IMAGE,
                    "huggingface-hub==1.33.0",
                    "diffusers==0.41.0",
                )
            ),
            worker_capabilities=(
                "diffusers.models.transformers.transformer_qwenimage21:QwenImage21Rope",
                "diffusers.models.transformers.transformer_qwenimage21:QwenImage21Transformer2DModel",
                "diffusers.models.autoencoders.autoencoder_kl_qwenimage21:AutoencoderKLQwenImage21",
            ),
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "images": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "default": [],
                        "description": "Local RGBA/RGB references fitted to the published capacities.",
                    },
                    "height": {"type": "integer", "minimum": 32, "multipleOf": 32},
                    "width": {"type": "integer", "minimum": 32, "multipleOf": 32},
                    "steps": {"type": "integer", "minimum": 1, "default": 20},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": 42},
                    "negative_prompt": {"type": "string"},
                    "guidance_scale": {"type": "number", "minimum": 1, "default": 1.0},
                },
            },
            output_schema={
                "type": "object",
                "required": ["image", "width", "height"],
                "properties": {
                    "image": {"type": "array", "description": "H x W x 4 uint8 RGBA"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
        ),
        provider=PROVIDER_ID,
    )
