"""Lightweight declaration for the complete three-graph action policy."""

from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
    VariantRecipe,
)
from xhmodel_merak.model_integrations._dependencies import IMAGE, runtime_dependencies


PROVIDER_ID = "wall_oss"
COMPONENTS = ("policy",)
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name="wall_oss"), provider=PROVIDER_ID)
    for kind in ("build", "inference"):
        catalog.register_adapter(
            AdapterSpec(
                name=f"wall_oss_{kind}",
                kind=kind,
                target=f"xhmodel_merak.model_integrations.wall_oss.{kind}"
                + (":adapter" if kind == "inference" else ""),
                artifact_formats=("wall_oss",),
            ),
            provider=PROVIDER_ID,
        )
    vector = {"type": "array", "items": {"type": "number"}, "minItems": 26, "maxItems": 26}
    mask = {"type": "array", "items": {"type": "integer", "enum": [0, 1]}, "minItems": 26, "maxItems": 26}
    catalog.register_recipe(
        ModelRecipe(
            model_id="wall-oss-0.5",
            display_name="Wall-OSS-0.5",
            family=PROVIDER_ID,
            export_name="wall_oss_05",
            task="action-prediction",
            modalities=("text", "image", "state", "action"),
            variants=(VariantRecipe("full", "configs_merak/workflows/xh2a/other_models/wall_oss/wall_oss_05.yaml"),),
            default_precision=ResolvedPrecision(weight="w16", activation="a16"),
            supported_precisions=(("w16", "a16"),),
            quant_methods=(),
            deployment_precision={"policy": "w16a16_sefp"},
            components=COMPONENTS,
            golden_components=(),
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    components=COMPONENTS,
                    config_path="export.model.quant_scheme",
                    choices=("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"),
                ),
            ),
            build_adapter="wall_oss_build",
            inference_adapter="wall_oss_inference",
            artifact_format="wall_oss",
            default_profile="p256-c1024-v1024-h32",
            profiles=("p256-c1024-v1024-h32",),
            artifact_name_template="hmquant_xh2_wall_oss_05_{deployment_precision}_p256_c1024_v1024_h32",
            config_resolver="xhmodel_merak.model_integrations.wall_oss.source:resolve_config",
            dependencies=runtime_dependencies(common=(*IMAGE, "qwen-vl-utils==0.0.14")),
            input_schema={
                "type": "object",
                "required": ["instruction", "images", "state"],
                "additionalProperties": False,
                "properties": {
                    "instruction": {"type": "string", "minLength": 1},
                    "images": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "state": vector,
                    "norm_key": {"type": "string", "default": "x2_normal"},
                    "dof_mask": mask,
                    "state_mask": mask,
                    "seed": {"type": "integer", "minimum": 0, "default": 0},
                    "steps": {"type": "integer", "minimum": 1, "default": 10},
                },
            },
            output_schema={
                "type": "object",
                "required": ["actions", "horizon", "action_dim"],
                "properties": {
                    "actions": {"type": "array"},
                    "horizon": {"type": "integer", "const": 32},
                    "action_dim": {"type": "integer", "const": 26},
                },
            },
        ),
        provider=PROVIDER_ID,
    )
