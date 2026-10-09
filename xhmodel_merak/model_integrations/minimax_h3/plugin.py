from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import IMAGE, runtime_dependencies
from xhmodel_merak.model_integrations._shared import MINIMAX_H3_BUILD, MINIMAX_H3_INFERENCE, variant
from xhmodel_merak.model_integrations.minimax_h3.artifact import COMPONENTS


PROVIDER_ID = "minimax_h3"

_WORKFLOW = "configs_merak/workflows/xh2a/other_models/minimax_h3/minimax_h3.yaml"


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "description": "Text prompt for MiniMax-H3 video/audio generation.",
            },
            "image": {
                "type": "string",
                "minLength": 1,
                "description": "Local first-frame image path required by fl2va; omit for t2va.",
            },
            "workflow": {
                "type": "string",
                "enum": ["fl2va", "t2va"],
                "default": "fl2va",
            },
            "output": {
                "type": "string",
                "minLength": 1,
                "description": "Optional local mp4 output path for the legacy demo path.",
            },
            "height": {"type": "integer", "minimum": 16, "default": 352},
            "width": {"type": "integer", "minimum": 16, "default": 608},
            "num_frames": {"type": "integer", "minimum": 1, "default": 124},
            "num_inference_steps": {"type": "integer", "minimum": 1, "default": 4},
            "seed": {"type": "integer", "minimum": 0, "default": 42},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "output_type": {"type": "string", "const": "video"},
            "output_path": {"type": "string"},
            "sample_rate": {"type": "integer", "minimum": 1},
        },
        "required": ["output_type", "output_path"],
    }


def _recipe() -> ModelRecipe:
    deployment = {component: "w8a8_ssfp" for component in COMPONENTS}
    return ModelRecipe(
        dependencies=runtime_dependencies(
            common=(*IMAGE, "diffusers==0.41.0"),
            unresolved=(
                "MiniMax-H3 unified inference still requires the host modular pipeline from model_dir",
                "Use examples_merak.aigc.minimaxh3.hmonnx_demo until a VideoInferenceResult contract exists",
            ),
        ),
        model_id="minimax-h3-fl2va",
        display_name="MiniMax-H3 FL2VA",
        family="minimax-h3",
        export_name="minimax_h3_fl2va",
        task="image-to-video",
        modalities=("text", "image", "video", "audio"),
        variants=(variant("full", _WORKFLOW),),
        default_precision=ResolvedPrecision(weight="w8", activation="a8"),
        supported_precisions=(("w8", "a8"),),
        quant_methods=(),
        deployment_precision=deployment,
        components=COMPONENTS,
        golden_components=(),
        build_adapter=MINIMAX_H3_BUILD,
        inference_adapter=MINIMAX_H3_INFERENCE,
        artifact_format="aigc",
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        default_profile="full",
        profiles=("full",),
        fixed_artifact_name="hmquant_xh2_minimax_h3_fl2va_w8a8_ssfp",
        recipe_version="1",
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name="aigc"), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(
            name=MINIMAX_H3_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.minimax_h3.build",
            artifact_formats=("aigc",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=MINIMAX_H3_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.minimax_h3.inference:adapter",
            artifact_formats=("aigc",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
