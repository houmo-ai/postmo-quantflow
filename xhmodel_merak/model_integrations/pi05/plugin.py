from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._shared import PI05_BUILD, PI05_INFERENCE, variant


PROVIDER_ID = "pi05"

_PI05 = "configs_merak/workflows/xh2a/other_models/pi05"


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "observation.images.image": {"type": "string"},
            "observation.images.image2": {"type": "string"},
            "observation.state": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 8,
                "maxItems": 8,
            },
            "seed": {"type": "integer", "minimum": 0, "default": 0},
        },
        "required": [
            "task",
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
        ],
        "additionalProperties": False,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "actions": {"type": "array"},
            "horizon": {"type": "integer", "const": 50},
            "action_dim": {"type": "integer", "const": 7},
        },
        "required": ["actions", "horizon", "action_dim"],
    }


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        model_id="pi0.5-libero",
        display_name="PI0.5 LIBERO",
        family="pi0.5",
        export_name="pi0_5_libero",
        task="action-prediction",
        modalities=("text", "image", "state", "action"),
        variants=(variant("lerobot-h50", f"{_PI05}/libero/pi05_libero_lerobot_h50.yaml"),),
        default_precision=ResolvedPrecision(weight="w16", activation="a16"),
        supported_precisions=(("w16", "a16"),),
        quant_methods=(),
        deployment_precision={
            "vision": "w8a8h1_sefp",
            "gemma": "w8a8h1_sefp",
            "expert": "w8a8h1_sefp",
            "other": "w8a8h1_sefp",
        },
        components=("vision", "gemma", "expert", "other"),
        golden_components=(),
        build_adapter=PI05_BUILD,
        inference_adapter=PI05_INFERENCE,
        artifact_format="vla",
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        default_profile="lerobot-h50",
        profiles=("lerobot-h50",),
        required_resources=("tokenizer",),
        resource_config_bindings=(("tokenizer", "export.config_dir"),),
        source_constraints=(
            ("type", "pi05"),
            ("chunk_size", 50),
            ("output_features.action.shape.0", 7),
        ),
        worker_requirements=("xhquant", "transformers>=5.3.0,<5.4.0", "lerobot==0.5.1"),
        fixed_artifact_name="hmquant_xh2_pi0_5_libero_lerobot_h50_w8a8h1_sefp",
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name="vla"), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(
            name=PI05_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.pi05.build",
            artifact_formats=("vla",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=PI05_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.pi05.inference:adapter",
            artifact_formats=("vla",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
