from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import DeploymentPrecisionBinding, ResolvedPrecision, VariantRecipe
from xhmodel_merak.model_catalog.dependencies import DependencySpec


AUTOREGRESSIVE_BUILD = "autoregressive.build"
PI05_BUILD = "pi05.build"
KOKORO_BUILD = "kokoro.build"
MINIMAX_H3_BUILD = "minimax_h3.build"
QWEN_INFERENCE = "qwen.inference"
GEMMA4_INFERENCE = "gemma4.inference"
LING_INFERENCE = "ling.inference"
PI05_INFERENCE = "pi05.inference"
KOKORO_INFERENCE = "kokoro.inference"
MINIMAX_H3_INFERENCE = "minimax_h3.inference"
MOSS_TRANSCRIBE_DIARIZE_BUILD = "moss_transcribe_diarize.build"
MOSS_TRANSCRIBE_DIARIZE_INFERENCE = "moss_transcribe_diarize.inference"
MINICPM_V_4_7_BUILD = "minicpm_v_4_7.build"
MINICPM_V_4_7_INFERENCE = "minicpm_v_4_7.inference"

PILE10K_RESOURCE = ("calibration", "data/calib_data/NeelNanda-pile-10k.jsonl")


def text_media_input_schema(media: tuple[str, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "minLength": 1},
        "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 64},
    }
    for modality in media:
        properties[modality] = {"type": "string", "description": f"Local {modality} path"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["prompt"],
        "additionalProperties": False,
        "maxMediaInputs": 1,
    }


def text_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "output_text": {"type": "string"},
            "output_tokens": {"type": "integer", "minimum": 0},
        },
        "required": ["output_text", "output_tokens"],
    }


def variant(
    name: str,
    config: str,
    *,
    speculative_mode: str | None = None,
    requires_lora: bool = False,
    context_workflows: tuple[tuple[int, str], ...] = (),
    export_modes: tuple[str, ...] = ("standard",),
    dependencies: DependencySpec | None = None,
) -> VariantRecipe:
    required_sources = (speculative_mode,) if speculative_mode in {"dflash", "dflash2"} else ()
    extra_components = ((speculative_mode,) if speculative_mode else ()) + (("lora",) if requires_lora else ())
    return VariantRecipe(
        name=name,
        workflow_config=config,
        speculative_mode=speculative_mode,
        requires_lora=requires_lora,
        required_sources=required_sources,
        extra_components=extra_components,
        context_workflows=context_workflows,
        export_modes=export_modes,
        dependencies=dependencies if dependencies is not None else DependencySpec(),
    )


def llm_deployment_precision(components: tuple[str, ...]) -> dict[str, str]:
    return {component: "w8a8h1_sefp" for component in components}


def autoregressive_precision_bindings(components: tuple[str, ...]) -> tuple[DeploymentPrecisionBinding, ...]:
    choices = ("w8a8h1_sefp", "w8a16h1_sefp")
    language = tuple(component for component in components if component in {"prefill", "decode"})
    bindings = [DeploymentPrecisionBinding(language, "export.model.quant_scheme", choices)] if language else []
    for component in components:
        if component in {"visual", "video_visual", "audio"}:
            bindings.append(
                DeploymentPrecisionBinding(
                    (component,),
                    f"export.model.{component}_config.quant_scheme",
                    choices,
                    inherit_from="export.model.visual_config.quant_scheme"
                    if component == "video_visual"
                    else "export.model.quant_scheme",
                )
            )
    return tuple(bindings)


def moe_precision() -> ResolvedPrecision:
    return ResolvedPrecision(
        weight="w4",
        activation="a8",
        components={
            "attention": {"weight": "w8", "activation": "a8"},
            "shared_expert": {"weight": "w8", "activation": "a8"},
            "routed_expert": {"weight": "w4", "activation": "a8"},
        },
    )


__all__ = [
    "AUTOREGRESSIVE_BUILD",
    "GEMMA4_INFERENCE",
    "KOKORO_BUILD",
    "KOKORO_INFERENCE",
    "LING_INFERENCE",
    "MINIMAX_H3_BUILD",
    "MINIMAX_H3_INFERENCE",
    "MINICPM_V_4_7_BUILD",
    "MINICPM_V_4_7_INFERENCE",
    "MOSS_TRANSCRIBE_DIARIZE_BUILD",
    "MOSS_TRANSCRIBE_DIARIZE_INFERENCE",
    "PI05_BUILD",
    "PI05_INFERENCE",
    "PILE10K_RESOURCE",
    "QWEN_INFERENCE",
    "llm_deployment_precision",
    "autoregressive_precision_bindings",
    "moe_precision",
    "text_media_input_schema",
    "text_output_schema",
    "variant",
]
