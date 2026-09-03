from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._shared import KOKORO_BUILD, KOKORO_INFERENCE, variant


PROVIDER_ID = "kokoro"

_KOKORO = "configs_merak/workflows/xh2a/other_models/kokoro"


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "中文、数字及中文标点；含首尾标记最多 256 token，超长由调用方切句。",
            },
            "voice": {"type": "string", "enum": ["zf_001"], "default": "zf_001"},
            "speed": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
        },
        "required": ["text"],
        "additionalProperties": False,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "waveform": {"type": "array", "items": {"type": "number"}},
            "sample_rate": {"type": "integer", "const": 24000},
        },
        "required": ["waveform", "sample_rate"],
    }


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        model_id="kokoro-82m-v1.1-zh",
        display_name="Kokoro 82M v1.1 Chinese",
        family="kokoro",
        export_name="kokoro_82m_v1_1_zh",
        task="text-to-speech",
        recipe_version="3",
        modalities=("text", "audio"),
        variants=(variant("full", f"{_KOKORO}/kokoro_xh2a_bucketed_w16.yaml"),),
        default_precision=ResolvedPrecision(weight="w16", activation="a16"),
        supported_precisions=(("w16", "a16"),),
        quant_methods=(),
        deployment_precision={
            "text_duration": "w16a16_sefp",
            "frame_acoustic": "w16a16_sefp",
            "phase_core": "fp32_ort",
            "generator_istft": "w16a16_sefp",
        },
        components=("text_duration", "frame_acoustic", "phase_core", "generator_istft"),
        golden_components=(),
        build_adapter=KOKORO_BUILD,
        inference_adapter=KOKORO_INFERENCE,
        artifact_format="tts",
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        default_profile="bucketed-t32f160-t64f320-t128f640-t256f1280",
        profiles=("bucketed-t32f160-t64f320-t128f640-t256f1280",),
        worker_requirements=(
            "xhquant",
            "misaki==0.9.4",
            "cn2an",
            "jieba",
            "ordered-set",
            "pypinyin",
            "pypinyin-dict",
        ),
        fixed_artifact_name=("hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t32-64-128-256_f160-320-640-1280"),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name="tts"), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(
            name=KOKORO_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.kokoro.build",
            artifact_formats=("tts",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=KOKORO_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.kokoro.inference:adapter",
            artifact_formats=("tts",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
