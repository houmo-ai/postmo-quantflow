from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import AUDIO, runtime_dependencies
from xhmodel_merak.model_integrations._shared import (
    MOSS_TRANSCRIBE_DIARIZE_BUILD,
    MOSS_TRANSCRIBE_DIARIZE_INFERENCE,
    variant,
)


PROVIDER_ID = "moss_transcribe_diarize"

_WORKFLOW = "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/moss_transcribe_diarize.yaml"
_WORKFLOW_16K = (
    "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/"
    "moss_transcribe_diarize_16k.yaml"
)
_WORKFLOW_64K = (
    "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/"
    "moss_transcribe_diarize_64k.yaml"
)
_WORKFLOW_128K = (
    "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/"
    "moss_transcribe_diarize_128k.yaml"
)


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "audio": {
                "type": "string",
                "minLength": 1,
                "description": "Local 16 kHz mono 16-bit PCM WAV/FLAC recording; encoded in 30-second windows and decoded jointly within the artifact KV capacity. 30-minute meetings require context_length=65536.",
            },
            "prompt": {"type": "string", "minLength": 1},
            "max_new_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "description": (
                    "Maximum generated transcript tokens. Omit to scale with audio duration "
                    "(10 tokens/s, floor 512, cap 65536). prompt plus this budget must fit "
                    "the published KV cache; overflow fails instead of chunking the meeting."
                ),
            },
        },
        "required": ["audio"],
        "additionalProperties": False,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"output_text": {"type": "string"}},
        "required": ["output_text"],
    }


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        dependencies=runtime_dependencies(common=(*AUDIO, "ninja==1.13.0")),
        model_id="moss-transcribe-diarize-0.9b",
        display_name="MOSS-Transcribe-Diarize 0.9B",
        family="moss-transcribe-diarize",
        export_name="moss_transcribe_diarize_09b",
        task="speech-transcription",
        modalities=("audio", "text"),
        variants=(
            variant(
                "full",
                _WORKFLOW,
                context_workflows=(
                    (4096, _WORKFLOW),
                    (16384, _WORKFLOW_16K),
                    (65536, _WORKFLOW_64K),
                    (131072, _WORKFLOW_128K),
                ),
            ),
        ),
        default_precision=ResolvedPrecision(weight="w8", activation="a8"),
        supported_precisions=(("w8", "a8"),),
        quant_methods=(),
        deployment_precision={
            "encode": "w8a8_sefp",
            "prefill_decode": "w8a8_sefp",
        },
        components=("encode", "prefill", "decode"),
        golden_components=("encode", "prefill", "decode"),
        build_adapter=MOSS_TRANSCRIBE_DIARIZE_BUILD,
        inference_adapter=MOSS_TRANSCRIBE_DIARIZE_INFERENCE,
        artifact_format="asr",
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        default_profile="full",
        profiles=("full",),
        default_context_length=2048,
        default_prefill_chunk_length=512,
        max_pe_length=262144,
        config_resolver="xhmodel_merak.model_integrations.moss_transcribe_diarize.resources:resolve_config",
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name="asr"), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(
            name=MOSS_TRANSCRIBE_DIARIZE_BUILD,
            kind="build",
            target="xhmodel_merak.model_integrations.moss_transcribe_diarize.build",
            artifact_formats=("asr",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name=MOSS_TRANSCRIBE_DIARIZE_INFERENCE,
            kind="inference",
            target="xhmodel_merak.model_integrations.moss_transcribe_diarize.inference:adapter",
            artifact_formats=("asr",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)


__all__ = ["PROVIDER_ID", "register"]
