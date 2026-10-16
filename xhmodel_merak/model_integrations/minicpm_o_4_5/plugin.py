from __future__ import annotations

from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision
from xhmodel_merak.model_catalog.registry import ModelCatalog
from xhmodel_merak.model_integrations._dependencies import AUDIO, GPTQ_REQUIREMENTS, IMAGE, runtime_dependencies
from xhmodel_merak.model_integrations._shared import variant


__all__ = ["PROVIDER_ID", "register"]

PROVIDER_ID = "minicpm_o_4_5"
BUILD = "minicpm_o_4_5.build"
INFERENCE = "minicpm_o_4_5.inference"
FORMAT = "minicpm_o_4_5"
COMPONENTS = (
    "vision", "audio", "llm", "tts", "speaker",
    "token2wav_flow_frontend", "token2wav_flow_decoder", "token2wav_hift",
)
_MODULE = "xhmodel_merak.model_integrations.minicpm_o_4_5"
_WORKFLOW = (
    "configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/"
    "minicpm_o_4_5_xh2a_w4a8_gptq.yaml"
)


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        model_id="minicpm-o-4.5",
        display_name="MiniCPM-o 4.5",
        family="minicpm_o_4_5",
        export_name="minicpm_o_4_5",
        fixed_artifact_name="hmquant_xh2_minicpm_o_4_5_w4a8",
        task="multimodal-generation",
        modalities=("text", "image", "audio", "video"),
        variants=tuple(variant(mode, _WORKFLOW) for mode in ("offline", "session", "stream")),
        default_profile="offline",
        profiles=("offline", "session", "stream"),
        default_precision=ResolvedPrecision(weight="w4", activation="a8"),
        supported_precisions=(("w4", "a8"),),
        quant_methods=("gptq",),
        quant_checkpoint_methods=("gptq",),
        deployment_precision={
            "vision": "w8a16_sefp", "audio": "w8a16_sefp", "llm": "w4a8h0_ssfp",
            "tts": "w8a8_sefp", "speaker": "w8a16_sefp",
            "token2wav_flow_frontend": "w8a16_sefp", "token2wav_flow_decoder": "w8a16_sefp",
            "token2wav_hift": "w8a16_sefp",
        },
        components=COMPONENTS,
        golden_components=COMPONENTS,
        build_adapter=BUILD,
        inference_adapter=INFERENCE,
        artifact_format=FORMAT,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
                **{name: {"type": "string", "minLength": 1} for name in ("image", "audio", "video")},
                "max_new_tokens": {
                    "type": "integer", "minimum": 1, "maximum": 32768, "default": 128,
                    "description": "offline/session: total tokens; stream: speaking tokens per input chunk",
                },
                "generate_audio": {"type": "boolean", "default": False},
                "reference_audio": {"type": "string", "minLength": 1},
                "language": {
                    "type": "string", "enum": ["zh", "en"],
                    "description": "Offline voice prompt language; session/stream use the official streaming prompt",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
            "maxMediaInputs": 2,
        },
        output_schema={
            "type": "object",
            "oneOf": [
                {"type": "object", "required": ["output_text", "output_tokens"]},
                {"type": "object", "required": ["waveform", "sample_rate", "stats"]},
            ],
        },
        source_constraints=(("model_type", "minicpmo"), ("version", "4.5"), ("architectures.0", "MiniCPMO")),
        config_resolver=f"{_MODULE}.build:resolve_config",
        checkpoint_config_projection=f"{_MODULE}.build:project_checkpoint_config",
        dependencies=runtime_dependencies(
            common=(*AUDIO, *IMAGE, "minicpmo-utils==1.0.6", "hyperpyyaml==1.2.3",
                    "torchaudio==2.8.0", "resampy==0.4.3"),
            inference=("moviepy==2.2.1", "ninja==1.13.0"),
            system=("FFmpeg and ffprobe for video input", "CUDA 12.8 toolkit for the W4 runtime extension"),
            checkpoint_build=GPTQ_REQUIREMENTS,
            quantization=(("gptq", GPTQ_REQUIREMENTS),),
            unresolved=(
                "Install minicpmo-utils==1.0.6 with --no-deps in a prepared shared environment; "
                "its Pillow/librosa/moviepy metadata conflicts remain unresolved. Do not install tts/all extras",
            ),
        ),
    )


def register(catalog: ModelCatalog) -> None:
    catalog.register_artifact_format(ArtifactFormatSpec(name=FORMAT), provider=PROVIDER_ID)
    catalog.register_adapter(AdapterSpec(BUILD, "build", f"{_MODULE}.build", (FORMAT,)), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(INFERENCE, "inference", f"{_MODULE}.inference:adapter", (FORMAT,)), provider=PROVIDER_ID
    )
    catalog.register_recipe(_recipe(), provider=PROVIDER_ID)
