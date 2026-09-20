from __future__ import annotations

from collections.abc import Mapping
import json
import time
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs
from xhmodel_merak.model_inference.inputs import collect_golden_dirs, snapshot_golden_dirs
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing import load_audio, SAMPLING_RATE
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import (
    transcribe_artifact,
    validate_transcript_text,
)
from .artifact import load_metadata, validate_runtime_contract


class _MossTranscribeDiarizeAdapter:
    def validate_artifact(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: Mapping[str, Any],
        meta_file: Path,
        **_: Any,
    ) -> None:
        del manifest
        try:
            meta = load_metadata(meta_file)
            validate_runtime_contract(
                meta,
                resolve_file=lambda value, field: confined_file(root, value, field),
            )
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"MOSS decoder contract invalid; re-export required: {error}",
                model=recipe.model_id,
            ) from error

    def run(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: dict[str, Any],
        meta_file: Path,
        inputs: dict[str, Any],
        device: str,
        components: tuple[str, ...],
        selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> Any:
        del manifest, components
        normalized = validate_inputs(recipe, inputs)
        audio = require_local_file(recipe, "audio", normalized["audio"])
        waveform = load_audio(audio)
        started = time.perf_counter()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        golden_before = snapshot_golden_dirs(root, selected_golden) if selected_golden else {}
        text = transcribe_artifact(
            root=root,
            meta=meta,
            audio=waveform,
            device=device,
            prompt=normalized.get("prompt"),
            max_new_tokens=normalized.get("max_new_tokens"),
            selected_golden=selected_golden,
        )
        text = validate_transcript_text(
            text,
            recipe.model_id,
            max_duration=len(waveform) / SAMPLING_RATE,
            max_duration_tolerance=0.1,
        )
        produced_golden = (
            collect_golden_dirs(root, selected_golden, before=golden_before)
            if selected_golden
            else golden_artifacts
        )
        latency_s = time.perf_counter() - started
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=recipe.components,
            golden_components=selected_golden,
            golden_artifacts=produced_golden,
            output_text=text,
            output_tokens=0,
            tokens_per_second=0.0,
        )


adapter = _MossTranscribeDiarizeAdapter()
