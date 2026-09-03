"""Thin public-API binding to Kokoro's own text frontend and release runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import AudioInferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs

from .artifact import load_metadata, validate_runtime_contract


class _KokoroAdapter:
    def validate_artifact(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: Mapping[str, Any],
        meta_file: Path,
        **_: Any,
    ) -> None:
        _load_runtime_contract(root, manifest, meta_file, recipe.components)

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
    ) -> AudioInferenceResult:
        normalized = validate_inputs(recipe, inputs)
        meta, config_path, voice_path = _load_runtime_contract(root, manifest, meta_file, components)

        from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import KokoroBucketedRuntime
        from xhmodel_merak.xh_other_model.models.kokoro.host import load_voice_style
        from xhmodel_merak.xh_other_model.models.kokoro.text import prepare_text

        tokens, phonemes = prepare_text(
            normalized["text"],
            config_path,
            max_tokens=max(route["token_max_length"] for route in meta["bucket_presets"]["routes"]),
        )
        style = load_voice_style(voice_path, phoneme_count=tokens.size - 2).numpy()
        runtime = KokoroBucketedRuntime(root, meta, backend="hmonnx", device=device)
        started = time.perf_counter()
        try:
            waveform, synthesis = runtime.synthesize(tokens, style, speed=float(normalized["speed"]))
            latency_s = time.perf_counter() - started
        finally:
            runtime.clear_runner_cache()
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("Kokoro runtime produced an empty or non-finite waveform")
        sample_rate = int(meta["sample_rate"])
        duration_s = float(waveform.size / sample_rate)
        return AudioInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="audio",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            waveform=waveform,
            sample_rate=sample_rate,
            num_samples=int(waveform.size),
            duration_s=duration_s,
            real_time_factor=latency_s / duration_s,
            voice=str(normalized["voice"]),
            speed=float(normalized["speed"]),
            segments=({"index": 0, "phonemes": phonemes, **synthesis},),
            stats={"phonemes": phonemes, "segment_count": 1},
        )


def _load_runtime_contract(
    root: Path,
    manifest: Mapping[str, Any],
    meta_file: Path,
    components: tuple[str, ...],
) -> tuple[dict[str, Any], Path, Path]:
    meta = load_metadata(meta_file)
    config_path, voice_path = validate_runtime_contract(
        meta,
        components=components,
        resolve_file=lambda value, field: artifact_file(root, manifest, value, field),
    )
    return meta, config_path, voice_path


adapter = _KokoroAdapter()


__all__ = ["adapter"]
