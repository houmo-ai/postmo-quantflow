from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class InferenceResult:
    model_id: str
    task: str
    output_type: str
    artifact_dir: str
    meta_file: str
    latency_s: float
    components: tuple[str, ...]
    golden_components: tuple[str, ...] = ()
    golden_artifacts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, kw_only=True)
class TextInferenceResult(InferenceResult):
    output_text: str
    output_tokens: int
    tokens_per_second: float
    spec_decode_mode: str | None = None


@dataclass(frozen=True, kw_only=True)
class ActionInferenceResult(InferenceResult):
    actions: tuple[tuple[float, ...], ...]
    horizon: int
    action_dim: int
    seed: int


@dataclass(frozen=True, kw_only=True)
class AudioInferenceResult(InferenceResult):
    waveform: Any
    sample_rate: int
    num_samples: int
    duration_s: float
    real_time_factor: float
    voice: str
    speed: float
    segments: tuple[dict[str, Any], ...] = ()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Inference results must contain only finite numbers")
    return value


__all__ = [
    "ActionInferenceResult",
    "AudioInferenceResult",
    "InferenceResult",
    "TextInferenceResult",
]
