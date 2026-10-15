"""Checkpoint-owned sampling contracts, usable without loading worker packages."""

import json
import math
from numbers import Real
from pathlib import Path


BASE_MODEL_ID = "qwen-image-2.1"
TURBO_MODEL_ID = "qwen-image-2.1-turbo"
MODEL_IDS = (BASE_MODEL_ID, TURBO_MODEL_ID)


def validate_sigmas(values, *, field="sigmas"):
    """Validate a decreasing grid excluding the scheduler's terminal zero."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} must be a nonempty sampling sequence")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1
        for value in values
    ):
        raise ValueError(f"{field} must contain finite numbers in (0, 1]")
    if any(left <= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{field} must be strictly decreasing")
    return tuple(float(value) for value in values)


def validate_sampling_config(pipeline_config, scheduler_config, *, model_id=BASE_MODEL_ID):
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported Qwen-Image model_id: {model_id}")
    if not isinstance(pipeline_config, dict) or not isinstance(scheduler_config, dict):
        raise ValueError("pipeline and scheduler configs must be JSON objects")
    if pipeline_config and pipeline_config.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("model_index.json requires QwenImage21Pipeline")
    values = pipeline_config.get("sample_sigmas")
    sigmas = None if values is None else validate_sigmas(values, field="sample_sigmas")
    if model_id == TURBO_MODEL_ID:
        if sigmas is None or len(sigmas) != 8:
            raise ValueError("Qwen-Image-2.1-Turbo requires checkpoint sample_sigmas with eight steps")
        if (
            scheduler_config.get("use_dynamic_shifting") is not False
            or scheduler_config.get("shift_terminal") is not None
            or scheduler_config.get("shift", 1.0) != 1.0
        ):
            raise ValueError("Turbo scheduler requires use_dynamic_shifting=false, shift_terminal=null and shift=1")
    elif sigmas is not None:
        raise ValueError("checkpoint sample_sigmas require the qwen-image-2.1-turbo model identity")
    return sigmas


def read_source_sampling(model_dir, *, model_id=BASE_MODEL_ID):
    root = Path(model_dir)
    pipeline_path = root / "model_index.json"
    # Earlier base fixtures and deployment sources omitted the pipeline config.
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.is_file() else {}
    scheduler = json.loads((root / "scheduler/scheduler_config.json").read_text())
    validate_sampling_config(pipeline, scheduler, model_id=model_id)
    return pipeline
