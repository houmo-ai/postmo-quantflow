"""Gemma4 uses a decode-only assistant reading target-owned KV."""

from xhmodel_merak.model_integrations.autoregressive.build import (
    normalize_export,
    package_runtime,
    sanitize_artifact,
    validate_artifact as validate_autoregressive_artifact,
)
from .artifact import validate_mtp


def validate_artifact(stage_path, plan):
    return validate_autoregressive_artifact(stage_path, plan, speculative_validator=validate_mtp)


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
