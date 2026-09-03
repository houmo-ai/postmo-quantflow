from __future__ import annotations

import importlib
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError

from .contracts import AdapterKind
from .registry import get_adapter_spec


_BUILD_HOOKS = ("normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact")


def load_adapter(kind: AdapterKind, name: str) -> Any:
    """Load only implementations declared in the frozen, installed catalog."""
    try:
        spec = get_adapter_spec(kind, name)
    except LookupError as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"{kind.capitalize()} adapter is not registered: {name!r}",
            adapter=name,
        ) from error
    module_name, separator, attribute = spec.target.partition(":")
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, attribute) if separator else module
    except (ImportError, AttributeError) as error:
        raise ModelError(
            ErrorCode.EXPORT_FAILED if kind == "build" else ErrorCode.INFERENCE_FAILED,
            f"Registered {kind} adapter {name!r} is unavailable",
            adapter=name,
            target=spec.target,
            reason=str(error),
        ) from error
    required = _BUILD_HOOKS if kind == "build" else ("run",)
    missing = [hook for hook in required if not callable(getattr(implementation, hook, None))]
    if missing:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Registered {kind} adapter {name!r} is missing required hooks: {missing!r}",
            adapter=name,
            missing=missing,
        )
    return implementation
