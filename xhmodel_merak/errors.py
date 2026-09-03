from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    UNSUPPORTED_COMBINATION = "UNSUPPORTED_COMBINATION"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    QUANT_ARTIFACT_MISMATCH = "QUANT_ARTIFACT_MISMATCH"
    EXPORT_FAILED = "EXPORT_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    GOLDEN_NOT_SUPPORTED = "GOLDEN_NOT_SUPPORTED"
    ARTIFACT_VALIDATION_FAILED = "ARTIFACT_VALIDATION_FAILED"


class ModelError(ValueError):
    def __init__(self, code: ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "details": self.details,
        }


class CatalogRegistrationError(RuntimeError):
    """Raised when the installed built-in catalog is internally inconsistent."""


def invalid_parameter(model: str, field: str, value: Any, constraint: str) -> ModelError:
    return ModelError(
        ErrorCode.INVALID_PARAMETER,
        f"Invalid {field!r} for model {model!r}: {value!r}. {constraint}",
        model=model,
        field=field,
        value=value,
        constraint=constraint,
    )


def unsupported_combination(model: str, constraint: str, **values: Any) -> ModelError:
    return ModelError(
        ErrorCode.UNSUPPORTED_COMBINATION,
        f"Unsupported parameter combination for model {model!r}: {constraint}",
        model=model,
        constraint=constraint,
        values=values,
    )


__all__ = [
    "CatalogRegistrationError",
    "ErrorCode",
    "ModelError",
    "invalid_parameter",
    "unsupported_combination",
]
