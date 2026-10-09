from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from xhmodel_merak._environment import scoped_environment
from xhmodel_merak.errors import ErrorCode, ModelError, invalid_parameter
from xhmodel_merak.model_artifact.files import artifact_file, matches_component_directory
from xhmodel_merak.model_catalog.contracts import ModelRecipe


def validate_inputs(recipe: ModelRecipe, inputs: Mapping[str, Any]) -> dict[str, Any]:
    schema = recipe.input_schema
    properties = schema.get("properties", {})
    allowed = set(properties)
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise invalid_parameter(recipe.model_id, "inputs", unknown, f"allowed fields are {sorted(allowed)}")
    missing = sorted(set(schema.get("required", ())) - set(inputs))
    if missing:
        raise invalid_parameter(recipe.model_id, "inputs", missing, "required input fields are missing")

    normalized = dict(inputs)
    for name, field_schema in properties.items():
        if name not in normalized and "default" in field_schema:
            normalized[name] = field_schema["default"]
        if name not in normalized:
            continue
        _validate_value(recipe, name, normalized[name], field_schema)
    media = set(recipe.modalities) - {"text"}
    if schema.get("maxMediaInputs") is not None:
        active_media = [name for name in media if normalized.get(name) is not None]
        if len(active_media) > int(schema["maxMediaInputs"]):
            raise invalid_parameter(
                recipe.model_id,
                "inputs",
                active_media,
                f"allows at most {schema['maxMediaInputs']} media input",
            )
    return normalized


def _validate_value(recipe: ModelRecipe, name: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected = schema.get("type")
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "array": isinstance(value, (list, tuple)),
    }.get(expected, True)
    if not valid:
        raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"must have type {expected}")
    if isinstance(value, str) and schema.get("minLength") is not None and len(value) < int(schema["minLength"]):
        raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, "must not be empty")
    if isinstance(value, (list, tuple)):
        if schema.get("minItems") is not None and len(value) < int(schema["minItems"]):
            raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"requires {schema['minItems']} items")
        if schema.get("maxItems") is not None and len(value) > int(schema["maxItems"]):
            raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"allows {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_value(recipe, f"{name}[{index}]", item, item_schema)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, "must be finite")
        if schema.get("minimum") is not None and value < schema["minimum"]:
            raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"must be >= {schema['minimum']}")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"must be <= {schema['maximum']}")
        if schema.get("exclusiveMinimum") is not None and value <= schema["exclusiveMinimum"]:
            raise invalid_parameter(
                recipe.model_id,
                f"inputs.{name}",
                value,
                f"must be > {schema['exclusiveMinimum']}",
            )
        if schema.get("multipleOf") is not None:
            divisor = Fraction(str(schema["multipleOf"]))
            if divisor <= 0 or (Fraction(str(value)) / divisor).denominator != 1:
                raise invalid_parameter(
                    recipe.model_id, f"inputs.{name}", value, f"must be a multiple of {schema['multipleOf']}"
                )
    if "enum" in schema and value not in schema["enum"]:
        raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"must be one of {schema['enum']}")


def require_local_file(
    recipe: ModelRecipe,
    name: str,
    value: Any,
    *,
    allow_directory: bool = False,
) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file() and not (allow_directory and path.is_dir()):
        kind = "file or directory" if allow_directory else "file"
        raise invalid_parameter(recipe.model_id, f"inputs.{name}", value, f"must point to an existing {kind}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Failed to load runtime metadata",
            path=str(path),
            reason=str(error),
        ) from error
    if not isinstance(value, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Runtime metadata must contain a mapping",
            path=str(path),
        )
    return value


def snapshot_golden_dirs(root: Path, components: tuple[str, ...]) -> dict[str, tuple[tuple[str, int, str], ...]]:
    return {
        relative: _golden_dir_state(root / relative)
        for component in components
        for relative in _golden_dirs_for_component(root, component)
    }


def collect_golden_dirs(
    root: Path,
    components: tuple[str, ...],
    *,
    before: Mapping[str, tuple[tuple[str, int, str], ...]] | None = None,
) -> dict[str, tuple[str, ...]]:
    before = before or {}
    result: dict[str, tuple[str, ...]] = {}
    for component in components:
        paths = [
            relative
            for relative in _golden_dirs_for_component(root, component)
            if before.get(relative) != _golden_dir_state(root / relative)
        ]
        result[component] = tuple(paths)
    missing = [component for component, paths in result.items() if not paths]
    if missing:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "Requested Golden components produced no artifacts",
            components=missing,
        )
    return result


def _golden_dirs_for_component(root: Path, component: str) -> list[str]:
    paths = []
    for path in root.rglob("step_*"):
        if re.fullmatch(r"step_[0-9]+", path.name) is None:
            continue
        if not path.is_dir() or not any(child.is_file() for child in path.rglob("*")):
            continue
        relative = path.relative_to(root)
        if any(matches_component_directory(part, component) for part in relative.parts[:-1]):
            paths.append(relative.as_posix())
    return sorted(paths)


def _golden_dir_state(path: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (child.relative_to(path).as_posix(), child.stat().st_size, _file_sha256(child))
        for child in sorted(path.rglob("*"))
        if child.is_file()
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "artifact_file",
    "collect_golden_dirs",
    "load_json",
    "require_local_file",
    "scoped_environment",
    "snapshot_golden_dirs",
    "validate_inputs",
]
