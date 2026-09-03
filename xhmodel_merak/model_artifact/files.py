from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError


def confined_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Artifact field {field!r} must be a relative path",
            field=field,
            value=value,
        )
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in relative.parts or "\\" in value:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact path must remain inside the Artifact root",
            field=field,
            path=value,
        )
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Artifact runtime paths must not use symlinks",
                field=field,
                path=value,
            )
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact runtime file is missing or escapes the Artifact root",
            field=field,
            path=value,
        )
    return path


def artifact_file(root: Path, manifest: Mapping[str, Any], value: Any, field: str) -> Path:
    path = confined_file(root, value, field)
    artifact = manifest.get("artifact", {})
    files = artifact.get("files", ()) if isinstance(artifact, Mapping) else ()
    inventoried = {
        Path(str(entry.get("path"))).as_posix()
        for entry in files
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    }
    normalized = path.relative_to(root).as_posix()
    if normalized not in inventoried:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Manifest runtime file is absent from the verified file inventory",
            field=field,
            path=normalized,
        )
    return path


def matches_component_directory(name: str, component: str) -> bool:
    """Match exact stage names, including flat visual token gears."""
    return name == component or (
        component == "visual" and (name == "vision" or re.fullmatch(r"visual_m[1-9][0-9]*", name) is not None)
    )


def path_contains_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


__all__ = ["artifact_file", "confined_file", "matches_component_directory", "path_contains_symlink"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
