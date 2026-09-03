"""Golden selection and narrowly scoped cleanup of runtime-generated outputs."""

from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_artifact.validation import _is_ephemeral_runtime_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe


def _remove_stale_golden_outputs(
    root: Path,
    artifact: Mapping[str, Any],
    components: Sequence[str],
) -> None:
    files = artifact.get("files", ())
    if not isinstance(files, list):
        return
    inventoried = {
        Path(str(entry.get("path"))).as_posix()
        for entry in files
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    }
    removable: list[Path] = []
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            return
        relative = path.relative_to(root)
        if (
            stat.S_ISREG(mode)
            and relative.as_posix() not in inventoried
            and _is_ephemeral_runtime_file(relative, components)
        ):
            removable.append(path)
    for path in removable:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Failed to remove stale unmanifested Golden output",
                path=path.relative_to(root).as_posix(),
                reason=str(error),
            ) from error


def _golden_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    entries = artifact.get("golden_artifacts", {})
    if not isinstance(entries, Mapping):
        return result
    for component, paths in entries.items():
        if not isinstance(component, str) or not isinstance(paths, Sequence) or isinstance(paths, str):
            continue
        checked = []
        for path in paths:
            checked.append(
                artifact_file(
                    root,
                    manifest,
                    path,
                    "artifact.golden_artifacts",
                )
                .relative_to(root)
                .as_posix()
            )
        result[component] = tuple(checked)
    return result


def _resolve_golden(
    recipe: ModelRecipe,
    components: tuple[str, ...],
    requested: Sequence[str] | None,
    artifacts: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if requested is None:
        return ()
    normalized = tuple(dict.fromkeys(str(value).strip() for value in requested if str(value).strip()))
    unknown = sorted(set(normalized) - set(recipe.golden_components))
    if unknown:
        raise ModelError(
            ErrorCode.GOLDEN_NOT_SUPPORTED,
            "Requested Golden component is not supported by this model",
            model=recipe.model_id,
            requested=unknown,
            supported=list(recipe.golden_components),
        )
    missing = sorted(set(normalized) - set(components))
    if missing:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Requested Golden component is missing from the Artifact",
            model=recipe.model_id,
            requested=missing,
        )
    requested_set = set(normalized)
    return tuple(component for component in recipe.golden_components if component in requested_set)
