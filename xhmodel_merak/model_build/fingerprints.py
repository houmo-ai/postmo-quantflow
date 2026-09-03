"""Content and filesystem identity checks for build inputs."""

from __future__ import annotations

import hashlib
import stat as stat_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import file_sha256

from .contracts import BuildPlan


def resource_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"Resource checksums do not accept symlinks: {path}")
    if path.is_file():
        return file_sha256(path)

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"Resource checksums do not accept symlinks: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(child)))
        digest.update(b"\0")
    return digest.hexdigest()


def source_binding(path: Path) -> dict[str, Any]:
    info = path.stat(follow_symlinks=False)
    if not stat_module.S_ISDIR(info.st_mode):
        raise ValueError(f"source root is not a regular directory: {path}")
    config_path = path / "config.json"
    try:
        config_info = config_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        config_sha256 = None
    else:
        if not stat_module.S_ISREG(config_info.st_mode):
            raise ValueError(f"source config is not a regular file: {config_path}")
        config_sha256 = file_sha256(config_path)
    final_info = path.stat(follow_symlinks=False)
    if (final_info.st_dev, final_info.st_ino) != (info.st_dev, info.st_ino):
        raise ValueError(f"source root changed while its identity was being captured: {path}")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "config_sha256": config_sha256,
    }


def verify_resource_fingerprints(plan: BuildPlan) -> None:
    changed: list[dict[str, str]] = []
    for name, resource in plan.resources.items():
        try:
            actual = resource_sha256(Path(resource.path))
        except (OSError, ValueError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Cannot revalidate resource {name!r} after packaging: {exc}",
                model=plan.model_id,
                resource=name,
                path=resource.path,
            ) from exc
        if actual != resource.sha256:
            changed.append({"resource": name, "expected": str(resource.sha256), "actual": actual})
    lora_paths = getattr(plan, "lora_paths", ())
    lora_sha256 = getattr(plan, "lora_sha256", ())
    for index, (path_value, expected) in enumerate(zip(lora_paths, lora_sha256, strict=True)):
        try:
            actual = resource_sha256(Path(path_value))
        except (OSError, ValueError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Cannot revalidate LoRA adapter {index} after export: {exc}",
                model=plan.model_id,
                path=path_value,
            ) from exc
        if actual != expected:
            changed.append({"resource": f"lora[{index}]", "expected": expected, "actual": actual})
    if changed:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Build resources changed while the Artifact was being packaged",
            model=plan.model_id,
            changed_resources=changed,
        )


def fingerprint_build_sources(plan: BuildPlan) -> dict[str, str]:
    paths = {"model": plan.model_dir, **{name: source.path for name, source in plan.sources.items()}}
    fingerprints = {}
    for name, path_value in paths.items():
        try:
            fingerprints[name] = resource_sha256(Path(path_value))
        except (OSError, ValueError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Cannot fingerprint build source {name!r}: {exc}",
                model=plan.model_id,
                source=name,
                path=path_value,
            ) from exc
    return fingerprints


def verify_resolved_source_bindings(plan: BuildPlan) -> None:
    expected = getattr(plan, "source_bindings", {})
    if not expected:
        return
    paths = {"model": plan.model_dir, **{name: source.path for name, source in plan.sources.items()}}
    changed: list[dict[str, Any]] = []
    for name in sorted(set(expected) | set(paths)):
        path_value = paths.get(name)
        try:
            actual = source_binding(Path(path_value)) if path_value is not None else None
        except (OSError, ValueError) as exc:
            actual = {"error": str(exc)}
        if actual != expected.get(name):
            changed.append(
                {
                    "source": name,
                    "path": path_value,
                    "expected": expected.get(name),
                    "actual": actual,
                }
            )
    if changed:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Build source identity or config changed after BuildPlan resolution",
            model=plan.model_id,
            changed_sources=changed,
        )


def verify_build_source_fingerprints(plan: BuildPlan, expected: Mapping[str, str]) -> None:
    actual = fingerprint_build_sources(plan)
    changed = [
        {"source": name, "expected": fingerprint, "actual": actual.get(name)}
        for name, fingerprint in expected.items()
        if actual.get(name) != fingerprint
    ]
    if changed:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Build sources changed while the Artifact was being produced",
            model=plan.model_id,
            changed_sources=changed,
        )
