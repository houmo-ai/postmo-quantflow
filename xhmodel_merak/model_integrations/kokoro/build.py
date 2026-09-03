"""Package the current Kokoro release without rewriting model or weight names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan

from .artifact import load_metadata, validate_runtime_contract


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    # Workflow.export creates a dated release inside the transaction directory.
    # Lift that release as a unit; paths inside it remain relative and unchanged.
    root = Path(export_result.work_dir)
    stage = stage_path.resolve()
    if root.is_symlink() or root.resolve().parent != stage or not root.is_dir():
        raise _invalid(plan, "Kokoro export must produce one release directory inside the build stage")
    entries = list(root.iterdir())
    if not (root / "export_meta_info.json").is_file():
        raise _invalid(plan, "Kokoro release is missing export_meta_info.json")
    collisions = [entry.name for entry in entries if (stage / entry.name).exists()]
    if collisions:
        raise _invalid(plan, f"Kokoro release would overwrite staged files: {collisions}")
    for entry in entries:
        entry.rename(stage / entry.name)
    root.rmdir()
    # None deliberately bypasses the generic filename/external_data rewriter.
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    # The workflow already exports hf_config and a safe NumPy voice pack.
    # Do not rebuild a second frontend or copy files from the source checkpoint.
    del stage_path, plan


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    # Source-path sanitization is handled by the shared publisher. Kokoro's
    # metadata already uses release-relative graph and runtime-asset paths.
    del stage_path, plan


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    root = stage_path.resolve()
    meta_path = confined_file(root, "export_meta_info.json", "metadata_file")
    try:
        meta = load_metadata(meta_path)
        validate_runtime_contract(
            meta,
            components=plan.components,
            resolve_file=lambda value, field: confined_file(root, value, field),
        )
    except (ValueError, OSError) as error:
        raise _invalid(plan, str(error)) from error
    return {
        "metadata_file": meta_path.name,
        "runtime_kind": "kokoro_bucketed_cpu_phase",
        "release_prefix": meta["release_prefix"],
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
    }


def _invalid(plan: BuildPlan, message: str) -> ModelError:
    return ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, message, model=plan.model_id)


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
