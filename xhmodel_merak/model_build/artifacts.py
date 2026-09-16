from __future__ import annotations

from pathlib import Path

from xhmodel_merak.errors import ErrorCode, ModelError


def flatten_single_export_root(stage_path: Path) -> str:
    export_roots = sorted(path for path in stage_path.iterdir() if path.is_dir() and path.name.startswith("hmquant_"))
    if len(export_roots) != 1:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Expected exactly one hmquant export root, found {len(export_roots)} under {stage_path}",
            export_roots=[path.name for path in export_roots],
        )
    export_root = export_roots[0]
    empty_directories = []
    collisions = []
    for child in export_root.iterdir():
        destination = stage_path / child.name
        if not destination.exists() and not destination.is_symlink():
            continue
        # Some exporters remove their intermediate files but leave an empty
        # component work directory (e.g. Gemma audio). Only that empty shell
        # may be replaced; never merge trees or overwrite a file/symlink.
        if (
            child.is_dir()
            and not child.is_symlink()
            and destination.is_dir()
            and not destination.is_symlink()
            and not any(destination.iterdir())
        ):
            empty_directories.append(destination)
        else:
            collisions.append(child.name)
    collisions.sort()
    if collisions:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Cannot flatten exported Artifact because names collide: {collisions}",
            collisions=collisions,
        )
    for directory in empty_directories:
        directory.rmdir()
    for child in list(export_root.iterdir()):
        child.replace(stage_path / child.name)
    export_root.rmdir()
    return export_root.name


__all__ = ["flatten_single_export_root"]
