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
    collisions = sorted(path.name for path in export_root.iterdir() if (stage_path / path.name).exists())
    if collisions:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Cannot flatten exported Artifact because names collide: {collisions}",
            collisions=collisions,
        )
    for child in list(export_root.iterdir()):
        child.replace(stage_path / child.name)
    export_root.rmdir()
    return export_root.name


__all__ = ["flatten_single_export_root"]
