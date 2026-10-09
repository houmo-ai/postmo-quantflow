"""Publish the workflow's complete bundle without rewriting graph names."""

import json
from pathlib import Path

from xhmodel_merak.model_artifact.export_info import validate_export_info
from xhmodel_merak.model_artifact.files import confined_file

from .artifact import validate_runtime


def normalize_export(stage_path, export_result, plan):
    stage, root = stage_path.resolve(), Path(export_result.work_dir)
    if root.is_symlink() or root.resolve().parent != stage or not root.is_dir():
        raise ValueError("Qwen-Image export must be one directory inside the build stage")
    validate_runtime(root, plan.components)
    entries = list(root.iterdir())
    if any((stage / entry.name).exists() for entry in entries):
        raise ValueError("Qwen-Image export collides with staged files")
    for entry in entries:
        entry.rename(stage / entry.name)
    root.rmdir()
    return None


def package_runtime(stage_path, plan):
    # The workflow already packages all processor/config/embedding resources.
    pass


def sanitize_artifact(stage_path, plan):
    # Shared publication sanitizes source paths in text resources.
    pass


def validate_artifact(stage_path, plan):
    root = stage_path.resolve()
    meta, entries = validate_runtime(root, plan.components,
                                     resolve_file=lambda path, field: confined_file(root, path, field))
    if plan.deployment_precision != {"image": meta["quant_type"]}:
        raise ValueError("runtime quant_type disagrees with planned deployment precision")
    # Direct workflow users need this index too. The public publisher reserves
    # the same name, so validate the workflow index before handing it ownership.
    index = root / "export_meta_info.json"
    if json.loads(index.read_text()) != entries:
        raise ValueError("workflow export inventory disagrees with runtime metadata")
    validate_export_info(root, entries)
    index.unlink()
    return dict(metadata_file="runtime_meta_info.json", export_entries=entries,
                runtime_kind="qwen_image21", components=list(plan.components),
                requires_source=False, loadable=True)
