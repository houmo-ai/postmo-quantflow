# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from xhmodel_merak.model_artifact.export_info import validate_export_info
from xhmodel_merak.model_artifact.files import confined_file

from .artifact import validate_precision, validate_runtime


def normalize_export(stage_path, export_result, plan):
    stage, root = stage_path.resolve(), Path(export_result.work_dir)
    if root.is_symlink() or root.resolve().parent != stage or not root.is_dir():
        raise ValueError("Sol-H3 export must produce a single directory inside the transaction stage")
    validate_runtime(root, plan.components)
    files = list(root.iterdir())
    if any((stage / file.name).exists() for file in files):
        raise ValueError("Sol-H3 export collides with staged files")
    for file in files:
        file.rename(stage / file.name)
    root.rmdir()
    return None


def package_runtime(stage_path, plan):
    pass


def sanitize_artifact(stage_path, plan):
    pass


def validate_artifact(stage_path, plan):
    root = stage_path.resolve()
    meta, entries = validate_runtime(
        root, plan.components, resolve_file=lambda path, field: confined_file(root, path, field)
    )
    validate_precision(meta, deployment=plan.deployment_precision, checkpoint_weight=plan.precision.weight)
    if plan.profile != meta["profile"]:
        raise ValueError("Sol-H3 graph canvas disagrees with the build profile")
    index = root / "export_meta_info.json"
    if json.loads(index.read_text()) != entries:
        raise ValueError("Sol-H3 workflow inventory disagrees with runtime metadata")
    validate_export_info(root, entries)
    index.unlink()
    return dict(
        metadata_file="runtime_meta_info.json",
        export_entries=entries,
        runtime_kind="sol_h3",
        components=list(plan.components),
        requires_source=False,
        loadable=True,
    )
