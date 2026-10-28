"""Publish the family's already-complete three-graph portable bundle."""

import json
from pathlib import Path


def normalize_export(stage_path, export_result, plan):
    root = Path(export_result.work_dir)
    if root.is_symlink() or root.resolve().parent != stage_path.resolve():
        raise ValueError("Wall export must remain in its build stage")
    for path in root.iterdir():
        target = stage_path / path.name
        if target.exists():
            raise ValueError("Wall stage collision")
        path.rename(target)
    root.rmdir()


def package_runtime(stage_path, plan):
    pass


def sanitize_artifact(stage_path, plan):
    pass


def validate_artifact(stage_path, plan):
    from xhmodel_merak.model_artifact.export_info import validate_export_info
    from xhmodel_merak.xh_other_model.models.wall_oss.build import export_entries
    from xhmodel_merak.xh_other_model.models.wall_oss.runtime import read_metadata

    meta = read_metadata(stage_path)
    if plan.deployment_precision != {"policy": meta["quant_type"]}:
        raise ValueError("deployment precision mismatch")
    entries = export_entries(stage_path, meta)
    index = stage_path / "export_meta_info.json"
    if json.loads(index.read_text()) != entries:
        raise ValueError("Wall export inventory mismatch")
    validate_export_info(stage_path, entries)
    index.unlink()
    return dict(
        metadata_file="runtime_meta_info.json",
        export_entries=entries,
        runtime_kind="wall_oss",
        components=list(plan.components),
        requires_source=False,
        loadable=True,
    )
