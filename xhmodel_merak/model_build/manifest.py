"""Write the self-contained artifact contract and build provenance."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from xhmodel_merak.model_artifact import EFFECTIVE_WORKFLOW_NAME, MANIFEST_NAME, SCHEMA_VERSION
from xhmodel_merak.model_artifact.files import file_sha256
from xhmodel_merak.model_catalog.registry import get_artifact_format_spec

from . import workspace
from .contracts import BuildPlan


def build_manifest(
    stage_path: Path,
    plan: BuildPlan,
    effective_workflow: Path,
    quant_artifact: str | None,
    quant_metadata: dict[str, Any] | None,
    embedding_metadata: list[dict[str, Any]],
    artifact_metadata: dict[str, Any],
    source_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    effective_data = yaml.safe_load(effective_workflow.read_text(encoding="utf-8")) or {}
    export_quant_type = (
        effective_data.get("export", {}).get("model", {}).get("quant_scheme", {}).get("quant_type")
        if isinstance(effective_data, Mapping)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": plan.model_id,
        "export": {"model_name": plan.export_name, "artifact_name": plan.artifact_name},
        "task": plan.task,
        "modalities": list(plan.modalities),
        "recipe": {
            "version": plan.recipe_version,
            "variant": plan.variant,
            "source_config": plan.workflow_config or _logical_config_path(plan.config_path),
            "source_config_sha256": plan.source_config_sha256,
            "effective_workflow": EFFECTIVE_WORKFLOW_NAME,
            "effective_workflow_sha256": file_sha256(effective_workflow),
        },
        "parameters": {
            "precision": {
                "checkpoint": plan.precision.to_dict(),
                "deployment": plan.deployment_precision,
            },
            "quant_method": plan.quant_method,
            "profile": plan.profile,
            "context_length": plan.context_length,
            "prefill_chunk_length": plan.prefill_chunk_length,
            "max_pe_length": plan.max_pe_length,
            "vision_profile": plan.vision_profile,
            "vision_gears": list(plan.vision_gears),
            "components": list(plan.components),
            "golden_components": list(plan.golden_components),
            "field_sources": plan.field_sources,
        },
        "sources": _source_descriptors(plan, source_fingerprints),
        "resources": _resource_descriptors(plan),
        "quantization": {
            "checkpoint": {
                "method": plan.quant_method,
                "source": (
                    "reused" if plan.quant_checkpoint else ("skipped" if plan.quant_method is None else "generated")
                ),
                "artifact": None,
                "external_retained": quant_artifact is not None,
                "verification": quant_metadata,
            },
            "export": {"quant_type": export_quant_type},
        },
        "artifact": {
            **artifact_metadata,
            "embedding": embedding_metadata,
            "files": file_manifest(stage_path),
        },
        "format": {
            "name": plan.artifact_format,
            "version": get_artifact_format_spec(plan.artifact_format).version,
        },
    }


def _logical_config_path(value: str) -> str:
    path = Path(value).as_posix()
    marker = "configs_merak/"
    return path[path.index(marker) :] if marker in path else Path(value).name


def _source_descriptors(plan: BuildPlan, source_fingerprints: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": {**_directory_identity(Path(plan.model_dir)), "sha256": source_fingerprints["model"]}
    }
    for name, source in plan.sources.items():
        result[name] = {
            **_directory_identity(Path(source.path)),
            "revision": source.revision,
            "sha256": source_fingerprints[name],
        }
    if plan.lora_paths:
        result["lora"] = [
            {**_directory_identity(Path(path_value)), "sha256": sha256}
            for path_value, sha256 in zip(plan.lora_paths, plan.lora_sha256, strict=True)
        ]
    return result


def _resource_descriptors(plan: BuildPlan) -> dict[str, Any]:
    result = {}
    for name, resource in plan.resources.items():
        path = Path(resource.path)
        result[name] = {
            "kind": "directory" if path.is_dir() else "file",
            "sha256": resource.sha256 or (file_sha256(path) if path.is_file() else None),
        }
    return result


def _directory_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    return {
        "kind": "directory",
        "config_sha256": file_sha256(config) if config.is_file() else None,
    }


def file_manifest(root: Path) -> list[dict[str, Any]]:
    workspace.reject_artifact_symlinks(root)
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return files
