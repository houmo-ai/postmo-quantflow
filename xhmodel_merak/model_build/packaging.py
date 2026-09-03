"""Normalize and validate exported files before publication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact import EFFECTIVE_WORKFLOW_NAME

from . import workspace
from .contracts import BuildPlan


_PUBLISHED_TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".jinja",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def normalize_effective_workflow(stage_path: Path, config_file: str) -> Path:
    source = Path(config_file).absolute()
    root = stage_path.resolve()
    resolved_source = source.resolve()
    if (
        source.is_symlink()
        or not resolved_source.is_relative_to(root)
        or workspace.path_has_symlink(root, source)
        or not resolved_source.is_file()
    ):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Workflow did not persist a regular effective config inside {stage_path}: {source}",
            config_file=str(source),
        )
    destination = stage_path / EFFECTIVE_WORKFLOW_NAME
    if resolved_source != destination.resolve():
        if destination.exists():
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Multiple effective Workflow files were produced under {stage_path}",
                config_file=str(source),
            )
        source.replace(destination)
    return destination


def sanitize_effective_workflow(path: Path, plan: BuildPlan) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    replacements = {
        str(Path(plan.model_dir).resolve()): "source://model",
    }
    if plan.quant_checkpoint:
        replacements[str(Path(plan.quant_checkpoint).resolve())] = "source://quant-checkpoint"
    for name, source in plan.sources.items():
        replacements[str(Path(source.path).resolve())] = f"source://{name}"
    for name, resource in plan.resources.items():
        replacements[str(Path(resource.path).resolve())] = f"resource://{name}"
    for index, path_value in enumerate(getattr(plan, "lora_paths", ())):
        replacements[str(Path(path_value).resolve())] = f"source://lora/{index}"
    sanitized = _replace_exact_string_values(data, replacements)
    path.write_text(yaml.safe_dump(sanitized, allow_unicode=True, sort_keys=False), encoding="utf-8")


def sanitize_published_text_artifacts(
    stage_path: Path,
    plan: BuildPlan,
    effective_workflow: Path,
    *,
    extra_paths: Mapping[str, str | None] | None = None,
) -> None:
    for path in sorted(stage_path.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if path.suffix.lower() == ".log" or "provenance" in lowered:
            path.unlink()

    source_paths: dict[str, str] = {
        str(Path(plan.model_dir).resolve()): "source://model",
        str(Path(plan.config_path).resolve()): "source://workflow",
        str(stage_path.resolve()): "artifact://",
        str(Path(plan.output_dir).resolve()): "artifact://",
    }
    if plan.quant_checkpoint:
        source_paths[str(Path(plan.quant_checkpoint).resolve())] = "source://quant-checkpoint"
    if plan.quant_output_dir:
        source_paths[str(Path(plan.quant_output_dir).resolve())] = "source://quant-output"
    for name, source in plan.sources.items():
        source_paths[str(Path(source.path).resolve())] = f"source://{name}"
    for name, resource in plan.resources.items():
        source_paths[str(Path(resource.path).resolve())] = f"resource://{name}"
    for index, path_value in enumerate(getattr(plan, "lora_paths", ())):
        source_paths[str(Path(path_value).resolve())] = f"source://lora/{index}"
    for name, path_value in (extra_paths or {}).items():
        if path_value:
            source_paths[str(Path(path_value).resolve())] = f"source://{name}"

    replacements = sorted(source_paths.items(), key=lambda item: len(item[0]), reverse=True)
    leaks: list[dict[str, str]] = []
    for path in sorted(stage_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _PUBLISHED_TEXT_SUFFIXES:
            continue
        relative_path = path.relative_to(stage_path).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            sanitized = text
            for source, replacement in replacements:
                sanitized = sanitized.replace(source, replacement)
            if sanitized != text:
                path.write_text(sanitized, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Published text Artifact could not be validated: {relative_path}",
                model=plan.model_id,
                file=relative_path,
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        for source, _ in replacements:
            if source in sanitized:
                leaks.append(
                    {
                        "file": relative_path,
                        "source": source_paths[source],
                    }
                )

    effective_data = yaml.safe_load(effective_workflow.read_text(encoding="utf-8")) or {}
    for path in sorted(stage_path.rglob("*")):
        if (
            not path.is_file()
            or path.resolve() == effective_workflow.resolve()
            or path.suffix.lower() not in {".yaml", ".yml"}
        ):
            continue
        try:
            candidate = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if candidate == effective_data:
            path.unlink()
    if leaks:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published text Artifacts retain local build source paths",
            model=plan.model_id,
            leaked_paths=leaks,
        )


def _replace_exact_string_values(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_exact_string_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_exact_string_values(item, replacements) for key, item in value.items()}
    return value


def normalize_published_names(stage_path: Path, exported_name: str, artifact_name: str) -> None:
    if exported_name == artifact_name:
        return

    for onnx_path in _graph_paths(stage_path):
        _rewrite_onnx_external_locations(onnx_path, exported_name, artifact_name)

    paths = sorted(stage_path.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if exported_name not in path.name:
            continue
        destination = path.with_name(path.name.replace(exported_name, artifact_name))
        if destination.exists():
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Published-name normalization would overwrite {destination}",
                source=str(path),
                destination=str(destination),
            )
        path.replace(destination)

    for json_path in stage_path.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        normalized = _replace_string_values(payload, exported_name, artifact_name)
        if normalized != payload:
            json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validate_onnx_external_locations(stage_path)


def _rewrite_onnx_external_locations(onnx_path: Path, exported_name: str, artifact_name: str) -> None:
    import onnx

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    changed = False
    for tensor in _model_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location" and exported_name in entry.value:
                entry.value = entry.value.replace(exported_name, artifact_name)
                changed = True
    if not changed:
        return
    temporary = onnx_path.with_name(f".{onnx_path.name}.rename.tmp.onnx")
    onnx.save_model(model, str(temporary))
    temporary.replace(onnx_path)


def _onnx_tensors(graph: Any):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _onnx_tensors(attribute.g)
            for nested_graph in attribute.graphs:
                yield from _onnx_tensors(nested_graph)


def _model_tensors(model: Any):
    yield from _onnx_tensors(model.graph)
    for function in getattr(model, "functions", ()):
        yield from _node_tensors(function.node)
    for training in getattr(model, "training_info", ()):
        yield from _onnx_tensors(training.initialization)
        yield from _onnx_tensors(training.algorithm)


def _node_tensors(nodes: Any):
    for node in nodes:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _onnx_tensors(attribute.g)
            for nested_graph in attribute.graphs:
                yield from _onnx_tensors(nested_graph)


def _replace_string_values(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_string_values(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string_values(item, old, new) for key, item in value.items()}
    return value


def validate_onnx_external_locations(stage_path: Path) -> None:
    import onnx

    workspace.reject_artifact_symlinks(stage_path)
    root = stage_path.resolve()
    invalid: list[dict[str, str]] = []
    for onnx_path in _graph_paths(stage_path):
        try:
            model = onnx.load_model(str(onnx_path), load_external_data=False)
            onnx.checker.check_model(str(onnx_path), full_check=False)
        except Exception as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Published graph is not a valid ONNX/HMONNX model: {onnx_path}: {exc}",
                graph=onnx_path.relative_to(stage_path).as_posix(),
                cause=type(exc).__name__,
            ) from exc
        for tensor in _model_tensors(model):
            entries = {entry.key: entry.value for entry in tensor.external_data}
            location = entries.get("location")
            if location is not None:
                reason = None
                if (
                    not location
                    or "\\" in location
                    or PureWindowsPath(location).drive
                    or location.startswith("/")
                    or any(segment in {"", ".", ".."} for segment in location.split("/"))
                ):
                    reason = "location must be a portable relative Artifact path"
                else:
                    relative = Path(location)
                    candidate = onnx_path.parent / relative
                    external_path = candidate.resolve()
                    if not external_path.is_relative_to(root):
                        reason = "location escapes the Artifact directory"
                    elif workspace.path_has_symlink(root, candidate):
                        reason = "location resolves through a symlink"
                    elif not external_path.is_file():
                        reason = "external data file does not exist"
                    else:
                        reason = _validate_external_data_range(external_path, entries)
                if reason is not None:
                    invalid.append(
                        {
                            "onnx": onnx_path.relative_to(stage_path).as_posix(),
                            "location": location,
                            "reason": reason,
                        }
                    )
    if invalid:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published ONNX files contain unsafe or missing external-data locations",
            invalid_external_data=invalid,
        )


def _graph_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".onnx", ".hmonnx"})


def _validate_external_data_range(path: Path, entries: Mapping[str, str]) -> str | None:
    try:
        offset = int(entries.get("offset", "0"))
        length = int(entries["length"]) if "length" in entries else path.stat().st_size - offset
    except (OSError, TypeError, ValueError):
        return "offset and length must be valid decimal integers"
    if offset < 0 or length < 0 or offset + length > path.stat().st_size:
        return "offset and length exceed the external data file"

    checksum = entries.get("checksum")
    if checksum is None:
        return None
    algorithm = {40: "sha1", 64: "sha256"}.get(len(checksum))
    if algorithm is None or re.fullmatch(r"[0-9a-fA-F]+", checksum) is None:
        return "checksum must be a hexadecimal SHA-1 or SHA-256 digest"
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                return "external data ended before the declared length"
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != checksum.lower():
        return "checksum does not match the declared external data range"
    return None
