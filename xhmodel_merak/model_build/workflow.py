"""Bridge the resolved build plan into existing quant/export Workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from xhmodel_merak.errors import ErrorCode, ModelError

from . import checkpoint, workspace
from .contracts import BuildPlan


def create_workflow(model_dir: str, config_path: str) -> Any:
    from xhmodel_merak.workflows import AutoWorkflow

    return AutoWorkflow.from_config(model_dir=model_dir, config_path=config_path)


def resolve_quant_result(
    workflow: Any,
    plan: BuildPlan,
    output_path: Path,
) -> tuple[Any, Path | None, tuple[int, int, int] | None, dict[str, Any] | None]:
    if plan.quant_checkpoint:
        from xhmodel_merak.xh_llm.workflows.result import QuantResult

        quant_metadata = checkpoint.inspect_quant_artifact(plan.quant_checkpoint, plan)
        return (
            QuantResult(
                raw_model_dir=str(Path(plan.model_dir).resolve()),
                quanted_model_dir=plan.quant_checkpoint,
                skipped=False,
                meta={"source": "public_quant_checkpoint"},
            ),
            None,
            None,
            quant_metadata,
        )

    if plan.quant_method is None:
        quant_root = workspace.unused_sibling(output_path, "quant-skip")
        quant_identity = workspace.reserve_owned_directory(quant_root, plan, "quant_output_dir")
        try:
            quant_result = workflow.quant(
                output_dir=str(quant_root),
                device=plan.device,
                config_overrides=plan.config_overrides,
            )
            if not workspace.owned_directory_matches(quant_root, quant_identity):
                raise ModelError(
                    ErrorCode.QUANT_ARTIFACT_MISMATCH,
                    "Quant output root was replaced while quantization was running",
                    model=plan.model_id,
                    quant_output_dir=str(quant_root),
                )
            if (
                getattr(quant_result, "skipped", None) is not True
                or getattr(quant_result, "quanted_model_dir", None) is not None
            ):
                raise ModelError(
                    ErrorCode.QUANT_ARTIFACT_MISMATCH,
                    "A build without quantization requires an explicitly skipped QuantResult without quanted_model_dir",
                    model=plan.model_id,
                )
        except BaseException:
            workspace.remove_owned_directory(quant_root, quant_identity)
            raise
        return quant_result, quant_root, quant_identity, None

    if plan.quant_output_dir:
        quant_root = Path(plan.quant_output_dir)
    else:
        quant_root = workspace.unused_sibling(output_path, "quant")
    quant_identity = workspace.reserve_owned_directory(quant_root, plan, "quant_output_dir")

    try:
        quant_result = workflow.quant(
            output_dir=str(quant_root),
            device=plan.device,
            config_overrides=plan.config_overrides,
        )
        if not workspace.owned_directory_matches(quant_root, quant_identity):
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant output root was replaced while quantization was running",
                model=plan.model_id,
                quant_output_dir=str(quant_root),
            )
        quanted_model_dir = getattr(quant_result, "quanted_model_dir", None)
        quanted_path = Path(quanted_model_dir) if quanted_model_dir else None
        if quanted_path is None or not quanted_path.is_dir():
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Checkpoint quantization did not return an existing quanted_model_dir",
                model=plan.model_id,
                quanted_model_dir=quanted_model_dir,
            )
        resolved_root = quant_root.resolve()
        resolved_quanted = quanted_path.resolve()
        if (
            quant_root.is_symlink()
            or workspace.path_has_symlink(resolved_root, quanted_path.absolute())
            or not (resolved_quanted == resolved_root or resolved_quanted.is_relative_to(resolved_root))
        ):
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Checkpoint quantization returned a directory outside quant_output_dir",
                model=plan.model_id,
                quanted_model_dir=str(quanted_path),
                quant_output_dir=str(quant_root),
            )
        quant_metadata = checkpoint.inspect_quant_artifact(str(quanted_path), plan)
    except BaseException:
        workspace.remove_owned_directory(quant_root, quant_identity)
        raise
    return quant_result, quant_root, quant_identity, quant_metadata


def materialize_workflow_config(plan: BuildPlan, destination: Path) -> None:
    source = Path(plan.config_path)
    source_bytes = source.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != plan.source_config_sha256:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Registry Workflow changed after BuildPlan resolution",
            model=plan.model_id,
            config_path=str(source),
            expected_sha256=plan.source_config_sha256,
            actual_sha256=actual_sha256,
        )
    data = yaml.safe_load(source_bytes.decode("utf-8")) or {}
    if not isinstance(data, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Registry Workflow is not a YAML mapping: {source}",
            model=plan.model_id,
            config_path=str(source),
        )
    for dotted_path, value in plan.config_replacements.items():
        _set_recipe_path(data, dotted_path, value, model=plan.model_id)
    destination.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _set_recipe_path(data: dict[str, Any], dotted_path: str, value: Any, *, model: str) -> None:
    current: Any = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Registry replacement path does not exist for model {model!r}: {dotted_path}",
                model=model,
                field=dotted_path,
            )
        current = current[part]
    if not isinstance(current, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Registry replacement parent is not a mapping for model {model!r}: {dotted_path}",
            model=model,
            field=dotted_path,
        )
    current[parts[-1]] = value
