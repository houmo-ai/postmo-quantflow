"""Build transaction orchestration. Model behavior lives in registered integrations."""

from __future__ import annotations

import json
from pathlib import Path

from xhmodel_merak._environment import scoped_environment as _scoped_environment
from xhmodel_merak._requirements import validate_worker_capabilities, validate_worker_requirements
from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact import EFFECTIVE_WORKFLOW_NAME, MANIFEST_NAME
from xhmodel_merak.model_artifact.export_info import publish_export_info

from . import checkpoint, embeddings, fingerprints, manifest, packaging, workflow, workspace
from .contracts import BuildPlan, BuildResult
from .dependencies import write_dependency_files


def execute_build(plan: BuildPlan) -> BuildResult:
    from xhmodel_merak.model_catalog.loading import load_adapter

    adapter = load_adapter("build", plan.build_adapter)
    validate_worker_requirements(plan.worker_requirements, model=plan.model_id, operation="build")
    validate_worker_capabilities(plan.worker_capabilities, model=plan.model_id, operation="build")
    output_path = Path(plan.output_dir)
    stage_path = workspace.unused_sibling(output_path, "xhbuild")
    recipe_path = workspace.unused_sibling(output_path, "recipe")
    stage_identity: tuple[int, int, int] | None = None
    recipe_identity: tuple[int, int, int] | None = None
    generated_quant_root: Path | None = None
    generated_quant_identity: tuple[int, int, int] | None = None
    quant_artifact = plan.quant_checkpoint
    build_succeeded = False
    output_lock = workspace.acquire_output_lock(output_path)
    try:
        recipe_identity = workspace.reserve_owned_directory(recipe_path, plan, "recipe_dir")
        stage_identity = workspace.reserve_owned_directory(stage_path, plan, "stage_dir")
        effective_source = recipe_path / EFFECTIVE_WORKFLOW_NAME
        fingerprints.verify_resolved_source_bindings(plan)
        workflow.materialize_workflow_config(plan, effective_source)
        source_fingerprints = fingerprints.fingerprint_build_sources(plan)
        fingerprints.verify_resolved_source_bindings(plan)
        with _scoped_environment(plan.environment):
            runner = workflow.create_workflow(plan.model_dir, str(effective_source))
            quant_result, generated_quant_root, generated_quant_identity, quant_metadata = (
                workflow.resolve_quant_result(
                    runner,
                    plan,
                    output_path,
                )
            )
            if getattr(quant_result, "quanted_model_dir", None):
                quant_artifact = str(Path(quant_result.quanted_model_dir).resolve())
            quant_inspection_path = plan.quant_checkpoint or quant_artifact

            export_result = runner.export(
                quant_result=quant_result,
                output_dir=str(stage_path),
                device=plan.device,
                config_overrides=plan.config_overrides,
            )
        workspace.reject_artifact_symlinks(stage_path)
        effective_workflow = packaging.normalize_effective_workflow(stage_path, export_result.config_file)
        workspace.reject_artifact_symlinks(stage_path)
        exported_name = adapter.normalize_export(stage_path, export_result, plan)
        workspace.reject_artifact_symlinks(stage_path)
        if exported_name is not None:
            packaging.normalize_published_names(stage_path, exported_name, plan.artifact_name)

        adapter.package_runtime(stage_path, plan)
        workspace.reject_artifact_symlinks(stage_path)
        fingerprints.verify_resource_fingerprints(plan)
        fingerprints.verify_resolved_source_bindings(plan)
        fingerprints.verify_build_source_fingerprints(plan, source_fingerprints)
        adapter.sanitize_artifact(stage_path, plan)
        workspace.reject_artifact_symlinks(stage_path)
        packaging.sanitize_effective_workflow(effective_workflow, plan)
        packaging.sanitize_published_text_artifacts(
            stage_path,
            plan,
            effective_workflow,
            extra_paths={
                "quant-artifact": quant_inspection_path,
                "recipe": str(recipe_path),
            },
        )
        packaging.validate_onnx_external_locations(stage_path)

        embedding_metadata = embeddings.write_embedding_bins(stage_path, plan)
        artifact_metadata = adapter.validate_artifact(stage_path, plan)
        final_quant_metadata = checkpoint.inspect_quant_artifact(quant_inspection_path, plan)
        if quant_metadata is not None and final_quant_metadata != quant_metadata:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact changed after its pre-export verification",
                model=plan.model_id,
                quant_artifact=quant_inspection_path,
            )
        quant_metadata = final_quant_metadata
        published_quant_artifact = (
            quant_artifact if plan.quant_checkpoint or plan.quant_output_dir is not None else None
        )
        write_dependency_files(stage_path, plan.model_id, plan.variant)
        manifest.write_child_manifests(stage_path, plan, effective_workflow, published_quant_artifact,
                                       quant_metadata, artifact_metadata, source_fingerprints, embedding_metadata)
        publish_export_info(stage_path, artifact_metadata, embedding_metadata=embedding_metadata)
        artifact_manifest = manifest.build_manifest(
            stage_path,
            plan,
            effective_workflow,
            published_quant_artifact,
            quant_metadata,
            embedding_metadata,
            artifact_metadata,
            source_fingerprints,
        )
        manifest_path = stage_path / MANIFEST_NAME
        manifest_path.write_text(json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        workspace.require_owned_directory(recipe_path, recipe_identity, plan, "recipe_dir")
        workspace.publish(stage_path, output_path, stage_identity)
        build_succeeded = True
        return BuildResult(
            output_dir=str(output_path),
            model_id=plan.model_id,
            task=plan.task,
            profile=plan.profile,
            manifest_file=str(output_path / MANIFEST_NAME),
            effective_workflow_file=str(output_path / EFFECTIVE_WORKFLOW_NAME),
            quant_artifact=published_quant_artifact,
            precision=plan.precision,
            deployment_precision=plan.deployment_precision,
            quant_method=plan.quant_method,
            context_length=plan.context_length,
            prefill_chunk_length=plan.prefill_chunk_length,
            max_pe_length=plan.max_pe_length,
            vision_profile=plan.vision_profile,
            vision_gears=plan.vision_gears,
            components=plan.components,
            loadable=bool(artifact_metadata["loadable"]),
        )
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(
            ErrorCode.EXPORT_FAILED,
            f"Build failed for model {plan.model_id!r} variant {plan.variant!r}: {exc}",
            model=plan.model_id,
            variant=plan.variant,
            output_dir=plan.output_dir,
            cause=type(exc).__name__,
        ) from exc
    finally:
        workspace.remove_owned_directory(recipe_path, recipe_identity)
        if generated_quant_root is not None:
            if plan.quant_output_dir is None or not build_succeeded:
                workspace.remove_owned_directory(generated_quant_root, generated_quant_identity)
            else:
                workspace.release_owned_directory(generated_quant_identity)
        workspace.remove_owned_directory(stage_path, stage_identity)
        workspace.release_output_lock(output_lock)
