"""Build a compiler-facing Macaron LoRA export after Qwen export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto

from xhmodel_merak.utils import calculate_file_md5
from xhquant.api import get_xhquant_logger

from .lora_constant_bundle import (
    MacaronLoRAConstantBundle,
    MacaronLoRAConstantBundleArtifact,
    append_canonical_lora_to_stage,
)
from .lora_export_config import MacaronLoRAExportConfig
from .lora_export_plan import (
    MacaronLoRAExportPlan,
    build_macaron_lora_export_plan,
)
from .lora_weights import (
    ADAPTER_NAMES,
    MacaronLoRAAdapters,
    open_macaron_lora_adapters,
)
from .non_routed_lora_injection import (
    bind_non_routed_lora_graph,
    write_non_routed_lora_constants,
)
from .routed_expert_lora_injection import (
    LORA_MOEBLOCK_INPUT_COUNT,
    bind_routed_expert_lora_graph,
    moeblock_node_for_layer,
    write_routed_expert_lora_constants,
)


def _copy_export_work_files(
    base_export_dir: str | Path,
    destination: str | Path,
) -> None:
    """Keep the parent workflow YAML and log beside the model export."""

    base_export_dir = Path(base_export_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for entry in base_export_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, destination / entry.name)


def _write_lora_bundle(
    output_dir: Path,
    plan: MacaronLoRAExportPlan,
    adapters: MacaronLoRAAdapters,
    export_config: MacaronLoRAExportConfig,
) -> MacaronLoRAConstantBundleArtifact:
    with MacaronLoRAConstantBundle(output_dir) as bundle:
        write_non_routed_lora_constants(
            bundle,
            adapters,
            plan.non_routed,
        )
        write_routed_expert_lora_constants(
            bundle,
            adapters,
            plan.routed_expert,
        )
        artifact = bundle.close(macaron_lora=export_config.manifest_config())
    if artifact.input_order != plan.initializer_order:
        raise ValueError("Generated LoRA initializer order differs from the export plan")
    if tuple(entry["name"] for entry in artifact.schema) != artifact.input_order:
        raise ValueError("Generated LoRA schema differs from its initializer order")
    return artifact


def _bind_phase(
    output_dir: Path,
    phase: str,
    graph_path: Path,
    plan: MacaronLoRAExportPlan,
    bundle: MacaronLoRAConstantBundleArtifact,
    *,
    scaling: float,
) -> None:
    logger = get_xhquant_logger()
    canonical_references = append_canonical_lora_to_stage(
        output_dir,
        phase,
        bundle.adapter_onnx(ADAPTER_NAMES[0]),
    )
    if tuple(canonical_references) != bundle.input_order:
        raise ValueError(f"{phase}: canonical L0 order differs from the LoRA bundle")
    schema_by_name = {entry["name"]: entry for entry in bundle.schema}
    for name, reference in canonical_references.items():
        descriptor = schema_by_name[name]
        if (
            reference.data_type != descriptor["dtype"]
            or tuple(reference.dims) != tuple(descriptor["shape"])
        ):
            raise ValueError(f"{phase}: canonical L0 schema mismatch for {name}")
    model = onnx.load(str(graph_path), load_external_data=False)
    non_routed_count = bind_non_routed_lora_graph(
        model.graph,
        plan.non_routed,
        canonical_references,
    )
    routed_count = bind_routed_expert_lora_graph(
        model.graph,
        plan.routed_expert,
        canonical_references,
        scaling=scaling,
    )
    bound_order = tuple(
        initializer.name for initializer in model.graph.initializer if initializer.name in canonical_references
    )
    if bound_order != bundle.input_order:
        raise ValueError(f"{phase}: graph initializer order differs from the LoRA bundle")
    onnx.save(model, graph_path)
    logger.info(
        f"{phase}: bound {non_routed_count} non-routed Linear nodes and " f"{routed_count} routed MoeBlock nodes"
    )


def _validate_final_export(
    plan: MacaronLoRAExportPlan,
    bundle: MacaronLoRAConstantBundleArtifact,
) -> None:
    """Validate the compiler ABI before publishing the final directory."""

    if not bundle.manifest_path.is_file():
        raise FileNotFoundError(f"Missing Macaron LoRA manifest: {bundle.manifest_path}")
    for adapter_name in ADAPTER_NAMES:
        onnx_path = bundle.adapter_onnx(adapter_name)
        if not onnx_path.is_file() or not onnx_path.with_suffix(onnx_path.suffix + ".data").is_file():
            raise FileNotFoundError(f"Incomplete Macaron LoRA bundle for {adapter_name}")

    expected_names = set(bundle.input_order)
    for phase, graph_path in plan.graph_paths:
        graph = onnx.load(str(graph_path), load_external_data=False).graph
        initializer_by_name = {initializer.name: initializer for initializer in graph.initializer}
        missing = expected_names - set(initializer_by_name)
        if missing:
            raise ValueError(f"{phase}: missing {len(missing)} LoRA initializers")
        for name in bundle.input_order:
            initializer = initializer_by_name[name]
            if initializer.data_location != TensorProto.EXTERNAL:
                raise ValueError(f"{phase}: LoRA initializer is not external: {name}")
            locations = {entry.value for entry in initializer.external_data if entry.key == "location"}
            if not locations or any("pending" in location for location in locations):
                raise ValueError(f"{phase}: invalid LoRA data location for {name}")
        for binding in plan.routed_expert:
            node = moeblock_node_for_layer(graph, binding.layer_idx)
            if node is None or len(node.input) != LORA_MOEBLOCK_INPUT_COUNT:
                raise ValueError(
                    f"{phase}: layer {binding.layer_idx} does not use "
                    "the 28-input MoeBlock LoRA ABI"
                )


def _refresh_export_metadata(
    output_dir: Path,
    manifest_path: Path,
    export_config: MacaronLoRAExportConfig,
) -> None:
    metadata_path = output_dir / "golden_meta_info.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing final export metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["compiler_lora_constants"] = {
        "manifest": str(manifest_path.relative_to(output_dir)),
        "canonical_adapter": "L0",
        "binding": "initializer_name_replacement",
        "macaron_lora": export_config.manifest_config(),
    }
    metadata["macaron_runtime"] = {
        "adapter_binding": "compiler_initializer_name_replacement",
        "python_runtime_adapters": ["L0"],
        "fast_mode_supported": False,
    }
    for phase in ("prefill", "decode"):
        graph_path = next(
            iter((output_dir / phase).glob("*_with_act.onnx")),
            None,
        )
        if graph_path is None:
            raise FileNotFoundError(f"{phase}: no *_with_act.onnx graph found")
        metadata[f"{phase}_hmonnx_md5"] = calculate_file_md5(graph_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def inject_macaron_lora_export(
    *,
    model_dir: str | Path,
    base_export_dir: str | Path,
    base_model_export_dir: str | Path,
    final_export_dir: str | Path,
    export_config: MacaronLoRAExportConfig,
) -> Path:
    """Build and atomically publish a complete Macaron LoRA export."""

    final_export_dir = Path(final_export_dir)
    staging_dir = final_export_dir.with_name(f".{final_export_dir.name}.tmp")
    if final_export_dir.exists() or staging_dir.exists():
        raise FileExistsError("Macaron export destination already exists: " f"{final_export_dir}")

    base_model_export_dir = Path(base_model_export_dir)
    if not (base_model_export_dir / "golden_meta_info.json").is_file():
        raise FileNotFoundError("Base model export must contain golden_meta_info.json: " f"{base_model_export_dir}")

    staged_model_dir = staging_dir / base_model_export_dir.name
    try:
        _copy_export_work_files(base_export_dir, staging_dir)
        shutil.copytree(base_model_export_dir, staged_model_dir)
        with open_macaron_lora_adapters(model_dir) as adapters:
            plan = build_macaron_lora_export_plan(
                staged_model_dir,
                adapters,
                export_config,
            )
            bundle = _write_lora_bundle(
                staged_model_dir,
                plan,
                adapters,
                export_config,
            )
            for phase, graph_path in plan.graph_paths:
                _bind_phase(
                    staged_model_dir,
                    phase,
                    graph_path,
                    plan,
                    bundle,
                    scaling=adapters.scaling,
                )
            _refresh_export_metadata(
                staged_model_dir,
                bundle.manifest_path,
                export_config,
            )
            _validate_final_export(plan, bundle)
        staging_dir.replace(final_export_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    final_model_export_dir = final_export_dir / base_model_export_dir.name
    get_xhquant_logger().info(f"Macaron final export: {final_model_export_dir}")
    return final_model_export_dir


def cleanup_macaron_base_export(
    *,
    base_export_dir: str | Path,
    final_export_dir: str | Path,
) -> None:
    """Move the active export log, then remove the disposable base export."""

    base_export_dir = Path(base_export_dir)
    base_resolved = base_export_dir.resolve()
    final_resolved = Path(final_export_dir).resolve()
    if (
        base_resolved == final_resolved
        or final_resolved in base_resolved.parents
        or base_resolved in final_resolved.parents
    ):
        raise ValueError("Refusing to delete final export or its parent: " f"{base_resolved}")
    if not base_export_dir.exists():
        return

    # BaseLLMWorkflow keeps export_hmonnx.log open for the rest of the
    # process. Unlinking it on NFS creates a temporary .nfs* file and makes
    # rmtree fail with ENOTEMPTY. Both roots are siblings, so rename the open
    # log into the completed final export first. The existing file descriptor
    # then continues writing to the delivered log without holding the base
    # directory open.
    base_log = base_export_dir / "export_hmonnx.log"
    final_log = Path(final_export_dir) / base_log.name
    if base_log.is_file():
        base_log.replace(final_log)
    shutil.rmtree(base_export_dir)


__all__ = [
    "cleanup_macaron_base_export",
    "inject_macaron_lora_export",
]
