from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError, invalid_parameter, unsupported_combination
from xhmodel_merak.model_catalog.dependencies import dependency_profile
from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe
from xhmodel_merak.model_catalog.identity import (
    build_artifact_name as _build_artifact_name,
)
from xhmodel_merak.model_catalog.identity import (
    select_context_workflow as _select_context_workflow,
)
from xhmodel_merak.model_catalog.identity import (
    source_constraint_mismatches as _source_constraint_mismatches,
)
from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_catalog.resources import resolve_workflow_config

from .contracts import (
    UNSET,
    BuildPlan,
    LoRARequest,
    PrecisionRequest,
    ResolvedPrecision,
    ResourceRef,
    SourceRef,
    SpeculativeRequest,
    UnsetType,
    VisionRequest,
)
from .fingerprints import resource_sha256 as _resource_sha256
from .fingerprints import source_binding as _source_binding
from .recipe_config import resolve_recipe_config


def resolve_build_plan(
    *,
    model_id: str,
    model_dir: str,
    output_dir: str | None = None,
    device: str | None = None,
    low_memory: bool | None = None,
    quant_output_dir: str | None = None,
    sources: Mapping[str, SourceRef | Mapping[str, Any] | str] | None = None,
    resources: Mapping[str, Any] | None = None,
    profile: str | None = None,
    precision: PrecisionRequest | Mapping[str, Any] | None = None,
    deployment_precision: Mapping[str, str] | None = None,
    quant_method: str | None | UnsetType = UNSET,
    quant_checkpoint: str | None = None,
    context_length: int | None = None,
    prefill_chunk_length: int | None = None,
    components: Sequence[str] | None = None,
    vision: VisionRequest | Mapping[str, Any] | None = None,
    attention: str | None = None,
    speculative: SpeculativeRequest | Mapping[str, Any] | None = None,
    lora: LoRARequest | Mapping[str, Any] | None = None,
) -> BuildPlan:
    model = model_id
    recipe = get_recipe(model_id)
    resolved_device = device if device is not None else os.getenv("XHMODEL_BUILD_DEVICE", "cuda:0")
    if not isinstance(resolved_device, str) or not resolved_device.strip():
        raise invalid_parameter(model_id, "device", device, "must be a non-empty device string")
    source_dir = _existing_directory(model, "model_dir", model_dir)
    source_refs = _coerce_source_refs(model, sources)
    resource_refs = _coerce_resource_refs(model, resources)

    speculative_request = _coerce_speculative(model, speculative)
    lora_request = _coerce_lora(model, lora)
    variant = _select_variant(recipe, speculative_request, lora_request)
    resolved_low_memory = _resolve_low_memory(model, variant, low_memory)
    _validate_sources(model, variant, source_refs)
    _validate_resources(recipe, resource_refs)
    resource_refs.update(_resolve_bundled_resources(recipe))

    checkpoint_precision = False
    if (quant_checkpoint and precision is None
            and ("wmix", recipe.default_precision.activation) in recipe.supported_precisions
            and (Path(quant_checkpoint).expanduser() / "config.json").is_file()):
        from .checkpoint import checkpoint_has_mixed_precision
        if checkpoint_has_mixed_precision(quant_checkpoint, model=model_id):
            precision = {"weight": "wmix", "activation": recipe.default_precision.activation}
            checkpoint_precision = True
    resolved_precision, precision_source = _resolve_precision(recipe, precision, quant_method)
    if checkpoint_precision:
        precision_source = "quant_checkpoint"
    resolved_method, method_source = _resolve_quant_method(
        recipe,
        resolved_precision,
        quant_method,
        quant_checkpoint,
    )
    checkpoint_path = _resolve_quant_checkpoint(model, quant_checkpoint, resolved_method)

    context, prefill = _resolve_autoregressive_lengths(
        recipe,
        context_length=context_length,
        prefill_chunk_length=prefill_chunk_length,
    )

    selected_components = _resolve_components(model, recipe, variant, components)
    resolved_profile = _resolve_profile(recipe, profile)
    _validate_vision(model, recipe, vision)
    _validate_attention(recipe, attention)

    overrides: dict[str, Any] = {}
    if context is not None:
        overrides["export.model.context_max_length"] = context
    if prefill is not None:
        overrides["export.model.prefill_chunk_length"] = prefill
    if resolved_method is None:
        overrides["quant"] = None
    if speculative_request and speculative_request.num_draft_tokens is not None:
        overrides["export.model.num_draft_tokens"] = speculative_request.num_draft_tokens
    if lora_request:
        overrides["export.model.lora.path"] = list(lora_request.paths)

    config_replacements = dict(recipe.config_replacements)
    # This is installed recipe metadata, not a user override of an existing
    # YAML field. Other-model workflows may omit model_name entirely.
    config_replacements["export.model.model_name"] = recipe.export_name
    for resource_name, config_path in recipe.resource_config_bindings:
        config_replacements[config_path] = resource_refs[resource_name].path
    if recipe.max_pe_length is not None:
        config_replacements["export.model.max_pe_length"] = recipe.max_pe_length

    expected_artifact_name = _build_artifact_name(
        recipe,
        variant,
        recipe.export_name,
        resolved_precision,
        prefill,
        context,
    )
    if output_dir is None:
        from datetime import datetime

        output_dir = f"{expected_artifact_name}_{datetime.now():%Y%m%d}"
    output_path = _normalize_output_path(model, output_dir)
    quant_output_path = _normalize_quant_output_path(model, quant_output_dir, output_path)
    if quant_output_path is not None and checkpoint_path is not None:
        raise unsupported_combination(
            model,
            "quant_output_dir cannot be combined with quant_checkpoint",
            quant_output_dir=str(quant_output_path),
            quant_checkpoint=checkpoint_path,
        )
    if quant_output_path is not None and resolved_method is None:
        raise unsupported_combination(
            model,
            "quant_output_dir requires checkpoint quantization",
            quant_output_dir=str(quant_output_path),
            quant_method=None,
        )
    _validate_disjoint_build_paths(
        model,
        output_path=output_path,
        quant_output_path=quant_output_path,
        model_dir=source_dir,
        source_refs=source_refs,
        resource_refs=resource_refs,
        quant_checkpoint=checkpoint_path,
        lora=lora_request,
    )
    source_bindings = _capture_build_source_bindings(recipe.model_id, source_dir, source_refs)
    _validate_source_constraints(recipe, source_dir)
    workflow_config = _select_context_workflow(recipe, variant, context)
    config_path = resolve_workflow_config(workflow_config)
    recipe_config = resolve_recipe_config(
        recipe,
        variant,
        config_path,
        model_dir=source_dir,
        sources=source_refs,
        deployment_precision=deployment_precision,
        quant_checkpoint=checkpoint_path,
        precision=resolved_precision,
    )
    config_replacements.update(recipe_config.replacements)
    # Whole quant sections returned by family policies precede resource bindings.
    for resource_name, binding in recipe.resource_config_bindings:
        config_replacements.pop(binding, None)
        config_replacements[binding] = resource_refs[resource_name].path
    _verify_source_bindings_during_resolution(recipe.model_id, source_dir, source_refs, source_bindings)
    lora_paths = lora_request.paths if lora_request is not None else ()
    lora_sha256 = tuple(_resource_sha256(Path(path)) for path in lora_paths)
    dependencies = dependency_profile(recipe, variant)
    worker_requirements = dependencies["build"]["pip"]
    worker_capabilities = list(recipe.worker_capabilities)
    if checkpoint_path is not None:
        worker_requirements = dependencies["checkpoint_build"]["pip"]
    elif resolved_method is not None:
        worker_requirements = dependencies["quantization"][resolved_method]["pip"]
        worker_capabilities.extend(dict(recipe.quant_capabilities).get(resolved_method, ()))

    field_sources = {
        "variant": "user_explicit" if speculative_request or lora_request else "recipe_default",
        "precision": precision_source,
        "deployment_precision": "user_explicit" if deployment_precision is not None else "recipe_default",
        "quant_method": method_source,
        "context_length": "user_explicit" if context_length is not None else "recipe_default",
        "prefill_chunk_length": "user_explicit" if prefill_chunk_length is not None else "recipe_default",
        "components": "user_explicit" if components is not None else "recipe_default",
        "profile": "user_explicit" if profile is not None else "recipe_default",
        "low_memory": "user_explicit" if low_memory is not None else "recipe_default",
    }
    return BuildPlan(
        model_id=recipe.model_id,
        recipe_version=recipe.recipe_version,
        task=recipe.task,
        modalities=recipe.modalities,
        variant=variant.name,
        config_path=str(config_path),
        source_config_sha256=recipe_config.sha256,
        model_dir=str(source_dir),
        output_dir=str(output_path),
        quant_output_dir=str(quant_output_path) if quant_output_path is not None else None,
        artifact_name=expected_artifact_name,
        export_name=recipe.export_name,
        profile=resolved_profile,
        precision=resolved_precision,
        deployment_precision=recipe_config.deployment_precision,
        quant_method=resolved_method,
        quant_checkpoint=checkpoint_path,
        context_length=context,
        prefill_chunk_length=prefill,
        max_pe_length=recipe.max_pe_length,
        vision_profile=recipe.vision_profile,
        vision_gears=recipe.vision_gears,
        components=selected_components,
        golden_components=recipe.golden_components,
        sources=source_refs,
        resources=resource_refs,
        lora_paths=lora_paths,
        lora_sha256=lora_sha256,
        config_overrides=overrides,
        config_replacements=config_replacements,
        field_sources=field_sources,
        build_adapter=recipe.build_adapter,
        artifact_format=recipe.artifact_format,
        required_runtime_files=recipe.required_runtime_files,
        worker_requirements=tuple(dict.fromkeys(worker_requirements)),
        # Always bind this switch, including False, so the caller's environment
        # cannot silently select a different export path. The executor restores it.
        environment={**recipe.environment, "HUGE_MODEL_EXPORT_ENABLED": "1" if resolved_low_memory else "0"},
        low_memory=resolved_low_memory,
        device=resolved_device.strip(),
        workflow_config=workflow_config,
        source_bindings=source_bindings,
        worker_capabilities=tuple(dict.fromkeys(worker_capabilities)),
    )


def _resolve_low_memory(model: str, variant: VariantRecipe, value: bool | None) -> bool:
    if value is not None and not isinstance(value, bool):
        raise invalid_parameter(model, "low_memory", value, "must be a bool or None")
    mode = variant.export_modes[0] if value is None else ("low_memory" if value else "standard")
    if mode not in variant.export_modes:
        raise unsupported_combination(
            model,
            "Export mode is not registered for this variant",
            variant=variant.name,
            export_mode=mode,
            allowed_modes=list(variant.export_modes),
        )
    return mode == "low_memory"


def _resolve_autoregressive_lengths(
    recipe: ModelRecipe,
    *,
    context_length: int | None,
    prefill_chunk_length: int | None,
) -> tuple[int | None, int | None]:
    if recipe.default_context_length is None:
        if context_length is not None or prefill_chunk_length is not None:
            raise invalid_parameter(
                recipe.model_id,
                "context_length/prefill_chunk_length",
                {"context_length": context_length, "prefill_chunk_length": prefill_chunk_length},
                "these autoregressive parameters do not apply to this task",
            )
        return None, None
    if recipe.default_prefill_chunk_length is None or recipe.max_pe_length is None:
        raise RuntimeError(f"Incomplete autoregressive Registry contract for {recipe.model_id}")
    context = _positive_int(
        recipe.model_id,
        "context_length",
        context_length,
        default=recipe.default_context_length,
    )
    if context > recipe.max_pe_length:
        raise invalid_parameter(
            recipe.model_id,
            "context_length",
            context,
            f"must not exceed Recipe max_pe_length={recipe.max_pe_length}",
        )
    prefill = _positive_int(
        recipe.model_id,
        "prefill_chunk_length",
        prefill_chunk_length,
        default=recipe.default_prefill_chunk_length,
    )
    if prefill > context:
        raise unsupported_combination(
            recipe.model_id,
            "prefill_chunk_length must not exceed context_length",
            prefill_chunk_length=prefill,
            context_length=context,
        )
    return context, prefill


def _select_variant(
    recipe: ModelRecipe,
    speculative: SpeculativeRequest | None,
    lora: LoRARequest | None,
) -> VariantRecipe:
    variant_name = (f"lora_{speculative.mode}" if speculative and lora else
                    "lora" if lora else speculative.mode if speculative else recipe.default_variant.name)
    variant = recipe.get_variant(variant_name)
    if variant is None:
        allowed_variants = [item.name for item in recipe.variants]
        raise unsupported_combination(
            recipe.model_id,
            f"variant {variant_name!r} is not registered; allowed variants are {allowed_variants}",
            variant=variant_name,
        )
    return variant


def _resolve_precision(
    recipe: ModelRecipe,
    value: PrecisionRequest | Mapping[str, Any] | None,
    quant_method: str | None | UnsetType,
) -> tuple[ResolvedPrecision, str]:
    request = _coerce_precision(recipe.model_id, value)
    if request is None:
        if quant_method is None:
            w16 = ("w16", recipe.default_precision.activation)
            if w16 not in recipe.supported_precisions:
                raise unsupported_combination(
                    recipe.model_id,
                    "quant_method=null is not registered for this model",
                    quant_method=None,
                    supported_precisions=list(recipe.supported_precisions),
                )
            return ResolvedPrecision(weight=w16[0], activation=w16[1]), "derived"
        return recipe.default_precision, "recipe_default"

    weight = request.weight or recipe.default_precision.weight
    activation = request.activation or recipe.default_precision.activation
    if request.components:
        raise invalid_parameter(
            recipe.model_id,
            "precision.components",
            request.components,
            "checkpoint component widths are fixed by the recipe; graph precision uses deployment_precision",
        )
    if (weight, activation) not in recipe.supported_precisions:
        raise invalid_parameter(
            recipe.model_id,
            "precision",
            {"weight": weight, "activation": activation},
            f"allowed profiles are {list(recipe.supported_precisions)!r}",
        )
    components = recipe.default_precision.components if weight == recipe.default_precision.weight else {}
    return ResolvedPrecision(weight=weight, activation=activation, components=components), "user_explicit"


def _resolve_quant_method(
    recipe: ModelRecipe,
    precision: ResolvedPrecision,
    value: str | None | UnsetType,
    quant_checkpoint: str | None,
) -> tuple[str | None, str]:
    if isinstance(value, UnsetType):
        if quant_checkpoint and precision.weight != "w16":
            from .checkpoint import detect_checkpoint_method

            method = detect_checkpoint_method(quant_checkpoint, model=recipe.model_id)
            if method not in recipe.quant_checkpoint_methods:
                raise unsupported_combination(
                    recipe.model_id,
                    "Checkpoint producer is not supported by this model exporter",
                    quant_method=method,
                    supported_methods=list(recipe.quant_checkpoint_methods),
                )
            return method, "checkpoint_metadata"
        if not recipe.quant_methods:
            return None, "recipe_default"
        if precision.weight == "w16":
            return None, "derived"
        return recipe.quant_methods[0], "recipe_default"
    if value is None:
        if precision.weight != "w16":
            raise unsupported_combination(
                recipe.model_id,
                "quant_method=null requires W16 checkpoint precision",
                quant_method=None,
                precision=precision.to_dict(),
            )
        if quant_checkpoint:
            raise unsupported_combination(
                recipe.model_id,
                "quant_checkpoint is forbidden when quant_method=null",
                quant_method=None,
                quant_checkpoint=quant_checkpoint,
            )
        return None, "user_explicit"

    normalized = str(value).strip().lower()
    allowed_methods = recipe.quant_checkpoint_methods if quant_checkpoint else recipe.quant_methods
    if normalized not in allowed_methods:
        raise invalid_parameter(
            recipe.model_id,
            "quant_method",
            value,
            f"allowed methods are {list(allowed_methods)!r} or null for W16",
        )
    if precision.weight == "w16":
        raise unsupported_combination(
            recipe.model_id,
            "W16 checkpoint precision cannot run a low-bit quantization method",
            quant_method=normalized,
            precision=precision.to_dict(),
        )
    return normalized, "user_explicit"


def _resolve_components(
    model: str,
    recipe: ModelRecipe,
    variant: VariantRecipe,
    value: Sequence[str] | None,
) -> tuple[str, ...]:
    available = recipe.components + variant.extra_components
    if value is None:
        return available
    selected = tuple(dict.fromkeys(value))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise invalid_parameter(model, "components", list(value), f"unknown components: {unknown}")
    if set(selected) != set(available):
        raise invalid_parameter(
            model,
            "components",
            list(value),
            "the public Workflow exports the complete registered component set",
        )
    return tuple(component for component in available if component in selected)


def _resolve_profile(recipe: ModelRecipe, value: str | None) -> str | None:
    if not recipe.profiles:
        if value is not None:
            raise invalid_parameter(recipe.model_id, "profile", value, "this Recipe has no selectable profile")
        return None
    resolved = recipe.default_profile if value is None else value
    if resolved not in recipe.profiles:
        raise invalid_parameter(
            recipe.model_id,
            "profile",
            value,
            f"allowed profiles are {list(recipe.profiles)!r}",
        )
    return resolved


def _validate_vision(
    model: str,
    recipe: ModelRecipe,
    vision: VisionRequest | Mapping[str, Any] | None,
) -> None:
    if recipe.vision_profile is None:
        if vision is not None:
            raise invalid_parameter(
                model,
                "vision",
                vision,
                "this Recipe has no user-selectable vision geometry; use its registered deployment profile",
            )
        return
    request = _coerce_vision(model, vision)
    if request is None:
        return
    if request.width is not None or request.height is not None:
        raise invalid_parameter(
            model,
            "vision",
            vision,
            "static width/height profiles are deprecated; select the dynamic visualm profile",
        )
    requested_gears = tuple(request.token_gears) if request.token_gears is not None else recipe.vision_gears
    if (
        request.profile is not None and request.profile != recipe.vision_profile
    ) or requested_gears != recipe.vision_gears:
        raise invalid_parameter(
            model,
            "vision",
            vision,
            f"only dynamic profile {recipe.vision_profile!r} with gears {list(recipe.vision_gears)!r} is registered",
        )


def _validate_attention(recipe: ModelRecipe, value: str | None) -> None:
    if value is None:
        return
    normalized = value.lower()
    if normalized not in recipe.attention_profiles:
        raise invalid_parameter(
            recipe.model_id,
            "attention",
            value,
            f"allowed attention profiles are {list(recipe.attention_profiles)!r}",
        )


def _validate_sources(
    model: str,
    variant: VariantRecipe,
    sources: Mapping[str, SourceRef | Mapping[str, Any] | str],
) -> None:
    unknown = sorted(set(sources) - set(variant.required_sources))
    if unknown:
        raise invalid_parameter(
            model, "sources", list(sources), f"unknown sources for variant {variant.name!r}: {unknown}"
        )
    missing = sorted(set(variant.required_sources) - set(sources))
    if missing:
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Model {model!r} variant {variant.name!r} requires sources: {missing}",
            model=model,
            variant=variant.name,
            missing=missing,
        )


def _validate_resources(recipe: ModelRecipe, resources: Mapping[str, ResourceRef]) -> None:
    unknown = sorted(set(resources) - set(recipe.required_resources))
    if unknown:
        raise invalid_parameter(
            recipe.model_id,
            "resources",
            list(resources),
            f"unknown resources for this Recipe: {unknown}",
        )
    missing = sorted(set(recipe.required_resources) - set(resources))
    if missing:
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Model {recipe.model_id!r} requires resources: {missing}",
            model=recipe.model_id,
            missing=missing,
        )


def _resolve_bundled_resources(recipe: ModelRecipe) -> dict[str, ResourceRef]:
    resources: dict[str, ResourceRef] = {}
    for name, relative_path in recipe.bundled_resources:
        path = _resolve_packaged_resource(relative_path)
        resources[name] = ResourceRef(path=str(path), sha256=_resource_sha256(path))
    return resources


def _resolve_packaged_resource(relative_path: str) -> Path:
    for root in Path(__file__).resolve().parents:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Packaged build resource does not exist: {relative_path}")


def _validate_source_constraints(recipe: ModelRecipe, model_dir: Path) -> None:
    if not recipe.source_constraints:
        return
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Model source for {recipe.model_id!r} is missing config.json: {config_path}",
            model=recipe.model_id,
            path=str(config_path),
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            f"Cannot parse source config for {recipe.model_id!r}: {exc}",
            model=recipe.model_id,
            path=str(config_path),
        ) from exc
    mismatches = _source_constraint_mismatches(recipe, config)
    if mismatches:
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            f"model_dir does not match Registry identity {recipe.model_id!r}",
            model=recipe.model_id,
            path=str(model_dir),
            mismatches=mismatches,
        )


def _resolve_quant_checkpoint(model: str, value: str | None, method: str | None) -> str | None:
    if value is None:
        return None
    if method is None:
        raise unsupported_combination(
            model,
            "quant_checkpoint requires a checkpoint quantization method",
            quant_checkpoint=value,
            quant_method=None,
        )
    return str(_existing_directory(model, "quant_checkpoint", value))


def _existing_directory(model: str, field: str, value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"{field} for model {model!r} is not an existing directory: {path}",
            model=model,
            field=field,
            path=str(path),
        )
    return path


def _normalize_output_path(model: str, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise invalid_parameter(model, "output_dir", value, "must be a non-empty directory path")
    requested = Path(value).expanduser()
    path = requested.resolve()
    if path.exists():
        raise invalid_parameter(model, "output_dir", value, "must not already exist")
    if not path.parent.is_dir():
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Parent directory for output_dir does not exist: {path.parent}",
            model=model,
            field="output_dir",
            path=str(path),
        )
    return path


def _normalize_quant_output_path(model: str, value: str | None, output_path: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise invalid_parameter(model, "quant_output_dir", value, "must be a non-empty directory path")
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise invalid_parameter(model, "quant_output_dir", value, "must not already exist")
    if path == output_path or path.is_relative_to(output_path) or output_path.is_relative_to(path):
        raise invalid_parameter(
            model,
            "quant_output_dir",
            value,
            "must not equal, contain, or be contained by output_dir",
        )
    if not path.parent.is_dir():
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Parent directory for quant_output_dir does not exist: {path.parent}",
            model=model,
            field="quant_output_dir",
            path=str(path),
        )
    return path


def _validate_disjoint_build_paths(
    model: str,
    *,
    output_path: Path,
    quant_output_path: Path | None,
    model_dir: Path,
    source_refs: Mapping[str, SourceRef],
    resource_refs: Mapping[str, ResourceRef],
    quant_checkpoint: str | None,
    lora: LoRARequest | None,
) -> None:
    inputs: dict[str, Path] = {"model_dir": model_dir}
    inputs.update({f"sources.{name}": Path(source.path) for name, source in source_refs.items()})
    inputs.update({f"resources.{name}": Path(resource.path) for name, resource in resource_refs.items()})
    if quant_checkpoint is not None:
        inputs["quant_checkpoint"] = Path(quant_checkpoint)
    if lora is not None:
        inputs.update({f"lora.paths[{index}]": Path(path) for index, path in enumerate(lora.paths)})

    outputs = {"output_dir": output_path}
    if quant_output_path is not None:
        outputs["quant_output_dir"] = quant_output_path
    for output_name, output in outputs.items():
        for input_name, input_path in inputs.items():
            if _paths_overlap(output, input_path):
                raise invalid_parameter(
                    model,
                    output_name,
                    str(output),
                    f"must not equal, contain, or be contained by {input_name}",
                )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _source_path(model: str, field: str, value: SourceRef | Mapping[str, Any] | str) -> str:
    if isinstance(value, SourceRef):
        path = value.path
    elif isinstance(value, str):
        path = value
    elif isinstance(value, Mapping):
        path = value.get("path") or value.get("model_dir")
    else:
        path = None
    if not isinstance(path, str) or not path:
        raise invalid_parameter(model, field, value, "must provide a non-empty path")
    return str(_existing_directory(model, field, path))


def _coerce_source_refs(
    model: str,
    values: Mapping[str, SourceRef | Mapping[str, Any] | str] | None,
) -> dict[str, SourceRef]:
    result: dict[str, SourceRef] = {}
    for name, value in (values or {}).items():
        if isinstance(value, SourceRef):
            path = value.path
            revision = value.revision
        elif isinstance(value, str):
            path = value
            revision = None
        elif isinstance(value, Mapping):
            path = value.get("path") or value.get("model_dir")
            revision = value.get("revision")
        else:
            path = None
            revision = None
        if not isinstance(path, str) or not path:
            raise invalid_parameter(model, f"sources.{name}", value, "must provide a non-empty directory path")
        resolved_path = _existing_directory(model, f"sources.{name}", path)
        normalized_revision = str(revision) if revision is not None else None
        if normalized_revision is not None:
            _verify_source_revision(model, str(name), resolved_path, normalized_revision)
        result[str(name)] = SourceRef(path=str(resolved_path), revision=normalized_revision)
    return result


def _coerce_resource_refs(
    model: str,
    values: Mapping[str, Any] | None,
) -> dict[str, ResourceRef]:
    result: dict[str, ResourceRef] = {}
    for name, value in (values or {}).items():
        if isinstance(value, ResourceRef):
            path = value.path
            sha256 = value.sha256
        elif isinstance(value, str):
            path = value
            sha256 = None
        elif isinstance(value, Mapping):
            path = value.get("path")
            sha256 = value.get("sha256")
        else:
            path = None
            sha256 = None
        if not isinstance(path, str) or not path:
            raise invalid_parameter(model, f"resources.{name}", value, "must provide a non-empty path")
        expanded = Path(path).expanduser().absolute()
        if _path_contains_symlink(expanded):
            raise invalid_parameter(model, f"resources.{name}.path", path, "must not resolve through a symlink")
        resolved = expanded.resolve()
        if not resolved.exists():
            raise ModelError(
                ErrorCode.RESOURCE_NOT_FOUND,
                f"resources.{name} for model {model!r} does not exist: {resolved}",
                model=model,
                field=f"resources.{name}",
                path=str(resolved),
            )
        if sha256 is not None and (not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None):
            raise invalid_parameter(
                model,
                f"resources.{name}.sha256",
                sha256,
                "must be a 64-character hexadecimal SHA-256 digest",
            )
        try:
            actual_sha256 = _resource_sha256(resolved)
        except ValueError as exc:
            raise invalid_parameter(
                model,
                f"resources.{name}.path",
                path,
                str(exc),
            ) from exc
        if sha256 is not None:
            if actual_sha256 != sha256.lower():
                raise invalid_parameter(
                    model,
                    f"resources.{name}.sha256",
                    sha256,
                    f"does not match the resource content (actual {actual_sha256})",
                )
        result[str(name)] = ResourceRef(
            path=str(resolved),
            sha256=actual_sha256,
        )
    return result


def _capture_build_source_bindings(
    model: str,
    model_dir: Path,
    source_refs: Mapping[str, SourceRef],
) -> dict[str, dict[str, Any]]:
    paths = {"model": model_dir, **{name: Path(source.path) for name, source in source_refs.items()}}
    bindings: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            bindings[name] = _source_binding(path)
        except (OSError, ValueError) as exc:
            field = "model_dir" if name == "model" else f"sources.{name}"
            raise invalid_parameter(model, field, str(path), f"cannot bind source identity: {exc}") from exc
    return bindings


def _verify_source_bindings_during_resolution(
    model: str,
    model_dir: Path,
    source_refs: Mapping[str, SourceRef],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    actual = _capture_build_source_bindings(model, model_dir, source_refs)
    changed = [name for name, binding in expected.items() if actual.get(name) != binding]
    if changed:
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            "Build sources changed while the BuildPlan was being resolved",
            model=model,
            changed_sources=changed,
        )


def _path_contains_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _verify_source_revision(model: str, name: str, path: Path, revision: str) -> None:
    config_path = path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise invalid_parameter(
            model,
            f"sources.{name}.revision",
            revision,
            f"requires a readable source config.json with _commit_hash: {exc}",
        ) from exc
    actual = config.get("_commit_hash") if isinstance(config, Mapping) else None
    if actual != revision:
        raise invalid_parameter(
            model,
            f"sources.{name}.revision",
            revision,
            f"does not match source config _commit_hash={actual!r}",
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_precision(model: str, value: PrecisionRequest | Mapping[str, Any] | None) -> PrecisionRequest | None:
    if value is None or isinstance(value, PrecisionRequest):
        return value
    if not isinstance(value, Mapping):
        raise invalid_parameter(model, "precision", value, "must be a PrecisionRequest or mapping")
    unknown = sorted(set(value) - {"weight", "activation", "components"})
    if unknown:
        raise invalid_parameter(model, "precision", value, f"unknown fields: {unknown}")
    components = value.get("components") or {}
    if not isinstance(components, Mapping):
        raise invalid_parameter(model, "precision.components", components, "must be a mapping")
    return PrecisionRequest(
        weight=value.get("weight"),
        activation=value.get("activation"),
        components=dict(components),
    )


def _coerce_speculative(
    model: str,
    value: SpeculativeRequest | Mapping[str, Any] | None,
) -> SpeculativeRequest | None:
    if value is None:
        return None
    if isinstance(value, SpeculativeRequest):
        request = value
    else:
        if not isinstance(value, Mapping):
            raise invalid_parameter(model, "speculative", value, "must be a SpeculativeRequest or mapping")
        unknown = sorted(set(value) - {"mode", "num_draft_tokens"})
        if unknown or value.get("mode") not in {"mtp", "dflash", "dflash2"}:
            raise invalid_parameter(model, "speculative", value, "requires mode='mtp', mode='dflash' or mode='dflash2'")
        request = SpeculativeRequest(mode=value["mode"], num_draft_tokens=value.get("num_draft_tokens"))
    num_draft_tokens = request.num_draft_tokens
    if num_draft_tokens is not None and (
        isinstance(num_draft_tokens, bool) or not isinstance(num_draft_tokens, int) or num_draft_tokens <= 0
    ):
        raise invalid_parameter(
            model,
            "speculative.num_draft_tokens",
            num_draft_tokens,
            "must be a positive integer",
        )
    return request


def _coerce_lora(model: str, value: LoRARequest | Mapping[str, Any] | None) -> LoRARequest | None:
    if value is None:
        return None
    if isinstance(value, LoRARequest):
        request = value
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - {"paths", "path"})
        paths = value.get("paths", value.get("path"))
        if unknown or not isinstance(paths, (list, tuple)):
            raise invalid_parameter(model, "lora", value, "requires a paths list")
        request = LoRARequest(paths=tuple(str(path) for path in paths))
    else:
        raise invalid_parameter(model, "lora", value, "must be a LoRARequest or mapping")
    if not request.paths:
        raise invalid_parameter(model, "lora.paths", request.paths, "must contain at least one adapter directory")
    resolved_paths = []
    for path in request.paths:
        expanded = Path(path).expanduser().absolute()
        if _path_contains_symlink(expanded):
            raise invalid_parameter(model, "lora.paths", path, "must not resolve through a symlink")
        resolved_paths.append(str(_existing_directory(model, "lora.paths", path)))
    resolved = tuple(resolved_paths)
    return LoRARequest(paths=resolved)


def _coerce_vision(model: str, value: VisionRequest | Mapping[str, Any] | None) -> VisionRequest | None:
    if value is None or isinstance(value, VisionRequest):
        return value
    if not isinstance(value, Mapping):
        raise invalid_parameter(model, "vision", value, "must be a VisionRequest or mapping")
    unknown = sorted(set(value) - {"width", "height", "profile", "token_gears"})
    if unknown:
        raise invalid_parameter(model, "vision", value, f"unknown fields: {unknown}")
    token_gears = value.get("token_gears")
    if token_gears is not None:
        if not isinstance(token_gears, (list, tuple)) or any(
            isinstance(gear, bool) or not isinstance(gear, int) or gear <= 0 for gear in token_gears
        ):
            raise invalid_parameter(model, "vision.token_gears", token_gears, "must be a list of positive integers")
        token_gears = tuple(token_gears)
    return VisionRequest(
        width=value.get("width"),
        height=value.get("height"),
        profile=value.get("profile"),
        token_gears=token_gears,
    )


def _positive_int(model: str, field: str, value: int | None, *, default: int) -> int:
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise invalid_parameter(model, field, value, "must be a positive integer")
    return resolved


__all__ = ["resolve_build_plan"]
