from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

from .dependencies import DependencySpec, dependency_profile, describe_dependencies


JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
AdapterKind: TypeAlias = Literal["build", "inference"]


@dataclass(frozen=True)
class ResolvedPrecision:
    weight: str
    activation: str
    components: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True)
class VariantRecipe:
    name: str
    workflow_config: str
    speculative_mode: str | None = None
    requires_lora: bool = False
    required_sources: tuple[str, ...] = ()
    extra_components: tuple[str, ...] = ()
    context_workflows: tuple[tuple[int, str], ...] = ()
    # First entry is the default; only list modes implemented by this variant.
    export_modes: tuple[str, ...] = ("standard",)
    dependencies: DependencySpec = field(default_factory=DependencySpec)

    def __post_init__(self) -> None:
        if (
            not self.export_modes
            or len(set(self.export_modes)) != len(self.export_modes)
            or not set(self.export_modes).issubset({"standard", "low_memory"})
        ):
            raise ValueError("export_modes must contain unique standard/low_memory modes")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_sources"] = list(self.required_sources)
        data["extra_components"] = list(self.extra_components)
        data["export_modes"] = list(self.export_modes)
        data["context_workflows"] = [
            {"max_context_length": max_context_length, "workflow_config": workflow_config}
            for max_context_length, workflow_config in self.context_workflows
        ]
        return data


@dataclass(frozen=True)
class DeploymentPrecisionBinding:
    """Components sharing one exporter QuantScheme, owned by their integration."""

    components: tuple[str, ...]
    config_path: str
    choices: tuple[str, ...]
    inherit_from: str | None = None

    def __post_init__(self) -> None:
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("precision binding components must be non-empty and unique")
        if not self.choices or any(not isinstance(value, str) or not value for value in self.choices):
            raise ValueError("precision binding choices must be non-empty quant_type strings")
        for path in (self.config_path, self.inherit_from):
            if path is not None and (
                not path.startswith("export.model.")
                or not path.endswith(".quant_scheme")
                or any(not part.isidentifier() for part in path.split("."))
            ):
                raise ValueError("precision binding must address an export.model QuantScheme")


@dataclass(frozen=True)
class ModelRecipe:
    model_id: str
    display_name: str
    family: str
    export_name: str
    task: str
    modalities: tuple[str, ...]
    variants: tuple[VariantRecipe, ...]
    default_precision: ResolvedPrecision
    supported_precisions: tuple[tuple[str, str], ...]
    quant_methods: tuple[str, ...]
    deployment_precision: dict[str, str]
    components: tuple[str, ...]
    golden_components: tuple[str, ...]
    build_adapter: str
    inference_adapter: str
    artifact_format: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    default_profile: str | None = None
    profiles: tuple[str, ...] = ()
    default_context_length: int | None = None
    default_prefill_chunk_length: int | None = None
    max_pe_length: int | None = None
    vision_profile: str | None = None
    vision_gears: tuple[int, ...] = ()
    attention_profiles: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    bundled_resources: tuple[tuple[str, str], ...] = ()
    resource_config_bindings: tuple[tuple[str, str], ...] = ()
    config_replacements: tuple[tuple[str, Any], ...] = ()
    source_constraints: tuple[tuple[str, Any], ...] = ()
    worker_capabilities: tuple[str, ...] = ()
    quant_capabilities: tuple[tuple[str, tuple[str, ...]], ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    fixed_artifact_name: str | None = None
    recipe_version: str = "2"
    required_runtime_files: tuple[str, ...] = ()
    config_resolver: str | None = None
    deployment_precision_bindings: tuple[DeploymentPrecisionBinding, ...] = ()
    # Reading an existing checkpoint is independent of registered quantizers.
    quant_checkpoint_methods: tuple[str, ...] = ()
    checkpoint_embedding_suffixes: tuple[str, ...] = ("embed_tokens.weight",)
    checkpoint_auxiliary_layers_field: str | None = None
    # Exact upstream sub-config type renames, owned by the model family.
    config_type_aliases: tuple[tuple[str, str, str], ...] = ()
    dependencies: DependencySpec = field(default_factory=DependencySpec)

    def __post_init__(self) -> None:
        alias_keys: set[tuple[str, str]] = set()
        for path, old, new in self.config_type_aliases:
            if (
                not path.endswith(".model_type")
                or any(not part.isidentifier() for part in path.split("."))
                or not old
                or not new
                or old == new
                or (path, old) in alias_keys
            ):
                raise ValueError("config type aliases must uniquely name a nested model_type rename")
            alias_keys.add((path, old))
        if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", self.export_name) is None:
            raise ValueError("export_name must contain lowercase tokens separated by underscores")
        if not self.checkpoint_embedding_suffixes or any(
            not isinstance(name, str) or not name.endswith(".weight") for name in self.checkpoint_embedding_suffixes
        ):
            raise ValueError("checkpoint embedding suffixes must identify weight tensors")
        if (
            self.checkpoint_auxiliary_layers_field is not None
            and not self.checkpoint_auxiliary_layers_field.isidentifier()
        ):
            raise ValueError("checkpoint auxiliary layer field must be a config key")
        if len(set(self.quant_checkpoint_methods)) != len(self.quant_checkpoint_methods) or any(
            method not in {"gptq", "autoround", "qat"} for method in self.quant_checkpoint_methods
        ):
            raise ValueError("quant_checkpoint_methods must contain unique supported producer names")
        if self.config_resolver is not None:
            _validate_internal_target(self.config_resolver, "recipe config resolver")
            if ":" not in self.config_resolver:
                raise ValueError("recipe config resolver must name a callable with module:attribute")
        bound_components: set[str] = set()
        bound_paths: set[str] = set()
        for binding in self.deployment_precision_bindings:
            if bound_components.intersection(binding.components) or binding.config_path in bound_paths:
                raise ValueError("deployment precision bindings must not overlap")
            defaults = {self.deployment_precision.get(component) for component in binding.components}
            if len(defaults) != 1 or not defaults.issubset(binding.choices):
                raise ValueError("bound components must share a supported default deployment precision")
            if not set(binding.components).issubset(self.components):
                raise ValueError("deployment precision bindings must refer to registered components")
            bound_components.update(binding.components)
            bound_paths.add(binding.config_path)

    @property
    def default_variant(self) -> VariantRecipe:
        return self.variants[0]

    def get_variant(self, name: str) -> VariantRecipe | None:
        return next((variant for variant in self.variants if variant.name == name), None)

    def to_summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "task": self.task,
            "modalities": list(self.modalities),
            "default_variant": self.default_variant.name,
            "variants": [variant.name for variant in self.variants],
            "default_precision": self.default_precision.to_dict(),
            "deployment_precision": dict(self.deployment_precision),
            "quant_methods": list(self.quant_methods),
            "worker_capabilities": list(self.worker_capabilities),
            "dependencies": {
                "schema_version": 1,
                "profiles": {
                    variant.name: {"status": dependency_profile(self, variant)["status"]}
                    for variant in self.variants
                },
            },
        }

    def to_description(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"profile": self.default_profile}
        if self.default_context_length is not None:
            defaults.update(
                context_length=self.default_context_length,
                prefill_chunk_length=self.default_prefill_chunk_length,
                max_pe_length=self.max_pe_length,
            )
        if self.vision_profile is not None:
            defaults["vision_profile"] = self.vision_profile
        return {
            **self.to_summary(),
            "recipe_version": self.recipe_version,
            "artifact_format": self.artifact_format,
            "dependencies": describe_dependencies(self),
            "workflow_variants": [
                {**variant.to_dict(), "dependencies": dependency_profile(self, variant)}
                for variant in self.variants
            ],
            "supported_precisions": [
                {"weight": weight, "activation": activation} for weight, activation in self.supported_precisions
            ],
            "deployment_precision_options": {
                component: {"choices": list(binding.choices), "shared_with": list(binding.components)}
                for binding in self.deployment_precision_bindings
                for component in binding.components
            },
            "components": list(self.components),
            "golden_components": list(self.golden_components),
            "profiles": list(self.profiles),
            "defaults": defaults,
            "vision": (
                {
                    "mode": "dynamic_token_gears",
                    "profile": self.vision_profile,
                    "image_token_gears": list(self.vision_gears),
                    "static_profiles": [],
                }
                if self.vision_profile
                else None
            ),
            "input_schema": deepcopy(self.input_schema),
            "output_schema": deepcopy(self.output_schema),
            "quant_capabilities": {method: list(values) for method, values in self.quant_capabilities},
            "quant_checkpoint_methods": list(self.quant_checkpoint_methods),
            "required_resources": list(self.required_resources),
            "bundled_resources": {name: path for name, path in self.bundled_resources},
        }


@dataclass(frozen=True)
class AdapterSpec:
    """An internal adapter name and lazy implementation; never persisted as artifact identity."""

    name: str
    kind: AdapterKind
    target: str
    artifact_formats: tuple[str, ...]
    source_config_for_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"build", "inference"}:
            raise ValueError(f"invalid adapter kind: {self.kind!r}")
        if not self.name or not self.target:
            raise ValueError("adapter name and target must be non-empty")
        _validate_internal_target(self.target, "adapter target")
        if not self.artifact_formats or len(set(self.artifact_formats)) != len(self.artifact_formats):
            raise ValueError("adapter artifact formats must be non-empty and unique")


@dataclass(frozen=True)
class ArtifactFormatSpec:
    """A registered artifact format contract shared by its adapters."""

    name: str
    validator: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact format name must be non-empty")
        if self.validator is not None:
            _validate_internal_target(self.validator, "artifact format validator")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise ValueError("artifact format version must be a positive integer")


def _validate_internal_target(target: str, label: str) -> None:
    module_name, separator, attribute = target.partition(":")
    names = module_name.split(".") + ([attribute] if separator else [])
    if not module_name.startswith("xhmodel_merak.") or any(
        re.fullmatch(r"[A-Za-z_]\w*", name) is None for name in names
    ):
        raise ValueError(f"{label} must be an internal dotted import: {target!r}")


__all__ = [
    "AdapterKind",
    "AdapterSpec",
    "ArtifactFormatSpec",
    "DeploymentPrecisionBinding",
    "JSONValue",
    "ModelRecipe",
    "ResolvedPrecision",
    "VariantRecipe",
]
