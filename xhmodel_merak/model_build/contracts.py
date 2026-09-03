from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from xhmodel_merak.model_catalog.contracts import JSONValue, ResolvedPrecision


QuantMethod: TypeAlias = Literal["gptq", "autoround", "qat"]


class UnsetType:
    """Sentinel used to distinguish an omitted value from explicit ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()


@dataclass(frozen=True)
class SourceRef:
    path: str
    revision: str | None = None


@dataclass(frozen=True)
class ResourceRef:
    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class PrecisionRequest:
    weight: str | None = None
    activation: str | None = None
    components: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionRequest:
    width: int | None = None
    height: int | None = None
    profile: str | None = None
    token_gears: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SpeculativeRequest:
    mode: Literal["mtp", "dflash"]
    num_draft_tokens: int | None = None


@dataclass(frozen=True)
class LoRARequest:
    paths: tuple[str, ...]


@dataclass(frozen=True)
class BuildPlan:
    model_id: str
    recipe_version: str
    task: str
    modalities: tuple[str, ...]
    variant: str
    config_path: str
    source_config_sha256: str
    model_dir: str
    output_dir: str
    quant_output_dir: str | None
    artifact_name: str
    export_name: str
    profile: str | None
    precision: ResolvedPrecision
    deployment_precision: dict[str, str]
    quant_method: QuantMethod | None
    quant_checkpoint: str | None
    context_length: int | None
    prefill_chunk_length: int | None
    max_pe_length: int | None
    vision_profile: str | None
    vision_gears: tuple[int, ...]
    components: tuple[str, ...]
    golden_components: tuple[str, ...]
    sources: dict[str, SourceRef]
    resources: dict[str, ResourceRef]
    lora_paths: tuple[str, ...]
    lora_sha256: tuple[str, ...]
    config_overrides: dict[str, Any]
    config_replacements: dict[str, Any]
    field_sources: dict[str, str]
    build_adapter: str
    artifact_format: str
    worker_requirements: tuple[str, ...]
    environment: dict[str, str]
    device: str
    workflow_config: str = ""
    source_bindings: dict[str, dict[str, JSONValue]] = field(default_factory=dict)
    worker_capabilities: tuple[str, ...] = ()
    required_runtime_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "precision",
            "deployment_precision",
            "sources",
            "resources",
            "config_overrides",
            "config_replacements",
            "field_sources",
            "environment",
            "source_bindings",
        ):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = list(self.components)
        data["golden_components"] = list(self.golden_components)
        data["required_runtime_files"] = list(self.required_runtime_files)
        data["worker_requirements"] = list(self.worker_requirements)
        data["worker_capabilities"] = list(self.worker_capabilities)
        return data


@dataclass(frozen=True)
class BuildResult:
    output_dir: str
    model_id: str
    task: str
    profile: str | None
    manifest_file: str
    effective_workflow_file: str
    quant_artifact: str | None
    precision: ResolvedPrecision
    deployment_precision: dict[str, str]
    quant_method: QuantMethod | None
    context_length: int | None
    prefill_chunk_length: int | None
    max_pe_length: int | None
    vision_profile: str | None
    vision_gears: tuple[int, ...]
    components: tuple[str, ...]
    loadable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", deepcopy(self.precision))
        object.__setattr__(self, "deployment_precision", deepcopy(self.deployment_precision))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = list(self.components)
        data["vision_gears"] = list(self.vision_gears)
        return data

    @property
    def manifest_path(self) -> Path:
        return Path(self.manifest_file)


__all__ = [
    "BuildPlan",
    "BuildResult",
    "JSONValue",
    "LoRARequest",
    "PrecisionRequest",
    "QuantMethod",
    "ResolvedPrecision",
    "ResourceRef",
    "SourceRef",
    "SpeculativeRequest",
    "UNSET",
    "UnsetType",
    "VisionRequest",
]
