from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from xhmodel_merak.errors import CatalogRegistrationError, ErrorCode, ModelError

from .contracts import AdapterKind, AdapterSpec, ArtifactFormatSpec, ModelRecipe


class ModelCatalog:
    """Mutable while built, immutable once published catalog of built-in integrations."""

    def __init__(self) -> None:
        self._recipes: dict[str, ModelRecipe] = {}
        self._recipe_providers: dict[str, str] = {}
        self._adapters: dict[tuple[AdapterKind, str], AdapterSpec] = {}
        self._adapter_providers: dict[tuple[AdapterKind, str], str] = {}
        self._artifact_formats: dict[str, ArtifactFormatSpec] = {}
        self._artifact_format_providers: dict[str, str] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register_recipe(self, recipe: ModelRecipe, *, provider: str) -> None:
        self._ensure_mutable()
        if not isinstance(recipe, ModelRecipe):
            raise CatalogRegistrationError(f"provider {provider!r} registered a non-ModelRecipe value")
        if not recipe.model_id:
            raise CatalogRegistrationError(f"provider {provider!r} registered an empty model ID")
        previous = self._recipe_providers.get(recipe.model_id)
        if previous is not None:
            raise CatalogRegistrationError(
                f"duplicate model ID {recipe.model_id!r}: providers {previous!r} and {provider!r}"
            )
        if not recipe.variants:
            raise CatalogRegistrationError(f"model {recipe.model_id!r} has no variants")
        variant_names = [variant.name for variant in recipe.variants]
        if len(set(variant_names)) != len(variant_names):
            raise CatalogRegistrationError(f"model {recipe.model_id!r} has duplicate variant names: {variant_names!r}")
        # A frozen dataclass can still contain mutable dictionaries. Keep the
        # provider's object graph outside the catalog so it cannot mutate a
        # published recipe through an alias retained during registration.
        self._recipes[recipe.model_id] = deepcopy(recipe)
        self._recipe_providers[recipe.model_id] = provider

    def register_adapter(self, adapter: AdapterSpec, *, provider: str) -> None:
        self._ensure_mutable()
        if not isinstance(adapter, AdapterSpec):
            raise CatalogRegistrationError(f"provider {provider!r} registered a non-AdapterSpec value")
        key = (adapter.kind, adapter.name)
        previous = self._adapter_providers.get(key)
        if previous is not None:
            raise CatalogRegistrationError(
                f"duplicate {adapter.kind} adapter name {adapter.name!r}: providers {previous!r} and {provider!r}"
            )
        self._adapters[key] = adapter
        self._adapter_providers[key] = provider

    def register_artifact_format(self, artifact_format: ArtifactFormatSpec, *, provider: str) -> None:
        self._ensure_mutable()
        if not isinstance(artifact_format, ArtifactFormatSpec):
            raise CatalogRegistrationError(f"provider {provider!r} registered a non-ArtifactFormatSpec value")
        previous = self._artifact_format_providers.get(artifact_format.name)
        if previous is not None:
            raise CatalogRegistrationError(
                f"duplicate artifact format {artifact_format.name!r}: providers {previous!r} and {provider!r}"
            )
        self._artifact_formats[artifact_format.name] = artifact_format
        self._artifact_format_providers[artifact_format.name] = provider

    def freeze(self) -> ModelCatalog:
        self._ensure_mutable()
        for adapter in self._adapters.values():
            self._validate_adapter_formats(adapter)
        for recipe in self._recipes.values():
            if recipe.artifact_format not in self._artifact_formats:
                provider = self._recipe_providers[recipe.model_id]
                raise CatalogRegistrationError(
                    f"model {recipe.model_id!r} from provider {provider!r} references unknown artifact format "
                    f"{recipe.artifact_format!r}"
                )
            self._validate_recipe_adapter(recipe, "build", recipe.build_adapter)
            self._validate_recipe_adapter(recipe, "inference", recipe.inference_adapter)
        self._frozen = True
        return self

    def list_recipes(self) -> tuple[ModelRecipe, ...]:
        self._ensure_frozen()
        return tuple(deepcopy(recipe) for _, recipe in sorted(self._recipes.items()))

    def get_recipe(self, model: str) -> ModelRecipe:
        self._ensure_frozen()
        recipe = self._recipes.get(model)
        if recipe is None:
            raise ModelError(
                ErrorCode.MODEL_NOT_FOUND,
                f"Unknown model Registry ID: {model!r}",
                model=model,
                allowed=sorted(self._recipes),
            )
        return deepcopy(recipe)

    def get_adapter(self, kind: AdapterKind, name: str) -> AdapterSpec:
        self._ensure_frozen()
        adapter = self._adapters.get((kind, name))
        if adapter is None:
            raise LookupError(f"unknown registered {kind} adapter name: {name!r}")
        return adapter

    def get_artifact_format(self, name: str) -> ArtifactFormatSpec:
        self._ensure_frozen()
        artifact_format = self._artifact_formats.get(name)
        if artifact_format is None:
            raise LookupError(f"unknown registered artifact format: {name!r}")
        return artifact_format

    def _validate_adapter_formats(self, adapter: AdapterSpec) -> None:
        missing = [name for name in adapter.artifact_formats if name not in self._artifact_formats]
        if missing:
            provider = self._adapter_providers[(adapter.kind, adapter.name)]
            raise CatalogRegistrationError(
                f"{adapter.kind} adapter {adapter.name!r} from provider {provider!r} references unknown artifact "
                f"formats: {missing!r}"
            )

    def _validate_recipe_adapter(self, recipe: ModelRecipe, kind: AdapterKind, name: str) -> None:
        adapter = self._adapters.get((kind, name))
        if adapter is None:
            provider = self._recipe_providers[recipe.model_id]
            raise CatalogRegistrationError(
                f"model {recipe.model_id!r} from provider {provider!r} references unknown {kind} adapter {name!r}"
            )
        if recipe.artifact_format not in adapter.artifact_formats:
            raise CatalogRegistrationError(
                f"model {recipe.model_id!r} uses artifact format {recipe.artifact_format!r}, which is not supported "
                f"by {kind} adapter {name!r}: {adapter.artifact_formats!r}"
            )

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise CatalogRegistrationError("model catalog is frozen")

    def _ensure_frozen(self) -> None:
        if not self._frozen:
            raise CatalogRegistrationError("model catalog has not been frozen")


@lru_cache(maxsize=1)
def get_catalog() -> ModelCatalog:
    from .discovery import build_builtin_catalog

    return build_builtin_catalog()


def list_recipes() -> tuple[ModelRecipe, ...]:
    return get_catalog().list_recipes()


def get_recipe(model: str) -> ModelRecipe:
    return get_catalog().get_recipe(model)


def get_adapter_spec(kind: AdapterKind, name: str) -> AdapterSpec:
    return get_catalog().get_adapter(kind, name)


def get_artifact_format_spec(name: str) -> ArtifactFormatSpec:
    return get_catalog().get_artifact_format(name)


__all__ = [
    "ModelCatalog",
    "get_adapter_spec",
    "get_artifact_format_spec",
    "get_catalog",
    "get_recipe",
    "list_recipes",
]
