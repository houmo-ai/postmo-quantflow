from __future__ import annotations

from dataclasses import replace

import pytest

from xhmodel_merak.errors import CatalogRegistrationError, ModelError
from xhmodel_merak.model_catalog.contracts import AdapterSpec, ArtifactFormatSpec, VariantRecipe
from xhmodel_merak.model_catalog.registry import ModelCatalog, get_recipe


def _adapter(name: str, kind: str) -> AdapterSpec:
    return AdapterSpec(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        target=f"xhmodel_merak.test_adapters.{kind}",
        artifact_formats=("test",),
    )


def _recipe(model_id: str = "test-model"):
    return replace(
        get_recipe("ling-3.0-flash"),
        model_id=model_id,
        variants=(VariantRecipe(name="full", workflow_config="test.yaml"),),
        build_adapter="test-build",
        inference_adapter="test-inference",
        artifact_format="test",
    )


def _catalog_with_adapters() -> ModelCatalog:
    catalog = ModelCatalog()
    catalog.register_artifact_format(ArtifactFormatSpec(name="test"), provider="test")
    catalog.register_adapter(_adapter("test-build", "build"), provider="test")
    catalog.register_adapter(_adapter("test-inference", "inference"), provider="test")
    return catalog


@pytest.mark.parametrize(
    "target", ["os:system", "user.plugin:resolve", "xhmodel_merak.model_integrations.qwen.planning"]
)
def test_recipe_rejects_unregistered_code_locations(target):
    with pytest.raises(ValueError, match="recipe config resolver"):
        replace(_recipe(), config_resolver=target)


def test_catalog_rejects_duplicate_model_ids_with_both_providers() -> None:
    catalog = _catalog_with_adapters()
    catalog.register_recipe(_recipe(), provider="first")

    with pytest.raises(CatalogRegistrationError, match="providers 'first' and 'second'"):
        catalog.register_recipe(_recipe(), provider="second")


def test_catalog_rejects_duplicate_adapter_names() -> None:
    catalog = ModelCatalog()
    catalog.register_adapter(_adapter("same", "build"), provider="first")

    with pytest.raises(CatalogRegistrationError, match="providers 'first' and 'second'"):
        catalog.register_adapter(_adapter("same", "build"), provider="second")


def test_catalog_rejects_duplicate_artifact_formats() -> None:
    catalog = ModelCatalog()
    catalog.register_artifact_format(ArtifactFormatSpec(name="same"), provider="first")

    with pytest.raises(CatalogRegistrationError, match="providers 'first' and 'second'"):
        catalog.register_artifact_format(ArtifactFormatSpec(name="same"), provider="second")


def test_catalog_rejects_duplicate_variant_names() -> None:
    catalog = _catalog_with_adapters()
    recipe = replace(
        _recipe(),
        variants=(
            VariantRecipe(name="full", workflow_config="one.yaml"),
            VariantRecipe(name="full", workflow_config="two.yaml"),
        ),
    )

    with pytest.raises(CatalogRegistrationError, match="duplicate variant"):
        catalog.register_recipe(recipe, provider="test")


def test_catalog_validates_adapter_format_and_freezes_mutation() -> None:
    catalog = _catalog_with_adapters()
    catalog.register_recipe(_recipe(), provider="test")
    catalog.freeze()

    assert catalog.get_recipe("test-model").model_id == "test-model"
    with pytest.raises(CatalogRegistrationError, match="frozen"):
        catalog.register_recipe(_recipe("late-model"), provider="late")


def test_catalog_rejects_missing_or_incompatible_adapters_on_freeze() -> None:
    missing = ModelCatalog()
    missing.register_artifact_format(ArtifactFormatSpec(name="test"), provider="test")
    missing.register_recipe(_recipe(), provider="test")
    with pytest.raises(CatalogRegistrationError, match="unknown build adapter"):
        missing.freeze()

    incompatible = ModelCatalog()
    incompatible.register_artifact_format(ArtifactFormatSpec(name="test"), provider="test")
    incompatible.register_artifact_format(ArtifactFormatSpec(name="other"), provider="test")
    incompatible.register_adapter(_adapter("test-build", "build"), provider="test")
    incompatible.register_adapter(
        replace(_adapter("test-inference", "inference"), artifact_formats=("other",)),
        provider="test",
    )
    incompatible.register_recipe(_recipe(), provider="test")
    with pytest.raises(CatalogRegistrationError, match="not supported"):
        incompatible.freeze()


def test_catalog_rejects_unregistered_artifact_formats_on_freeze() -> None:
    catalog = ModelCatalog()
    catalog.register_adapter(_adapter("test-build", "build"), provider="test")
    catalog.register_adapter(_adapter("test-inference", "inference"), provider="test")
    catalog.register_recipe(_recipe(), provider="test")

    with pytest.raises(CatalogRegistrationError, match="unknown artifact format"):
        catalog.freeze()


def test_frozen_catalog_isolated_from_nested_recipe_mutation() -> None:
    catalog = _catalog_with_adapters()
    recipe = _recipe()
    catalog.register_recipe(recipe, provider="test")
    recipe.deployment_precision["provider_alias"] = "mutated"
    catalog.freeze()

    first = catalog.get_recipe("test-model")
    assert "provider_alias" not in first.deployment_precision
    first.deployment_precision["reader_alias"] = "mutated"
    first.input_schema["reader_alias"] = {"nested": ["mutated"]}

    second = catalog.get_recipe("test-model")
    listed = catalog.list_recipes()[0]
    assert "reader_alias" not in second.deployment_precision
    assert "reader_alias" not in second.input_schema
    assert "reader_alias" not in listed.deployment_precision


def test_unknown_build_adapter_name_never_reaches_importlib(monkeypatch: pytest.MonkeyPatch) -> None:
    import xhmodel_merak.model_catalog.loading as adapter_loader

    get_recipe("ling-3.0-flash")  # Initialize the installed catalog before observing runtime imports.
    imported = []
    monkeypatch.setattr(adapter_loader.importlib, "import_module", lambda name: imported.append(name))

    with pytest.raises(ModelError, match="not registered"):
        adapter_loader.load_adapter("build", "untrusted.module")

    assert imported == []


def test_unknown_inference_adapter_name_never_reaches_importlib(monkeypatch: pytest.MonkeyPatch) -> None:
    import xhmodel_merak.model_catalog.loading as adapter_loader

    get_recipe("ling-3.0-flash")
    imported = []
    monkeypatch.setattr(adapter_loader.importlib, "import_module", lambda name: imported.append(name))

    with pytest.raises(ModelError, match="not registered"):
        adapter_loader.load_adapter("inference", "untrusted.module")

    assert imported == []


def test_new_family_is_discovered_and_registers_without_a_central_model_list(monkeypatch):
    from types import ModuleType, SimpleNamespace

    from xhmodel_merak.model_catalog import discovery

    recipe = _recipe("new-family-model")
    provider = ModuleType("xhmodel_merak.model_integrations.new_family.plugin")
    provider.PROVIDER_ID = "new-family"

    def register(catalog):
        catalog.register_recipe(recipe, provider="new-family")
        catalog.register_adapter(_adapter("test-build", "build"), provider="new-family")
        catalog.register_adapter(_adapter("test-inference", "inference"), provider="new-family")
        catalog.register_artifact_format(ArtifactFormatSpec(name="test"), provider="new-family")

    provider.register = register
    original_import = discovery.importlib.import_module
    monkeypatch.setattr(
        discovery.pkgutil, "iter_modules", lambda _paths: [SimpleNamespace(ispkg=True, name="new_family")]
    )
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: provider if name == provider.__name__ else original_import(name),
    )

    catalog = discovery.build_builtin_catalog()
    assert catalog.frozen
    assert [item.model_id for item in catalog.list_recipes()] == ["new-family-model"]


def test_builtin_plugins_export_only_existing_declarations():
    from importlib import import_module

    from xhmodel_merak.model_catalog.discovery import _discover_builtin_providers

    providers = _discover_builtin_providers()
    assert providers
    for provider in providers:
        module = import_module(provider.module_name)
        assert "PROVIDER_ORDER" not in module.__all__
        for name in module.__all__:
            assert hasattr(module, name), f"{provider.module_name} exports missing {name!r}"
