from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from xhmodel_merak.errors import CatalogRegistrationError

from .registry import ModelCatalog


@dataclass(frozen=True)
class _Provider:
    provider_id: str
    module_name: str
    register: Callable[[ModelCatalog], None]


def build_builtin_catalog() -> ModelCatalog:
    catalog = ModelCatalog()
    for provider in _discover_builtin_providers():
        provider.register(catalog)
    return catalog.freeze()


def _discover_builtin_providers() -> tuple[_Provider, ...]:
    integrations = importlib.import_module("xhmodel_merak.model_integrations")
    package_paths = getattr(integrations, "__path__", None)
    if package_paths is None:
        raise CatalogRegistrationError("built-in model integrations package has no package path")

    providers = []
    for item in pkgutil.iter_modules(package_paths):
        if not item.ispkg or item.name.startswith("_"):
            continue
        module_name = f"{integrations.__name__}.{item.name}.plugin"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                raise CatalogRegistrationError(f"built-in integration {item.name!r} has no plugin.py") from error
            raise
        providers.append(_provider_from_module(module_name, module))

    seen: dict[str, str] = {}
    for provider in providers:
        previous = seen.get(provider.provider_id)
        if previous is not None:
            raise CatalogRegistrationError(
                f"duplicate provider ID {provider.provider_id!r}: modules {previous!r} and {provider.module_name!r}"
            )
        seen[provider.provider_id] = provider.module_name
    return tuple(sorted(providers, key=lambda provider: (provider.provider_id, provider.module_name)))


def _provider_from_module(module_name: str, module: ModuleType) -> _Provider:
    provider_id = getattr(module, "PROVIDER_ID", None)
    register = getattr(module, "register", None)
    if not isinstance(provider_id, str) or not provider_id:
        raise CatalogRegistrationError(f"provider module {module_name!r} must declare a non-empty PROVIDER_ID")
    if not callable(register):
        raise CatalogRegistrationError(f"provider module {module_name!r} must declare register(catalog)")
    return _Provider(provider_id=provider_id, module_name=module_name, register=register)


__all__ = ["build_builtin_catalog"]
