from __future__ import annotations

import importlib
import importlib.util
from importlib import metadata
from typing import Iterable

from packaging.requirements import Requirement
from packaging.version import InvalidVersion

from xhmodel_merak.errors import ErrorCode, ModelError


def validate_worker_requirements(requirements: Iterable[str], *, model: str, operation: str) -> None:
    incompatible = []
    for value in requirements:
        requirement = Requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        module_name = requirement.name.lower().replace("-", "_")
        installed = _installed_distribution_version(requirement.name)
        if installed is None and requirement.specifier:
            installed = _installed_module_version(module_name)
        if installed is None:
            if not requirement.specifier and _module_available(module_name):
                continue
            incompatible.append({"requirement": value, "installed": None})
        elif requirement.specifier:
            try:
                compatible = installed in requirement.specifier
            except InvalidVersion:
                compatible = False
            if not compatible:
                incompatible.append({"requirement": value, "installed": installed})
    if incompatible:
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"{operation} worker does not satisfy the registered dependency profile",
            model=model,
            operation=operation,
            incompatible_requirements=incompatible,
        )


def validate_worker_capabilities(capabilities: Iterable[str], *, model: str, operation: str) -> None:
    incompatible = [value for value in capabilities if not _capability_available(value)]
    if incompatible:
        raise ModelError(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"{operation} worker does not provide the registered Python capabilities",
            model=model,
            operation=operation,
            incompatible_capabilities=incompatible,
        )


def _capability_available(value: str) -> bool:
    predicate_prefix = "python-predicate:"
    if not value.startswith(predicate_prefix):
        return _python_symbol_available(value)

    predicate = _load_python_symbol(value.removeprefix(predicate_prefix))
    if not callable(predicate):
        return False
    try:
        return predicate() is True
    except Exception:
        return False


def _python_symbol_available(value: str) -> bool:
    return callable(_load_python_symbol(value))


def _load_python_symbol(value: str):
    module_name, separator, symbol_path = value.partition(":")
    if not separator or not module_name or not symbol_path:
        return None
    try:
        target = importlib.import_module(module_name)
        for part in symbol_path.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError):
        return None
    return target


def _installed_distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _installed_module_version(name: str) -> str | None:
    try:
        value = getattr(importlib.import_module(name), "__version__", None)
        return str(value) if value is not None else None
    except Exception:
        return None


__all__ = ["validate_worker_capabilities", "validate_worker_requirements"]
