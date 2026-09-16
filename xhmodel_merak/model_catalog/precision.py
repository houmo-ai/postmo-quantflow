"""Lightweight deployment precision contracts shared by build and artifact loading.

These choices select the base QuantScheme of exported components. Existing
packed checkpoint weights and explicit YAML node/operator exceptions are not
rewritten. No SDK import or checkpoint quantization belongs here.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .contracts import DeploymentPrecisionBinding, ModelRecipe


def resolve_deployment_precision(
    recipe: ModelRecipe,
    request: Mapping[str, str] | None,
    *,
    defaults: Mapping[str, str] | None = None,
) -> dict[str, str]:
    result = dict(recipe.deployment_precision if defaults is None else defaults)
    if request is None:
        return result
    if not isinstance(request, Mapping):
        raise ValueError("deployment_precision must map component names to quant_type strings")
    available = {component for binding in recipe.deployment_precision_bindings for component in binding.components}
    unknown = set(request) - available
    if unknown:
        raise ValueError(f"deployment precision is not configurable for components: {sorted(unknown, key=str)!r}")
    for binding in recipe.deployment_precision_bindings:
        values = [request[component] for component in binding.components if component in request]
        if not values:
            continue
        if any(not isinstance(value, str) or value not in binding.choices for value in values):
            raise ValueError(f"{binding.components!r} require a quant_type from {binding.choices!r}")
        if len(set(values)) != 1:
            raise ValueError(f"{binding.components!r} share one exporter QuantScheme and must use the same precision")
        result.update(dict.fromkeys(binding.components, values[0]))
    return result


def _get(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _scheme(data: Mapping[str, Any], binding: DeploymentPrecisionBinding) -> dict[str, Any]:
    value = _get(data, binding.config_path)
    if value is None and binding.inherit_from is not None:
        value = _get(data, binding.inherit_from)
    if not isinstance(value, Mapping) or not isinstance(value.get("quant_type"), str):
        raise ValueError(f"workflow has no valid QuantScheme for {binding.config_path}")
    return deepcopy(dict(value))


def deployment_precision_from_workflow(recipe: ModelRecipe, data: Mapping[str, Any]) -> dict[str, str]:
    result = dict(recipe.deployment_precision)
    for binding in recipe.deployment_precision_bindings:
        quant_type = _scheme(data, binding)["quant_type"]
        if quant_type not in binding.choices:
            raise ValueError(f"workflow quant_type {quant_type!r} is not supported for {binding.components!r}")
        result.update(dict.fromkeys(binding.components, quant_type))
    return result


def deployment_precision_replacements(
    recipe: ModelRecipe,
    data: Mapping[str, Any],
    resolved: Mapping[str, str],
) -> dict[str, Any]:
    """Materialize inherited schemes before changing their parents.

    For example, an omitted Gemma audio scheme inherits the language scheme.
    An explicit language-only choice must not silently change audio as well.
    All paths come from installed bindings, never from user input.
    """
    if not recipe.deployment_precision_bindings:
        return {}
    if not isinstance(_get(data, "export.model"), Mapping):
        raise ValueError("workflow export.model must be a mapping")
    if dict(resolved) == deployment_precision_from_workflow(recipe, data):
        return {}
    replacements: dict[str, Any] = {}
    for binding in recipe.deployment_precision_bindings:
        scheme = _scheme(data, binding)
        scheme["quant_type"] = resolved[binding.components[0]]
        parts = binding.config_path.split(".")
        current: Any = data
        for index, part in enumerate(parts[:-1]):
            if part not in current or current[part] is None:
                nested: Any = scheme
                for child in reversed(parts[index + 1 :]):
                    nested = {child: nested}
                replacements[".".join(parts[: index + 1])] = nested
                break
            current = current[part]
            if not isinstance(current, Mapping):
                raise ValueError(f"workflow parent of {binding.config_path} must be a mapping")
        else:
            replacements[binding.config_path] = scheme
    return replacements
