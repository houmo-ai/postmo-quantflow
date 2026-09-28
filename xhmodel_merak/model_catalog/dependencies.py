"""Lightweight dependency declarations; querying never inspects the host environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


if TYPE_CHECKING:
    from .contracts import ModelRecipe, VariantRecipe


def merge_requirements(*groups: tuple[str, ...]) -> list[str]:
    """Keep markers intact and intersect constraints rather than overwriting them.

    Conditional requirements remain separate. Full platform/transitive conflict
    resolution belongs to pip; unconditional exact-pin conflicts fail here.
    """
    parsed: dict[str, list[Requirement]] = {}
    output: list[str] = []
    for group in groups:
        for value in group:
            requirement = Requirement(value)
            if requirement.url:
                url = urlsplit(requirement.url)
                if url.scheme not in {"https", "git+https"} or not url.netloc or url.username or url.password:
                    raise ValueError("Dependency URLs must be credential-free remote HTTPS sources")
                if url.scheme == "git+https":
                    revision = url.path.rsplit("@", 1)[-1]
                    if len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision):
                        raise ValueError("VCS dependencies must pin a full commit")
            name = canonicalize_name(requirement.name)
            previous = parsed.setdefault(name, [])
            unconditional = [r for r in [*previous, requirement] if r.marker is None]
            urls = {r.url for r in unconditional if r.url}
            if len(urls) > 1:
                raise ValueError(f"Conflicting dependency sources for {name}")
            specs = SpecifierSet(",".join(str(r.specifier) for r in unconditional if str(r.specifier)))
            pins = [s.version for s in specs if s.operator == "==" and "*" not in s.version]
            if any(not specs.contains(pin, prereleases=True) for pin in pins):
                raise ValueError(f"Conflicting dependency versions for {name}: {specs}")
            normalized = str(requirement)
            if normalized not in output:
                output.append(normalized)
                previous.append(requirement)
    return sorted(output, key=str.casefold)


@dataclass(frozen=True)
class DependencySpec:
    """Requirements are declarations, not a resolved transitive lockfile."""

    python: str = "==3.12.13"
    common: tuple[str, ...] = ()
    build: tuple[str, ...] = ()
    inference: tuple[str, ...] = ()
    quantization: tuple[tuple[str, tuple[str, ...]], ...] = ()
    checkpoint_build: tuple[str, ...] = ()
    system: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        SpecifierSet(self.python)
        merge_requirements(self.common, self.build)
        merge_requirements(self.common, self.inference)
        merge_requirements(self.common, self.build, self.checkpoint_build)
        if len(dict(self.quantization)) != len(self.quantization):
            raise ValueError("Duplicate quantization dependency method")
        for _, requirements in self.quantization:
            merge_requirements(self.common, self.build, requirements)


def dependency_profile(recipe: ModelRecipe, variant: VariantRecipe) -> dict:
    spec, extra = recipe.dependencies, variant.dependencies
    common = merge_requirements(spec.common, extra.common)
    build = merge_requirements(tuple(common), spec.build, extra.build)
    inference = merge_requirements(tuple(common), spec.inference, extra.inference)
    unresolved = sorted(set((*spec.unresolved, *extra.unresolved)))
    if not common:
        unresolved.append("Model dependency declaration is not populated")
    quantization = {}
    declared_quantizers = dict(spec.quantization)
    extra_quantizers = dict(extra.quantization)
    for method in recipe.quant_methods:
        requirements = declared_quantizers.get(method, ())
        extra_requirements = extra_quantizers.get(method, ())
        quantization[method] = {
            "pip": merge_requirements(tuple(build), requirements, extra_requirements),
            "unresolved": [
                "Release the tested modified GPTQModel source and its bundled AutoRound when selected",
            ] if method in {"gptq", "autoround"} else [],
        }
    return {
        "python": str(SpecifierSet(f"{spec.python},{extra.python}")),
        "build": {"pip": build},
        "inference": {"pip": inference},
        "quantization": quantization,
        "checkpoint_build": {
            "pip": merge_requirements(tuple(build), spec.checkpoint_build, extra.checkpoint_build),
        },
        "system": sorted(set((*spec.system, *extra.system))),
        "status": "blocked_distribution" if unresolved else "pending_install_validation",
        "unresolved": unresolved,
        "validation": {"clean_install": False, "variant_runtime": False},
        "audit": {"necessary_dependencies_complete": False},
    }


def describe_dependencies(recipe: ModelRecipe) -> dict:
    return {
        "schema_version": 1,
        "kind": "declared_requirements",
        "profiles": {v.name: dependency_profile(recipe, v) for v in recipe.variants},
    }
