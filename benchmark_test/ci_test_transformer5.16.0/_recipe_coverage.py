"""Repository consistency checks, without importing model implementations or Git.

Start from the implementation registry, not the catalog being checked. Workflow
YAML already identifies the implementation; recipes do not need another model
name or a duplicate list of source directories.
"""

from __future__ import annotations

import ast
import json
import re
import runpy
from pathlib import Path
from typing import TYPE_CHECKING

import yaml


if TYPE_CHECKING:
    from collections.abc import Iterable

    from xhmodel_merak.model_catalog.contracts import ModelRecipe


MODEL_ROOTS = {"xh_llm": "register_llm_model", "xh_other_model": "register_other_model"}
BASELINE = Path(__file__).with_name("recipe_legacy_gaps.json")


def _check_static_registrations(path: Path, register_name: str) -> None:
    """Reject declarations the production AST scanner would silently miss."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    decorators = {
        id(decorator)
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == register_name and alias.asname for alias in node.names):
                raise ValueError(f"{path}:{node.lineno}: do not alias {register_name}; use a literal decorator")
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name != register_name:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        model_type = keywords.get("model_type", node.args[0] if node.args else None)
        valid = (
            id(node) in decorators
            and len(node.args) <= 1
            and None not in keywords
            and "name" not in keywords
            and isinstance(model_type, ast.Constant)
            and isinstance(model_type.value, str)
            and bool(model_type.value.strip())
            and all(
                key not in keywords or isinstance(keywords[key], ast.Constant) and type(keywords[key].value) is bool
                for key in ("master", "force")
            )
        )
        if not valid:
            raise ValueError(
                f"{path}:{node.lineno}: {register_name} requires a literal model_type decorator "
                "and literal keyword master/force flags; dynamic registration is not discoverable"
            )


def scan_implementations(repo_root: Path) -> tuple[set[str], dict[tuple[str, str], str]]:
    packages: set[str] = set()
    model_types: dict[tuple[str, str], str] = {}
    for family, register_name in MODEL_ROOTS.items():
        package_root = repo_root / "xhmodel_merak" / family
        for path in sorted((package_root / "models").rglob("*.py")):
            relative = path.relative_to(package_root / "models")
            if len(relative.parts) == 1:
                if path.name != "__init__.py":
                    raise ValueError(f"{path}: model implementations must live in models/<family>/")
            else:
                packages.add(f"{family}/{relative.parts[0]}")
            _check_static_registrations(path, register_name)
        # Loading the scanners by filename avoids xh_llm/__init__.py and its SDK
        # imports. Keep forced-registration precedence identical to production.
        scanner = runpy.run_path(str(package_root / "scan_model_types.py"))
        for model_type, package in scanner["get_support_master_model_types"]().items():
            model_types[family, model_type] = f"{family}/{package}"
    return packages, model_types


def _workflow_target(repo_root: Path, config: str) -> tuple[str, str]:
    relative = Path(config)
    workflow_root = (repo_root / "configs_merak" / "workflows").resolve()
    path = (repo_root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(workflow_root):
        raise ValueError(f"{config}: recipe workflow must stay inside configs_merak/workflows/")
    if path.suffix not in {".yaml", ".yml"} or not path.is_file():
        raise ValueError(f"{config}: recipe workflow does not exist or is not YAML")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    export = data.get("export") if isinstance(data, dict) else None
    model = export.get("model") if isinstance(export, dict) else None
    if not isinstance(model, dict) or ("model_type" in model) == ("type" in model):
        raise ValueError(f"{config}: export.model must have exactly one of model_type or type")
    key, family = ("model_type", "xh_llm") if "model_type" in model else ("type", "xh_other_model")
    if not isinstance(model[key], str) or not model[key].strip():
        raise ValueError(f"{config}: export.model.{key} must be a non-empty string")
    return family, model[key]


def recipe_gaps(repo_root: Path, recipes: Iterable[ModelRecipe]) -> set[str]:
    """Return uncovered package/type identities; reject dangling recipe targets."""
    packages, model_types = scan_implementations(repo_root)
    covered: set[tuple[str, str]] = set()
    targets: dict[str, tuple[str, str]] = {}
    for recipe in recipes:
        if not recipe.variants:
            raise ValueError(f"{recipe.model_id}: recipe needs at least one workflow variant")
        for variant in recipe.variants:
            configs = (variant.workflow_config, *(path for _, path in variant.context_workflows))
            for config in configs:
                if config not in targets:
                    targets[config] = _workflow_target(repo_root, config)
                target = targets[config]
                if target not in model_types:
                    raise ValueError(
                        f"{recipe.model_id}: {config} targets {target[0]}:{target[1]}, "
                        "which is not a registered master model"
                    )
                covered.add(target)
    covered_packages = {model_types[target] for target in covered}
    return {f"{package}/" for package in packages - covered_packages} | {
        f"{package}:{model_type}"
        for (family, model_type), package in model_types.items()
        if (family, model_type) not in covered
    }


def load_legacy_gaps(path: Path = BASELINE) -> set[str]:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "frozen_at", "missing_recipes"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["frozen_at"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", data["frozen_at"])
    ):
        raise ValueError(f"{path}: invalid frozen recipe-gap inventory")
    gaps = data["missing_recipes"]
    if (
        not isinstance(gaps, list)
        or any(
            not isinstance(gap, str)
            or not re.fullmatch(r"xh_(?:llm|other_model)/[a-zA-Z0-9_]+(?:/|:[a-zA-Z0-9_]+)", gap)
            for gap in gaps
        )
        or len(set(gaps)) != len(gaps)
    ):
        raise ValueError(f"{path}: gaps must be unique exact package/type identities, never globs")
    return set(gaps)


def check_recipe_coverage(repo_root: Path, recipes: Iterable[ModelRecipe], legacy: set[str]) -> None:
    gaps = recipe_gaps(repo_root, recipes)
    missing, stale = gaps - legacy, legacy - gaps
    errors = []
    if missing:
        errors.append(
            "New Merak models require a registered ModelRecipe:\n  "
            + "\n  ".join(sorted(missing))
            + "\nAdd model_integrations/<family>/plugin.py registration and a real workflow. "
            "Do not extend recipe_legacy_gaps.json. See model_integrations/README.md."
        )
    if stale:
        errors.append("Remove resolved/deleted legacy recipe gaps:\n  " + "\n  ".join(sorted(stale)))
    if errors:
        raise ValueError("\n".join(errors))
