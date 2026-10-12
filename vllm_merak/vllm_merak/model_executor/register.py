import ast
import importlib
from functools import lru_cache
from pathlib import Path

from xhquant.utils.registry import Registry


MERAK_MODEL_REGISTRY = Registry("MerakModelRegistry")

_model_modules_imported = False


def _should_import_module_file(module_file: Path) -> bool:
    name = module_file.name
    stem = module_file.stem

    if name == "__init__.py":
        return False
    if stem.startswith("_"):
        return False
    if stem.endswith("_native"):
        return False
    if stem.startswith("test_") or stem.endswith("_test"):
        return False
    return True


@lru_cache(maxsize=1)
def _discover_model_paths() -> tuple[dict[str, str], dict[str, str]]:
    """Read literal registrations without importing optional model dependencies."""
    source_paths: dict[str, str] = {}
    custom_paths: dict[str, str] = {}
    model_loader_dir = Path(__file__).resolve().parent / "model_loader"
    for module_file in sorted(model_loader_dir.glob("*.py")):
        if not _should_import_module_file(module_file):
            continue
        tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "MERAK_MODEL_REGISTRY"
                    and decorator.func.attr == "register_module"
                ):
                    continue
                name_arg = next(
                    (kw.value for kw in decorator.keywords if kw.arg == "name"),
                    decorator.args[0] if decorator.args else None,
                )
                try:
                    names = (
                        ast.literal_eval(name_arg) if name_arg is not None else node.name
                    )
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{module_file}:{node.lineno}: model registration names must be literals"
                    ) from exc
                names = [names] if isinstance(names, str) else names
                if not isinstance(names, (list, tuple)) or not names or any(
                    not isinstance(name, str) or not name for name in names
                ):
                    raise ValueError(
                        f"{module_file}:{node.lineno}: invalid model registration names"
                    )
                path = f"{__package__}.model_loader.{module_file.stem}:{node.name}"
                for name in names:
                    if name in source_paths and source_paths[name] != path:
                        raise ValueError(f"Duplicate Merak source architecture: {name}")
                    source_paths[name] = path
                if node.name in custom_paths and custom_paths[node.name] != path:
                    raise ValueError(f"Duplicate Merak loader architecture: {node.name}")
                custom_paths[node.name] = path
    return source_paths, custom_paths


def get_model_registration_paths() -> dict[str, str | type]:
    """Return vLLM's lazy module:class registrations, including runtime additions."""
    paths: dict[str, str | type] = dict(_discover_model_paths()[1])
    for model_cls in MERAK_MODEL_REGISTRY.module_dict.values():
        paths[model_cls.__name__] = model_cls
    return paths


def import_model_modules() -> None:
    """Import model_loader Python modules so decorator registrations run."""
    global _model_modules_imported
    if _model_modules_imported:
        return

    model_loader_dir = Path(__file__).resolve().parent / "model_loader"
    module_prefix = f"{__package__}.model_loader"

    for module_file in sorted(model_loader_dir.glob("*.py")):
        if not _should_import_module_file(module_file):
            continue
        module_name = f"{module_prefix}.{module_file.stem}"
        importlib.import_module(module_name)

    _model_modules_imported = True


def get_registered_model_cls(source_architecture: str):
    """Load only the selected HF architecture or concrete Merak loader."""
    model_cls = MERAK_MODEL_REGISTRY.module_dict.get(source_architecture)
    if model_cls is None:
        model_cls = {
            cls.__name__: cls for cls in MERAK_MODEL_REGISTRY.module_dict.values()
        }.get(source_architecture)
    if model_cls is not None:
        return model_cls
    source_paths, custom_paths = _discover_model_paths()
    path = source_paths.get(source_architecture) or custom_paths.get(source_architecture)
    if path is None:
        return None
    module_name, class_name = path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def get_registered_custom_architectures() -> dict[str, type]:
    """Return vLLM registration names to Merak model classes.

    The registry stores source HF architecture names as keys. vLLM model
    registration should use the concrete Merak class name to avoid
    collisions with built-in vLLM architectures.
    """
    import_model_modules()

    custom_architectures: dict[str, type] = {}
    for model_cls in MERAK_MODEL_REGISTRY.module_dict.values():
        custom_architectures[model_cls.__name__] = model_cls
    return custom_architectures
