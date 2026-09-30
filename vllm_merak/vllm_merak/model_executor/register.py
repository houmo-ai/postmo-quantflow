import importlib
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
    """Return the registered Merak model class for a HF architecture name."""
    import_model_modules()
    return MERAK_MODEL_REGISTRY.get(source_architecture)


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
