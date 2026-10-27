"""Lazy Transformers Auto registration for native and local MiniCPM-V-4.7 code."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def get_minicpmv47_modeling_module() -> ModuleType:
    """Prefer the native Transformers implementation, then import local code.

    Only fall back when the MiniCPM module itself is absent. Missing dependencies
    inside an installed implementation must retain their original traceback.
    """

    native_module = "transformers.models.minicpmv4_7.modeling_minicpmv4_7"
    try:
        return import_module(native_module)
    except ModuleNotFoundError as exc:
        if exc.name not in {"transformers.models.minicpmv4_7", native_module}:
            raise

    try:
        return import_module("modeling_minicpmv4_7")
    except ModuleNotFoundError as exc:
        if exc.name != "modeling_minicpmv4_7":
            raise
        raise ImportError(
            "MiniCPM-V-4.7 model classes are unavailable. Use a Transformers build "
            "with native minicpmv4_7 support, or make configuration_minicpmv4_7.py "
            "and modeling_minicpmv4_7.py importable on PYTHONPATH. "
            "Auto registration alone cannot supply the model implementation."
        ) from exc


def register_minicpmv47_auto_classes() -> None:
    """Register the matching config/model pair before a family loader runs."""

    from transformers import AutoConfig, AutoModelForImageTextToText

    modeling = get_minicpmv47_modeling_module()
    model_class = modeling.MiniCPMV4_7ForConditionalGeneration
    config_class = model_class.config_class
    AutoConfig.register(config_class.model_type, config_class, exist_ok=True)
    AutoModelForImageTextToText.register(config_class, model_class, exist_ok=True)
    # Some 5.18 builds ignore register() for all transformers.* config classes,
    # including a native model whose built-in Auto mapping was omitted. Fill
    # only that missing entry; preserve an existing native mapping.
    mapping = AutoModelForImageTextToText._model_mapping
    if config_class not in mapping:
        mapping._extra_content[config_class] = model_class


__all__ = ["register_minicpmv47_auto_classes"]
