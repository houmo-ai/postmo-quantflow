from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .backend import ExecutionBackend
from .builtin_backends import BACKEND_ALIASES, BACKEND_POLICIES, BACKENDS


ENV_INFERENCE_BACKEND = "XHMODEL_INFERENCE_BACKEND"


def get_inference_backend() -> str:
    """Select a backend; configure the environment before creating sessions/caches.

    Keep this setting unchanged for the lifetime of those objects.
    """
    value = os.getenv(ENV_INFERENCE_BACKEND, "hmonnx").strip().lower() or "hmonnx"
    backend = BACKEND_ALIASES.get(value, value)
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported {ENV_INFERENCE_BACKEND}={value!r}; expected one of {', '.join(sorted(BACKENDS))}"
        )
    return backend


def get_backend() -> ExecutionBackend:
    """Capture one backend and runtime mode for a model's entire lifetime."""
    name = get_inference_backend()
    try:
        policy = BACKEND_POLICIES[name]
    except KeyError:
        raise ValueError(f"Backend {name!r} does not provide a runtime policy") from None
    return ExecutionBackend(name, BACKENDS[name], policy, policy.use_inference_v2())


def resolve_backend_class(name: str):
    backend = get_inference_backend()
    try:
        module_name, class_name = BACKENDS[backend][name]
    except KeyError:
        raise ValueError(f"Backend {backend!r} does not provide interface {name!r}") from None
    return getattr(import_module(module_name), class_name)


class _BackendFacadeMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, resolve_backend_class(cls._backend_class_name))

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, resolve_backend_class(cls._backend_class_name))


class _BackendFacade(metaclass=_BackendFacadeMeta):
    _backend_class_name = ""

    def __new__(cls, *args: Any, **kwargs: Any):
        implementation = resolve_backend_class(cls._backend_class_name)
        return implementation(*args, **kwargs)


class HMONNXInference(_BackendFacade):
    _backend_class_name = "HMONNXInference"


class HMONNXGoldenInference(_BackendFacade):
    _backend_class_name = "HMONNXGoldenInference"


class HMONNXGraphGoldenInference(_BackendFacade):
    _backend_class_name = "HMONNXGraphGoldenInference"


HMONNXGrapInference = HMONNXGraphGoldenInference


class HMONNXCUDAGraphInference(_BackendFacade):
    _backend_class_name = "HMONNXCUDAGraphInference"


class HMONNXInferenceV2(_BackendFacade):
    _backend_class_name = "HMONNXInferenceV2"

    @classmethod
    def from_onnx_graph(cls, *args: Any, **kwargs: Any):
        implementation = resolve_backend_class(cls._backend_class_name)
        return implementation.from_onnx_graph(*args, **kwargs)


class HMONNXInferenceConfig(_BackendFacade):
    _backend_class_name = "HMONNXInferenceConfig"


class LLMHMONNXLoader(_BackendFacade):
    _backend_class_name = "LLMHMONNXLoader"


class MultiHMONNXLoader(_BackendFacade):
    _backend_class_name = "MultiHMONNXLoader"


class CacheTensor(_BackendFacade):
    _backend_class_name = "CacheTensor"


class HybridCacheTensor(_BackendFacade):
    _backend_class_name = "HybridCacheTensor"


__all__ = [
    "CacheTensor",
    "ENV_INFERENCE_BACKEND",
    "HMONNXCUDAGraphInference",
    "HMONNXGoldenInference",
    "HMONNXGrapInference",
    "HMONNXGraphGoldenInference",
    "HMONNXInference",
    "HMONNXInferenceConfig",
    "HMONNXInferenceV2",
    "HybridCacheTensor",
    "LLMHMONNXLoader",
    "MultiHMONNXLoader",
    "get_backend",
    "get_inference_backend",
    "resolve_backend_class",
]
