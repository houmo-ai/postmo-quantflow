from __future__ import annotations

import os
from importlib import import_module
from typing import Any


ENV_INFERENCE_BACKEND = "XHMODEL_INFERENCE_BACKEND"


def get_inference_backend() -> str:
    value = os.getenv(ENV_INFERENCE_BACKEND, "hmonnx").strip().lower()
    if value in {"", "hmonnx"}:
        return "hmonnx"
    if value in {"tcim", "tcim_runtime", "tcim_lite"}:
        return "tcim"
    raise ValueError(f"Unsupported {ENV_INFERENCE_BACKEND}={value!r}; expected hmonnx or tcim")


_HMONNX_CLASSES = {
    "HMONNXInference": ("xhquant.api", "HMONNXInference"),
    "HMONNXGoldenInference": ("xhquant.api", "HMONNXGoldenInference"),
    "HMONNXGraphGoldenInference": ("xhquant.api", "HMONNXGraphGoldenInference"),
    "HMONNXCUDAGraphInference": (
        "xhquant.xhonnxruntime.hmonnx_graph_inference",
        "HMONNXCUDAGraphInference",
    ),
    "HMONNXInferenceV2": (
        "xhquant.xhonnxruntime.hmonnx_inference_v2",
        "HMONNXInferenceV2",
    ),
    "HMONNXInferenceConfig": (
        "xhquant.xhonnxruntime.hmonnx_inference_v2",
        "HMONNXInferenceConfig",
    ),
    "LLMHMONNXLoader": (
        "xhquant.xhonnxruntime.llm_hmonnx_loader",
        "LLMHMONNXLoader",
    ),
    "MultiHMONNXLoader": (
        "xhquant.xhonnxruntime.llm_hmonnx_loader",
        "MultiHMONNXLoader",
    ),
    "CacheTensor": ("xhquant.core", "CacheTensor"),
    "HybridCacheTensor": ("xhquant.core", "HybridCacheTensor"),
}

_TCIM_CLASSES = {
    "HMONNXInference": "TcimHMONNXInference",
    "HMONNXGoldenInference": "TcimHMONNXGoldenInference",
    "HMONNXGraphGoldenInference": "TcimHMONNXGraphGoldenInference",
    "HMONNXCUDAGraphInference": "TcimHMONNXCUDAGraphInference",
    "HMONNXInferenceV2": "TcimHMONNXInferenceV2",
    "HMONNXInferenceConfig": "TcimHMONNXInferenceConfig",
    "LLMHMONNXLoader": "TcimLLMHMONNXLoader",
    "MultiHMONNXLoader": "TcimMultiHMONNXLoader",
    "CacheTensor": "TcimCacheTensor",
    "HybridCacheTensor": "TcimHybridCacheTensor",
}


def resolve_backend_class(name: str):
    backend = get_inference_backend()
    if backend == "tcim":
        module = import_module(".tcim_hmonnx", package=__package__)
        return getattr(module, _TCIM_CLASSES[name])
    module_name, class_name = _HMONNX_CLASSES[name]
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
    "get_inference_backend",
    "resolve_backend_class",
]
