"""HMONNX interface declarations; SDK imports are deferred until resolution."""

NAME = "hmonnx"
ALIASES = ()

CLASSES = {
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

SUPPORTS_CUDA_GRAPH = True
SUPPORTS_AUTO_OFFLOAD = True
ENV_ENABLE_INFERENCE_V2 = "ENABLE_HMINFERENCE_V2"


def use_inference_v2():
    import os

    return os.getenv(ENV_ENABLE_INFERENCE_V2, "").strip().lower() in {"1", "true", "yes", "on"}


def prepare_graphs(prefill, decode, *, prepare, **options):
    return prepare(prefill, decode, **options)


def create_session(
    backend,
    path,
    *,
    onnx_graph=None,
    enable_cuda_graph=False,
    enable_auto_offload=False,
    enable_golden=False,
    device_map=None,
    layer_infos=None,
):
    from .backend import create_v2_session

    if backend.use_inference_v2:
        return create_v2_session(
            backend,
            path,
            onnx_graph=onnx_graph,
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=device_map,
            layer_infos=layer_infos,
        )

    import warnings
    from pathlib import Path

    import torch

    warnings.warn(
        "Legacy HMONNX runtime fallback is deprecated and may be removed in a future release. "
        f"Set {ENV_ENABLE_INFERENCE_V2}=1 to route compatible single-device models through HMONNXInferenceV2.",
        FutureWarning,
        stacklevel=2,
    )
    if enable_cuda_graph:
        session = backend.resolve_class("HMONNXCUDAGraphInference")(path)
    else:
        session = backend.resolve_class("HMONNXGoldenInference")(path)
        session.golden_dir = str(Path(path).parent)
        session.legacy_mode = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    session.to(device)
    return session, device
