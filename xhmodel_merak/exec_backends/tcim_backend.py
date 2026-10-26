"""TCIM interface declarations; importing this module does not load the runtime."""

NAME = "tcim"
ALIASES = ("tcim_runtime", "tcim_lite")

_IMPLEMENTATIONS = {
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

CLASSES = {
    name: ("xhmodel_merak.exec_backends.tcim_hmonnx", implementation)
    for name, implementation in _IMPLEMENTATIONS.items()
}

SUPPORTS_CUDA_GRAPH = False
SUPPORTS_AUTO_OFFLOAD = False


def use_inference_v2():
    return True


def prepare_graphs(
    prefill,
    decode,
    *,
    prepare,
    enable_parallel_linear_fusion,
    enable_page_attention,
    convert_unfused_sliding_kv_cache=False,
):
    if enable_parallel_linear_fusion or enable_page_attention:
        raise NotImplementedError(
            "TCIM requires HMMs compiled with the requested graph ABI; runtime HMONNX graph conversion is unsupported"
        )
    return str(prefill), str(decode)


def create_session(backend, path, **options):
    from .backend import create_v2_session

    return create_v2_session(backend, path, **options)
