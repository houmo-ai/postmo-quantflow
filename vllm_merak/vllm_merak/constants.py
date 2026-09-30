MERAK_AUTO_OFFLOAD = "VLLM_MERAK_AUTO_OFFLOAD"
MERAK_USE_V2 = "VLLM_MERAK_USE_V2"
MERAK_CUDA_GRAPH = "VLLM_MERAK_CUDA_GRAPH"
MERAK_SPEC_DRAFT_CUDA_GRAPH = "VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH"
MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH = "VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH"
MERAK_KV_CACHE_FORMAT = "VLLM_MERAK_KV_CACHE_FORMAT"
MERAK_ENABLE_PAGE_ATTENTION = "VLLM_MERAK_ENABLE_PAGE_ATTENTION"
MERAK_PARALLEL_LINEAR_FUSION = "VLLM_MERAK_PARALLEL_LINEAR_FUSION"

# Full-attention state ownership for models that also carry recurrent/conv
# state.  Paged models project request state through vLLM's block manager;
# contiguous-single-request models retain their asymmetric cache inside the
# HMONNX runtime and therefore forbid concurrent sequences and prefix reuse.
HYBRID_CACHE_POLICY_PAGED = "paged"
HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST = "contiguous_single_request"
SUPPORTED_HYBRID_CACHE_POLICIES = frozenset(
    {
        HYBRID_CACHE_POLICY_PAGED,
        HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    }
)

# Packed-W4 narrow-token execution contract.  A normal autoregressive decode
# has one target row; speculative verification has K+1 rows and is bounded by
# the native M<=16 CUDA/Triton kernels.
MERAK_DECODE_QUERY_LENGTH = 1
MERAK_MIN_VERIFY_QUERY_LENGTH = 2
MERAK_MAX_NARROW_QUERY_LENGTH = 16
