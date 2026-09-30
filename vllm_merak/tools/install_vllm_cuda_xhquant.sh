#!/usr/bin/env bash
# Build the CUDA-target vLLM plugin host in the dedicated service environment.

set -euo pipefail

VLLM_SOURCE="${VLLM_SOURCE:?set VLLM_SOURCE to the checked-out vLLM source tree}"
CONDA_ENV="${CONDA_ENV:-vllm}"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS_VALUE="${MAX_JOBS:-16}"
NVCC_THREADS_VALUE="${NVCC_THREADS:-2}"
USE_PRECOMPILED_VALUE="${VLLM_USE_PRECOMPILED:-1}"

if [[ ! -f "$VLLM_SOURCE/pyproject.toml" || ! -d "$VLLM_SOURCE/vllm" ]]; then
    echo "VLLM_SOURCE is not a vLLM source tree: $VLLM_SOURCE" >&2
    exit 2
fi
BUILD_ENV=(
    "CUDA_HOME=$CUDA_HOME_VALUE"
    "VLLM_TARGET_DEVICE=cuda"
)

if [[ "$USE_PRECOMPILED_VALUE" =~ ^(1|true|TRUE)$ ]]; then
    SOURCE_COMMIT="$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
    CUDA_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-$(
        conda run -n "$CONDA_ENV" python -c \
            'import torch; print("cu" + torch.version.cuda.replace(".", "")[:3])'
    )}"
    BUILD_ENV+=(
        "VLLM_USE_PRECOMPILED=1"
        "VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-$SOURCE_COMMIT}"
        "VLLM_PRECOMPILED_WHEEL_VARIANT=$CUDA_VARIANT"
    )
else
    if [[ ! -x "$CUDA_HOME_VALUE/bin/nvcc" ]]; then
        echo "CUDA compiler not found: $CUDA_HOME_VALUE/bin/nvcc" >&2
        exit 2
    fi
    CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(
        conda run -n "$CONDA_ENV" python -c \
            'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
    )}"
    BUILD_ENV+=(
        "MAX_JOBS=$MAX_JOBS_VALUE"
        "NVCC_THREADS=$NVCC_THREADS_VALUE"
        "TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST"
        "VLLM_USE_PRECOMPILED=0"
    )
fi

cd "$VLLM_SOURCE"
conda run -n "$CONDA_ENV" env \
    "${BUILD_ENV[@]}" \
    uv pip install -v -e . --no-build-isolation --no-deps

conda run -n "$CONDA_ENV" python -c '
import importlib
import torch

importlib.import_module("vllm._C_stable_libtorch")
if not hasattr(torch.ops._C, "apply_repetition_penalties_"):
    raise RuntimeError("CUDA vLLM build did not register apply_repetition_penalties_")
logits = torch.tensor([[2.0, -3.0, 4.0]], device="cuda")
prompt_mask = torch.tensor([[True, False, False]], device="cuda")
output_mask = torch.tensor([[False, True, False]], device="cuda")
penalties = torch.tensor([2.0], device="cuda")
torch.ops._C.apply_repetition_penalties_(
    logits, prompt_mask, output_mask, penalties
)
torch.cuda.synchronize()
torch.testing.assert_close(
    logits.cpu(), torch.tensor([[1.0, -6.0, 4.0]]), rtol=0.0, atol=0.0
)
print("CUDA-target vLLM sampler ABI: PASS")
'
