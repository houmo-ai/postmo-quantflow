#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

MODEL_DIR=${MODEL_DIR:-./data/models/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260723}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "error: MODEL_DIR is not a directory: $MODEL_DIR" >&2
    exit 2
fi


cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES

echo "Merak chat launch: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MODEL_DIR=$MODEL_DIR" >&2

# HMONNX auto-offload owns placement across the visible cards. Keep one vLLM
# worker and do not add --tensor-parallel-size.
exec python examples/vllm_merak_chat.py \
--model "$MODEL_DIR" \
--use-v2 \
--auto-offload \
--cuda-graph
