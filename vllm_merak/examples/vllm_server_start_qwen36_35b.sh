#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

MODEL_DIR=${MODEL_DIR:-/data01/home/yujy/work/v4/xh2modelzoo/work_dirs/qwen3_6_35b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260710}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PORT=${PORT:-18036}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3_6_35b_a3b_merak}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
AUDIT_LOG_MODE=${AUDIT_LOG_MODE:-off}
AUDIT_LOG_DIR=${AUDIT_LOG_DIR:-logs/audit}

if [[ "$MAX_NUM_SEQS" != "1" ]]; then
    echo "error: MAX_NUM_SEQS must be exactly 1; fixed-address CUDA Graph metadata staging does not support concurrent forward (got '$MAX_NUM_SEQS')" >&2
    exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "error: MODEL_DIR is not a directory: $MODEL_DIR" >&2
    exit 2
fi

# if [[ ! -f "$MODEL_DIR/merak_config.json" ]]; then
#     echo "error: missing $MODEL_DIR/merak_config.json (copy data/merak_config.json first)" >&2
#     exit 2
# fi

invalid_device_list() {
    echo "error: CUDA_VISIBLE_DEVICES must contain exactly 1, 2, or 3 unique non-negative integers; got '$CUDA_VISIBLE_DEVICES'" >&2
    exit 2
}

if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+){0,2}$ ]]; then
    invalid_device_list
fi
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
device_count=${#visible_devices[@]}
declare -A seen_devices=()
for device in "${visible_devices[@]}"; do
  if [[ ! "$device" =~ ^(0|[1-9][0-9]*)$ ]]; then
    invalid_device_list
  fi
  normalized_device=$((10#$device))
  if [[ -n "${seen_devices[$normalized_device]+present}" ]]; then
    invalid_device_list
  fi
  seen_devices[$normalized_device]=1
done

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ -n "${CPU_KV+x}" && -n "${VLLM_CPU_KVCACHE_SPACE+x}" && "$CPU_KV" != "$VLLM_CPU_KVCACHE_SPACE" ]]; then
  echo "error: CPU_KV ($CPU_KV) conflicts with VLLM_CPU_KVCACHE_SPACE ($VLLM_CPU_KVCACHE_SPACE); set only one override or give both the same GiB value" >&2
  exit 2
fi
if [[ -n "${CPU_KV+x}" ]]; then
  export VLLM_CPU_KVCACHE_SPACE=$CPU_KV
  VIRTUAL_KV="VLLM_CPU_KVCACHE_SPACE=${CPU_KV}GiB-explicit-virtual-override-via-CPU_KV"
elif [[ -n "${VLLM_CPU_KVCACHE_SPACE+x}" ]]; then
  export VLLM_CPU_KVCACHE_SPACE
  VIRTUAL_KV="VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}GiB-explicit-virtual-override"
else
  VIRTUAL_KV="planner-default"
fi

echo "Merak launch config: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES card_count=$device_count MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS VIRTUAL_KV=$VIRTUAL_KV PORT=$PORT AUDIT_LOG_MODE=$AUDIT_LOG_MODE AUDIT_LOG_DIR=$AUDIT_LOG_DIR" >&2

# HMONNX auto-offload owns placement across the visible cards. Keep one vLLM
# worker and do not add --tensor-parallel-size.
exec python examples/vllm_chat_server.py \
  --model "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --config-format merak_llm \
  --load-format merak_llm \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --use-v2 \
  --auto-offload \
  --cuda-graph \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --generation-config vllm \
  --audit-log-mode "$AUDIT_LOG_MODE" \
  --audit-log-dir "$AUDIT_LOG_DIR"
