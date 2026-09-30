#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

model_id=${1:?usage: vllm_server_start.sh MODEL_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
PORT=${PORT:-18006}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
AUDIT_LOG_MODE=${AUDIT_LOG_MODE:-off}
AUDIT_LOG_DIR=${AUDIT_LOG_DIR:-logs/audit}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3_5_9b_merak}

if [[ "$MAX_NUM_SEQS" != "1" ]]; then
    echo "error: MAX_NUM_SEQS must be exactly 1 for the Phase 1 sparse Linear state store (got '$MAX_NUM_SEQS')" >&2
    exit 2
fi

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

echo "Merak Phase 1 launch config: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS VIRTUAL_KV=$VIRTUAL_KV PORT=$PORT SERVED_MODEL_NAME=$SERVED_MODEL_NAME AUDIT_LOG_MODE=$AUDIT_LOG_MODE AUDIT_LOG_DIR=$AUDIT_LOG_DIR" >&2

cd "$PLUGIN_ROOT"
exec python "$SCRIPT_DIR/vllm_chat_server.py" \
--model "$model_id" \
--host 127.0.0.1 \
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
