#!/usr/bin/env bash
# Fail-closed acceptance gate for one generative HMONNX release.

set -euo pipefail

MODEL_DIR="${1:?usage: run_release_acceptance.sh MODEL_DIR [MAX_MODEL_LEN]}"
MAX_MODEL_LEN="${2:-512}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export VLLM_PLUGINS="${VLLM_PLUGINS:-merak}"
export ENABLE_HMINFERENCE_V2="${ENABLE_HMINFERENCE_V2:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

"$PYTHON_BIN" "$SCRIPT_DIR/check_runtime_environment.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_hmonnx_attention_abi.py" --json "$MODEL_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/run_hmonnx_release_smoke.py" \
    --model "$MODEL_DIR" \
    --max-model-len "$MAX_MODEL_LEN"
