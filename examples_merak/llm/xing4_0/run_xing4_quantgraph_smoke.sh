#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Make the checkout importable while keeping the caller's environment.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Use the caller's environment and GPU selection.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XHQUANT_MOE_LOW_M_TRITON=0
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

MODEL_DIR="${MODEL_DIR:-work_dirs/xing4_0_29b_a4b_quant}"
CONFIG_PATH="${CONFIG_PATH:-configs_merak/workflows/xh2a/llm_models/xing4_0/29b_a4b/xing4_0_29b_a4b_xh2a_w4a8.yaml}"
EVAL_TYPE="${EVAL_TYPE:-quanted_fast}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PROMPT="${PROMPT:-请用中文简单介绍 Xing4.0，并给出 1+1 的结果。}"

LOG_DIR="${REPO_ROOT}/hm_eval/outputs/xing4_quantgraph_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "[xing4-quantgraph] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee "${LOG_DIR}/run.log"
echo "[xing4-quantgraph] PYTHONPATH=${PYTHONPATH}" | tee -a "${LOG_DIR}/run.log"
echo "[xing4-quantgraph] model_dir=${MODEL_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "[xing4-quantgraph] eval_type=${EVAL_TYPE} max_new_tokens=${MAX_NEW_TOKENS}" | tee -a "${LOG_DIR}/run.log"
echo "[xing4-quantgraph] log_dir=${LOG_DIR}" | tee -a "${LOG_DIR}/run.log"

python examples_merak/llm/xing4_0/xing4_0_xh_generate_workflow.py \
  --model-dir "${MODEL_DIR}" \
  --config-path "${CONFIG_PATH}" \
  --eval-type "${EVAL_TYPE}" \
  --think \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt "${PROMPT}" \
  2>&1 | tee -a "${LOG_DIR}/run.log"

echo "[xing4-quantgraph] done; log=${LOG_DIR}/run.log"
