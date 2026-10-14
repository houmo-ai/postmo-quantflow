#!/usr/bin/env bash
set -euo pipefail

model_dir=${1:?usage: deepseek_v4_dspark_service.sh MODEL_DIR}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_PLUGINS=merak
export VLLM_MERAK_USE_V2=1
export ENABLE_HMINFERENCE_V2=1
export VLLM_MERAK_AUTO_OFFLOAD=1
export VLLM_MERAK_CUDA_GRAPH=1
export XHQUANT_PACKED_W4=1
export XHQUANT_FORBID_CUDA_W4_UNPACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

exec python -m vllm.entrypoints.openai.api_server \
  --model "$model_dir" --tokenizer "$model_dir/hf_config" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18185}" \
  --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}" \
  --tokenizer-mode deepseek_v4 --trust-remote-code --dtype float16 \
  --max-model-len 262144 --max-num-batched-tokens 256 --max-num-seqs 1 \
  --config-format merak_llm --load-format merak_llm --generation-config vllm \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"max"}' \
  --speculative-config '{"method":"custom_class","model":"vllm_merak.spec_decode.dspark.DeepSeekV4DSparkProposer","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
