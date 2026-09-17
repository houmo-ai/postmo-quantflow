#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OFFICIAL="${MMEB_OFFICIAL:-$ROOT/work_dirs/wemm_embedding/mmeb_v3_official/mmeb_v3_eval}"
DATA="${MMEB_V3_DATA_DIR:-$ROOT/work_dirs/wemm_embedding/mmeb_v3_data}"
OUT="${MMEB_OUT:-$ROOT/work_dirs/wemm_embedding/mmeb_v3_runs}"
EXPORT_ROOT="${WEMM_EXPORT_ROOT:-$ROOT/work_dirs/wemm_embedding}"
MODEL_ROOT="${WEMM_MODEL_ROOT:-/data01/datasets}"
PY="${PY:-python}"
PY_BIN="$(command -v "$PY")"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-65536}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1572864}"

export PYTHONPATH="$ROOT:$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$PY_BIN"):$PATH"
export ENABLE_HMINFERENCE_V2="${ENABLE_HMINFERENCE_V2:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MMEB_V3_DATA_DIR="$DATA"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MMEB_DEVICE="${MMEB_DEVICE:-cuda:0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

for size in ${WEMM_SIZES:-2b 4b 9b}; do
  upper_size="${size^^}"
  model_dir="$MODEL_ROOT/WeMM-Embedding-$upper_size"
  export_dir="$EXPORT_ROOT/$size/wemm_embedding_export"
  for task in ${MMEB_TASKS:-Visual7W ChartQA}; do
    (
    flock -x 9
    comparison="$OUT/${size}_${task}_comparison.json"
    if [[ "${MMEB_FORCE:-0}" != 1 && -s "$comparison" ]]; then
      echo "MMEB_V3_REUSE size=$size task=$task"
      exit 0
    fi
    if [[ "$task" == Visual7W ]]; then
      task_config="$ROOT/examples_merak/llm/wemm_embedding/mmeb_v3_image_first.yaml"
    else
      task_config="$ROOT/examples_merak/llm/wemm_embedding/mmeb_v3_image_chartqa.yaml"
    fi
    for backend in hf hmonnx; do
      run_dir="$OUT/${size}_${backend}/${task}"
      mkdir -p "$run_dir"
      if [[ "$backend" == hf && "${MMEB_REUSE_HF:-0}" == 1 \
        && -s "$run_dir/${task}_score.json" && -s "$run_dir/${task}_qry" \
        && -s "$run_dir/${task}_tgt" && -s "$run_dir/protocol.json" ]]; then
        echo "MMEB_V3_REUSE_HF size=$size task=$task"
        continue
      fi
      args=(
        --backend "$backend"
        --protocol-file "$run_dir/protocol.json"
        --image-min-pixels "$IMAGE_MIN_PIXELS"
        --image-max-pixels "$IMAGE_MAX_PIXELS"
        --model_name "$model_dir"
        --model_backbone wemm_embedding
        --pooling last --normalize true
        --per_device_eval_batch_size 1 --dataloader_num_workers 0
        --dataset_config "$task_config"
        --encode_output_path "$run_dir"
        --data_basedir "$DATA"
      )
      if [[ "$backend" == hmonnx ]]; then
        args+=(--hmonnx-meta "$export_dir")
      fi
      echo "MMEB_V3_START size=$size task=$task backend=$backend"
      (
        cd "$OFFICIAL"
        "$PY" -u "$ROOT/examples_merak/llm/wemm_embedding/mmeb_v3_eval.py" "${args[@]}"
      )
      echo "MMEB_V3_DONE size=$size task=$task backend=$backend"
    done
    "$PY" -u "$ROOT/examples_merak/llm/wemm_embedding/compare_mmeb_v3_runs.py" \
      "$OUT/${size}_hf/${task}" "$OUT/${size}_hmonnx/${task}" "$task" \
      > "$comparison.tmp"
    mv "$comparison.tmp" "$comparison"
    ) 9> "$OUT/.${size}_${task}.lock"
  done
done

echo "MMEB_V3_SUITE_OK: $OUT"
