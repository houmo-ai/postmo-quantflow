#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PY="${PY:-python}"
PY_BIN="$(command -v "$PY")"
export PATH="$(dirname "$PY_BIN"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
MODEL_ROOT="${WEMM_MODEL_ROOT:-/data01/datasets}"
EXPORT_ROOT="${WEMM_EXPORT_ROOT:-work_dirs/wemm_embedding}"
DATA="${DATA:-work_dirs/wemm_embedding/datasets/flickr30k}"
OUT="${OUT:-work_dirs/wemm_embedding/eval_flickr30k_1000}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda:0}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-65536}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1572864}"
NINE_B_WORKERS="${NINE_B_WORKERS:-4}"
OTHER_WORKERS="${OTHER_WORKERS:-8}"
EVAL="$ROOT/examples_merak/llm/wemm_embedding/flickr/eval_flickr30k.py"
PARALLEL_EVAL="$ROOT/examples_merak/llm/wemm_embedding/flickr/eval_flickr30k_parallel.py"
COMPARE="$ROOT/examples_merak/llm/wemm_embedding/flickr/compare_full_cosine.py"

mkdir -p "$OUT"

run() {
  echo
  echo "======== $* ========"
  echo
  "$@"
}

for size in ${WEMM_SIZES:-2b 4b 9b}; do
  upper_size="${size^^}"
  model_dir="$MODEL_ROOT/WeMM-Embedding-$upper_size"
  export_dir="$EXPORT_ROOT/${size}_256k_context/wemm_embedding_export"
  native_dir="$OUT/${size}_native_dynamic"
  hmonnx_dir="$OUT/${size}_hmonnx_w4a8_dynamic"

  if [[ "${WEMM_REUSE_NATIVE:-0}" == 1 && -s "$OUT/${size}_native_dynamic_flickr1000.json" \
    && -s "$native_dir/text_emb.pt" && -s "$native_dir/image_emb.pt" ]]; then
    echo "Reusing completed native Flickr30K embeddings for $size"
  else
    run "$PY" -u "$EVAL" \
      --backend native \
      --model-dir "$model_dir" \
      --dataset-type flickr30k \
      --dataset-dir "$DATA" \
      --device "$DEVICE" \
      --image-min-pixels "$IMAGE_MIN_PIXELS" \
      --image-max-pixels "$IMAGE_MAX_PIXELS" \
      --text-batch-size 4 \
      --image-batch-size 1 \
      --max-images "$MAX_IMAGES" \
      --save-embeddings "$native_dir" \
      --report "$OUT/${size}_native_dynamic_flickr1000.json"
  fi

  if [[ "$size" == "9b" ]]; then
    workers="$NINE_B_WORKERS"
  else
    workers="$OTHER_WORKERS"
  fi
  if [[ "$workers" -gt 1 ]]; then
    run "$PY" -u "$PARALLEL_EVAL" \
      --export-dir "$export_dir" \
      --dataset-type flickr30k \
      --dataset-dir "$DATA" \
      --device "$DEVICE" \
      --workers "$workers" \
      --image-min-pixels "$IMAGE_MIN_PIXELS" \
      --image-max-pixels "$IMAGE_MAX_PIXELS" \
      --max-images "$MAX_IMAGES" \
      --output-dir "$hmonnx_dir" \
      --report "$OUT/${size}_hmonnx_w4a8_dynamic_flickr1000.json"
  else
    run "$PY" -u "$EVAL" \
      --backend hmonnx \
      --export-dir "$export_dir" \
      --dataset-type flickr30k \
      --dataset-dir "$DATA" \
      --device "$DEVICE" \
      --image-min-pixels "$IMAGE_MIN_PIXELS" \
      --image-max-pixels "$IMAGE_MAX_PIXELS" \
      --text-batch-size 1 \
      --image-batch-size 1 \
      --max-images "$MAX_IMAGES" \
      --save-embeddings "$hmonnx_dir" \
      --report "$OUT/${size}_hmonnx_w4a8_dynamic_flickr1000.json"
  fi

  run "$PY" -u "$COMPARE" \
    --native-dir "$native_dir" \
    --hmonnx-dir "$hmonnx_dir" \
    --report "$OUT/${size}_full_cosine_dynamic.json"
done

if [[ "${WEMM_SKIP_SUMMARY:-0}" != 1 ]]; then
  run "$PY" -u "$ROOT/examples_merak/llm/wemm_embedding/flickr/summarize_accuracy.py" \
    --report-dir "$OUT" --output "$OUT/summary.json"
fi

echo
echo "SUITE_OK: $OUT"
