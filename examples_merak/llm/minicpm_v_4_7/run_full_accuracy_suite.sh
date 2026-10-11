#!/usr/bin/env bash
# Full MiniCPM-V-4.7 Dense 1B Torch vs HMONNX accuracy suite.
# Protocol: thinking-off, downsample 4x, max_slice_nums=9, greedy, ChartQA short-answer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-/data01/home/xuchen/miniconda3/envs/xhquant/bin/python}"
DEVICE="${DEVICE:-cuda:0}"  # after CUDA_VISIBLE_DEVICES remap
OUT_ROOT="${OUT_ROOT:-$ROOT/work_dirs/minicpm_v_4_7_runs/accuracy_bench}"
EXPORT_META="${EXPORT_META:-$ROOT/work_dirs/minicpm_v_4_7_1b_export/export_meta_info.json}"
MODEL_DIR="${MODEL_DIR:-/data01/datasets/Minicpm-v4.7/weights/1b}"
DATASETS=(ocrbench chartqa ai2d)

mkdir -p "$OUT_ROOT"

run_one() {
  local backend="$1"
  local out_dir="$OUT_ROOT/full_${backend}"
  local log_file="$OUT_ROOT/full_${backend}.log"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START backend=$backend out=$out_dir device=$DEVICE" | tee -a "$log_file"
  "$PY" examples_merak/llm/minicpm_v_4_7/run_accuracy_bench.py \
    --backend "$backend" \
    --datasets "${DATASETS[@]}" \
    --model-dir "$MODEL_DIR" \
    --export-meta "$EXPORT_META" \
    --device "$DEVICE" \
    --downsample-mode 4x \
    --max-slice-nums 9 \
    --max-new-tokens 64 \
    --resume \
    --output-dir "$out_dir" \
    2>&1 | tee -a "$log_file"
  echo "[$(date '+%F %T')] DONE backend=$backend" | tee -a "$log_file"
}

run_one torch
run_one hmonnx

"$PY" examples_merak/llm/minicpm_v_4_7/summarize_accuracy_table.py \
  --torch-summary "$OUT_ROOT/full_torch/summary.json" \
  --hmonnx-summary "$OUT_ROOT/full_hmonnx/summary.json" \
  --output "$OUT_ROOT/comparison_table.json" \
  2>&1 | tee "$OUT_ROOT/comparison_table.md" | tee -a "$OUT_ROOT/summarize.log"

echo "[$(date '+%F %T')] Suite complete. Table: $OUT_ROOT/comparison_table.md"
