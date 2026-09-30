#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${WEMM_REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
OFFICIAL=${MMEB_OFFICIAL:-$REPO/work_dirs/wemm_embedding/mmeb_v3_official/mmeb_v3_eval}
DATA=${MMEB_V3_DATA_DIR:-$REPO/work_dirs/wemm_embedding/mmeb_v3_data}
RUN_ROOT=${MMEB_OUT:-$REPO/work_dirs/wemm_embedding/mmeb_v3_runs}
PYTHON_BIN=${WEMM_PYTHON:-python}
VLLM_MERAK=${VLLM_MERAK_ROOT:-$REPO/vllm_merak}
RUNNER=$SCRIPT_DIR/run_mmeb_v3_vllm_merak_full.py
AGGREGATE=$SCRIPT_DIR/aggregate_mmeb_v3.py
COMPARE=$SCRIPT_DIR/compare_mmeb_v3_aggregates.py
GPUS=${WEMM_GPUS:-0,1,2,3}
MODEL_ROOT=${WEMM_MODEL_ROOT:-/data01/datasets}
EXPORT_ROOT=${WEMM_EXPORT_ROOT:-$REPO/work_dirs/wemm_embedding}

mkdir -p "$RUN_ROOT"

exec 9>"$RUN_ROOT/wemm_256k_queue.lock"
if ! flock -n 9; then
    echo "Another WEMM MMEB-v3 256K queue already owns the lock." >&2
    exit 2
fi

run_suite() {
    local backend=$1 model=$2 source_model=$3 output=$4
    local max_num_seqs=1
    local attempt
    if [[ "$backend" == "vllm" ]]; then
        max_num_seqs=16
    fi
    for attempt in 1 2 3; do
        echo "[$(date --iso-8601=seconds)] backend=$backend attempt=$attempt output=$output"
        "$PYTHON_BIN" -u "$RUNNER" \
            --repo "$REPO" \
            --backend "$backend" \
            --vllm-merak "$VLLM_MERAK" \
            --official "$OFFICIAL" \
            --data "$DATA" \
            --model "$model" \
            --source-model "$source_model" \
            --output "$output" \
            --python "$PYTHON_BIN" \
            --gpus "$GPUS" \
            --batch-size 8 \
            --dataloader-workers 4 \
            --datasets-per-engine 8 \
            --vllm-max-num-seqs "$max_num_seqs" \
            --gpu-memory-utilization 0.60 \
            --retries 1 && return 0
        echo "[$(date --iso-8601=seconds)] retrying incomplete suite: $output"
    done
    echo "[$(date --iso-8601=seconds)] suite still incomplete after three passes: $output" >&2
    return 1
}

run_model() {
    local size=$1 float_model=$2 quant_model=$3
    local float_run=$RUN_ROOT/wemm_${size}_float_256k_vllm_full
    local quant_run=$RUN_ROOT/wemm_${size}_w4a8_256k_vllm_merak_full
    local report=$RUN_ROOT/wemm_${size}_256k_float_vs_w4a8

    run_suite vllm "$float_model" "$float_model" "$float_run" || return 1
    "$PYTHON_BIN" "$AGGREGATE" "$float_run" \
        --title "WEMM-Embedding-${size^^} Float vLLM 256K MMEB-v3"

    run_suite vllm_merak "$quant_model" "$float_model" "$quant_run" || return 1
    "$PYTHON_BIN" "$AGGREGATE" "$quant_run" \
        --title "WEMM-Embedding-${size^^} W4A8 vLLM Merak 256K MMEB-v3"

    "$PYTHON_BIN" "$COMPARE" "$float_run" "$quant_run" \
        --title "WEMM-Embedding-${size^^} 256K MMEB-v3: Float vs W4A8" \
        --output "$report"
}

resolve_quant_model() {
    local size=$1 override_name override_value export_dir
    local -a matches
    override_name=WEMM_${size^^}_QUANT_MODEL
    override_value=${!override_name:-}
    if [[ -n "$override_value" ]]; then
        printf '%s\n' "$override_value"
        return
    fi

    export_dir=$EXPORT_ROOT/${size}_256k_context/wemm_embedding_export
    mapfile -t matches < <(
        find "$export_dir" -mindepth 1 -maxdepth 1 -type d \
            -name "hmquant_xh2_wemm_embedding_${size}_w4a8_256_256k_*" \
            -print 2>/dev/null | sort
    )
    if (( ${#matches[@]} != 1 )); then
        echo "Expected one ${size} quantized export under $export_dir; " \
             "set $override_name explicitly." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

for size in 2b 4b 9b; do
    quant_model=$(resolve_quant_model "$size") || exit 1
    run_model "$size" \
        "$MODEL_ROOT/WeMM-Embedding-${size^^}" \
        "$quant_model" || exit 1
done

echo "[$(date --iso-8601=seconds)] all WEMM MMEB-v3 256K suites completed"
