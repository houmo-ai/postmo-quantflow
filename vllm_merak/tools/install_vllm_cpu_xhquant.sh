#!/usr/bin/env bash
# Install the vLLM CPU-target plugin host in the dedicated service environment.

set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE="${VLLM_SOURCE:?set VLLM_SOURCE to the checked-out vLLM source tree}"
CONDA_ENV="${CONDA_ENV:-vllm}"
ONEDNN_VERSION="${ONEDNN_VERSION:-v3.10}"
NVCC_PATH="${NVCC_PATH:-/usr/local/cuda/bin/nvcc}"
DEPS_DIR="${DEPS_DIR:-$PLUGIN_ROOT/.deps}"
SOURCE_CACHE_DIR="${SOURCE_CACHE_DIR:-$DEPS_DIR/source-cache}"
ONEDNN_SOURCE_DIR="${ONEDNN_SOURCE_DIR:-$DEPS_DIR/onednn-${ONEDNN_VERSION}-source}"
ONEDNN_ARCHIVE="$SOURCE_CACHE_DIR/onednn-${ONEDNN_VERSION}.tar.gz"
ONEDNN_URL="https://github.com/oneapi-src/oneDNN/archive/refs/tags/${ONEDNN_VERSION}.tar.gz"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
PROGRESS_TOTAL="${PROGRESS_TOTAL:-479}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-$DEPS_DIR/uv-cache}"
UV_LOCK_TIMEOUT_VALUE="${UV_LOCK_TIMEOUT:-1800}"
MAX_JOBS_VALUE="${MAX_JOBS:-16}"

usage() {
        cat <<EOF
Usage: $(basename "$0") [--clean-onednn]

Environment variables:
    CONDA_ENV            Conda environment name. Default: vllm
    ONEDNN_VERSION       oneDNN tag to use. Default: v3.10
    NVCC_PATH            CUDA compiler path for Torch CMake. Default: /usr/local/cuda/bin/nvcc
    VLLM_SOURCE          Required checked-out vLLM source tree to install
    DEPS_DIR             Build dependency cache dir. Default: <plugin>/.deps
    ONEDNN_SOURCE_DIR    Local oneDNN source dir. Default: <deps>/onednn-v3.10-source
    UV_LINK_MODE         uv link mode. Default: copy
    UV_CACHE_DIR         uv cache dir. Default: <deps>/uv-cache
    UV_LOCK_TIMEOUT      uv lock wait timeout in seconds. Default: 1800
    MAX_JOBS             Maximum compiler jobs. Default: 16
    PROGRESS_INTERVAL    Seconds between progress updates. Default: 10
    PROGRESS_TOTAL       Expected ninja build edges for progress bar. Default: 479

The script downloads oneDNN as a tarball and points CMake at the local source
tree, avoiding FetchContent git clone hangs at "Downloading oneDNN from GitHub".
EOF
}

CLEAN_ONEDNN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean-onednn)
            CLEAN_ONEDNN=1
            shift
        ;;
        -h|--help)
            usage
            exit 0
        ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
        ;;
    esac
done

require_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

format_elapsed() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

print_progress_bar() {
    local completed="$1"
    local total="$2"
    local width=40
    local percent=0
    local filled=0
    local empty=0
    local bar=""

    if [[ "$total" -gt 0 ]]; then
        if [[ "$completed" -gt "$total" ]]; then
            completed="$total"
        fi
        percent=$((completed * 100 / total))
        filled=$((completed * width / total))
    fi
    empty=$((width - filled))

    printf -v bar '%*s' "$filled" ''
    bar="${bar// /#}"
    printf -v empty_bar '%*s' "$empty" ''
    empty_bar="${empty_bar// /-}"

    printf '[%s%s] %3d%% (%d/%d)' "$bar" "$empty_bar" "$percent" "$completed" "$total"
}

show_compile_progress() {
    local ninja_log="$DEPS_DIR/onednn-build/.ninja_log"
    local completed=0
    local latest_target=""

    if [[ -f "$ninja_log" ]]; then
        completed="$(awk 'NR > 1 {count++} END {print count + 0}' "$ninja_log" 2>/dev/null || true)"
        latest_target="$(awk 'NR > 1 {target=$4} END {print target}' "$ninja_log" 2>/dev/null || true)"
    fi

    printf '[%s] Compile progress: ' "$(timestamp)"
    print_progress_bar "$completed" "$PROGRESS_TOTAL"
    if [[ -n "$latest_target" ]]; then
        printf ' latest=%s' "$latest_target"
    else
        printf ' waiting-for-ninja'
    fi
    printf '\n'
}

show_python_environment() {
    log "Python/Torch environment:"
    conda run -n "$CONDA_ENV" python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"  python: {sys.executable}")
    print(f"  torch: failed to import: {exc}")
    raise

print(f"  python: {sys.executable}")
print(f"  torch: {torch.__version__}")
print(f"  torch cuda: {torch.version.cuda}")
PY
}

run_install() {
    local start_time pid status elapsed
    start_time="$SECONDS"

    log "Install command is starting. Progress interval: ${PROGRESS_INTERVAL}s"
    log "CMAKE_ARGS=${CMAKE_ARGS_VALUE}"

    set +e
    conda run -n "$CONDA_ENV" env \
    CC="$CC_PATH" \
    CXX="$CXX_PATH" \
    CPATH="$CONDA_PREFIX_VALUE/include" \
    CPLUS_INCLUDE_PATH="$CONDA_PREFIX_VALUE/include" \
    LIBRARY_PATH="$CONDA_PREFIX_VALUE/lib" \
    LD_LIBRARY_PATH="$CONDA_PREFIX_VALUE/lib:$TORCH_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MAX_JOBS="$MAX_JOBS_VALUE" \
    CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
    VLLM_TARGET_DEVICE=cpu \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.23.0}" \
    VLLM_USE_PRECOMPILED=0 \
    FETCHCONTENT_SOURCE_DIR_ONEDNN="$ONEDNN_SOURCE_DIR" \
    CMAKE_ARGS="$CMAKE_ARGS_VALUE" \
    UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" \
    UV_LOCK_TIMEOUT="$UV_LOCK_TIMEOUT_VALUE" \
    UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
    uv pip install -v -e . --no-build-isolation --no-deps &
    pid="$!"
    set -e

    trap 'log "Interrupted; stopping install process ${pid}"; kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; exit 130' INT TERM

    log "Install process started with pid ${pid}"
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$PROGRESS_INTERVAL"
        if kill -0 "$pid" 2>/dev/null; then
            elapsed="$((SECONDS - start_time))"
            log "Still installing vLLM CPU build; elapsed $(format_elapsed "$elapsed")"
            show_compile_progress
        fi
    done

    set +e
    wait "$pid"
    status="$?"
    set -e
    trap - INT TERM

    elapsed="$((SECONDS - start_time))"
    if [[ "$status" -ne 0 ]]; then
        log "Install failed after $(format_elapsed "$elapsed") with exit code ${status}"
        return "$status"
    fi

    log "Install finished in $(format_elapsed "$elapsed")"
}

require_command conda
require_command curl
require_command tar

if [[ ! -f "$VLLM_SOURCE/pyproject.toml" || ! -d "$VLLM_SOURCE/vllm" ]]; then
    echo "VLLM_SOURCE is not a vLLM source tree: $VLLM_SOURCE" >&2
    exit 2
fi

CONDA_PREFIX_VALUE="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
CC_PATH="${CC_PATH:-$CONDA_PREFIX_VALUE/bin/x86_64-conda-linux-gnu-gcc}"
CXX_PATH="${CXX_PATH:-$CONDA_PREFIX_VALUE/bin/x86_64-conda-linux-gnu-g++}"
TORCH_LIB_PATH="$(conda run -n "$CONDA_ENV" python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"

if [[ ! -x "$CC_PATH" || ! -x "$CXX_PATH" ]]; then
    echo "GCC/G++ toolchain not found in service environment: $CC_PATH / $CXX_PATH" >&2
    exit 1
fi

log "Plugin root: $PLUGIN_ROOT"
log "vLLM source: $VLLM_SOURCE"
log "Conda environment: $CONDA_ENV"
log "oneDNN version: $ONEDNN_VERSION"
log "oneDNN source dir: $ONEDNN_SOURCE_DIR"
log "uv cache dir: $UV_CACHE_DIR_VALUE"
log "uv lock timeout: ${UV_LOCK_TIMEOUT_VALUE}s"
log "compiler jobs: $MAX_JOBS_VALUE"
log "C compiler: $CC_PATH"
log "C++ compiler: $CXX_PATH"
log "Progress interval: ${PROGRESS_INTERVAL}s"
log "Progress total: ${PROGRESS_TOTAL} build edges"

if [[ ! -x "$NVCC_PATH" ]]; then
    echo "CUDA compiler not found or not executable: $NVCC_PATH" >&2
    echo "Set NVCC_PATH to the nvcc path visible to Torch CMake." >&2
    exit 1
fi
log "nvcc path: $NVCC_PATH"
show_python_environment

mkdir -p "$SOURCE_CACHE_DIR" "$UV_CACHE_DIR_VALUE"

if [[ ! -f "$ONEDNN_ARCHIVE" ]]; then
    log "Downloading oneDNN ${ONEDNN_VERSION}: $ONEDNN_URL"
    curl -L --fail --retry 3 --connect-timeout 30 \
    "$ONEDNN_URL" \
    -o "$ONEDNN_ARCHIVE"
else
    log "Using cached oneDNN archive: $ONEDNN_ARCHIVE"
fi

if [[ ! -f "$ONEDNN_SOURCE_DIR/CMakeLists.txt" ]]; then
    log "Extracting oneDNN to: $ONEDNN_SOURCE_DIR"
    rm -rf "$ONEDNN_SOURCE_DIR"
    mkdir -p "$ONEDNN_SOURCE_DIR"
    tar -xzf "$ONEDNN_ARCHIVE" -C "$ONEDNN_SOURCE_DIR" --strip-components=1
else
    log "Using existing oneDNN source: $ONEDNN_SOURCE_DIR"
fi

log "Removing stale oneDNN FetchContent clone state"
rm -rf "$DEPS_DIR/onednn-src" "$DEPS_DIR/onednn-subbuild"
if [[ "$CLEAN_ONEDNN" == "1" ]]; then
    log "Removing oneDNN build directory: $DEPS_DIR/onednn-build"
    rm -rf "$DEPS_DIR/onednn-build"
fi

cd "$VLLM_SOURCE"

CMAKE_ARGS_VALUE="-DCMAKE_CUDA_COMPILER=${NVCC_PATH} -DFETCHCONTENT_SOURCE_DIR_ONEDNN=${ONEDNN_SOURCE_DIR}"

log "Installing vLLM CPU editable build in conda env: $CONDA_ENV"
run_install

log "vLLM install completed. Installed package:"
conda run -n "$CONDA_ENV" uv pip show vllm
log "Checking service environment dependency metadata"
conda run -n "$CONDA_ENV" python -m pip check
