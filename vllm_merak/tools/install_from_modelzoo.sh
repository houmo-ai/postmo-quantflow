#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUNTIME_PROFILE=${RUNTIME_PROFILE:-vllm_023_local}

if [[ "${1:-}" != "--check-only" ]]; then
    "$PYTHON_BIN" -m pip install -v -e "$PLUGIN_ROOT" \
        --no-build-isolation --no-deps
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/check_runtime_environment.py" \
    --profile "$RUNTIME_PROFILE"
