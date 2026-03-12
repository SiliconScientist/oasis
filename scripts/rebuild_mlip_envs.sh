#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
UV_BIN="${UV_BIN:-uv}"
MODELS=(mace mattersim orb_v3 sevennet uma)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: Python interpreter '$PYTHON_BIN' was not found in PATH." >&2
    echo "Set PYTHON_BIN=/path/to/python3.13 and rerun." >&2
    exit 1
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "error: uv executable '$UV_BIN' was not found in PATH." >&2
    echo "Set UV_BIN=/path/to/uv and rerun." >&2
    exit 1
fi

rebuild_env() {
    local model="$1"
    local env_dir="$ROOT_DIR/envs/$model/.venv"
    local req_file="$ROOT_DIR/envs/$model/requirements.txt"

    if [[ ! -f "$req_file" ]]; then
        echo "error: requirements file not found: $req_file" >&2
        exit 1
    fi

    echo "==> Rebuilding $model"
    rm -rf "$env_dir"
    "$UV_BIN" venv --python "$PYTHON_BIN" "$env_dir"
    "$UV_BIN" pip install --python "$env_dir/bin/python" -r "$req_file"
    "$UV_BIN" pip install --python "$env_dir/bin/python" -e "$ROOT_DIR"
}

for model in "${MODELS[@]}"; do
    rebuild_env "$model"
done

echo
echo "All MLIP environments rebuilt."
echo "CatBench is supplied at runtime from $ROOT_DIR/vendor/catbench."
