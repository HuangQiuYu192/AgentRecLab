#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "${BASH_SOURCE[0]%/*}/.." && pwd)"
cd -- "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif [[ -f "$ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$ROOT/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

MODEL="${1:-iagent}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODEL" in
    iagent|i2agent|memrec) ;;
    *) echo "Usage: bash scripts/run.sh {iagent|i2agent|memrec} [experiment options]" >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" "$ROOT/main.py" --config "$ROOT/configs/experiments/books100.yaml" --model "$MODEL" "$@"
