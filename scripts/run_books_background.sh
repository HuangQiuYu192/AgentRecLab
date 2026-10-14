#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

if [[ "${1:-}" == "--worker" ]]; then
    PYTHON_BIN="$2"
    RUN_DIR="$3"
    trap 'code=$?; if (( code != 0 )); then printf "failed (exit %s)\n" "$code" > "$RUN_DIR/status"; fi' EXIT
    for MODEL in iagent i2agent; do
        printf 'running %s\n' "$MODEL" > "$RUN_DIR/status"
        "$PYTHON_BIN" -u main.py run \
            --config "configs/experiments/books_deepseek41flash_${MODEL}.yaml" \
            --resume --allow-paid > "$RUN_DIR/${MODEL}.log" 2>&1
    done
    printf 'finished; inspect both experiment metrics for failed cases\n' > "$RUN_DIR/status"
    exit 0
fi

mkdir -p outputs/background
exec 9> outputs/background/books_full.lock
if ! flock -n 9; then
    echo 'A full Books background launcher is already running.' >&2
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON:-python}")"
for MODEL in iagent i2agent; do
    "$PYTHON_BIN" main.py run \
        --config "configs/experiments/books_deepseek41flash_${MODEL}.yaml" --dry-run
done

# Read the replacement key without echoing it or putting it in shell history.
read -rsp 'Enter the new AutoDL API Key (hidden): ' AUTODL_API_KEY
printf '\n'
if [[ -z "$AUTODL_API_KEY" ]]; then
    echo 'API Key must not be empty.' >&2
    exit 1
fi
export AUTODL_API_KEY
umask 077
RUN_DIR="$ROOT/outputs/background/books_full_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
printf 'starting\n' > "$RUN_DIR/status"
nohup bash "$ROOT/scripts/run_books_background.sh" --worker "$PYTHON_BIN" "$RUN_DIR" \
    </dev/null > "$RUN_DIR/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_DIR/pid"
printf '%s\n' "$RUN_DIR" > outputs/background/books_full_latest.txt
echo "Background PID: $(cat "$RUN_DIR/pid")"
echo "Logs and status: $RUN_DIR"
echo 'SSH may now be disconnected. iAgent runs first, then i2Agent.'
