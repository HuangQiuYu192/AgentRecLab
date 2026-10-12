#!/usr/bin/env bash
set -euo pipefail
PREFIX="${1:-}"
if [[ -z "$PREFIX" ]]; then printf -v PREFIX 'acceptance_%(%Y%m%d_%H%M%S)T' -1; fi
exec "$BASH" "${BASH_SOURCE[0]%/*}/run.sh" acceptance --prefix "$PREFIX"
