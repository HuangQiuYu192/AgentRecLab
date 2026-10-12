#!/usr/bin/env bash
set -euo pipefail
exec "$BASH" "${BASH_SOURCE[0]%/*}/run.sh" iagent "$@"
