#!/usr/bin/env bash
# Runs evaluate.py with the given arguments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: Python not found. Install Python 3.11+ or set PYTHON=/path/to/python." >&2
  exit 1
fi
exec "$PYTHON" evaluate.py "$@"
