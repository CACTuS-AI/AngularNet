#!/usr/bin/env bash
# Same as `python evaluate.py ...`, for bash users (Linux, macOS, WSL):
#
#   bash run_eval.sh tinyimagenet resnet18_ggm_1bit_N1   # one model
#   bash run_eval.sh imagenet all                        # every checkpoint of a dataset
#   bash run_eval.sh cifar10 vgg16_ggm_1bit_N1
#   bash run_eval.sh list                                # all available models
#
# Extra arguments are passed to evaluate.py (e.g. --device cpu, --batch_size 32).
# Set PYTHON to choose the interpreter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: Python not found. Install Python 3.11+ or set PYTHON=/path/to/python." >&2
  exit 1
fi
exec "$PYTHON" evaluate.py "$@"
