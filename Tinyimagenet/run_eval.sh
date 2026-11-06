#!/usr/bin/env bash
# One-command evaluation of a GGM ResNet checkpoint on the TinyImageNet test set.
#
#   bash run_eval.sh resnet18_ggm_1bit_N1     # one model
#   bash run_eval.sh all                      # every checkpoint in ./checkpoints
#   bash run_eval.sh list                     # available models
#
# Extra arguments are passed to evaluate.py (e.g. --device cpu, --batch_size 32).
# The test set is downloaded to ./data on first use. Set PYTHON to choose the interpreter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found. Install Python 3.9+ or set PYTHON=/path/to/python." >&2
  exit 1
fi
if ! "$PYTHON" -c "import torch, torchvision, PIL, numpy" >/dev/null 2>&1; then
  echo "error: missing Python packages. Install them with:" >&2
  echo "  $PYTHON -m pip install -r requirements.txt" >&2
  echo "(for an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org first)" >&2
  exit 1
fi

if [ $# -lt 1 ]; then
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  exec "$PYTHON" evaluate.py --list
fi

case "$1" in
  list|--list) exec "$PYTHON" evaluate.py --list ;;
  *) model="$1"; shift; exec "$PYTHON" evaluate.py --model "$model" "$@" ;;
esac
