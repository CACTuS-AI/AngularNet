#!/usr/bin/env bash
# One-command evaluation of GGM ResNet checkpoints.
#
#   bash run_eval.sh tinyimagenet resnet18_ggm_1bit_N1   # one model
#   bash run_eval.sh imagenet all                        # every checkpoint of a dataset
#   bash run_eval.sh list                                # all available models
#
# Extra arguments are passed to evaluate.py (e.g. --device cpu, --batch_size 32).
# TinyImageNet is downloaded to ./data/tinyimagenet on first use; ImageNet must be placed in
# ./data/imagenet (see README). Set PYTHON to choose the interpreter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found. Install Python 3.10+ or set PYTHON=/path/to/python." >&2
  exit 1
fi
need="torch, torchvision, PIL, numpy"
if [ "${1:-}" = "imagenet" ]; then need="$need, timm"; fi
if ! "$PYTHON" -c "import $need" >/dev/null 2>&1; then
  echo "error: missing Python packages. Install them with:" >&2
  echo "  $PYTHON -m pip install -r requirements.txt" >&2
  echo "(see Setup in README.md)" >&2
  exit 1
fi

usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; echo; }
case "${1:-}" in
  "")              usage; exec "$PYTHON" evaluate.py --list ;;
  list|--list)     exec "$PYTHON" evaluate.py --list ;;
  tinyimagenet|imagenet)
    dataset="$1"; shift
    case "${1:-list}" in
      list|--list) exec "$PYTHON" evaluate.py --dataset "$dataset" --list ;;
      *) model="$1"; shift; exec "$PYTHON" evaluate.py --dataset "$dataset" --model "$model" "$@" ;;
    esac ;;
  *) echo "error: unknown dataset '$1' (use tinyimagenet or imagenet)" >&2; usage; exit 2 ;;
esac
