#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 MODEL_DIR OUT_BASE DEVICE [MAT_DIR]"
  echo "Example: $0 test_model outputs cuda:0 ./data/sliced_hsi_mat"
  exit 1
fi

MODEL_DIR=$1
OUT_BASE=$2
DEVICE=$3
MAT_DIR=${4:-./data/sliced_hsi_mat}

CHKPT_ROOT="${MODEL_DIR}/checkpoints"
if [ ! -d "$CHKPT_ROOT" ]; then
  echo "Error: checkpoint directory not found: $CHKPT_ROOT"
  exit 1
fi

# basename of model, e.g. "test_model"
MODEL_NAME=$(basename "$MODEL_DIR")

for CKPT_PATH in "$CHKPT_ROOT"/*; do
  [ -e "$CKPT_PATH" ] || continue
  CKPT_NAME=$(basename "$CKPT_PATH")
  OUT_DIR="${OUT_BASE}/${MODEL_NAME}/${CKPT_NAME}"
  mkdir -p "$OUT_DIR"

  echo "→ Running layout on checkpoint ‘$CKPT_NAME’"
  python3 layout_predictions.py \
    --checkpoint_path "$CKPT_PATH" \
    --mat_dir "$MAT_DIR" \
    --out_dir "$OUT_DIR" \
    --layout_image \
    --full_load_ds \
    --device "$DEVICE"
done
