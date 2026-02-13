#!/bin/bash

BASE_DIR="" # MegaUnScene root
DATA="unscene" # Dataset: unscene-t | unscene
CKPT=ckpts/VGGT_changed_bias.pth  # Checkpoint to load (empty = base model only)
OUT="./eval_out" # Output directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
python scripts/eval.py \
  --data "$DATA" \
  --base_dir "$BASE_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT"
