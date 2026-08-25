#!/bin/bash

TRAIN_NPY="" # Training pairs npy built from MegaScenes (see README)
BASE_DIR="" # Image root; leave empty if TRAIN_NPY stores absolute paths
MODEL="vggt" # Model: vggt | pi3 | wm
GPUS=4
OUT="./train_out/${MODEL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
torchrun --nproc_per_node="$GPUS" --master_port=29500 scripts/train.py \
  --model "$MODEL" \
  --train_npy "$TRAIN_NPY" \
  --base_dir "$BASE_DIR" \
  --out "$OUT" \
  --epochs 2 \
  --lr 5e-5 \
  --batch_size 1
