#!/bin/bash

BASE_DIR="/share/phoenix/nfs06/S9/jt664/megascenes_local/data/megaunscene_release" # MegaUnScene root
DATA="unscene-t" # Dataset: unscene-t | unscene
MODEL="wm" # Model: vggt | pi3 | wm
NO_CKPT=1 # Set to 1 to use pre-trained model only; 0 = load fine-tuned ckpt

if [ "$NO_CKPT" = "1" ]; then
  CKPT=""
  OUT="./eval_out/${MODEL}/pre-trained/${DATA}"
else
  if [ "$MODEL" = "vggt" ]; then CKPT=ckpts/VGGT_changed_bias.pth; elif [ "$MODEL" = "wm" ]; then CKPT=ckpts/WM_changed_bias.pth; else CKPT=ckpts/PI3_changed_bias.pth; fi
  OUT="./eval_out/${MODEL}/fine-tuned/${DATA}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
python scripts/eval.py \
  --data "$DATA" \
  --base_dir "$BASE_DIR" \
  --model "$MODEL" \
  --ckpt "$CKPT" \
  --out "$OUT"
