#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE="${1:-ans}"
case "$MODE" in
  ans)
    TRAIN="$TRAINING_DATA_DIR/aug_ans_train.jsonl"
    VAL="$TRAINING_DATA_DIR/aug_ans_val.jsonl"
    OUT="$FAN_WEIGHTS_DIR/gtl_ans_ckpt_aug"
    ;;
  qa)
    TRAIN="$TRAINING_DATA_DIR/aug_qa_train.jsonl"
    VAL="$TRAINING_DATA_DIR/aug_qa_val.jsonl"
    OUT="$FAN_WEIGHTS_DIR/gtl_ans_ckpt_qa"
    ;;
  *)
    echo "usage: $0 <ans|qa>" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"

export ANS_GT_RESUME="$BASE_REPO/dataset/checkpoints/loc_rainbow.pth"
export ANS_GT_TRAIN="$TRAIN"
export ANS_GT_VAL="$VAL"
export ANS_GT_OUT="$OUT"
export ANS_GT_NEIGHBOR=1
export ANS_GT_NEIGHBOR_WEIGHT=0.3
export ANS_GT_EPOCHS=2
export ANS_GT_BATCH=8
export ANS_GT_LR=1e-5
export ANS_GT_MAX_SAMPLES=20000
export GTL_SCENE_CACHE_DIR="$SCENE_CACHE_DIR"
export GTL_SCENE_CACHE_MAX=4
export GTL_PANO_CACHE_MAX=1000

"$DIALNAV_PYTHON" -u "$FAN_RELEASE_ROOT/src/tools/train_ans_grounding.py"

