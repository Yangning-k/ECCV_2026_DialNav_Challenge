#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

required=(
  "$BASE_REPO/dataset/checkpoints/nav_rainbow"
  "$BASE_REPO/dataset/checkpoints/q_rainbow"
  "$BASE_REPO/dataset/checkpoints/a_rainbow"
  "$BASE_REPO/dataset/checkpoints/loc_rainbow.pth"
  "$BASE_REPO/dataset/RAIN_holistic/train.json"
  "$BASE_REPO/dataset/RAIN_holistic/val_seen.json"
  "$BASE_REPO/dataset/RAIN_holistic/val_unseen.json"
  "$BASE_REPO/dataset/RAIN_holistic/test.json"
  "$BASE_REPO/dataset/features/clip_vit-h14_mp3d_original.hdf5"
  "$BASE_REPO/dataset/features/CLIP-ViT-B-16-views.tsv"
  "$BASE_REPO/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz"
  "$BASE_REPO/dataset/RAINbow/aug_train.jsonl"
  "$FAN_WEIGHTS_DIR/gtl_ans_ckpt_aug/snapshot5000.pth"
  "$FAN_WEIGHTS_DIR/gtl_ans_ckpt_qa/final.pth"
  "$FAN_WEIGHTS_DIR/gtl_rerank_ckpt_hard/best.pth"
  "$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_union.jsonl"
  "$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_offloc.jsonl"
  "$FAN_WEIGHTS_DIR/gtl_rerank_ckpt_avg_uo_6/best.pth"
  "$FAN_RELEASE_ROOT/assets/fan/final/FAN_k80.json"
)

for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done

if [[ -f "$FAN_RELEASE_ROOT/assets/manifest/assets.sha256" ]]; then
  (
    cd "$FAN_RELEASE_ROOT"
    grep -v ' assets/manifest/assets.sha256$' assets/manifest/assets.sha256 | sha256sum -c -
  )
fi

"$DIALNAV_PYTHON" - <<'PY'
import os
import torch
from transformers import AutoTokenizer
tok = os.environ["DIALNAV_BERT_TOKENIZER_DIR"]
AutoTokenizer.from_pretrained(tok)
print("assets and imports OK", torch.__version__)
PY

echo "verify OK"
