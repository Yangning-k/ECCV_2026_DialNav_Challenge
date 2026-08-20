#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# End-to-end K80 reranker training.
#
# Trains two contrastive path-answer rerankers on the included 12k
# hard-negative candidate sets (union candidates from the inference
# localizers, and released-localizer candidates), then weight-averages them
# 0.6/0.4. Measured end-to-end test SR: 80.35, matching the submitted result.

OUT_BASE="${RERANK_OUT_BASE:-$FAN_WEIGHTS_DIR/gtl_rerank_ckpt_final}"
UNION_DIR="$OUT_BASE/union"
OFFLOC_DIR="$OUT_BASE/offloc"
mkdir -p "$UNION_DIR" "$OFFLOC_DIR"

for f in \
  "$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_union.jsonl" \
  "$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_offloc.jsonl" \
  "$TRAINING_DATA_DIR/aug_rerank_val.jsonl"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

export RERANK_HARD_NEG=1
export RERANK_NEG=4
export RERANK_BATCH=1
export RERANK_STEPS="${RERANK_STEPS:-3000}"
export RERANK_LR=1e-5
export RERANK_MAX_VAL="${RERANK_MAX_VAL:-128}"
export GTL_SCENE_CACHE_MAX=4
export RERANK_VAL="$TRAINING_DATA_DIR/aug_rerank_val.jsonl"

echo "== training reranker A (union candidates)"
export RERANK_TRAIN="$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_union.jsonl"
export RERANK_OUT="$UNION_DIR"
export GTL_SCENE_CACHE_DIR="$SCENE_CACHE_DIR"
"$DIALNAV_PYTHON" -u "$FAN_RELEASE_ROOT/src/tools/train_path_reranker.py"

echo "== training reranker B (offloc candidates)"
export RERANK_TRAIN="$TRAINING_DATA_DIR/aug_rerank_hard_train_12000_offloc.jsonl"
export RERANK_OUT="$OFFLOC_DIR"
"$DIALNAV_PYTHON" -u "$FAN_RELEASE_ROOT/src/tools/train_path_reranker.py"

echo "== averaging 0.6*A + 0.4*B -> $OUT_BASE/best.pth"
"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/average_rerank_ckpts.py" \
  "$OUT_BASE/best.pth" 0.6 "$UNION_DIR/best.pth" 0.4 "$OFFLOC_DIR/best.pth"

echo "== done. For inference:"
echo "   export RERANK_CKPT_OVERRIDE=$OUT_BASE/best.pth"
echo "   bash scripts/run_all.sh   # or run the shards in scripts/infer.sh"
