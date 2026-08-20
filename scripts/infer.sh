#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <test|val_unseen|val_seen> <shard_idx> <gpu> <out_id>" >&2
  exit 2
fi

SPLIT="$1"
SHARD_IDX="$2"
GPU="$3"
OUT_ID="$4"
SHARD_PREFIX="${SHARD_PREFIX:-}"

OUTPUT_DIR="$FAN_RELEASE_ROOT/outputs/$OUT_ID"
mkdir -p "$OUTPUT_DIR"

COMMON_ANNO=(
  --connectivity_dir "$BASE_REPO/dataset/connectivity/"
  --val_seen_anno_paths "$BASE_REPO/dataset/RAIN_holistic/val_seen.json"
  --val_unseen_anno_paths "$BASE_REPO/dataset/RAIN_holistic/val_unseen.json"
  --test_anno_paths ''
  --qa_clip_tokenizer_path "$BASE_REPO/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz"
)

if [[ "$SPLIT" == "test" ]]; then
  TEST_PREFIX="${SHARD_PREFIX:-test}"
  ANNO_ARGS=(
    "${COMMON_ANNO[@]}"
    --test_anno_paths "$SHARD_DIR/${TEST_PREFIX}_${SHARD_IDX}.json"
    --env_names test
  )
elif [[ "$SPLIT" == "val_unseen" ]]; then
  VU_PREFIX="${SHARD_PREFIX:-val_unseen}"
  ANNO_ARGS=(
    "${COMMON_ANNO[@]}"
    --val_unseen_anno_paths "$SHARD_DIR/${VU_PREFIX}_${SHARD_IDX}.json"
    --env_names val_unseen
  )
else
  VS_PREFIX="${SHARD_PREFIX:-val_seen}"
  ANNO_ARGS=(
    "${COMMON_ANNO[@]}"
    --val_seen_anno_paths "$SHARD_DIR/${VS_PREFIX}_${SHARD_IDX}.json"
    --env_names val_seen
  )
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export ANS_CKPT="$FAN_WEIGHTS_DIR/gtl_ans_ckpt_aug/snapshot5000.pth"
export ANS_CKPT2="$FAN_WEIGHTS_DIR/gtl_ans_ckpt_qa/final.pth"
export RERANK_CKPT="${RERANK_CKPT_OVERRIDE:-$FAN_WEIGHTS_DIR/gtl_rerank_ckpt_avg_uo_6/best.pth}"
export RERANK_K=80
export RERANK_ALPHA=5
export RERANK_BATCH=2
export RERANK_TEXTS=tail,last,qa
export LOC_CAND_K=20
export GTL_SCENE_CACHE_DIR="${GTL_SCENE_CACHE_DIR_OVERRIDE:-$SCENE_CACHE_DIR}"
export GTL_SCENE_CACHE_MAX=100
export WTA_MODE=ct_0.6_cap_3

cd "$SRC_DIR/holistic"
"$DIALNAV_PYTHON" -u main.py \
  --id "$OUT_ID" \
  --output_path "$OUTPUT_DIR" \
  --basepath "$BASE_REPO" \
  "${ANNO_ARGS[@]}" \
  --batch_size 1 \
  --max_action_len 50 \
  --nav_resume_file "$BASE_REPO/dataset/checkpoints/nav_rainbow" --nav_model DST --nav_act_visited_nodes \
  --qg_resume_file "$BASE_REPO/dataset/checkpoints/q_rainbow" \
  --wta_mode "$WTA_MODE" \
  --ag_resume_file "$BASE_REPO/dataset/checkpoints/a_rainbow" \
  --loc_resume_file "$BASE_REPO/dataset/checkpoints/loc_rainbow.pth" --loc_model GTL
