#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUT_JSON="${1:-$FAN_RELEASE_ROOT/outputs/FAN_k80_repro.json}"

seen_dirs=""
unseen_dirs=""
test_dirs=""

for i in 0 1 2 3; do
  test_dirs+="$FAN_RELEASE_ROOT/outputs/k80_test_${i},"
  unseen_dirs+="$FAN_RELEASE_ROOT/outputs/k80_val_unseen_${i},"
done

for i in 0 1 2 3 4 5 6 7; do
  seen_dirs+="$FAN_RELEASE_ROOT/outputs/k80_val_seen_${i},"
done

"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/merge_submit.py" "$OUT_JSON" \
  --seen "${seen_dirs%,}" \
  --unseen "${unseen_dirs%,}" \
  --test "${test_dirs%,}"

echo "merged submission: $OUT_JSON"

