#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "$SHARD_DIR"

"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/shard_anno.py" \
  "$BASE_REPO/dataset/RAIN_holistic/test.json" "$SHARD_DIR/test" 4

"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/shard_anno.py" \
  "$BASE_REPO/dataset/RAIN_holistic/val_unseen.json" "$SHARD_DIR/val_unseen" 4

"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/shard_anno.py" \
  "$BASE_REPO/dataset/RAIN_holistic/val_seen.json" "$SHARD_DIR/val_seen" 8

echo "shards written to $SHARD_DIR"

