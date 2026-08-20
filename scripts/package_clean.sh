#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/../FAN_k80_release_clean}"

mkdir -p "$DEST"
rsync -a \
  --exclude='.env' \
  --exclude='.cache/' \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$ROOT/" "$DEST/"

mkdir -p "$DEST/outputs"
touch "$DEST/outputs/.gitkeep"

echo "clean release exported to $DEST"
