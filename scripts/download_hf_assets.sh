#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${HF_REPO_ID:?Set HF_REPO_ID to the Hugging Face repository containing assets/}"
: "${HF_REPO_TYPE:=dataset}"

mkdir -p "$ROOT/assets"
huggingface-cli download "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --local-dir "$ROOT/assets"

echo "assets downloaded from $HF_REPO_ID to $ROOT/assets"
