#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "Created $ROOT/.env. Edit it before running again."
  exit 0
fi

source "$ROOT/scripts/common.sh"

if [[ "${DIALNAV_INSTALL:-0}" == "1" ]]; then
  "$DIALNAV_PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi

"$DIALNAV_PYTHON" -c 'import torch, numpy, transformers; print("python env OK")'
test -d "$DIALNAV_MATTERPORT_SIM_BUILD" || {
  echo "Matterport3D build directory does not exist: $DIALNAV_MATTERPORT_SIM_BUILD" >&2
  exit 1
}

bash "$ROOT/scripts/verify.sh"

