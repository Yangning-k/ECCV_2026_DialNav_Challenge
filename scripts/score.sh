#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SUBMIT="${1:-$FAN_RELEASE_ROOT/outputs/FAN_k80_repro.json}"
"$DIALNAV_PYTHON" "$FAN_RELEASE_ROOT/src/tools/score_release.py" --submit "$SUBMIT"

