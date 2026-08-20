#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

GPU_COUNT="${GPU_COUNT:-8}"

bash "$FAN_RELEASE_ROOT/scripts/make_shards.sh"

run_group() {
  local split="$1"
  local n="$2"
  local wave_size=4
  for ((start=0; start<n; start+=wave_size)); do
    local end=$((start+wave_size))
    [[ $end -gt $n ]] && end=$n
    local pids=()
    for ((idx=start; idx<end; idx++)); do
      local gpu=$((idx % GPU_COUNT))
      local cache_dir="$FAN_RELEASE_ROOT/.cache/scene_cache_${split}_${idx}"
      mkdir -p "$cache_dir"
      GTL_SCENE_CACHE_DIR_OVERRIDE="$cache_dir" \
        bash "$FAN_RELEASE_ROOT/scripts/infer.sh" "$split" "$idx" "$gpu" "k80_${split}_${idx}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
}

run_group test 4
run_group val_unseen 4
run_group val_seen 8

bash "$FAN_RELEASE_ROOT/scripts/merge.sh"
