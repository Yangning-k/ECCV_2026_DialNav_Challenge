#!/usr/bin/env python3
"""Stream the large RAINbow aug_train.jsonl and build answer->destination data."""
import json
import os
import random

random.seed(0)
ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
AUG = os.environ.get("DIALNAV_SOURCE_AUG", os.path.join(ROOT, "assets", "base", "dataset", "RAINbow", "aug_train.jsonl"))
OUT_DIR = os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data"))
MAX = int(os.environ.get("AUG_MAX", "60000"))


def main():
    cache_dir = os.environ.get("DIALNAV_SCENE_CACHE_DIR")
    cache_scans = None
    if cache_dir and os.path.isdir(cache_dir):
        cache_scans = {f[:-3] for f in os.listdir(cache_dir) if f.endswith(".pt")}
    rows = []
    with open(AUG) as f:
        for line in f:
            if len(rows) >= MAX:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            if cache_scans is not None and d.get("scan") not in cache_scans:
                continue
            traj = d.get("_full_trajectory") or []
            if not traj:
                continue
            dest = traj[-1]
            for turn in d.get("_full_dialog", []):
                a = turn.get("a", "").strip()
                if not a:
                    continue
                path = turn.get("instruction_path") or []
                rows.append({
                    "instr_id": d.get("instr_id"),
                    "scan": d["scan"],
                    "q": a,
                    "start_pano": dest,
                    "neighbor_viewpoint_list": [p for p in path[:-1]],
                })
    # split by instr_id
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["instr_id"], []).append(r)
    eps = sorted(by_ep.keys())
    random.shuffle(eps)
    n_val = max(1, int(len(eps) * 0.05))
    val_eps = set(eps[:n_val])
    train, val = [], []
    for r in rows:
        (val if r["instr_id"] in val_eps else train).append(r)
    for name, arr in [("train", train), ("val", val)]:
        p = os.path.join(OUT_DIR, f"aug_ans_{name}.jsonl")
        with open(p, "w") as f:
            for r in arr:
                f.write(json.dumps(r) + "\n")
        print(name, len(arr), "->", p)


if __name__ == "__main__":
    main()
