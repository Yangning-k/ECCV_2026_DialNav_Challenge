#!/usr/bin/env python3
import json, random, os

random.seed(0)
ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
base = os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data"))
rows = [json.loads(l) for l in open(os.path.join(base, "ans_train.jsonl"))]
by_ep = {}
for r in rows:
    by_ep.setdefault(r["instr_id"], []).append(r)
ep_ids = sorted(by_ep.keys())
random.shuffle(ep_ids)
n_val = max(1, int(len(ep_ids) * 0.1))
val_eps = set(ep_ids[:n_val])
train_rows, val_rows = [], []
for r in rows:
    (val_rows if r["instr_id"] in val_eps else train_rows).append(r)
for name, arr in [("train", train_rows), ("val", val_rows)]:
    p = os.path.join(base, f"ans_{name}.jsonl")
    with open(p, "w") as f:
        for r in arr:
            f.write(json.dumps(r) + "\n")
    print(name, len(arr), "->", p)
