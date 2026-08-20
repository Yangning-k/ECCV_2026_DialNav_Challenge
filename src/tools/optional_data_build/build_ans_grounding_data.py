#!/usr/bin/env python3
"""Convert RAIN_holistic/train.json into answer->destination localization data.

The provided training split contains human dialogs and navigation trajectories.
For each dialog turn we create a supervised sample:
  text = answer (or question + answer)
  viewpointId = the final node of the navigation trajectory (the destination).
This is provided training data only; val/test splits are never touched.
"""
import json
import os

ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
SRC = os.environ.get("DIALNAV_TRAIN_ANNO", os.path.join(ROOT, "assets", "base", "dataset", "RAIN_holistic", "train.json"))
OUT_DIR = os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data"))
OUT_ANS = os.path.join(OUT_DIR, "ans_train.jsonl")
OUT_QA = os.path.join(OUT_DIR, "qa_train.jsonl")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(SRC))
    ans_rows = []
    qa_rows = []
    for ep in data:
        traj = ep.get("nav_trajectory") or []
        if not traj:
            continue
        dest = traj[-1]
        for turn in ep.get("dialog", []):
            q = turn.get("q", "").strip()
            a = turn.get("a", "").strip()
            if not a:
                continue
            ans_rows.append({
                "instr_id": ep["instr_id"],
                "scan": ep["scan"],
                "q": a,
                "start_pano": dest,
            })
            qa_rows.append({
                "instr_id": ep["instr_id"],
                "scan": ep["scan"],
                "q": (q + " " + a).strip(),
                "start_pano": dest,
            })
    with open(OUT_ANS, "w") as f:
        for r in ans_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT_QA, "w") as f:
        for r in qa_rows:
            f.write(json.dumps(r) + "\n")
    print("ans rows", len(ans_rows), "->", OUT_ANS)
    print("qa rows", len(qa_rows), "->", OUT_QA)


if __name__ == "__main__":
    main()
