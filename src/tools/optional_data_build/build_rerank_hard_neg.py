#!/usr/bin/env python3
"""Build hard-negative path reranker data from RAINbow augmented dialogs.

For each augmented turn we use the current answer-grounding GTL to generate the
same top-k candidate goals/paths that inference would consider.  The positive
path is the turn's ``instruction_path``.  This creates a much harder and more
realistic negative set than sampling arbitrary graph destinations.
"""

import json
import os
import random
import sys

import torch

ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
REPO = os.environ.get("DIALNAV_BASE_REPO", os.path.join(ROOT, "assets", "base"))
sys.path.insert(0, os.path.join(ROOT, "src", "holistic"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluator import Evaluator
from holistic_models.GTL.GTL import GraphVlnAgentModel

AUG = os.environ.get("DIALNAV_SOURCE_AUG", os.path.join(ROOT, "assets", "base", "dataset", "RAINbow", "aug_train.jsonl"))
OUT_DIR = os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data"))
MAX = int(os.environ.get("HARD_MAX", "12000"))
TOPK = int(os.environ.get("HARD_TOPK", "20"))
ALPHA = float(os.environ.get("HARD_ALPHA", "5"))
TEXTS = os.environ.get("HARD_TEXTS", "full,tail,last,qa")
CKPT = os.environ.get("HARD_ANS_CKPT", os.path.join(ROOT, "assets", "fan", "weights", "gtl_ans_ckpt_aug", "snapshot5000.pth"))
OUT_NAME = os.environ.get("HARD_OUT", "aug_rerank_hard_train.jsonl")


def load_rows():
    rows = []
    with open(AUG) as f:
        for line in f:
            if len(rows) >= MAX:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            for turn in d.get("_full_dialog", []):
                a = turn.get("a", "").strip()
                path = turn.get("instruction_path") or []
                if not a or len(path) < 2:
                    continue
                rows.append({
                    "instr_id": d.get("instr_id"),
                    "scan": d["scan"],
                    "answer": a,
                    "positive_path": path,
                })
    random.shuffle(rows)
    return rows


def candidates_for(model, evaluator, scan, qvp, text, k, alpha):
    try:
        model.localize([scan], [text])
    except Exception:
        return []
    agent = getattr(model, "agent", None)
    logits = getattr(agent, "last_loc_logits", None)
    nodes = getattr(agent, "last_loc_nodes", None)
    if logits is None or nodes is None or not nodes:
        return []
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, min(k, probs.shape[0]))
    dist_map = evaluator.shortest_distances[scan][qvp]
    out = []
    for idx, val in zip(topk.indices, topk.values):
        c = nodes[0][int(idx)]
        p = float(val)
        if c == qvp:
            continue
        d = dist_map.get(c, -1)
        score = d + alpha * p
        try:
            path = list(evaluator.shortest_paths[scan][qvp][c])
        except Exception:
            continue
        out.append({"goal": c, "prob": p, "score": score, "path": path[:20]})
    return out


def candidates_for_texts(model, evaluator, scan, qvp, answer, k, alpha, texts):
    merged = {}
    for spec in texts.split(","):
        spec = spec.strip()
        if spec == "full":
            text = answer
        elif spec == "tail":
            text = " ".join(answer.split()[-20:])
        elif spec == "last":
            parts = [p.strip() for p in answer.replace(" .", ".").split(". ")]
            parts = [p for p in parts if p]
            text = parts[-1] if parts else answer
        elif spec == "qa":
            # Hard-negative builder does not have the question text; skip qa.
            continue
        else:
            continue
        for cand in candidates_for(model, evaluator, scan, qvp, text, k, alpha):
            g = cand["goal"]
            old = merged.get(g)
            if old is None or cand["prob"] > old["prob"]:
                merged[g] = cand
    return sorted(merged.values(), key=lambda x: -x["prob"])[:k]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(0)
    rows = load_rows()
    scans = sorted({r["scan"] for r in rows})
    evaluator = Evaluator(f"{REPO}/dataset/connectivity/", scans)
    model = GraphVlnAgentModel(REPO, {"resume_file": CKPT, "scan_list": scans})

    out = []
    for i, r in enumerate(rows):
        pos = r["positive_path"][:20]
        cands = candidates_for_texts(model, evaluator, r["scan"], pos[0],
                                     r["answer"], TOPK, ALPHA, TEXTS)
        cand_paths = [c["path"] for c in cands if c["path"] != pos]
        # Keep the positive target if it is present as a candidate.
        rank = None
        for j, c in enumerate(cands):
            if c["path"] == pos:
                rank = j
                break
        if rank is None:
            rank = -1
        out.append({
            "instr_id": r["instr_id"],
            "scan": r["scan"],
            "answer": r["answer"],
            "positive_path": pos,
            "candidate_paths": cand_paths,
            "positive_rank": rank,
        })
        if i % 1000 == 0:
            print("processed", i, "/", len(rows), flush=True)

    path = os.path.join(OUT_DIR, OUT_NAME)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    hit = sum(1 for r in out if r["positive_rank"] >= 0) / max(len(out), 1)
    top1 = sum(1 for r in out if r["positive_rank"] == 0) / max(len(out), 1)
    print("wrote", len(out), "rows ->", path)
    print("positive in candidates", hit, "top1", top1)


if __name__ == "__main__":
    main()
