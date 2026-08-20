#!/usr/bin/env python3
"""Fine-tune the GTL localizer to ground Guide answers to destination
viewpoints using only the provided training split."""
import os
import sys
import time
import random
import json

import numpy as np
import torch

ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
REPO = os.environ.get("DIALNAV_BASE_REPO", os.path.join(ROOT, "assets", "base"))
sys.path.insert(0, os.path.join(ROOT, "src", "holistic"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "modules", "loc", "GTL"))
sys.path.insert(0, os.path.join(ROOT, "src", "modules", "nav", "DST", "map_nav_src"))

from gtl.utils.data import ImageFeaturesDB
from gtl.main.env import GraphEnvBatch
from gtl.main.graph_agent import GraphVlnAgent
from gtl.main.data_loader import VLNDialogDataset
from holistic_models.GTL.default_args import get_default_args
from transformers import AutoTokenizer

CKPT = os.environ.get("ANS_GT_RESUME", f"{REPO}/dataset/checkpoints/loc_rainbow.pth")
TRAIN = os.environ.get("ANS_GT_TRAIN", os.path.join(os.environ.get("DIALNAV_TRAINING_DATA_DIR", REPO), "aug_ans_train.jsonl"))
VAL = os.environ.get("ANS_GT_VAL", os.path.join(os.environ.get("DIALNAV_TRAINING_DATA_DIR", REPO), "aug_ans_val.jsonl"))
OUT_DIR = os.environ.get("ANS_GT_OUT", os.path.join(ROOT, "assets", "fan", "weights", "gtl_ans_ckpt_aug"))
TOKENIZER = os.environ.get("DIALNAV_BERT_TOKENIZER_DIR", os.path.join(ROOT, "src", "modules", "qa", "LANA", "tokenizer_files", "bert-base-uncase"))


def evaluate(agent, dataset, batch=16):
    agent.vln_bert.eval()
    preds, gts, scans = [], [], []
    with torch.no_grad():
        for i in range(0, len(dataset), batch):
            items = [dataset[j] for j in range(i, min(i + batch, len(dataset)))]
            pred = agent.localize([it["instruction"] for it in items],
                                  [it["scanName"] for it in items])
            preds.extend(pred)
            gts.extend([it["viewpointId"] for it in items])
            scans.extend([it["scanName"] for it in items])
    dists = [agent.env.shortest_distances[s][p][g] for s, p, g in zip(scans, preds, gts)]
    n = len(dists)
    return {
        "0m": sum(d == 0 for d in dists) / n,
        "3m": sum(d <= 3 for d in dists) / n,
        "5m": sum(d <= 5 for d in dists) / n,
        "10m": sum(d <= 10 for d in dists) / n,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    max_steps = int(os.environ.get("ANS_GT_MAX_STEPS", "0"))
    max_samples = int(os.environ.get("ANS_GT_MAX_SAMPLES", "0"))
    max_scans = int(os.environ.get("ANS_GT_MAX_SCANS", "0"))
    max_val = int(os.environ.get("ANS_GT_MAX_VAL", "500"))
    batch_env = int(os.environ.get("ANS_GT_BATCH", "0"))
    epochs = int(os.environ.get("ANS_GT_EPOCHS", "4"))
    args = get_default_args(REPO)
    args.resume_file = CKPT
    args.use_neighbor_loss = os.environ.get("ANS_GT_NEIGHBOR") == "1"
    args.neighbor_loss_weight = float(os.environ.get("ANS_GT_NEIGHBOR_WEIGHT", "0.3"))
    args.lr = float(os.environ.get("ANS_GT_LR", "1e-5"))
    args.batch_size = batch_env if batch_env else 8
    args.max_dialog_len = 200
    args.max_instr_len = 200
    args.optim = "adamW"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    train_ds = VLNDialogDataset(TRAIN, tokenizer=tokenizer, max_dialog_len=200,
                                connectivity_dir=args.connectivity_dir, debug=False)
    val_ds = VLNDialogDataset(VAL, tokenizer=tokenizer, max_dialog_len=200,
                              connectivity_dir=args.connectivity_dir, debug=False)
    if args.use_neighbor_loss:
        src = json.load(open(f"{REPO}/dataset/RAIN_holistic/train.json"))
        ep_ends = {ep["instr_id"]: ep.get("end_panos", []) for ep in src}
        for ds in (train_ds, val_ds):
            for it in ds.data:
                gt = it["viewpointId"]
                if not it.get("neighbor_viewpoint_list"):
                    it["neighbor_viewpoint_list"] = [p for p in ep_ends.get(it["episodeId"], []) if p != gt]
        print("attached neighbor end_panos", flush=True)
    if max_scans:
        scan_list = sorted(set(it["scanName"] for it in train_ds.data))[:max_scans]
        train_ds.data = [it for it in train_ds.data if it["scanName"] in scan_list]
        val_ds.data = [it for it in val_ds.data if it["scanName"] in scan_list]
        print("restricted to scans", scan_list, flush=True)
    if max_samples:
        random.shuffle(train_ds.data)
        random.shuffle(val_ds.data)
        train_ds.data = train_ds.data[:max_samples]
        val_ds.data = val_ds.data[:min(max_val, len(val_ds.data))]
        print("restricted to samples", len(train_ds.data), len(val_ds.data), flush=True)
    scans = sorted(set(it["scanName"] for it in train_ds.data) |
                   set(it["scanName"] for it in val_ds.data))
    print("scans", len(scans))

    feat_db = ImageFeaturesDB(args.feat_path, args.image_feat_size, use_gpu=True, preload_all=False)
    env = GraphEnvBatch(args.connectivity_dir, feat_db=feat_db, batch_size=args.batch_size,
                        angle_feat_size=args.angle_feat_size)
    env._load_nav_graphs(scans)

    agent = GraphVlnAgent(args, env)
    agent.load(args.resume_file)
    agent.vln_bert.train()
    agent.critic.eval()

    n = len(train_ds)
    random.seed(args.seed)
    from collections import defaultdict
    by_scan = defaultdict(list)
    for i, item in enumerate(train_ds.data):
        by_scan[item["scanName"]].append(i)
    step = 0
    for epoch in range(epochs):
        # Keep each mini-batch within a single scan.  This makes the GTL scene
        # cache effective even when its bound is small, and it is much faster
        # than random scan-hopping batches.
        scan_order = list(by_scan.keys())
        random.shuffle(scan_order)
        idx = []
        for s in scan_order:
            ids = list(by_scan[s])
            random.shuffle(ids)
            idx.extend(ids)
        for b in range(0, n, args.batch_size):
            items = [train_ds[idx[j]] for j in range(b, min(b + args.batch_size, n))]
            t0 = time.time()
            loss, pred, gt = agent.train_step(items)
            step += 1
            if step % 10 == 0:
                torch.cuda.empty_cache()
            if max_steps and step > max_steps:
                break
            if step % 25 == 0:
                acc = sum(1 for p, g in zip(pred, gt) if p == g) / len(gt)
                print(f"[train] epoch={epoch} step={step} loss={loss:.4f} acc={acc:.3f} "
                      f"time={time.time()-t0:.2f}s", flush=True)
            if step % 500 == 0:
                met = evaluate(agent, val_ds)
                print(f"[val] step={step} " + " ".join(f"{k}={v:.3f}" for k, v in met.items()), flush=True)
                agent.vln_bert.train()
                agent.save(step, os.path.join(OUT_DIR, "latest.pth"))
                agent.save(step, os.path.join(OUT_DIR, f"snapshot{step}.pth"))
        if max_steps and step > max_steps:
            break
    met = evaluate(agent, val_ds)
    print("FINAL VAL", met)
    agent.vln_bert.train()
    agent.save(step, os.path.join(OUT_DIR, "final.pth"))


if __name__ == "__main__":
    main()
