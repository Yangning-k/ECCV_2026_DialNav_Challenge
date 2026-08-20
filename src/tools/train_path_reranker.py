#!/usr/bin/env python3
"""Contrastively fine-tune the LANA answer model for path-answer reranking."""

import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.environ.get("DIALNAV_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
REPO = os.environ.get("DIALNAV_BASE_REPO", os.path.join(ROOT, "assets", "base"))
sys.path.insert(0, os.path.join(ROOT, "src", "holistic"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "modules", "nav", "DST", "map_nav_src"))

import utils.ops  # noqa: F401  (register DST utils namespace before LANA import)
from evaluator import Evaluator
from holistic_models.LANA.LANA import LANA

CONN = os.path.join(REPO, "dataset", "connectivity", "")
CLIP = os.path.join(REPO, "dataset", "modules", "clip_tokenizer", "bpe_simple_vocab_16e6.txt.gz")
AG_CKPT = os.path.join(REPO, "dataset", "checkpoints", "a_rainbow")
TRAIN = os.environ.get("RERANK_TRAIN", os.path.join(os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data")), "aug_rerank_hard_train.jsonl"))
VAL = os.environ.get("RERANK_VAL", os.path.join(os.environ.get("DIALNAV_TRAINING_DATA_DIR", os.path.join(ROOT, "assets", "fan", "training_data")), "aug_rerank_val.jsonl"))
HARD_NEG = os.environ.get("RERANK_HARD_NEG", "1")
OUT_DIR = os.environ.get("RERANK_OUT", os.path.join(ROOT, "assets", "fan", "weights", "gtl_rerank_ckpt_hard"))
MAX_TRAIN = int(os.environ.get("RERANK_MAX_TRAIN", "0"))
MAX_VAL = int(os.environ.get("RERANK_MAX_VAL", "128"))
NEG = int(os.environ.get("RERANK_NEG", "4"))
BATCH = int(os.environ.get("RERANK_BATCH", "1"))
STEPS = int(os.environ.get("RERANK_STEPS", "200"))
LR = float(os.environ.get("RERANK_LR", "1e-5"))


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def score_paths(agent, env, scan, paths, answer):
    env.reset([scan] * len(paths),
              [p[0] for p in paths],
              [3.14] * len(paths),
              [])
    t_hist, t_act, hist_lens, action_lens, _ = agent.get_history_and_actions_for_speaker(env, paths)
    bs = len(paths)
    hist_embeds = [agent.vln_bert('history').expand(bs, -1, -1)]
    action_embeds = []
    for t, action_input in enumerate(t_act):
        hist_embeds.append(agent.vln_bert(**t_hist[t]))
        action_embeds.append(agent.vln_bert(**action_input).unsqueeze(1))
    token_ids = agent.tokenizer.encode(answer)[:200]
    max_len = len(token_ids)
    words = torch.zeros(bs, max_len, dtype=torch.long, device='cuda')
    for j in range(bs):
        words[j, :max_len] = torch.tensor(token_ids, dtype=torch.long, device='cuda')
    future_mask = agent.make_future_mask(words.shape[1], hist_embeds[0].dtype, words.device)
    caption_lengths = (words != 0).sum(-1)
    ones = torch.ones_like(words)
    caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)
    language_inputs = {
        'mode': 'language',
        'txt_ids': words,
        'txt_masks': caption_mask,
        'future_mask': future_mask,
    }
    txt_embeds = agent.vln_bert(**language_inputs)
    caption_input = {
        'mode': 'visual',
        'hist_embeds': hist_embeds,
        'txt_embeds': txt_embeds,
        'txt_masks': caption_mask,
        'hist_lens': hist_lens,
        'action_embeds': action_embeds,
        'action_lens': action_lens,
        'is_train_caption': True,
        'future_mask': future_mask,
    }
    logits = agent.vln_bert(**caption_input)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = words[:, 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=agent.pad_token_id,
        reduction='none',
    ).view(bs, -1)
    valid = (shift_labels != agent.pad_token_id)
    return -ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def evaluate(agent, env, evaluator, rows):
    agent.vln_bert.eval()
    rank_hits = 0
    n = 0
    with torch.no_grad():
        for r in rows:
            pos = r["positive_path"][:20]
            if len(pos) < 2:
                continue
            start = pos[0]
            if r.get("candidate_paths"):
                negs = [p for p in r["candidate_paths"] if p != pos]
            else:
                cands = list(evaluator.shortest_paths[r["scan"]][start].keys())
                negs = [evaluator.shortest_paths[r["scan"]][start][d][:20]
                        for d in cands if d != pos[-1]]
            if not negs:
                continue
            sample = random.sample(negs, min(NEG, len(negs)))
            if r.get("candidate_paths"):
                paths = [pos] + sample
            else:
                paths = [pos] + sample
            scores = score_paths(agent, env, r["scan"], paths, r["answer"])
            if scores[0] == max(scores):
                rank_hits += 1
            n += 1
    agent.vln_bert.train()
    return rank_hits / max(n, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    train_rows = load_rows(TRAIN)
    val_rows = load_rows(VAL)
    if MAX_TRAIN:
        train_rows = train_rows[:MAX_TRAIN]
    if MAX_VAL:
        val_rows = val_rows[:MAX_VAL]
    print("train rows", len(train_rows), "val rows", len(val_rows), flush=True)
    scans = sorted({r["scan"] for r in train_rows[:MAX_TRAIN or len(train_rows)]})
    evaluator = Evaluator(CONN, scans)
    ag = LANA(REPO, {
        "scan_list": scans,
        "resume_file": AG_CKPT,
        "connectivity_dir": CONN,
        "bpe_path": CLIP,
        "max_action_len": 20,
    }, type="ag")
    agent = ag.agent
    env = ag.language_env
    agent.vln_bert.train()
    optimizer = torch.optim.AdamW(
        [p for p in agent.vln_bert.parameters() if p.requires_grad], lr=LR)
    best_val = 0.0
    for step in range(1, STEPS + 1):
        samples = random.sample(train_rows, BATCH)
        losses = []
        for r in samples:
            pos = r["positive_path"][:20]
            if len(pos) < 2:
                continue
            start = pos[0]
            if r.get("candidate_paths"):
                negs = [p for p in r["candidate_paths"] if p != pos]
            else:
                cands = list(evaluator.shortest_paths[r["scan"]][start].keys())
                negs = [evaluator.shortest_paths[r["scan"]][start][d][:20]
                        for d in cands if d != pos[-1]]
            if not negs:
                continue
            neg_sample = random.sample(negs, min(NEG, len(negs)))
            paths = [pos] + neg_sample
            scores = score_paths(agent, env, r["scan"], paths, r["answer"])
            loss = -scores[0] + torch.logsumexp(scores, dim=0)
            losses.append(loss)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"[train] step={step} loss={loss.item():.4f}", flush=True)
        if step % 25 == 0:
            val = evaluate(agent, env, evaluator, val_rows)
            print(f"[val] step={step} rank_acc={val:.3f}", flush=True)
            agent.vln_bert.train()
            if val > best_val:
                best_val = val
                torch.save({"vln_bert": agent.vln_bert.state_dict(),
                            "critic": agent.critic.state_dict()},
                           os.path.join(OUT_DIR, "best.pth"))
        if step % 100 == 0:
            torch.save({"vln_bert": agent.vln_bert.state_dict(),
                        "critic": agent.critic.state_dict()},
                       os.path.join(OUT_DIR, "latest.pth"))
    print("best_val", best_val)


if __name__ == "__main__":
    main()
