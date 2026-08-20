#!/usr/bin/env python3
"""Weight-average reranker checkpoints. Existing checkpoints are only read.

Usage:
  python average_rerank_ckpts.py OUT.pth W1 CKPT1 [W2 CKPT2 ...]

Weights are normalized to sum to 1. The vln_bert and critic state dicts are
averaged element-wise. Existing checkpoints are only read.
"""

import sys
import torch


def _avg_into(acc, src, w):
    for k, v in src.items():
        if isinstance(v, dict):
            sub = acc.setdefault(k, {})
            _avg_into(sub, v, w)
        elif isinstance(v, torch.Tensor):
            if k in acc:
                acc[k] = acc[k] + w * v
            else:
                acc[k] = w * v.clone()
        else:
            if k not in acc:
                acc[k] = v


def main():
    if len(sys.argv) < 5 or (len(sys.argv) - 2) % 2 != 0:
        print("usage: average_rerank_ckpts.py OUT.pth W1 CKPT1 [W2 CKPT2 ...]")
        sys.exit(2)
    out = sys.argv[1]
    pairs = []
    for i in range(2, len(sys.argv), 2):
        w = float(sys.argv[i])
        p = sys.argv[i + 1]
        pairs.append((w, p))
    total = sum(w for w, _ in pairs)
    weights = [w / total for w, _ in pairs]
    print("normalized weights:", list(zip(weights, [p for _, p in pairs])))

    summed = None
    for w, p in pairs:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if summed is None:
            summed = {}
            _avg_into(summed, d, w)
        else:
            _avg_into(summed, d, w)
    torch.save(summed, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
