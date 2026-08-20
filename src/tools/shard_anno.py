"""Split an annotation json into K shard files (contiguous episode ranges).

Usage: python shard_anno.py <anno.json> <out_prefix> <K>
Writes <out_prefix>_0.json ... <out_prefix>_K-1.json, each with the same schema.
Contiguous ranges keep same-scan episodes together, improving scene-cache reuse.
"""
import json
import sys


def main():
    anno_path, out_prefix, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data = json.load(open(anno_path))
    n = len(data)
    base = n // k
    rem = n % k
    start = 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        shard = data[start:end]
        out = f"{out_prefix}_{i}.json"
        with open(out, "w") as f:
            json.dump(shard, f)
        print(f"{out}: {len(shard)} episodes (instr_id {shard[0]['instr_id']}..{shard[-1]['instr_id']})")
        start = end
    assert start == n


if __name__ == "__main__":
    main()
