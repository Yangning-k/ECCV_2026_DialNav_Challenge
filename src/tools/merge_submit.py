"""Merge per-split run outputs into the split-keyed submission.json.

Usage:
  python merge_submit.py <out_submit.json> [--seen d1,d2,...] [--unseen d1,d2,...] [--test d1,d2,...]

Each split accepts one or more comma-separated output directories (shards are
merged by instr_id). Old positional form is still supported:
  python merge_submit.py <out_submit.json> <val_seen_dir> <val_unseen_dir> <test_dir>
"""
import json
import sys


def load_split(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_split_shards(dirs, split_name):
    """Merge multiple output dirs for one split into one list of episodes."""
    by_id = {}
    for d in dirs:
        for item in load_split(f"{d.rstrip('/')}/{split_name}.json"):
            by_id[item["instr_id"]] = item
    return list(by_id.values())


def flatten_path(path):
    flat_path = []
    for step in path:
        if isinstance(step, list):
            flat_path.extend(flatten_path(step))
        else:
            flat_path.append(step)
    return flat_path


def make_submit_output(output):
    submit_output = []
    for item in output:
        submit_item = {k: v for k, v in item.items() if k != "navigation_detail"}
        submit_item["path"] = flatten_path(item.get("path", []))
        submit_item.pop("start_pano", None)
        submit_item.pop("end_panos", None)
        submit_item.pop("nav_error", None)
        submit_item.pop("gt_path", None)
        dialog = []
        for detail in item.get("navigation_detail", []):
            if detail.get("ask"):
                dialog.append({
                    "nav_idx": detail.get("nav_idx"),
                    "question": detail.get("question"),
                    "answer": detail.get("answer"),
                    "localized_viewpoint": detail.get("localized_viewpoint"),
                    "viewpoint": detail.get("gt_viewpoint"),
                })
        submit_item["dialog"] = dialog
        submit_output.append(submit_item)
    return submit_output


def main():
    # args: <out_submit.json> <val_seen_dir> <val_unseen_dir> <test_dir>
    out_path = sys.argv[1]
    if len(sys.argv) == 5 and "--" not in sys.argv[2]:
        dirs = {
            "val_seen": [sys.argv[2]],
            "val_unseen": [sys.argv[3]],
            "test": [sys.argv[4]],
        }
    else:
        dirs = {}
        i = 2
        while i < len(sys.argv):
            if sys.argv[i].startswith("--"):
                sp = sys.argv[i][2:]
                sp = {"seen": "val_seen", "unseen": "val_unseen"}.get(sp, sp)
                dirs[sp] = [d for d in sys.argv[i + 1].split(",") if d]
                i += 2
            else:
                raise SystemExit(f"unexpected arg: {sys.argv[i]}")
    submission = {}
    for split, shards in dirs.items():
        items = load_split_shards(shards, split)
        submission[split] = make_submit_output(items)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    for split in submission:
        print(f"{split}: {len(submission[split])} episodes")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
