#!/usr/bin/env python3
"""Reproduce the K80 local validation metrics from a merged submission."""

import argparse
import json
import os

import numpy as np

from evaluator import Evaluator


def flatten_path(path):
    flat = []
    for step in path:
        if isinstance(step, list):
            flat.extend(flatten_path(step))
        else:
            flat.append(step)
    return flat


def episode_score(dtc, dtc_gt, nsc_gt, success):
    denom = nsc_gt - dtc_gt
    if denom <= 0:
        efficiency = 0.0
    else:
        efficiency = 1.0 - min(max(dtc - dtc_gt, 0) / denom, 1.0)
    return efficiency * float(success)


def load_gt(root, split):
    path = os.path.join(root, "assets", "base", "dataset", "RAIN_holistic", f"{split}.json")
    with open(path) as f:
        return {str(item["instr_id"]): item for item in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", required=True)
    ap.add_argument("--splits", default="test,val_unseen,val_seen")
    args = ap.parse_args()

    root = os.environ.get(
        "DIALNAV_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    with open(args.submit) as f:
        submit = json.load(f)

    all_scans = sorted({
        item.get("scan")
        for split_items in submit.values()
        for item in split_items
        if item.get("scan")
    })
    connectivity = os.path.join(root, "assets", "base", "dataset", "connectivity")
    evaluator = Evaluator(connectivity, all_scans)

    for split in args.splits.split(","):
        gt_map = load_gt(root, split)
        items = submit.get(split, [])
        sr, oracle, dtc, score, steps, spl, nav_error = [], [], [], [], [], [], []
        for item in items:
            gt = gt_map.get(str(item.get("instr_id")))
            if gt is None:
                continue
            path = flatten_path(item.get("path", []))
            scan = item.get("scan")
            final = path[-1] if path else None
            end_panos = gt.get("end_panos", [])
            success = 1.0 if final in end_panos else 0.0
            oracle_success = 1.0 if any(p in end_panos for p in path) else 0.0
            dialogs = item.get("dialog", []) or []
            turn_count = len(dialogs)
            nsc_gt = len(gt.get("nav_trajectory", []))
            dtc_gt = len(gt.get("dialog", []))
            sr.append(success)
            oracle.append(oracle_success)
            dtc.append(turn_count)
            score.append(episode_score(turn_count, dtc_gt, nsc_gt, success))
            if scan and len(path) >= 2:
                dist = evaluator.shortest_distances[scan]
                traj_len = float(np.sum([
                    dist[path[i]][path[i + 1]]
                    for i in range(len(path) - 1)
                ]))
                start = path[0]
                shortest_distance = float(np.min([
                    dist[start][end]
                    for end in end_panos
                ]))
                spl.append(
                    success
                    * shortest_distance
                    / max(traj_len, shortest_distance, 0.01)
                )
                nav_error.append(float(np.min([
                    dist[final][end]
                    for end in end_panos
                ])))
            else:
                spl.append(success)
                nav_error.append(float("nan"))
            steps.append(max(len(path) - 1, 0))

        n = len(sr)
        if n == 0:
            print(f"{split}: empty")
            continue
        if split == "test":
            print(
                f"{split}: n={n} "
                f"SR={100 * np.mean(sr):.2f} "
                f"oracle_SR={100 * np.mean(oracle):.2f} "
                f"DTC={np.mean(dtc):.2f} "
                f"SPL={100 * np.mean(spl):.2f} "
                f"steps={np.mean(steps):.2f} "
                f"nav_error={np.nanmean(nav_error):.2f} "
                "score=offline"
            )
        else:
            print(
                f"{split}: n={n} "
                f"SR={100 * np.mean(sr):.2f} "
                f"oracle_SR={100 * np.mean(oracle):.2f} "
                f"DTC={np.mean(dtc):.2f} "
                f"SPL={100 * np.mean(spl):.2f} "
                f"steps={np.mean(steps):.2f} "
                f"nav_error={np.nanmean(nav_error):.2f} "
                f"score={np.mean(score):.4f}"
            )


if __name__ == "__main__":
    main()
