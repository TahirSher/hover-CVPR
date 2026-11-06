#!/usr/bin/env python3
"""Are the 24 games of our development and test splits representative of the benchmark? Annotation statistics
of the 12 development games, the 12 test games and the 96 games never used: clips, duration and genre
(annotations.json), glitch-free and multi-glitch shares and glitch coverage of the clip from the ground truth
the evaluator scores (groundtruth_gilde.json, which omits reports without a time span), splits/*.json.

  python v4_split_stats.py [--root /path/to/VideoGlitchBench]       -> results/v4_splitstats.csv
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/path/to/VideoGlitchBench")
a = ap.parse_args()
ann = json.load(open(os.path.join(a.root, "annotations.json")))
gt = {r["video_id"]: r for r in json.load(open(os.path.join(a.root, "groundtruth_gilde.json")))}
split = {}
for s in ("train", "dev", "test"):
    for v in json.load(open(os.path.join(a.root, "splits", f"{s}.json"))):
        split[v] = {"train": "unused_96_games", "dev": "dev_12_games", "test": "test_12_games"}[s]


def coverage(r, dur):
    """Share of the clip covered by the union of all ground-truth intervals."""
    iv = sorted((max(0.0, s), min(dur, e)) for sp in gt[r["video_id"]].get("time_nodes", []) for s, e in sp if e > s)
    tot, cur = 0.0, None
    for s, e in iv:
        if cur and s <= cur[1]:
            cur[1] = max(cur[1], e)
        else:
            if cur:
                tot += cur[1] - cur[0]
            cur = [s, e]
    if cur:
        tot += cur[1] - cur[0]
    return tot / dur if dur > 0 else float("nan")


rows = []
for grp in ("dev_12_games", "test_12_games", "unused_96_games", "all"):
    rs = [r for r in ann if grp == "all" or split.get(r["video_id"]) == grp]
    n_rep = [len([x for x in gt[r["video_id"]].get("bugs", []) if str(x).strip()]) for r in rs]
    cov = [coverage(r, float(r["duration_sec"])) for r, k in zip(rs, n_rep) if k > 0]
    gen = Counter(r.get("genre", "") for r in rs)
    rows.append({"group": grp, "n_games": len({r["game"] for r in rs}), "n_clips": len(rs),
                 "mean_duration_s": float(np.mean([float(r["duration_sec"]) for r in rs])),
                 "glitch_free_share": float(np.mean([k == 0 for k in n_rep])),
                 "multi_share": float(np.mean([k >= 2 for k in n_rep])),
                 "glitches_per_clip": float(np.mean(n_rep)),
                 "mean_coverage": float(np.mean(cov)), "short_share_lt50": float(np.mean([c < 0.5 for c in cov])),
                 "top_genres": "; ".join(f"{g} {c / len(rs):.0%}" for g, c in gen.most_common(3))})
    print(rows[-1])
with open(os.path.join(HERE, "results", "v4_splitstats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
