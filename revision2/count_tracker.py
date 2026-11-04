#!/usr/bin/env python3
"""Count the outcomes of GliDe's object_tracking (SAM 3) tool calls in the per-clip logs of each re-run."""
import csv, glob, os, re, sys
GLIDE = sys.argv[1] if len(sys.argv) > 1 else "/path/to/GliDe"
OUT = sys.argv[2] if len(sys.argv) > 2 else "glide_tracker_counts.csv"
RUNS = {"qwen2.5-vl-3b": "data_qwen2.5-vl-3b_sam3", "qwen2.5-vl-7b": "data_qwen2.5-vl-7b_sam3",
        "internvl2.5-4b": "data_internvl2.5-4b_sam3", "internvl2.5-8b": "data_internvl2.5-8b_sam3",
        "ui-tars-1.5-7b": "data_ui-tars-1.5-7b_sam3", "qwen2.5-vl-3b (no SAM 3)": "data_qwen2.5-vl-3b"}
KINDS = [("success", "[Executor/object_tracking] success"), ("no_match", "found no objects"),
         ("oom", "FAILED: CUDA out of memory"), ("frame_loading", "FAILED: Failure in frame loading"),
         ("not_initialised", "tracker could not be initialized")]
rows = []
for name, d in RUNS.items():
    c = {k: 0 for k, _ in KINDS}
    planned = 0
    for p in glob.glob(os.path.join(GLIDE, d, "logs", "*.log")):
        for line in open(p, errors="ignore"):
            if "[Planner] → object_tracking" in line:
                planned += 1
            if "[Executor/object_tracking]" in line:
                for k, pat in KINDS:
                    if pat in line:
                        c[k] += 1
    rows.append({"run": name, "dir": d, "tracking_calls": planned, **c})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows: print(r)
