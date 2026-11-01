#!/usr/bin/env python3
"""Rebuild data/profile_examples_qwen7b.json (Supp. Fig. S9) from the MAIN run of Qwen2.5-VL-7B."""
import json, sys
R = "/path/to/VideoGlitchBench/HOVER_v8_work/v10"
run = json.load(open(f"{R}/runs/qwen2.5-vl-7b/HOVER_window_test.json"))
cal = json.load(open(f"{R}/params/qwen2.5-vl-7b/calibration_window.json"))
old = json.load(open(sys.argv[1]))
out = {}
for o in run["outputs"]:
    if o["video_name"] not in old:
        continue
    hyps = [{k: h[k] for k in ("description", "entity", "profile", "prior", "support", "utility", "members")}
            for h in o["hypotheses"]]
    reps = [{"description": d, "spans": sp} for d, sp in zip(o["bugs"], o["time_nodes"])]
    out[o["video_name"]] = {"game": o["game"], "duration": o["duration"], "n_windows": o["n_windows"],
                            "hypotheses": hyps, "reports": reps, "tau": cal["seg_tau"],
                            "lam": cal["seg_penalty"], "thr": cal["decision_threshold"],
                            "refine_mode": "window", "run": "main"}
assert set(out) == set(old), set(old) - set(out)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("rebuilt", sorted(out))
