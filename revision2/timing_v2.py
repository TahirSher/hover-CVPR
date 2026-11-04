#!/usr/bin/env python3
"""Re-time the SCORED configuration of every backbone (main Tab. 1: calibration and refine mode of
the main run) on the 100-clip timing subset of Tab. 7, with per-clip logs, plus 'H + top-1 + whole
clip' (stage H only). Same protocol as Tab. 7: wall clock from the raw .mp4 to the final report,
sequential requests, an empty work directory (no cross-run cache), two untimed warm-up clips.

  python timing_v2.py --backbone internvl2.5-8b --methods hover,h_top1
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--backbone", required=True)
ap.add_argument("--methods", default="hover,h_top1")
ap.add_argument("--subset", default="/path/to/VideoGlitchBench/crux_v8_work/v10/timing/subset_test_100.json")
ap.add_argument("--scratch", default="HOVER_v2_timing_work")
ap.add_argument("--out", default=os.path.join(HERE, "results", "v2_timing_per_video.csv"))
a = ap.parse_args()

H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
ctx = R.Ctx(R.read_runs(a.runs), a.backbone,
            argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants=""))
sub = json.load(open(a.subset))
vids = list(sub["vids"])
by_vid = {r.vid: r for r in ctx.recs}
warm = sorted(ctx.splits["dev"], key=lambda r: r.vid)[:2]
fields = ["backbone", "method", "mode", "vid", "game", "duration", "wall_s", "preprocess_s", "model_s",
          "calls", "model_calls", "prompt_tokens", "completion_tokens", "n_reports"]
new = not os.path.exists(a.out)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fout = open(a.out, "a", newline="")
w = csv.DictWriter(fout, fieldnames=fields)
if new:
    w.writeheader()

for method in a.methods.split(","):
    base = ctx.hover_cfg() if method == "hover" else ctx.cfg
    sw = {"use_hypothesis_scoring": False} if method == "h_top1" else {}
    work = f"{a.scratch}/{a.backbone}_{method}"
    cfg = H.Config(**{**asdict(base), **sw, "work_subdir": work})
    shutil.rmtree(cfg.paths()["work"], ignore_errors=True)
    H._PREP_MEMO.clear()
    vlm = H.make_vlm(cfg)
    try:
        import requests
        requests.post(cfg.api_base.replace("/v1", "") + "/reset_prefix_cache", timeout=10)
    except Exception:
        pass

    def one(r):
        b0 = asdict(vlm.budget)
        t0 = time.perf_counter()
        wins = H.prepare_split([r], cfg)[r.vid]
        t1 = time.perf_counter()
        if method == "hover":
            o = H.run_HOVER(vlm, r, wins, cfg)
            n = len(o.get("bugs", []))
        else:
            vh = H.collect_video(vlm, r, wins, cfg)
            n = 1 if vh.hyps else 0          # top-1 hypothesis, whole-clip span: no further calls
        t2 = time.perf_counter()
        b1 = asdict(vlm.budget)
        return {"wall_s": t2 - t0, "preprocess_s": t1 - t0, "model_s": t2 - t1,
                "calls": b1["calls"] - b0["calls"], "model_calls": b1["model_calls"] - b0["model_calls"],
                "prompt_tokens": b1["prompt_tokens"] - b0["prompt_tokens"],
                "completion_tokens": b1["completion_tokens"] - b0["completion_tokens"], "n_reports": n}

    for r in warm:
        one(r)
    t_all = time.time()
    for i, v in enumerate(vids):
        r = by_vid[v]
        res = one(r)
        w.writerow({"backbone": a.backbone, "method": method, "mode": cfg.refine_mode if method == "hover" else "whole",
                    "vid": v, "game": r.game, "duration": r.duration, **res})
        fout.flush()
        if (i + 1) % 10 == 0:
            print(f"{a.backbone} {method} {i + 1}/{len(vids)} | {(time.time() - t_all) / (i + 1):.1f} s/clip", flush=True)
    shutil.rmtree(cfg.paths()["work"], ignore_errors=True)
fout.close()
