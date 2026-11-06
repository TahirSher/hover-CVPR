#!/usr/bin/env python3
"""Same-session timing (paired_timing.sh): vanilla and GliDe (crux_v10.py timing) and the core (H + top-1) and HOVER
(timing_v2.py) timed back to back on one vLLM server per backbone, 20 game-stratified test clips. Per method mean,
RTF (95% game-cluster intervals) and calls/tokens; total-time ratios core/GliDe, HOVER/GliDe, core/vanilla with the
statistics code of the first session (timing_stats.py; same B and seed). Also the ratio of each method's time here
to its time in Tab. 5 on the same clips (vanilla, GliDe: data/timing/all_backbones_timing_per_video.csv; core,
HOVER: results/v2_timing_per_video.csv), as a check.

  python v4_timing_paired_stats.py      -> results/v4_timing_paired_stats.csv, results/v4_timing_paired_per_clip.csv
  python v4_timing_paired_stats.py --per_clip results/v4_timing_paired_per_clip.csv   (from the released per-clip times)
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_stats as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--glide_dir", default="/path/to/VideoGlitchBench/paired_timing_v4/v10/timing")
ap.add_argument("--ours", default=os.path.join(HERE, "results", "v4_timing_paired.csv"))
ap.add_argument("--first", default=os.path.join(HERE, "..", "HOVER_CVPR2027", "data", "timing",
                                                "all_backbones_timing_per_video.csv"))   # vanilla, GliDe
ap.add_argument("--per_clip", default="", help="read all four methods from this per-clip CSV instead")
ap.add_argument("--first_ours", default=os.path.join(HERE, "results", "v2_timing_per_video.csv"))   # core, HOVER
a = ap.parse_args()
cfg = P.Config()
BACKBONES = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "internvl2.5-8b", "ui-tars-1.5-7b"]
NAME = {"glide": "glide", "vanilla": "vanilla", "h_top1": "core", "hover": "hover"}


def row(vid, game, dur, r):
    out = {"vid": vid, "game": game, "duration": float(dur), "n_repeats": 1, "memo_hits": float(r.get("memo_hits", 0)),
           "request_failures": 0.0, "stage_model_s": {}}
    for k in ("wall_s", "preprocess_s", "model_s", "calls", "model_calls", "prompt_tokens", "completion_tokens"):
        out[k] = float(r.get(k) or 0.0)
    return out


pv = defaultdict(lambda: defaultdict(dict))
bad = defaultdict(int)
for r in (csv.DictReader(open(a.per_clip)) if a.per_clip else []):
    pv[r["backbone"]][r["method"]][r["vid"]] = row(r["vid"], r["game"], r["duration_s"], r)
for b in ([] if a.per_clip else BACKBONES):
    d = json.load(open(os.path.join(a.glide_dir, b, "timing_test_20.json")))
    for e in d["entries"]:
        if not e.get("ok", True):
            bad[(b, e["method"])] += 1
            continue
        pv[b][NAME[e["method"]]][e["vid"]] = row(e["vid"], e["game"], e["duration"], e)
for r in ([] if a.per_clip else csv.DictReader(open(a.ours))):
    pv[r["backbone"]][NAME[r["method"]]][r["vid"]] = row(r["vid"], r["game"], r["duration"], r)
first = defaultdict(dict)
for r in csv.DictReader(open(a.first)):
    if r["method"] in ("vanilla", "glide"):
        first[(r["backbone"], r["method"])][r["vid"]] = float(r["wall_s"])
for r in csv.DictReader(open(a.first_ours)):
    first[(r["backbone"], NAME[r["method"]])][r["vid"]] = float(r["wall_s"])

out = []
for b in BACKBONES:
    for m in ("vanilla", "glide", "core", "hover"):
        rows = list(pv[b][m].values())
        s = P.summarise_times(rows, cfg.n_bootstrap, cfg.seed)
        old = first.get((b, m), {})
        common = [v for v in pv[b][m] if v in old]
        out.append({"backbone": b, "kind": "method", "method": m, "n": s["n_videos"], "n_games": s["n_games"],
                    "failed": bad[(b, m)], "mean_s": s["mean_s"], "mean_lo": s["mean_ci"][0],
                    "mean_hi": s["mean_ci"][1], "rtf": s["rtf"], "rtf_lo": s["rtf_ci"][0], "rtf_hi": s["rtf_ci"][1],
                    "calls": s["requests_mean"], "prompt_tokens": s["prompt_tokens_mean"],
                    "completion_tokens": s["completion_tokens_mean"],
                    "n_first": len(common),
                    "vs_first": (sum(pv[b][m][v]["wall_s"] for v in common) / sum(old[v] for v in common))
                    if common else float("nan")})
    for x, y in (("core", "glide"), ("hover", "glide"), ("core", "vanilla"), ("hover", "vanilla")):
        c = P.paired_time_contrast(pv[b][x], pv[b][y], cfg)
        out.append({"backbone": b, "kind": "contrast", "method": f"{x}/{y}", "n": c["n_videos"], "n_games": c["n_games"],
                    "ratio": c["total_time_ratio"], "ratio_lo": c["total_time_ratio_ci"][0],
                    "ratio_hi": c["total_time_ratio_ci"][1], "geo_ratio": c["geo_mean_ratio"],
                    "frac_faster": c["fraction_videos_faster"]})
PER = ("wall_s", "preprocess_s", "model_s", "model_calls", "prompt_tokens", "completion_tokens")
with open(os.path.join(HERE, "results", "v4_timing_paired_per_clip.csv"), "w", newline="") as f:   # all four methods
    w = csv.writer(f)
    w.writerow(["backbone", "method", "vid", "game", "duration_s", *PER])
    for b in BACKBONES:
        for m in ("vanilla", "glide", "core", "hover"):
            for v, r in sorted(pv[b][m].items()):
                w.writerow([b, m, v, r["game"], r["duration"], *("%.12g" % r[k] for k in PER)])
fields = list(dict.fromkeys(k for r in out for k in r))
with open(os.path.join(HERE, "results", "v4_timing_paired_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in out:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in out:
    print(r["backbone"][:14].ljust(15), r["method"].ljust(14),
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("backbone", "kind", "method")})
