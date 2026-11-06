#!/usr/bin/env python3
"""Intervals and paired contrasts of the first timing session for every backbone, from the per-clip logs
(data/timing/all_backbones_timing_per_video.csv), with the statistics code that produced report_test_100.md
(summarise_times, paired_time_contrast in timing_stats.py, copied verbatim; same B and seed). The report lacks the HOVER
rows of InternVL2.5-4B/8B and UI-TARS-1.5-7B; this fills them, and re-derives the report's other rows as a check.

  python v4_timing_ci.py --per_video <csv>                                    -> results/v4_timing_ci.csv
"""
import argparse
import csv
import itertools
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--per_video", default=os.path.join(HERE, "..", "HOVER_CVPR2027", "data", "timing",
                                                    "all_backbones_timing_per_video.csv"))
a = ap.parse_args()
import sys  # noqa: E402
sys.path.insert(0, HERE)
import timing_stats as P  # noqa: E402
cfg = P.Config()   # n_bootstrap, seed, permutation settings as in the report

pv = defaultdict(dict)
for r in csv.DictReader(open(a.per_video)):
    row = {"vid": r["vid"], "game": r["game"], "duration": float(r["duration_s"]), "n_repeats": 1}
    for k in ("wall_s", "preprocess_s", "model_s", "model_calls", "prompt_tokens", "completion_tokens"):
        row[k] = float(r[k])
    row.update({"calls": row["model_calls"], "memo_hits": 0.0, "request_failures": 0.0, "stage_model_s": {}})
    pv[r["backbone"]][r["method"]] = {**pv[r["backbone"]].get(r["method"], {}), r["vid"]: row}

out = []
for b, meths in pv.items():
    for m, rows in meths.items():
        s = P.summarise_times(list(rows.values()), cfg.n_bootstrap, cfg.seed)
        out.append({"backbone": b, "kind": "method", "method": m, "n": s["n_videos"], "mean_s": s["mean_s"],
                    "mean_lo": s["mean_ci"][0], "mean_hi": s["mean_ci"][1], "rtf": s["rtf"],
                    "rtf_lo": s["rtf_ci"][0], "rtf_hi": s["rtf_ci"][1]})
    for x, y in itertools.combinations(sorted(meths, key=["hover", "glide", "vanilla"].index), 2):
        c = P.paired_time_contrast(meths[x], meths[y], cfg)
        out.append({"backbone": b, "kind": "contrast", "method": f"{x}/{y}", "n": c.get("n_videos", c.get("n")),
                    **{k: v for k, v in c.items() if isinstance(v, (int, float))},
                    **{f"{k}_{kk}": vv for k, v in c.items() if isinstance(v, dict)
                       for kk, vv in v.items() if isinstance(vv, (int, float))},
                    **{f"{k}_lo": v[0] for k, v in c.items() if isinstance(v, (list, tuple)) and len(v) == 2},
                    **{f"{k}_hi": v[1] for k, v in c.items() if isinstance(v, (list, tuple)) and len(v) == 2}})
fields = list(dict.fromkeys(k for r in out for k in r))
with open(os.path.join(HERE, "results", "v4_timing_ci.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in out:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in out:
    print(r["backbone"], r["kind"], r["method"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                                  if k not in ("backbone", "kind", "method")})
