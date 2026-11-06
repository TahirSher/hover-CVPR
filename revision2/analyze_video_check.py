#!/usr/bin/env python3
"""Results of the video-grounded human check (make_video_sheet.py), once both annotators returned their sheets.

Per system (core = H + top-1 whole clip, GliDe re-run), from both annotators and their consensus (mean):
  visible      share of reports whose content is visible in the clip (rating 2; 'partly' = 1 reported apart)
  hallucinated share of reports rated 0 (not in the clip / normal gameplay called a glitch)
  main         share of reports describing the clip's main glitch (rating 2), on clips with a ground-truth glitch
  clip hit     share of glitch clips where at least one report of the system describes the main glitch
  false alarm  reports on clips that both annotators judge glitch-free
Differences core - GliDe with a permutation test over clips (system labels swapped within clip, 10,000 draws)
and a clip bootstrap interval; quadratic-weighted kappa between annotators; annotators' clip labels vs ground truth.

  python analyze_video_check.py [--dir human_check_video]          -> results/v4_video_check*.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "human_check_video"))
ap.add_argument("--draws", type=int, default=10000)
a = ap.parse_args()
key = {int(r["row_id"]): r for r in csv.DictReader(open(os.path.join(a.dir, "key.csv")))}
rep, clip = {}, {}
for ann in ("A", "B"):
    x = os.path.join(a.dir, f"annotator_{ann}.xlsx")
    s = pd.read_excel(x, sheet_name="Reports")
    for _, r in s.iterrows():
        if pd.notna(r["visible_0_1_2"]) and pd.notna(r["main_glitch_0_1_2"]):
            rep.setdefault(int(r["row_id"]), {})[ann] = (int(r["visible_0_1_2"]), int(r["main_glitch_0_1_2"]))
    c = pd.read_excel(x, sheet_name="Clips")
    rs = pd.read_excel(x, sheet_name="Reports")
    vf2clip = {key[int(r["row_id"])]["video_file"]: key[int(r["row_id"])]["clip"] for _, r in rs.iterrows()}
    for _, r in c.iterrows():
        if pd.notna(r["contains_glitch_0_1"]):
            clip.setdefault(vf2clip[r["video_file"]], {})[ann] = int(r["contains_glitch_0_1"])
ids = sorted(i for i in key if len(rep.get(i, {})) == 2)
print(len(ids), "of", len(key), "reports rated by both annotators;", sum(len(v) == 2 for v in clip.values()), "clips")


def kappa_w(x, y, k):
    O = np.zeros((k, k))
    for p, q in zip(x, y):
        O[p, q] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def table(rater):
    """Per clip and system: lists of (visible, main) ratings of rater 'A', 'B' or 'H' (mean)."""
    per = defaultdict(lambda: defaultdict(list))
    for i in ids:
        k = key[i]
        v = rep[i]
        val = (np.mean([v["A"][0], v["B"][0]]), np.mean([v["A"][1], v["B"][1]])) if rater == "H" else v[rater]
        per[k["clip"]][k["system"]].append(val)
    return per


clean_by_humans = {c for c, v in clip.items() if len(v) == 2 and v["A"] == 0 and v["B"] == 0}
gt_glitch = {key[i]["clip"]: key[i]["gt_has_glitch"] == "1" for i in key}


def metrics(per, sysname, clips):
    reps = [x for c in clips for x in per[c].get(sysname, [])]
    g = [c for c in clips if gt_glitch[c]]
    greps = [x for c in g for x in per[c].get(sysname, [])]
    return {"reports": len(reps),
            "visible": float(np.mean([x[0] >= 2 for x in reps])) if reps else float("nan"),
            "partly_or_visible": float(np.mean([x[0] >= 1 for x in reps])) if reps else float("nan"),
            "hallucinated": float(np.mean([x[0] < 0.5 for x in reps])) if reps else float("nan"),
            "main": float(np.mean([x[1] >= 2 for x in greps])) if greps else float("nan"),
            "clip_hit": float(np.mean([any(x[1] >= 2 for x in per[c].get(sysname, [])) for c in g])) if g else float("nan"),
            "false_alarm_reports": sum(len(per[c].get(sysname, [])) for c in clips if c in clean_by_humans)}


rows, tests = [], []
clips_all = sorted({key[i]["clip"] for i in ids})
rng = np.random.default_rng(0)
for rater in ("A", "B", "H"):
    per = table(rater)
    m = {s: metrics(per, s, clips_all) for s in ("core", "glide")}
    for s in m:
        rows.append({"rater": rater, "system": s, **m[s]})
    for stat in ("visible", "hallucinated", "main", "clip_hit"):
        obs = m["core"][stat] - m["glide"][stat]
        null, boot = [], []
        for _ in range(a.draws):   # swap system labels within each clip
            pp = {c: ({"core": per[c].get("glide", []), "glide": per[c].get("core", [])} if rng.random() < 0.5
                      else per[c]) for c in clips_all}
            null.append(metrics(pp, "core", clips_all)[stat] - metrics(pp, "glide", clips_all)[stat])
        for _ in range(2000):
            bc = list(rng.choice(clips_all, len(clips_all)))
            pb = {f"{c}#{j}": per[c] for j, c in enumerate(bc)}
            for j, c in enumerate(bc):
                gt_glitch[f"{c}#{j}"] = gt_glitch[c]
            boot.append(metrics(pb, "core", list(pb))[stat] - metrics(pb, "glide", list(pb))[stat])
        null = np.array(null)
        tests.append({"rater": rater, "stat": stat, "core_minus_glide": obs,
                      "lo": float(np.nanpercentile(boot, 2.5)), "hi": float(np.nanpercentile(boot, 97.5)),
                      "p": float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))})
agree = {"n_reports": len(ids),
         "kappa_visible": kappa_w([rep[i]["A"][0] for i in ids], [rep[i]["B"][0] for i in ids], 3),
         "kappa_main": kappa_w([rep[i]["A"][1] for i in ids], [rep[i]["B"][1] for i in ids], 3),
         "clip_label_agree_AB": float(np.mean([v["A"] == v["B"] for v in clip.values() if len(v) == 2])),
         "clip_label_agree_gt_A": float(np.mean([v["A"] == int(gt_glitch[c]) for c, v in clip.items() if "A" in v])),
         "clip_label_agree_gt_B": float(np.mean([v["B"] == int(gt_glitch[c]) for c, v in clip.items() if "B" in v])),
         "clips_glitch_free_by_both": len(clean_by_humans)}
for name, data in (("v4_video_check.csv", rows), ("v4_video_check_tests.csv", tests), ("v4_video_check_agree.csv", [agree])):
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        for r in data:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(name, *data, sep="\n  ")
