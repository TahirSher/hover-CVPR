#!/usr/bin/env python3
"""Video-grounded human check (make_video_sheet.py sample: 60 test clips, 12 per backbone, 45 with a ground-truth
glitch; every report of the core (H + top-1, whole clip) and of the GliDe re-run, 150 reports, system hidden, order
random). Raters WATCH the clip and score each report 0-5 against what they see (rubric in INSTRUCTIONS_RATING.md);
three raters, each clip scored by one fixed pair of them (50 reports per pair), so both systems of a clip share
their raters and within-clip differences are free of rater severity.

  consensus   mean of the two raters of a report; 'correct' if >= 3 (the judge's threshold), 'wrong' if both gave 0
  core-GliDe  per clip, mean core score minus mean GliDe score (56 clips with reports from both); sign-flip
              permutation test over clips (10,000 draws) and clip bootstrap interval (2,000); also per rater and
              without the rater who is an author (R3)
  agreement   Krippendorff's alpha (interval) over all ratings; per pair quadratic-weighted kappa and Spearman
  judges      cached ratings of both judges (no new requests) for the same reports against the ground-truth text,
              best over the clip's ground-truth glitches (a report on a glitch-free clip has none: 0); Spearman with
              the consensus, and each judge's leniency (share of reports it rates correct that the humans also do)

  python analyze_video_ratings.py   (reads human_check_video/key.csv, ratings_merged.csv; not released)
  python analyze_video_ratings.py --per_report results/v4_video_ratings_per_report.csv   (released ratings)
-> results/v4_video_ratings.csv, v4_video_ratings_tests.csv, v4_video_ratings_agree.csv,
   v4_video_ratings_per_report.csv (anonymised raters R1-R3)
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "human_check_video"))   # raw export, not released
ap.add_argument("--per_report", default="", help="read the released per-report CSV instead (no judge cache needed)")
ap.add_argument("--draws", type=int, default=10000)
a = ap.parse_args()
RATERS = {"rater_1": "R1", "rater_2": "R2", "rater_3": "R3"}   # user ids of the raw export (not released); R3 is an author
AUTHOR = "R3"
J1, J2 = "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct-AWQ"

if a.per_report:
    k = pd.read_csv(a.per_report)
else:
    k = pd.read_csv(os.path.join(a.dir, "key.csv"))
    r = pd.read_csv(os.path.join(a.dir, "ratings_merged.csv"), encoding="utf-8-sig")
    assert r.groupby("row_id").size().eq(2).all() and r.score.between(0, 5).all()
    w = r.assign(rater=r.user.map(RATERS)).pivot(index="row_id", columns="rater", values="score")
    k = k.join(w, on="row_id")
    sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
    import hover_revision as R  # noqa: E402
    H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
    spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
    ARGS = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")

    def rating(c, model, g, d):   # cached rating only, as judge_agreement.py
        j = c.judge
        key = H.sha1(json.dumps([j.name, model, j.cfg.judge_temperature, H.sha1(j.prompt), g, d]))
        p = os.path.join(os.path.dirname(j.cache_dir.rstrip("/")), model.replace("/", "_"), key[:2], key + ".json")
        if not os.path.exists(p):
            raise RuntimeError(f"judge cache miss ({model})")
        return int(H.read_json(p)["rating"])

    jr = {}
    for b in k.backbone.unique():
        R.H._PREP_MEMO.clear()
        c = R.Ctx(spec, b, ARGS)
        Hm, Gm = c.run("hover"), c.run("glide")
        for v in k[k.backbone == b]["clip"].unique():
            texts = {"core": [R.cached_hypotheses(Hm[v])[0].description] if Hm[v].get("hypotheses") else [],
                     "glide": list(Gm[v].get("bugs", []))}
            gts = [g for g in c.gt[v].get("bugs", []) if str(g).strip()]
            for _, x in k[k["clip"] == v].iterrows():
                d = next(t for t in texts[x.system] if str(t).strip() == x.report)   # text as rated by the judge
                jr[x.row_id] = [max([rating(c, m, g, d) for g in gts], default=0) for m in (J1, J2)]
    k["llama"] = k.row_id.map(lambda i: jr[i][0])
    k["qwen72b"] = k.row_id.map(lambda i: jr[i][1])
    out = k[["row_id", "clip", "game", "backbone", "system", "gt_has_glitch", "R1", "R2", "R3", "llama", "qwen72b"]]
    out.to_csv(os.path.join(HERE, "results", "v4_video_ratings_per_report.csv"), index=False)

RS = ["R1", "R2", "R3"]
k["H"] = k[RS].mean(axis=1)                                   # consensus of the report's two raters
k["Hna"] = k[[x for x in RS if x != AUTHOR]].mean(axis=1)     # without the author (every report has >= 1 other rater)
k["both0"] = k[RS].fillna(-1).apply(lambda s: (s == 0).sum() == 2, axis=1)
rng = np.random.default_rng(0)


def summary(x, col):
    return {"reports": len(x), "mean": x[col].mean(), "correct": (x[col] >= 3).mean()}


rows = []
for s in ("core", "glide"):
    for sub, cond in (("all", True), ("glitch", k.gt_has_glitch == 1), ("glitch_free", k.gt_has_glitch == 0)):
        x = k[(k.system == s) & cond]
        g = x.groupby("clip")
        rows.append({"system": s, "clips": sub, "n_clips": x["clip"].nunique(), **summary(x, "H"),
                     "wrong_both0": x.both0.mean(),
                     "clip_hit": g.H.max().ge(3).mean(),   # at least one report rated correct
                     **{f"mean_{c}": x[c].mean() for c in RS + ["Hna", "llama", "qwen72b"]},
                     **{f"correct_{c}": (x[c] >= 3).sum() / x[c].notna().sum() for c in RS + ["Hna", "llama", "qwen72b"]}})


def contrast(col, stat):
    """Per clip core minus GliDe (mean score or share >= 3) of column col, clips with both systems."""
    f = (lambda s: s.mean()) if stat == "mean" else (lambda s: s.ge(3).mean())
    d = []
    for v, x in k[k[col].notna()].groupby("clip"):
        c, g = x[x.system == "core"][col], x[x.system == "glide"][col]
        if len(c) and len(g):
            d.append(f(c) - f(g))
    d = np.array(d)
    null = np.abs((rng.choice([-1, 1], (a.draws, len(d))) * d).mean(1))
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return {"col": col, "stat": stat, "n_clips": len(d), "core_minus_glide": d.mean(),
            "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "p": (1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (1 + a.draws)}


tests = [contrast(c, s) for c in ["H", "Hna"] + RS + ["llama", "qwen72b"] for s in ("mean", "correct")]


def alpha_interval(m):
    """Krippendorff's alpha, interval metric, units x raters with NaN for missing."""
    vals = [u[~np.isnan(u)] for u in m]
    vals = [u for u in vals if len(u) >= 2]
    n = sum(len(u) for u in vals)
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in vals) / n
    allv = np.concatenate(vals)
    de = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return 1 - do / de


def kappa_w(x, y, n=6):
    O = np.zeros((n, n))
    for p, q in zip(x, y):
        O[int(p), int(q)] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


agree = [{"what": "alpha_all_raters", "n": len(k), "value": alpha_interval(k[RS].to_numpy(float))}]
for p, q in (("R1", "R2"), ("R1", "R3"), ("R2", "R3")):
    x = k[k[p].notna() & k[q].notna()]
    agree += [{"what": f"kappa_{p}_{q}", "n": len(x), "value": kappa_w(x[p], x[q])},
              {"what": f"spearman_{p}_{q}", "n": len(x), "value": spearmanr(x[p], x[q]).correlation},
              {"what": f"within1_{p}_{q}", "n": len(x), "value": (x[p] - x[q]).abs().le(1).mean()}]
for j in ("llama", "qwen72b"):
    jc, hc = k[j] >= 3, k.H >= 3
    agree += [{"what": f"spearman_{j}_H", "n": len(k), "value": spearmanr(k[j], k.H).correlation},
              {"what": f"leniency_{j}", "n": int(jc.sum()), "value": hc[jc].mean()},   # judge-correct also human-correct
              {"what": f"recall_{j}", "n": int(hc.sum()), "value": jc[hc].mean()},     # human-correct also judge-correct
              {"what": f"agree_correct_{j}", "n": len(k), "value": (jc == hc).mean()}]
for name, data in (("v4_video_ratings.csv", rows), ("v4_video_ratings_tests.csv", tests),
                   ("v4_video_ratings_agree.csv", agree)):
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in data for c in r)))
        w.writeheader()
        for r in data:
            w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in r.items()})
    print(name)
    for r in data:
        print("  ", {c: (round(v, 3) if isinstance(v, float) else v) for c, v in r.items()})
