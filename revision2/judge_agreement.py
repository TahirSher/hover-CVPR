#!/usr/bin/env python3
"""Agreement of the two judges on every (ground truth, prediction) pair rated for the test split
(HOVER, H + top-1, GliDe re-run, vanilla; five backbones)                 -> v2_judge_agreement.csv"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
ARGS = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
J1, J2 = "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct-AWQ"


def rating(c, model, g, d):
    j = c.judge
    key = H.sha1(json.dumps([j.name, model, j.cfg.judge_temperature, H.sha1(j.prompt), g, d]))
    p = os.path.join(os.path.dirname(j.cache_dir.rstrip("/")), model.replace("/", "_"), key[:2], key + ".json")
    if not os.path.exists(p):
        return None
    x = H.read_json(p)
    return None if not str(x.get("raw", "")).strip() else int(x["rating"])


def kappa_w(a, b, k=6):
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


pairs, missing = set(), 0
for b in ("qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "internvl2.5-8b", "ui-tars-1.5-7b"):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm, Vm = c.run("hover"), c.run("glide"), c.run("vanilla")
    descs = {}
    for v in c.gt:
        ds = []
        if v in Hm:
            ds += list(Hm[v].get("bugs", []))
            hy = R.cached_hypotheses(Hm[v])
            ds += [hy[0].description] if hy else []
        for run in (Gm, Vm):
            if v in run:
                ds += list(run[v].get("bugs", []))
        for d in set(ds):
            for g in c.gt[v].get("bugs", []):
                pairs.add((g, d))
    judge_ctx = c
rows = []
a, b2 = [], []
for g, d in pairs:
    r1, r2 = rating(judge_ctx, J1, g, d), rating(judge_ctx, J2, g, d)
    if r1 is None or r2 is None:
        missing += 1
        continue
    a.append(r1); b2.append(r2)
a, b2 = np.array(a), np.array(b2)
out = {"pairs": len(a), "missing_or_failed": missing, "spearman": spearmanr(a, b2).correlation,
       "kappa_quadratic": kappa_w(a, b2), "mean_j1": a.mean(), "mean_j2": b2.mean(),
       "agree_correct_ge3": float(np.mean((a >= 3) == (b2 >= 3))),
       "share_ge3_j1": float(np.mean(a >= 3)), "share_ge3_j2": float(np.mean(b2 >= 3)),
       "j2_le_j1": float(np.mean(b2 <= a))}
print(out)
with open(os.path.join(HERE, "results", "v2_judge_agreement.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out))
    w.writeheader()
    w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in out.items()})
