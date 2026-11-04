#!/usr/bin/env python3
"""Human check, per method and per judge (human_check/): mean human rating and share rated correct (>= 3) per
method for each annotator and their consensus; differences between methods (two-sided permutation test over
pairs, 10,000 draws; 95% bootstrap interval); each judge's leniency against the humans (share of pairs a judge
rates correct that the consensus also rates correct); single annotator against each judge, beside A against B.
                                                     -> results/v3_human_methods.csv, v3_human_judges.csv"""
import csv
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "human_check")
k = pd.read_csv(os.path.join(D, "key.csv"))
for ann in ("A", "B"):
    s = pd.read_excel(os.path.join(D, f"human_sheet_annotator_{ann}.xlsx"), sheet_name="Annotate")
    k = k.merge(s[["id", "human_rating_0_to_5"]].rename(columns={"human_rating_0_to_5": ann}), on="id")
k["H"] = (k.A + k.B) / 2   # consensus: mean of the two annotators; correct if >= 3
rng = np.random.default_rng(0)
METHODS = ("core", "hover", "glide", "tfidf")


def kappa_w(a, b, n=6):
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


rows = []
for m in METHODS:
    x = k[k.method == m]
    rows.append({"kind": "method", "method": m, "n": len(x),
                 **{f"mean_{c}": x[c].mean() for c in ("A", "B", "H", "llama", "qwen72b")},
                 **{f"correct_{c}": (x[c] >= 3).mean() for c in ("A", "B", "H", "llama", "qwen72b")}})
for a, b in (("core", "glide"), ("hover", "glide"), ("core", "tfidf"), ("hover", "tfidf"), ("glide", "tfidf"),
             ("hover", "core")):
    for c in ("H", "A", "B", "llama", "qwen72b"):
        x, y = k[k.method == a][c].values, k[k.method == b][c].values
        d = x.mean() - y.mean()
        z = np.concatenate([x, y])
        perm = [abs(p[:len(x)].mean() - p[len(x):].mean()) for p in (rng.permutation(z) for _ in range(10000))]
        bs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(2000)]
        rows.append({"kind": "diff", "method": f"{a}-{b}", "rater": c, "n": len(x) + len(y), "diff": d,
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                     "p": (1 + sum(p >= abs(d) - 1e-12 for p in perm)) / 10001})
fields = list(dict.fromkeys(f for r in rows for f in r))
with open(os.path.join(HERE, "results", "v3_human_methods.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({q: (f"{v:.4g}" if isinstance(v, float) else v) for q, v in r.items()})

jr = []
for j in ("llama", "qwen72b"):
    for m in METHODS + ("all",):
        x = k if m == "all" else k[k.method == m]
        pos = x[x[j] >= 3]
        jr.append({"judge": j, "method": m, "n": len(x), "judge_correct": (x[j] >= 3).mean(),
                   "human_correct": (x.H >= 3).mean(), "precision_vs_human": (pos.H >= 3).mean() if len(pos) else "",
                   "recall_vs_human": ((x[j] >= 3) & (x.H >= 3)).sum() / max(1, (x.H >= 3).sum()),
                   "mean_bias": (x[j] - x.H).mean(),
                   "kappa_A": kappa_w(x.A, x[j]), "kappa_B": kappa_w(x.B, x[j]), "kappa_AB": kappa_w(x.A, x.B)})
with open(os.path.join(HERE, "results", "v3_human_judges.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(jr[0]))
    w.writeheader()
    for r in jr:
        w.writerow({q: (f"{v:.4g}" if isinstance(v, float) else v) for q, v in r.items()})
print(pd.DataFrame([r for r in rows if r["kind"] == "method"]).round(2).to_string())
print(pd.DataFrame([r for r in rows if r["kind"] == "diff"])[["method", "rater", "diff", "lo", "hi", "p"]].round(3).to_string())
print(pd.DataFrame(jr).round(2).to_string())
