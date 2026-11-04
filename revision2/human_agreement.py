#!/usr/bin/env python3
"""Agreement of two blinded human annotators with each other and with both judges (human_check/).
Quadratic-weighted kappa, Spearman, and agreement on 'correct' (rating >= 3), overall and per method;
95% intervals from a pair bootstrap (B = 2000).                          -> results/v3_human_agreement.csv"""
import csv
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "human_check")
key = {r["id"]: r for r in csv.DictReader(open(os.path.join(D, "key.csv")))}
hum = {}
for ann in ("A", "B"):
    xlsx = os.path.join(D, f"human_sheet_annotator_{ann}.xlsx")
    if os.path.exists(xlsx):   # returned workbook (sheet 'Annotate'); the CSV is the blank template
        import pandas as pd
        sheet = pd.read_excel(xlsx, sheet_name="Annotate").fillna("")
        recs = [{"id": str(int(r["id"])), "human_rating_0_to_5": str(r["human_rating_0_to_5"])} for _, r in sheet.iterrows()]
    else:
        recs = csv.DictReader(open(os.path.join(D, f"human_sheet_annotator_{ann}.csv")))
    for r in recs:
        v = r["human_rating_0_to_5"].strip()
        if v != "":
            hum.setdefault(r["id"], {})[ann] = int(float(v))
ids = sorted((i for i in key if len(hum.get(i, {})) == 2), key=int)
print(len(ids), "pairs rated by both annotators")


def kappa_w(a, b, k=6):
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def stats(a, b, rng, B=2000):
    a, b = np.asarray(a), np.asarray(b)
    f = lambda x, y: (kappa_w(x, y), spearmanr(x, y).correlation, float(np.mean((x >= 3) == (y >= 3))))
    est = f(a, b)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        bs.append(f(a[i], b[i]))
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    return est, lo, hi


rng = np.random.default_rng(0)
rows = []
for subset in ("all", "hover", "core", "glide", "tfidf"):
    S = [i for i in ids if subset == "all" or key[i]["method"] == subset]
    A = [hum[i]["A"] for i in S]
    Bv = [hum[i]["B"] for i in S]
    M = [round((x + y) / 2 + 1e-9) for x, y in zip(A, Bv)]   # human consensus, half up
    J1 = [int(key[i]["llama"]) for i in S]
    J2 = [int(key[i]["qwen72b"]) for i in S]
    for name, x, y in (("A_vs_B", A, Bv), ("human_vs_llama8b", M, J1), ("human_vs_qwen72b", M, J2),
                       ("llama8b_vs_qwen72b", J1, J2)):
        (k, s, c), lo, hi = stats(x, y, rng)
        rows.append({"subset": subset, "pair": name, "n": len(S), "kappa_w": k, "kappa_lo": lo[0], "kappa_hi": hi[0],
                     "spearman": s, "agree_ge3": c, "mean_x": float(np.mean(x)), "mean_y": float(np.mean(y))})
        print(subset, name, len(S), "kappa %.2f [%.2f, %.2f]  rho %.2f  agree>=3 %.2f  means %.2f/%.2f"
              % (k, lo[0], hi[0], s, c, np.mean(x), np.mean(y)))
with open(os.path.join(HERE, "results", "v3_human_agreement.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
