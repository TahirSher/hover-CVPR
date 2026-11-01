#!/usr/bin/env python3
"""Selection effect of the development-medoid control (hover_revision.py 'controls', E3).

1. Scores every sentence of the judge short-list (the --medoid_shortlist most TF-IDF-central
   development descriptions) on the test split as a fixed whole-clip prediction, next to its
   development mean rating -> revision_medoid_candidates.csv.
2. Re-runs the E3 fixed-sentence block on every backbone with a fifth sentence, the TF-IDF
   development medoid (chosen without the judge), so Holm covers five sentences
   -> replaces the E3_constant rows of revision_controls.csv.

  python medoid_candidates.py --runs runs.json --out revision_results
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs.json")
ap.add_argument("--out", default="revision_results")
ap.add_argument("--medoid_shortlist", type=int, default=40)
ap.add_argument("--backbones", default=",".join(R.BACKBONES))
a = ap.parse_args()

R.load_hover(os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOVER.py"))
spec = R.read_runs(a.runs)
args = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=a.medoid_shortlist, game_pool="loo",
                          ablate_variants="")
bks = [b for b in a.backbones.split(",") if b]

# ---- 1. candidates (backbone-independent: ground truth, dev texts and judge are shared)
ctx = R.Ctx(spec, bks[0], args)
dev_texts = list(dict.fromkeys(d for r in ctx.splits["dev"] for d in r.bugs if str(d).strip()))
order = [int(i) for i in np.argsort(-R._tfidf_centrality(dev_texts), kind="mergesort")[:a.medoid_shortlist]]
R.prefetch_ratings(ctx, [(dev_texts[j], dev_texts[i]) for i in order for j in range(len(dev_texts)) if j != i])
dev_mean = {i: float(np.mean([ctx.judge.rate(dev_texts[j], dev_texts[i])
                              for j in range(len(dev_texts)) if j != i])) for i in order}
judge_pick = max(order, key=lambda i: (dev_mean[i], -order.index(i)))
tfidf_pick = order[0]
names = sorted(set(ctx.by_name) & set(ctx.gt) & set(ctx.run("hover")))
rows = []
for rank, i in enumerate(order):
    preds = {v: R.pred([dev_texts[i]], [R.whole(ctx, v)]) for v in names}
    s = R.summarise(R.evaluate(ctx, preds, names))
    rows.append({"rank_tfidf": rank, "dev_mean_rating": round(dev_mean[i], 4),
                 "test_f1_iou": round(s["f1_iou"], 6), "test_mean_score": round(s["mean_score"], 4),
                 "judge_pick": int(i == judge_pick), "tfidf_pick": int(i == tfidf_pick),
                 "sentence": dev_texts[i]})
    print(f"  {rank:2d} dev {dev_mean[i]:.3f} test {100 * s['f1_iou']:.2f}  {dev_texts[i][:70]}")
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "revision_medoid_candidates.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
t = np.array([r["test_f1_iou"] for r in rows]) * 100
print(f"test F1xIoU over {len(t)} candidates: median {np.median(t):.2f}, "
      f"IQR {np.percentile(t, 25):.2f}-{np.percentile(t, 75):.2f}, range {t.min():.2f}-{t.max():.2f}")
print(f"judge pick: {dev_texts[judge_pick][:80]}\nTF-IDF pick: {dev_texts[tfidf_pick][:80]}")

# ---- 2. E3 block with five sentences, paired against HOVER of the main run on every backbone
out_rows = []
for b in bks:
    # HOVER.py memoises prepared windows per process; a second Ctx would then take clip durations
    # from the last window instead of the manifest. Clear it so every backbone sees the manifest
    # durations, as in the one-backbone-per-process 'controls' runs.
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, args)
    Hm = c.run("hover")
    hover_preds = {v: R.pred(o.get("bugs", []), R.spans_of(o)) for v, o in Hm.items()}
    names3 = sorted(set(c.by_name) & set(c.gt) & set(Hm))
    by_game_dev = defaultdict(list)
    for r in c.splits["dev"]:
        by_game_dev[r.game].extend([d for d in r.bugs if str(d).strip()])
    by_game_test = defaultdict(list)
    for r in c.test:
        by_game_test[r.game].append((r.video_name, [d for d in r.bugs if str(d).strip()]))
    keys = ("const_generic", "const_glide_example", "const_dev_medoid", "const_dev_tfidf", "const_game_loo")
    const = {k: {} for k in keys}
    for v in names3:
        w_ = R.whole(c, v)
        const["const_generic"][v] = R.pred([R.GENERIC_SENTENCE], [w_])
        const["const_glide_example"][v] = R.pred([R.GLIDE_EXAMPLE], [w_])
        const["const_dev_medoid"][v] = R.pred([dev_texts[judge_pick]], [w_])
        const["const_dev_tfidf"][v] = R.pred([dev_texts[tfidf_pick]], [w_])
        g = c.by_name[v].game
        pool = list(by_game_dev[g]) + [d for n, ds in by_game_test[g] if n != v for d in ds]
        const["const_game_loo"][v] = R.pred([pool[R.text_medoid(pool)] if pool else dev_texts[judge_pick]], [w_])
    r3, _ = R.family(c, "E3_constant", "hover", hover_preds, [(k, const[k]) for k in keys], names3)
    out_rows += r3
    print(b, {r["variant"]: (round(100 * r["f1_iou"], 2), r.get("p", "")) for r in r3})
R.write_rows(os.path.join(a.out, "revision_controls.csv"), out_rows)
