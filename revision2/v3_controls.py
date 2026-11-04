#!/usr/bin/env python3
"""Third-round controls for the core (stage H + top-1, whole clip), test split, main judge.

  abstain   the core without the forced branch: a clip on which no window reported a glitch gets no
            report (the forced hypothesis is dropped)
  shuffle   the core's description of each clip is given to another clip of the same game (a fixed
            derangement within each game, three seeds), span = whole clip of the receiving clip; if
            the descriptions only encoded a class prior, shuffling would not lower the score
  xshuffle  the same across all games

  python v3_controls.py [--backbones a,b] [--judge_model M --judge_api_base URL --tag j2]
-> results/v3_controls[_tag].csv, v3_controls_tests[_tag].csv, v3_detection[_tag].csv
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--backbones", default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl2.5-4b,internvl2.5-8b,ui-tars-1.5-7b")
ap.add_argument("--judge_model", default="")
ap.add_argument("--judge_api_base", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--mock_judge", action="store_true", help="dry run: mock judge, writes no ratings")
a = ap.parse_args()
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(a.runs)
if a.mock_judge:   # no requests, nothing cached
    H.JudgeBase.rate = lambda self, g, d: (len(g) + len(d)) % 6
if a.judge_model:
    spec["judge_model"], spec["judge_api_base"] = a.judge_model, a.judge_api_base
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
sfx = (f"_{a.tag}" if a.tag else "") + ("_mock" if a.mock_judge else "")


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {len(rows)} rows -> {name}")


def derange(items, rng):
    """A permutation of items without fixed points (items are distinct clip names)."""
    if len(items) < 2:
        return list(items)
    while True:
        p = items[:]
        rng.shuffle(p)
        if all(x != y for x, y in zip(items, p)):
            return p


rows, tests, det = [], [], []
for b in a.backbones.split(","):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm = c.run("hover"), c.run("glide")
    names = sorted(set(Hm) & set(Gm) & set(c.gt) & set(c.by_name))
    whole = {v: [[0.0, round(c.duration(v), 2)]] for v in names}
    top = {v: (R.cached_hypotheses(Hm[v])[0] if Hm[v].get("hypotheses") else None) for v in names}
    P = {"top1_whole": {v: (R.pred([top[v].description], [whole[v]]) if top[v] else R.pred([], [])) for v in names},
         "top1_abstain": {v: (R.pred([top[v].description], [whole[v]]) if top[v] and not top[v].forced
                              else R.pred([], [])) for v in names},
         "glide": {v: R.pred(Gm[v].get("bugs", []), Gm[v].get("time_nodes", [])) for v in names}}
    by_game = defaultdict(list)
    for v in names:
        by_game[c.by_name[v].game].append(v)
    for s in range(a.seeds):
        rng = random.Random(1000 + s)
        src = {}
        for g in sorted(by_game):
            vs = sorted(by_game[g])
            src.update(dict(zip(vs, derange(vs, rng))))
        P[f"shuffle_game_s{s}"] = {v: (R.pred([top[src[v]].description], [whole[v]]) if top[src[v]] else R.pred([], []))
                                   for v in names}
        allv = derange(names, random.Random(2000 + s))
        P[f"shuffle_all_s{s}"] = {v: (R.pred([top[w].description], [whole[v]]) if top[w] else R.pred([], []))
                                  for v, w in zip(names, allv)}
    per = {k: R.evaluate(c, p, names) for k, p in P.items()}
    groups = [c.by_name[v].game for v in names]
    for k in P:
        rows.append({"backbone": b, "method": k, **R.summarise(per[k])})
    for ref, var in [("top1_whole", "top1_abstain"), ("top1_abstain", "glide"), ("top1_whole", "glide")] + \
                    [("top1_whole", k) for k in P if k.startswith("shuffle")]:
        st = R.paired(c, per[ref], per[var], groups)
        tests.append({"backbone": b, "ref": ref, "var": var, "f1_iou_ref": R.summarise(per[ref])["f1_iou"],
                      "f1_iou_var": R.summarise(per[var])["f1_iou"], **st})
    y = np.array([1 if c.gt[v].get("bugs") else 0 for v in names])
    emit = np.array([1 if top[v] and not top[v].forced else 0 for v in names])
    det.append({"backbone": b, "n_clean": int((y == 0).sum()), "abstain_fpr_clean": float(emit[y == 0].mean()),
                "abstain_tpr_glitch": float(emit[y == 1].mean()),
                "forced_share_all": float(np.mean([1 if top[v] and top[v].forced else 0 for v in names]))})
    print(b, {r["method"]: round(100 * r["f1_iou"], 2) for r in rows if r["backbone"] == b}, det[-1], flush=True)
write(f"v3_controls{sfx}.csv", rows)
write(f"v3_controls_tests{sfx}.csv", tests)
write(f"v3_detection{sfx}.csv", det)
