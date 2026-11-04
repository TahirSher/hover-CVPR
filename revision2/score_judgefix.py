#!/usr/bin/env python3
"""InternVL2.5-8B re-tuned on development labels without failed judge calls (tag judgefix):
test F1xIoU vs the main run, H + top-1 + whole clip, and the GliDe re-run   -> v2_judgefix.csv"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
ROOT = "/path/to/VideoGlitchBench/HOVER_v8_work/v10"
c = R.Ctx(spec, "internvl2.5-8b", ARGS)
c.files["hover_fix"] = f"{ROOT}/runs/internvl2.5-8b/HOVER_judgefix_test.json"
Hm, Fm, Gm = c.run("hover"), c.run("hover_fix"), c.run("glide")
names = sorted(set(Hm) & set(Fm) & set(Gm) & set(c.gt) & set(c.by_name))
P = {"hover": {v: R.pred(Hm[v].get("bugs", []), R.spans_of(Hm[v])) for v in names},
     "hover_judgefix": {v: R.pred(Fm[v].get("bugs", []), R.spans_of(Fm[v])) for v in names},
     "glide": {v: R.pred(Gm[v].get("bugs", []), Gm[v].get("time_nodes", [])) for v in names}}
per = {k: R.evaluate(c, p, names) for k, p in P.items()}
groups = [c.by_name[v].game for v in names]
rows = []
for ref, var in (("hover_judgefix", "glide"), ("hover", "hover_judgefix")):
    st = R.paired(c, per[ref], per[var], groups)
    rows.append({"ref": ref, "var": var, "f1_iou_ref": R.summarise(per[ref])["f1_iou"],
                 "f1_iou_var": R.summarise(per[var])["f1_iou"], **st})
for k in P:
    s = R.summarise(per[k])
    print(k, round(100 * s["f1_iou"], 2), "reports/clip", round(sum(len(P[k][v]["bugs"]) if isinstance(P[k][v], dict) and "bugs" in P[k][v] else 0 for v in names) / len(names), 3))
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(os.path.join(HERE, "results", "v2_judgefix.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print(r)
