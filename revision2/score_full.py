#!/usr/bin/env python3
"""Core (stage H + top-1, whole clip) and vanilla on the full VideoGlitchBench (all 5,238 clips, 120 games),
the clip set of GliDe's published results, with the released evaluator and the main judge.

Core and vanilla have no tuned parameter, so every clip is a test clip for them. Reported on all clips, on
the 96 games outside our development and test splits (never used for any decision), and on the 24 games of those splits.
Paired core-vs-vanilla tests with the game as unit.
  python score_full.py [--backbones a,b] [--mock_judge]
-> results/v3_full.csv, v3_full_tests.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbones", default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl2.5-4b,internvl2.5-8b,ui-tars-1.5-7b")
ap.add_argument("--mock_judge", action="store_true", help="dry run: fake ratings, nothing cached")
a = ap.parse_args()
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
if a.mock_judge:
    H.JudgeBase.rate = lambda self, g, d: (len(g) + len(d)) % 6
spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
sfx = "_mock" if a.mock_judge else ""


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            x = json.loads(line)
            out[x["video_name"]] = x
    return out


rows, tests = [], []
for b in a.backbones.split(","):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    recs = c.splits["train"] + c.splits["dev"] + c.splits["test"]
    c.by_name = {r.video_name: r for r in recs}
    c.gt = H.load_gt_official(c.cfg, recs)
    split_of = {r.video_name: s for s in ("train", "dev", "test") for r in c.splits[s]}
    core = {**load(os.path.join(HERE, "h_only", f"{b}_orig_dev_*.jsonl")),
            **load(os.path.join(HERE, "h_only", f"{b}_orig_test_*.jsonl")),
            **load(os.path.join(HERE, "h_full", f"{b}_orig_train_*.jsonl"))}
    van = {**load(os.path.join(HERE, "h_full", f"{b}_vanilla_train_*.jsonl")),
           **load(os.path.join(HERE, "h_full", f"{b}_vanilla_dev_*.jsonl"))}
    for v, o in c.run("vanilla").items():
        van.setdefault(v, o)
    names = sorted(set(core) & set(van) & set(c.gt) & set(c.by_name))
    missing = len(recs) - len(names)

    def dur(v):
        d = float(core[v].get("duration") or 0)
        return d if d > 0 else c.duration(v)

    P = {"core": {v: (R.pred([core[v]["hyps"][0]["description"]], [[[0.0, round(dur(v), 2)]]])
                      if core[v]["hyps"] else R.pred([], [])) for v in names},
         "vanilla": {v: R.pred(van[v].get("bugs", []), van[v].get("time_nodes", [])) for v in names}}
    per = {k: R.evaluate(c, p, names) for k, p in P.items()}
    for subset, keep in (("all", lambda v: True), ("unused_96_games", lambda v: split_of[v] == "train"),
                         ("our_24_games", lambda v: split_of[v] != "train")):
        idx = [i for i, v in enumerate(names) if keep(v)]
        sub = {k: [per[k][i] for i in idx] for k in per}
        groups = [c.by_name[names[i]].game for i in idx]
        for k in sub:
            rows.append({"backbone": b, "subset": subset, "method": k, "n_games": len(set(groups)),
                         "missing_clips": missing if subset == "all" else "", **R.summarise(sub[k])})
        st = R.paired(c, sub["core"], sub["vanilla"], groups)
        tests.append({"backbone": b, "subset": subset, "ref": "core", "var": "vanilla",
                      "f1_iou_ref": R.summarise(sub["core"])["f1_iou"], "f1_iou_var": R.summarise(sub["vanilla"])["f1_iou"], **st})
    print(b, len(names), "clips;", {(r["subset"], r["method"]): round(100 * r["f1_iou"], 2)
                                     for r in rows if r["backbone"] == b}, flush=True)

for name, rr in ((f"v3_full{sfx}.csv", rows), (f"v3_full_tests{sfx}.csv", tests)):
    fields = list(dict.fromkeys(k for r in rr for k in r))
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rr:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", name)
