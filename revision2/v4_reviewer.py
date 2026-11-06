#!/usr/bin/env python3
"""Pre-submission checks, from cached outputs and cached judge ratings only (no GPU, no new judge requests;
a cache miss raises instead of being scored 0).

  multi     the test clips with several ground-truth glitches vs. those with one: F1xIoU and recall of the
            core (H + top-1, whole clip), HOVER and the GliDe re-run; the recall ceiling of one report per clip
  games     per test game, dataset-level F1xIoU of the core and HOVER minus the GliDe re-run (12 games)
  gtstats   number of ground-truth glitches per clip on the whole benchmark (5,238 clips) and per split

  python v4_reviewer.py [--backbones a,b]
-> results/v4_multi.csv, v4_games.csv, v4_gtstats.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--backbones", default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl2.5-4b,internvl2.5-8b,ui-tars-1.5-7b")
a = ap.parse_args()
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))


def _no_request(self, system, user):
    raise RuntimeError("judge cache miss: this script must not send judge requests")


H.JudgeBase._chat = _no_request
os.environ["JUDGE_WORKERS"] = "1"
spec = R.read_runs(a.runs)
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {len(rows)} rows -> {name}")


multi, games, gtstats = [], [], []
for i, b in enumerate(a.backbones.split(",")):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm = c.run("hover"), c.run("glide")
    names = sorted(set(Hm) & set(Gm) & set(c.gt) & set(c.by_name))
    whole = {v: [[0.0, round(c.duration(v), 2)]] for v in names}
    top = {v: (R.cached_hypotheses(Hm[v])[0] if Hm[v].get("hypotheses") else None) for v in names}
    P = {"core": {v: (R.pred([top[v].description], [whole[v]]) if top[v] else R.pred([], [])) for v in names},
         "hover": {v: R.pred(Hm[v].get("bugs", []), Hm[v].get("time_nodes", [])) for v in names},
         "glide": {v: R.pred(Gm[v].get("bugs", []), Gm[v].get("time_nodes", [])) for v in names}}
    per = {k: {d["video_name"]: d for d in R.evaluate(c, p, names)} for k, p in P.items()}
    ngt = {v: len([d for d in c.gt[v].get("bugs", []) if str(d).strip()]) for v in names}
    subsets = {"all": names, "gt0": [v for v in names if ngt[v] == 0], "gt1": [v for v in names if ngt[v] == 1],
               "gt2plus": [v for v in names if ngt[v] >= 2]}
    for s, vs in subsets.items():
        m = sum(ngt[v] for v in vs)
        row = {"backbone": b, "subset": s, "n_clips": len(vs), "n_gt": m,
               "recall_ceiling_one_report": (sum(1 for v in vs if ngt[v] > 0) / m) if m else float("nan")}
        for k in P:
            sm = R.summarise([per[k][v] for v in vs])
            row.update({f"{k}_f1_iou": sm["f1_iou"], f"{k}_recall": sm["recall"], f"{k}_reports": sm["num_pred_bugs"]})
        multi.append(row)
    by_game = defaultdict(list)
    for v in names:
        by_game[c.by_name[v].game].append(v)
    for g, vs in sorted(by_game.items()):
        f = {k: R.summarise([per[k][v] for v in vs])["f1_iou"] for k in P}
        games.append({"backbone": b, "game": g, "n_clips": len(vs), "core_f1_iou": f["core"], "hover_f1_iou": f["hover"],
                      "glide_f1_iou": f["glide"], "core_minus_glide": f["core"] - f["glide"],
                      "hover_minus_glide": f["hover"] - f["glide"]})
    gw = [r for r in games if r["backbone"] == b]
    print(b, "games core>glide", sum(r["core_minus_glide"] > 0 for r in gw), "/", len(gw),
          "hover>glide", sum(r["hover_minus_glide"] > 0 for r in gw),
          {r["subset"]: (r["n_clips"], round(100 * r["core_f1_iou"], 2), round(100 * r["glide_f1_iou"], 2))
           for r in multi if r["backbone"] == b}, flush=True)
    if i == 0:   # ground-truth statistics do not depend on the backbone
        allgt = H.load_gt_official(c.cfg, c.recs)
        for split, recs in [("all", c.recs), ("dev", c.splits["dev"]), ("test", c.test)]:
            cnt = Counter(len([d for d in allgt.get(r.video_name, {}).get("bugs", []) if str(d).strip()])
                          for r in recs if r.video_name in allgt)
            n, m = sum(cnt.values()), sum(k * v for k, v in cnt.items())
            gtstats.append({"split": split, "n_clips": n, "n_gt": m, "clips_0": cnt[0], "clips_1": cnt[1],
                            "clips_2plus": sum(v for k, v in cnt.items() if k >= 2),
                            "share_2plus": sum(v for k, v in cnt.items() if k >= 2) / n,
                            "recall_ceiling_one_report": sum(v for k, v in cnt.items() if k >= 1) / m})
        print(gtstats, flush=True)
write("v4_multi.csv", multi)
write("v4_games.csv", games)
write("v4_gtstats.csv", gtstats)
