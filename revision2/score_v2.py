#!/usr/bin/env python3
"""Score the second-revision runs with the main judge (Llama-3.1-8B-Instruct, T = 0.3).

  honly   'H + top-1 + whole clip' for every H-only output (h_only/*.jsonl): test split, paired
          against HOVER of the main run and against the original definition; dev split for the
          development-test gap                                                     -> v2_honly.csv
  new     new backbones (runs_v2.json entries without a GliDe run): HOVER, HOVER whole-clip,
          H + top-1 + whole clip, vanilla, paired tests                             -> v2_newbb.csv

  python score_v2.py honly
  python score_v2.py new --backbones qwen3-vl-8b,qwen2.5-vl-32b
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=("honly", "new"))
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--backbones", default="")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
a = ap.parse_args()
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(a.runs)
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")


def write(name, rows, keep_other=False):
    path = os.path.join(a.out, name)
    if keep_other and os.path.exists(path):   # keep rows of backbones not scored in this call
        done = {r["backbone"] for r in rows}
        rows = [r for r in csv.DictReader(open(path)) if r["backbone"] not in done] + rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {len(rows)} rows -> {name}")


def ctx_for(b, split="test"):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    if split != "test":
        recs = c.splits[split]
        c.by_name = {r.video_name: r for r in recs}
        c.gt = H.load_gt_official(c.cfg, recs)
    return c


def whole(c, v):
    r = c.by_name[v]
    if r.duration <= 0:
        c.wins(r)
    return [[0.0, round(float(r.duration), 2)]]


def test_row(c, tag, ref_name, ref, var_name, var, names):
    groups = [c.by_name[v].game for v in names]
    st = R.paired(c, ref, var, groups)
    return {"backbone": c.b, "tag": tag, "ref": ref_name, "var": var_name,
            "f1_iou_ref": R.summarise(ref)["f1_iou"], "f1_iou_var": R.summarise(var)["f1_iou"], **st}


rows, tests = [], []
if a.what == "honly":
    files = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HERE, "h_only", "*.jsonl"))):
        b, d, sp = os.path.basename(p).split("_")[:3]
        files[(b, d, sp)].append(p)
    for b in sorted({k[0] for k in files}):
        if a.backbones and b not in a.backbones.split(","):
            continue
        for sp in ("dev", "test"):
            defs = sorted(d for (bb, d, s) in files if bb == b and s == sp)
            if not defs:
                continue
            c = ctx_for(b, sp)
            per, preds = {}, {}
            for d in defs:
                out = {}
                for p in files[(b, d, sp)]:
                    for l in open(p):
                        x = json.loads(l)
                        out[x["video_name"]] = x
                names = sorted(set(out) & set(c.gt) & set(c.by_name))
                preds[d] = {v: (R.pred([out[v]["hyps"][0]["description"]], [whole(c, v)]) if out[v]["hyps"]
                                else R.pred([], [])) for v in names}
                per[d] = (names, R.evaluate(c, preds[d], names))
                s = R.summarise(per[d][1])
                L = list(out.values())
                rows.append({"backbone": b, "split": sp, "definition": d, "n_clips": len(names), **s,
                             "calls_per_clip": sum(x["calls"] for x in L) / len(L),
                             "forced_share": sum(1 for x in L if x["hyps"] and x["hyps"][0]["forced"]) / len(L)})
                print(b, sp, d, round(100 * s["f1_iou"], 2), flush=True)
            if sp == "test":
                Hm = c.run("hover")
                for d in defs:
                    names, pv = per[d]
                    hv = R.evaluate(c, {v: R.pred(Hm[v].get("bugs", []), R.spans_of(Hm[v])) for v in names}, names)
                    tests.append(test_row(c, "vs_hover", "hover", hv, f"top1_whole_{d}", pv, names))
                    if d != "orig" and "orig" in per:
                        common = sorted(set(names) & set(per["orig"][0]))
                        ix_o = {v: i for i, v in enumerate(per["orig"][0])}
                        ix_d = {v: i for i, v in enumerate(names)}
                        tests.append(test_row(c, "definition", "top1_whole_orig", [per["orig"][1][ix_o[v]] for v in common],
                                              f"top1_whole_{d}", [pv[ix_d[v]] for v in common], common))
    write("v2_honly.csv", rows)
    write("v2_honly_tests.csv", tests)

else:
    for b in [x for x in a.backbones.split(",") if x]:
        c = ctx_for(b)
        Hm, Vm = c.run("hover"), c.run("vanilla")
        names = sorted(set(Hm) & set(Vm) & set(c.gt) & set(c.by_name))
        P = {"hover": {v: R.pred(Hm[v].get("bugs", []), R.spans_of(Hm[v])) for v in names},
             "hover_whole": {v: R.pred(Hm[v].get("bugs", []), [whole(c, v)] * len(Hm[v].get("bugs", []))) for v in names},
             "top1_whole": {v: (R.pred([R.cached_hypotheses(Hm[v])[0].description], [whole(c, v)])
                                if Hm[v].get("hypotheses") else R.pred([], [])) for v in names},
             "vanilla": {v: R.pred(Vm[v].get("bugs", []), Vm[v].get("time_nodes", [])) for v in names}}
        per = {k: R.evaluate(c, p, names) for k, p in P.items()}
        for k in P:
            run = Hm if k.startswith(("hover", "top1")) else Vm
            rows.append({"backbone": b, "mode": c.main_mode(), "method": k, **R.summarise(per[k]),
                         "calls_per_clip": sum(run[v].get("calls", 0) for v in names) / len(names)})
        for var in ("vanilla", "top1_whole", "hover_whole"):
            tests.append(test_row(c, "new_backbone", "hover", per["hover"], var, per[var], names))
        print(b, {r["method"]: round(100 * r["f1_iou"], 2) for r in rows if r["backbone"] == b})
    write("v2_newbb.csv", rows, keep_other=True)
    write("v2_newbb_tests.csv", tests, keep_other=True)
