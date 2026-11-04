#!/usr/bin/env python3
"""Fill the VLM cache of a new backbone for HOVER (stages H and V) on one shard of a split, so the
sequential 'calibrate' and 'run' afterwards read cached calls. Several shards run as parallel
clients of one vLLM server; results are identical to a sequential run (greedy decoding, same prompts).

  python warm.py --backbone qwen3-vl-8b --split dev --shard 0 --nshards 6
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--backbone", required=True)
ap.add_argument("--split", required=True, choices=("dev", "test"))
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, required=True)
ap.add_argument("--vanilla", action="store_true", help="warm the vanilla baseline instead")
a = ap.parse_args()

H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
ctx = R.Ctx(R.read_runs(a.runs), a.backbone,
            argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants=""))
cfg = ctx.cfg
recs = H.take_limit(ctx.splits[a.split], cfg.limit, cfg.stratify_limit)[a.shard::a.nshards]
wm = H.prepare_split(recs, cfg)
if a.vanilla:
    for r in recs:
        H.run_vanilla(ctx.vlm, r, wm[r.vid], cfg)
else:
    H.gather(cfg, ctx.vlm, recs, wm, True)
b = ctx.vlm.budget
print(json.dumps({"backbone": a.backbone, "split": a.split, "shard": a.shard, "videos": len(recs),
                  "calls": b.calls, "run": b.model_calls, "cached": b.cache_hits}))
