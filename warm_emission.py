#!/usr/bin/env python3
"""Fill the VLM cache for the 'emission rule' control (hover_revision.glide_emission)
on one shard of the test clips, so the sequential 'controls' pass reads it from the cache.

  python warm_emission.py --backbone ui-tars-1.5-7b --shard 0 --nshards 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs.json")
ap.add_argument("--backbone", required=True)
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, required=True)
a = ap.parse_args()

R.load_hover(os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOVER.py"))
spec = R.read_runs(a.runs)
args = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
ctx = R.Ctx(spec, a.backbone, args)
names = sorted(set(ctx.by_name) & set(ctx.gt) & set(ctx.run("hover")) & set(ctx.run("glide")))
R.glide_emission(ctx, names[a.shard::a.nshards])
b = ctx.vlm.budget
print({"backbone": a.backbone, "shard": a.shard, "calls": b.calls, "run": b.model_calls, "cached": b.cache_hits})
