#!/usr/bin/env python3
"""Fill HOVER.py's VLM cache for one ablation variant on one shard of a split.

Runs exactly the calls of HOVER.py's evidence stage (H.gather, the same function
cmd_calibrate and run_HOVER go through) with the configuration hover_revision.py's
'ablate' uses, so the sequential ablate pass afterwards reads them from the cache.
Several shards run as parallel clients of one vLLM server.

  python warm_gather.py --runs runs.json --backbone qwen2.5-vl-3b \
      --variant no_event_resolution --split dev --shard 0 --nshards 6
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs.json")
ap.add_argument("--backbone", required=True)
ap.add_argument("--variant", required=True, choices=sorted(R.ABLATION_SWITCHES) + ["full"])
ap.add_argument("--split", required=True, choices=("dev", "test"))
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, required=True)
a = ap.parse_args()

H = R.load_hover(os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOVER.py"))
spec = R.read_runs(a.runs)
args = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
ctx = R.Ctx(spec, a.backbone, args)
sw = R.ABLATION_SWITCHES.get(a.variant, {})
cfg = H.Config(**{**asdict(ctx.cfg), **sw, "fail_on_degenerate": False})
recs = H.take_limit(ctx.splits[a.split], cfg.limit, cfg.stratify_limit)[a.shard::a.nshards]
wm = H.prepare_split(recs, cfg)
H.gather(cfg, ctx.vlm, recs, wm, True)
b = ctx.vlm.budget
print(json.dumps({"backbone": a.backbone, "variant": a.variant, "split": a.split, "shard": a.shard,
                  "videos": len(recs), "calls": b.calls, "run": b.model_calls, "cached": b.cache_hits}))
