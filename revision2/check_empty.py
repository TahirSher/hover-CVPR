#!/usr/bin/env python3
"""Which scored (gt, pred) pairs hit a judge-cache entry whose request failed (empty response -> 0)?"""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hover_revision as R
H = R.load_hover(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HOVER.py"))
spec = R.read_runs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_v2.json"))
A = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
for b in R.BACKBONES:
    R.H._PREP_MEMO.clear(); c = R.Ctx(spec, b, A); j = c.judge
    def status(g, d):
        k = H.sha1(json.dumps([j.name, j.cfg.judge_model, j.cfg.judge_temperature, H.sha1(j.prompt), g, d]))
        p = os.path.join(j.cache_dir, k[:2], k + ".json")
        if not os.path.exists(p): return "missing"
        return "ok" if json.load(open(p)).get("raw") else "EMPTY"
    for key in ("hover", "glide", "vanilla"):
        run = c.run(key); cnt = collections.Counter()
        for v, o in run.items():
            if v not in c.gt: continue
            for d in o.get("bugs", []):
                for g in c.gt[v].get("bugs", []): cnt[status(g, d)] += 1
        print(b, key, dict(cnt), flush=True)
