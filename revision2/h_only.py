#!/usr/bin/env python3
"""Stage H alone: describe every window, resolve events, rank by support (no V, S or U).

Emitting the best-supported hypothesis of each clip with a whole-clip span is the
'H + top-1 + whole clip' row of the second revision. --definition swaps the glitch
definition shared by the detection and description prompts:
  orig     the definition of the paper (Supp. Sec. S4.2)
  neutral  no list of glitch kinds, only "a visible defect the developers did not intend"
  glide    the glitch categories of GliDe's own scanner prompt (scanner/system_prompt.txt)
The exclusion sentence (stylised art, menus, ...) is kept in all three, so only the list of
glitch kinds changes. With 'orig' every call is a cache hit of the main run.

  python h_only.py --backbone qwen2.5-vl-7b --definition neutral --split test --shard 0 --nshards 4
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

EXCLUDE = ("Stylised art, intended abilities, menus, HUD, cut-scenes and ordinary camera motion "
           "are NOT glitches.")
DEFS = {
    "neutral": "A glitch is a visible defect in the game that the developers did not intend. " + EXCLUDE,
    "glide": ("A glitch is a visible defect the developers did not intend, of one of these kinds. "
              "Visual: texture/lighting errors, color glitches, massive light effects; model "
              "distortion, stretching, T-pose, limb twisting; flickering, disappearing objects, "
              "loading anomalies; camera clipping through walls, abnormal visibility. "
              "Physics: clipping/penetration (falling through map, stuck in terrain, phasing through "
              "objects); launching/flying (abnormally thrown/ejected, excessive force); unstable "
              "motion (shaking, spinning, uncontrolled rolling); floating/suspension (objects "
              "hovering, ignoring gravity); jittering/trembling. "
              "Game logic: AI/pathfinding issues (NPCs stuck, can't navigate obstacles, abnormal "
              "behavior); animation freeze/desync; interaction/trigger failures; spawning/duplication "
              "anomalies. Other: network desync, mod/version conflicts, projectile anomalies, "
              "unclassifiable abnormal behavior. " + EXCLUDE),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--definition", default="orig", choices=["orig"] + sorted(DEFS))
    ap.add_argument("--split", default="test", choices=("dev", "test", "train"))
    ap.add_argument("--vanilla", action="store_true", help="run the vanilla baseline instead of stage H")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "h_only"))
    ap.add_argument("--also_done", default="", help="glob of other output files whose clips count as done "
                    "(lets a later pass use a different shard count without recomputing clips)")
    ap.add_argument("--skip_errors", action="store_true",
                    help="log a clip whose VLM requests fail and go on (it stays missing for a later pass)")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the shard from its end and re-read --also_done before every clip, stopping once a "
                         "forward stream has reached it (requests are disk-cached, so any overlap is cache hits)")
    a = ap.parse_args()

    H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
    if a.definition != "orig":
        old, new = H._GLITCH_DEF, DEFS[a.definition]
        for name in ("P_DETECT_SYS", "P_DESC_SYS", "P_DESC_OPEN_SYS"):
            s = getattr(H, name)
            H.require(old in s, f"{name} does not contain the glitch definition")
            setattr(H, name, s.replace(old, new))
        H._GLITCH_DEF = new
    spec = R.read_runs(a.runs)
    args = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
    ctx = R.Ctx(spec, a.backbone, args)
    cfg = H.Config(**{**asdict(ctx.cfg), "use_hypothesis_scoring": False})
    vlm = ctx.vlm
    recs = sorted(ctx.splits[a.split], key=lambda r: r.vid)[a.shard::a.nshards]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.backbone}_{'vanilla' if a.vanilla else a.definition}_{a.split}_{a.shard}of{a.nshards}"
                               f"{'_rev' if a.reverse else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["video_name"] for l in open(path)}
    def others():
        out = set()
        for p in (sorted(glob.glob(a.also_done)) if a.also_done else []):
            if os.path.abspath(p) != os.path.abspath(path):
                out |= {json.loads(l)["video_name"] for l in open(p) if l.strip().endswith("}")}
        return out
    done |= others()
    if a.reverse:
        recs = recs[::-1]
    with open(path, "a") as f:
        for i, r in enumerate(recs):
            if r.video_name in done:
                continue
            if a.reverse and r.video_name in others():
                break
            t0, c0, m0 = time.time(), vlm.budget.calls, vlm.budget.model_calls
            try:
                wins = H.prepare_split([r], cfg)[r.vid]
                o = H.run_vanilla(vlm, r, wins, cfg) if a.vanilla else None
                vh = None if a.vanilla else H.collect_video(vlm, r, wins, cfg)
            except RuntimeError as e:
                if not a.skip_errors:
                    raise
                print(f"SKIP {r.video_name}: {str(e)[:200]}", flush=True)
                continue
            if a.vanilla:
                f.write(json.dumps({"video_name": r.video_name, "game": r.game, "duration": r.duration,
                                    "bugs": o.get("bugs", []), "time_nodes": o.get("time_nodes", []),
                                    "calls": vlm.budget.calls - c0, "model_calls": vlm.budget.model_calls - m0,
                                    "seconds": time.time() - t0}) + "\n")
                f.flush()
                continue
            f.write(json.dumps({
                "video_name": r.video_name, "game": r.game, "duration": r.duration,
                "definition": a.definition, "n_described": vh.n_described,
                "hyps": [{"description": h.description, "entity": h.entity, "glitch_type": h.glitch_type,
                          "members": h.members, "support": len(h.members), "forced": bool(h.forced)}
                         for h in vh.hyps],
                "calls": vlm.budget.calls - c0, "model_calls": vlm.budget.model_calls - m0,
                "seconds": time.time() - t0}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{a.backbone} {a.definition} shard {a.shard}: {i + 1}/{len(recs)}", flush=True)
    print(f"done {path}")


if __name__ == "__main__":
    main()
