#!/usr/bin/env python3
"""Blinded human check of the judges: matched (ground truth, prediction) pairs from the test split for
HOVER, the core (H + top-1, whole clip), the GliDe re-run and the TF-IDF fixed sentence, matched as the
released evaluator does (Hungarian on main-judge ratings); 35 pairs per method drawn evenly over the five
backbones. Two sheets with the same pairs in different orders, one per annotator; method, backbone and judge
ratings are hidden (human_check/key.csv).
"""
import argparse, csv, json, os, random, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
ARGS = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
J1, J2 = "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct-AWQ"
med = list(csv.DictReader(open(os.path.join(os.path.dirname(HERE), "revision_results", "revision_medoid_candidates.csv"))))
S_TFIDF = next(r["sentence"] for r in med if r["tfidf_pick"] == "1")
BB = ("qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "internvl2.5-8b", "ui-tars-1.5-7b")


def rating(c, model, g, d):
    j = c.judge
    key = H.sha1(json.dumps([j.name, model, j.cfg.judge_temperature, H.sha1(j.prompt), g, d]))
    p = os.path.join(os.path.dirname(j.cache_dir.rstrip("/")), model.replace("/", "_"), key[:2], key + ".json")
    if not os.path.exists(p):
        return None
    x = H.read_json(p)
    return None if not str(x.get("raw", "")).strip() else int(x["rating"])


def matched(c, gts, preds):
    """Pairs assigned by the evaluator's Hungarian matching on main-judge ratings."""
    if not gts or not preds:
        return []
    M = np.array([[rating(c, J1, g, d) or 0 for g in gts] for d in preds], float)
    ri, ci = linear_sum_assignment(-M)
    return [(gts[j], preds[i]) for i, j in zip(ri, ci)]


pool = {m: [] for m in ("hover", "core", "glide", "tfidf")}
for b in BB:
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm = c.run("hover"), c.run("glide")
    for v in sorted(set(Hm) & set(Gm) & set(c.gt)):
        gts = c.gt[v].get("bugs", [])
        hy = R.cached_hypotheses(Hm[v])
        for m, preds in (("hover", Hm[v].get("bugs", [])), ("core", [hy[0].description] if hy else []),
                         ("glide", Gm[v].get("bugs", [])), ("tfidf", [S_TFIDF])):
            for g, d in matched(c, gts, preds):
                r1, r2 = rating(c, J1, g, d), rating(c, J2, g, d)
                if r1 is not None and r2 is not None:
                    pool[m].append({"gt": g, "pred": d, "backbone": b, "method": m, "video": v, "llama": r1, "qwen72b": r2})
    jc = c
rng = random.Random(11)
sample, seen = [], set()
for m in ("hover", "core", "glide", "tfidf"):
    per_b = {b: [x for x in pool[m] if x["backbone"] == b] for b in BB}
    picked = []
    for i in range(35):
        b = BB[i % 5]
        cand = [x for x in per_b[b] if (x["gt"], x["pred"]) not in seen]
        x = rng.choice(cand)
        seen.add((x["gt"], x["pred"])); picked.append(x)
    sample += picked
for i, x in enumerate(sample):
    x["id"] = i + 1
os.makedirs(os.path.join(HERE, "human_check"), exist_ok=True)
for ann, seed in (("A", 21), ("B", 22)):
    order = sample[:]
    random.Random(seed).shuffle(order)
    with open(os.path.join(HERE, "human_check", f"human_sheet_annotator_{ann}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ground_truth_description", "predicted_description", "human_rating_0_to_5"])
        for x in order:
            w.writerow([x["id"], x["gt"], x["pred"], ""])
with open(os.path.join(HERE, "human_check", "key.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "method", "backbone", "video", "llama", "qwen72b"])
    w.writeheader()
    for x in sample:
        w.writerow({k: x[k] for k in ("id", "method", "backbone", "video", "llama", "qwen72b")})
print({m: len(v) for m, v in pool.items()}, "sampled", len(sample))
