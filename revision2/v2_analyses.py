#!/usr/bin/env python3
"""Second-revision analyses on the main run (paired with main Tab. 1), from cached outputs only.

  1. methods     every method scored under the released evaluator (rating matching) and under
                 rating x IoU matching (the protocol of the benchmark paper)       -> v2_methods.csv
  2. tests       paired game-level tests under both protocols                       -> v2_tests.csv
  3. detection   clip-level detection on the glitch-free test clips: FPR, AUROC     -> v2_detection.csv
  4. subset      short-glitch clips (ground truth covers < half of the clip)        -> v2_subset.csv
  5. jackknife   leave-one-game-out t(11) intervals for the deltas of Tab. 1         -> in v2_tests.csv

The judge is used read-only: a rating missing from the cache raises instead of querying a server.

  python v2_analyses.py --runs ../runs.json --out results
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs.json"))
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--backbones", default=",".join(R.BACKBONES))
ap.add_argument("--glide_nosam3", default="/path/to/GliDe/data_qwen2.5-vl-3b/batch_report_fixed.json",
                help="GliDe re-run of Qwen2.5-VL-3B without SAM 3 (tracker never initialised)")
ap.add_argument("--allow_judge", action="store_true", help="query the judge server for missing ratings")
ap.add_argument("--judge_model", default="", help="score with another judge (separate cache)")
ap.add_argument("--judge_api_base", default="")
ap.add_argument("--tag", default="", help="suffix of the output files")
a = ap.parse_args()

H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(a.runs)
if a.judge_model:
    spec["judge_model"], spec["judge_api_base"] = a.judge_model, a.judge_api_base
args = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
os.makedirs(a.out, exist_ok=True)
MISSING = []


class CacheOnly(Exception):
    pass


def cache_only(judge):
    def _chat(system, user):
        MISSING.append(user[:200])
        raise CacheOnly("judge rating not cached")
    if not a.allow_judge:
        judge._chat = _chat
    return judge


# ----------------------------------------------------------------------------- evaluation
def eval_video(judge, gt, pr, rule):
    """rule 'rating': released evaluator (Hungarian on ratings, IoU afterwards).
    rule 'rxiou': Hungarian on rating x IoU, the weighting described in the benchmark paper."""
    if rule == "rating":
        return H.official_eval_video(judge, gt, pr)
    gb, pb = list(gt.get("bugs", [])), list(pr.get("bugs", []))
    gtn, ptn = list(gt.get("time_nodes", [])), list(pr.get("time_nodes", []))
    res = {"gt_num": len(gb), "pred_num": len(pb), "matched_scores": [], "matched_ious": [], "matches": []}
    if not gb or not pb:
        return res
    S = np.array([[judge.rate(g, p) for g in gb] for p in pb], float)
    I = np.array([[H.official_iou(gtn[j] if j < len(gtn) else [], ptn[i] if i < len(ptn) else [])
                   for j in range(len(gb))] for i in range(len(pb))])
    pi, gi = linear_sum_assignment(-(S * I))
    for x, y in zip(pi, gi):
        res["matched_scores"].append(int(S[x, y]))
        res["matched_ious"].append(float(I[x, y]))
    return res


def evaluate(ctx, preds, names, rule):
    empty = R.pred([], [])
    if a.allow_judge:
        R.prefetch_ratings(ctx, [(g, d) for v in names for d in preds.get(v, empty)["bugs"]
                                 for g in ctx.gt[v].get("bugs", [])])
    return [dict(eval_video(ctx.judge, ctx.gt[v], preds.get(v, empty), rule), video_name=v) for v in names]


def jackknife(ref, var, groups, key="f1_iou"):
    """Delete-one-game jackknife SE of the difference, t(G-1) interval (CR3-type)."""
    G = sorted(set(groups))
    full = H.official_aggregate(ref)[key] - H.official_aggregate(var)[key]
    loo = []
    for g in G:
        keep = [i for i, x in enumerate(groups) if x != g]
        loo.append(H.official_aggregate([ref[i] for i in keep])[key]
                   - H.official_aggregate([var[i] for i in keep])[key])
    loo = np.array(loo)
    se = math.sqrt((len(G) - 1) / len(G) * np.sum((loo - loo.mean()) ** 2))
    q = float(student_t.ppf(0.975, len(G) - 1))
    return full, full - q * se, full + q * se


def roc_auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def write(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {len(rows)} rows -> {name}")


# ----------------------------------------------------------------------------- per backbone
med = list(csv.DictReader(open(os.path.join(os.path.dirname(HERE), "revision_results",
                                            "revision_medoid_candidates.csv"))))
S_MEDOID = next(r["sentence"] for r in med if r["judge_pick"] == "1")
S_TFIDF = next(r["sentence"] for r in med if r["tfidf_pick"] == "1")
rows_m, rows_t, rows_d, rows_s = [], [], [], []
for b in [x for x in a.backbones.split(",") if x]:
    R.H._PREP_MEMO.clear()
    ctx = R.Ctx(spec, b, args)
    cache_only(ctx.judge)
    mode = ctx.main_mode()
    Hm, Gm, Vm = ctx.run("hover"), ctx.run("glide"), ctx.run("vanilla")
    names_all = sorted(set(ctx.by_name) & set(ctx.gt) & set(Hm))
    names3 = sorted(set(names_all) & set(Gm) & set(Vm))
    groups = {v: ctx.by_name[v].game for v in names_all}
    P = {}
    P["hover"] = {v: R.pred(Hm[v].get("bugs", []), R.spans_of(Hm[v])) for v in names_all}
    P["hover_whole"] = {v: R.pred(Hm[v].get("bugs", []), [R.whole(ctx, v)] * len(Hm[v].get("bugs", [])))
                        for v in names_all}
    P["top1_whole"] = {v: (R.pred([R.cached_hypotheses(Hm[v])[0].description], [R.whole(ctx, v)])
                           if Hm[v].get("hypotheses") else R.pred([], [])) for v in names_all}
    P["glide"] = {v: R.pred(Gm[v].get("bugs", []), Gm[v].get("time_nodes", [])) for v in Gm}
    P["glide_whole"] = {v: R.pred(Gm[v].get("bugs", []), [R.whole(ctx, v)] * len(Gm[v].get("bugs", [])))
                        for v in Gm if v in ctx.by_name}
    P["vanilla"] = {v: R.pred(Vm[v].get("bugs", []), Vm[v].get("time_nodes", [])) for v in Vm}
    for k, s in (("const_generic", R.GENERIC_SENTENCE), ("const_dev_tfidf", S_TFIDF),
                 ("const_dev_medoid", S_MEDOID)):
        P[k] = {v: R.pred([s], [R.whole(ctx, v)]) for v in names_all}
    if b == "qwen2.5-vl-3b" and os.path.exists(a.glide_nosam3):
        outs, _ = H.import_glide_report(a.glide_nosam3, ctx.test)
        P["glide_nosam3"] = {o["video_name"]: R.pred(o["bugs"], o["time_nodes"]) for o in outs}

    # 1-2. all methods on the three-method clip set, both matching rules; tests vs HOVER and vs GliDe
    per = {}
    for rule in ("rating", "rxiou"):
        for m, pr in P.items():
            try:
                per[(m, rule)] = evaluate(ctx, pr, names3, rule)
            except CacheOnly:
                print(f"  {b} {m} {rule}: ratings missing from the judge cache, skipped")
                continue
            s = R.summarise(per[(m, rule)])
            rows_m.append({"backbone": b, "mode": mode, "rule": rule, "method": m, "n_clips": len(names3), **s})
        grp = [groups[v] for v in names3]
        fam = defaultdict(dict)
        for ref, var, family in (("hover", "glide", "vs_glide"), ("vanilla", "glide", "vs_glide"),
                                 ("top1_whole", "glide", "top1_vs_glide"),
                                 ("hover", "top1_whole", "vs_hover"), ("hover", "hover_whole", "vs_hover"),
                                 ("hover", "glide_whole", "vs_hover"), ("hover", "const_generic", "vs_hover"),
                                 ("hover", "const_dev_tfidf", "vs_hover"), ("hover", "const_dev_medoid", "vs_hover"),
                                 ("glide", "glide_nosam3", "sam3"), ("hover", "glide_nosam3", "sam3")):
            if (ref, rule) not in per or (var, rule) not in per:
                continue
            A_, B_ = per[(ref, rule)], per[(var, rule)]
            pm = H.paired_cluster_permutation(A_, B_, "f1_iou", 4096, 0, grp)
            bs = H.paired_bootstrap(A_, B_, "f1_iou", 2000, 0, grp)
            d, jlo, jhi = jackknife(A_, B_, grp)
            r = {"backbone": b, "rule": rule, "family": family, "ref": ref, "var": var,
                 "f1_iou_ref": H.official_aggregate(A_)["f1_iou"], "f1_iou_var": H.official_aggregate(B_)["f1_iou"],
                 "delta": d, "pct_lo": bs["ci_lo"], "pct_hi": bs["ci_hi"], "jk_lo": jlo, "jk_hi": jhi, "p": pm["p"]}
            rows_t.append(r)
            fam[family][var + "|" + ref] = pm["p"]
        for family, pv in fam.items():
            hb = H.holm_bonferroni(pv)
            for r in rows_t:
                if r["backbone"] == b and r["rule"] == rule and r["family"] == family:
                    r["holm_reject_block"] = hb[r["var"] + "|" + r["ref"]]["reject"]

    # 3. clip-level detection: positives = clips with >= 1 ground-truth glitch
    y = np.array([int(bool(ctx.gt[v].get("bugs"))) for v in names_all])
    clean = y == 0
    gl = [groups[v] for v in names_all]

    def det_row(method, score, emits):
        score, emits = np.asarray(score, float), np.asarray(emits, bool)
        auc = H.cluster_bootstrap_auc(score, y, gl, 2000, 0) if np.unique(score).size > 1 else \
            {"auc": 0.5, "lo": float("nan"), "hi": float("nan")}
        return {"backbone": b, "method": method, "n_clips": len(y), "n_clean": int(clean.sum()),
                "fpr_clean": float(emits[clean].mean()), "tpr_glitch": float(emits[~clean].mean()),
                "auroc": auc["auc"], "auroc_lo": auc["lo"], "auroc_hi": auc["hi"],
                "auroc_pct_lo": auc.get("pct_lo", float("nan")), "auroc_pct_hi": auc.get("pct_hi", float("nan"))}

    umax = [max([float(h.get("utility", 0.0)) for h in Hm[v].get("hypotheses", [])] or [0.0]) for v in names_all]
    nonforced = [float(any(not h.get("forced") for h in Hm[v].get("hypotheses", []))) for v in names_all]
    gen = [max([float(h.get("gen_z", -9)) for h in Hm[v].get("hypotheses", [])] or [-9.0]) for v in names_all]
    support = [max([float(h.get("support", 0)) for h in Hm[v].get("hypotheses", [])] or [0.0]) for v in names_all]
    nh = [len(Hm[v].get("bugs", [])) for v in names_all]
    rows_d.append(det_row("hover_selector_umax", umax, np.array(nh) > 0))
    rows_d.append(det_row("hover_description_found", nonforced, np.array(nonforced) > 0))
    rows_d.append(det_row("hover_support_max", support, np.array(nh) > 0))
    rows_d.append(det_row("hover_generic_detector_z", gen, np.array(nh) > 0))
    rows_d.append(det_row("top1", [1.0] * len(y), [True] * len(y)))
    for m, run in (("glide", Gm), ("vanilla", Vm)):
        n = [len(run[v].get("bugs", [])) if v in run else 0 for v in names_all]
        rows_d.append(det_row(m, n, np.array(n) > 0))
    print(b, {r["method"]: (round(r["fpr_clean"], 3), round(r["auroc"], 3)) for r in rows_d if r["backbone"] == b})

    # 4. short-glitch subset: ground-truth union covers < 50% of the clip
    def cover(v):
        iv = sorted([list(map(float, x)) for tn in ctx.gt[v].get("time_nodes", []) for x in tn])
        tot, cur = 0.0, None
        for s0, e0 in iv:
            if cur and s0 <= cur[1]:
                cur[1] = max(cur[1], e0)
            else:
                if cur:
                    tot += cur[1] - cur[0]
                cur = [s0, e0]
        if cur:
            tot += cur[1] - cur[0]
        return tot / max(ctx.duration(v), 1e-6)
    cov = {v: cover(v) for v in names3 if ctx.gt[v].get("bugs")}
    for thr in (0.5, 0.3):
        sub = sorted(v for v, c in cov.items() if c < thr)
        sub_groups = [groups[v] for v in sub]
        idx = {v: i for i, v in enumerate(names3)}
        for rule in ("rating", "rxiou"):
            ref = [per[("hover", rule)][idx[v]] for v in sub]
            for m in ("hover_whole", "top1_whole", "glide", "glide_whole", "const_dev_tfidf"):
                if (m, rule) not in per:
                    continue
                var = [per[(m, rule)][idx[v]] for v in sub]
                pm = H.paired_cluster_permutation(ref, var, "f1_iou", 4096, 0, sub_groups)
                rows_s.append({"backbone": b, "mode": mode, "cover_lt": thr, "rule": rule, "n_clips": len(sub),
                               "n_games": len(set(sub_groups)), "method": m,
                               "f1_iou_hover": H.official_aggregate(ref)["f1_iou"],
                               "iou_r_hover": R.summarise(ref)["iou_r"],
                               "f1_iou_var": H.official_aggregate(var)["f1_iou"],
                               "iou_r_var": R.summarise(var)["iou_r"],
                               "delta": pm["delta"], "p": pm["p"]})

write(f"v2_methods{a.tag and '_' + a.tag}.csv", rows_m)
write(f"v2_tests{a.tag and '_' + a.tag}.csv", rows_t)
write(f"v2_detection{a.tag and '_' + a.tag}.csv", rows_d)
write(f"v2_subset{a.tag and '_' + a.tag}.csv", rows_s)
if MISSING:
    print(f"{len(MISSING)} judge ratings were not cached (rerun with --allow_judge and a judge server)")
