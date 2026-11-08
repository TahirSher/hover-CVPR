#!/usr/bin/env python3
"""Final-round audit (answers to a mock review), from cached outputs and cached judge ratings only (no GPU, no
new judge requests; a cache miss raises).

  annot     the released annotations (VideoGlitchBench annotations.json) against the ground truth the released
            evaluator scores (groundtruth.json of the GliDe repository): descriptions without a time span, clips
            flagged no_bugs = false without any description; per split
  detect    clip-level detection on the test split with positives = clips with a ground-truth glitch and
            negatives = (a) all clips without one (as in Supp. S9.3) or (b) only the clips annotated no_bugs = true
  emission  reports per ground-truth glitch of the GliDe re-run (M/N) against the published P/R of each backbone
            (with dataset-level sums, P/R = M/N), and the core against a GliDe re-run cut to the published
            emission rate (last report of the clip with most reports removed first, until M/N matches)
  controls  zero-cost controls: GliDe's first / longest report with a whole-clip span; the core (whole clip) with
            a random (seed 0) or the lowest-support hypothesis instead of the best-supported one
  support   does support rank hypotheses? AUROC of support for 'the hypothesis is correct' (best judge rating
            against the clip's ground truth >= 3), over all cached hypotheses of clips with a glitch

  python v5_audit.py --annotations <annotations.json>
-> results/v5_annot.csv, v5_detect.csv, v5_emission.csv, v5_controls.csv, v5_support.csv
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--annotations", required=True, help="annotations.json of the VideoGlitchBench dataset release")
ap.add_argument("--backbones", default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl2.5-4b,internvl2.5-8b,ui-tars-1.5-7b")
a = ap.parse_args()
H = R.load_hover(os.path.join(HERE, "HOVER.py"))


def _no_request(self, system, user):
    raise RuntimeError("judge cache miss: this script must not send judge requests")


H.JudgeBase._chat = _no_request
os.environ["JUDGE_WORKERS"] = "1"
spec = R.read_runs(a.runs)
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
GLIDE_PUB_PR = {"qwen2.5-vl-3b": (32.36, 39.38), "qwen2.5-vl-7b": (34.55, 45.09), "internvl2.5-4b": (29.15, 36.87),
                "internvl2.5-8b": (32.64, 40.97), "ui-tars-1.5-7b": (34.31, 49.28)}   # published P, R (%)


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {len(rows)} rows -> {name}")


def nb(x):
    return [b for b in x.get("bugs", []) if str(b).strip()]


# ------------------------------------------------------------------ annot
rel = {x["video_name"].replace(".mp4", ""): x for x in json.load(open(a.annotations))}
annot = []
splits = {s: [str(v).replace(".mp4", "").split("::")[-1] for v in json.load(open(os.path.join(spec["root"], "splits", f"{s}.json")))]
          for s in ("dev", "test", "train")} if os.path.exists(os.path.join(spec["root"], "splits", "dev.json")) else {}
ev = {x["video_name"].replace(".mp4", ""): x for x in json.load(open(spec["gt_json"]))}
assert set(ev) == set(rel)
for s, names in [("all", sorted(rel))] + sorted(splits.items()):
    names = [v for v in names if v in rel]
    free = [v for v in names if not nb(ev[v])]
    annot.append({"split": s, "clips": len(names),
                  "released_descriptions": sum(len(nb(rel[v])) for v in names),
                  "released_spanless": sum(1 for v in names for i, _ in enumerate(nb(rel[v]))
                                           if i >= len(rel[v].get("time_nodes", [])) or not rel[v]["time_nodes"][i]),
                  "evaluator_glitches": sum(len(nb(ev[v])) for v in names),
                  "released_no_bugs_true": sum(bool(rel[v]["no_bugs"]) for v in names),
                  "released_flagged_without_description": sum(1 for v in names if not rel[v]["no_bugs"] and not nb(rel[v])),
                  "released_no_bugs_true_with_description": sum(1 for v in names if rel[v]["no_bugs"] and nb(rel[v])),
                  "evaluator_glitch_free": len(free),
                  "evaluator_free_true_negative": sum(1 for v in free if rel[v]["no_bugs"]),
                  "evaluator_free_flagged": sum(1 for v in free if not rel[v]["no_bugs"] and not nb(rel[v])),
                  "evaluator_free_spanless_only": sum(1 for v in free if nb(rel[v])),
                  "multi_released": sum(1 for v in names if len(nb(rel[v])) > 1),
                  "multi_evaluator": sum(1 for v in names if len(nb(ev[v])) > 1),
                  "recall_cap_evaluator": sum(1 for v in names if nb(ev[v])) / max(1, sum(len(nb(ev[v])) for v in names)),
                  "recall_cap_released": sum(1 for v in names if nb(rel[v])) / max(1, sum(len(nb(rel[v])) for v in names))})
write("v5_annot.csv", annot)

detect, emission, controls, support = [], [], [], []
for b in a.backbones.split(","):
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm, Vm = c.run("hover"), c.run("glide"), c.run("vanilla")
    names = sorted(set(Hm) & set(Gm) & set(c.gt) & set(c.by_name))
    grp = [c.by_name[v].game for v in names]
    whole = {v: [[0.0, round(c.duration(v), 2)]] for v in names}
    hyps = {v: (Hm[v].get("hypotheses") or []) for v in names}

    # ---------------- detect
    y = np.array([int(bool(nb(c.gt[v]))) for v in names])
    tn = np.array([bool(rel[v]["no_bugs"]) and not nb(rel[v]) for v in names])
    scores = {"hover_selector_umax": [max([float(h.get("utility", 0)) for h in hyps[v]] or [0.0]) for v in names],
              "hover_description_found": [float(any(not h.get("forced") for h in hyps[v])) for v in names],
              "hover_support_max": [max([float(h.get("support", 0)) for h in hyps[v]] or [0.0]) for v in names],
              "glide_reports": [len(Gm[v].get("bugs", [])) for v in names],
              "vanilla_reports": [len(Vm[v].get("bugs", [])) if v in Vm else 0 for v in names]}
    for m, s in scores.items():
        s = np.asarray(s, float)
        for neg, keep in (("all_glitch_free", np.ones(len(y), bool)), ("true_negatives", (y == 1) | tn)):
            yy, ss, gg = y[keep], s[keep], [g for g, k in zip(grp, keep) if k]
            auc = H.cluster_bootstrap_auc(ss, yy, gg, 2000, 0) if np.unique(ss).size > 1 else {"auc": 0.5, "lo": float("nan"), "hi": float("nan")}
            emits = ss > 0 if m.endswith("reports") or m == "hover_description_found" else None
            detect.append({"backbone": b, "method": m, "negatives": neg, "n_pos": int(yy.sum()), "n_neg": int((yy == 0).sum()),
                           "auroc": auc["auc"], "auroc_lo": auc["lo"], "auroc_hi": auc["hi"],
                           "fpr": float(emits[yy == 0].mean()) if emits is not None else float("nan"),
                           "tpr": float(emits[yy == 1].mean()) if emits is not None else float("nan")})

    # ---------------- predictions for emission / controls
    top = {v: (R.cached_hypotheses(Hm[v])[0] if hyps[v] else None) for v in names}
    core = {v: (R.pred([top[v].description], [whole[v]]) if top[v] else R.pred([], [])) for v in names}
    glide = {v: R.pred(Gm[v].get("bugs", []), Gm[v].get("time_nodes", [])) for v in names}
    first_w, longest_w = {}, {}
    for v in names:
        bugs = [d for d in Gm[v].get("bugs", [])]
        if bugs:
            il = max(range(len(bugs)), key=lambda i: (len(str(bugs[i])), -i))
            first_w[v], longest_w[v] = R.pred([bugs[0]], [whole[v]]), R.pred([bugs[il]], [whole[v]])
        else:
            first_w[v] = longest_w[v] = R.pred([], [])
    rng = random.Random(0)
    rand, low = {}, {}
    for v in names:
        hs = R.cached_hypotheses(Hm[v]) if hyps[v] else []
        if hs:
            sup = [float(h.get("support", 0)) for h in hyps[v]]
            il = min(range(len(hs)), key=lambda i: (sup[i], -i))   # lowest support, latest on ties
            rand[v] = R.pred([hs[rng.randrange(len(hs))].description], [whole[v]])
            low[v] = R.pred([hs[il].description], [whole[v]])
        else:
            rand[v] = low[v] = R.pred([], [])
    # GliDe cut to the published emission rate
    n_gt = sum(len(nb(c.gt[v])) for v in names)
    P, Rr = GLIDE_PUB_PR[b]
    target = round(n_gt * Rr / P)
    keepn = {v: len(Gm[v].get("bugs", [])) for v in names}
    while sum(keepn.values()) > target:
        v = max(names, key=lambda x: (keepn[x], x))
        keepn[v] -= 1
    cut = {v: R.pred(Gm[v].get("bugs", [])[:keepn[v]], Gm[v].get("time_nodes", [])[:keepn[v]]) for v in names}

    P_ = {"core": core, "glide": glide, "glide_first_whole": first_w, "glide_longest_whole": longest_w,
          "core_random_hypothesis": rand, "core_lowest_support": low, "glide_published_rate": cut}
    per = {k: R.evaluate(c, p, names) for k, p in P_.items()}
    summ = {k: R.summarise(x) for k, x in per.items()}
    pm = lambda x, y_: H.paired_cluster_permutation(per[x], per[y_], "f1_iou", 4096, 0, grp)
    bs = lambda x, y_: H.paired_bootstrap(per[x], per[y_], "f1_iou", 2000, 0, grp)
    for k in P_:
        row = {"backbone": b, "method": k, "f1_iou": summ[k]["f1_iou"], "reports": summ[k]["num_pred_bugs"],
               "precision": summ[k]["precision"], "recall": summ[k]["recall"]}
        for ref in ("core", "glide"):
            if k != ref:
                t, q = pm(ref, k), bs(ref, k)
                row.update({f"{ref}_minus": t["delta"], f"{ref}_minus_lo": q["ci_lo"], f"{ref}_minus_hi": q["ci_hi"],
                            f"{ref}_minus_p": t["p"]})
        controls.append(row)
    m_re = summ["glide"]["num_pred_bugs"]
    emission.append({"backbone": b, "n_gt": n_gt, "glide_reports": m_re, "rerun_reports_per_glitch": m_re / n_gt,
                     "published_reports_per_glitch": Rr / P, "ratio_rerun_to_published": (m_re / n_gt) / (Rr / P),
                     "cut_reports": summ["glide_published_rate"]["num_pred_bugs"],
                     "glide_f1_iou": summ["glide"]["f1_iou"], "cut_f1_iou": summ["glide_published_rate"]["f1_iou"],
                     "core_f1_iou": summ["core"]["f1_iou"],
                     "core_minus_glide": pm("core", "glide")["delta"], "core_minus_glide_p": pm("core", "glide")["p"],
                     "core_minus_cut": pm("core", "glide_published_rate")["delta"],
                     "core_minus_cut_p": pm("core", "glide_published_rate")["p"]})

    # ---------------- support
    ys, ss_, top_correct = [], [], []
    for v in names:
        gts = nb(c.gt[v])
        if not gts or not hyps[v]:
            continue
        for i, h in enumerate(hyps[v]):
            r = max(c.judge.rate(g, h["description"]) for g in gts)
            ys.append(int(r >= 3)); ss_.append(float(h.get("support", 0)))
            if i == 0:
                top_correct.append(int(r >= 3))
    ys, ss_ = np.array(ys), np.array(ss_)
    from sklearn.metrics import roc_auc_score  # noqa: E402
    support.append({"backbone": b, "hypotheses": len(ys), "share_correct": ys.mean(),
                    "auroc_support": roc_auc_score(ys, ss_) if 0 < ys.sum() < len(ys) else float("nan"),
                    "top1_correct": float(np.mean(top_correct)), "clips": len(top_correct),
                    "forced_share": float(np.mean([bool(hyps[v] and hyps[v][0].get("forced")) for v in names]))})
    print(b, "done", flush=True)

write("v5_detect.csv", detect)
write("v5_emission.csv", emission)
write("v5_controls.csv", controls)
write("v5_support.csv", support)
