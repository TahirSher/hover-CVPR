#!/usr/bin/env python3
"""
hover_revision.py -- revision analyses for HOVER, computed on the MAIN run
==========================================================================

Computes every analysis of the CVPR revision (controls E1-E3, decomposition E4,
post-hoc emission x span table and re-tuned ablations E5, extended segmentation
grid E6, re-judging, window probe, bisection repairs) PAIRED WITH THE MAIN RUN of
main Tab. 1, so that one HOVER run underlies every table of the paper.

It imports HOVER.py (v10) as a library and uses its official-evaluator port,
judge, statistics, caches and pipeline functions, so every number is computed
exactly as in the main evaluation:
  * official protocol: judge ratings 0-5, Hungarian matching on ratings, IoU after
  * paired statistics: game-cluster bootstrap CI (B = n_bootstrap) and exact
    game-level Fisher randomisation test (2^12 patterns), Holm within each family
  * VLM and judge calls are cached by HOVER.py, so re-running is cheap

Output CSVs have exactly the schema of revision_results.zip and are read by
paper/scripts/make_revision_tables.py:
  revision_controls.csv  revision_posthoc.csv  revision_ablation.csv
  revision_decompose.csv revision_seg_grid.csv revision_rejudge.csv
  revision_probe.csv     revision_repairs.csv

Usage (see REVISION_RUN.md for the full recipe):
  python hover_revision.py selftest                       # mock end-to-end check
  python hover_revision.py all       --runs runs.json --out revision_results
  python hover_revision.py controls  --runs runs.json --out revision_results --backbones qwen2.5-vl-3b
  (subcommands: controls posthoc decompose ablate seg_grid rejudge probe all)

runs.json maps every backbone to the files of the MAIN run, e.g.
  {"root": "/path/to/VideoGlitchBench",
   "gt_json": "/path/to/groundtruth.json",
   "judge_api_base": "http://localhost:8001/v1",
   "backbones": {
     "qwen2.5-vl-3b": {"hover": "<runs>/HOVER_whole_test.json",
                       "glide": "<runs>/glide_official_test.json",
                       "vanilla": "<runs>/vanilla_test.json",
                       "calibration": "<params>/calibration.json",
                       "api_base": "http://localhost:8000/v1"}, ...}}
Relative paths are resolved against the directory of runs.json.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

H = None  # the HOVER.py module, loaded by load_hover()

BACKBONES = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "internvl2.5-8b",
             "ui-tars-1.5-7b"]
FIELDS = ["backbone", "tag", "variant", "n_videos", "num_pred_bugs", "num_matched",
          "mean_score", "precision", "recall", "f1", "iou_r", "mean_iou", "f1_iou",
          "reports_per_clip", "clips_multi", "delta_ref_minus", "ci_lo", "ci_hi", "p",
          "holm_reject", "reference"]
# Sentences of the fixed-sentence baselines (E3).
GLIDE_EXAMPLE = ("A red sports car is floating 2 meters above the road surface with no "
                 "visible support, near the highway overpass.")
GENERIC_SENTENCE = "A character or object floats in mid-air without visible support."
COPY_KEYS = ("red sports car", "highway overpass")
GENERIC_ENTITY = "the affected object or character"
ABLATION_SWITCHES = {"no_ctx_calib": {"use_contextual_calibration": False},
                     "no_hypothesis_scoring": {"use_hypothesis_scoring": False},
                     "no_event_resolution": {"use_event_resolution": False},
                     "no_utility_selection": {"use_utility_selection": False}}
EXT_TAUS = "-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"


# =============================================================================
# Loading
# =============================================================================

def load_hover(path: str):
    global H
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit(f"HOVER.py not found at {path} (use --hover_py)")
    spec = importlib.util.spec_from_file_location("hover_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hover_main"] = mod
    spec.loader.exec_module(mod)
    H = mod
    return mod


def _abs(base: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def read_runs(path: str) -> Dict[str, Any]:
    with open(path) as f:
        spec = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for b, d in spec.get("backbones", {}).items():
        for k in ("hover", "glide", "vanilla", "calibration"):
            d[k] = _abs(base, d.get(k))
    for k in ("root", "gt_json", "glide_repo"):
        if spec.get(k):
            spec[k] = _abs(base, spec[k])
    return spec


class Ctx:
    """Per-backbone context: config, records, ground truth, judge and VLM (lazy)."""

    def __init__(self, spec: Dict[str, Any], backbone: str, args: argparse.Namespace,
                 vlm=None, judge=None):
        self.spec, self.b, self.args = spec, backbone, args
        bd = spec["backbones"][backbone]
        self.files = bd
        base = H.Config()
        over = {k: spec[k] for k in ("root", "gt_json", "glide_repo", "work_subdir", "judge_model",
                                     "judge_api_base", "backend", "judge_backend")
                if spec.get(k)}
        cfg = H.Config(**{**asdict(base), **over}).with_backbone(backbone)
        over_b = {}
        if bd.get("api_base"):
            over_b["api_base"] = bd["api_base"]
        if args.n_bootstrap:
            over_b["n_bootstrap"] = args.n_bootstrap
        if bd.get("model"):
            over_b["model"] = bd["model"]
        self.cfg = H.Config(**{**asdict(cfg), **over_b})
        self.recs = H.load_records(self.cfg)
        self.splits = H.load_splits(self.cfg, self.recs)
        self.test = self.splits["test"]
        self.by_name = {r.video_name: r for r in self.test}
        self.gt = H.load_gt_official(self.cfg, self.test)
        self._vlm, self._judge = vlm, judge
        self._wins: Dict[str, Any] = {}
        self._runs: Dict[str, Dict[str, Dict]] = {}
        self._cal = None

    # -- lazy resources --------------------------------------------------------
    @property
    def judge(self):
        if self._judge is None:
            self._judge = H.make_judge(self.cfg)
        return self._judge

    @property
    def vlm(self):
        if self._vlm is None:
            self._vlm = H.make_vlm(self.cfg)
        return self._vlm

    def wins(self, rec) -> List[Any]:
        if rec.vid not in self._wins:
            self._wins.update(H.prepare_split([rec], self.cfg))
        return self._wins[rec.vid]

    def duration(self, name: str) -> float:
        r = self.by_name[name]
        if r.duration <= 0:
            self.wins(r)
        return float(r.duration)

    def run(self, key: str) -> Dict[str, Dict]:
        """video_name -> output dict of a run file (hover / glide / vanilla)."""
        if key not in self._runs:
            path = self.files.get(key)
            H.require(path and os.path.exists(path), f"{self.b}: run file '{key}' missing: {path}")
            d = H.read_json(path)
            self._runs[key] = {o["video_name"]: o for o in d["outputs"]}
            self._runs[key + "__meta"] = d
        return self._runs[key]

    def run_meta(self, key: str) -> Dict:
        self.run(key)
        return self._runs[key + "__meta"]

    def calibration(self) -> Dict:
        if self._cal is None:
            p = self.files.get("calibration")
            H.require(p and os.path.exists(p), f"{self.b}: calibration file missing: {p}")
            self._cal = H.read_json(p)
        return self._cal

    def main_mode(self) -> str:
        cfgd = self.run_meta("hover").get("config", {})
        return str(self.files.get("mode") or cfgd.get("refine_mode") or self.calibration()["refine_mode"])

    def hover_cfg(self):
        """Config of the main run: calibration parameters + the scored refine mode."""
        cal = self.calibration()
        sw = cal.get("config_switches", {})
        return H.Config(**{**asdict(self.cfg), **{k: v for k, v in sw.items() if hasattr(self.cfg, k)},
                           "fusion": cal["fusion"], "decision_threshold": float(cal["decision_threshold"]),
                           "seg_tau": float(cal["seg_tau"]), "seg_penalty": float(cal["seg_penalty"]),
                           "refine_mode": self.main_mode()})


# =============================================================================
# Evaluation helpers (official protocol through HOVER.py)
# =============================================================================

def pred(bugs: Sequence[str], spans: Sequence[Sequence[Sequence[float]]]) -> Dict[str, Any]:
    return {"bugs": list(bugs), "time_nodes": [[list(map(float, iv)) for iv in sp] for sp in spans],
            "no_bugs": not bugs}


def prefetch_ratings(ctx: Ctx, pairs) -> None:
    """Fill the judge cache for (gt, pred) description pairs with concurrent requests.
    Ratings and cache keys are those of judge.rate, so results do not change."""
    pairs = list(dict.fromkeys(pairs))
    workers = int(os.environ.get("JUDGE_WORKERS", "16"))
    if ctx.judge.name == "mock" or workers <= 1 or len(pairs) < 2:
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda gp: ctx.judge.rate(*gp), pairs))


def evaluate(ctx: Ctx, preds: Dict[str, Dict], names: Sequence[str]) -> List[Dict]:
    empty = pred([], [])
    prefetch_ratings(ctx, [(g, d) for v in names for d in preds.get(v, empty)["bugs"]
                           for g in ctx.gt[v].get("bugs", [])])
    return [dict(H.official_eval_video(ctx.judge, ctx.gt[v], preds.get(v, empty)), video_name=v)
            for v in names]


def summarise(per: List[Dict]) -> Dict[str, float]:
    a = H.official_aggregate(per)
    n = len(per)
    return {"n_videos": n, "num_pred_bugs": a["num_pred_bugs"], "num_matched": a["num_matched"],
            "mean_score": a["mean_score"], "precision": a["precision"], "recall": a["recall"],
            "f1": a["f1"], "iou_r": (a["precision_iou"] / a["precision"]) if a["precision"] > 0 else float("nan"),
            "mean_iou": a["mean_iou"], "f1_iou": a["f1_iou"],
            "reports_per_clip": a["num_pred_bugs"] / n if n else float("nan"),
            "clips_multi": sum(1 for v in per if v["pred_num"] > 1)}


def paired(ctx: Ctx, ref: List[Dict], var: List[Dict], groups: List[str]) -> Dict[str, float]:
    c = ctx.cfg
    bs = H.paired_bootstrap(ref, var, "f1_iou", c.n_bootstrap, c.seed, groups)
    pm = H.paired_cluster_permutation(ref, var, "f1_iou", c.n_permutations, c.seed, groups)
    return {"delta_ref_minus": bs["delta"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p": pm["p"]}


def family(ctx: Ctx, tag: str, ref_name: str, ref_preds: Dict[str, Dict],
           variants: List[Tuple[str, Dict[str, Dict]]], names: Sequence[str],
           reference_label: Optional[str] = None,
           ref_per: Optional[List[Dict]] = None) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Score a reference and its variants on the same clips; paired tests vs the
    reference, Holm over the family. Returns CSV rows and the per-video results."""
    names = sorted(names)
    groups = [ctx.by_name[v].game for v in names]
    ref = ref_per if ref_per is not None else evaluate(ctx, ref_preds, names)
    per = {ref_name: ref}
    rows = [{"backbone": ctx.b, "tag": tag, "variant": ref_name, **summarise(ref),
             "reference": reference_label or ref_name}]
    pv = {}
    for name, pr in variants:
        vr = evaluate(ctx, pr, names)
        per[name] = vr
        st = paired(ctx, ref, vr, groups)
        pv[name] = st["p"]
        rows.append({"backbone": ctx.b, "tag": tag, "variant": name, **summarise(vr), **st,
                     "reference": reference_label or ref_name})
    holm = H.holm_bonferroni(pv) if pv else {}
    for r in rows[1:]:
        r["holm_reject"] = bool(holm[r["variant"]]["reject"])
    return rows, per


def write_rows(path: str, rows: List[Dict], key: Tuple[str, ...] = ("backbone", "tag")) -> None:
    """Merge rows into a CSV: rows of the same (backbone, tag) are replaced."""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = list(csv.DictReader(f))
    done = {tuple(str(r[k]) for k in key) for r in rows}
    keep = [r for r in old if tuple(str(r.get(k, "")) for k in key) not in done]
    fields = FIELDS if set(FIELDS) >= set(rows[0]) else list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in keep + rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    print(f"  wrote {len(rows)} rows -> {path}")


def _fmt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6f}"
    return v


# =============================================================================
# Span utilities
# =============================================================================

def spans_of(o: Dict) -> List[List[List[float]]]:
    return [[list(map(float, iv)) for iv in sp] for sp in o.get("time_nodes", [])]


def span_len(sp: Sequence[Sequence[float]]) -> float:
    return sum(e - s for s, e in H.normalize_intervals([tuple(map(float, iv)) for iv in sp]))


def span_start(sp: Sequence[Sequence[float]]) -> float:
    return min((float(iv[0]) for iv in sp), default=float("inf"))


def overlaps(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> bool:
    return any(min(x[1], y[1]) > max(x[0], y[0]) for x in a for y in b)


def whole(ctx: Ctx, name: str) -> List[List[float]]:
    return [[0.0, round(ctx.duration(name), 2)]]


# =============================================================================
# E1-E3: controls on the agent's outputs, the vanilla baseline, fixed sentences
# =============================================================================

def glide_variants(ctx: Ctx, names: Sequence[str]) -> List[Tuple[str, Dict[str, Dict]]]:
    G = ctx.run("glide")
    out = {k: {} for k in ("glide", "glide_whole", "glide_nocopy", "glide_merge", "glide_first",
                           "glide_longest")}
    for v in names:
        o = G.get(v, {"bugs": [], "time_nodes": []})
        bugs, sps = list(o.get("bugs", [])), spans_of(o)
        sps += [[] for _ in range(len(bugs) - len(sps))]
        out["glide"][v] = pred(bugs, sps)
        out["glide_whole"][v] = pred(bugs, [whole(ctx, v) for _ in bugs])
        keep = [i for i, d in enumerate(bugs) if not any(k in d.lower() for k in COPY_KEYS)]
        out["glide_nocopy"][v] = pred([bugs[i] for i in keep], [sps[i] for i in keep])
        # merge reports whose spans overlap (connected components), keep the earliest description
        parent = list(range(len(bugs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(bugs)):
            for j in range(i + 1, len(bugs)):
                if overlaps(sps[i], sps[j]):
                    parent[find(i)] = find(j)
        comp = defaultdict(list)
        for i in range(len(bugs)):
            comp[find(i)].append(i)
        mb, ms = [], []
        for members in sorted(comp.values(), key=lambda m: min(span_start(sps[i]) for i in m)):
            first = min(members, key=lambda i: (span_start(sps[i]), i))
            mb.append(bugs[first])
            ms.append([list(iv) for iv in H.normalize_intervals(
                [tuple(iv) for i in members for iv in sps[i]])])
        out["glide_merge"][v] = pred(mb, ms)
        if bugs:
            i1 = min(range(len(bugs)), key=lambda i: (span_start(sps[i]), i))
            il = max(range(len(bugs)), key=lambda i: (span_len(sps[i]), -i))
            out["glide_first"][v] = pred([bugs[i1]], [sps[i1]])
            out["glide_longest"][v] = pred([bugs[il]], [sps[il]])
        else:
            out["glide_first"][v] = out["glide_longest"][v] = pred([], [])
    return [(k, out[k]) for k in ("glide", "glide_whole", "glide_nocopy", "glide_merge",
                                  "glide_first", "glide_longest")]


def glide_emission(ctx: Ctx, names: Sequence[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """HOVER's stages V and U applied to GliDe's reports: each report becomes a
    hypothesis (members = windows its spans overlap, support = |members|), is
    verified on every window, featurised exactly as in HOVER.py and emitted iff the
    main run's selector predicts u_hat >= t*. At most max_hypotheses reports per clip
    (ranked by support, as in stage H) are considered."""
    G, hc = ctx.run("glide"), ctx.hover_cfg()
    tau, lam = hc.seg_tau, hc.seg_penalty
    keep_span, keep_whole = {}, {}
    for v in names:
        o = G.get(v, {"bugs": [], "time_nodes": []})
        bugs, sps = list(o.get("bugs", [])), spans_of(o)
        rec = ctx.by_name[v]
        if not bugs:
            keep_span[v] = keep_whole[v] = pred([], [])
            continue
        wins = ctx.wins(rec)
        gen = H.collect_generic_evidence(ctx.vlm, rec, wins, hc)
        cands = []
        for i, d in enumerate(bugs):
            sp = sps[i] if i < len(sps) else []
            mem = [w.idx for w in wins if any(min(w.end, b) > max(w.start, a) for a, b in sp)]
            if not mem:
                continue
            prior, prof = (H.verify_profile(ctx.vlm, rec, wins, d, GENERIC_ENTITY, hc)
                           if hc.use_hypothesis_scoring else (0.0, list(gen.lo_abs)))
            cands.append((i, H.Hypothesis(d, GENERIC_ENTITY, "", mem, False, float(prior),
                                          [float(x) for x in prof])))
        cands.sort(key=lambda c: (-len(c[1].members), c[0]))
        cands = cands[:hc.max_hypotheses]
        rows = [H.hypothesis_features(h, gen, k + 1, hc, tau, lam) for k, (_, h) in enumerate(cands)]
        util = H.fusion_predict(hc.fusion, rows) if rows else np.zeros(0)
        emit = [cands[k][0] for k in np.argsort(-util, kind="mergesort")
                if util[k] >= hc.decision_threshold][:hc.max_events_per_video]
        keep_span[v] = pred([bugs[i] for i in emit], [sps[i] for i in emit])
        keep_whole[v] = pred([bugs[i] for i in emit], [whole(ctx, v) for _ in emit])
    return keep_span, keep_whole


def text_medoid(texts: Sequence[str]) -> int:
    """Index of the text with the highest mean TF-IDF cosine similarity to the others."""
    toks = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
    if not vocab:
        return 0
    df = np.zeros(len(vocab))
    for t in toks:
        for w in set(t):
            df[vocab[w]] += 1
    idf = np.log((1 + len(toks)) / (1 + df)) + 1
    X = np.zeros((len(toks), len(vocab)))
    for i, t in enumerate(toks):
        for w in t:
            X[i, vocab[w]] += 1
    X *= idf
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    return int(np.argmax(S.sum(1)))


def judge_medoid(ctx: Ctx, texts: Sequence[str], shortlist: int) -> str:
    """Medoid under the judge: the development description with the highest mean
    rating when used as the prediction for every other development description.
    Candidates are short-listed by TF-IDF centrality to bound the judge calls
    (shortlist x number of descriptions ratings)."""
    texts = list(dict.fromkeys(texts))
    if len(texts) == 1:
        return texts[0]
    order = list(range(len(texts)))
    if shortlist and len(texts) > shortlist:
        order = [int(i) for i in np.argsort(-_tfidf_centrality(texts), kind="mergesort")[:shortlist]]
    best, best_s = texts[order[0]], -1.0
    prefetch_ratings(ctx, [(texts[j], texts[i]) for i in order for j in range(len(texts)) if j != i])
    for i in order:
        s = float(np.mean([ctx.judge.rate(texts[j], texts[i]) for j in range(len(texts)) if j != i]))
        if s > best_s:
            best, best_s = texts[i], s
    print(f"    development medoid (mean rating {best_s:.3f}): {best}")
    return best


def _tfidf_centrality(texts: Sequence[str]) -> np.ndarray:
    toks = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
    X = np.zeros((len(toks), max(1, len(vocab))))
    df = np.zeros(max(1, len(vocab)))
    for t in toks:
        for w in set(t):
            df[vocab[w]] += 1
    idf = np.log((1 + len(toks)) / (1 + df)) + 1
    for i, t in enumerate(toks):
        for w in t:
            X[i, vocab[w]] += 1
    X *= idf
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    return S.sum(1)


def cmd_controls(ctx: Ctx, out_dir: str) -> None:
    print(f"[controls] {ctx.b}")
    Hm, V = ctx.run("hover"), ctx.run("vanilla")
    hover_preds = {v: pred(o.get("bugs", []), spans_of(o)) for v, o in Hm.items()}
    G = ctx.run("glide")
    test_names = set(ctx.by_name) & set(ctx.gt)
    rows = []
    # E1: agent outputs (clips with output from HOVER and GliDe)
    names1 = sorted(test_names & set(Hm) & set(G))
    var = glide_variants(ctx, names1)
    es, ew = glide_emission(ctx, names1)
    var += [("glide_emit_span", es), ("glide_emit_whole", ew)]
    r1, per1 = family(ctx, "E1_glide_controls", "hover", hover_preds, var, names1)
    rows += r1
    # E4 reference: HOVER's own reports with whole-clip spans on the same clips (no test)
    hw = {v: pred(hover_preds[v]["bugs"], [whole(ctx, v) for _ in hover_preds[v]["bugs"]]) for v in names1}
    rows.append({"backbone": ctx.b, "tag": "E4_reference", "variant": "hover_whole",
                 **summarise(evaluate(ctx, hw, names1)), "reference": "hover"})
    # E2: vanilla controls
    names2 = sorted(test_names & set(Hm) & set(V))
    vp, vt = {}, {}
    for v in names2:
        o = V.get(v, {"bugs": [], "time_nodes": []})
        bugs, sps = list(o.get("bugs", [])), spans_of(o)
        sps += [[] for _ in range(len(bugs) - len(sps))]
        vp[v] = pred(bugs, sps)
        if bugs:
            i1 = min(range(len(bugs)), key=lambda i: (span_start(sps[i]), i))
            vt[v] = pred([bugs[i1]], [whole(ctx, v)])
        else:
            vt[v] = pred([], [])
    r2, _ = family(ctx, "E2_vanilla_controls", "hover", hover_preds,
                   [("vanilla", vp), ("vanilla_top1_whole", vt)], names2)
    rows += r2
    # E3: one fixed sentence per clip, whole-clip span (no video input)
    names3 = sorted(test_names & set(Hm))
    dev = ctx.splits["dev"]
    dev_texts = [d for r in dev for d in r.bugs if str(d).strip()]
    medoid = judge_medoid(ctx, dev_texts, ctx.args.medoid_shortlist)
    by_game_dev = defaultdict(list)
    for r in dev:
        by_game_dev[r.game].extend([d for d in r.bugs if str(d).strip()])
    by_game_test = defaultdict(list)
    for r in ctx.test:
        by_game_test[r.game].append((r.video_name, [d for d in r.bugs if str(d).strip()]))
    const = {k: {} for k in ("const_generic", "const_glide_example", "const_dev_medoid", "const_game_loo")}
    for v in names3:
        w = whole(ctx, v)
        const["const_generic"][v] = pred([GENERIC_SENTENCE], [w])
        const["const_glide_example"][v] = pred([GLIDE_EXAMPLE], [w])
        const["const_dev_medoid"][v] = pred([medoid], [w])
        g = ctx.by_name[v].game
        pool = list(by_game_dev[g])
        if ctx.args.game_pool == "loo":     # also the other TEST clips of the game (leave-one-out)
            pool += [d for name, ds in by_game_test[g] if name != v for d in ds]
        sent = pool[text_medoid(pool)] if pool else medoid
        const["const_game_loo"][v] = pred([sent], [w])
    r3, _ = family(ctx, "E3_constant", "hover", hover_preds,
                   [(k, const[k]) for k in ("const_generic", "const_glide_example",
                                            "const_dev_medoid", "const_game_loo")], names3)
    rows += r3
    write_rows(os.path.join(out_dir, "revision_controls.csv"), rows)


# =============================================================================
# E5 post-hoc: emission rule x span resolution from the cached hypotheses
# =============================================================================

def cached_hypotheses(o: Dict) -> List[Any]:
    return [H.Hypothesis(h["description"], h.get("entity", ""), h.get("glitch_type", ""),
                         list(h["members"]), bool(h.get("forced", False)), float(h.get("prior", 0.0)),
                         [float(x) for x in h["profile"]]) for h in o.get("hypotheses", [])]


def cmd_posthoc(ctx: Ctx, out_dir: str) -> None:
    print(f"[posthoc] {ctx.b}")
    Hm, hc = ctx.run("hover"), ctx.hover_cfg()
    tau, lam, thr = hc.seg_tau, hc.seg_penalty, hc.decision_threshold
    names = sorted(set(Hm) & set(ctx.by_name) & set(ctx.gt))
    preds = defaultdict(dict)
    audits = defaultdict(lambda: {"monotonicity_violations": 0, "frame_conflicts": 0})
    for v in names:
        o, rec = Hm[v], ctx.by_name[v]
        hyps = cached_hypotheses(o)
        H.require(hyps or not o.get("bugs"), f"{v}: run file has no cached 'hypotheses'")
        util = np.array([float(h.get("utility", 0.0)) for h in o.get("hypotheses", [])])
        order = list(np.argsort(-util, kind="mergesort"))
        rules = {"selector": [int(i) for i in order if util[i] >= thr][:hc.max_events_per_video],
                 "top1": [0] if hyps else [],
                 "all": [int(i) for i in order][:hc.max_events_per_video]}
        wins = ctx.wins(rec) if hyps else []
        for rule, idx in rules.items():
            for mode in H.REFINE_MODES:
                bugs, sps = [], []
                for i in idx:
                    sp, au = H.hypothesis_spans(ctx.vlm, rec, wins, hyps[i], hc, tau, lam, mode)
                    for k in au:
                        audits[f"{rule}_{mode}"][k] += au[k]
                    if sp:
                        bugs.append(hyps[i].description)
                        sps.append(H.to_time_nodes(sp, hc))
                preds[f"{rule}_{mode}"][v] = pred(bugs, sps)
    ref = f"selector_{ctx.main_mode()}"
    variants = [(k, preds[k]) for k in sorted(preds) if k != ref]
    rows, per = family(ctx, "E5_posthoc", ref, preds[ref], variants, names, reference_label=ref)
    main_per = evaluate(ctx, {v: pred(Hm[v].get("bugs", []), spans_of(Hm[v])) for v in names}, names)
    main_row = {"backbone": ctx.b, "tag": "E5_posthoc", "variant": "main", **summarise(main_per),
                "reference": ref}
    a, b_ = summarise(main_per)["f1_iou"], summarise(per[ref])["f1_iou"]
    if abs(a - b_) > 1e-9:
        print(f"  WARNING: recomputed {ref} ({b_:.4f}) differs from the run file ({a:.4f}); "
              "check that runs.json points to the calibration of this run")
    write_rows(os.path.join(out_dir, "revision_posthoc.csv"), [main_row] + rows)
    rep = [{"backbone": ctx.b, "variant": k, **audits[k]} for k in sorted(audits)]
    _write_simple(os.path.join(out_dir, "revision_repairs.csv"), rep,
                  ["backbone", "variant", "monotonicity_violations", "frame_conflicts"], ("backbone",))


def _write_simple(path: str, rows: List[Dict], fields: List[str], key: Tuple[str, ...]) -> None:
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = list(csv.DictReader(f))
    done = {tuple(str(r[k]) for k in key) for r in rows}
    keep = [r for r in old if tuple(str(r.get(k, "")) for k in key) not in done]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in keep + rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    print(f"  wrote {len(rows)} rows -> {path}")


# =============================================================================
# E4: decomposition against the GliDe re-run (needs the controls)
# =============================================================================

def cmd_decompose(out_dir: str, backbones: Sequence[str]) -> None:
    print("[decompose]")
    ctrl = _read(os.path.join(out_dir, "revision_controls.csv"))
    rows = []
    for b in backbones:
        def g(rows_, tag, var, k):
            for r in rows_:
                if r["backbone"] == b and r["tag"] == tag and r["variant"] == var:
                    return float(r[k])
            raise KeyError((b, tag, var))
        if not any(r["backbone"] == b for r in ctrl):
            print(f"  {b}: no control rows, skipped")
            continue
        Hf, Hi = g(ctrl, "E1_glide_controls", "hover", "f1_iou"), g(ctrl, "E1_glide_controls", "hover", "iou_r")
        Gf, Gi = g(ctrl, "E1_glide_controls", "glide", "f1_iou"), g(ctrl, "E1_glide_controls", "glide", "iou_r")
        F1h, F1g = g(ctrl, "E1_glide_controls", "hover", "f1"), g(ctrl, "E1_glide_controls", "glide", "f1")
        Gwi = g(ctrl, "E1_glide_controls", "glide_whole", "iou_r")
        Hwi = g(ctrl, "E4_reference", "hover_whole", "iou_r")
        tot = math.log(Hf / Gf)
        parts = {"log_f1": math.log(F1h / F1g), "log_glide_fragmentation": math.log(Gwi / Gi),
                 "log_matched_set": math.log(Hwi / Gwi), "log_hover_localisation": math.log(Hi / Hwi)}
        r = {"backbone": b, "total_log_ratio": tot, **parts,
             "share_f1": parts["log_f1"] / tot, "share_glide_fragmentation": parts["log_glide_fragmentation"] / tot,
             "share_matched_set": parts["log_matched_set"] / tot,
             "share_hover_localisation": parts["log_hover_localisation"] / tot,
             "identity_residual": tot - sum(parts.values())}
        rows.append(r)
    if rows:
        _write_simple(os.path.join(out_dir, "revision_decompose.csv"), rows, list(rows[0]), ("backbone",))


def _read(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# =============================================================================
# E5 ablations: each variant re-tuned on dev and run on test, paired with the MAIN run
# =============================================================================

def cmd_ablate(ctx: Ctx, out_dir: str) -> None:
    print(f"[ablate] {ctx.b}")
    Hm = ctx.run("hover")
    names = sorted(set(Hm) & set(ctx.by_name) & set(ctx.gt))
    base = ctx.cfg
    variants = []
    only = [t for t in getattr(ctx.args, "ablate_variants", "").split(",") if t]
    for tag, sw in ABLATION_SWITCHES.items():
        if only and tag not in only:
            continue
        vcfg = H.Config(**{**asdict(base), **sw, "fail_on_degenerate": False})
        cal = H.cmd_calibrate(vcfg, ctx.vlm, ctx.splits, "dev", tag, ctx.judge)
        vcfg = H.Config(**{**asdict(vcfg), "refine_mode": cal["refine_mode"]})
        path = H.cmd_run(vcfg, ctx.vlm, ctx.splits, "HOVER", "test", tag)
        d = {o["video_name"]: o for o in H.read_json(path)["outputs"]}
        variants.append((tag, {v: pred(d[v].get("bugs", []), spans_of(d[v])) for v in names if v in d}))
    if only:    # warm-up of single variants (parallel processes); the CSV is written by a full pass
        print(f"  variants {only} done (no CSV written)")
        return
    rows, _ = family(ctx, "E5_ablate_main", "full",
                     {v: pred(Hm[v].get("bugs", []), spans_of(Hm[v])) for v in names}, variants, names,
                     reference_label="full")
    write_rows(os.path.join(out_dir, "revision_ablation.csv"), rows)


# =============================================================================
# E6: extended segmentation grid on the development split
# =============================================================================

def cmd_seg_grid(ctx: Ctx, out_dir: str) -> None:
    print(f"[seg_grid] {ctx.b}")
    hc = ctx.hover_cfg()
    recs = H.take_limit(ctx.splits["dev"], hc.limit, hc.stratify_limit)
    wm = H.prepare_split(recs, hc)
    ev = H.gather(hc, ctx.vlm, recs, wm, True)
    by = {r.vid: r for r in recs}
    correct = []
    for r in recs:
        for i, (j, rt) in H.match_hypotheses(hc, ctx.judge, r, wm[r.vid], ev[r.vid]).items():
            if rt >= hc.min_correct_rating:
                correct.append((r.vid, i, j))
    taus, lams = H.parse_grid(EXT_TAUS), H.parse_grid(hc.seg_penalties)

    def iou(item, t, l, mode):
        vid, i, j = item
        sp, _ = H.hypothesis_spans(ctx.vlm, by[vid], wm[vid], ev[vid].hyps[i], hc, t, l, mode)
        return H.official_iou(H.gt_bug_list(by[vid])[j]["spans"], [list(x) for x in sp])
    rows, best = [], (-1.0, None, None)
    for t in taus:
        for l in lams:
            m = float(np.mean([iou(it, t, l, "window") for it in correct])) if correct else float("nan")
            rows.append({"backbone": ctx.b, "tau": t, "lambda": l, "mean_iou_correct": m})
            if m > best[0]:
                best = (m, t, l)
    for mode in H.REFINE_MODES:
        m = float(np.mean([iou(it, best[1], best[2], mode) for it in correct])) if correct else float("nan")
        rows.append({"backbone": ctx.b, "tau": f"ref:{mode}", "lambda": "", "mean_iou_correct": m})
    _write_simple(os.path.join(out_dir, "revision_seg_grid.csv"), rows,
                  ["backbone", "tau", "lambda", "mean_iou_correct"], ("backbone",))


# =============================================================================
# Re-judging: A0 = cached judging, A1..A3 = fresh judge samples (separate caches)
# =============================================================================

def _weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 6) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.size == 0:
        return float("nan")
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def cmd_rejudge(ctx: Ctx, out_dir: str, repeats: int) -> None:
    print(f"[rejudge] {ctx.b}")
    runs = {"crux": ctx.run("hover"), "glide_official": ctx.run("glide"), "vanilla": ctx.run("vanilla")}
    names = sorted(set.intersection(*[set(r) for r in runs.values()]) & set(ctx.by_name) & set(ctx.gt))
    groups = [ctx.by_name[v].game for v in names]
    preds = {m: {v: pred(r[v].get("bugs", []), spans_of(r[v])) for v in names} for m, r in runs.items()}
    base_dir = ctx.judge.cache_dir
    rows, scores, ratings = [], defaultdict(list), {}
    for a in range(repeats + 1):
        ctx.judge.cache_dir = base_dir if a == 0 else f"{base_dir}__repeat{a}"
        per = {m: evaluate(ctx, preds[m], names) for m in runs}
        ratings[a] = [x for m in runs for v in per[m] for row in v.get("score_matrix", []) for x in row]
        pv, st = {}, {}
        for m in ("glide_official", "vanilla"):
            st[m] = paired(ctx, per["crux"], per[m], groups)
            pv[m] = st[m]["p"]
        holm = H.holm_bonferroni(pv)
        for m in runs:
            f = H.official_aggregate(per[m])["f1_iou"]
            scores[m].append(f)
            r = {"backbone": ctx.b, "judging": f"A{a}", "method": m, "f1_iou": f}
            if m in st:
                r.update(delta_ref_minus=st[m]["delta_ref_minus"], p=st[m]["p"],
                         holm_reject=bool(holm[m]["reject"]))
            rows.append(r)
    ctx.judge.cache_dir = base_dir
    for m in runs:
        rows.append({"backbone": ctx.b, "judging": "sd_A_repeats", "method": m,
                     "f1_iou": float(np.std(scores[m], ddof=1)) if len(scores[m]) > 1 else float("nan")})
    kap = [_weighted_kappa(ratings[0], ratings[a]) for a in range(1, repeats + 1)
           if len(ratings[a]) == len(ratings[0])]
    rows.append({"backbone": ctx.b, "judging": "pair_kappa_A_repeat", "method": "",
                 "f1_iou": float(np.mean(kap)) if kap else float("nan")})
    _write_simple(os.path.join(out_dir, "revision_rejudge.csv"), rows,
                  ["backbone", "judging", "method", "f1_iou", "delta_ref_minus", "p", "holm_reject"],
                  ("backbone",))


# =============================================================================
# Window probe on 72 development clips (HOVER.py 'probe'), exported to CSV
# =============================================================================

def cmd_probe(ctx: Ctx, out_dir: str, limit: int) -> None:
    print(f"[probe] {ctx.b}")
    cfg = H.Config(**{**asdict(ctx.cfg), "limit": limit})
    d = H.cmd_diagnose(cfg, ctx.vlm, ctx.splits, "dev", hypotheses=True)
    c = d.get("hyp_vs_generic", {})
    rows = []
    for sig in ("hyp_max", "hyp_top1", "lo_abs", "lo_raw"):
        s = d["signals"][sig]
        rows.append({"backbone": ctx.b, "signal": sig,
                     "within_auc": s["within_video_auc"]["auc"], "within_lo": s["within_video_auc"]["lo"],
                     "within_hi": s["within_video_auc"]["hi"], "pooled_auc": s["pooled_auc"]["auc"],
                     "pooled_lo": s["pooled_auc"]["lo"], "pooled_hi": s["pooled_auc"]["hi"],
                     "icc": s["variance"]["icc"],
                     "delta_per_video_auc": c.get("mean_delta_per_video_auc", float("nan")),
                     "delta_lo": (c.get("game_bootstrap_ci") or [float("nan")] * 2)[0],
                     "delta_hi": (c.get("game_bootstrap_ci") or [float("nan")] * 2)[1],
                     "wilcoxon_p": c.get("wilcoxon_p", float("nan")),
                     "n_windows": d["base"]["n_windows"], "n_videos": d["n_videos"]})
    _write_simple(os.path.join(out_dir, "revision_probe.csv"), rows, list(rows[0]), ("backbone",))


# =============================================================================
# Self-test on HOVER.py's synthetic benchmark with the mock VLM and judge
# =============================================================================

def selftest() -> None:
    import shutil
    print("== hover_revision selftest (mock VLM / judge, synthetic benchmark) ==")
    root = tempfile.mkdtemp(prefix="hover_rev_")
    H._build_synthetic_benchmark(root)
    base = H.Config(root=root, backend="mock", judge_backend="mock", n_bootstrap=200, cv_folds=4)
    recs = H.load_records(base)
    spl = H.load_splits(base, recs)
    wm = H.prepare_split(spl["dev"] + spl["test"] + spl["train"], base)
    label_of, frame_label_of = {}, {}
    for r in spl["dev"] + spl["test"]:
        gt_iv = [iv for spans in r.time_nodes for iv in spans]
        for w_, y_ in zip(wm[r.vid], H.window_labels(r, wm[r.vid], base.label_min_overlap)):
            label_of[os.path.abspath(w_.stitch_path)] = int(y_)
            d_ = os.path.dirname(os.path.abspath(w_.stitch_path))
            for fid, ft in zip(w_.frame_ids, w_.frame_times):
                frame_label_of[(d_, fid)] = int(any(a_ <= ft < b_ for a_, b_ in gt_iv))
    vbias = lambda v: H._noise("video-bias" + v)
    oracle = H._mock_oracle(label_of, vbias, True, frame_label_of=frame_label_of)
    vlm = H.make_vlm(base, oracle)
    judge = H.make_judge(base, lambda g, p: 5 if ("float" in p.lower() and "float" in g.lower()) else
                         (2 if "car" in p.lower() and "car" in g.lower() else 0))
    H.cmd_calibrate(base, vlm, spl, "dev", "", judge)
    cal_path = os.path.join(base.paths()["params"], "calibration.json")
    cal = H.read_json(cal_path)
    run_cfg = H.Config(**{**asdict(base), "refine_mode": cal["refine_mode"]})
    f_h = H.cmd_run(run_cfg, vlm, spl, "HOVER", "test")
    f_v = H.cmd_run(run_cfg, vlm, spl, "vanilla", "test")
    gl = []
    for k, r in enumerate(spl["test"][:-1]):
        bugs = ["Something odd happens.", "The red car floats above the road, near the highway overpass."]
        tns = [[[0, 1]], [[3, 9]]]
        if k % 2:
            bugs.append("A car floats in the air.")
            tns.append([[4, 7]])
        gl.append({"video_name": r.video_name, "game_name": r.game, "no_bugs": False, "bugs": bugs,
                   "time_nodes": tns})
    gpath = os.path.join(root, "glide_batch_report.json")
    H.atomic_write_json(gpath, gl)
    f_g = H.cmd_import_glide(base, spl, "test", gpath)
    spec = {"root": root, "backend": "mock", "judge_backend": "mock",
            "backbones": {"qwen2.5-vl-7b": {"hover": f_h, "glide": f_g, "vanilla": f_v,
                                            "calibration": cal_path}}}
    args = argparse.Namespace(n_bootstrap=200, medoid_shortlist=5, game_pool="dev", repeats=1,
                              probe_limit=None)
    ctx = Ctx(spec, "qwen2.5-vl-7b", args, vlm=vlm, judge=judge)
    ctx.cfg = H.Config(**{**asdict(ctx.cfg), "root": root, "backend": "mock", "judge_backend": "mock",
                          "n_bootstrap": 200, "cv_folds": 4})
    out = os.path.join(root, "revision_results")
    checks = []

    def chk(name, cond, detail=""):
        checks.append(bool(cond))
        print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    cmd_controls(ctx, out)
    cmd_decompose(out, ["qwen2.5-vl-7b"])
    cmd_posthoc(ctx, out)
    cmd_rejudge(ctx, out, 1)
    cmd_seg_grid(ctx, out)
    cmd_probe(ctx, out, 16)
    cmd_ablate(ctx, out)
    C = _read(os.path.join(out, "revision_controls.csv"))
    g = lambda tag, var, k: float([r for r in C if r["tag"] == tag and r["variant"] == var][0][k])
    chk("whole-clip spans leave F1 unchanged (E1)",
        abs(g("E1_glide_controls", "glide", "f1") - g("E1_glide_controls", "glide_whole", "f1")) < 1e-9)
    chk("prompt-example reports removed",
        g("E1_glide_controls", "glide_nocopy", "num_pred_bugs") < g("E1_glide_controls", "glide", "num_pred_bugs"))
    chk("first/longest emit at most one report per clip",
        g("E1_glide_controls", "glide_first", "clips_multi") == 0 and
        g("E1_glide_controls", "glide_longest", "clips_multi") == 0)
    chk("fixed sentences: one report per clip",
        all(g("E3_constant", v, "reports_per_clip") == 1.0 for v in
            ("const_generic", "const_glide_example", "const_dev_medoid", "const_game_loo")))
    chk("controls reference equals the main-run HOVER file",
        abs(g("E1_glide_controls", "hover", "f1_iou") -
            summarise(evaluate(ctx, {v: pred(o.get("bugs", []), spans_of(o)) for v, o in ctx.run("hover").items()},
                               sorted(set(ctx.run("hover")) & set(ctx.run("glide")))))["f1_iou"]) < 1e-12)
    P = _read(os.path.join(out, "revision_posthoc.csv"))
    ref = P[0]["reference"]
    m = [r for r in P if r["variant"] == "main"][0]
    s = [r for r in P if r["variant"] == ref][0]
    chk("post-hoc selector in the run's mode reproduces the run file",
        abs(float(m["f1_iou"]) - float(s["f1_iou"])) < 1e-9, f"{m['f1_iou']} vs {s['f1_iou']}")
    D = _read(os.path.join(out, "revision_decompose.csv"))
    chk("decomposition identity holds (CSV values have 6 decimals)", D and abs(float(D[0]["identity_residual"])) < 1e-4)
    J = _read(os.path.join(out, "revision_rejudge.csv"))
    chk("re-judging wrote A0 and A1", {r["judging"] for r in J} >= {"A0", "A1", "sd_A_repeats"})
    A = _read(os.path.join(out, "revision_ablation.csv"))
    chk("ablation wrote four variants + full", len(A) == 5)
    chk("probe and seg grid written",
        os.path.exists(os.path.join(out, "revision_probe.csv")) and
        os.path.exists(os.path.join(out, "revision_seg_grid.csv")))
    print(f"== {sum(checks)}/{len(checks)} checks passed; outputs in {out} ==")
    if not all(checks):
        sys.exit(1)
    shutil.rmtree(root, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="HOVER revision analyses on the main run")
    ap.add_argument("command", choices=["selftest", "controls", "posthoc", "decompose", "ablate",
                                        "seg_grid", "rejudge", "probe", "all"])
    ap.add_argument("--hover_py", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOVER.py"))
    ap.add_argument("--runs", default="runs.json", help="JSON with the main-run files per backbone")
    ap.add_argument("--out", default="revision_results")
    ap.add_argument("--backbones", default="", help="comma list (default: every backbone in --runs)")
    ap.add_argument("--n_bootstrap", type=int, default=0, help="0 = HOVER.py default (2000)")
    ap.add_argument("--medoid_shortlist", type=int, default=40,
                    help="development descriptions short-listed by TF-IDF centrality before the judge medoid")
    ap.add_argument("--game_pool", choices=("dev", "loo"), default="loo",
                    help="per-game sentence from development descriptions (dev) or also from the "
                         "other test clips of the game, leaving the clip out (loo)")
    ap.add_argument("--repeats", type=int, default=3, help="re-judgings beyond the cached one")
    ap.add_argument("--ablate_variants", default="",
                    help="ablate: only run these variants (comma list) and write no CSV")
    ap.add_argument("--probe_limit", type=int, default=72)
    a = ap.parse_args()
    load_hover(a.hover_py)
    H.setup_logging(tempfile.gettempdir(), "revision")
    if a.command == "selftest":
        selftest()
        return
    spec = read_runs(a.runs)
    bbs = [b for b in (a.backbones.split(",") if a.backbones else spec["backbones"]) if b]
    os.makedirs(a.out, exist_ok=True)
    steps = {"controls": ["controls"], "posthoc": ["posthoc"], "decompose": ["decompose"],
             "ablate": ["ablate"], "seg_grid": ["seg_grid"], "rejudge": ["rejudge"], "probe": ["probe"],
             "all": ["controls", "decompose", "posthoc", "rejudge", "seg_grid", "probe", "ablate"]}[a.command]
    for step in steps:
        if step == "decompose":
            cmd_decompose(a.out, bbs)
            continue
        for b in bbs:
            ctx = Ctx(spec, b, a)
            {"controls": lambda: cmd_controls(ctx, a.out), "posthoc": lambda: cmd_posthoc(ctx, a.out),
             "ablate": lambda: cmd_ablate(ctx, a.out), "seg_grid": lambda: cmd_seg_grid(ctx, a.out),
             "rejudge": lambda: cmd_rejudge(ctx, a.out, a.repeats),
             "probe": lambda: cmd_probe(ctx, a.out, a.probe_limit)}[step]()
    print("done; copy the CSVs to paper/data/revision/ and run scripts/make_revision_tables.py")


if __name__ == "__main__":
    main()
