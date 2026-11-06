"""Timing statistics of the inference-time study (main Tab. 7, Supp. timing tables), copied verbatim from the
pipeline script that produced data/timing/report_test_100.md, so v4_timing_ci.py reproduces that report exactly."""
from __future__ import annotations

import itertools  # noqa: F401
import math
from collections import defaultdict  # noqa: F401
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class Config:   # the three settings these functions read, with the pipeline's defaults
    seed: int = 20260924
    n_bootstrap: int = 2000
    n_permutations: int = 10000

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def _game_units(games: Sequence[str]) -> List[np.ndarray]:
    gi: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(games):
        gi[g].append(i)
    return [np.asarray(v, int) for _, v in sorted(gi.items())]

def _cluster_boot(fn: Callable[[np.ndarray], float], units: List[np.ndarray], B: int,
                  seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = len(units)
    return np.asarray([fn(np.concatenate([units[u] for u in rng.integers(0, G, G)]))
                       for _ in range(B)], float)

def _t_interval(point: float, boot: np.ndarray, G: int) -> Tuple[float, float]:
    """Bootstrap-SE interval with a t(G-1) quantile and the G/(G-1) small-sample
    variance correction (same construction as cluster_bootstrap_auc)."""
    boot = boot[np.isfinite(boot)]
    if G < 2 or boot.size < 2 or not math.isfinite(point):
        return float("nan"), float("nan")
    from scipy.stats import t as _t
    se = float(np.std(boot, ddof=1)) * math.sqrt(G / (G - 1.0))
    q = float(_t.ppf(0.975, G - 1))
    return point - q * se, point + q * se

def cluster_sign_flip(d: np.ndarray, units: List[np.ndarray], B: int, seed: int) -> Dict[str, Any]:
    """Randomisation test of H0: within every game the two methods' labels are
    exchangeable, i.e. each game's paired log-time differences are symmetric
    about 0 as a block. Statistic: mean difference over videos. Exact when
    2^G <= B, otherwise Monte Carlo with the (1+k)/(1+B) correction."""
    d = np.asarray(d, float)
    D = np.array([d[u].sum() for u in units])
    G, n = len(units), d.size
    obs = float(D.sum() / n)
    tol = 1e-12
    if G <= 20 and 2 ** G <= B:
        S = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
        p, exact = float(np.mean(np.abs(S @ D / n) >= abs(obs) - tol)), True
    else:
        S = np.random.default_rng(seed).choice((1.0, -1.0), size=(B, G))
        p, exact = float((1 + np.sum(np.abs(S @ D / n) >= abs(obs) - tol)) / (1 + B)), False
    return {"mean_log_diff": obs, "p": p, "exact": exact, "n_clusters": G,
            "min_attainable_p": float(2.0 ** (1 - G)) if G else 1.0}

def summarise_times(rows: Sequence[Dict], B: int, seed: int) -> Dict[str, Any]:
    t = np.array([r["wall_s"] for r in rows], float)
    dur = np.array([r["duration"] for r in rows], float)
    units = _game_units([r["game"] for r in rows])
    G = len(units)
    lt = np.log(np.maximum(t, 1e-9))
    mean = float(t.mean())
    m_lo, m_hi = _t_interval(mean, _cluster_boot(lambda ix: t[ix].mean(), units, B, seed), G)
    m_lo = max(0.0, m_lo) if math.isfinite(m_lo) else m_lo     # times are positive
    g_lo, g_hi = _t_interval(float(lt.mean()), _cluster_boot(lambda ix: lt[ix].mean(), units, B,
                                                             seed + 1), G)
    has_dur = float(dur.sum()) > 0 and bool(np.all(dur > 0))
    rtf = float(t.sum() / dur.sum()) if has_dur else float("nan")
    r_lo, r_hi = (_t_interval(math.log(rtf), _cluster_boot(
        lambda ix: math.log(t[ix].sum() / dur[ix].sum()), units, B, seed + 2), G)
        if has_dur else (float("nan"), float("nan")))
    col = lambda k: np.array([float(r.get(k) or 0.0) for r in rows])
    stages: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for s, v in (r.get("stage_model_s") or {}).items():
            stages[s].append(float(v))
    return {"n_videos": len(rows), "n_games": G, "mean_s": mean, "mean_ci": [m_lo, m_hi],
            "sd_s": float(t.std(ddof=1)) if t.size > 1 else float("nan"),
            "median_s": float(np.median(t)), "q25_s": float(np.percentile(t, 25)),
            "q75_s": float(np.percentile(t, 75)), "p90_s": float(np.percentile(t, 90)),
            "min_s": float(t.min()), "max_s": float(t.max()),
            "geo_mean_s": float(math.exp(lt.mean())), "geo_mean_ci": [math.exp(g_lo), math.exp(g_hi)]
            if math.isfinite(g_lo) else [float("nan")] * 2,
            "rtf": rtf, "rtf_ci": [math.exp(r_lo), math.exp(r_hi)] if math.isfinite(r_lo)
            else [float("nan")] * 2,
            "total_hours": float(t.sum() / 3600.0), "video_hours": float(dur.sum() / 3600.0),
            "mean_duration_s": float(dur.mean()),
            "preprocess_s_mean": float(col("preprocess_s").mean()),
            "model_s_mean": float(col("model_s").mean()),
            "model_time_fraction": float(col("model_s").sum() / max(t.sum(), 1e-12)),
            "requests_mean": float(col("model_calls").mean()),
            "logical_calls_mean": float(col("calls").mean()),
            "memo_hits_mean": float(col("memo_hits").mean()),
            "prompt_tokens_mean": float(col("prompt_tokens").mean()),
            "completion_tokens_mean": float(col("completion_tokens").mean()),
            "request_failures_total": int(col("request_failures").sum()),
            "stage_model_s_mean": {s: float(np.sum(v) / len(rows)) for s, v in sorted(stages.items())}}

def paired_time_contrast(a: Dict[str, Dict], b: Dict[str, Dict], cfg: Config) -> Dict[str, Any]:
    vids = sorted(set(a) & set(b))
    if len(vids) < 2:
        return {"n_videos": len(vids)}
    ta = np.array([a[v]["wall_s"] for v in vids], float)
    tb = np.array([b[v]["wall_s"] for v in vids], float)
    d = np.log(np.maximum(ta, 1e-9)) - np.log(np.maximum(tb, 1e-9))
    units = _game_units([a[v]["game"] for v in vids])
    G, B = len(units), cfg.n_bootstrap
    lo, hi = _t_interval(float(d.mean()), _cluster_boot(lambda ix: d[ix].mean(), units, B, cfg.seed), G)
    tr = float(ta.sum() / tb.sum())
    t_lo, t_hi = _t_interval(math.log(tr), _cluster_boot(
        lambda ix: math.log(ta[ix].sum() / tb[ix].sum()), units, B, cfg.seed + 1), G)
    nz = int(np.sum(np.abs(d) > 1e-12))
    wil = {"stat": float("nan"), "p": float("nan")}
    if nz >= 10:
        st, pw = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        wil = {"stat": float(st), "p": float(pw)}
    return {"n_videos": len(vids), "n_games": G,
            "geo_mean_ratio": float(math.exp(d.mean())),
            "geo_mean_ratio_ci": [math.exp(lo), math.exp(hi)] if math.isfinite(lo) else [float("nan")] * 2,
            "total_time_ratio": tr,
            "total_time_ratio_ci": [math.exp(t_lo), math.exp(t_hi)] if math.isfinite(t_lo)
            else [float("nan")] * 2,
            "fraction_videos_faster": float(np.mean(d < 0)),
            "permutation_game": cluster_sign_flip(d, units, cfg.n_permutations, cfg.seed),
            "wilcoxon_cluster_naive": {**wil, "n_nonzero": nz}}
