"""
HOVER v10 -- Hypothesise, Verify, Segment, Select: calibrated likelihood
verification for open-ended video-game glitch detection and temporal grounding
(VideoGlitchBench; baseline and evaluation parity with the official GliDe code,
https://github.com/SandyyyZheng/GliDe, arXiv:2604.07818).

Task (GliDe problem statement): for each gameplay video output every glitch as
a free-text description plus time span(s); scored by an LLM judge (0-5) with
Hungarian matching on the ratings, and by rating x temporal IoU.

Why v9 failed (probe on 72 dev videos / 12 games / 709 windows)
--------------------------------------------------------------
  * The generic question "does this window show a glitch?" measures the VIDEO,
    not the window: sd within videos 0.44 < sd between videos 0.60 logits;
    pooled AUC 0.557-0.569, per-video median AUC 0.583. Per-video AUC is
    identical with and without the content-free term, because that term is
    a per-video constant. This agrees with GlitchBench (Taesiri et al., CVPR 2024):
    LMMs are poor at OPEN glitch detection.
  * The self-reference assumption of v9 is false for this benchmark: the window
    base rate is 0.518, the median video is 50% glitch and 47% of videos are
    more than half glitch. Median normalisation (C2), 'normal-half' references
    (C3) and the HMM prior all assume that glitches are rare within a video.
  * C3 (pairwise) spent 67% of all calls for an AUC CI that contains 0.5;
    the HMM LOWERED the AUC (0.483 vs 0.529) with gamma pinned at the grid
    minimum and a log-loss of ln 2, i.e. uninformative emissions.
  * With ~50% coverage the trivial span "whole video" already reaches IoU ~0.5;
    v9 never reported that baseline.

v10 method: turn open detection into calibrated verification of hypotheses
---------------------------------------------------------------------------
  H  Hypothesise: every window is described with an explicit "no glitch" option;
     observations are merged into entity-anchored hypotheses and ranked by
     SUPPORT (how many windows produced them independently; self-consistency,
     Wang et al., ICLR 2023). If nothing is reported, one forced hypothesis from
     the top generic window is kept and the selector decides.
  V  Verify: for hypothesis h and every window w
        s_h(w) = logit P(Yes | w, "is <h> visible?") - logit P(Yes | grey, same)
     A specific, entity-anchored question is a verification task, which VLMs
     do far better than open anomaly detection. The subtracted term is h's own
     prompt prior (contextual calibration per query, Zhao et al., ICML 2021),
     so profiles of different hypotheses share one scale.
  S  Segment: exact O(T) dynamic programme for
        argmax over non-empty sets of disjoint segments of
        sum_k sum_{t in seg_k} (s_h(t) - tau)  -  lambda (K - 1),
     i.e. penalised-likelihood change-point segmentation (Kadane when
     lambda -> inf). tau, lambda are fitted on correct hypotheses by IoU with
     game-grouped cross-validation. Boundary windows are refined to frames by
     masked-frame likelihood bisection (from v9). The whole-video span competes
     as a refine mode, so the calibration output SAYS when localisation does
     not beat the trivial baseline.
  U  Select: a fractional-response logistic model (Papke & Wooldridge, 1996)
     predicts u_h = E[rating/5 x IoU] from profile, support and prior
     features. Labels come from the official protocol on dev: Hungarian on judge
     ratings, then IoU. Because the official F1xIoU is 2 S / (5 (N + M)),
     adding a prediction helps iff u_h > F*/2. This generalises the result of
     Lipton, Elkan & Naryanaswamy (2014) that the F1-optimal threshold is
     half the optimal F1. The threshold is the exact maximiser on out-of-fold
     predictions. The selector is declared degenerate unless it ranks correct
     hypotheses above chance (game-cluster CI of the out-of-fold AUC > 0.5).

  Statistics (kept from v9): exact / Monte-Carlo game sign-flip permutation test
  with Holm correction; game-cluster bootstrap CIs; official evaluator port.
  New diagnostics: ICC(1) variance decomposition and within-video (median-
  centred) AUC with game-cluster CIs, i.e. the numbers of the v9 probe, computed
  reproducibly for every evidence type; paired per-video AUC test of the
  hypothesis profile against the generic detector.

Honest scope: nothing is trained on pixels. The self-test verifies mechanics
(exactness of the segmentation DP, the F*/2 property, ICC recovery, test
validity) with a mock that reproduces the v9 probe pattern; whether a real
VLM verifies hypotheses better than it detects glitches is exactly what
'probe' measures, and must be established before any full run.

Commands
--------
  python HOVER_v10.py selftest [--glide_repo /path/to/GliDe]
  python HOVER_v10.py diagnose  --split dev --limit 72     # generic detector only (cached)
  python HOVER_v10.py probe     --split dev --limit 72     # v10 go/no-go (no judge needed)
  python HOVER_v10.py calibrate --calib_splits dev         # needs the judge server
  python HOVER_v10.py run       --method HOVER    --split test
  python HOVER_v10.py run       --method vanilla --split test
  python HOVER_v10.py import_glide --batch_report /path/batch_report.json --split test
  python HOVER_v10.py export    --run_file <runs/..../HOVER_test.json>
  python HOVER_v10.py evaluate  --split test --run_files HOVER_test.json glide_official_test.json
  python HOVER_v10.py ablate    --split test

--limit always samples round-robin over games (--no-stratify_limit disables it).

Fair-comparison recipe (same backbone, same server, same judge, same videos)
  1. vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-logprobs 20
  2. official GliDe, unmodified:
       cd GliDe && python run.py --video-dir <root>/raw --api-base http://localhost:8000/v1 \\
                                 --model Qwen/Qwen2.5-VL-7B-Instruct
  3. python HOVER_v10.py import_glide --split test --batch_report GliDe/data/results/batch_report.json
  4. python HOVER_v10.py probe --split dev --limit 72     (stop if the verdict is NO-GO)
  5. vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
  6. python HOVER_v10.py calibrate --calib_splits dev
  7. python HOVER_v10.py run --method HOVER --split test ; run --method vanilla --split test
  8. python HOVER_v10.py evaluate --split test --gt_json <official groundtruth.json> \\
       --run_files HOVER_test.json glide_official_test.json vanilla_test.json
     (first file = reference; evaluation uses the intersection of video sets)

The work directory defaults to HOVER_v8_work so that the windows and the generic
detector calls cached by v8/v9 are reused (identical prompts); v10 writes its
calibrations, runs and evaluations under a separate 'v10' sub-tree.

UAL-Bench extension (dataset-isolated; every command above is unchanged)
------------------------------------------------------------------------
UAL-Bench (Abdullah et al., WACV 2025; https://github.com/Hasnat79/UAL_Bench):
UAG-OOPS, UAG-SSBD and UAG-FunQA, one query and one GT interval per item,
scored by the official R@1 IoU / R@1 TD / mTD evaluator (ported exactly).
Mode A, the primary and only implemented UAL mode, is a query-conditioned
evaluation of the HOVER LOCALISATION CORE: the official query is the
hypothesis, V and S run unchanged with a fixed rule (tau = 0, lambda = inf,
i.e. one interval; frame refinement), H and U are not used. It is not an
evaluation of the full open-ended H -> V -> S -> U pipeline.
Only Qwen2.5-VL-7B-Instruct, InternVL2.5-8B and UI-TARS-1.5-7B are allowed.

  python HOVER_v10.py selftest_ual [--ual_repo /path/UAL_Bench]
  python HOVER_v10.py inspect_ual  --ual_root /path/UAL_Bench --ual_dataset all
  python HOVER_v10.py run_ual      --ual_root /path/UAL_Bench --ual_dataset uag_oops \\
                                  --backbone qwen2.5-vl-7b --limit 1          # smoke test
  python HOVER_v10.py run_ual      --ual_root /path/UAL_Bench --ual_dataset all --backbone all \\
      --ual_api_bases qwen2.5-vl-7b=http://localhost:8000/v1,internvl2.5-8b=http://localhost:8002/v1,ui-tars-1.5-7b=http://localhost:8003/v1
  python HOVER_v10.py evaluate_ual --ual_root /path/UAL_Bench --ual_dataset all --backbone all
  python HOVER_v10.py evaluate_ual --ual_predictions predictions_official.json [--ual_repo ...]

Outputs go to <parent of --ual_root>/HOVER_v10_ual_work/ual_bench/ (or
--ual_work_root), never into the VideoGlitchBench root, HOVER_v8_work, the GliDe
repository or the UAL-Bench data. See the UAL-Bench section near the end.

Portions (evaluation prompt, summariser prompt, preprocessing geometry,
evaluator logic) follow the GliDe repository, MIT licence, (c) its authors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp
from scipy.stats import wilcoxon

_CV2_ERR: Optional[Exception] = None
try:
    import cv2
except Exception as _e:  # pragma: no cover - environment dependent
    cv2 = None
    _CV2_ERR = _e

LOG = logging.getLogger("HOVER")
EPS = 1e-9
VERSION = "10.0"

# Backbones for the comparison matrix. With backend=openai the same names are
# what the vLLM server must be serving (e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`).
BACKBONES: Dict[str, Dict[str, str]] = {
    "qwen2.5-vl-3b": {"hf": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen"},
    "qwen2.5-vl-7b": {"hf": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen"},
    "internvl2.5-4b": {"hf": "OpenGVLab/InternVL2_5-4B", "family": "internvl"},
    "internvl2.5-8b": {"hf": "OpenGVLab/InternVL2_5-8B", "family": "internvl"},
    "ui-tars-1.5-7b": {"hf": "ByteDance-Seed/UI-TARS-1.5-7B", "family": "qwen"},
}
# Hypothesis-level features used by the utility model U (section "U" below).
HYP_FEATURES = ("support", "rank_inv", "forced", "prior", "prof_max", "prof_top2",
                "prof_mean", "prof_contrast", "seg_frac", "gen_lo", "gen_z")
REFINE_MODES = ("window", "frame", "whole")
STOPWORDS = {"the", "a", "an", "is", "are", "of", "in", "on", "and", "or", "to", "with",
             "that", "this", "it", "its", "as", "by", "for", "be", "can", "may", "game",
             "player", "character", "when", "which", "from", "at", "while", "into"}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # ---- paths ---------------------------------------------------------------
    root: str = "/home/tahir/VLM-Project/VideoGlitchBench"
    raw_subdir: str = "raw"
    splits_subdir: str = "splits"
    work_subdir: str = "HOVER_v8_work"   # shared with v8/v9: windows and VLM caches are reused
    metadata_name: str = "annotations.json"
    gt_json: str = ""            # official groundtruth.json; used verbatim by evaluate
    glide_repo: str = ""         # optional clone of the official GliDe repo

    # ---- preprocessing: official GliDe defaults (config.py) ------------------
    fps: float = 4.0
    window_size: int = 8
    jpeg_quality: int = 95       # official writes frames and stitches at q=95
    pair_max_side: int = 2688    # downscaled copies (used by the vanilla baseline)

    # ---- detector backbone -----------------------------------------------------
    backbone: str = "qwen2.5-vl-7b"
    backend: str = "openai"      # 'openai' (vLLM, same stack as GliDe) | 'hf' | 'mock'
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    family: str = "qwen"
    hf_dtype: str = "bfloat16"
    hf_device_map: str = "auto"
    load_in_4bit: bool = False
    max_pixels: int = 0          # 0 = processor/server default (match your GliDe server)
    request_timeout: int = 300
    max_retries: int = 3

    # Generation is ALWAYS fully specified; nothing is inherited from the
    # checkpoint's generation_config (the v7 top_k=1 bug).
    temperature: float = 0.0     # descriptions/summaries: greedy => reproducible
    top_p: float = 1.0
    top_k: int = -1              # -1 / 0 disables top-k
    repetition_penalty: float = 1.0
    desc_max_tokens: int = 256
    summary_max_tokens: int = 512
    top_logprobs: int = 20

    # ---- evaluation judge: official protocol ------------------------------------
    judge_backend: str = "openai"    # 'openai' | 'hf' | 'mock'
    judge_api_base: str = "http://localhost:8001/v1"
    judge_api_key: str = "EMPTY"
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_temperature: float = 0.3   # Evaluator default in the official repo
    judge_max_tokens: int = 512

    # ---- HOVER v10 components (each is an ablation switch) -------------------------
    use_contextual_calibration: bool = True   # per-query content-free prior removal
    use_hypothesis_scoring: bool = True       # V: hypothesis-conditioned profile (else generic)
    use_event_resolution: bool = True         # merge window observations into hypotheses
    use_segmentation: bool = True             # S: exact penalised segmentation (else member runs)
    use_subframe: bool = True                 # frame-level masked likelihood bisection
    use_verified_bisection: bool = True
    use_utility_selection: bool = True        # U: F1xIoU-optimal hypothesis selection

    describe_budget: int = 0      # windows described per video (0 = all; else top-k by lo_abs)
    max_hypotheses: int = 3       # hypotheses verified per video (ranked by support)
    max_intervals_per_event: int = 4
    max_events_per_video: int = 6
    seg_taus: str = "-1.0,-0.5,0.0,0.5,1.0,1.5,2.0"   # grid for the presence level tau
    seg_penalties: str = "0.0,0.5,1.0,2.0,4.0,1e9"    # grid for the segment-start cost lambda
    seg_tau: float = 0.0          # fitted by 'calibrate'
    seg_penalty: float = 1.0      # fitted by 'calibrate'
    refine_mode: str = "frame"    # 'window' | 'frame' | 'whole'; fitted by 'calibrate'
    min_correct_rating: int = 3   # judge rating that counts a hypothesis as correct

    # ---- selection / calibration ---------------------------------------------------
    calib_label: str = "judge"    # 'judge' (official judge on dev) | 'iou' (description-blind)
    label_min_overlap: float = 0.25
    fusion_l2: float = 1.0
    cv_folds: int = 5
    decision_threshold: float = 0.0   # utility threshold, fitted by 'calibrate'
    fusion: Dict[str, Any] = field(default_factory=dict)
    min_auc_ci_lo: float = 0.44
    fail_on_degenerate: bool = True
    quantize_seconds: bool = False   # emulate GliDe's int(frame // fps) output
    stratify_limit: bool = True      # --limit samples round-robin over games

    # ---- statistics / bookkeeping -------------------------------------------------
    seed: int = 20260924
    n_bootstrap: int = 2000
    n_permutations: int = 10000
    limit: Optional[int] = None

    def paths(self) -> Dict[str, str]:
        w = os.path.join(self.root, self.work_subdir)
        b = self.backbone
        return {"raw": os.path.join(self.root, self.raw_subdir),
                "splits": os.path.join(self.root, self.splits_subdir),
                "meta": os.path.join(self.root, self.metadata_name),
                "work": w,
                "windows": os.path.join(w, "windows"),
                "cache": os.path.join(w, "cache", b),
                "judge_cache": os.path.join(w, "cache", "judge",
                                            re.sub(r"[^A-Za-z0-9_.-]", "_", self.judge_model)),
                "runs": os.path.join(w, "v10", "runs", b),
                "params": os.path.join(w, "v10", "params", b),
                "eval": os.path.join(w, "v10", "eval", b)}

    def with_backbone(self, key: str) -> "Config":
        require(key in BACKBONES, f"unknown backbone {key}; have {sorted(BACKBONES)}")
        spec = BACKBONES[key]
        return Config(**{**asdict(self), "backbone": key, "model": spec["hf"],
                         "family": spec["family"]})

    def prep_fingerprint(self) -> str:
        blob = json.dumps({"fps": self.fps, "window_size": self.window_size,
                           "q": self.jpeg_quality, "pair": self.pair_max_side,
                           "v": "glide-official-geometry-1"}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# =============================================================================
# Utilities
# =============================================================================

def setup_logging(work_dir: str, tag: str) -> None:
    os.makedirs(work_dir, exist_ok=True)
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(os.path.join(work_dir, f"HOVER_v10_{tag}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically. Retries the rename on a transient
    FileNotFoundError: under heavy write load (sharded cache) the parent
    directory entry can be flushed asynchronously, and os.replace then fails
    even though os.makedirs just succeeded."""
    parent = os.path.dirname(path) or "."
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            os.makedirs(parent, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(_jsonable(obj), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return
        except FileNotFoundError as e:
            last = e
            time.sleep(0.1 * (attempt + 1))
        except OSError as e:
            last = e
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"atomic_write_json failed after 3 attempts: {path} ({last})")


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _repair_json_str(s: str) -> str:
    """Port of GliDe LLMClient._repair_json_str (identical heuristics)."""
    s = re.sub(r"(?<=[{,\s])\s*'([^']+?)'\s*:", r' "\1":', s)
    s = re.sub(r":\s*'([^']*?)'\s*([,}\]])", r': "\1"\2', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def _try_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_json_str(s))
    except json.JSONDecodeError:
        return None


def parse_json_from_text(text: str) -> Optional[Dict]:
    """Port of GliDe LLMClient._parse_json_from_text: fences first, then the
    first balanced {...}, then the whole string. Used for every JSON reply so
    that parse failures behave identically for GliDe-style and HOVER calls."""
    if not text:
        return None
    if "```json" in text:
        st = text.find("```json") + 7
        en = text.find("```", st)
        if en != -1:
            r = _try_loads(text[st:en].strip())
            if isinstance(r, dict):
                return r
    if "```" in text:
        st = text.find("```") + 3
        en = text.find("```", st)
        if en != -1:
            r = _try_loads(text[st:en].strip())
            if isinstance(r, dict):
                return r
    m = re.search(r"\{", text)
    if m:
        depth = 0
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    r = _try_loads(text[m.start():i + 1])
                    if isinstance(r, dict):
                        return r
                    break
    r = _try_loads(text.strip())
    return r if isinstance(r, dict) else None


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]+", str(text).lower())
            if t not in STOPWORDS and len(t) > 2}


# =============================================================================
# Interval algebra (used by grounding; evaluation uses the official IoU port)
# =============================================================================

Interval = Tuple[float, float]


def normalize_intervals(ivs: Sequence[Sequence[float]]) -> List[Interval]:
    clean = []
    for iv in ivs or []:
        if iv is None or len(iv) < 2:
            continue
        a, b = float(iv[0]), float(iv[1])
        if b < a:
            a, b = b, a
        if b - a <= 0:
            continue
        clean.append((a, b))
    if not clean:
        return []
    clean.sort()
    out = [clean[0]]
    for a, b in clean[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def intersect_length(A: Sequence[Interval], B: Sequence[Interval]) -> float:
    i = j = 0
    acc = 0.0
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if hi > lo:
            acc += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return acc


def parse_time_nodes(raw: Any) -> List[List[Interval]]:
    """VideoGlitchBench time_nodes -> one list of intervals per bug."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list) or not raw:
        return []
    if len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
        return [[(float(raw[0]), float(raw[1]))]]
    out: List[List[Interval]] = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) == 2 and all(
                isinstance(x, (int, float)) for x in entry):
            out.append(normalize_intervals([entry]))
        elif isinstance(entry, list):
            out.append(normalize_intervals(entry))
    return [o for o in out if o]


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class Record:
    vid: str
    video_name: str
    game: str
    genre: str
    video_path: str
    no_bugs: bool
    bugs: List[str]
    time_nodes: List[List[Interval]]
    duration: float = 0.0

    def gt_reports(self) -> List[Dict]:
        return [{"description": d, "spans": self.time_nodes[i]}
                for i, d in enumerate(self.bugs)
                if i < len(self.time_nodes) and self.time_nodes[i]]


def load_records(cfg: Config) -> List[Record]:
    """Accepts both VideoGlitchBench schemas: (A) the official GliDe format
    {video_name, game_name, bugs, time_nodes, no_bugs} and (B) a report list
    [{description, time_nodes|time_nodes_validated}]."""
    p = cfg.paths()
    require(os.path.exists(p["meta"]), f"metadata not found at {p['meta']}")
    rows = read_json(p["meta"])
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("records", []))
    recs: List[Record] = []
    missing = 0
    for r in rows:
        name = (r.get("video_name") or r.get("file") or r.get("video") or r.get("path") or "")
        name = os.path.basename(name) if name else f"{r.get('id') or r.get('video_id')}.mp4"
        if not name.endswith(".mp4"):
            name = name + ".mp4"
        game = str(r.get("game_name") or r.get("game") or "unknown")
        cands = [os.path.join(p["raw"], name), os.path.join(p["raw"], game, name)]
        ap = r.get("path") or r.get("video_path")
        if isinstance(ap, str) and ap.startswith("/"):
            cands.append(ap)
        vp = next((c for c in cands if os.path.exists(c)), None)
        if vp is None:
            import glob as _glob
            hits = _glob.glob(os.path.join(p["raw"], "*", name))
            vp = hits[0] if hits else None
        if vp is None:
            missing += 1
            continue
        bugs: List[str] = []
        tns: List[List[Interval]] = []
        raw_bugs = r.get("bugs")
        if isinstance(raw_bugs, str):
            raw_bugs = _try_loads(raw_bugs) or [raw_bugs]
        if isinstance(raw_bugs, list) and raw_bugs:
            bugs = [str(b) for b in raw_bugs]
            tns = parse_time_nodes(r.get("time_nodes"))
        else:
            for rep in (r.get("reports") or []):
                if not isinstance(rep, dict):
                    continue
                desc = str(rep.get("description", "")).strip()
                ivs = parse_time_nodes(rep.get("time_nodes_validated")
                                       or rep.get("time_nodes")
                                       or rep.get("time_nodes_original") or [])
                if desc and ivs:
                    bugs.append(desc)
                    tns.append(ivs[0])
        explicit = r.get("no_bugs")
        recs.append(Record(
            vid=str(r.get("id") or r.get("video_id") or name.replace(".mp4", "")),
            video_name=name.replace(".mp4", ""), game=game,
            genre=str(r.get("genre", "unknown")), video_path=vp,
            no_bugs=bool(explicit) if explicit is not None else (len(bugs) == 0),
            bugs=bugs, time_nodes=tns))
    require(recs, "no usable records; check raw/ and metadata paths")
    if missing:
        LOG.warning("%d annotated videos have no file on disk and were dropped", missing)
    LOG.info("loaded %d records over %d games", len(recs), len({r.game for r in recs}))
    return recs


def load_splits(cfg: Config, recs: List[Record]) -> Dict[str, List[Record]]:
    p = cfg.paths()
    require(os.path.isdir(p["splits"]), f"splits dir missing: {p['splits']}")
    index = {r.vid: r for r in recs}
    by_name = {r.video_name: r for r in recs}
    out: Dict[str, List[Record]] = {}
    for name in ("train", "dev", "test"):
        hits = sorted(f for f in os.listdir(p["splits"]) if f.startswith(name))
        require(hits, f"no split file for '{name}' in {p['splits']}")
        with open(os.path.join(p["splits"], hits[0])) as f:
            raw = f.read().strip()
        ids = _try_loads(raw)
        if isinstance(ids, dict):
            ids = ids.get("ids", [])
        if not isinstance(ids, list):
            ids = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        sel = []
        for i in ids:
            # Split entries are VideoGlitchBench video_id strings of the form
            # "Game::clip.mp4" and Record.vid is the same string, so match on
            # it verbatim first. Fall back to the extensionless form and then
            # to basename matching only if the direct lookup fails.
            key = str(i)
            rr = index.get(key) or index.get(key.replace(".mp4", ""))
            if rr is None:
                base = os.path.basename(key).replace(".mp4", "")
                rr = by_name.get(base)
            if rr is not None:
                sel.append(rr)
        require(sel, f"split '{name}' resolved to 0 records -- id format mismatch")
        out[name] = sel
        LOG.info("split %-5s: %4d videos, %3d games", name, len(sel), len({r.game for r in sel}))
    tr, te = {r.game for r in out["train"]}, {r.game for r in out["test"]}
    LOG.info("train/test game overlap: %d of %d test games", len(tr & te), len(te))
    return out


def take_limit(recs: Sequence[Record], limit: Optional[int], stratify: bool = True
               ) -> List[Record]:
    """First `limit` records, drawn round-robin over games when stratify is set.
    Split files are usually grouped by game, so a plain prefix of 60 videos can
    contain one or two games and makes game-grouped cross-fitting impossible."""
    recs = list(recs)
    if not limit or limit >= len(recs):
        return recs
    if not stratify:
        return recs[:limit]
    by: Dict[str, List[Record]] = defaultdict(list)
    for r in recs:
        by[r.game].append(r)
    games, out, k = sorted(by), [], 0
    while len(out) < limit:
        for g in games:
            if k < len(by[g]) and len(out) < limit:
                out.append(by[g][k])
        k += 1
    return out


def window_labels(rec: Record, wins: Sequence["Window"], min_overlap: float) -> np.ndarray:
    gt = normalize_intervals([iv for spans in rec.time_nodes for iv in spans])
    y = np.zeros(len(wins), np.int64)
    for i, w in enumerate(wins):
        if w.end > w.start:
            y[i] = int(intersect_length([(w.start, w.end)], gt) / (w.end - w.start)
                       >= min_overlap)
    return y


# =============================================================================
# Preprocessing -- geometry of the official GliDe VideoPreprocessor
#   * time-based sampling: t = 0, 1/fps, 2/fps, ... while t < duration, seek+read
#   * non-overlapping windows of `window_size` frames (last window may be short)
#   * stitch: native resolution, 2 rows x ceil(n/2) cols, GLOBAL frame index
#     label "#<id>" on a black box, JPEG quality 95
# =============================================================================

@dataclass
class Window:
    idx: int
    start: float
    end: float
    stitch_path: str
    frame_ids: List[int]
    frame_times: List[float]
    pair_path: str = ""


def extract_frames_official(video_path: str, fps: float, quality: int = 95
                            ) -> Tuple[List[np.ndarray], List[float], float]:
    require(cv2 is not None, f"opencv is required ({_CV2_ERR})")
    cap = cv2.VideoCapture(video_path)
    require(cap.isOpened(), f"cannot open video {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / native if native > 0 else 0.0
    interval = 1.0 / fps
    targets, t = [], 0.0
    while t < duration:                 # float accumulation, exactly as official
        targets.append(t)
        t += interval
    frames, times = [], []
    for tt in targets:
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        # Official GliDe writes each frame as JPEG (q=95) and stitches from the
        # decoded files; the round trip reproduces those pixels exactly.
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames.append(cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else fr)
        times.append(tt)
    cap.release()
    require(frames, f"decoded 0 frames from {video_path}")
    return frames, times, float(duration)


def stitch_official(frames: Sequence[np.ndarray], ids: Sequence[int]) -> np.ndarray:
    n = len(frames)
    require(n > 0, "no frames to stitch")
    fh, fw = frames[0].shape[:2]
    cols, rows = (n + 1) // 2, 2
    canvas = np.zeros((fh * rows, fw * cols, 3), dtype=np.uint8)
    for k, (fr, fid) in enumerate(zip(frames, ids)):
        r, c = k // cols, k % cols
        y0, x0 = r * fh, c * fw
        if fr.shape[:2] != (fh, fw):
            fr = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + fh, x0:x0 + fw] = fr
        _draw_label(canvas, x0, y0, fid)
    return canvas


def _draw_label(canvas: np.ndarray, x0: int, y0: int, fid: int) -> None:
    font, fs, th, pad = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, 5
    label = f"#{fid}"
    (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
    cv2.rectangle(canvas, (x0, y0), (x0 + tw + pad * 2, y0 + tht + pad * 2), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x0 + pad, y0 + tht + pad), font, fs, (255, 255, 255), th)


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def prepare_record(rec: Record, cfg: Config, force: bool = False) -> List[Window]:
    out_dir = os.path.join(cfg.paths()["windows"], cfg.prep_fingerprint(), rec.vid)
    manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest) and not force:
        data = read_json(manifest)
        rec.duration = float(data["duration"])
        return [Window(**w) for w in data["windows"]]
    frames, times, duration = extract_frames_official(rec.video_path, cfg.fps,
                                                      cfg.jpeg_quality)
    rec.duration = duration
    os.makedirs(out_dir, exist_ok=True)
    wins: List[Window] = []
    for wi, s in enumerate(range(0, len(frames), cfg.window_size)):
        e = min(s + cfg.window_size, len(frames))
        ids = list(range(s, e))
        st = stitch_official(frames[s:e], ids)
        sp = os.path.join(out_dir, f"window_{wi:04d}_stitched.jpg")
        require(cv2.imwrite(sp, st, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]),
                f"failed to write {sp}")
        pp = os.path.join(out_dir, f"window_{wi:04d}_pair.jpg")
        cv2.imwrite(pp, _downscale(st, cfg.pair_max_side), [cv2.IMWRITE_JPEG_QUALITY, 92])
        t0 = times[s]
        t1 = min(duration, times[e - 1] + 1.0 / cfg.fps)
        wins.append(Window(wi, float(t0), float(max(t1, t0 + EPS)), sp, ids,
                           [float(x) for x in times[s:e]], pp))
    require(wins, f"created 0 windows for {rec.vid}")
    # Content-free probe: a uniform grey image with the stitched geometry.
    h, w = cv2.imread(wins[0].stitch_path).shape[:2]
    cf = os.path.join(out_dir, "content_free.jpg")
    cv2.imwrite(cf, np.full((h, w, 3), 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    atomic_write_json(manifest, {"duration": duration, "fps": cfg.fps,
                                 "windows": [asdict(x) for x in wins]})
    return wins


def content_free_path(wins: Sequence[Window]) -> str:
    return os.path.join(os.path.dirname(wins[0].stitch_path), "content_free.jpg")


# =============================================================================
# Derived images. All are built lazily from the stitched window and inherit
# its exact geometry, so the preprocessing fingerprint (and v8 caches) is kept.
#   static(w):  every cell = per-pixel temporal median of the window's frames
#   keep(w,a,b): cells outside frame positions [a, b] are grey (labels kept)
# =============================================================================

def _split_cells(img: np.ndarray, n: int) -> Tuple[List[np.ndarray], int, int, int]:
    cols = (n + 1) // 2
    fh, fw = img.shape[0] // 2, img.shape[1] // cols
    cells = [img[(k // cols) * fh:(k // cols + 1) * fh, (k % cols) * fw:(k % cols + 1) * fw]
             for k in range(n)]
    return cells, fh, fw, cols


def _derived_path(win: Window, suffix: str) -> str:
    base = win.stitch_path
    stem = base[:-len("_stitched.jpg")] if base.endswith("_stitched.jpg") else base.rsplit(".", 1)[0]
    return f"{stem}_{suffix}.jpg"


def masked_window_path(win: Window, a: int, b: int, quality: int = 95) -> str:
    """Keep frame POSITIONS a..b (inclusive) of the window, grey out the rest.
    The grid geometry and the frame labels are unchanged, so the model still
    sees where in time the kept frames sit."""
    n = len(win.frame_ids)
    require(0 <= a <= b < n, f"bad keep range {a}..{b} for {n} frames")
    out = _derived_path(win, f"keep_{win.frame_ids[a]}_{win.frame_ids[b]}")
    if os.path.exists(out):
        return out
    img = cv2.imread(win.stitch_path)
    require(img is not None, f"cannot read {win.stitch_path}")
    _, fh, fw, cols = _split_cells(img, n)
    canvas = img.copy()
    for k, fid in enumerate(win.frame_ids):
        if a <= k <= b:
            continue
        y0, x0 = (k // cols) * fh, (k % cols) * fw
        canvas[y0:y0 + fh, x0:x0 + fw] = 128
        _draw_label(canvas, x0, y0, fid)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out


_PREP_MEMO: Dict[Tuple[str, str], List[Window]] = {}


def prepare_split(recs: Sequence[Record], cfg: Config) -> Dict[str, List[Window]]:
    fp, out, t0 = cfg.prep_fingerprint(), {}, time.time()
    for i, r in enumerate(recs):
        key = (fp, r.vid)
        if key not in _PREP_MEMO:
            _PREP_MEMO[key] = prepare_record(r, cfg)
        out[r.vid] = _PREP_MEMO[key]
        if r.duration <= 0 and out[r.vid]:
            r.duration = out[r.vid][-1].end
        if (i + 1) % 100 == 0:
            LOG.info("prepared %d/%d videos (%.0f s)", i + 1, len(recs), time.time() - t0)
    return out


# =============================================================================
# VLM backends. Two primitives only:
#   generate(system, user, images) -> text          (fully specified sampling)
#   score(system, user, images, options) -> {option: normalised log-prob}
# =============================================================================

@dataclass
class Budget:
    calls: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprob_truncations: int = 0
    by_stage: Dict[str, int] = field(default_factory=dict)


def _file_sig(path: str) -> str:
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        return sha1(f"{ap}:{st.st_size}:{int(st.st_mtime)}")
    except OSError:
        return sha1(ap)


def build_generation_kwargs(cfg: Config, max_tokens: int) -> Dict[str, Any]:
    """Every sampling knob is explicit. v7 omitted top_k and silently inherited
    top_k=1 from the Qwen2.5-VL generation_config, turning every 'sample' into
    the greedy decode."""
    do_sample = cfg.temperature > 0
    kw: Dict[str, Any] = {"max_new_tokens": int(max_tokens), "do_sample": do_sample,
                          "repetition_penalty": float(cfg.repetition_penalty)}
    if do_sample:
        kw.update({"temperature": float(cfg.temperature), "top_p": float(cfg.top_p),
                   "top_k": int(cfg.top_k) if cfg.top_k and cfg.top_k > 0 else 0})
    else:
        kw.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0})
    return kw


def normalise_option_logprobs(raw: Dict[str, float]) -> Dict[str, float]:
    keys = list(raw)
    z = float(logsumexp([raw[k] for k in keys]))
    return {k: float(raw[k] - z) for k in keys}


def binary_log_odds(lp: Dict[str, float], pos: str, neg: str) -> float:
    return float(lp[pos] - lp[neg])


class VLMBase:
    name = "base"
    needs_pixels = True

    def __init__(self, cfg: Config, cache_dir: str):
        self.cfg = cfg
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.budget = Budget()

    # ---- caching ---------------------------------------------------------------
    def _key(self, kind: str, system: str, user: str, images: Sequence[str],
             extra: Dict) -> str:
        return sha1(json.dumps({"k": kind, "b": self.name, "m": self.cfg.model,
                                "s": system, "u": user, "x": extra,
                                "px": self.cfg.max_pixels,
                                "i": [_file_sig(p) for p in images]}, sort_keys=True))

    def _cached(self, key: str) -> Optional[Any]:
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        return read_json(path)["v"] if os.path.exists(path) else None

    def _store(self, key: str, value: Any) -> None:
        atomic_write_json(os.path.join(self.cache_dir, key[:2], key + ".json"), {"v": value})

    def _count(self, stage: str, cached: bool) -> None:
        self.budget.calls += 1
        self.budget.by_stage[stage] = self.budget.by_stage.get(stage, 0) + 1
        if cached:
            self.budget.cache_hits += 1
        else:
            self.budget.model_calls += 1

    # ---- public ---------------------------------------------------------------
    def generate(self, system: str, user: str, images: Sequence[str] = (),
                 max_tokens: int = 512, stage: str = "other") -> str:
        extra = build_generation_kwargs(self.cfg, max_tokens)
        key = self._key("gen", system, user, images, extra)
        hit = self._cached(key)
        if hit is not None:
            self._count(stage, True)
            return hit
        out = self._generate(system, user, images, max_tokens)
        self._count(stage, False)
        self._store(key, out)
        return out

    def generate_json(self, system: str, user: str, images: Sequence[str] = (),
                      max_tokens: int = 512, stage: str = "other",
                      retries: int = 1) -> Optional[Dict]:
        for a in range(retries + 1):
            hint = "" if a == 0 else "\nReturn ONLY a single valid JSON object."
            obj = parse_json_from_text(self.generate(system, user + hint, images,
                                                     max_tokens, stage))
            if obj is not None:
                return obj
        return None

    def score(self, system: str, user: str, images: Sequence[str],
              options: Dict[str, Sequence[str]], stage: str = "other") -> Dict[str, float]:
        extra = {"opts": {k: list(v) for k, v in sorted(options.items())}}
        key = self._key("score", system, user, images, extra)
        hit = self._cached(key)
        if hit is not None:
            self._count(stage, True)
            return hit
        raw = self._score(system, user, images, options)
        out = normalise_option_logprobs(raw)
        self._count(stage, False)
        self._store(key, out)
        return out

    def _generate(self, system, user, images, max_tokens) -> str:
        raise NotImplementedError

    def _score(self, system, user, images, options) -> Dict[str, float]:
        raise NotImplementedError


def _variants(word: str) -> List[str]:
    return list(dict.fromkeys([word, word.lower(), word.upper(), word.capitalize()]))


def match_top_logprobs(top: Sequence[Dict], options: Dict[str, Sequence[str]]
                       ) -> Tuple[Dict[str, float], bool]:
    """Collapse an OpenAI-style top_logprobs list onto answer options. An option
    absent from the list gets the smallest listed log-prob, which is an UPPER
    bound on its true value; the second return flags that truncation."""
    lps = [float(t["logprob"]) for t in top]
    floor = min(lps) if lps else -30.0
    out, truncated = {}, False
    for opt, words in options.items():
        wanted = {w.strip().lower() for word in words for w in _variants(word)}
        got = [float(t["logprob"]) for t in top
               if str(t.get("token", "")).strip().lower() in wanted]
        if got:
            out[opt] = float(logsumexp(got))
        else:
            out[opt] = floor
            truncated = True
    return out, truncated


class OpenAIVLM(VLMBase):
    """OpenAI-compatible client (vLLM). Same serving stack as official GliDe,
    so detector capacity and image handling are identical by construction."""
    name = "openai"

    def _post(self, payload: Dict) -> Dict:
        import requests
        hdr = {"Content-Type": "application/json"}
        if self.cfg.api_key and self.cfg.api_key != "EMPTY":
            hdr["Authorization"] = f"Bearer {self.cfg.api_key}"
        if self.cfg.max_pixels > 0:
            payload["mm_processor_kwargs"] = {"max_pixels": int(self.cfg.max_pixels)}
        last = None
        for a in range(self.cfg.max_retries):
            try:
                r = requests.post(f"{self.cfg.api_base.rstrip('/')}/chat/completions",
                                  headers=hdr, json=payload, timeout=self.cfg.request_timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # pragma: no cover - network
                last = e
                # DEBUG: dump the failing payload AND the response body
                try:
                    import json as _json
                    _dump = _json.dumps(payload)[:2000]
                    LOG.error("VLM request failed (attempt %d/%d): %s", a + 1,
                              self.cfg.max_retries, e)
                    LOG.error("VLM payload (first 2000 chars): %s", _dump)
                    # try to read the response body if the exception carries one
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            LOG.error("VLM response status: %s", resp.status_code)
                            LOG.error("VLM response body: %s", resp.text[:2000])
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(2 ** a)
        raise RuntimeError(f"VLM request failed after {self.cfg.max_retries} attempts: {last}")

    @staticmethod
    def _content(user: str, images: Sequence[str]) -> Any:
        if not images:
            return user
        parts: List[Dict] = [{"type": "text", "text": user}]
        for p in images:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return parts

    def _generate(self, system, user, images, max_tokens) -> str:
        g = build_generation_kwargs(self.cfg, max_tokens)
        payload = {"model": self.cfg.model, "max_tokens": int(max_tokens),
                   "temperature": float(self.cfg.temperature),
                   "top_p": g["top_p"], "top_k": g["top_k"] if g["top_k"] > 0 else -1,
                   "repetition_penalty": g["repetition_penalty"],
                   # InternVL's template requires the content-list form on every
                   # message; a string system message triggers
                   # "can only concatenate str (not 'list') to str".
                   "messages": [{"role": "system", "content": [{"type": "text", "text": system}]},
                                {"role": "user", "content": self._content(user, images)}]}
        d = self._post(payload)
        u = d.get("usage") or {}
        self.budget.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
        self.budget.completion_tokens += int(u.get("completion_tokens", 0) or 0)
        return str(d["choices"][0]["message"]["content"] or "").strip()

    def _score(self, system, user, images, options) -> Dict[str, float]:
        # Both messages must use the modern content-list form. InternVL's
        # template rejects a string system message when any user message
        # contains a list, raising "can only concatenate str (not 'list') to str".
        sys_content = [{"type": "text", "text": system}]
        user_content = self._content(user, images)
        if not isinstance(user_content, list):
            user_content = [{"type": "text", "text": user_content}]
        payload = {"model": self.cfg.model, "max_tokens": 1, "temperature": 0.0,
                   "logprobs": True, "top_logprobs": int(self.cfg.top_logprobs),
                   "messages": [{"role": "system", "content": sys_content},
                                {"role": "user", "content": user_content}]}
        d = self._post(payload)
        u = d.get("usage") or {}
        self.budget.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
        content = ((d["choices"][0].get("logprobs") or {}).get("content") or [])
        require(content, "server returned no logprobs; start vLLM with logprobs enabled "
                         "(--max-logprobs >= top_logprobs)")
        out, trunc = match_top_logprobs(content[0].get("top_logprobs") or [], options)
        self.budget.logprob_truncations += int(trunc)
        return out


class HFVLM(VLMBase):
    """Local transformers backend (Qwen2.5-VL family). InternVL's remote-code
    .chat() hides the logits, so InternVL is supported through vLLM only."""
    name = "hf"

    def __init__(self, cfg: Config, cache_dir: str):
        super().__init__(cfg, cache_dir)
        require(cfg.family == "qwen", "the hf backend supports the qwen family only; "
                                      "serve InternVL with vLLM and use --backend openai")
        self._model = None
        self._proc = None
        self._opt_ids: Dict[Tuple[str, ...], List[int]] = {}

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLM
        except ImportError:  # pragma: no cover
            from transformers import Qwen2VLForConditionalGeneration as VLM
        kw: Dict[str, Any] = {"device_map": self.cfg.hf_device_map}
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = getattr(torch, self.cfg.hf_dtype)
        self._model = VLM.from_pretrained(self.cfg.model, **kw).eval()
        pk = {"max_pixels": int(self.cfg.max_pixels)} if self.cfg.max_pixels > 0 else {}
        self._proc = AutoProcessor.from_pretrained(self.cfg.model, **pk)
        LOG.info("loaded %s (hf); checkpoint generation_config is IGNORED: %s",
                 self.cfg.model, self._model.generation_config.to_diff_dict())

    def _inputs(self, system, user, images):
        from PIL import Image
        pil = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "text", "text": user}] + [{"type": "image"} for _ in pil]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._proc(text=[text], images=pil or None, return_tensors="pt",
                          padding=True).to(self._model.device)

    def _generate(self, system, user, images, max_tokens) -> str:
        import torch
        from transformers import GenerationConfig
        self._lazy()
        inp = self._inputs(system, user, images)
        gc = GenerationConfig(**build_generation_kwargs(self.cfg, max_tokens),
                              eos_token_id=self._model.generation_config.eos_token_id,
                              pad_token_id=self._model.generation_config.pad_token_id)
        with torch.no_grad():
            out = self._model.generate(**inp, generation_config=gc)
        gen = out[0][inp["input_ids"].shape[1]:]
        self.budget.prompt_tokens += int(inp["input_ids"].shape[1])
        self.budget.completion_tokens += int(gen.shape[0])
        return self._proc.decode(gen, skip_special_tokens=True).strip()

    def _first_ids(self, words: Sequence[str]) -> List[int]:
        key = tuple(words)
        if key not in self._opt_ids:
            tok = self._proc.tokenizer
            ids = set()
            for w in words:
                for v in _variants(w):
                    for s in (v, " " + v):
                        t = tok.encode(s, add_special_tokens=False)
                        if t:
                            ids.add(int(t[0]))
            self._opt_ids[key] = sorted(ids)
        return self._opt_ids[key]

    def _score(self, system, user, images, options) -> Dict[str, float]:
        import torch
        self._lazy()
        inp = self._inputs(system, user, images)
        with torch.no_grad():
            # Only the last position is needed; materialising logits for ~10^4
            # visual tokens x 152k vocab would cost several GB.
            out = None
            for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
                try:
                    out = self._model(**inp, **kw)
                    break
                except TypeError:
                    continue
            logits = out.logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        self.budget.prompt_tokens += int(inp["input_ids"].shape[1])
        return {o: float(logsumexp(lp[self._first_ids(w)])) for o, w in options.items()}


class MockVLM(VLMBase):
    """Scripted backend for offline tests. oracle(kind, system, user, images, options)
    returns text for kind='gen' and a raw {option: logprob} dict for kind='score'."""
    name = "mock"
    needs_pixels = False

    def __init__(self, cfg: Config, cache_dir: str, oracle: Callable):
        super().__init__(cfg, cache_dir)
        self.oracle = oracle

    def _cached(self, key):      # never cache in tests
        return None

    def _store(self, key, value):
        return None

    def _generate(self, system, user, images, max_tokens) -> str:
        return str(self.oracle("gen", system, user, list(images), None))

    def _score(self, system, user, images, options) -> Dict[str, float]:
        return dict(self.oracle("score", system, user, list(images), options))


def make_vlm(cfg: Config, oracle: Optional[Callable] = None) -> VLMBase:
    cache = cfg.paths()["cache"]
    if cfg.backend == "openai":
        return OpenAIVLM(cfg, cache)
    if cfg.backend == "hf":
        return HFVLM(cfg, cache)
    if cfg.backend == "mock":
        require(oracle is not None, "mock backend needs an oracle")
        return MockVLM(cfg, cache, oracle)
    raise ValueError(f"unknown backend {cfg.backend}")


# =============================================================================
# Prompts
# =============================================================================

_GLITCH_DEF = (
    "A glitch is a visible defect the developers did not intend: objects or characters "
    "clipping through geometry, floating without support, being launched or flung, "
    "teleporting, stretching or deforming, T-posing or frozen/broken animation while "
    "moving, missing or corrupted textures, flickering, lighting or shadow errors, AI "
    "stuck in place, physically impossible motion. Stylised art, intended abilities, "
    "menus, HUD, cut-scenes and ordinary camera motion are NOT glitches.")

P_DETECT_SYS = (
    "You are a video game QA expert. The image is a stitched sequence of consecutive "
    "gameplay frames, each labelled '#<frame index>' in temporal order (left to right, "
    "top row then bottom row). " + _GLITCH_DEF)
P_PRESENCE_SYS = (
    "You are a video game QA expert. You are given one stitched window of consecutive "
    "gameplay frames and the description of one specific glitch. Decide whether that "
    "same glitch, on that same entity, is visibly occurring in this window. Being "
    "plausible is not enough; it must be observable.")
P_DESC_SYS = (
    "You are a video game QA analyst writing one line of a bug report. The image is a "
    "stitched sequence of consecutive gameplay frames labelled '#<frame index>'. "
    + _GLITCH_DEF + " Name the affected object or character by its visual appearance, "
    "state exactly what is abnormal, and where in the scene it happens, in 1-3 sentences. "
    "No frame numbers or timestamps. Reply with one JSON object only, keys: description, "
    "entity, glitch_type.")
P_DESC_OPEN_SYS = (
    "You are a video game QA analyst. The image is a stitched sequence of consecutive "
    "gameplay frames labelled '#<frame index>'. " + _GLITCH_DEF + " If no glitch is visible, "
    "say so; do not invent one. If a glitch is visible, name the affected object or "
    "character by its visual appearance and state exactly what is abnormal, in 1-2 "
    "sentences, without frame numbers. Reply with one JSON object only, keys: glitch "
    "(true or false), description, entity, glitch_type.")
P_EAR_SYS = (
    "You consolidate per-window glitch observations from one gameplay video into distinct "
    "glitch EVENTS. Two observations belong to the same event if they concern the same "
    "entity (allowing for wording variance) and the same kind of anomaly, even if far "
    "apart in time. Different entities, or different anomaly kinds on the same entity, "
    "are different events. Reply with one JSON object only, key 'events': list of "
    "{member_ids (list of int), canonical_entity, glitch_type, description}.")

# Verbatim from GliDe summarizer/system_prompt.txt (MIT). {placeholders} filled by format().
P_SUMMARY_TEMPLATE = (
    "You are a video game glitch analyst. Your task is to summarize multiple fragmented "
    "glitch descriptions into a single, clear, and coherent description.\n\n"
    "## Input Descriptions:\n{descriptions}\n\n## Glitch Category: {category}\n"
    "## Glitch Subtype: {subtype}\n## Time Range: {time_range}\n\n## Instructions:\n"
    "1. Read all the fragmented descriptions carefully\n"
    "2. Identify the core glitch phenomenon being described\n"
    "3. Write a single, coherent paragraph that:\n"
    "   - Clearly describes what the glitch is\n"
    "   - Mentions the visual/behavioral anomaly observed\n"
    "   - Is concise but complete (2-4 sentences)\n"
    "   - Does NOT include frame numbers or technical details\n"
    "   - Does NOT include JSON formatting or code blocks\n"
    "   - Uses natural, descriptive language\n\n"
    "## Output:\nWrite ONLY the summarized description paragraph, nothing else.")

P_VANILLA_SYS = (
    "You are a video game QA expert. Watch the gameplay windows and report every glitch: "
    "for each, a natural-language description and its start/end time in seconds. Reply "
    "with one JSON object only, keys: no_bugs (bool), bugs (list of {description, "
    "time_nodes: [[start,end], ...]}).")

YES_NO = {"yes": ["Yes"], "no": ["No"]}
A_B = {"A": ["A"], "B": ["B"]}


def detect_user(game: str) -> str:
    return (f"Game: {game}.\nDoes this window show a gameplay glitch? "
            f"Answer with exactly one word: Yes or No.")


def presence_user(game: str, desc: str, entity: str) -> str:
    return (f"Game: {game}.\nGlitch: {desc}\nAffected entity: {entity}\n"
            f"Is this same glitch visibly occurring in this window? "
            f"Answer with exactly one word: Yes or No.")


# =============================================================================
# E: generic evidence (C1, C2). Weak on VideoGlitchBench (see module docstring),
#    kept only because it is cheap, cached from v8/v9, and useful as a feature.
# =============================================================================

def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD). 1.4826 makes MAD a consistent estimator of
    sigma under normality; the floor keeps a flat video from exploding."""
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return (x - med) / max(mad, 0.25)


@dataclass
class GenericEvidence:
    lo_raw: List[float]
    lo_cf: float
    lo_abs: List[float]
    rel_z: List[float]


def collect_generic_evidence(vlm: VLMBase, rec: Record, wins: Sequence[Window],
                             cfg: Config) -> GenericEvidence:
    usr = detect_user(rec.game)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_DETECT_SYS, usr, [path], YES_NO, stage), "yes", "no")

    lo_raw = np.array([lo_of(w.stitch_path, "c1_detect") for w in wins])
    lo_cf = 0.0
    if cfg.use_contextual_calibration:
        lo_cf = lo_of(content_free_path(wins) if vlm.needs_pixels else "__content_free__",
                      "c1_calib")
    lo_abs = lo_raw - lo_cf
    return GenericEvidence(lo_raw.tolist(), float(lo_cf), lo_abs.tolist(),
                           robust_z(lo_abs).tolist())


# =============================================================================
# H + V: hypotheses and hypothesis-conditioned likelihood profiles
#   H  every window is described with an explicit "no glitch" option; the
#      observations are merged into hypotheses (entity + anomaly), ranked by
#      support (how many windows independently produced them).
#   V  for hypothesis h and window w:
#        s_h(w) = logit P(Yes | w, "is <h> visible?") - logit P(Yes | grey, same question)
#      The second term is h's own prompt prior (contextual calibration per query),
#      so profiles of different hypotheses are on a common scale.
# =============================================================================

@dataclass
class Hypothesis:
    description: str
    entity: str
    glitch_type: str
    members: List[int]
    forced: bool
    prior: float
    profile: List[float]


@dataclass
class VideoHypotheses:
    vid: str
    generic: GenericEvidence
    hyps: List[Hypothesis]
    n_described: int


def verify_profile(vlm: VLMBase, rec: Record, wins: Sequence[Window], desc: str, entity: str,
                   cfg: Config) -> Tuple[float, List[float]]:
    usr = presence_user(rec.game, desc, entity)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_PRESENCE_SYS, usr, [path], YES_NO, stage), "yes", "no")

    prior = 0.0
    if cfg.use_contextual_calibration:
        prior = lo_of(content_free_path(wins) if vlm.needs_pixels else "__content_free__",
                      "v_calib")
    return prior, [lo_of(w.stitch_path, "v_profile") - prior for w in wins]


def collect_video(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config
                  ) -> VideoHypotheses:
    gen = collect_generic_evidence(vlm, rec, wins, cfg)
    lo = np.asarray(gen.lo_abs)
    order = list(range(len(wins)))
    if cfg.describe_budget and cfg.describe_budget < len(wins):
        order = sorted(int(i) for i in np.argsort(-lo, kind="mergesort")[:cfg.describe_budget])
    obs = []
    for i in order:
        d = describe_window_open(vlm, wins[i], rec.game)
        if d is not None:
            obs.append({"window": i, **d})
    forced = not obs
    if forced:
        # The model reports no glitch anywhere. Keep one forced hypothesis from
        # the window the generic detector ranks highest; U decides whether it is
        # worth emitting (the 'forced' feature lets it learn how often it is).
        i = int(np.argmax(lo)) if lo.size else 0
        obs = [{"window": i, **describe_window(vlm, wins[i], rec.game)}]
    events = resolve_events(vlm, obs, cfg)
    rz = np.asarray(gen.rel_z)
    for e in events:
        e["members"] = sorted({obs[k]["window"] for k in e["member_ids"]})
        e["support"] = len(e["members"])
        e["peak_z"] = float(max(rz[m] for m in e["members"])) if rz.size else 0.0
    events.sort(key=lambda e: (-e["support"], -e["peak_z"], e["members"][0]))
    hyps = []
    for e in events[:cfg.max_hypotheses]:
        descs = [obs[k]["description"] for k in e["member_ids"]]
        desc = summarise_event(vlm, descs, e["glitch_type"], windows_to_spans(e["members"], wins))
        if cfg.use_hypothesis_scoring:
            prior, prof = verify_profile(vlm, rec, wins, desc, e["canonical_entity"], cfg)
        else:
            prior, prof = 0.0, list(gen.lo_abs)
        hyps.append(Hypothesis(desc, e["canonical_entity"], e["glitch_type"], e["members"],
                               forced, float(prior), [float(x) for x in prof]))
    return VideoHypotheses(rec.vid, gen, hyps, len(order))


# =============================================================================
# S: exact penalised segmentation of a likelihood profile
#   argmax over NON-EMPTY sets of disjoint segments of
#        sum_k sum_{t in seg_k} (s_t - tau)  -  lambda * (K - 1)
#   tau is the presence level (log-odds units), lambda the cost of every segment
#   after the first. Exact O(T) dynamic programme with three states: before the
#   first segment, inside a segment, between segments. lambda -> inf gives the
#   single maximum-sum segment (Kadane); lambda = 0 gives every run with s > tau.
# =============================================================================

def segment_profile(s: Sequence[float], tau: float, lam: float) -> List[Tuple[int, int]]:
    x = np.asarray(s, float) - float(tau)
    T = x.size
    if T == 0:
        return []
    NEG = -np.inf
    V = np.full((T, 3), NEG)          # 0 before, 1 inside, 2 between
    bp = np.zeros((T, 3), int)
    V[0, 0], V[0, 1] = 0.0, x[0]
    for t in range(1, T):
        V[t, 0] = 0.0
        cand = (V[t - 1, 1], V[t - 1, 0], V[t - 1, 2] - lam)    # continue, first, new
        k = int(np.argmax(cand))
        V[t, 1], bp[t, 1] = cand[k] + x[t], (1, 0, 2)[k]
        if V[t - 1, 2] >= V[t - 1, 1]:
            V[t, 2], bp[t, 2] = V[t - 1, 2], 2
        else:
            V[t, 2], bp[t, 2] = V[t - 1, 1], 1
    st = 1 if V[T - 1, 1] >= V[T - 1, 2] else 2
    inside = np.zeros(T, bool)
    for t in range(T - 1, -1, -1):
        inside[t] = st == 1
        if t > 0:
            st = bp[t, st] if st != 0 else 0
    segs, a = [], None
    for t in range(T):
        if inside[t] and a is None:
            a = t
        if a is not None and (not inside[t] or t == T - 1):
            segs.append((a, t if inside[t] else t - 1))
            a = None
    return segs


def segment_score(s: Sequence[float], segs: Sequence[Tuple[int, int]], tau: float,
                  lam: float) -> float:
    x = np.asarray(s, float) - tau
    return float(sum(x[a:b + 1].sum() for a, b in segs) - lam * max(0, len(segs) - 1))


def member_runs(members: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for m in sorted(set(members)):
        if out and m == out[-1][1] + 1:
            out[-1] = (out[-1][0], m)
        else:
            out.append((m, m))
    return out


def hypothesis_segments(h: Hypothesis, cfg: Config, tau: float, lam: float
                        ) -> List[Tuple[int, int]]:
    if not cfg.use_segmentation:
        return member_runs(h.members)
    segs = segment_profile(h.profile, tau, lam)
    if len(segs) > cfg.max_intervals_per_event:
        x = np.asarray(h.profile) - tau
        segs = sorted(sorted(segs, key=lambda ab: -x[ab[0]:ab[1] + 1].sum())
                      [:cfg.max_intervals_per_event])
    return segs


def hypothesis_features(h: Hypothesis, gen: GenericEvidence, rank: int, cfg: Config,
                        tau: float, lam: float) -> Dict[str, float]:
    s = np.asarray(h.profile, float)
    n = max(1, s.size)
    segs = hypothesis_segments(h, cfg, tau, lam)
    inside = np.zeros(s.size, bool)
    for a, b in segs:
        inside[a:b + 1] = True
    contrast = float(s[inside].mean() - s[~inside].mean()) if inside.any() and (~inside).any() else 0.0
    lo, rz = np.asarray(gen.lo_abs), np.asarray(gen.rel_z)
    top = np.sort(s)[::-1]
    return {"support": len(h.members) / n, "rank_inv": 1.0 / rank, "forced": float(h.forced),
            "prior": h.prior, "prof_max": float(top[0]) if s.size else 0.0,
            "prof_top2": float(top[:2].mean()) if s.size else 0.0,
            "prof_mean": float(s.mean()) if s.size else 0.0, "prof_contrast": contrast,
            "seg_frac": float(inside.mean()) if s.size else 0.0,
            "gen_lo": float(lo[h.members].mean()) if lo.size else 0.0,
            "gen_z": float(rz[h.members].max()) if rz.size else 0.0}


# =============================================================================
# Metrics
# =============================================================================

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    srt = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0:
        return float("nan")
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, y.size + 1)) * y).sum() / y.sum())


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def icc_oneway(values: Sequence[Sequence[float]]) -> Dict[str, float]:
    """One-way random-effects ANOVA decomposition of a window score into a
    between-video and a within-video component (unbalanced groups, ANOVA
    estimator with the n0 correction). ICC(1) near 1 means the score is a
    property of the VIDEO and cannot localise anything inside it."""
    groups = [np.asarray(v, float) for v in values if len(v) >= 2]
    if len(groups) < 2:
        return {"icc": float("nan"), "sd_within": float("nan"), "sd_between": float("nan"),
                "n_videos": len(groups)}
    k, N = len(groups), sum(g.size for g in groups)
    grand = float(np.concatenate(groups).mean())
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(g.size ** 2 for g in groups) / N) / (k - 1)
    vb = max(0.0, (msb - msw) / n0)
    return {"icc": float(vb / (vb + msw)) if vb + msw > 0 else float("nan"),
            "sd_within": float(math.sqrt(msw)), "sd_between": float(math.sqrt(vb)),
            "n_videos": k}


def within_video_scores(scores: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """Centre every video's scores on its median: a pooled AUC of the centred
    scores measures WITHIN-video discrimination only (the 'when' signal)."""
    return [np.asarray(v, float) - (float(np.median(v)) if len(v) else 0.0) for v in scores]


def per_video_auc(scores: Sequence[Sequence[float]], labels: Sequence[Sequence[int]]) -> np.ndarray:
    out = [roc_auc(s, y) for s, y in zip(scores, labels)]
    return np.asarray([a for a in out if not math.isnan(a)], float)


def cluster_bootstrap_auc(scores: Sequence[float], labels: Sequence[int],
                          groups: Sequence[str], B: int, seed: int) -> Dict[str, float]:
    """Resample GAMES with replacement. Windows of one video/game are not
    independent, so an i.i.d. bootstrap over windows would understate the
    variance and overstate significance."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        a = roc_auc(s[take], y[take])
        if not math.isnan(a):
            vals.append(a)
    point = roc_auc(s, y)
    G = int(uniq.size)
    if not vals or G < 2:
        return {"auc": point, "lo": float("nan"), "hi": float("nan"), "n_groups": G}
    # Percentile cluster-bootstrap intervals under-cover when the number of
    # clusters is small (null false-pass rate ~12% at G=8 in the selftest study).
    # The interval used for decisions is therefore a bootstrap-SE interval with a
    # Student-t(G-1) quantile and the G/(G-1) small-sample variance correction.
    from scipy.stats import t as _t
    se = float(np.std(vals, ddof=1)) * math.sqrt(G / (G - 1.0))
    q = float(_t.ppf(0.975, G - 1))
    return {"auc": point, "lo": point - q * se, "hi": point + q * se,
            "pct_lo": float(np.percentile(vals, 2.5)), "pct_hi": float(np.percentile(vals, 97.5)),
            "se": se, "n_groups": G, "p_le_half": float(np.mean(np.asarray(vals) <= 0.5))}


def select_threshold_mcc(p: Sequence[float], y: Sequence[int]) -> Tuple[float, Dict]:
    p, y = np.asarray(p, float), np.asarray(y, int)
    grid = np.unique(np.round(np.concatenate([np.quantile(p, np.linspace(0.01, 0.99, 99)),
                                              np.arange(0.05, 0.96, 0.05)]), 6))
    best_t, best, curve = 0.5, -2.0, []
    for t in grid:
        pr = (p >= t).astype(int)
        tp, fp = int(((pr == 1) & (y == 1)).sum()), int(((pr == 1) & (y == 0)).sum())
        fn, tn = int(((pr == 0) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
        m = mcc(tp, fp, fn, tn)
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        curve.append({"t": float(t), "mcc": m, "precision": prec, "recall": rec,
                      "pos_rate": float(pr.mean())})
        if m > best:
            best_t, best = float(t), m
    at = min(curve, key=lambda r: abs(r["t"] - best_t))
    return best_t, {"mcc": best, **{k: at[k] for k in ("precision", "recall", "pos_rate")},
                    "base_rate": float(y.mean()) if y.size else 0.0, "curve": curve}


# =============================================================================
# C5: fusion with game-grouped cross-fitting
# =============================================================================

def fit_logistic(X: np.ndarray, y: Sequence[int], l2: float = 1.0,
                 iters: int = 100) -> Tuple[np.ndarray, float]:
    """Newton-Raphson / IRLS for ridge logistic regression (intercept unpenalised)."""
    yv = np.asarray(y, float)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        pr = expit(Xb @ w)
        g = Xb.T @ (yv - pr) - reg @ w
        H = Xb.T @ (Xb * np.clip(pr * (1 - pr), 1e-6, None)[:, None]) + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return w[:-1], float(w[-1])


def fusion_fit(rows: Sequence[Dict[str, float]], y: Sequence[int], feats: Sequence[str],
               l2: float) -> Dict[str, Any]:
    X = np.array([[r[f] for f in feats] for r in rows], float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    w, b = fit_logistic((X - mu) / sd, y, l2)
    return {"features": list(feats), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": w.tolist(), "b": b, "l2": l2}


def fusion_predict(model: Dict[str, Any], rows: Sequence[Dict[str, float]]) -> np.ndarray:
    feats = model["features"]
    X = np.array([[r[f] for f in feats] for r in rows], float).reshape(len(rows), len(feats))
    Z = (X - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return expit(Z @ np.asarray(model["w"]) + float(model["b"]))


def game_folds(games: Sequence[str], k: int, seed: int) -> Dict[str, int]:
    uniq = sorted(set(games))
    random.Random(seed).shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    return {g: i % k for i, g in enumerate(uniq)}


# =============================================================================
# G: frame-level grounding with a likelihood presence oracle
# =============================================================================

def bisect_onset(present: Callable[[int], bool], lo: int, hi: int) -> int:
    """Smallest i in [lo, hi] with present(i), assuming present(hi) and monotone."""
    require(lo <= hi, "bisect_onset: empty range")
    while lo < hi:
        mid = (lo + hi) // 2
        if present(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bisect_offset(present: Callable[[int], bool], lo: int, hi: int) -> int:
    require(lo <= hi, "bisect_offset: empty range")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if present(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


class PresenceOracle:
    """Event-specific likelihood oracle: present iff the calibrated log-odds of
    'this same glitch is visibly occurring' exceeds the margin. Callable on a
    window index; .image(path) scores any derived image of the same geometry."""

    def __init__(self, vlm: VLMBase, wins: Sequence[Window], game: str, desc: str,
                 entity: str, cfg: Config, margin: float = 0.0):
        self.vlm, self.cfg, self.margin = vlm, cfg, float(margin)
        self.by_idx = {w.idx: w for w in wins}
        self.usr = presence_user(game, desc, entity)
        self.base = 0.0
        if cfg.use_contextual_calibration and wins:
            cf = content_free_path(wins) if vlm.needs_pixels else "__content_free__"
            self.base = binary_log_odds(vlm.score(P_PRESENCE_SYS, self.usr, [cf], YES_NO,
                                                  "g_calib"), "yes", "no")
        self.memo: Dict[Any, bool] = {}

    def image(self, path: str, stage: str = "g_frame") -> bool:
        if path not in self.memo:
            lo = binary_log_odds(self.vlm.score(P_PRESENCE_SYS, self.usr, [path], YES_NO, stage),
                                 "yes", "no")
            self.memo[path] = (lo - self.base) > self.margin
        return self.memo[path]

    def __call__(self, i: int) -> bool:
        w = self.by_idx.get(i)
        return False if w is None else self.image(w.stitch_path, "g_presence")


def frame_onset(oracle: PresenceOracle, win: Window, audit: Dict, verify: bool = True) -> int:
    """First glitch frame of the onset window: the smallest position k such that
    the glitch is visible when only frames 0..k are kept. 'The prefix 0..k
    contains a glitch frame' is monotone non-decreasing in k, so bisection needs
    ceil(log2 n) oracle calls; k = n-1 is the full window (known present). A
    verified step repairs a violated monotonicity."""
    n = len(win.frame_ids)
    keep = lambda k: True if k == n - 1 else oracle.image(masked_window_path(win, 0, k))
    k = bisect_onset(keep, 0, n - 1)
    if verify and k > 0 and keep(k - 1):
        audit["monotonicity_violations"] += 1
        while k - 1 >= 0 and keep(k - 1):
            k -= 1
    return k


def frame_offset(oracle: PresenceOracle, win: Window, audit: Dict, verify: bool = True) -> int:
    """Last glitch frame of the offset window: the largest position k such that
    the glitch is visible when only frames k..n-1 are kept (mirror of
    frame_onset; the suffix predicate is monotone non-increasing in k)."""
    n = len(win.frame_ids)
    keep = lambda k: True if k == 0 else oracle.image(masked_window_path(win, k, n - 1))
    k = bisect_offset(keep, 0, n - 1)
    if verify and k < n - 1 and keep(k + 1):
        audit["monotonicity_violations"] += 1
        while k + 1 <= n - 1 and keep(k + 1):
            k += 1
    return k


def frame_time_bounds(win: Window, k_on: int, k_off: int, fps: float) -> Interval:
    """Decoded timestamps, not id / fps (the float accumulation of the sampler
    makes those differ)."""
    t = win.frame_times
    a = clamp(t[k_on], win.start, win.end)
    b = clamp(t[k_off] + 1.0 / fps, a, win.end)
    return (a, b)


def windows_to_spans(members: Sequence[int], wins: Sequence[Window]) -> List[Interval]:
    by = {w.idx: w for w in wins}
    return normalize_intervals([(by[m].start, by[m].end) for m in members if m in by])


def hypothesis_spans(vlm: VLMBase, rec: Record, wins: Sequence[Window], h: Hypothesis,
                     cfg: Config, tau: float, lam: float, mode: str
                     ) -> Tuple[List[Interval], Dict[str, int]]:
    """Time spans of a hypothesis under a refine mode:
         window  segments of the exact segmentation, at window resolution
         frame   same segments, boundaries refined by masked-frame bisection with
                 the hypothesis' own presence oracle at level tau
         whole   the entire video (the trivial baseline; kept as a candidate so
                 that calibration can report honestly when localisation does not
                 beat it)"""
    audit = {"monotonicity_violations": 0, "frame_conflicts": 0}
    dur = float(rec.duration or (wins[-1].end if wins else 0.0))
    if mode == "whole" or not wins:
        return ([(0.0, dur)] if dur > 0 else []), audit
    segs = hypothesis_segments(h, cfg, tau, lam)
    refine = mode == "frame" and cfg.use_subframe and cfg.use_hypothesis_scoring
    oracle: Optional[PresenceOracle] = None
    out = []
    for a, b in segs:
        lo_t, hi_t = wins[a].start, wins[b].end
        if refine:
            oracle = oracle or PresenceOracle(vlm, wins, rec.game, h.description, h.entity,
                                              cfg, tau)
            v = cfg.use_verified_bisection
            on_t, off_t = lo_t, hi_t
            if h.profile[a] > tau:
                k = frame_onset(oracle, wins[a], audit, v)
                on_t = frame_time_bounds(wins[a], k, k, cfg.fps)[0]
            if h.profile[b] > tau:
                k = frame_offset(oracle, wins[b], audit, v)
                off_t = frame_time_bounds(wins[b], k, k, cfg.fps)[1]
            if off_t > on_t:
                lo_t, hi_t = on_t, off_t
            else:
                audit["frame_conflicts"] += 1
        out.append((clamp(lo_t, 0.0, dur), clamp(hi_t, 0.0, dur)))
    return normalize_intervals(out), audit


# =============================================================================
# Descriptions and entity-anchored event resolution
# =============================================================================

def describe_window(vlm: VLMBase, win: Window, game: str) -> Dict[str, str]:
    obj = vlm.generate_json(P_DESC_SYS, f"Game: {game}. Describe the glitch in this window.",
                            [win.stitch_path], 384, "describe") or {}
    d = str(obj.get("description", "")).strip()
    return {"description": d or "A visual or physics anomaly affecting an object in the scene.",
            "entity": str(obj.get("entity", "") or "entity"),
            "glitch_type": str(obj.get("glitch_type", "") or "other")}


def describe_window_open(vlm: VLMBase, win: Window, game: str) -> Optional[Dict[str, str]]:
    """Open description with an explicit no-glitch option. Returns None when the
    model reports no glitch or returns nothing usable."""
    obj = vlm.generate_json(P_DESC_OPEN_SYS, f"Game: {game}. Is there a glitch in this window?",
                            [win.stitch_path], 256, "describe") or {}
    g = obj.get("glitch")
    if isinstance(g, str):
        g = g.strip().lower() in ("true", "yes", "1")
    d = str(obj.get("description", "") or "").strip()
    if not g or not d:
        return None
    return {"description": d, "entity": str(obj.get("entity", "") or "entity"),
            "glitch_type": str(obj.get("glitch_type", "") or "other")}


def _fallback_events(obs: Sequence[Dict]) -> List[Dict]:
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, o in enumerate(obs):
        key = (" ".join(sorted(_tokens(o["entity"]))) or o["entity"], o["glitch_type"].lower())
        buckets[key].append(i)
    return [{"member_ids": ids, "canonical_entity": obs[ids[0]]["entity"],
             "glitch_type": k[1], "description": obs[ids[0]]["description"]}
            for k, ids in buckets.items()]


def resolve_events(vlm: VLMBase, obs: Sequence[Dict], cfg: Config) -> List[Dict]:
    if len(obs) <= 1 or not cfg.use_event_resolution:
        return [{"member_ids": [i], "canonical_entity": o["entity"],
                 "glitch_type": o["glitch_type"], "description": o["description"]}
                for i, o in enumerate(obs)]
    payload = [{"id": i, "window": o["window"], "entity": o["entity"],
                "glitch_type": o["glitch_type"], "observation": o["description"][:300]}
               for i, o in enumerate(obs)]
    obj = vlm.generate_json(P_EAR_SYS, f"Observations (chronological):\n"
                                       f"{json.dumps(payload)[:4000]}", (), 768,
                            "resolution") or {}
    events, seen = [], set()
    for e in obj.get("events") or []:
        if not isinstance(e, dict):
            continue
        ids = []
        for i in e.get("member_ids") or []:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(obs) and ii not in seen and ii not in ids:
                ids.append(ii)
        if not ids:
            continue
        seen.update(ids)
        events.append({"member_ids": sorted(ids),
                       "canonical_entity": str(e.get("canonical_entity") or obs[ids[0]]["entity"]),
                       "glitch_type": str(e.get("glitch_type") or obs[ids[0]]["glitch_type"]),
                       "description": str(e.get("description") or obs[ids[0]]["description"])})
    left = [i for i in range(len(obs)) if i not in seen]
    for grp in _fallback_events([obs[i] for i in left]):
        grp["member_ids"] = [left[i] for i in grp["member_ids"]]
        events.append(grp)
    return events


def summarise_event(vlm: VLMBase, descs: Sequence[str], gtype: str,
                    spans: Sequence[Interval]) -> str:
    if len(descs) == 1:
        return descs[0]
    prompt = P_SUMMARY_TEMPLATE.format(
        descriptions="\n".join(f"- {d}" for d in descs), category=gtype, subtype="Unknown",
        time_range=", ".join(f"{a:.1f}s - {b:.1f}s" for a, b in spans))
    txt = vlm.generate("You are a video game glitch analyst.", prompt, (), 512, "summary")
    txt = re.sub(r"```.*?```", "", txt, flags=re.S).strip()
    return txt or descs[0]


def to_time_nodes(spans: Sequence[Interval], cfg: Config) -> List[List[float]]:
    if cfg.quantize_seconds:
        # GliDe summariser: int(frame // fps) on both ends.
        return [[int(math.floor(a)), int(math.floor(b))] for a, b in spans]
    return [[round(float(a), 2), round(float(b), 2)] for a, b in spans]


# =============================================================================
# HOVER end-to-end
# =============================================================================

def hypothesis_rows(vh: VideoHypotheses, cfg: Config, tau: float, lam: float
                    ) -> List[Dict[str, float]]:
    return [hypothesis_features(h, vh.generic, k + 1, cfg, tau, lam) for k, h in enumerate(vh.hyps)]


def run_HOVER(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config,
             vh: Optional[VideoHypotheses] = None) -> Dict[str, Any]:
    require(not cfg.use_utility_selection or str(cfg.fusion.get("version", "")).startswith("10"),
            "run_HOVER needs a HOVER v10 utility model (run 'calibrate')")
    vh = vh or collect_video(vlm, rec, wins, cfg)
    tau, lam = cfg.seg_tau, cfg.seg_penalty
    rows = hypothesis_rows(vh, cfg, tau, lam)
    util = (fusion_predict(cfg.fusion, rows) if rows and cfg.use_utility_selection
            else np.ones(len(rows)))
    keep = [i for i in np.argsort(-util, kind="mergesort")
            if not cfg.use_utility_selection or util[i] >= cfg.decision_threshold]
    keep = [int(i) for i in keep][:cfg.max_events_per_video]
    out: Dict[str, Any] = {"vid": rec.vid, "video_name": rec.video_name, "game": rec.game,
                           "n_windows": len(wins), "n_described": vh.n_described,
                           "hypotheses": [{"description": h.description, "entity": h.entity,
                                           "members": h.members, "forced": h.forced,
                                           "prior": h.prior, "profile": h.profile,
                                           "utility": float(u), **r}
                                          for h, u, r in zip(vh.hyps, util, rows)],
                           "monotonicity_violations": 0, "frame_conflicts": 0}
    reps = []
    for i in keep:
        h = vh.hyps[i]
        spans, audit = hypothesis_spans(vlm, rec, wins, h, cfg, tau, lam, cfg.refine_mode)
        for k in audit:
            out[k] += audit[k]
        if spans:
            reps.append({"description": h.description, "spans": spans, "utility": float(util[i]),
                         "entity": h.entity})
    out.update({"no_bugs": not reps, "reports": reps, "bugs": [r["description"] for r in reps],
                "time_nodes": [to_time_nodes(r["spans"], cfg) for r in reps]})
    return out


# =============================================================================
# Baselines
# =============================================================================

def run_vanilla(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config,
                max_images: int = 6) -> Dict[str, Any]:
    """In-house single-pass baseline (not part of the official repo). Windows are
    sampled evenly across the video and their true time ranges are stated, so the
    model is never asked to extrapolate timestamps it was not shown."""
    n = len(wins)
    pick = sorted(set(np.linspace(0, n - 1, min(n, max_images)).round().astype(int).tolist()))
    imgs = [wins[i].pair_path or wins[i].stitch_path for i in pick]
    ranges = "; ".join(f"image {k + 1} = [{wins[i].start:.1f}s, {wins[i].end:.1f}s]"
                       for k, i in enumerate(pick))
    obj = vlm.generate_json(P_VANILLA_SYS,
                            f"Game: {rec.game}. Video duration: {rec.duration:.1f} s. "
                            f"Images in temporal order: {ranges}. Report all glitches.",
                            imgs, 1024, "vanilla") or {}
    bugs, tns = [], []
    for b in obj.get("bugs") or []:
        if not isinstance(b, dict):
            continue
        # Defensive parsing: InternVL occasionally returns time_nodes as
        # [5] or [[5]] or [["5"]] instead of [[start, end], ...]. Skip any
        # entry that is not a two-element numeric pair, so the stage does
        # not crash mid-run on a malformed response.
        pairs = []
        for entry in (b.get("time_nodes") or []):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            a, c = entry[0], entry[1]
            try:
                pairs.append((clamp(float(a), 0, rec.duration),
                              clamp(float(c), 0, rec.duration)))
            except (TypeError, ValueError):
                continue
        sp = normalize_intervals(pairs)
        d = str(b.get("description", "")).strip()
        if d and sp:
            bugs.append(d)
            tns.append(to_time_nodes(sp, cfg))
    return {"vid": rec.vid, "video_name": rec.video_name, "game": rec.game,
            "no_bugs": not bugs, "bugs": bugs, "time_nodes": tns}


def import_glide_report(path: str, recs: Sequence[Record]) -> Tuple[List[Dict], Dict]:
    """Ingest an official GliDe batch_report.json verbatim: bugs and time_nodes are
    passed to the evaluator exactly as the official code wrote them (integer
    seconds from int(frame // fps)); nothing is re-normalised."""
    data = read_json(path)
    require(isinstance(data, list), f"{path} is not a GliDe batch_report (expected a list)")
    by_name = {r.video_name: r for r in recs}
    outs, unknown = [], 0
    for item in data:
        vn = str(item.get("video_name", "")).replace(".mp4", "")
        r = by_name.get(vn)
        if r is None:
            unknown += 1
            continue
        outs.append({"vid": r.vid, "video_name": vn, "game": r.game,
                     "no_bugs": bool(item.get("no_bugs", True)),
                     "bugs": list(item.get("bugs", [])),
                     "time_nodes": list(item.get("time_nodes", []))})
    missing = sorted(set(by_name) - {o["video_name"] for o in outs})
    info = {"n_report": len(data), "n_matched": len(outs), "n_not_in_split": unknown,
            "n_split_missing_from_report": len(missing), "missing_examples": missing[:10]}
    if missing:
        LOG.warning("GliDe report lacks %d of %d split videos; evaluation will use the "
                    "INTERSECTION of video sets for every method, as fairness requires",
                    len(missing), len(by_name))
    return outs, info


def to_batch_report(outputs: Sequence[Dict]) -> List[Dict]:
    return [{"video_name": o["video_name"], "game_name": o.get("game", "Unknown"),
             "no_bugs": bool(o.get("no_bugs", not o.get("bugs"))),
             "bugs": list(o.get("bugs", [])), "time_nodes": list(o.get("time_nodes", []))}
            for o in outputs]


# =============================================================================
# Official GliDe evaluation protocol (evaluation/evaluator.py), ported exactly:
#   * text-only LLM judge, official prompt, system message, key 'rating' (0-5)
#   * Hungarian matching on the RATING matrix only
#   * IoU computed for each matched pair afterwards (official compute_iou)
#   * EVERY Hungarian pair is appended, including rating-0 pairs
#   * P = sum(r)/(5 N_pred), R = sum(r)/(5 N_gt); *_iou use sum(r * IoU)
#   * mean_iou = mean over all matched pairs; only videos present in both files
# =============================================================================

JUDGE_SYSTEM = ("You must respond with a JSON object containing 'rating' (int 0-5) and "
                "'reasoning' (string).")
# Verbatim from GliDe evaluation/prompt.txt (MIT). Loaded from --glide_repo when given.
JUDGE_PROMPT = (
    "You are an expert in video game glitch analysis. Your task is to evaluate the quality "
    "of a predicted glitch description based on the given ground truth glitch description, "
    "focusing on the accurate identification/detection of the glitch.\n\n"
    "**Ground Truth Description**:\n{gt_description}\n\n"
    "**Predicted Description**:\n{pred_description}\n\n"
    "**Instructions**:\n\n"
    "- **Rate the quality of the predicted description on a scale from 0 to 5**, using the "
    "following criteria:\n\n"
    "- **0: No quality**.\n"
    "  - The predicted description is completely unrelated to the ground truth.\n"
    "  - The descriptions refer to entirely different events or glitches.\n"
    "  - The predicted description is irrelevant or nonsensical in the context of the glitch.\n\n"
    "- **1: Very low quality**.\n"
    "  - The predicted description mentions some elements, but they are incorrect or irrelevant.\n"
    "  - Key details about the glitch are missing or inaccurately described.\n"
    "  - The descriptions have minimal overlap in content or context.\n\n"
    "- **2: Low quality**.\n"
    "  - The predicted description includes minor aspects related to the glitch but misses "
    "the main behavior.\n"
    "  - Important elements are absent or significantly misrepresented.\n"
    "  - The description has some relevant terms but lacks context or specificity.\n\n"
    "- **3: Moderate quality**.\n"
    "  - The predicted description identifies the general type of glitch but lacks "
    "significant details.\n"
    "  - Some key behaviors or elements are mentioned, but important specifics are missing.\n"
    "  - The description captures part of the glitch but omits certain actions or effects.\n\n"
    "- **4: High quality**.\n"
    "  - The predicted description captures the main glitch and includes most key details.\n"
    "  - Minor differences exist (e.g., general terms used instead of specific names), but "
    "the overall glitch is accurately described.\n"
    "  - The description reflects the primary behaviors and impacts of the glitch on the "
    "game experience.\n\n"
    "- **5: Nearly identical**.\n"
    "  - The predicted description accurately identifies the glitch with all key details "
    "matching the ground truth.\n"
    "  - There are no significant differences; both descriptions refer to the same event "
    "with similar specificity.\n"
    "  - The description includes all important behaviors, entities involved, and impacts "
    "on gameplay.\n\n"
    "- **Provide a brief reasoning before your answer**, focusing on:\n"
    "  - How well the predicted description matches the ground truth in identifying the glitch.\n"
    "  - Specific similarities or differences in the behaviors, entities involved, and "
    "details mentioned.\n"
    "  - You should first identify what is the main abnormal behavior of the glitch; as long "
    "as the main behavior is predicted to be the same as the ground truth, you should give a "
    "high rating to the prediction.\n"
    "  - The impact of any missing or additional information on the accuracy of the glitch "
    "identification.\n\n"
    "- **Output Format**:\n\n"
    "Please provide your analysis and your rating in the following JSON format:\n\n"
    "```json\n{{\n  \"reasoning\": \"<brief reasoning of the evaluation>\",\n"
    "  \"rating\": <number from 0 to 5>\n}}\n```\n")


def load_judge_prompt(cfg: Config) -> str:
    if cfg.glide_repo:
        p = os.path.join(cfg.glide_repo, "evaluation", "prompt.txt")
        require(os.path.exists(p), f"--glide_repo given but {p} is missing")
        with open(p) as f:
            txt = f.read()
        if txt != JUDGE_PROMPT:
            LOG.warning("official prompt.txt differs from the embedded copy (sha1 %s vs %s); "
                        "using the repository file", sha1(txt)[:10], sha1(JUDGE_PROMPT)[:10])
        return txt
    return JUDGE_PROMPT


def official_iou(gt: List[List[float]], pred: List[List[float]]) -> float:
    """Exact port of Evaluator.compute_iou (including its conventions: both empty
    -> 1.0; one empty -> 0.0; pairwise intersections merged)."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0

    def merge(iv):
        if not iv:
            return []
        s = sorted(iv, key=lambda x: x[0])
        m = [list(s[0])]
        for cur in s[1:]:
            last = m[-1]
            if cur[0] <= last[1]:
                m[-1] = [last[0], max(last[1], cur[1])]
            else:
                m.append(list(cur))
        return m

    def dur(iv):
        return sum(e - s for s, e in merge(iv))

    inter = []
    for gs, ge in gt:
        for ps, pe in pred:
            a, b = max(gs, ps), min(ge, pe)
            if a < b:
                inter.append([a, b])
    g, p, i = dur(gt), dur(pred), dur(inter)
    u = g + p - i
    if u == 0:
        return 1.0 if i == 0 else 0.0
    return max(0.0, min(1.0, i / u))


class JudgeBase:
    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt = load_judge_prompt(cfg)
        self.cache_dir = cfg.paths()["judge_cache"]
        self.calls = 0
        self.parse_failures = 0

    def rate(self, gt_desc: str, pred_desc: str) -> int:
        key = sha1(json.dumps([self.name, self.cfg.judge_model, self.cfg.judge_temperature,
                               sha1(self.prompt), gt_desc, pred_desc]))
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path) and self.name != "mock":
            return int(read_json(path)["rating"])
        content = self._chat(JUDGE_SYSTEM, self.prompt.format(gt_description=gt_desc,
                                                              pred_description=pred_desc))
        self.calls += 1
        rating = self._parse(content)
        if self.name != "mock":
            atomic_write_json(path, {"rating": rating, "raw": content})
        return rating

    def _parse(self, content: str) -> int:
        """Official order: json.loads first, then the text parser; failure -> 0."""
        for loader in (lambda s: json.loads(s), parse_json_from_text):
            try:
                d = loader(content)
            except (json.JSONDecodeError, TypeError):
                d = None
            if isinstance(d, dict):
                try:
                    return int(d.get("rating", 0))
                except (TypeError, ValueError):
                    break
        self.parse_failures += 1
        return 0

    def _chat(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAIJudge(JudgeBase):
    """Payload shape identical to GliDe LLMClient._chat_openai (model,
    temperature, max_tokens, messages); no extra sampling fields are sent."""
    name = "openai"

    def _chat(self, system, user) -> str:
        import requests
        hdr = {"Content-Type": "application/json"}
        if self.cfg.judge_api_key and self.cfg.judge_api_key != "EMPTY":
            hdr["Authorization"] = f"Bearer {self.cfg.judge_api_key}"
        payload = {"model": self.cfg.judge_model, "temperature": self.cfg.judge_temperature,
                   "max_tokens": self.cfg.judge_max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        last = None
        for a in range(3):
            try:
                r = requests.post(f"{self.cfg.judge_api_base.rstrip('/')}/chat/completions",
                                  headers=hdr, json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # pragma: no cover - network
                last = e
                time.sleep(2 ** a)
        LOG.warning("judge request failed (%s); official behaviour is rating 0", last)
        return ""


class HFJudge(JudgeBase):
    name = "hf"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._m = self._t = None

    def _chat(self, system, user) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if self._m is None:
            self._t = AutoTokenizer.from_pretrained(self.cfg.judge_model)
            self._m = AutoModelForCausalLM.from_pretrained(
                self.cfg.judge_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        ids = self._t.apply_chat_template([{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
                                          add_generation_prompt=True, return_tensors="pt"
                                          ).to(self._m.device)
        gc = GenerationConfig(max_new_tokens=self.cfg.judge_max_tokens, do_sample=True,
                              temperature=self.cfg.judge_temperature, top_p=1.0, top_k=0,
                              pad_token_id=self._t.eos_token_id)
        with torch.no_grad():
            out = self._m.generate(ids, generation_config=gc)
        return self._t.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


class MockJudge(JudgeBase):
    name = "mock"

    def __init__(self, cfg: Config, fn: Callable[[str, str], int]):
        super().__init__(cfg)
        self.fn = fn

    def _chat(self, system, user) -> str:
        gt = user.split("**Ground Truth Description**:\n", 1)[1].split("\n\n**Predicted", 1)[0]
        pr = user.split("**Predicted Description**:\n", 1)[1].split("\n\n**Instructions", 1)[0]
        return json.dumps({"reasoning": "mock", "rating": int(self.fn(gt, pr))})


def make_judge(cfg: Config, fn: Optional[Callable] = None) -> JudgeBase:
    if cfg.judge_backend == "openai":
        return OpenAIJudge(cfg)
    if cfg.judge_backend == "hf":
        return HFJudge(cfg)
    require(fn is not None, "mock judge needs a scoring function")
    return MockJudge(cfg, fn)


def official_eval_video(judge: JudgeBase, gt: Dict, pred: Dict) -> Dict[str, Any]:
    gb, pb = list(gt.get("bugs", [])), list(pred.get("bugs", []))
    gtn, ptn = list(gt.get("time_nodes", [])), list(pred.get("time_nodes", []))
    res = {"gt_num": len(gb), "pred_num": len(pb), "matched_scores": [], "matched_ious": [],
           "matches": []}
    if not gb or not pb:
        return res
    S = np.zeros((len(pb), len(gb)))
    for i, pd in enumerate(pb):
        for j, gd in enumerate(gb):
            S[i, j] = judge.rate(gd, pd)
    pi, gi = linear_sum_assignment(-S)
    for a, b in zip(pi, gi):
        s = int(S[a][b])
        iou = official_iou(gtn[b] if b < len(gtn) else [], ptn[a] if a < len(ptn) else [])
        res["matched_scores"].append(s)
        res["matched_ious"].append(iou)
        res["matches"].append({"pred_idx": int(a), "gt_idx": int(b), "score": s, "iou": iou})
    res["score_matrix"] = S.tolist()
    return res


def official_aggregate(vr: Sequence[Dict]) -> Dict[str, float]:
    N = sum(v["pred_num"] for v in vr)
    M = sum(v["gt_num"] for v in vr)
    sc = [s for v in vr for s in v["matched_scores"]]
    io = [i for v in vr for i in v["matched_ious"]]
    sxi = [s * i for v in vr for s, i in zip(v["matched_scores"], v["matched_ious"])]
    P = sum(sc) / (5 * N) if N else 0.0
    R = sum(sc) / (5 * M) if M else 0.0
    Pi = sum(sxi) / (5 * N) if N else 0.0
    Ri = sum(sxi) / (5 * M) if M else 0.0
    hm = lambda a, b: 2 * a * b / (a + b) if a + b > 0 else 0.0
    return {"num_videos": len(vr), "num_gt_bugs": M, "num_pred_bugs": N,
            "num_matched": len(sc), "mean_score": float(np.mean(sc)) if sc else 0.0,
            "mean_iou": float(np.mean(io)) if io else 0.0,
            "precision": P, "recall": R, "f1": hm(P, R),
            "precision_iou": Pi, "recall_iou": Ri, "f1_iou": hm(Pi, Ri)}


def load_gt_official(cfg: Config, recs: Sequence[Record]) -> Dict[str, Dict]:
    """Ground truth as the official evaluator sees it. With --gt_json the official
    groundtruth.json is used verbatim; otherwise the loaded annotations are
    converted (and a warning says so, because v7 preferred time_nodes_validated)."""
    if cfg.gt_json:
        out = {}
        for it in read_json(cfg.gt_json):
            out[str(it["video_name"]).replace(".mp4", "")] = {
                "game_name": it.get("game_name", "Unknown"), "bugs": it.get("bugs", []),
                "time_nodes": it.get("time_nodes", []), "no_bugs": it.get("no_bugs", True)}
        return out
    LOG.warning("--gt_json not given: ground truth is taken from %s. For numbers that are "
                "comparable with the GliDe paper pass the official groundtruth.json.",
                cfg.metadata_name)
    return {r.video_name: {"game_name": r.game, "bugs": list(r.bugs),
                           "time_nodes": [[list(iv) for iv in sp] for sp in r.time_nodes],
                           "no_bugs": r.no_bugs} for r in recs}


# =============================================================================
# Statistics over per-video evaluation results
# =============================================================================

def per_video_f1_iou(v: Dict) -> float:
    return official_aggregate([v])["f1_iou"]


def bootstrap_ci(vr: Sequence[Dict], key: str, B: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(vr)
    vals = [official_aggregate([vr[i] for i in rng.integers(0, n, n)])[key] for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(a: Sequence[Dict], b: Sequence[Dict], key: str, B: int, seed: int,
                     groups: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Paired over videos, or over GAMES when groups is given (cluster bootstrap)."""
    require(len(a) == len(b), "paired bootstrap needs aligned lists")
    rng = np.random.default_rng(seed)
    obs = official_aggregate(a)[key] - official_aggregate(b)[key]
    if groups is None:
        units = [[i] for i in range(len(a))]
    else:
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            gi[g].append(i)
        units = list(gi.values())
    d = []
    for _ in range(B):
        take = [i for u in rng.integers(0, len(units), len(units)) for i in units[u]]
        d.append(official_aggregate([a[i] for i in take])[key]
                 - official_aggregate([b[i] for i in take])[key])
    d = np.asarray(d)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    # The percentile 'p' is reported for continuity with v8 only; it is NOT a
    # test of H0 (see paired_cluster_permutation) and is never used for decisions.
    return {"delta": obs, "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "p_percentile_not_a_test": clamp(p, 0.0, 1.0)}


def paired_cluster_permutation(a: Sequence[Dict], b: Sequence[Dict], key: str, B: int,
                               seed: int, groups: Optional[Sequence[str]] = None
                               ) -> Dict[str, float]:
    """Fisher randomisation test of H0: the two methods' per-video results are
    exchangeable within each cluster (game). Each cluster's A/B assignment is
    flipped jointly; the statistic is the difference of the official aggregate
    (a ratio of sums, so per-video differences cannot simply be sign-flipped).
    Exact enumeration when 2^G <= B, otherwise Monte Carlo with the (1+k)/(1+B)
    correction. The minimum attainable two-sided p is 2^(1-G): with G = 2 games
    no difference can ever be significant at 0.05, which the bootstrap hid."""
    require(len(a) == len(b), "permutation test needs aligned lists")
    if groups is None:
        units = [[i] for i in range(len(a))]
    else:
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            gi[g].append(i)
        units = list(gi.values())
    G = len(units)
    stat = lambda A, Bv: official_aggregate(A)[key] - official_aggregate(Bv)[key]
    obs = stat(a, b)

    def flipped(mask: Sequence[bool]) -> float:
        A, Bv = list(a), list(b)
        for u, f in zip(units, mask):
            if f:
                for i in u:
                    A[i], Bv[i] = b[i], a[i]
        return stat(A, Bv)

    tol = 1e-12
    if G <= 20 and 2 ** G <= B:
        import itertools
        ds = [flipped(m) for m in itertools.product((False, True), repeat=G)]
        p = float(np.mean([abs(d) >= abs(obs) - tol for d in ds]))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        k = sum(abs(flipped(rng.random(G) < 0.5)) >= abs(obs) - tol for _ in range(B))
        p = (1.0 + k) / (1.0 + B)
        exact = False
    return {"delta": obs, "p": float(clamp(p, 0.0, 1.0)), "n_clusters": G, "exact": exact,
            "min_attainable_p": float(2.0 ** (1 - G)) if G > 0 else 1.0}


def paired_wilcoxon(a: Sequence[Dict], b: Sequence[Dict]) -> Dict[str, float]:
    xa = np.array([per_video_f1_iou(v) for v in a])
    xb = np.array([per_video_f1_iou(v) for v in b])
    nz = int(np.sum(np.abs(xa - xb) > 1e-12))
    if nz < 10:
        return {"stat": float("nan"), "p": 1.0, "n_nonzero": nz}
    st, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(st), "p": float(p), "n_nonzero": nz}


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, still = len(items), {}, True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        still = still and p <= thr
        out[k] = {"p": p, "threshold": thr, "reject": bool(still)}
    return out


# =============================================================================
# Evidence collection, diagnostics and calibration
# =============================================================================

def gather(cfg: Config, vlm: VLMBase, recs: Sequence[Record], wins_map: Dict[str, List[Window]],
           hypotheses: bool = True) -> Dict[str, Any]:
    out, t0 = {}, time.time()
    for i, r in enumerate(recs):
        w = wins_map[r.vid]
        out[r.vid] = (collect_video(vlm, r, w, cfg) if hypotheses
                      else collect_generic_evidence(vlm, r, w, cfg))
        if (i + 1) % 25 == 0:
            LOG.info("evidence %d/%d videos | %d logical calls (%d run, %d cached, %d "
                     "top-logprob truncations) | %.1f min", i + 1, len(recs), vlm.budget.calls,
                     vlm.budget.model_calls, vlm.budget.cache_hits,
                     vlm.budget.logprob_truncations, (time.time() - t0) / 60)
    return out


def _signal_report(name: str, scores: List[np.ndarray], labels: List[np.ndarray],
                   games: List[str], cfg: Config) -> Dict[str, Any]:
    s_all = np.concatenate(scores)
    y_all = np.concatenate(labels)
    g_all = [g for g, v in zip(games, scores) for _ in range(len(v))]
    B = min(cfg.n_bootstrap, 1000)
    pooled = cluster_bootstrap_auc(s_all, y_all, g_all, B, cfg.seed)
    within = cluster_bootstrap_auc(np.concatenate(within_video_scores(scores)), y_all, g_all, B,
                                   cfg.seed + 1)
    pv = per_video_auc(scores, labels)
    return {"name": name, "pooled_auc": pooled, "within_video_auc": within,
            "per_video_auc_median": float(np.median(pv)) if pv.size else float("nan"),
            "per_video_auc_mean": float(pv.mean()) if pv.size else float("nan"),
            "n_videos_both_classes": int(pv.size), "variance": icc_oneway(scores)}


def cmd_diagnose(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], split: str,
                 hypotheses: bool) -> Dict[str, Any]:
    """Signal audit on one split (never 'test'). Without hypotheses it only reads
    the generic detector, which is cached from v8/v9, so it costs no model calls
    on a warm cache. With hypotheses it is the v10 go/no-go probe: is the
    hypothesis-conditioned profile a better WITHIN-video ('when') signal than
    the generic detector on the same windows?"""
    require(split != "test", "diagnostics must not touch the test split")
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
    LOG.info("diagnose: %d videos over %d games", len(recs), len({r.game for r in recs}))
    wm = prepare_split(recs, cfg)
    ev = gather(cfg, vlm, recs, wm, hypotheses)
    labels = [window_labels(r, wm[r.vid], cfg.label_min_overlap) for r in recs]
    games = [r.game for r in recs]
    gen = [e.generic if hypotheses else e for e in (ev[r.vid] for r in recs)]
    series = {"lo_raw": [np.asarray(g.lo_raw) for g in gen],
              "lo_abs": [np.asarray(g.lo_abs) for g in gen]}
    if hypotheses:
        series["hyp_max"] = [np.max(np.stack([h.profile for h in ev[r.vid].hyps]), axis=0)
                             if ev[r.vid].hyps else np.asarray(ev[r.vid].generic.lo_abs)
                             for r in recs]
        series["hyp_top1"] = [np.asarray(ev[r.vid].hyps[0].profile) if ev[r.vid].hyps
                              else np.asarray(ev[r.vid].generic.lo_abs) for r in recs]
    rep = {k: _signal_report(k, v, labels, games, cfg) for k, v in series.items()}
    y_all = np.concatenate(labels)
    pos = np.array([float(y.mean()) if y.size else 0.0 for y in labels])
    whole = []
    for r in recs:
        gt = normalize_intervals([iv for sp in r.time_nodes for iv in sp])
        if gt:
            whole.append(official_iou([list(x) for x in gt], [[0.0, float(r.duration)]]))
    base = {"n_windows": int(y_all.size), "base_rate": float(y_all.mean()),
            "median_video_positive_fraction": float(np.median(pos)),
            "videos_over_half_positive": float(np.mean(pos > 0.5)),
            "whole_video_span_iou_mean": float(np.mean(whole)) if whole else float("nan")}
    LOG.info("windows %d | base rate %.3f | median positive fraction per video %.3f | %.1f%% "
             "of videos over half glitch", base["n_windows"], base["base_rate"],
             base["median_video_positive_fraction"], 100 * base["videos_over_half_positive"])
    LOG.info("TRIVIAL GROUNDING BASELINE: span = whole video gives mean IoU %.3f with the "
             "union of GT spans; any localisation must beat this", base["whole_video_span_iou_mean"])
    for k, d in rep.items():
        v = d["variance"]
        LOG.info("%-8s pooled AUC %.3f [%.3f, %.3f] | within-video AUC %.3f [%.3f, %.3f] | "
                 "per-video AUC median %.3f (n=%d) | sd within %.3f between %.3f ICC %.3f", k,
                 d["pooled_auc"]["auc"], d["pooled_auc"]["lo"], d["pooled_auc"]["hi"],
                 d["within_video_auc"]["auc"], d["within_video_auc"]["lo"],
                 d["within_video_auc"]["hi"], d["per_video_auc_median"],
                 d["n_videos_both_classes"], v["sd_within"], v["sd_between"], v["icc"])
    cmp_ = {}
    if hypotheses:
        a_h = [roc_auc(s, y) for s, y in zip(series["hyp_max"], labels)]
        a_g = [roc_auc(s, y) for s, y in zip(series["lo_abs"], labels)]
        keep = [i for i in range(len(recs)) if not (math.isnan(a_h[i]) or math.isnan(a_g[i]))]
        d = np.array([a_h[i] - a_g[i] for i in keep])
        gk = [games[i] for i in keep]
        if d.size >= 10 and np.any(np.abs(d) > 1e-12):
            _, pw = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        else:
            pw = 1.0
        rng = np.random.default_rng(cfg.seed)
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(gk):
            gi[g].append(i)
        units = list(gi.values())
        boot = [float(np.mean(np.concatenate([d[units[u]] for u in
                                              rng.integers(0, len(units), len(units))])))
                for _ in range(min(cfg.n_bootstrap, 2000))] if units else [float("nan")]
        cmp_ = {"mean_delta_per_video_auc": float(d.mean()) if d.size else float("nan"),
                "game_bootstrap_ci": [float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))],
                "wilcoxon_p": float(pw), "n_videos": int(d.size)}
        LOG.info("hyp_max vs lo_abs per-video AUC: delta %+.3f, game-bootstrap 95%% CI "
                 "[%+.3f, %+.3f], Wilcoxon p=%.4f (n=%d)", cmp_["mean_delta_per_video_auc"],
                 *cmp_["game_bootstrap_ci"], cmp_["wilcoxon_p"], cmp_["n_videos"])
    key = "hyp_max" if hypotheses else "lo_abs"
    w = rep[key]["within_video_auc"]
    go = not math.isnan(w["lo"]) and w["lo"] > 0.5
    LOG.info("VERDICT (%s within-video AUC): %s", key,
             "GO -- the score localises glitches inside videos" if go else
             "NO-GO -- no evidence that the score localises glitches inside videos")
    out = {"split": split, "n_videos": len(recs), "hypotheses": hypotheses, "base": base,
           "signals": rep, "hyp_vs_generic": cmp_, "go": go, "budget": asdict(vlm.budget)}
    atomic_write_json(os.path.join(cfg.paths()["params"],
                                   f"{'probe' if hypotheses else 'diagnose'}_{split}.json"), out)
    return out


def parse_grid(txt: str) -> List[float]:
    g = [float(x) for x in str(txt).split(",") if x.strip()]
    require(g, f"empty grid '{txt}'")
    return g


def gt_bug_list(rec: Record) -> List[Dict[str, Any]]:
    return [{"description": b, "spans": [list(iv) for iv in rec.time_nodes[i]]}
            for i, b in enumerate(rec.bugs) if i < len(rec.time_nodes) and rec.time_nodes[i]]


def match_hypotheses(cfg: Config, judge: Optional[JudgeBase], rec: Record, wins: Sequence[Window],
                     vh: VideoHypotheses) -> Dict[int, Tuple[int, int]]:
    """Official protocol applied to the CANDIDATE set: Hungarian assignment on
    the judge's 0-5 rating matrix (spans play no role in the matching, exactly
    as in evaluation/evaluator.py). Returns {hyp_idx: (gt_idx, rating)}.
    calib_label='iou' is a description-blind fallback when no judge is served:
    Hungarian on window-level IoU, rating 5 for any overlap. It ignores what
    the judge rewards and is only for debugging the pipeline."""
    gts = gt_bug_list(rec)
    if not gts or not vh.hyps:
        return {}
    R = np.zeros((len(vh.hyps), len(gts)))
    if cfg.calib_label == "judge":
        require(judge is not None, "calib_label=judge needs a judge")
        for i, h in enumerate(vh.hyps):
            for j, g in enumerate(gts):
                R[i, j] = judge.rate(g["description"], h.description)
    else:
        for i, h in enumerate(vh.hyps):
            sp = windows_to_spans([t for a, b in hypothesis_segments(h, cfg, 0.0, 1.0)
                                   for t in range(a, b + 1)], wins)
            for j, g in enumerate(gts):
                R[i, j] = official_iou(g["spans"], [list(x) for x in sp])
    pi, gi = linear_sum_assignment(-R)
    if cfg.calib_label == "judge":
        return {int(a): (int(b), int(R[a, b])) for a, b in zip(pi, gi)}
    return {int(a): (int(b), 5 if R[a, b] > 0 else 0) for a, b in zip(pi, gi)}


def select_f1_threshold(u_hat: np.ndarray, u: np.ndarray, M: int) -> Dict[str, Any]:
    """Exact maximiser over thresholds of the plug-in dataset F1xIoU
         F(t) = 2 sum_{u_hat >= t} u / (N(t) + M),   u = rating/5 * IoU in [0, 1].
    Lipton, Elkan & Naryanaswamy (2014) showed that for calibrated probabilities
    the F1-optimal threshold is F*/2; the same argument with graded utilities
    gives 'emit iff E[u] > F*/2', reported as a calibration check."""
    order = np.argsort(-u_hat, kind="mergesort")
    cs = np.cumsum(u[order])
    Ns = np.arange(1, u.size + 1)
    F = 2 * cs / (Ns + M) if M > 0 else np.zeros(u.size)
    if F.size == 0 or F.max() <= 0:
        return {"threshold": float("inf"), "f1_iou": 0.0, "n_emitted": 0, "lipton_ratio": float("nan")}
    k = int(np.argmax(F))
    t = float(u_hat[order[k]])
    return {"threshold": t, "f1_iou": float(F[k]), "n_emitted": int(k + 1),
            "lipton_ratio": float(t / F[k]) if F[k] > 0 else float("nan")}


def _const_util_model(feats: Sequence[str], p: float) -> Dict[str, Any]:
    p = clamp(p, 1e-4, 1 - 1e-4)
    return {"features": list(feats), "mu": [0.0] * len(feats), "sd": [1.0] * len(feats),
            "w": [0.0] * len(feats), "b": float(math.log(p / (1 - p))), "l2": 0.0}


def fit_utility(rows: Sequence[Dict[str, float]], u: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Fractional-response logistic regression (Papke & Wooldridge 1996): the
    Bernoulli quasi-likelihood with targets in [0, 1] is consistent for
    E[u | x]; IRLS as in fit_logistic."""
    if len(rows) < 5 or float(np.max(u)) <= 0 or float(np.min(u)) >= 1:
        return _const_util_model(HYP_FEATURES, float(np.mean(u)) if len(u) else 0.0)
    m = fusion_fit(rows, u, HYP_FEATURES, cfg.fusion_l2)
    m["version"] = VERSION
    return m


def _iou_of(spans: Sequence[Interval], gt: Sequence[Sequence[float]]) -> float:
    return official_iou([list(x) for x in gt], [list(x) for x in spans])


def cmd_calibrate(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]],
                  calib_splits: str = "dev", tag: str = "",
                  judge: Optional[JudgeBase] = None) -> Dict:
    recs = _calib_recs(cfg, splits, calib_splits)
    LOG.info("calibration pool: %d videos, %d games (splits: %s, labels: %s)", len(recs),
             len({r.game for r in recs}), calib_splits, cfg.calib_label)
    require(len({r.game for r in recs}) >= 3, "calibration needs >= 3 games")
    if cfg.calib_label == "iou":
        LOG.warning("calib_label=iou ignores descriptions; use it for debugging only")
    wm = prepare_split(recs, cfg)
    ev = gather(cfg, vlm, recs, wm, True)
    by = {r.vid: r for r in recs}
    match = {r.vid: match_hypotheses(cfg, judge, r, wm[r.vid], ev[r.vid]) for r in recs}
    M = sum(len(gt_bug_list(r)) for r in recs)

    # ---- 1. segmentation (tau, lambda) on CORRECT hypotheses, by game-grouped CV --
    correct = [(r.vid, i, j) for r in recs for i, (j, rt) in match[r.vid].items()
               if rt >= cfg.min_correct_rating]
    LOG.info("candidate hypotheses: %d over %d GT bugs; correct (rating >= %d): %d",
             sum(len(ev[r.vid].hyps) for r in recs), M, cfg.min_correct_rating, len(correct))
    taus, lams = parse_grid(cfg.seg_taus), parse_grid(cfg.seg_penalties)
    grid = [(t, l) for t in taus for l in lams]

    def win_iou(item, t, l) -> float:
        vid, i, j = item
        sp, _ = hypothesis_spans(vlm, by[vid], wm[vid], ev[vid].hyps[i], cfg, t, l, "window")
        return _iou_of(sp, gt_bug_list(by[vid])[j]["spans"])

    seg_table, tau, lam, seg_oof = {}, cfg.seg_tau, cfg.seg_penalty, float("nan")
    if correct and cfg.use_segmentation:
        mat = np.array([[win_iou(it, t, l) for (t, l) in grid] for it in correct])
        seg_table = {f"{t:g}|{l:g}": float(v) for (t, l), v in zip(grid, mat.mean(0))}
        k = int(np.argmax(mat.mean(0)))
        tau, lam = grid[k]
        fold = game_folds([by[v].game for v, _, _ in correct], cfg.cv_folds, cfg.seed)
        oof = []
        for f in sorted(set(fold.values())):
            te = [n for n, (v, _, _) in enumerate(correct) if fold[by[v].game] == f]
            tr = [n for n, (v, _, _) in enumerate(correct) if fold[by[v].game] != f]
            kk = int(np.argmax(mat[tr].mean(0))) if tr else k
            oof.extend(mat[te, kk].tolist())
        seg_oof = float(np.mean(oof)) if oof else float("nan")
        LOG.info("segmentation: tau=%g lambda=%g | window IoU on correct hypotheses %.3f "
                 "(game-CV estimate %.3f)", tau, lam, float(mat[:, k].mean()), seg_oof)

    # ---- 2. refine mode: localised spans versus the trivial whole-video span -----
    mode_iou = {}
    for mode in REFINE_MODES:
        vals = []
        for vid, i, j in correct:
            sp, _ = hypothesis_spans(vlm, by[vid], wm[vid], ev[vid].hyps[i], cfg, tau, lam, mode)
            vals.append(_iou_of(sp, gt_bug_list(by[vid])[j]["spans"]))
        mode_iou[mode] = float(np.mean(vals)) if vals else float("nan")
    valid = {m: v for m, v in mode_iou.items() if not math.isnan(v)}
    refine = max(valid, key=lambda m: (valid[m], m == cfg.refine_mode)) if valid else cfg.refine_mode
    LOG.info("refine mode %s selected; mean IoU on correct hypotheses: %s", refine,
             {m: round(v, 3) for m, v in mode_iou.items()})
    if refine == "whole":
        LOG.warning("the WHOLE-VIDEO span beats every localised span on calibration: the "
                    "profile does not localise glitches better than the trivial baseline")

    # ---- 3. utility labels, fractional-logistic model, F1xIoU-optimal threshold --
    rows, u, correct_flag, games = [], [], [], []
    for r in recs:
        vh, gts = ev[r.vid], gt_bug_list(r)
        for k, row in enumerate(hypothesis_rows(vh, cfg, tau, lam)):
            val = 0.0
            if k in match[r.vid] and match[r.vid][k][1] > 0:
                j, rt = match[r.vid][k]
                sp, _ = hypothesis_spans(vlm, r, wm[r.vid], vh.hyps[k], cfg, tau, lam, refine)
                val = rt / 5.0 * _iou_of(sp, gts[j]["spans"])
            rows.append(row)
            u.append(val)
            correct_flag.append(int(k in match[r.vid] and match[r.vid][k][1] >= cfg.min_correct_rating))
            games.append(r.game)
    u = np.asarray(u, float)
    require(len(rows) >= 20, f"only {len(rows)} candidate hypotheses; need >= 20")
    fold = game_folds(games, cfg.cv_folds, cfg.seed)
    oof = np.zeros(len(rows))
    for f in sorted(set(fold.values())):
        te = np.array([fold[g] == f for g in games])
        m = fit_utility([rows[i] for i in np.where(~te)[0]], u[~te], cfg)
        oof[te] = fusion_predict(m, [rows[i] for i in np.where(te)[0]])
    sel = select_f1_threshold(oof, u, M)
    auc = cluster_bootstrap_auc(oof, correct_flag, games, cfg.n_bootstrap, cfg.seed)
    top1 = np.array([rw["rank_inv"] == 1.0 for rw in rows])
    f_all = float(2 * u.sum() / (u.size + M)) if M else 0.0
    f_top1 = float(2 * u[top1].sum() / (int(top1.sum()) + M)) if M else 0.0
    LOG.info("utility model (out-of-fold): AUC for 'correct' %.3f CI [%.3f, %.3f] | F1xIoU "
             "proxy %.4f at t=%.4f with %d emitted (t/F* = %.2f, theory 0.5) | emit-all %.4f | "
             "emit-top1 %.4f", auc["auc"], auc["lo"], auc["hi"], sel["f1_iou"], sel["threshold"],
             sel["n_emitted"], sel["lipton_ratio"], f_all, f_top1)
    reasons = []
    if math.isnan(auc["lo"]) or auc["lo"] <= cfg.min_auc_ci_lo:
        reasons.append(f"utility model cannot rank correct hypotheses: AUC {auc['auc']:.3f} "
                       f"CI [{auc['lo']:.3f}, {auc['hi']:.3f}]")
    if sel["f1_iou"] <= max(f_all, f_top1) + 1e-12:
        LOG.warning("selection does not beat emitting all / top-1 hypotheses on calibration")
    final = fit_utility(rows, u, cfg)
    LOG.info("final utility weights (standardised): %s",
             {f: round(w, 3) for f, w in zip(final["features"], final["w"])})
    out: Dict[str, Any] = {
        "version": VERSION, "fusion": final, "decision_threshold": sel["threshold"],
        "seg_tau": tau, "seg_penalty": lam, "refine_mode": refine,
        "segmentation": {"table": seg_table, "oof_window_iou": seg_oof},
        "refine_mode_iou": mode_iou,
        "selection": {**sel, "f1_iou_emit_all": f_all, "f1_iou_emit_top1": f_top1,
                      "oof_auc_correct": auc},
        "n_hypotheses": len(rows), "n_correct": int(sum(correct_flag)), "n_gt_bugs": M,
        "calib_splits": calib_splits, "calib_label": cfg.calib_label, "n_videos": len(recs),
        "degenerate": bool(reasons), "degeneracy_reasons": reasons,
        "config_switches": {k: getattr(cfg, k) for k in asdict(cfg) if k.startswith("use_")}}
    if reasons:
        path = _calib_path(cfg, (tag + "_" if tag else "") + "degenerate")
        atomic_write_json(path, out)
        msg = "DEGENERATE SELECTOR: " + "; ".join(reasons) + f". Diagnostics: {path}."
        if cfg.fail_on_degenerate:
            raise RuntimeError(msg + " (--allow_degenerate exists for ablation bookkeeping, "
                                     "not for headline numbers)")
        LOG.error(msg)
    atomic_write_json(_calib_path(cfg, tag), out)
    return out


def _calib_path(cfg: Config, tag: str = "") -> str:
    return os.path.join(cfg.paths()["params"], f"calibration{('_' + tag) if tag else ''}.json")


def _calib_recs(cfg: Config, splits: Dict[str, List[Record]], names: str) -> List[Record]:
    names_l = [s.strip() for s in names.split(",") if s.strip()]
    require("test" not in names_l, "calibration may never read the test split")
    recs: List[Record] = []
    for n in names_l:
        recs.extend(splits[n])
    return take_limit(recs, cfg.limit, cfg.stratify_limit)


def apply_calibration(cfg: Config, tag: str = "") -> Config:
    path = _calib_path(cfg, tag)
    require(os.path.exists(path), f"calibration missing at {path}: run 'calibrate' first")
    cal = read_json(path)
    require(str(cal.get("version", "")).startswith("10"),
            f"{path} was not written by HOVER v10; re-run 'calibrate'")
    sw = cal.get("config_switches", {})
    diff = {k: (v, getattr(cfg, k)) for k, v in sw.items() if getattr(cfg, k) != v}
    require(not diff, f"component switches differ from calibration {path}: {diff}")
    return Config(**{**asdict(cfg), "fusion": cal["fusion"],
                     "decision_threshold": float(cal["decision_threshold"]),
                     "seg_tau": float(cal["seg_tau"]), "seg_penalty": float(cal["seg_penalty"]),
                     "refine_mode": str(cal["refine_mode"])})


# =============================================================================
# run / import / export
# =============================================================================

def cmd_run(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], method: str,
            split: str, tag: str = "") -> str:
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
    wm = prepare_split(recs, cfg)
    forced_mode = cfg.refine_mode  # remember what the caller asked for
    if method == "HOVER":
        cfg = apply_calibration(cfg, tag)
        # If the caller named a specific refine_mode on the CLI, that overrides
        # the mode stored in the calibration. This is what makes the per-mode
        # rows genuinely distinct instead of three copies of the same run.
        cfg = Config(**{**asdict(cfg), "refine_mode": forced_mode})
    b0 = asdict(vlm.budget)
    outs = []
    for i, r in enumerate(recs):
        before, st0 = vlm.budget.calls, dict(vlm.budget.by_stage)
        if method == "HOVER":
            o = run_HOVER(vlm, r, wm[r.vid], cfg)
        elif method == "vanilla":
            o = run_vanilla(vlm, r, wm[r.vid], cfg)
        else:
            raise ValueError(f"unknown method {method} (GliDe is ingested with import_glide)")
        o["calls"] = vlm.budget.calls - before
        o["calls_by_stage"] = {k: v - st0.get(k, 0) for k, v in vlm.budget.by_stage.items()
                               if v - st0.get(k, 0) > 0}
        o["duration"] = r.duration
        outs.append(o)
        if (i + 1) % 25 == 0:
            LOG.info("%s %d/%d | %.1f logical calls/video", method, i + 1, len(recs),
                     (vlm.budget.calls - b0["calls"]) / (i + 1))
    name = f"{method}{('_' + tag) if tag else ''}_{split}.json"
    path = os.path.join(cfg.paths()["runs"], name)
    atomic_write_json(path, {"method": method + (f"_{tag}" if tag else ""), "split": split,
                             "config": asdict(cfg), "outputs": outs,
                             "calls_per_video": (vlm.budget.calls - b0["calls"]) / max(1, len(recs)),
                             "logprob_truncations": vlm.budget.logprob_truncations
                             - b0["logprob_truncations"]})
    atomic_write_json(path.replace(".json", "_batch_report.json"), to_batch_report(outs))
    LOG.info("wrote %s (+ official-format batch report)", path)
    return path


def cmd_import_glide(cfg: Config, splits: Dict[str, List[Record]], split: str,
                     batch_report: str) -> str:
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
    outs, info = import_glide_report(batch_report, recs)
    path = os.path.join(cfg.paths()["runs"], f"glide_official_{split}.json")
    atomic_write_json(path, {"method": "glide_official", "split": split, "outputs": outs,
                             "source": os.path.abspath(batch_report), "import_info": info,
                             "calls_per_video": None})
    LOG.info("imported official GliDe report: %s -> %s", info, path)
    return path


def cmd_export(run_file: str) -> str:
    data = read_json(run_file)
    out = run_file.replace(".json", "_batch_report.json")
    atomic_write_json(out, to_batch_report(data["outputs"]))
    LOG.info("wrote %s (feed to GliDe's evaluation/run.py for an independent check)", out)
    return out


# =============================================================================
# evaluate / ablate
# =============================================================================

def cmd_evaluate(cfg: Config, judge: JudgeBase, splits: Dict[str, List[Record]], split: str,
                 run_files: Sequence[str]) -> Dict:
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
    gt = load_gt_official(cfg, recs)
    split_names = {r.video_name for r in recs}
    game_of = {r.video_name: r.game for r in recs}
    runs = []
    for rf in run_files:
        path = rf if os.path.isabs(rf) or os.path.exists(rf) else os.path.join(cfg.paths()["runs"], rf)
        d = read_json(path)
        runs.append((d["method"], {o["video_name"]: o for o in d["outputs"]}, d))
    common = sorted(split_names & set(gt) & set.intersection(*[set(m) for _, m, _ in runs]))
    require(common, "no video is shared by the split, the ground truth and every run file")
    LOG.info("evaluating %d videos common to all %d runs (official protocol; judge=%s @ T=%.1f)",
             len(common), len(runs), cfg.judge_model, cfg.judge_temperature)
    per: Dict[str, List[Dict]] = {}
    for name, m, _ in runs:
        per[name] = [dict(official_eval_video(judge, gt[v], m[v]), video_name=v) for v in common]
    groups = [game_of.get(v, "unknown") for v in common]
    table, cmp_ = {}, {}
    keys = ("precision", "recall", "f1", "mean_iou", "precision_iou", "recall_iou", "f1_iou")
    for name, _, d in runs:
        agg = official_aggregate(per[name])
        ci = {k: bootstrap_ci(per[name], k, cfg.n_bootstrap, cfg.seed) for k in ("f1", "f1_iou")}
        table[name] = {**agg, "ci": ci, "calls_per_video": d.get("calls_per_video")}
    ref = runs[0][0]
    pv = {}
    for name, _, _ in runs[1:]:
        c = {"bootstrap_ci_video": paired_bootstrap(per[ref], per[name], "f1_iou",
                                                    cfg.n_bootstrap, cfg.seed),
             "bootstrap_ci_game": paired_bootstrap(per[ref], per[name], "f1_iou",
                                                   cfg.n_bootstrap, cfg.seed, groups),
             "permutation_game": paired_cluster_permutation(per[ref], per[name], "f1_iou",
                                                            cfg.n_permutations, cfg.seed, groups),
             "wilcoxon": paired_wilcoxon(per[ref], per[name])}
        cmp_[f"{ref} vs {name}"] = c
        pv[f"{ref} vs {name}"] = c["permutation_game"]["p"]
    holm = holm_bonferroni(pv) if pv else {}
    print("\n" + "=" * 112)
    print(f"Official GliDe protocol | split={split} | videos={len(common)} | judge={cfg.judge_model}")
    print("-" * 112)
    print(f"{'method':<24}" + "".join(f"{k:>12}" for k in keys) + f"{'calls/vid':>12}")
    for name, row in table.items():
        cpv = row.get("calls_per_video")
        print(f"{name:<24}" + "".join(f"{100 * row[k] if k != 'mean_iou' else row[k]:>12.2f}"
                                     for k in keys) + f"{(cpv if cpv is not None else float('nan')):>12.1f}")
    for k, c in cmp_.items():
        g, pm = c["bootstrap_ci_game"], c["permutation_game"]
        print(f"  {k}: dF1xIoU {100 * g['delta']:+.2f} game-bootstrap 95% CI "
              f"[{100 * g['ci_lo']:+.2f}, {100 * g['ci_hi']:+.2f}] | game sign-flip permutation "
              f"p={pm['p']:.4f} ({'exact' if pm['exact'] else 'MC'}, G={pm['n_clusters']}, "
              f"min p={pm['min_attainable_p']:.4f}; Holm reject={holm.get(k, {}).get('reject')}) "
              f"| Wilcoxon p={c['wilcoxon']['p']:.4f}")
    print("=" * 112)
    res = {"split": split, "n_videos": len(common), "table": table, "comparisons": cmp_,
           "holm": holm, "judge": {"model": cfg.judge_model, "temperature": cfg.judge_temperature,
                                   "calls": judge.calls, "parse_failures": judge.parse_failures},
           "per_video": per}
    atomic_write_json(os.path.join(cfg.paths()["eval"], f"eval_{split}.json"), res)
    return res


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_ctx_calib": {"use_contextual_calibration": False},
    "no_hypothesis_scoring": {"use_hypothesis_scoring": False},   # generic profile
    "no_event_resolution": {"use_event_resolution": False},
    "no_segmentation": {"use_segmentation": False},               # member runs
    "no_subframe": {"use_subframe": False},
    "no_utility_selection": {"use_utility_selection": False},     # emit every hypothesis
}


def cmd_ablate(cfg: Config, vlm: VLMBase, judge: JudgeBase, splits: Dict[str, List[Record]],
               split: str, calib_splits: str) -> Dict:
    """Each variant is RE-CALIBRATED (utility model, threshold, segmentation and
    refine mode) on the calibration splits; VLM and judge calls are cached, so
    a variant costs only the calls its own switches add. Degenerate variants are
    reported, not hidden."""
    full = cmd_run(cfg, vlm, splits, "HOVER", split)
    files = [full]
    for tag, over in ABLATIONS.items():
        vcfg = Config(**{**asdict(cfg), **over, "fail_on_degenerate": False})
        cmd_calibrate(vcfg, vlm, splits, calib_splits, tag, judge)
        files.append(cmd_run(vcfg, vlm, splits, "HOVER", split, tag))
    return cmd_evaluate(cfg, judge, splits, split, files)


# =============================================================================
# Self-tests: unit tests, parity with the official repository, end-to-end mock
# =============================================================================

def _noise(key: str) -> float:
    """Deterministic ~N(0,1) from a string (sum of 12 uniforms - 6)."""
    h = hashlib.sha256(key.encode()).digest()
    return float(sum(b / 255.0 for b in h[:12]) - 6.0)


def _write_video(path: str, seconds: float, fps: int = 10, size=(96, 64), seed: int = 0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, 3)
    for k in range(int(seconds * fps)):
        fr = np.zeros((size[1], size[0], 3), np.uint8)
        fr[:] = base
        x = (k * 3) % (size[0] - 10)
        fr[20:30, x:x + 10] = (255, 255, 255)
        vw.write(fr)
    vw.release()


def _build_synthetic_benchmark(root: str) -> None:
    games = {"dev": ["AlphaRace", "BetaFight", "GammaSim", "DeltaShoot", "IotaPlat",
                     "KappaPuzzle", "LambdaRPG", "MuSport"],
             "test": ["EpsRace", "ZetaSim"], "train": ["EtaRPG"]}
    rows, splits = [], defaultdict(list)
    k = 0
    for split, gl in games.items():
        for g in gl:
            for v in range(3 if split in ("dev", "test") else 1):
                name = f"{split}_{g}_{v}"
                buggy = (v % 3) != 2
                _write_video(os.path.join(root, "raw", g, name + ".mp4"), 12.0, seed=k)
                k += 1
                rows.append({"video_name": name + ".mp4", "game_name": g,
                             "bugs": ["The red car floats above the road surface."] if buggy else [],
                             "time_nodes": [[[4, 8]]] if buggy else [], "no_bugs": not buggy})
                splits[split].append(name)
    atomic_write_json(os.path.join(root, "annotations.json"), rows)
    for s, ids in splits.items():
        atomic_write_json(os.path.join(root, "splits", f"{s}.json"), ids)


_KEEP_RE = re.compile(r"_keep_(\d+)_(\d+)\.jpg$")


def _mock_oracle(label_of: Dict[str, int], video_bias: Callable[[str], float], signal: bool,
                 salt: str = "rnd", frame_label_of: Optional[Dict[Tuple[str, int], int]] = None):
    """Scripted VLM that reproduces the HOVER v9 probe on VideoGlitchBench:
      * the GENERIC detector is dominated by a per-video offset (sd 1.5) and
        only weakly tied to the window label (+0.3 logits), i.e. high ICC and a
        near-chance within-video AUC;
      * open descriptions name the true glitch on glitch windows and sometimes
        hallucinate an unrelated one on normal windows;
      * the HYPOTHESIS-conditioned presence query is informative for a true
        hypothesis and flat for a hallucinated one.
    With signal=False descriptions and presence are independent of the labels.
    The self-test therefore checks that v10 USES such evidence when it exists
    and REFUSES to calibrate when it does not; it says nothing about how much
    of it a real VLM provides (that is what 'probe' measures)."""
    frame_label_of = frame_label_of or {}

    def win_key(path: str) -> str:
        ap = os.path.abspath(path)
        stem = re.sub(r"_(stitched|pair|keep_\d+_\d+)\.jpg$", "", ap)
        return stem + "_stitched.jpg"

    def rel(path: str) -> str:
        # Noise keys use the last two path components only, so the self-test is
        # reproducible although its data live in a random temporary directory.
        return "/".join(os.path.abspath(path).split(os.sep)[-2:])

    def vid_of(path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.abspath(path)))

    def lab(path: str) -> int:
        m = _KEEP_RE.search(path)
        if m:
            d = os.path.dirname(os.path.abspath(path))
            return int(any(frame_label_of.get((d, f), 0)
                           for f in range(int(m.group(1)), int(m.group(2)) + 1)))
        return int(label_of.get(win_key(path), 0))

    CAR = "The red car floats above the road surface."
    WALL = "A wall texture flickers briefly."

    def oracle(kind, system, user, images, options):
        if kind == "score":
            img = images[0]
            if system == P_DETECT_SYS:
                if img == "__content_free__":
                    lo = 0.6
                else:
                    lo = (0.6 + 1.5 * video_bias(vid_of(img)) + 0.3 * lab(img)
                          + 0.4 * _noise(salt + "d" + rel(img)))
                return {"yes": lo / 2, "no": -lo / 2}
            if system == P_PRESENCE_SYS:
                if img == "__content_free__":
                    lo = -1.0
                elif not signal:
                    lo = -1.0 + 1.0 * _noise(salt + "v" + rel(img) + user[-40:])
                elif "red car" in user:
                    lo = 3.0 * lab(img) - 1.5 + 0.3 * _noise(salt + "v" + rel(img))
                else:
                    lo = -2.0 + 0.3 * _noise(salt + "w" + rel(img))
                return {"yes": lo / 2, "no": -lo / 2}
            raise AssertionError(f"unexpected score prompt: {system[:40]}")
        if system in (P_DESC_OPEN_SYS, P_DESC_SYS):
            img = images[0]
            if signal:
                pos = lab(img)
                hall = _noise(salt + "h" + rel(img)) > 0.9
            else:
                pos = int(_noise(salt + "r" + rel(img)) > 0.3)
                hall = not pos and _noise(salt + "h" + rel(img)) > 0.0
            if system == P_DESC_SYS:        # forced description: always a glitch
                pos = pos or not hall
            if pos:
                return json.dumps({"glitch": True, "description": CAR, "entity": "red car",
                                   "glitch_type": "floating"})
            if hall:
                return json.dumps({"glitch": True, "description": WALL, "entity": "wall",
                                   "glitch_type": "texture"})
            return json.dumps({"glitch": False, "description": "", "entity": "", "glitch_type": ""})
        if system == P_EAR_SYS:
            return json.dumps({"events": []})
        if system == P_VANILLA_SYS:
            return json.dumps({"no_bugs": False, "bugs": [
                {"description": "A car floats in the air.", "time_nodes": [[3, 9]]}]})
        m = re.search(r"^- (.+)$", user, flags=re.M)      # summariser: echo the first input
        return m.group(1) if m else CAR
    return oracle


def selftest(glide_repo: str = "", verbose: bool = True) -> None:
    import tempfile
    fails, n = [], [0]

    def chk(name: str, cond: bool, detail: str = "") -> None:
        n[0] += 1
        if not cond:
            fails.append(f"{name}: {detail}")
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    if not LOG.handlers:
        logging.basicConfig(level=logging.WARNING)
    print("== unit tests ==")
    chk("official_iou both empty = 1", official_iou([], []) == 1.0)
    chk("official_iou one empty = 0", official_iou([[0, 1]], []) == 0.0)
    chk("official_iou half", abs(official_iou([[0, 4]], [[0, 2]]) - 0.5) < 1e-12)
    chk("official_iou multi", abs(official_iou([[0, 2], [4, 6]], [[1, 5]]) - 2 / 6) < 1e-12)
    chk("official_iou zero-length pred", official_iou([[3, 4]], [[3, 3]]) == 0.0)
    chk("normalize merges", normalize_intervals([(3, 4), (0, 1), (0.5, 2)]) == [(0, 2), (3, 4)])
    chk("json fenced", parse_json_from_text('x ```json\n{"a": 1}\n``` y') == {"a": 1})
    chk("json repaired", parse_json_from_text("{'a': True,}") == {"a": True})
    chk("json none", parse_json_from_text("no json here") is None)

    c0 = Config(temperature=0.5, top_k=-1)
    g = build_generation_kwargs(c0, 32)
    chk("sampling: top_k disabled explicitly (v7 inherited top_k=1)", g["top_k"] == 0 and g["do_sample"])
    g = build_generation_kwargs(Config(temperature=0.0), 32)
    chk("greedy: fully specified", g["do_sample"] is False and g["top_k"] == 0)

    top = [{"token": "Yes", "logprob": -0.2}, {"token": " yes", "logprob": -2.0},
           {"token": "Maybe", "logprob": -3.0}]
    m, tr = match_top_logprobs(top, YES_NO)
    chk("top_logprobs: variants summed", abs(m["yes"] - float(logsumexp([-0.2, -2.0]))) < 1e-9)
    chk("top_logprobs: absent option -> floor + flag", m["no"] == -3.0 and tr)
    nl = normalise_option_logprobs({"a": -1.0, "b": -2.0})
    chk("option normalisation", abs(math.exp(nl["a"]) + math.exp(nl["b"]) - 1) < 1e-12)

    z = robust_z(np.array([0, 0, 0, 0, 10.0]))
    chk("robust z: outlier stands out, floor on MAD", z[-1] > 5 and abs(z[0]) < 1e-9)

    # ---- S: exact penalised segmentation == brute force -------------------------
    import itertools

    def brute_seg(x, tau, lam):
        T, best = len(x), -np.inf
        for mask in itertools.product((0, 1), repeat=T):
            if not any(mask):
                continue
            segs = member_runs([t for t in range(T) if mask[t]])
            best = max(best, segment_score(x, segs, tau, lam))
        return best
    rng_s = np.random.default_rng(11)
    ok_seg = True
    for trial in range(40):
        T = int(rng_s.integers(1, 8))
        x = rng_s.normal(0, 1.5, T)
        tau, lam = float(rng_s.choice([-0.5, 0.0, 0.7])), float(rng_s.choice([0.0, 0.8, 3.0, 1e9]))
        segs = segment_profile(x, tau, lam)
        ok_seg &= bool(segs) and abs(segment_score(x, segs, tau, lam) - brute_seg(x, tau, lam)) < 1e-9
        ok_seg &= all(b2 >= a2 for a2, b2 in segs) and all(
            segs[k + 1][0] > segs[k][1] + 1 for k in range(len(segs) - 1))
    chk("segmentation DP attains the brute-force optimum (40 random profiles)", ok_seg)
    xk = np.array([1., -3., 2., 2., -0.5, 1., -4., 0.5])
    chk("lambda -> inf gives the single maximum-sum segment (Kadane)",
        segment_profile(xk, 0.0, 1e9) == [(2, 5)], str(segment_profile(xk, 0.0, 1e9)))
    chk("lambda = 0 gives every run above tau",
        segment_profile(xk, 0.0, 0.0) == [(0, 0), (2, 3), (5, 5), (7, 7)], str(segment_profile(xk, 0.0, 0.0)))
    chk("all-negative profile still yields its best window",
        segment_profile([-3., -1., -2.], 0.0, 1.0) == [(1, 1)])

    # ---- U: F1xIoU-optimal selection --------------------------------------------
    rng_u = np.random.default_rng(5)
    uh = rng_u.random(60)
    uu = (rng_u.random(60) < uh) * rng_u.random(60)
    sel = select_f1_threshold(uh, uu, 25)
    brute = max(2 * uu[uh >= t].sum() / ((uh >= t).sum() + 25) for t in uh)
    chk("F1xIoU threshold equals brute-force maximum", abs(sel["f1_iou"] - brute) < 1e-12)
    pc = rng_u.random(200000)
    yc = (rng_u.random(200000) < pc).astype(float)
    selc = select_f1_threshold(pc, yc, int(yc.sum()))
    chk("calibrated scores: optimal threshold = F*/2 (Lipton et al. 2014)",
        abs(selc["lipton_ratio"] - 0.5) < 0.02, f"{selc['lipton_ratio']:.3f}")
    Xf = rng_u.normal(size=(3000, 2))
    uf = expit(Xf @ np.array([1.2, -0.8]) - 0.5)
    wf, bf = fit_logistic(Xf, uf, l2=1e-6)
    chk("fractional logistic recovers E[u|x] for targets in [0, 1]",
        np.allclose(wf, [1.2, -0.8], atol=1e-3) and abs(bf + 0.5) < 1e-3, f"{wf}, {bf}")

    # ---- diagnostics: variance decomposition and within-video AUC ----------------
    vids = [rng_u.normal(0, 1) + rng_u.normal(0, 1, 30) for _ in range(400)]
    ic = icc_oneway(vids)
    chk("ICC(1) recovers 0.5 for equal variance components", abs(ic["icc"] - 0.5) < 0.05, str(ic))
    offs = rng_u.normal(0, 3, 200)
    labs = [(rng_u.random(10) < 0.5).astype(int) for _ in range(200)]
    scs = [o + 1.0 * y_ + rng_u.normal(0, 0.5, 10) for o, y_ in zip(offs, labs)]
    raw_auc = roc_auc(np.concatenate(scs), np.concatenate(labs))
    cen_auc = roc_auc(np.concatenate(within_video_scores(scs)), np.concatenate(labs))
    chk("within-video centring removes video offsets", cen_auc > raw_auc + 0.1,
        f"{raw_auc:.3f} -> {cen_auc:.3f}")
    rr_ = [Record(f"v{i}", f"v{i}", g_, "u", "", False, [], []) for i, g_ in
           enumerate(["A"] * 6 + ["B"] * 6 + ["C"] * 6)]
    chk("stratified --limit covers every game", [r_.game for r_ in take_limit(rr_, 5)]
        == ["A", "B", "C", "A", "B"])

    # ---- cluster permutation test validity ---------------------------------------
    mk = lambda sc, io: {"gt_num": 1, "pred_num": 1, "matched_scores": [sc], "matched_ious": [io]}
    A2 = [mk(5, 1.0) for _ in range(6)]
    B2 = [mk(0, 0.0) for _ in range(6)]
    pm2 = paired_cluster_permutation(A2, B2, "f1_iou", 1000, 0, ["g1"] * 3 + ["g2"] * 3)
    chk("permutation: G=2 can never reach p<0.05 (v8 bootstrap reported 0.0000)",
        pm2["exact"] and pm2["p"] >= 0.5, str(pm2))
    g8 = [f"g{i // 2}" for i in range(16)]
    pm8 = paired_cluster_permutation([mk(5, 1.0)] * 16, [mk(0, 0.0)] * 16, "f1_iou", 1000, 0, g8)
    chk("permutation: G=8 exact, maximal effect reaches min p = 2^(1-G)",
        pm8["exact"] and abs(pm8["p"] - 2 ** -7) < 1e-12, str(pm8))
    rng_p = np.random.default_rng(3)
    Ar = [mk(int(rng_p.integers(0, 6)), float(rng_p.random())) for _ in range(40)]
    Br = [mk(int(rng_p.integers(0, 6)), float(rng_p.random())) for _ in range(40)]
    gr = [f"g{i % 20}" for i in range(40)]
    pmr = paired_cluster_permutation(Ar, Br, "f1_iou", 999, 0, gr)
    chk("permutation: Monte-Carlo branch, null data not rejected", (not pmr["exact"]) and pmr["p"] > 0.05,
        str(pmr))

    # ---- derived images and frame-level likelihood bisection ----------------------
    if cv2 is not None:
        td0 = tempfile.mkdtemp()
        frs = []
        for k in range(8):
            f_ = np.full((120, 200, 3), 60, np.uint8)
            if k == 3:
                f_[70:110, 120:190] = 255         # transient glitch in ONE frame
            frs.append(f_)
        ids = list(range(40, 48))
        sp_ = os.path.join(td0, "window_0005_stitched.jpg")
        cv2.imwrite(sp_, stitch_official(frs, ids), [cv2.IMWRITE_JPEG_QUALITY, 95])
        w5 = Window(5, 10.0, 12.0, sp_, ids, [10.0 + 0.25 * k for k in range(8)])
        mk_ = cv2.imread(masked_window_path(w5, 2, 4))
        mc_, *_ = _split_cells(mk_, 8)
        chk("masked window greys frames outside the kept range",
            all(abs(float(mc_[k][60:110, 10:110].mean()) - (60 if 2 <= k <= 4 else 128)) < 6
                for k in range(8)))

        class _FrameStub:
            def __init__(self, glitch_ids):
                self.g, self.calls = set(glitch_ids), 0

            def image(self, path, stage="x"):
                self.calls += 1
                m = _KEEP_RE.search(path)
                return any(f in self.g for f in range(int(m.group(1)), int(m.group(2)) + 1))
        for gl_ids in ([43, 44, 45], [40], [47], [41, 42, 43, 44, 45, 46, 47]):
            stub = _FrameStub(gl_ids)
            au_ = {"monotonicity_violations": 0}
            on_, off_ = frame_onset(stub, w5, au_), frame_offset(stub, w5, au_)
            chk(f"frame bisection recovers first/last glitch frame {gl_ids[0]}..{gl_ids[-1]}",
                ids[on_] == min(gl_ids) and ids[off_] == max(gl_ids) and stub.calls <= 8,
                f"{ids[on_]}..{ids[off_]} in {stub.calls} calls")
        chk("frame times use decoded timestamps", frame_time_bounds(w5, 3, 5, 4.0) == (10.75, 11.5))

    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    brute = np.mean([(a > c) + 0.5 * (a == c) for a, ya in zip(s, y) if ya == 1
                     for c, yc in zip(s, y) if yc == 0])
    chk("roc_auc equals brute force", abs(roc_auc(s, y) - brute) < 1e-12)
    chk("roc_auc constant = 0.5", roc_auc(np.ones(10), [0, 1] * 5) == 0.5)
    X = rng.normal(size=(4000, 2))
    yy = (rng.random(4000) < expit(X @ np.array([1.5, -1.0]) + 0.3)).astype(int)
    w, bb = fit_logistic(X, yy, l2=1e-6)
    chk("IRLS recovers logistic weights", np.allclose(w, [1.5, -1.0], atol=0.15) and abs(bb - 0.3) < 0.15,
        f"{w}, {bb}")
    gms = [f"g{i % 7}" for i in range(70)]
    fd = game_folds(gms, 5, 1)
    chk("game folds are game-disjoint", all(len({fd[gm]}) == 1 for gm in set(gms)))
    grp = [f"g{i % 10}" for i in range(400)]
    ss = rng.normal(size=400)
    yy2 = (ss + rng.normal(size=400) > 0).astype(int)
    cb = cluster_bootstrap_auc(ss, yy2, grp, 300, 0)
    chk("cluster bootstrap CI brackets point", cb["lo"] <= cb["auc"] <= cb["hi"] and cb["lo"] > 0.5)
    cb0 = cluster_bootstrap_auc(rng.normal(size=400), yy2, grp, 300, 0)
    chk("cluster bootstrap CI includes 0.5 under no signal", cb0["lo"] <= 0.5 <= cb0["hi"])

    # ---- official evaluator port on a hand-computed case -------------------------
    cfgj = Config(judge_backend="mock", work_subdir="selftest_work", root=tempfile.mkdtemp())
    table = {("gA", "pA"): 5, ("gB", "pC"): 4}
    judge = MockJudge(cfgj, lambda gt_, pr_: table.get((gt_, pr_), 0))
    v1 = official_eval_video(judge, {"bugs": ["gA", "gB"], "time_nodes": [[[0, 4]], [[10, 12]]]},
                             {"bugs": ["pA", "pB", "pC"],
                              "time_nodes": [[[0, 2]], [[20, 22]], [[10, 12]]]})
    v2 = official_eval_video(judge, {"bugs": [], "time_nodes": []},
                             {"bugs": ["pX"], "time_nodes": [[[0, 1]]]})
    v3 = official_eval_video(judge, {"bugs": ["gQ"], "time_nodes": [[[0, 5]]]},
                             {"bugs": ["pQ"], "time_nodes": [[[0, 5]]]})
    agg = official_aggregate([v1, v2, v3])
    N, M = 5, 3
    P, R = 9 / (5 * N), 9 / (5 * M)
    Pi, Ri = 6.5 / (5 * N), 6.5 / (5 * M)
    chk("port: rating-0 pairs are counted", agg["num_matched"] == 3 and v3["matched_scores"] == [0])
    chk("port: P/R/F1", abs(agg["precision"] - P) < 1e-12 and abs(agg["recall"] - R) < 1e-12
        and abs(agg["f1"] - 2 * P * R / (P + R)) < 1e-12, str(agg))
    chk("port: *_iou", abs(agg["f1_iou"] - 2 * Pi * Ri / (Pi + Ri)) < 1e-12)
    chk("port: mean_iou over all pairs", abs(agg["mean_iou"] - (0.5 + 1.0 + 1.0) / 3) < 1e-12)

    repo = glide_repo or os.environ.get("GLIDE_REPO", "")
    if repo and os.path.isdir(repo):
        print("== parity with the official GliDe repository ==")
        sys.path.insert(0, repo)
        try:
            from evaluation.evaluator import Evaluator as OfficialEvaluator  # type: ignore

            class _Stub:
                def chat(self, system_msg, user_msg, images=None):
                    gt_ = user_msg.split("**Ground Truth Description**:\n", 1)[1].split("\n\n**Pred", 1)[0]
                    pr_ = user_msg.split("**Predicted Description**:\n", 1)[1].split("\n\n**Instr", 1)[0]
                    return json.dumps({"reasoning": "", "rating": table.get((gt_, pr_), 0)})
            td = tempfile.mkdtemp()
            gtf, prf = os.path.join(td, "gt.json"), os.path.join(td, "pr.json")
            atomic_write_json(gtf, [
                {"video_name": "v1.mp4", "bugs": ["gA", "gB"], "time_nodes": [[[0, 4]], [[10, 12]]], "no_bugs": False},
                {"video_name": "v2.mp4", "bugs": [], "time_nodes": [], "no_bugs": True},
                {"video_name": "v3.mp4", "bugs": ["gQ"], "time_nodes": [[[0, 5]]], "no_bugs": False}])
            atomic_write_json(prf, [
                {"video_name": "v1", "bugs": ["pA", "pB", "pC"], "time_nodes": [[[0, 2]], [[20, 22]], [[10, 12]]]},
                {"video_name": "v2", "bugs": ["pX"], "time_nodes": [[[0, 1]]]},
                {"video_name": "v3", "bugs": ["pQ"], "time_nodes": [[[0, 5]]]}])
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                off = OfficialEvaluator(llm_client=_Stub(), verbose=False).evaluate(gtf, prf).to_dict()
            keys = ("num_gt_bugs", "num_pred_bugs", "num_matched", "mean_iou", "precision",
                    "recall", "f1", "precision_iou", "recall_iou", "f1_iou")
            chk("evaluator port == official Evaluator (all metrics)",
                all(abs(float(off[k]) - float(agg[k])) < 1e-12 for k in keys),
                str({k: (off[k], agg[k]) for k in keys}))
            with open(os.path.join(repo, "evaluation", "prompt.txt")) as f:
                chk("embedded judge prompt == official prompt.txt", f.read() == JUDGE_PROMPT)
            with open(os.path.join(repo, "summarizer", "system_prompt.txt")) as f:
                chk("embedded summariser prompt == official", f.read().strip() == P_SUMMARY_TEMPLATE.strip())
            if cv2 is not None:
                from preprocess.video_preprocessor import VideoPreprocessor  # type: ignore
                vp = os.path.join(td, "clip.mp4")
                _write_video(vp, 5.3, fps=30, size=(160, 96), seed=3)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    VideoPreprocessor(output_path=__import__("pathlib").Path(td), target_fps=4.0,
                                      window_size=8).process_video(vp)
                rr = Record("clip", "clip", "g", "u", vp, False, [], [])
                cP = Config(root=td, work_subdir="w")
                ws = prepare_record(rr, cP, force=True)
                offw = sorted(os.listdir(os.path.join(td, "windows", "clip")))
                offw = [x for x in offw if x.endswith("_stitched.jpg")]
                chk("preprocessing: same number of windows", len(offw) == len(ws), f"{len(offw)} vs {len(ws)}")
                diffs = []
                for w_, name_ in zip(ws, offw):
                    a_ = cv2.imread(w_.stitch_path).astype(int)
                    b_ = cv2.imread(os.path.join(td, "windows", "clip", name_)).astype(int)
                    diffs.append(np.abs(a_ - b_).max() if a_.shape == b_.shape else 999)
                chk("preprocessing: stitched windows pixel-identical", max(diffs) == 0, str(diffs))
        except Exception as e:  # pragma: no cover
            chk("official repository parity checks ran", False, f"{type(e).__name__}: {e}")
        finally:
            sys.path.remove(repo)
    else:
        print("== parity with the official repository: SKIPPED (pass --glide_repo) ==")

    if cv2 is None:
        print("== end-to-end mock pipeline: SKIPPED (opencv missing) ==")
    else:
        print("== end-to-end mock pipeline ==")
        root = tempfile.mkdtemp()
        _build_synthetic_benchmark(root)
        base = Config(root=root, backend="mock", judge_backend="mock", n_bootstrap=300,
                      cv_folds=4)
        recs = load_records(base)
        spl = load_splits(base, recs)
        wm = prepare_split(spl["dev"] + spl["test"] + spl["train"], base)
        label_of, frame_label_of = {}, {}
        for r in spl["dev"] + spl["test"]:
            gt_iv = [iv for spans in r.time_nodes for iv in spans]
            for w_, y_ in zip(wm[r.vid], window_labels(r, wm[r.vid], base.label_min_overlap)):
                label_of[os.path.abspath(w_.stitch_path)] = int(y_)
                d_ = os.path.dirname(os.path.abspath(w_.stitch_path))
                for fid, ft in zip(w_.frame_ids, w_.frame_times):
                    frame_label_of[(d_, fid)] = int(any(a_ <= ft < b_ for a_, b_ in gt_iv))
        chk("synthetic labels contain both classes", 0 < sum(label_of.values()) < len(label_of))
        vbias = lambda v: _noise("video-bias" + v)
        vlm = make_vlm(base, _mock_oracle(label_of, vbias, True, frame_label_of=frame_label_of))
        judge_fn = lambda gt_, pr_: 5 if ("float" in pr_.lower() and "float" in gt_.lower()) else 0
        judge = make_judge(base, judge_fn)

        dg = cmd_diagnose(base, vlm, spl, "dev", hypotheses=True)
        sg_ = dg["signals"]
        chk("diagnose: generic detector is video-dominated (ICC > 0.5), as in the v9 probe",
            sg_["lo_abs"]["variance"]["icc"] > 0.5, str(sg_["lo_abs"]["variance"]))
        chk("diagnose: hypothesis profile localises better within videos than the generic score",
            sg_["hyp_max"]["within_video_auc"]["auc"] > sg_["lo_abs"]["within_video_auc"]["auc"] + 0.1,
            f"{sg_['hyp_max']['within_video_auc']['auc']:.3f} vs "
            f"{sg_['lo_abs']['within_video_auc']['auc']:.3f}")
        chk("diagnose: trivial whole-video IoU reported",
            abs(dg["base"]["whole_video_span_iou_mean"] - 4 / 12) < 0.05,
            str(dg["base"]["whole_video_span_iou_mean"]))

        cal = cmd_calibrate(base, vlm, spl, "dev", "", judge)
        chk("signal oracle: selector not degenerate", not cal["degenerate"], str(cal["degeneracy_reasons"]))
        chk("localised spans beat the whole-video span on calibration",
            cal["refine_mode"] != "whole" and cal["refine_mode_iou"]["whole"] < cal["refine_mode_iou"]["frame"],
            str(cal["refine_mode_iou"]))
        chk("selection beats emitting every hypothesis on calibration",
            cal["selection"]["f1_iou"] > cal["selection"]["f1_iou_emit_all"], str(cal["selection"]))
        f_HOVER = cmd_run(base, vlm, spl, "HOVER", "test")
        run_ = read_json(f_HOVER)
        byname = {r.video_name: r for r in spl["test"]}
        buggy = [o for o in run_["outputs"] if not byname[o["video_name"]].no_bugs]
        chk("frame-level grounding recovers the synthetic [4, 8] s span on every glitch video",
            all(o["time_nodes"] == [[[4.0, 8.0]]] for o in buggy), str([o["time_nodes"] for o in buggy]))
        chk("clean test videos produce no report",
            all(not o["bugs"] for o in run_["outputs"] if byname[o["video_name"]].no_bugs))
        f_van = cmd_run(base, vlm, spl, "vanilla", "test")
        gl = [{"video_name": r.video_name, "game_name": r.game, "no_bugs": False,
               "bugs": ["Something odd happens."], "time_nodes": [[[0, 1]]]} for r in spl["test"][:-1]]
        gpath = os.path.join(root, "glide_batch_report.json")
        atomic_write_json(gpath, gl)
        f_gl = cmd_import_glide(base, spl, "test", gpath)
        res = cmd_evaluate(base, judge, spl, "test", [f_HOVER, f_van, f_gl])
        chk("evaluation restricted to common videos", res["n_videos"] == len(spl["test"]) - 1)
        chk("HOVER beats a content-free baseline on the mock",
            res["table"]["HOVER"]["f1_iou"] > res["table"]["glide_official"]["f1_iou"])
        chk("batch report written in official format",
            os.path.exists(f_HOVER.replace(".json", "_batch_report.json")))
        vg = Config(**{**asdict(base), "use_hypothesis_scoring": False, "fail_on_degenerate": False})
        cmd_calibrate(vg, vlm, spl, "dev", "no_hypothesis_scoring", judge)
        chk("ablation with the generic profile calibrates and runs",
            os.path.exists(cmd_run(vg, vlm, spl, "HOVER", "test", "no_hypothesis_scoring")))
        vlm0 = make_vlm(base, _mock_oracle(label_of, vbias, False, frame_label_of=frame_label_of))
        try:
            cmd_calibrate(base, vlm0, spl, "dev", "", judge)
            chk("no-signal oracle is rejected as degenerate", False, "calibrate did not raise")
        except RuntimeError as e:
            chk("no-signal oracle is rejected as degenerate", "DEGENERATE" in str(e), str(e)[:200])
        ab = Config(**{**asdict(base), "use_segmentation": False})
        try:
            apply_calibration(ab)
            chk("switch mismatch with calibration is refused", False)
        except RuntimeError:
            chk("switch mismatch with calibration is refused", True)

    print(f"\n{n[0] - len(fails)}/{n[0]} checks passed")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        raise SystemExit(1)


# =============================================================================
# UAL-Bench: query-conditioned HOVER localisation core (Mode A)
#
# Everything in this block is new and dataset-isolated. No function above it
# was modified: VideoGlitchBench commands, paths, caches, calibrations and runs
# are untouched (enforced by ual_resolve_work_root and tested in selftest_ual).
#
# Official protocol, verified in https://github.com/Hasnat79/UAL_Bench:
#   annotations  data/uag_oops/oops_uag_paper_version.json
#                  dict {video_id: {start_time, end_time, description}}   1589 items
#                data/uag_ssbd/ssbd_paper_version.json
#                  list [[video_tag, {id, time 'MMSS:MMSS', category, intensity, ...}]]
#                  104 items over 58 videos (up to 7 behaviours per video)
#                data/uag_funqa/uag_funqa_dataset.json
#                  list [{instruction, visual_input, ID, start_time, end_time, output, task}]  172 items
#   item ids     src/data_loaders/ssbd_loader.py: video_tag + '_' + behaviour id; OOPS: dict key;
#                FunQA: visual_input (the key of the official prediction files)
#   query text   src/llama3_x_*_uag_oops.py: description;  *_ssbd.py:
#                f"A person is {category} with {intensity} intensity.";  *_funqa.py: instruction
#   evaluator    src/eval/eval_your_results.py (ported exactly in ual_official_metrics)
#   predictions  src/eval/output_test_template.json:
#                  [{video_id, start_time, end_time, pred_start, pred_end}, ...]
#
# Mode A (the primary and only implemented UAL mode). The official query is the
# HOVER hypothesis; H (open description, event resolution, summary) is skipped
# because the query is given, and U (utility selection) is bypassed because
# every item needs exactly one interval and the VideoGlitchBench-fitted model is
# not valid here. What runs is HOVER V -> S plus its frame refinement:
#   V  s_q(w) = logit P(Yes | w, q) - logit P(Yes | grey, q)       (verify_profile)
#   S  argmax over single segments of sum_t (s_q(t) - tau): segment_profile with
#      lambda = 1e9, i.e. the K = 1 case of the HOVER segmentation (Kadane)
#   G  boundary windows refined to frames by masked-frame bisection with a
#      presence oracle at margin tau (frame_onset / frame_offset, unchanged)
# UAL-Bench publishes no train/dev split for the three evaluation sets, so
# nothing is calibrated: tau = 0 (the zero point of contextual calibration),
# lambda = infinity (one interval per item, as the official protocol scores)
# and refine = frame are fixed before any UAL result is seen.
# The only task adapter is the presence prompt (P_UAL_PRESENCE_SYS /
# ual_presence_user): the VideoGlitchBench prompt names games and glitches.
# =============================================================================

UAL_ADAPTER_VERSION = "ual-modeA-1.0"
UAL_PROMPT_VERSION = "ual-presence-1"
UAL_SUBSETS: Tuple[str, ...] = ("uag_oops", "uag_ssbd", "uag_funqa")
UAL_DISPLAY = {"uag_oops": "UAG-OOPS", "uag_ssbd": "UAG-SSBD", "uag_funqa": "UAG-FunQA"}
UAL_ANNOTATION_FILES = {"uag_oops": "oops_uag_paper_version.json",
                        "uag_ssbd": "ssbd_paper_version.json",
                        "uag_funqa": "uag_funqa_dataset.json"}
# The only backbones of the UAL-Bench experiment: key -> (display name, HF id).
UAL_BACKBONES: Dict[str, Tuple[str, str]] = {
    "qwen2.5-vl-7b": ("Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl2.5-8b": ("InternVL2.5-8B", "OpenGVLab/InternVL2_5-8B"),
    "ui-tars-1.5-7b": ("UI-TARS-1.5-7B", "ByteDance-Seed/UI-TARS-1.5-7B")}
# OOPS-UAG-Instruct is training data. The Drive folder names it *_v1.json, the
# official repository data/uag_oops_train_instruct/uag_oops_train_instruct_dataset.json.
UAL_TRAINING_FILES = ("uag_oops_train_instruct_dataset_v1.json",
                      "uag_oops_train_instruct_dataset.json")
UAL_TRAINING_DIRS = ("uag_oops_instruct", "uag_oops_train_instruct")
UAL_MODES = ("query_conditioned", "open_ended_full_pipeline")
UAL_REFINE_MODES = ("frame", "window")
UAL_IOU_THRESHOLDS = (0.3, 0.5, 0.7)          # eval_recall_1_iou
UAL_TD_THRESHOLDS = (0, 1, 3, 5, 7)           # abs_dist
UAL_OFFICIAL_KEYS = ("video_id", "start_time", "end_time", "pred_start", "pred_end")
UAL_MAIN_HEADER = ("Model | R@1,0.3 | R@1,0.5 | R@1,0.7 | R@1,TD≤0 | R@1,TD≤1 | R@1,TD≤3 | "
                   "R@1,TD≤5 | mTD")
UAL_COMMANDS = ("inspect_ual", "run_ual", "evaluate_ual", "selftest_ual")
UAL_PREDECLARED = {"seg_tau": 0.0, "seg_penalty": 1e9, "refine_mode": "frame"}

MODE_B_MESSAGE = (
    "Mode B (--ual_mode open_ended_full_pipeline) is intentionally not implemented. HOVER's H "
    "stage (P_DESC_OPEN_SYS, P_EAR_SYS, the GliDe summariser) asks for video-game glitches and "
    "its U selector was fitted on VideoGlitchBench judge ratings; running them on UAL-Bench "
    "needs new prompts and a new selector, i.e. a different method, plus a validated rule for "
    "matching generated hypotheses to the query. Use the default --ual_mode query_conditioned "
    "(Mode A).")

# P_PRESENCE_SYS with the domain changed from 'game glitch' to 'event', plus the
# frame-layout sentence of P_DETECT_SYS. One prompt for all three subsets, fixed
# before any UAL-Bench result was computed.
P_UAL_PRESENCE_SYS = (
    "You are a video analyst. You are given one stitched window of consecutive video frames, "
    "each labelled '#<frame index>' in temporal order (left to right, top row then bottom "
    "row), and a query about one specific event. Decide whether that event is visibly "
    "occurring in this window. Being plausible is not enough; it must be observable.")


def ual_presence_user(query: str) -> str:
    """The query is inserted verbatim. Nothing else about the item (and never a
    timestamp) reaches the model."""
    return (f"Query: {query}\nIs the event described or requested by this query visibly "
            f"occurring in this window? Answer with exactly one word: Yes or No.")


# -----------------------------------------------------------------------------
# Records: the model-visible part and the evaluator-only part are separate types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UALQuery:
    """Model-visible part of one evaluation item. It has no timestamp fields on
    purpose: ground truth lives in UALGroundTruth and only the evaluator joins it."""
    subset: str
    item_id: str           # official evaluation key ('video_id' in the prediction schema)
    video_id: str          # video file stem (several SSBD items share one video)
    video_file: str
    query: str
    meta: Tuple[Tuple[str, str], ...] = ()   # official non-temporal fields, audit only


@dataclass(frozen=True)
class UALGroundTruth:
    item_id: str
    start_time: float
    end_time: float
    source: str            # verbatim source of the timestamps


@dataclass
class UALDataset:
    subset: str
    annotation_path: str
    annotation_sha256: str     # depends on GT: metadata only, never in a cache key
    inputs_fingerprint: str    # GT-free: (item_id, video_file, query) of every item
    queries: List[UALQuery]
    gt: Dict[str, UALGroundTruth]
    video_paths: Dict[str, Optional[str]]
    video_dirs: List[str]


@dataclass
class UALSettings:
    subsets: Tuple[str, ...] = ()
    root: str = ""
    annotations: str = ""
    video_root: str = ""
    subset_annotations: Dict[str, str] = field(default_factory=dict)
    subset_videos: Dict[str, str] = field(default_factory=dict)
    work_root: str = ""
    mode: str = "query_conditioned"
    seg_tau: float = 0.0
    seg_penalty: float = 1e9
    refine_mode: str = "frame"
    ssbd_time_format: str = "mmss"
    allow_missing_videos: bool = False
    api_bases: Dict[str, str] = field(default_factory=dict)
    limit: Optional[int] = None
    official_repo: str = ""

    @property
    def partial(self) -> bool:
        return bool(self.limit)

    @property
    def run_tag(self) -> str:
        return f"partial_limit{self.limit}" if self.limit else "full"

    def predeclared(self) -> bool:
        return (self.seg_tau, self.seg_penalty, self.refine_mode) == (
            UAL_PREDECLARED["seg_tau"], UAL_PREDECLARED["seg_penalty"], UAL_PREDECLARED["refine_mode"])


def _ual_short(subset: str) -> str:
    return subset[len("uag_"):]


def _ual_num(x: Any, what: str) -> float:
    require(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)),
            f"{what}: expected a finite number, got {x!r}")
    return float(x)


def ual_parse_ssbd_time(t: Any, fmt: str) -> Tuple[float, float]:
    """SSBD 'time' -> (start, end) seconds. The paper-version file stores
    'MMSS:MMSS' (e.g. '0105:0125' = 65 s to 85 s): among the 104 records no
    seconds field reaches 60 although 38 have a field >= 0100, which is
    incompatible with plain seconds. data/README.md calls the values 'seconds'
    and the repository contains no conversion code, so the format is a flag
    (inspect_ual checks both readings against the decoded video durations)."""
    m = re.fullmatch(r"\s*(\d+):(\d+)\s*", str(t))
    require(m is not None, f"SSBD time {t!r} is not of the form 'dddd:dddd'")
    a, b = m.group(1), m.group(2)
    if fmt == "seconds":
        return float(int(a)), float(int(b))
    require(fmt == "mmss", f"unknown --ual_ssbd_time_format {fmt!r}")

    def mmss(x: str) -> float:
        require(len(x) >= 3, f"SSBD time field {x!r} is too short for MMSS")
        mm, ss = int(x[:-2]), int(x[-2:])
        require(ss < 60, f"SSBD time field {x!r} has seconds {ss} >= 60, so it is not MMSS; "
                         f"pass --ual_ssbd_time_format seconds if the file stores seconds")
        return float(60 * mm + ss)
    return mmss(a), mmss(b)


def ual_reject_training_file(path: str) -> None:
    """OOPS-UAG-Instruct is instruction-tuning data, never benchmark input."""
    ap = os.path.abspath(path)
    parts = set(os.path.normpath(ap).split(os.sep))
    base = os.path.basename(ap)
    if base in UAL_TRAINING_FILES or "train_instruct" in base or parts & set(UAL_TRAINING_DIRS):
        raise RuntimeError(
            f"rejected UAL-Bench annotation file {ap}: it is OOPS-UAG-Instruct "
            f"training/instruction-format data, not evaluation data. For UAG-OOPS evaluation use "
            f"oops_uag_paper_version.json (official path data/uag_oops/oops_uag_paper_version.json).")


def ual_annotation_candidates(root: str, subset: str) -> List[str]:
    """Official repository layout first, then the Drive folder layouts."""
    f = UAL_ANNOTATION_FILES[subset]
    return [os.path.join(root, "data", subset, f), os.path.join(root, subset, f),
            os.path.join(root, f)]


def ual_find_annotation(us: UALSettings, subset: str) -> str:
    explicit = us.subset_annotations.get(subset) or us.annotations
    if explicit:
        ual_reject_training_file(explicit)
        require(os.path.isfile(explicit), f"{UAL_DISPLAY[subset]} annotation file not found: "
                                          f"{explicit}")
        return os.path.abspath(explicit)
    require(bool(us.root), f"no annotation path for {UAL_DISPLAY[subset]}: pass --ual_root (the "
                           f"UAL-Bench repository or dataset folder) or "
                           f"--ual_{_ual_short(subset)}_annotations. VideoGlitchBench is never "
                           f"used as a fallback.")
    cands = ual_annotation_candidates(us.root, subset)
    hit = next((c for c in cands if os.path.isfile(c)), None)
    require(hit is not None,
            f"{UAL_DISPLAY[subset]} evaluation annotations ({UAL_ANNOTATION_FILES[subset]}) not "
            f"found; looked in {cands}. Copy the file from the official repository "
            f"(data/{subset}/{UAL_ANNOTATION_FILES[subset]}) or the corrected Drive folder, or pass "
            f"--ual_{_ual_short(subset)}_annotations. VideoGlitchBench is never used as a fallback.")
    ual_reject_training_file(hit)
    return os.path.abspath(hit)


def ual_parse_annotations(subset: str, data: Any, path: str, ssbd_time_format: str = "mmss"
                          ) -> Tuple[List[UALQuery], Dict[str, UALGroundTruth]]:
    """Schema-checked parser of the three paper-version files. Timestamps are
    stored verbatim (SSBD: converted from its 'time' string only); nothing is
    clipped, shortened or repaired. Malformed records abort the load."""
    if (isinstance(data, list) and data and isinstance(data[0], dict)
            and "QA" in data[0] and "video" in data[0]):
        ual_reject_training_file(os.path.join(os.path.dirname(path), UAL_TRAINING_FILES[0]))
    qs: List[UALQuery] = []
    gts: Dict[str, UALGroundTruth] = {}

    def schema(expected: str) -> str:
        return (f"{path} does not have the {UAL_DISPLAY[subset]} paper-version schema (expected "
                f"{expected}, got {type(data).__name__}); is it the file of another subset?")
    if subset == "uag_oops":
        require(isinstance(data, dict), schema("a JSON object {video_id: {start_time, end_time, "
                                               "description}}"))
        for k, v in data.items():
            require(isinstance(v, dict) and {"start_time", "end_time", "description"} <= set(v),
                    f"{path}: record {k!r} lacks start_time/end_time/description")
            iid = str(k)
            qs.append(UALQuery(subset, iid, iid, iid + ".mp4", str(v["description"])))
            gts[iid] = UALGroundTruth(iid, _ual_num(v["start_time"], f"{iid}.start_time"),
                                      _ual_num(v["end_time"], f"{iid}.end_time"),
                                      json.dumps([v["start_time"], v["end_time"]]))
    elif subset == "uag_ssbd":
        require(isinstance(data, list), schema("a JSON list [[video_tag, behaviour], ...]"))
        for n, e in enumerate(data):
            require(isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], str)
                    and isinstance(e[1], dict), f"{path}: entry {n} is not [video_tag, behaviour]")
            tag, b = e
            require({"id", "time", "category", "intensity"} <= set(b),
                    f"{path}: entry {n} ({tag}) lacks id/time/category/intensity")
            iid = f"{tag}_{b['id']}"          # Ssbd_DataLoader.__iter__
            query = f"A person is {b['category']} with {b['intensity']} intensity."
            s, t = ual_parse_ssbd_time(b["time"], ssbd_time_format)
            meta = tuple((k, str(b[k])) for k in ("id", "category", "intensity", "bodypart",
                                                  "modality", "description") if k in b)
            qs.append(UALQuery(subset, iid, tag, tag + ".mp4", query, meta))
            gts[iid] = UALGroundTruth(iid, s, t, str(b["time"]))
    elif subset == "uag_funqa":
        require(isinstance(data, list) and all(isinstance(r, dict) for r in data),
                schema("a JSON list of {instruction, visual_input, start_time, end_time, ...}"))
        for n, r in enumerate(data):
            require({"instruction", "visual_input", "start_time", "end_time"} <= set(r),
                    f"{path}: entry {n} lacks instruction/visual_input/start_time/end_time")
            iid = str(r["visual_input"])
            vf = iid if iid.lower().endswith(".mp4") else iid + ".mp4"
            meta = tuple((k, str(r[k])) for k in ("ID", "task") if k in r)
            qs.append(UALQuery(subset, iid, os.path.splitext(vf)[0], vf, str(r["instruction"]),
                               meta))
            gts[iid] = UALGroundTruth(iid, _ual_num(r["start_time"], f"{iid}.start_time"),
                                      _ual_num(r["end_time"], f"{iid}.end_time"),
                                      json.dumps({"start_time": r["start_time"],
                                                  "end_time": r["end_time"],
                                                  "output": r.get("output")}))
    else:
        raise ValueError(f"unknown UAL-Bench subset {subset}")
    ids = [q.item_id for q in qs]
    dup = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    require(not dup, f"{path}: duplicate evaluation ids {dup[:5]}")
    bad = [i for i, g in gts.items() if g.start_time < 0 or g.end_time < g.start_time]
    require(not bad, f"{path}: {len(bad)} records with negative or reversed GT intervals, e.g. "
                     f"{bad[:5]}; refusing to repair ground truth")
    require(qs, f"{path}: no evaluation items")
    return qs, gts


def ual_video_dirs(us: UALSettings, subset: str) -> List[str]:
    if us.subset_videos.get(subset):
        return [os.path.abspath(us.subset_videos[subset])]
    out: List[str] = []
    for r in ([us.video_root] if us.video_root else []) + ([us.root] if us.root else []):
        out += [os.path.join(r, "data", subset, "videos"), os.path.join(r, subset, "videos"),
                os.path.join(r, subset)]
    if us.video_root and len(us.subsets) == 1:
        out.append(us.video_root)
    return list(dict.fromkeys(os.path.abspath(d) for d in out))


def ual_inputs_fingerprint(queries: Sequence[UALQuery]) -> str:
    blob = json.dumps([[q.item_id, q.video_file, q.query] for q in queries])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def ual_load_dataset(us: UALSettings, subset: str) -> UALDataset:
    path = ual_find_annotation(us, subset)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{path} is not valid JSON ({e})")
    qs, gts = ual_parse_annotations(subset, data, path, us.ssbd_time_format)
    dirs = ual_video_dirs(us, subset)
    vp: Dict[str, Optional[str]] = {}
    for q in qs:
        if q.video_file not in vp:
            vp[q.video_file] = next((os.path.join(d, q.video_file) for d in dirs
                                     if os.path.isfile(os.path.join(d, q.video_file))), None)
    return UALDataset(subset, path, hashlib.sha256(raw).hexdigest(), ual_inputs_fingerprint(qs),
                      qs, gts, vp, dirs)


def ual_select_items(ds: UALDataset, limit: Optional[int]) -> List[UALQuery]:
    """The first `limit` items in official annotation order (deterministic)."""
    return list(ds.queries[:limit]) if limit else list(ds.queries)


def ual_check_media(ds: UALDataset, items: Sequence[UALQuery], allow_missing: bool) -> List[str]:
    missing = sorted({q.video_file for q in items if ds.video_paths.get(q.video_file) is None})
    if missing:
        msg = (f"{UAL_DISPLAY[ds.subset]}: {len(missing)} of {len({q.video_file for q in items})} "
               f"videos not found in {ds.video_dirs} (e.g. {missing[:3]}). Download the videos "
               f"(official README: wget -O {ds.subset}.tar "
               f"https://huggingface.co/datasets/hasnat79/ual_bench/resolve/main/{ds.subset}.tar, "
               f"then tar -xvf) and point --ual_video_root or --ual_{_ual_short(ds.subset)}_videos "
               f"at them")
        require(allow_missing, msg + ", or pass --ual_allow_missing_videos to score those items "
                                     "as null predictions (official convention: IoU 0, TD = "
                                     "gt end_time).")
        LOG.warning("%s; scoring %d items as null predictions (--ual_allow_missing_videos)", msg,
                    sum(1 for q in items if q.video_file in set(missing)))
    return missing


# -----------------------------------------------------------------------------
# Isolation: work root, paths, cache namespace
# -----------------------------------------------------------------------------

def _ual_within(path: str, parent: str) -> bool:
    p, q = os.path.realpath(path), os.path.realpath(parent)
    return p == q or p.startswith(q.rstrip(os.sep) + os.sep)


def ual_resolve_work_root(us: UALSettings, cfg: Config) -> str:
    """Default: a sibling 'HOVER_v10_ual_work' of --ual_root, so the UAL-Bench data
    (or repository clone) is never written to. Refuses any location inside the
    VideoGlitchBench root or work dir, the GliDe repository or the UAL-Bench data."""
    if us.work_root:
        w = us.work_root
    else:
        require(bool(us.root), "pass --ual_work_root (or --ual_root, whose sibling "
                               "'HOVER_v10_ual_work' is then used)")
        w = os.path.join(os.path.dirname(os.path.abspath(us.root.rstrip(os.sep))),
                         "HOVER_v10_ual_work")
    w = os.path.abspath(w)
    protected = [("VideoGlitchBench root (--root)", cfg.root),
                 ("VideoGlitchBench work dir", cfg.paths()["work"])]
    if cfg.glide_repo:
        protected.append(("GliDe repository (--glide_repo)", cfg.glide_repo))
    if us.root:
        protected.append(("UAL-Bench data root (--ual_root)", us.root))
    for d in [us.video_root, *us.subset_videos.values()]:
        if d:
            protected.append(("UAL-Bench video directory", d))
    for p in [us.annotations, *us.subset_annotations.values()]:
        if p:
            protected.append(("UAL-Bench annotation directory", os.path.dirname(os.path.abspath(p))))
    for name, d in protected:
        require(not _ual_within(w, d), f"refusing UAL work root {w}: it lies inside the {name} "
                                       f"{d}; pass --ual_work_root outside it")
    return w


def ual_paths(us: UALSettings, subset: str = "", backbone: str = "") -> Dict[str, str]:
    base = os.path.join(us.work_root, "ual_bench")
    out = {"base": base, "logs": os.path.join(base, "logs"),
           "inspect": os.path.join(base, "inspect"),
           "summary": os.path.join(base, "summary", us.mode, us.run_tag)}
    if subset:
        out["subset"] = os.path.join(base, subset)
    if subset and backbone:
        out["cache"] = os.path.join(base, subset, "cache", backbone)
        out["run"] = os.path.join(base, subset, backbone, us.mode, us.run_tag)
    return out


def ual_prep_config(cfg: Config, us: UALSettings, subset: str) -> Config:
    """A Config whose paths() point into the UAL subtree, so prepare_record (the
    official GliDe geometry, unchanged) writes windows to
    <work>/ual_bench/<subset>/windows/<prep fingerprint>/<video key>/."""
    return Config(**{**asdict(cfg), "root": us.work_root,
                     "work_subdir": os.path.join("ual_bench", subset)})


def ual_cache_namespace(cfg: Config, ds: UALDataset, us: UALSettings) -> Dict[str, Any]:
    """Part of every UAL VLM cache key. It is GT-free on purpose: the dataset is
    identified by inputs_fingerprint, not by the annotation hash."""
    return {"dataset": f"ual_bench/{ds.subset}", "inputs": ds.inputs_fingerprint,
            "adapter": UAL_ADAPTER_VERSION, "prompt": UAL_PROMPT_VERSION, "HOVER": VERSION,
            "mode": us.mode, "prep": cfg.prep_fingerprint(), "backbone": cfg.backbone,
            "model": cfg.model, "top_logprobs": cfg.top_logprobs,
            "sampling": build_generation_kwargs(cfg, 1)}


class _UALKeyMixin:
    ual_namespace: Dict[str, Any] = {}

    def _key(self, kind, system, user, images, extra):
        return sha1(json.dumps({"ual": self.ual_namespace,
                                "k": super()._key(kind, system, user, images, extra)},
                               sort_keys=True))


class UALOpenAIVLM(_UALKeyMixin, OpenAIVLM):
    pass


class UALHFVLM(_UALKeyMixin, HFVLM):
    pass


class UALMockVLM(_UALKeyMixin, MockVLM):
    pass


def make_ual_vlm(cfg: Config, cache_dir: str, namespace: Dict[str, Any],
                 oracle: Optional[Callable] = None) -> VLMBase:
    if cfg.backend == "openai":
        v: VLMBase = UALOpenAIVLM(cfg, cache_dir)
    elif cfg.backend == "hf":
        v = UALHFVLM(cfg, cache_dir)
    elif cfg.backend == "mock":
        require(oracle is not None, "mock backend needs an oracle")
        v = UALMockVLM(cfg, cache_dir, oracle)
    else:
        raise ValueError(f"unknown backend {cfg.backend}")
    v.ual_namespace = dict(namespace)
    return v


def _ual_switch_subset(vlm: VLMBase, cache_dir: str, namespace: Dict[str, Any]) -> None:
    """Reuse one VLM (and, for hf, one loaded model) across subsets."""
    os.makedirs(cache_dir, exist_ok=True)
    vlm.cache_dir = cache_dir
    vlm.ual_namespace = dict(namespace)


# -----------------------------------------------------------------------------
# Backbones: the three approved checkpoints only, no silent substitution
# -----------------------------------------------------------------------------

def ual_resolve_backbones(spec: Optional[str], default_all: bool = False) -> List[str]:
    if not spec:
        require(default_all, "pass --backbone {qwen2.5-vl-7b,internvl2.5-8b,ui-tars-1.5-7b,all}")
        return list(UAL_BACKBONES)
    keys = list(UAL_BACKBONES) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    for k in keys:
        require(k in UAL_BACKBONES, f"backbone {k!r} is not part of the UAL-Bench experiment; "
                                    f"allowed: {sorted(UAL_BACKBONES)} or 'all'")
        require(BACKBONES[k]["hf"] == UAL_BACKBONES[k][1],
                f"BACKBONES[{k!r}] = {BACKBONES[k]['hf']} differs from the approved "
                f"{UAL_BACKBONES[k][1]}")
    return list(dict.fromkeys(keys))


def ual_backbone_config(cfg: Config, key: str, us: UALSettings, model_override: str = ""
                        ) -> Config:
    c = cfg.with_backbone(key)
    require(not model_override or model_override == c.model,
            f"--model {model_override} would replace the approved checkpoint {c.model} of "
            f"{key}; the UAL-Bench experiment never substitutes checkpoints")
    if key in us.api_bases:
        c.api_base = us.api_bases[key]
    return c


def ual_preflight_backbone(c: Config) -> Dict[str, Any]:
    """Cheap checks before any inference. openai: the server must list the exact
    approved model id. hf: Qwen2.5-VL architecture only (Qwen2.5-VL-7B and
    UI-TARS-1.5-7B); InternVL needs vLLM because its remote code hides logits."""
    info: Dict[str, Any] = {"backend": c.backend, "model": c.model, "family": c.family}
    if c.backend == "hf":
        require(c.family == "qwen", f"{c.model}: the hf backend supports Qwen2.5-VL-architecture "
                                    f"checkpoints only (Qwen2.5-VL-7B, UI-TARS-1.5-7B). Serve "
                                    f"InternVL2.5-8B with vLLM (vllm serve {c.model} "
                                    f"--trust-remote-code --max-logprobs 20) and use --backend "
                                    f"openai --ual_api_bases internvl2.5-8b=<url>")
        try:
            from transformers import AutoConfig
            arch = list(AutoConfig.from_pretrained(c.model).architectures or [])
        except Exception as e:
            raise RuntimeError(f"cannot read the config of {c.model}: {type(e).__name__}: {e}")
        require("Qwen2_5_VLForConditionalGeneration" in arch,
                f"{c.model} declares architectures {arch}; HFVLM loads "
                f"Qwen2_5_VLForConditionalGeneration and would not run it correctly")
        info["architectures"] = arch
    elif c.backend == "openai":
        import requests
        url = f"{c.api_base.rstrip('/')}/models"
        hdr = {"Authorization": f"Bearer {c.api_key}"} if c.api_key and c.api_key != "EMPTY" else {}
        try:
            r = requests.get(url, headers=hdr, timeout=30)
            r.raise_for_status()
            served = [str(m.get("id")) for m in (r.json().get("data") or [])]
        except Exception as e:
            raise RuntimeError(f"{c.backbone}: no OpenAI-compatible server at {c.api_base} "
                               f"({type(e).__name__}: {e}). Start it, e.g. vllm serve {c.model} "
                               f"--max-logprobs 20" + (" --trust-remote-code"
                                                       if c.family == "internvl" else "")
                               + f", and pass --ual_api_bases {c.backbone}=<url>/v1")
        require(c.model in served, f"{c.backbone}: the server at {c.api_base} serves {served}, "
                                   f"not {c.model}; refusing to substitute a checkpoint")
        info["served_models"] = served
        info["api_base"] = c.api_base
    return info


def ual_smoke_test(vlm: VLMBase, c: Config, ds: UALDataset, wins: Sequence[Window],
                   q: UALQuery) -> Dict[str, Any]:
    """One real presence score on a real UAL-Bench window plus its content-free
    probe, before the full run. Fails with the exact backend error."""
    usr = ual_presence_user(q.query)
    try:
        lo_w = binary_log_odds(vlm.score(P_UAL_PRESENCE_SYS, usr, [wins[0].stitch_path], YES_NO,
                                         "ual_smoke"), "yes", "no")
        cf = content_free_path(wins) if vlm.needs_pixels else "__content_free__"
        lo_c = binary_log_odds(vlm.score(P_UAL_PRESENCE_SYS, usr, [cf], YES_NO, "ual_smoke"),
                               "yes", "no")
    except Exception as e:
        raise RuntimeError(f"smoke test failed for {c.backbone} ({c.model}, backend {c.backend}) "
                           f"on {UAL_DISPLAY[ds.subset]} item {q.item_id!r}: "
                           f"{type(e).__name__}: {e}")
    require(math.isfinite(lo_w) and math.isfinite(lo_c),
            f"smoke test for {c.backbone}: non-finite log-odds ({lo_w}, {lo_c})")
    return {"item_id": q.item_id, "window_log_odds": lo_w, "content_free_log_odds": lo_c}


# -----------------------------------------------------------------------------
# Preprocessing and Mode A inference (GT never passes through these functions)
# -----------------------------------------------------------------------------

@dataclass
class _UALVideo:
    """The three attributes prepare_record reads and writes."""
    vid: str
    video_path: str
    duration: float = 0.0


def ual_video_key(subset: str, video_file: str, path: Optional[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(video_file)[0])[:80]
    size = os.path.getsize(path) if path and os.path.exists(path) else -1
    return f"{stem}_{sha1(f'{subset}/{video_file}:{size}')[:10]}"


def ual_prepare_video(cfg: Config, us: UALSettings, ds: UALDataset, video_file: str,
                      memo: Dict) -> Tuple[List[Window], float]:
    """Official GliDe windowing via the unchanged prepare_record; one preparation
    per video (SSBD items share videos), failures are memoised too."""
    key = (ds.subset, cfg.prep_fingerprint(), video_file)
    if key not in memo:
        path = ds.video_paths.get(video_file)
        try:
            require(path is not None, f"video {video_file} not found")
            shim = _UALVideo(ual_video_key(ds.subset, video_file, path), str(path))
            wins = prepare_record(shim, ual_prep_config(cfg, us, ds.subset))
            memo[key] = (wins, float(shim.duration or (wins[-1].end if wins else 0.0)))
        except Exception as e:
            memo[key] = e
    val = memo[key]
    if isinstance(val, Exception):
        raise val
    return val


def ual_verify_query(vlm: VLMBase, wins: Sequence[Window], query: str, cfg: Config
                     ) -> Tuple[float, List[float]]:
    """HOVER V (verify_profile) with the official query as the hypothesis."""
    usr = ual_presence_user(query)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_UAL_PRESENCE_SYS, usr, [path], YES_NO, stage),
                               "yes", "no")
    prior = 0.0
    if cfg.use_contextual_calibration:
        prior = lo_of(content_free_path(wins) if vlm.needs_pixels else "__content_free__",
                      "ual_v_calib")
    return prior, [lo_of(w.stitch_path, "ual_v_profile") - prior for w in wins]


class UALPresenceOracle(PresenceOracle):
    """PresenceOracle with the UAL prompt (the parent hard-codes the glitch prompt)."""

    def __init__(self, vlm: VLMBase, wins: Sequence[Window], query: str, cfg: Config,
                 margin: float = 0.0):
        self.vlm, self.cfg, self.margin = vlm, cfg, float(margin)
        self.by_idx = {w.idx: w for w in wins}
        self.usr = ual_presence_user(query)
        self.base = 0.0
        if cfg.use_contextual_calibration and wins:
            cf = content_free_path(wins) if vlm.needs_pixels else "__content_free__"
            self.base = binary_log_odds(vlm.score(P_UAL_PRESENCE_SYS, self.usr, [cf], YES_NO,
                                                  "ual_g_calib"), "yes", "no")
        self.memo: Dict[Any, bool] = {}

    def image(self, path: str, stage: str = "ual_g_frame") -> bool:
        if path not in self.memo:
            lo = binary_log_odds(self.vlm.score(P_UAL_PRESENCE_SYS, self.usr, [path], YES_NO,
                                                stage), "yes", "no")
            self.memo[path] = (lo - self.base) > self.margin
        return self.memo[path]


def ual_select_segment(profile: Sequence[float], tau: float, lam: float
                       ) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    """S with one output interval. With lambda = 1e9 segment_profile already
    returns one segment (Kadane); for a finite lambda the segment with the
    largest sum of (s - tau) is emitted (ties -> earliest)."""
    segs = segment_profile(profile, tau, lam)
    require(bool(segs), "segment_profile returned no segment")
    x = np.asarray(profile, float) - float(tau)
    best = max(segs, key=lambda ab: (float(x[ab[0]:ab[1] + 1].sum()), -ab[0]))
    return (int(best[0]), int(best[1])), [(int(a), int(b)) for a, b in segs]


def ual_infer_item(vlm: VLMBase, q: UALQuery, wins: Sequence[Window], duration: float,
                   cfg: Config, us: UALSettings) -> Dict[str, Any]:
    """Mode A for one item. Inputs are the query, the windows and the decoded
    duration; no ground truth exists in this scope."""
    require(len(wins) > 0, f"{q.item_id}: no windows")
    audit = {"monotonicity_violations": 0, "frame_conflicts": 0}
    prior, prof = ual_verify_query(vlm, wins, q.query, cfg)
    tau, lam = float(us.seg_tau), float(us.seg_penalty)
    (a, b), segs = ual_select_segment(prof, tau, lam)
    lo_t, hi_t = wins[a].start, wins[b].end
    window_span = (float(lo_t), float(hi_t))
    if us.refine_mode == "frame" and cfg.use_subframe:
        oracle = UALPresenceOracle(vlm, wins, q.query, cfg, tau)
        v = cfg.use_verified_bisection
        on_t, off_t = lo_t, hi_t
        if prof[a] > tau:
            k = frame_onset(oracle, wins[a], audit, v)
            on_t = frame_time_bounds(wins[a], k, k, cfg.fps)[0]
        if prof[b] > tau:
            k = frame_offset(oracle, wins[b], audit, v)
            off_t = frame_time_bounds(wins[b], k, k, cfg.fps)[1]
        if off_t > on_t:
            lo_t, hi_t = on_t, off_t
        else:
            audit["frame_conflicts"] += 1
    ps, pe = clamp(float(lo_t), 0.0, duration), clamp(float(hi_t), 0.0, duration)
    ok = pe > ps
    return {"status": "ok" if ok else "degenerate_interval", "error": None,
            "pred_start": round(ps, 6) if ok else None, "pred_end": round(pe, 6) if ok else None,
            "decoded_duration": float(duration), "n_windows": len(wins),
            "window_bounds": [[w.start, w.end] for w in wins], "prior": float(prior),
            "profile": [float(x) for x in prof], "segments": [list(s) for s in segs],
            "selected_segment": [a, b], "window_span": list(window_span),
            "refine_mode": us.refine_mode, **audit}


def ual_run_backbone(c: Config, us: UALSettings, ds: UALDataset, items: Sequence[UALQuery],
                     vlm: VLMBase, memo: Dict, preflight: Dict, smoke: Optional[Dict]) -> str:
    rp = ual_paths(us, ds.subset, c.backbone)
    os.makedirs(rp["run"], exist_ok=True)
    out_path = os.path.join(rp["run"], "raw_outputs.json")
    outs: List[Dict[str, Any]] = []
    t0, b_calls = time.time(), vlm.budget.calls

    def payload(done: bool) -> Dict[str, Any]:
        return {"dataset": "ual_bench", "subset": ds.subset, "backbone": c.backbone,
                "model": c.model, "mode": us.mode, "partial": us.partial, "limit": us.limit,
                "complete": done, "n_items": len(items),
                "inputs_fingerprint": ds.inputs_fingerprint, "items": outs}
    for i, q in enumerate(items):
        before, st0, tr0 = vlm.budget.calls, dict(vlm.budget.by_stage), vlm.budget.logprob_truncations
        rec: Dict[str, Any] = {"item_id": q.item_id, "video_id": q.video_id,
                               "video_file": q.video_file, "query": q.query}
        try:
            wins, dur = ual_prepare_video(c, us, ds, q.video_file, memo)
        except Exception as e:
            st = "video_missing" if ds.video_paths.get(q.video_file) is None else "video_error"
            if st == "video_error":
                LOG.error("%s: cannot decode %s (%s: %s); scored as a null prediction", q.item_id,
                          q.video_file, type(e).__name__, e)
            rec.update({"status": st, "error": f"{type(e).__name__}: {e}", "pred_start": None,
                        "pred_end": None, "decoded_duration": None})
        else:
            rec.update(ual_infer_item(vlm, q, wins, dur, c, us))   # VLM errors abort the run
        rec["calls"] = vlm.budget.calls - before
        rec["calls_by_stage"] = {k: v - st0.get(k, 0) for k, v in vlm.budget.by_stage.items()
                                 if v - st0.get(k, 0) > 0}
        rec["logprob_truncations"] = vlm.budget.logprob_truncations - tr0
        outs.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            LOG.info("%s | %s | %d/%d items | %.1f calls/item | %.1f min", UAL_DISPLAY[ds.subset],
                     c.backbone, i + 1, len(items), (vlm.budget.calls - b_calls) / (i + 1),
                     (time.time() - t0) / 60)
        if (i + 1) % 100 == 0:
            atomic_write_json(out_path.replace(".json", ".inprogress.json"), payload(False))
    atomic_write_json(out_path, payload(True))
    ip = out_path.replace(".json", ".inprogress.json")
    if os.path.exists(ip):
        os.remove(ip)
    atomic_write_json(os.path.join(rp["run"], "manifest.json"),
                      ual_manifest(c, us, ds, items, {"preflight": preflight, "smoke_test": smoke,
                                                      "budget": asdict(vlm.budget)}))
    LOG.info("wrote %s", out_path)
    return out_path


def ual_manifest(c: Config, us: UALSettings, ds: UALDataset, items: Sequence[UALQuery],
                 extra: Dict[str, Any]) -> Dict[str, Any]:
    import platform
    vers: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "numpy": np.__version__,
                            "opencv": getattr(cv2, "__version__", None) if cv2 else None}
    try:
        import scipy as _sp
        vers["scipy"] = _sp.__version__
    except Exception:
        pass
    for mod in ("torch", "transformers", "requests"):
        if mod in sys.modules:
            vers[mod] = getattr(sys.modules[mod], "__version__", None)
    return {"HOVER_version": VERSION, "ual_adapter": UAL_ADAPTER_VERSION,
            "prompt_version": UAL_PROMPT_VERSION, "mode": us.mode,
            "mode_description": "Mode A: query-conditioned HOVER localisation core (V -> S + frame "
                                "refinement); H and U not used",
            "prompts": {"system": P_UAL_PRESENCE_SYS, "user_template": ual_presence_user("{query}")},
            "selection_rule": {"seg_tau": us.seg_tau, "seg_penalty": us.seg_penalty,
                               "refine_mode": us.refine_mode, "predeclared": us.predeclared(),
                               "calibrated_on_ual": False},
            "subset": ds.subset, "annotation_path": ds.annotation_path,
            "annotation_sha256": ds.annotation_sha256, "inputs_fingerprint": ds.inputs_fingerprint,
            "ssbd_time_format": us.ssbd_time_format if ds.subset == "uag_ssbd" else None,
            "video_dirs": ds.video_dirs, "n_items": len(items), "n_items_in_annotation":
            len(ds.queries), "partial": us.partial, "limit": us.limit,
            "backbone": c.backbone, "model": c.model, "backend": c.backend,
            "api_base": c.api_base if c.backend == "openai" else None,
            "generation": build_generation_kwargs(c, 1), "top_logprobs": c.top_logprobs,
            "max_pixels": c.max_pixels, "fps": c.fps, "window_size": c.window_size,
            "prep_fingerprint": c.prep_fingerprint(), "seed": c.seed,
            "HOVER_switches": {k: getattr(c, k) for k in asdict(c) if k.startswith("use_")},
            "versions": vers, "argv": list(sys.argv),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}


# -----------------------------------------------------------------------------
# Official UAL-Bench evaluator: exact port of src/eval/eval_your_results.py
# -----------------------------------------------------------------------------

def ual_calculate_iou(gt_start: float, gt_end: float, pred_start: float, pred_end: float
                      ) -> float:
    """calculate_iou, verbatim (no clipping; raises on a zero union like the original)."""
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    return 1.0 * (intersection / union)


def ual_calculate_abs_dist(gt_start: float, gt_end: float, pred_start: float, pred_end: float
                           ) -> float:
    """calculate_abs_dist: TD = |start error| + |end error| (a SUM, not a max)."""
    return abs(gt_start - pred_start) + abs(gt_end - pred_end)


def ual_per_item(result: Sequence[Dict]) -> Dict[str, List[float]]:
    """Per-item quantities exactly as the official functions compute them:
      iou      0 if pred_start or pred_end is None (eval_recall_1_iou)
      td       gt end_time ('duration = gt_end') if either is None (abs_dist)
      td_mean  td, replaced by gt end_time when td > 10000 (the mTD list)
      onset_*  0 if either is None, else |gt_start - pred_start| <= 1 / 0.25"""
    iou, td, tdm, on1, on025 = [], [], [], [], []
    for v in result:
        gs, ge = v["start_time"], v["end_time"]
        duration = ge
        none = v["pred_start"] is None or v["pred_end"] is None
        if none:
            i, d = 0.0, duration
        else:
            i = ual_calculate_iou(gs, ge, v["pred_start"], v["pred_end"])
            d = ual_calculate_abs_dist(gs, ge, v["pred_start"], v["pred_end"])
        iou.append(i)
        td.append(d)
        tdm.append(duration if d > 10000 else d)
        on1.append(0.0 if none else float(abs(gs - v["pred_start"]) <= 1))
        on025.append(0.0 if none else float(abs(gs - v["pred_start"]) <= 0.25))
    return {"iou": iou, "td": td, "td_mean": tdm, "onset_1s": on1, "onset_025s": on025}


def ual_official_metrics(result: Sequence[Dict]) -> Dict[str, Any]:
    """All numbers printed by eval_your_results.py. Recalls and onset accuracies
    are percentages (100 * correct / N); mIoU is a fraction; mTD is in seconds."""
    n = len(result)
    require(n > 0, "the official evaluator divides by len(result); got 0 items")
    for v in result:
        for k in UAL_OFFICIAL_KEYS[1:]:
            x = v[k]
            ok = ((x is None and k.startswith("pred"))
                  or (isinstance(x, (int, float)) and not isinstance(x, bool)))
            require(ok, f"{v.get('video_id')}: {k} must be a number"
                        f"{' or null' if k.startswith('pred') else ''}, got {x!r}")
    p = ual_per_item(result)
    out: Dict[str, Any] = {"n": n, "n_null_predictions":
                           sum(1 for v in result if v["pred_start"] is None or v["pred_end"] is None)}
    for m in UAL_IOU_THRESHOLDS:
        out[f"r1_iou_{m}"] = 100 * (sum(1 for i in p["iou"] if i >= m) / n)
    out["miou"] = float(np.mean(p["iou"]))
    for t in UAL_TD_THRESHOLDS:
        out[f"r1_td_{t}"] = 100 * (sum(1 for d in p["td"] if d <= t) / n)
    out["mtd"] = float(np.mean(p["td_mean"]))
    out["onset_acc_1s"] = 100 * (sum(p["onset_1s"]) / n)
    out["onset_acc_0.25s"] = 100 * (sum(p["onset_025s"]) / n)
    return out


def ual_format_like_official(m: Dict[str, Any]) -> Dict[str, str]:
    """The values the official script prints, formatted as it prints them (:.2f)."""
    keys = ([f"r1_iou_{x}" for x in UAL_IOU_THRESHOLDS] + ["miou"]
            + [f"r1_td_{t}" for t in UAL_TD_THRESHOLDS] + ["mtd", "onset_acc_1s", "onset_acc_0.25s"])
    return {k: f"{m[k]:.2f}" for k in keys}


def ual_run_official_evaluator(repo: str, results_file: str) -> Dict[str, str]:
    """Runs the unmodified official script (python -I) and parses its stdout."""
    import subprocess
    script = os.path.join(repo, "src", "eval", "eval_your_results.py")
    require(os.path.isfile(script), f"official evaluator not found at {script}")
    p = subprocess.run([sys.executable, "-I", script, "--results_file", results_file],
                       capture_output=True, text=True, timeout=1800)
    require(p.returncode == 0, f"official evaluator failed ({p.returncode}): {p.stderr[-2000:]}")
    txt, out = p.stdout, {}
    for thr, r1, miou in re.findall(r"IoU = ([\d.]+) R@1: ([\d.]+); mIoU: ([\d.]+)", txt):
        out[f"r1_iou_{float(thr):g}"] = r1
        out["miou"] = miou
    for thr, r1, mtd in re.findall(r"Threshold m = (\d+)s R@1: ([\d.]+) mean abs distances: "
                                   r"([\d.]+)", txt):
        out[f"r1_td_{int(thr)}"] = r1
        out["mtd"] = mtd
    m1 = re.search(r"Accuracy within 1 sec: ([\d.]+)", txt)
    m2 = re.search(r"Accuracy within 0.25 sec: ([\d.]+)", txt)
    require(m1 is not None and m2 is not None, "could not parse the official evaluator output")
    out["onset_acc_1s"], out["onset_acc_0.25s"] = m1.group(1), m2.group(1)
    return out


def ual_build_official_predictions(ds: UALDataset, raw_items: Sequence[Dict]) -> List[Dict]:
    """The evaluator-ready file (output_test_template.json schema). This is the
    only place where GT and predictions meet."""
    out = []
    for r in raw_items:
        g = ds.gt[r["item_id"]]
        out.append({"video_id": r["item_id"], "start_time": g.start_time, "end_time": g.end_time,
                    "pred_start": r["pred_start"], "pred_end": r["pred_end"]})
    return out


# -----------------------------------------------------------------------------
# Uncertainty (not part of the official protocol; reported separately)
# -----------------------------------------------------------------------------

def _ual_cluster_index(clusters: Sequence[str]) -> List[np.ndarray]:
    g: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        g[c].append(i)
    return [np.asarray(v, int) for v in g.values()]


def ual_cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], B: int,
                             seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of a mean, resampling VIDEOS (SSBD items of one video are
    not independent). With 58-1589 clusters the percentile interval is adequate;
    the small-G correction of cluster_bootstrap_auc is not needed here."""
    v = np.asarray(values, float)
    units = _ual_cluster_index(clusters)
    if not units:
        return float("nan"), float("nan")
    sums = np.array([v[u].sum() for u in units])
    sizes = np.array([u.size for u in units], float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, B, 500):
        d = rng.integers(0, len(units), size=(min(500, B - len(means)), len(units)))
        means.extend((sums[d].sum(1) / sizes[d].sum(1)).tolist())
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def ual_paired_signflip(a: Sequence[float], b: Sequence[float], clusters: Sequence[str],
                        B: int, seed: int) -> Dict[str, Any]:
    """Randomisation test of H0: the two backbones' per-item values are
    exchangeable within every video. Statistic: mean difference; each video's
    differences flip sign jointly. Exact when 2^G <= B, else Monte Carlo with
    the (1 + k) / (1 + B) correction."""
    d = np.asarray(a, float) - np.asarray(b, float)
    units = _ual_cluster_index(clusters)
    D = np.array([d[u].sum() for u in units])
    n, G = d.size, len(units)
    obs = float(d.mean()) if n else 0.0
    tol = 1e-12
    if G <= 20 and 2 ** G <= B:
        import itertools
        st = [float(np.dot(s, D)) / n for s in itertools.product((-1.0, 1.0), repeat=G)]
        return {"delta": obs, "p": float(np.mean([abs(x) >= abs(obs) - tol for x in st])),
                "n_clusters": G, "exact": True}
    rng = np.random.default_rng(seed)
    k, done = 0, 0
    while done < B:
        m = min(1000, B - done)
        s = rng.choice((-1.0, 1.0), size=(m, G))
        k += int(np.sum(np.abs(s @ D / n) >= abs(obs) - tol))
        done += m
    return {"delta": obs, "p": float((1 + k) / (1 + B)), "n_clusters": G, "exact": False}


# -----------------------------------------------------------------------------
# Evaluation of runs and the summary tables
# -----------------------------------------------------------------------------

def ual_evaluate_run(us: UALSettings, ds: UALDataset, raw_path: str) -> Dict[str, Any]:
    raw = read_json(raw_path)
    require(raw.get("complete", False), f"{raw_path} is an unfinished run")
    require(raw.get("inputs_fingerprint") == ds.inputs_fingerprint,
            f"{raw_path} was produced from a different annotation version "
            f"({raw.get('inputs_fingerprint')} vs {ds.inputs_fingerprint})")
    items = raw["items"]
    ids = [r["item_id"] for r in items]
    require(len(set(ids)) == len(ids) and set(ids) <= set(ds.gt),
            f"{raw_path}: item ids are duplicated or unknown to the annotation file")
    preds = ual_build_official_predictions(ds, items)
    metrics = ual_official_metrics(preds)
    run_dir = os.path.dirname(raw_path)
    pred_path = os.path.join(run_dir, "predictions_official.json")
    atomic_write_json(pred_path, preds)
    parity = None
    if us.official_repo:
        off = ual_run_official_evaluator(us.official_repo, pred_path)
        mine = ual_format_like_official(metrics)
        diff = {k: (mine[k], off.get(k)) for k in mine if mine[k] != off.get(k)}
        require(not diff, f"port disagrees with the official evaluator on {pred_path}: {diff}")
        parity = {"official_script": os.path.join(us.official_repo, "src/eval/eval_your_results.py"),
                  "identical_to_2_decimals": True}
    status = defaultdict(int)
    for r in items:
        status[r["status"]] += 1
    calls = sum(int(r.get("calls", 0)) for r in items)
    trunc = sum(int(r.get("logprob_truncations", 0)) for r in items)
    beyond = [ds.gt[r["item_id"]].end_time - r["decoded_duration"] for r in items
              if r.get("decoded_duration") is not None
              and ds.gt[r["item_id"]].end_time > r["decoded_duration"] + 1e-6]
    diag = {"status_counts": dict(status), "calls_per_item": calls / max(1, len(items)),
            "logprob_truncation_rate": trunc / max(1, calls),
            "monotonicity_violations": sum(int(r.get("monotonicity_violations", 0)) for r in items),
            "frame_conflicts": sum(int(r.get("frame_conflicts", 0)) for r in items),
            "gt_end_beyond_decoded_duration": {"n": len(beyond),
                                               "max_excess_s": max(beyond) if beyond else 0.0}}
    if ds.subset == "uag_ssbd" and beyond and max(beyond) > 1.0:
        LOG.warning("UAG-SSBD: %d GT intervals end more than 1 s after the decoded video (max "
                    "%.1f s); check --ual_ssbd_time_format (%s)", len(beyond), max(beyond),
                    us.ssbd_time_format)
    res = {"subset": ds.subset, "backbone": raw["backbone"], "model": raw["model"],
           "mode": raw["mode"], "partial": raw["partial"], "limit": raw["limit"],
           "official_metrics": metrics, "official_script_parity": parity,
           "diagnostics_not_official": diag, "raw_outputs": raw_path,
           "predictions_official": pred_path}
    atomic_write_json(os.path.join(run_dir, "metrics.json"), res)
    res["_items"] = items
    res["_preds"] = preds
    return res


def _ual_fmt(x: Optional[float], nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ual_selection_tag(subsets: Sequence[str], backbones: Sequence[str]) -> str:
    st = "all" if set(subsets) == set(UAL_SUBSETS) else "+".join(subsets)
    bt = "all" if set(backbones) == set(UAL_BACKBONES) else "+".join(backbones)
    return f"{st}__{bt}"


_UAL_ECHO = [True]


def ual_write_summary(cfg: Config, us: UALSettings, datasets: Dict[str, UALDataset],
                      results: Dict[str, Dict[str, Dict[str, Any]]],
                      backbones: Sequence[str]) -> str:
    """Markdown / CSV / JSON summary: one official table per subset with the
    exact required header, then supplementary official metrics, then clearly
    labelled non-official diagnostics. No aggregation across subsets. Written
    to summary/<mode>/<full|partial_limitN>/<subsets>__<backbones>/."""
    sp = os.path.join(ual_paths(us)["summary"], ual_selection_tag(us.subsets, backbones))
    os.makedirs(sp, exist_ok=True)
    L: List[str] = []
    status = (f"PARTIAL RUN (--limit {us.limit}): the first {us.limit} items of each subset; NOT "
              f"benchmark results" if us.partial else "FULL RUN (all evaluation items)")
    L += ["# HOVER v10 on UAL-Bench: Mode A (query-conditioned HOVER localisation core)", "",
          f"**Status: {status}**", "",
          "- Mode A: the official query is the HOVER hypothesis; HOVER V (contextually calibrated "
          "presence profile) and S (exact segmentation, K = 1) with masked-frame boundary "
          "refinement. The open-ended H stage and the U selector are not used, so this is not an "
          "evaluation of the full H -> V -> S -> U pipeline.",
          f"- Selection rule fixed before any UAL result: tau = {us.seg_tau:g}, lambda = "
          f"{us.seg_penalty:g}, refine = {us.refine_mode}"
          + ("" if us.predeclared() else " **(deviates from the predeclared tau = 0, lambda = 1e9, "
                                         "refine = frame)**") + "; nothing was calibrated on UAL-Bench.",
          "- Metrics: exact port of the official evaluator (Hasnat79/UAL_Bench "
          "src/eval/eval_your_results.py)" + (", cross-checked against the unmodified official "
                                              "script on every predictions file"
                                              if us.official_repo else "") + ".",
          "- R@1 values and onset accuracies are percentages (100 x correct / N). mIoU is a "
          "fraction in [0, 1]. mTD is in seconds; TD = |start error| + |end error|; a null "
          "prediction counts as IoU 0 and TD = gt end_time (official convention).",
          "- Results are not aggregated across subsets (different domains, sizes and query "
          "types). Published UAL-Bench numbers are not included here; compare only with the "
          "same subset, metric and protocol, and label them as published.", ""]
    rows_csv: List[Dict[str, Any]] = []
    out_json: Dict[str, Any] = {"status": status, "partial": us.partial, "limit": us.limit,
                                "mode": us.mode, "selection_rule": {
                                    "seg_tau": us.seg_tau, "seg_penalty": us.seg_penalty,
                                    "refine_mode": us.refine_mode,
                                    "predeclared": us.predeclared()}, "subsets": {}}
    for s in us.subsets:
        res = results.get(s, {})
        ds = datasets[s]
        L += [f"## {UAL_DISPLAY[s]}", ""]
        if not res:
            L += ["No completed run for this subset.", ""]
            continue
        # Align backbones by item id in official order; compare on the intersection
        # only if the item sets differ (and say so).
        by = {k: {r["item_id"]: (r, p) for r, p in zip(v["_items"], v["_preds"])}
              for k, v in res.items()}
        common = [q.item_id for q in ds.queries if all(q.item_id in by[k] for k in by)]
        same = all(len(by[k]) == len(common) for k in by)
        n_videos = len({by[next(iter(by))][i][0]["video_file"] for i in common})
        L += [f"N = {len(common)} evaluation items over {n_videos} videos"
              + ("" if same else " (INTERSECTION; the backbones were run on different item "
                                 "sets: " + ", ".join(f"{k}: {len(v)}" for k, v in by.items())
                 + ")") + ".", ""]
        if s == "uag_ssbd":
            L += [f"SSBD 'time' read as {us.ssbd_time_format.upper()} (see UAL_BENCH_NOTES.md).", ""]
        if s == "uag_funqa":
            L += ["Every UAG-FunQA query is the same instruction (\"Identify the video's funny "
                  "moment.\"), so conditioning on it carries no item-specific information.", ""]
        sub: Dict[str, Any] = {}
        for k, v in res.items():
            items = [by[k][i][0] for i in common]
            preds = [by[k][i][1] for i in common]
            sub[k] = {"m": ual_official_metrics(preds), "p": ual_per_item(preds), "items": items,
                      "clusters": [r["video_file"] for r in items],
                      "diag": v["diagnostics_not_official"]}
        L += [UAL_MAIN_HEADER, " | ".join(["---"] * 9)]
        for k in UAL_BACKBONES:
            if k in sub:
                m = sub[k]["m"]
                L.append(" | ".join([UAL_BACKBONES[k][0]] + [_ual_fmt(m[f"r1_iou_{x}"]) for x in
                                                            UAL_IOU_THRESHOLDS]
                                    + [_ual_fmt(m[f"r1_td_{t}"]) for t in (0, 1, 3, 5)]
                                    + [_ual_fmt(m["mtd"])]))
        L += ["", "Supplementary official metrics (outputs of the official evaluator that are not "
                  "in the main table):", "",
              "Model | mIoU | R@1,TD≤7 | Onset acc. ≤1 s | Onset acc. ≤0.25 s",
              "--- | --- | --- | --- | ---"]
        for k in UAL_BACKBONES:
            if k in sub:
                m = sub[k]["m"]
                L.append(f"{UAL_BACKBONES[k][0]} | {_ual_fmt(m['miou'], 3)} | "
                         f"{_ual_fmt(m['r1_td_7'])} | {_ual_fmt(m['onset_acc_1s'])} | "
                         f"{_ual_fmt(m['onset_acc_0.25s'])}")
        L += ["", "Diagnostics (NOT official metrics; 95% CIs by video-cluster bootstrap, "
                  f"B = {cfg.n_bootstrap}):", "",
              "Model | null predictions | Yes/No outside top-logprobs (share of calls) | VLM "
              "calls/item | 95% CI mIoU | 95% CI R@1,0.5 | 95% CI mTD",
              "--- | --- | --- | --- | --- | --- | ---"]
        for k in UAL_BACKBONES:
            if k not in sub:
                continue
            p, cl, d = sub[k]["p"], sub[k]["clusters"], sub[k]["diag"]
            ci_i = ual_cluster_bootstrap_ci(p["iou"], cl, cfg.n_bootstrap, cfg.seed)
            ci_r = ual_cluster_bootstrap_ci([100.0 * (x >= 0.5) for x in p["iou"]], cl,
                                            cfg.n_bootstrap, cfg.seed + 1)
            ci_t = ual_cluster_bootstrap_ci(p["td_mean"], cl, cfg.n_bootstrap, cfg.seed + 2)
            sub[k]["ci"] = {"miou": ci_i, "r1_iou_0.5": ci_r, "mtd": ci_t}
            L.append(f"{UAL_BACKBONES[k][0]} | {sub[k]['m']['n_null_predictions']} | "
                     f"{100 * d['logprob_truncation_rate']:.1f}% | {d['calls_per_item']:.1f} | "
                     f"[{ci_i[0]:.3f}, {ci_i[1]:.3f}] | [{ci_r[0]:.2f}, {ci_r[1]:.2f}] | "
                     f"[{ci_t[0]:.2f}, {ci_t[1]:.2f}]")
        ref = next(iter(sub.values()))
        triv = [{"video_id": r["item_id"], "start_time": ds.gt[r["item_id"]].start_time,
                 "end_time": ds.gt[r["item_id"]].end_time,
                 "pred_start": 0.0 if r.get("decoded_duration") else None,
                 "pred_end": float(r["decoded_duration"]) if r.get("decoded_duration") else None}
                for r in ref["items"]]
        tm = ual_official_metrics(triv)
        L += ["", "Reference, NOT a model: the whole-video span [0, decoded duration] scored by "
                  "the same evaluator (any localiser should beat it):", "", UAL_MAIN_HEADER,
              " | ".join(["---"] * 9),
              " | ".join(["Whole video (no model)"] + [_ual_fmt(tm[f"r1_iou_{x}"]) for x in
                                                       UAL_IOU_THRESHOLDS]
                         + [_ual_fmt(tm[f"r1_td_{t}"]) for t in (0, 1, 3, 5)]
                         + [_ual_fmt(tm["mtd"])]), ""]
        keys = [k for k in UAL_BACKBONES if k in sub]
        pairs = [(x, y) for i, x in enumerate(keys) for y in keys[i + 1:]]
        tests: Dict[str, Any] = {}
        if pairs:
            for metric, field_ in (("miou", "iou"), ("mtd", "td_mean")):
                pv = {}
                for x, y in pairs:
                    t = ual_paired_signflip(sub[x]["p"][field_], sub[y]["p"][field_],
                                            sub[x]["clusters"], cfg.n_permutations, cfg.seed)
                    tests[f"{x} vs {y}|{metric}"] = t
                    pv[f"{x} vs {y}|{metric}"] = t["p"]
                for kk, hv in holm_bonferroni(pv).items():
                    tests[kk]["holm_reject_0.05"] = hv["reject"]
            L += ["Paired backbone comparisons (NOT official; video-cluster sign-flip "
                  f"randomisation test, B = {cfg.n_permutations}, Holm-corrected per metric):", "",
                  "Comparison | ΔmIoU | p | Holm reject | ΔmTD (s) | p | Holm reject",
                  "--- | --- | --- | --- | --- | --- | ---"]
            for x, y in pairs:
                a_, b_ = tests[f"{x} vs {y}|miou"], tests[f"{x} vs {y}|mtd"]
                L.append(f"{UAL_BACKBONES[x][0]} vs {UAL_BACKBONES[y][0]} | {a_['delta']:+.3f} | "
                         f"{a_['p']:.4f} | {a_['holm_reject_0.05']} | {b_['delta']:+.2f} | "
                         f"{b_['p']:.4f} | {b_['holm_reject_0.05']}")
            L.append("")
        out_json["subsets"][s] = {
            "n_items": len(common), "same_item_set_for_all_backbones": same,
            "item_counts": {k: len(v) for k, v in by.items()},
            "official": {k: v["m"] for k, v in sub.items()},
            "diagnostics_not_official": {k: {**v["diag"], "ci95": v.get("ci")} for k, v in
                                         sub.items()},
            "whole_video_reference": tm, "paired_tests": tests,
            "annotation_sha256": ds.annotation_sha256, "inputs_fingerprint": ds.inputs_fingerprint}
        for k, v in sub.items():
            rows_csv.append({"subset": s, "model": UAL_BACKBONES[k][0], "hf_id": UAL_BACKBONES[k][1],
                             "partial": us.partial, **{kk: v["m"][kk] for kk in v["m"]}})
    md = os.path.join(sp, "results.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    atomic_write_json(os.path.join(sp, "results.json"), out_json)
    if rows_csv:
        import csv
        cols = list(rows_csv[0])
        with open(os.path.join(sp, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_csv)
    if _UAL_ECHO[0]:
        print("\n".join(L))
    LOG.info("wrote %s (+ results.csv, results.json)", md)
    return md


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def ual_header_duration(path: str) -> Optional[float]:
    """frame count / fps from the container header: the duration formula of
    extract_frames_official, without decoding."""
    if cv2 is None or not path:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(n / fps) if fps and fps > 0 else None


def ual_banner(cfg: Config, us: UALSettings, backbones: Sequence[str],
               datasets: Dict[str, UALDataset], items: Dict[str, List[UALQuery]], phase: str
               ) -> None:
    LOG.info("=" * 96)
    LOG.info("HOVER v%s | UAL-Bench | %s | mode=%s (%s)", VERSION, phase, us.mode, UAL_ADAPTER_VERSION)
    LOG.info("subsets: %s", ", ".join(f"{UAL_DISPLAY[s]} ({len(items[s])} of "
                                      f"{len(datasets[s].queries)} items)" for s in us.subsets))
    for s in us.subsets:
        LOG.info("  %s annotations: %s (sha256 %s)", UAL_DISPLAY[s], datasets[s].annotation_path,
                 datasets[s].annotation_sha256[:12])
        LOG.info("  %s video dirs: %s", UAL_DISPLAY[s], datasets[s].video_dirs)
    for k in backbones:
        LOG.info("backbone %s -> %s (backend %s%s)", k, UAL_BACKBONES[k][1], cfg.backend,
                 f", api {us.api_bases.get(k, cfg.api_base)}" if cfg.backend == "openai" else "")
    LOG.info("input: fps=%g window=%d frames max_pixels=%d prep=%s | selection: tau=%g "
             "lambda=%g refine=%s%s", cfg.fps, cfg.window_size, cfg.max_pixels,
             cfg.prep_fingerprint(), us.seg_tau, us.seg_penalty, us.refine_mode,
             "" if us.predeclared() else "  ** NOT the predeclared rule **")
    LOG.info("output: %s", ual_paths(us)["base"])
    if us.partial:
        LOG.info("PARTIAL RUN: --limit %d; outputs go to '%s' and are NOT benchmark results",
                 us.limit, us.run_tag)
    LOG.info("=" * 96)


def cmd_inspect_ual(cfg: Config, us: UALSettings) -> Dict[str, Any]:
    """Schema audit without any VLM: paths, hashes, counts, fields, GT statistics,
    media availability and the timestamp conventions (OOPS end = clip end, SSBD
    time format) checked against container durations when videos are present."""
    report: Dict[str, Any] = {}
    for s in us.subsets:
        ds = ual_load_dataset(us, s)
        data = read_json(ds.annotation_path)
        rows = list(data.values()) if isinstance(data, dict) else [
            (e[1] if s == "uag_ssbd" else e) for e in data]
        fields = sorted({k for r in rows for k in r})
        g = list(ds.gt.values())
        lens = [x.end_time - x.start_time for x in g]
        per_video = defaultdict(int)
        for q in ds.queries:
            per_video[q.video_file] += 1
        qcount = defaultdict(int)
        for q in ds.queries:
            qcount[q.query] += 1
        present = {vf: p for vf, p in ds.video_paths.items() if p}
        a: Dict[str, Any] = {
            "annotation_path": ds.annotation_path, "annotation_sha256": ds.annotation_sha256,
            "inputs_fingerprint": ds.inputs_fingerprint, "fields": fields,
            "n_records": len(rows), "n_items": len(ds.queries), "n_videos": len(per_video),
            "max_items_per_video": max(per_video.values()),
            "videos_with_multiple_items": sum(1 for v in per_video.values() if v > 1),
            "n_unique_queries": len(qcount),
            "most_common_queries": sorted(qcount.items(), key=lambda kv: -kv[1])[:3],
            "gt_start_range": [min(x.start_time for x in g), max(x.start_time for x in g)],
            "gt_end_range": [min(x.end_time for x in g), max(x.end_time for x in g)],
            "gt_length_mean_median_min_max": [float(np.mean(lens)), float(np.median(lens)),
                                              min(lens), max(lens)],
            "n_zero_length_gt": sum(1 for x in lens if x == 0),
            "video_dirs": ds.video_dirs, "n_videos_present": len(present),
            "n_videos_missing": len(per_video) - len(present),
            "missing_examples": sorted(set(per_video) - set(present))[:5],
            "example_items": [{"item_id": q.item_id, "query": q.query,
                               "start_time": ds.gt[q.item_id].start_time,
                               "end_time": ds.gt[q.item_id].end_time,
                               "source": ds.gt[q.item_id].source} for q in ds.queries[:2]]}
        dur = {vf: ual_header_duration(p) for vf, p in present.items()}
        if s == "uag_oops":
            a["end_time_multiple_of_0.125"] = sum(1 for x in g if abs(x.end_time * 8 -
                                                                       round(x.end_time * 8)) < 1e-9)
        if dur:
            ex = [(q.item_id, ds.gt[q.item_id].end_time, dur[q.video_file]) for q in ds.queries
                  if dur.get(q.video_file)]
            a["checked_against_video_durations"] = len(ex)
            a["gt_end_beyond_duration"] = sum(1 for _, e, d in ex if e > d + 1e-6)
            a["gt_end_within_0.25s_of_duration"] = sum(1 for _, e, d in ex if abs(e - d) <= 0.25)
            a["max_gt_end_minus_duration_s"] = max(e - d for _, e, d in ex) if ex else None
            a["duration_examples"] = [{"item_id": i, "gt_end": e, "header_duration": d}
                                      for i, e, d in ex[:3]]
            if s == "uag_ssbd":
                alt = "seconds" if us.ssbd_time_format == "mmss" else "mmss"
                verdict = {}
                for fmt in (us.ssbd_time_format, alt):
                    beyond, bad = 0, 0
                    for q in ds.queries:
                        d = dur.get(q.video_file)
                        if not d:
                            continue
                        try:
                            _, e = ual_parse_ssbd_time(ds.gt[q.item_id].source, fmt)
                        except RuntimeError:
                            bad += 1
                            continue
                        beyond += int(e > d + 1.0)
                    verdict[fmt] = {"intervals_ending_>1s_after_video": beyond,
                                    "unparseable": bad}
                a["ssbd_time_format_check"] = verdict
        report[s] = a
        print(f"\n== {UAL_DISPLAY[s]} schema audit ==")
        print(json.dumps(_jsonable(a), indent=2))
    out = os.path.join(ual_paths(us)["inspect"], "schema_audit.json")
    atomic_write_json(out, report)
    LOG.info("wrote %s", out)
    return report


def cmd_run_ual(cfg: Config, us: UALSettings, backbones: Sequence[str],
                oracle: Optional[Callable] = None, model_override: str = "") -> str:
    """Load and validate every subset, check every backbone, then infer and
    evaluate. Nothing expensive starts before all checks pass."""
    require(us.mode == "query_conditioned", MODE_B_MESSAGE)
    require(cfg.use_segmentation and cfg.use_hypothesis_scoring,
            "UAL Mode A needs HOVER S and V (use_segmentation, use_hypothesis_scoring)")
    if cfg.use_utility_selection:
        LOG.info("U (utility selection) is bypassed in Mode A: one interval per item, and the "
                 "VideoGlitchBench-fitted selector is not valid for UAL-Bench")
    datasets = {s: ual_load_dataset(us, s) for s in us.subsets}
    items = {s: ual_select_items(ds, us.limit) for s, ds in datasets.items()}
    for s, ds in datasets.items():
        ual_check_media(ds, items[s], us.allow_missing_videos)
    bcfg = {k: ual_backbone_config(cfg, k, us, model_override) for k in backbones}
    pre = {k: ual_preflight_backbone(c) for k, c in bcfg.items()}
    ual_banner(cfg, us, backbones, datasets, items, "inference + official evaluation")
    memo: Dict = {}
    runs: Dict[str, Dict[str, str]] = defaultdict(dict)
    for k in backbones:
        c, vlm = bcfg[k], None
        for s in us.subsets:
            ds = datasets[s]
            ns, cache = ual_cache_namespace(c, ds, us), ual_paths(us, s, k)["cache"]
            if vlm is None:
                vlm = make_ual_vlm(c, cache, ns, oracle)
            else:
                _ual_switch_subset(vlm, cache, ns)
            smoke = None
            first = next((q for q in items[s] if ds.video_paths.get(q.video_file)), None)
            if first is not None:
                try:
                    w0, _ = ual_prepare_video(c, us, ds, first.video_file, memo)
                except Exception as e:
                    LOG.warning("smoke test skipped: %s does not decode (%s)", first.video_file, e)
                else:
                    smoke = ual_smoke_test(vlm, c, ds, w0, first)
                    LOG.info("smoke test %s on %s: %s", k, UAL_DISPLAY[s], smoke)
            runs[s][k] = ual_run_backbone(c, us, ds, items[s], vlm, memo, pre[k], smoke)
        del vlm
        if c.backend == "hf":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    results = {s: {k: ual_evaluate_run(us, datasets[s], p) for k, p in runs[s].items()}
               for s in us.subsets}
    return ual_write_summary(cfg, us, datasets, results, backbones)


def cmd_evaluate_ual(cfg: Config, us: UALSettings, backbones: Sequence[str]) -> str:
    """Re-evaluate saved runs (no model calls) and rebuild the summary."""
    datasets = {s: ual_load_dataset(us, s) for s in us.subsets}
    items = {s: ual_select_items(ds, us.limit) for s, ds in datasets.items()}
    ual_banner(cfg, us, backbones, datasets, items, "evaluation only (no model calls)")
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for s in us.subsets:
        results[s] = {}
        for k in backbones:
            p = os.path.join(ual_paths(us, s, k)["run"], "raw_outputs.json")
            if os.path.exists(p):
                results[s][k] = ual_evaluate_run(us, datasets[s], p)
            else:
                LOG.warning("no %s run for %s / %s at %s", us.run_tag, UAL_DISPLAY[s], k, p)
    require(any(results.values()), "no completed UAL-Bench run found; run 'run_ual' first")
    return ual_write_summary(cfg, us, datasets, results,
                             [k for k in backbones if any(k in r for r in results.values())])


def cmd_evaluate_ual_file(path: str, official_repo: str = "") -> Dict[str, Any]:
    """Official metrics of any file in the output_test_template.json schema."""
    data = read_json(path)
    require(isinstance(data, list) and all(isinstance(r, dict) and set(UAL_OFFICIAL_KEYS) <= set(r)
                                           for r in data),
            f"{path} is not in the official schema (a list of {list(UAL_OFFICIAL_KEYS)})")
    m = ual_official_metrics(data)
    print(json.dumps({"file": os.path.abspath(path), "official_metrics": m}, indent=2))
    if official_repo:
        off, mine = ual_run_official_evaluator(official_repo, path), ual_format_like_official(m)
        diff = {k: (mine[k], off.get(k)) for k in mine if mine[k] != off.get(k)}
        require(not diff, f"port disagrees with the official evaluator: {diff}")
        print("identical to the official evaluator (to its printed 2 decimals)")
    return m


def _parse_api_bases(txt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in [p.strip() for p in (txt or "").split(",") if p.strip()]:
        require("=" in part, f"--ual_api_bases entry {part!r} is not backbone=url")
        k, v = part.split("=", 1)
        require(k.strip() in UAL_BACKBONES, f"--ual_api_bases: unknown backbone {k.strip()!r}")
        out[k.strip()] = v.strip()
    return out


def ual_settings_from_args(a: argparse.Namespace) -> UALSettings:
    require(bool(a.ual_dataset), "pass --ual_dataset {uag_oops,uag_ssbd,uag_funqa,all}")
    subsets = UAL_SUBSETS if a.ual_dataset == "all" else (a.ual_dataset,)
    require(not (a.ual_annotations and len(subsets) > 1),
            "--ual_annotations names one file; with --ual_dataset all use "
            "--ual_oops_annotations / --ual_ssbd_annotations / --ual_funqa_annotations")
    us = UALSettings(
        subsets=tuple(subsets), root=a.ual_root or "", annotations=a.ual_annotations or "",
        video_root=a.ual_video_root or "",
        subset_annotations={s: getattr(a, f"ual_{_ual_short(s)}_annotations") for s in UAL_SUBSETS
                            if getattr(a, f"ual_{_ual_short(s)}_annotations")},
        subset_videos={s: getattr(a, f"ual_{_ual_short(s)}_videos") for s in UAL_SUBSETS
                       if getattr(a, f"ual_{_ual_short(s)}_videos")},
        work_root=a.ual_work_root or "", mode=a.ual_mode, seg_tau=float(a.ual_seg_tau),
        seg_penalty=float(a.ual_seg_penalty), refine_mode=a.ual_refine_mode,
        ssbd_time_format=a.ual_ssbd_time_format, allow_missing_videos=bool(a.ual_allow_missing_videos),
        api_bases=_parse_api_bases(a.ual_api_bases), limit=a.limit,
        official_repo=a.ual_repo or "")
    require(us.mode in UAL_MODES, f"--ual_mode must be one of {UAL_MODES}")
    require(us.refine_mode in UAL_REFINE_MODES, f"--ual_refine_mode must be one of {UAL_REFINE_MODES}")
    require(math.isfinite(us.seg_tau) and us.seg_penalty >= 0, "invalid --ual_seg_tau/penalty")
    require(us.limit is None or us.limit > 0, "--limit must be positive")
    return us


def _legacy_dataset_guard(a: argparse.Namespace) -> None:
    """VideoGlitchBench commands stay VideoGlitchBench-only."""
    require((a.dataset or "videoglitchbench") == "videoglitchbench",
            f"'{a.command}' is a VideoGlitchBench command; for UAL-Bench use inspect_ual, run_ual "
            f"or evaluate_ual (got --dataset {a.dataset})")


def main_ual(a: argparse.Namespace) -> None:
    require(a.dataset in (None, "ual_bench"),
            f"'{a.command}' is a UAL-Bench command; --dataset must be ual_bench (got {a.dataset})")
    if a.command == "selftest_ual":
        selftest_ual(a.ual_repo or "")
        return
    require(a.ual_mode == "query_conditioned", MODE_B_MESSAGE)
    require(a.method == "HOVER", "UAL-Bench evaluates HOVER only; GliDe and vanilla are not run on "
                                "UAL-Bench (explicit guard)")
    if a.command == "evaluate_ual" and a.ual_predictions:
        cmd_evaluate_ual_file(a.ual_predictions, a.ual_repo or "")
        return
    bb = a.backbone
    a.backbone = None
    try:
        cfg = cfg_from_args(a)
    finally:
        a.backbone = bb
    us = ual_settings_from_args(a)
    us.work_root = ual_resolve_work_root(us, cfg)
    setup_logging(ual_paths(us)["logs"], a.command)
    set_seed(cfg.seed)
    if a.command == "inspect_ual":
        cmd_inspect_ual(cfg, us)
    elif a.command == "run_ual":
        cmd_run_ual(cfg, us, ual_resolve_backbones(bb), model_override=a.model or "")
    elif a.command == "evaluate_ual":
        cmd_evaluate_ual(cfg, us, ual_resolve_backbones(bb, default_all=True))


# -----------------------------------------------------------------------------
# Self-test (no model downloads, no GPU)
# -----------------------------------------------------------------------------

# Verbatim records of the official paper-version files (see UAL_BENCH_NOTES.md).
_UAL_FIX_OOPS = {
    "34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0": {
        "start_time": 2.714329, "end_time": 3.75,
        "description": "A guy jumps onto a bed where his son is. When the guy jumps, the son flies "
                       "up and hits the wall."},
    "34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)10": {
        "start_time": 3.53317, "end_time": 7.375,
        "description": "a little boy is running around with cover on head Boy trips over toys and "
                       "falls."}}
_UAL_FIX_SSBD = [
    ["v_ArmFlapping_01", {"id": "b_01", "time": "0018:0024", "bodypart": "hand",
                          "category": "armflapping", "intensity": "high", "modality": "video",
                          "description": "a person is armflapping with high intensity"}],
    ["v_ArmFlapping_01", {"id": "b_02", "time": "0027:0032", "bodypart": "full",
                          "category": "headbanging", "intensity": "high", "modality": "video",
                          "description": "a person is headbanging with high intensity"}]]
_UAL_FIX_FUNQA = [{"instruction": "Identify the video's funny moment.",
                   "visual_input": "H_A_101_1433_1631.mp4", "output": "[0000,  0195]",
                   "task": "H1", "ID": "test_0", "start_time": 0.0, "end_time": 6.5}]
_UAL_FIX_INSTRUCT = [{"video": "25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July "
                               "2017)11.mp4",
                      "QA": [{"q": "Find the start time and end time of the query below from the "
                                   "video.\n          Query: man attempted to jump from trampoline "
                                   "into pool man tripped on trampoline and fell onto the ground",
                              "a": "start_time: 1.689396, end_time: 6.0"}]}]
# src/eval/output_test_template.json and the output printed in the official README.
_UAL_TEMPLATE = [
    {"video_id": "video_01", "start_time": 2.714329, "end_time": 3.75, "pred_start": 1.0, "pred_end": 1.5},
    {"video_id": "video_02", "start_time": 5.123456, "end_time": 6.789012, "pred_start": 4.0, "pred_end": 5.5},
    {"video_id": "video_03", "start_time": 1.234567, "end_time": 2.345678, "pred_start": 0.5, "pred_end": 1.0},
    {"video_id": "video_04", "start_time": 7.890123, "end_time": 9.012345, "pred_start": 6.5, "pred_end": 7.0},
    {"video_id": "video_05", "start_time": 3.456789, "end_time": 4.567890, "pred_start": 2.0, "pred_end": 3.0},
    {"video_id": "video_06", "start_time": 8.901234, "end_time": 10.123456, "pred_start": 7.5, "pred_end": 8.0}]
_UAL_TEMPLATE_README = {"r1_iou_0.3": "0.00", "r1_iou_0.5": "0.00", "r1_iou_0.7": "0.00",
                        "miou": "0.02", "r1_td_0": "0.00", "r1_td_1": "0.00", "r1_td_3": "33.33",
                        "r1_td_5": "100.00", "r1_td_7": "100.00", "mtd": "3.07",
                        "onset_acc_1s": "16.67", "onset_acc_0.25s": "0.00"}
# Synthetic video lengths: OOPS clips end slightly BEFORE their annotated end_time,
# as on the real '... (May 2017)0' clip (end_time 3.75 s, mp4 3.64 s).
_UAL_FIX_SECONDS = {"uag_oops": {"34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0": 3.6,
                                 "34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)10": 7.3},
                    "uag_ssbd": {"v_ArmFlapping_01": 34.0},
                    "uag_funqa": {"H_A_101_1433_1631": 6.6}}
# Frame-exact Mode A output of a perfect presence oracle on the fixtures (4 fps).
# The 7.3 s clip ends at 7.25 s, not 7.3 s: the sampler's last seek (t = 7.25 s,
# frame 72.5 of 73) reads nothing, so the last decoded frame is t = 7.0 s and its
# window ends at 7.0 + 1/fps. Predictions never extend past decoded frames.
_UAL_FIX_EXPECTED = {"34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0": (2.75, 3.6),
                     "34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)10": (3.75, 7.25),
                     "v_ArmFlapping_01_b_01": (18.0, 24.0), "v_ArmFlapping_01_b_02": (27.0, 32.0),
                     "H_A_101_1433_1631.mp4": (0.0, 6.5)}


def _ual_build_fixture(root: str) -> None:
    for s, data in (("uag_oops", _UAL_FIX_OOPS), ("uag_ssbd", _UAL_FIX_SSBD),
                    ("uag_funqa", _UAL_FIX_FUNQA)):
        atomic_write_json(os.path.join(root, "data", s, UAL_ANNOTATION_FILES[s]), data)
        for k, (stem, sec) in enumerate(_UAL_FIX_SECONDS[s].items()):
            _write_video(os.path.join(root, "data", s, "videos", stem + ".mp4"), sec, seed=11 + k)
    atomic_write_json(os.path.join(root, "data", "uag_oops_train_instruct",
                                   "uag_oops_train_instruct_dataset_v1.json"), _UAL_FIX_INSTRUCT)


def _ual_mock_oracle(frame_pos: Dict[Tuple[str, str, int], int], win_frames: Dict[str, Tuple],
                     log: List[Tuple[str, str]]):
    """A perfect presence oracle for the fixtures: Yes iff a kept frame lies in
    the queried GT interval. The oracle is the simulated world and knows GT; the
    method under test only sees its Yes/No log-probs."""
    def oracle(kind, system, user, images, options):
        log.append((system, user))
        require(kind == "score" and system == P_UAL_PRESENCE_SYS, f"unexpected call {system[:40]}")
        q = user.split("\n", 1)[0][len("Query: "):]
        img = images[0]
        if img == "__content_free__":
            lo = -1.0
        else:
            ap = os.path.abspath(img)
            m = _KEEP_RE.search(ap)
            d = os.path.dirname(ap)
            ids = (range(int(m.group(1)), int(m.group(2)) + 1) if m
                   else win_frames[re.sub(r"_(stitched|pair)\.jpg$", "_stitched.jpg", ap)])
            lo = 3.0 if any(frame_pos.get((q, d, f), 0) for f in ids) else -3.0
        return {"yes": lo / 2, "no": -lo / 2}
    return oracle


def _ual_snapshot(root: str) -> Dict[str, Tuple[int, int]]:
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def selftest_ual(official_repo: str = "", verbose: bool = True) -> None:
    import inspect as _inspect
    import shutil
    import tempfile
    fails, n = [], [0]

    def chk(name: str, cond: bool, detail: str = "") -> None:
        n[0] += 1
        if not cond:
            fails.append(f"{name}: {detail}")
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    def raises(name: str, fn: Callable, needle: str) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - the message is what is tested
            chk(name, needle.lower() in str(e).lower(), f"message lacks {needle!r}: {str(e)[:300]}")
            return
        chk(name, False, "no exception raised")

    if not LOG.handlers:
        logging.basicConfig(level=logging.WARNING)
    _UAL_ECHO[0] = False
    print("== UAL-Bench: selector, backbones, guards ==")
    ap = build_parser()
    a0 = ap.parse_args(["run"])
    chk("default dataset of legacy commands is VideoGlitchBench",
        a0.dataset is None and (_legacy_dataset_guard(a0) is None))
    raises("legacy command with --dataset ual_bench is refused",
           lambda: _legacy_dataset_guard(ap.parse_args(["run", "--dataset", "ual_bench"])),
           "VideoGlitchBench command")
    raises("UAL command with --dataset videoglitchbench is refused",
           lambda: main_ual(ap.parse_args(["run_ual", "--dataset", "videoglitchbench"])),
           "UAL-Bench command")
    c0 = Config()
    chk("VideoGlitchBench Config and paths are unchanged",
        c0.work_subdir == "HOVER_v8_work" and c0.paths()["runs"].endswith(
            os.path.join("HOVER_v8_work", "v10", "runs", "qwen2.5-vl-7b"))
        and c0.seg_penalty == 1.0 and c0.refine_mode == "frame" and VERSION == "10.0")
    a1 = ap.parse_args(["run_ual", "--ual_dataset", "uag_oops", "--backbone", "qwen2.5-vl-7b"])
    chk("Mode A (query_conditioned) is the default UAL mode", a1.ual_mode == "query_conditioned")
    raises("Mode B needs an explicit flag and is refused as not implemented",
           lambda: main_ual(ap.parse_args(["run_ual", "--ual_dataset", "uag_oops", "--backbone",
                                           "qwen2.5-vl-7b", "--ual_mode",
                                           "open_ended_full_pipeline"])), "Mode B")
    raises("vanilla / GliDe cannot be run on UAL-Bench",
           lambda: main_ual(ap.parse_args(["run_ual", "--ual_dataset", "uag_oops", "--backbone",
                                           "qwen2.5-vl-7b", "--method", "vanilla"])), "GliDe")
    chk("UAL backbones are exactly the three approved checkpoints",
        {k: v[1] for k, v in UAL_BACKBONES.items()} == {
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
            "ui-tars-1.5-7b": "ByteDance-Seed/UI-TARS-1.5-7B"})
    chk("UAL backbone ids agree with BACKBONES",
        all(BACKBONES[k]["hf"] == v[1] for k, v in UAL_BACKBONES.items()))
    chk("--backbone all expands to the three approved backbones",
        ual_resolve_backbones("all") == list(UAL_BACKBONES))
    raises("a non-approved backbone is refused", lambda: ual_resolve_backbones("qwen2.5-vl-3b"),
           "not part of the UAL-Bench experiment")
    raises("--model substitution is refused",
           lambda: ual_backbone_config(Config(), "qwen2.5-vl-7b", UALSettings(),
                                       "Qwen/Qwen2.5-VL-3B-Instruct"), "never substitutes")
    raises("hf backend refuses InternVL explicitly (no logits through remote code)",
           lambda: ual_preflight_backbone(Config(**{**asdict(Config().with_backbone(
               "internvl2.5-8b")), "backend": "hf"})), "InternVL2.5-8B with vLLM")
    raises("openai backend without a server fails with a remedy",
           lambda: ual_preflight_backbone(Config(**{**asdict(Config().with_backbone(
               "ui-tars-1.5-7b")), "api_base": "http://127.0.0.1:9/v1"})), "vllm serve")

    print("== UAL-Bench: timestamps and official metric port ==")
    chk("SSBD MMSS: '0105:0125' -> 65 s .. 85 s", ual_parse_ssbd_time("0105:0125", "mmss") == (65.0, 85.0))
    chk("SSBD MMSS: '0018:0024' -> 18 s .. 24 s", ual_parse_ssbd_time("0018:0024", "mmss") == (18.0, 24.0))
    chk("SSBD seconds reading is available as a flag",
        ual_parse_ssbd_time("0105:0125", "seconds") == (105.0, 125.0))
    raises("SSBD seconds field >= 60 is not MMSS", lambda: ual_parse_ssbd_time("0170:0180", "mmss"),
           "not MMSS")
    raises("malformed SSBD time is refused", lambda: ual_parse_ssbd_time("18-24", "mmss"), "dddd:dddd")
    tm = ual_official_metrics(_UAL_TEMPLATE)
    chk("port reproduces the official README output on output_test_template.json",
        ual_format_like_official(tm) == _UAL_TEMPLATE_README,
        str({k: (v, _UAL_TEMPLATE_README[k]) for k, v in ual_format_like_official(tm).items()
             if v != _UAL_TEMPLATE_README[k]}))
    h = ual_official_metrics([{"video_id": "a", "start_time": 2.0, "end_time": 4.0,
                               "pred_start": 3.0, "pred_end": 5.0},
                              {"video_id": "b", "start_time": 1.0, "end_time": 6.0,
                               "pred_start": None, "pred_end": None}])
    chk("hand case: IoU([2,4],[3,5]) = 1/3, null -> IoU 0 (mIoU 1/6)", abs(h["miou"] - 1 / 6) < 1e-12)
    chk("hand case: TD is the SUM of boundary errors (2), null -> TD = gt end (6); mTD = 4",
        abs(h["mtd"] - 4.0) < 1e-12 and h["r1_td_3"] == 50.0 and h["r1_td_5"] == 50.0
        and h["r1_td_7"] == 100.0, str(h))
    chk("hand case: recalls are percentages, onset within 1 s counted", h["r1_iou_0.3"] == 50.0
        and h["onset_acc_1s"] == 50.0 and h["onset_acc_0.25s"] == 0.0, str(h))
    big = ual_official_metrics([{"video_id": "z", "start_time": 0.0, "end_time": 3.0,
                                 "pred_start": 0.0, "pred_end": 20000.0}])
    chk("TD > 10000 enters mTD as gt end (official rule)", big["mtd"] == 3.0 and big["r1_td_7"] == 0.0)
    raises("string-valued predictions are refused (the official script would crash)",
           lambda: ual_official_metrics([{"video_id": "s", "start_time": 0.0, "end_time": 1.0,
                                          "pred_start": "0.0", "pred_end": "1.0"}]), "number")
    rng = np.random.default_rng(7)
    xa = rng.random(400)
    cl = [f"v{i // 2}" for i in range(400)]
    chk("sign-flip test: identical backbones give p = 1",
        ual_paired_signflip(xa, xa, cl, 2000, 0)["p"] == 1.0)
    chk("sign-flip test: a clear effect is detected",
        ual_paired_signflip(xa + 0.2, xa + rng.normal(0, 0.05, 400), cl, 2000, 0)["p"] < 0.01)
    lo_, hi_ = ual_cluster_bootstrap_ci(xa, cl, 1000, 0)
    chk("video-cluster bootstrap CI brackets the mean", lo_ < float(np.mean(xa)) < hi_)
    chk("GT never enters prompts: the user prompt is the query alone",
        ual_presence_user("q") == "Query: q\nIs the event described or requested by this query "
                                  "visibly occurring in this window? Answer with exactly one word: "
                                  "Yes or No.")
    sig = set(_inspect.signature(ual_infer_item).parameters)
    qa, _ = ual_parse_annotations("uag_ssbd", _UAL_FIX_SSBD, "f.json")
    qf, _ = ual_parse_annotations("uag_funqa", _UAL_FIX_FUNQA, "f.json")
    chk("model-visible metadata carries no timestamps (SSBD 'time', FunQA 'output')",
        all("time" not in dict(q.meta) for q in qa) and all("output" not in dict(q.meta) for q in qf))
    chk("inference functions receive no ground truth (UALQuery has no time fields)",
        not {f for f in UALQuery.__dataclass_fields__ if "time" in f or "start" in f or "end" in f}
        and "gt" not in sig and "ds" not in sig, str(sig))

    if cv2 is None:
        print("== UAL-Bench end-to-end: SKIPPED (opencv missing) ==")
    else:
        print("== UAL-Bench: real-schema fixtures, isolation, end-to-end Mode A ==")
        tmp = tempfile.mkdtemp(prefix="ual_selftest_")
        root = os.path.join(tmp, "UAL_Bench")
        _ual_build_fixture(root)
        vgb = os.path.join(tmp, "VideoGlitchBench")
        atomic_write_json(os.path.join(vgb, "annotations.json"), [{"video_name": "x.mp4"}])
        atomic_write_json(os.path.join(vgb, "HOVER_v8_work", "cache", "qwen2.5-vl-7b", "ab",
                                       "abc.json"), {"v": 1})
        atomic_write_json(os.path.join(vgb, "HOVER_v8_work", "v10", "runs", "qwen2.5-vl-7b",
                                       "HOVER_test.json"), {"outputs": []})
        work = os.path.join(tmp, "HOVER_v10_ual_work")
        base = Config(root=vgb, backend="mock", n_bootstrap=300, n_permutations=500)

        def mk(**kw) -> UALSettings:
            d = dict(subsets=UAL_SUBSETS, root=root, work_root=work)
            d.update(kw)
            return UALSettings(**d)
        for s in UAL_SUBSETS:
            chk(f"default mapping: {UAL_DISPLAY[s]} -> {UAL_ANNOTATION_FILES[s]}",
                os.path.basename(ual_find_annotation(mk(), s)) == UAL_ANNOTATION_FILES[s])
        raises("--ual_annotations = uag_oops_train_instruct_dataset_v1.json is rejected by name",
               lambda: ual_load_dataset(mk(subsets=("uag_oops",), annotations=os.path.join(
                   root, "data", "uag_oops_train_instruct",
                   "uag_oops_train_instruct_dataset_v1.json")), "uag_oops"),
               "oops_uag_paper_version.json")
        odd = os.path.join(tmp, "renamed_annotations.json")
        atomic_write_json(odd, _UAL_FIX_INSTRUCT)
        raises("instruction-format data under another name is rejected by schema",
               lambda: ual_load_dataset(mk(subsets=("uag_oops",), annotations=odd), "uag_oops"),
               "training/instruction-format")
        raises("another subset's file is refused by schema",
               lambda: ual_load_dataset(mk(subsets=("uag_oops",), annotations=os.path.join(
                   root, "data", "uag_ssbd", "ssbd_paper_version.json")), "uag_oops"),
               "paper-version schema")
        raises("missing --ual_root is an actionable error, never a VideoGlitchBench fallback",
               lambda: ual_load_dataset(UALSettings(subsets=("uag_oops",)), "uag_oops"),
               "--ual_root")
        raises("missing annotation file names every searched path",
               lambda: ual_load_dataset(mk(root=os.path.join(tmp, "nowhere")), "uag_ssbd"),
               "never used as a fallback")
        raises("UAL work root inside the VideoGlitchBench root is refused",
               lambda: ual_resolve_work_root(mk(work_root=os.path.join(vgb, "ual")), base),
               "VideoGlitchBench root")
        raises("UAL work root inside the UAL-Bench data is refused",
               lambda: ual_resolve_work_root(mk(work_root=os.path.join(root, "w")), base),
               "UAL-Bench data root")
        chk("default UAL work root is a sibling of --ual_root",
            ual_resolve_work_root(mk(work_root=""), base) == os.path.join(tmp, "HOVER_v10_ual_work"))

        dss = {s: ual_load_dataset(mk(), s) for s in UAL_SUBSETS}
        oo, ss, fq = dss["uag_oops"], dss["uag_ssbd"], dss["uag_funqa"]
        chk("UAG-OOPS fixture parses with official ids, query = description",
            [q.item_id for q in oo.queries] == list(_UAL_FIX_OOPS)
            and oo.queries[0].query == _UAL_FIX_OOPS[oo.queries[0].item_id]["description"])
        chk("UAG-OOPS end_time stored verbatim (duration-valued 3.75 s)",
            oo.gt["34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0"].end_time == 3.75)
        chk("UAG-SSBD keeps every behaviour as its own item with the official id",
            [q.item_id for q in ss.queries] == ["v_ArmFlapping_01_b_01", "v_ArmFlapping_01_b_02"]
            and len({q.video_file for q in ss.queries}) == 1)
        chk("UAG-SSBD query follows the official script",
            [q.query for q in ss.queries] == ["A person is armflapping with high intensity.",
                                              "A person is headbanging with high intensity."])
        chk("UAG-SSBD GT from MMSS", (ss.gt["v_ArmFlapping_01_b_02"].start_time,
                                      ss.gt["v_ArmFlapping_01_b_02"].end_time) == (27.0, 32.0))
        chk("UAG-FunQA id = visual_input, query = instruction",
            fq.queries[0].item_id == "H_A_101_1433_1631.mp4"
            and fq.queries[0].query == "Identify the video's funny moment.")
        chk("every fixture video resolves", all(all(v for v in d.video_paths.values())
                                                for d in dss.values()))
        g2 = json.loads(json.dumps(_UAL_FIX_OOPS))
        first = next(iter(g2))
        g2[first]["end_time"] = 99.0
        q2, _ = ual_parse_annotations("uag_oops", g2, "x.json")
        chk("cache namespace is GT-free (changing GT leaves the input fingerprint unchanged)",
            ual_inputs_fingerprint(q2) == oo.inputs_fingerprint
            and ual_cache_namespace(base, oo, mk()) == ual_cache_namespace(
                base, UALDataset(**{**oo.__dict__, "annotation_sha256": "other"}), mk()))
        vg_vlm = MockVLM(base, os.path.join(tmp, "k"), lambda *a_: {})
        ua_vlm = make_ual_vlm(base, os.path.join(tmp, "k2"), ual_cache_namespace(base, oo, mk()),
                              lambda *a_: {})
        ua_vlm2 = make_ual_vlm(base, os.path.join(tmp, "k3"), ual_cache_namespace(base, ss, mk()),
                               lambda *a_: {})
        kargs = ("score", P_PRESENCE_SYS, "u", [], {})
        chk("UAL cache keys cannot collide with VideoGlitchBench or across subsets",
            len({vg_vlm._key(*kargs), ua_vlm._key(*kargs), ua_vlm2._key(*kargs)}) == 3)
        chk("UAL caches live under <work>/ual_bench/<subset>/cache/<backbone>",
            ual_paths(mk(), "uag_oops", "qwen2.5-vl-7b")["cache"] == os.path.join(
                work, "ual_bench", "uag_oops", "cache", "qwen2.5-vl-7b")
            and not _ual_within(work, vgb))

        # frame-level presence labels of the simulated world
        frame_pos: Dict[Tuple[str, str, int], int] = {}
        win_frames: Dict[str, Tuple] = {}
        memo: Dict = {}
        for s, d in dss.items():
            for q in d.queries:
                wins, _ = ual_prepare_video(base, mk(), d, q.video_file, memo)
                g = d.gt[q.item_id]
                for w in wins:
                    sp = os.path.abspath(w.stitch_path)
                    win_frames[sp] = tuple(w.frame_ids)
                    for fid, ft in zip(w.frame_ids, w.frame_times):
                        frame_pos[(q.query, os.path.dirname(sp), fid)] = int(
                            g.start_time <= ft < g.end_time)
        chk("UAL windows are written under <work>/ual_bench/<subset>/windows",
            all(os.path.abspath(w.stitch_path).startswith(os.path.join(work, "ual_bench"))
                for v in memo.values() if not isinstance(v, Exception) for w in v[0]))
        prompts: List[Tuple[str, str]] = []
        oracle = _ual_mock_oracle(frame_pos, win_frames, prompts)
        snap_vgb, snap_data = _ual_snapshot(vgb), _ual_snapshot(root)
        g_ = globals()
        saved = (g_["load_records"], g_["load_splits"])

        def _boom(*_a, **_k):
            raise AssertionError("VideoGlitchBench loader called from a UAL command")
        g_["load_records"] = g_["load_splits"] = _boom
        try:
            md = cmd_run_ual(base, mk(), ["qwen2.5-vl-7b", "ui-tars-1.5-7b"], oracle=oracle)
            routed = True
        except AssertionError as e:
            routed, md = False, ""
            chk("UAL run never calls the VideoGlitchBench loaders", False, str(e))
        finally:
            g_["load_records"], g_["load_splits"] = saved
        if routed:
            chk("UAL run never calls the VideoGlitchBench loaders", True)
            chk("VideoGlitchBench tree is byte-for-byte untouched", _ual_snapshot(vgb) == snap_vgb)
            chk("UAL-Bench data tree is untouched", _ual_snapshot(root) == snap_data)
            new = set(_ual_snapshot(tmp)) - set(snap_vgb) - set(snap_data)
            chk("every file the UAL run wrote is inside the UAL work root",
                all(_ual_within(p, work) or p.startswith(os.path.join(tmp, "k"))
                    or p == odd for p in new), str(sorted(p for p in new if not _ual_within(p, work))[:3]))
            chk("prompts are exactly the UAL presence prompt with the official query",
                {p[0] for p in prompts} == {P_UAL_PRESENCE_SYS}
                and {p[1] for p in prompts} <= {ual_presence_user(q.query) for d in dss.values()
                                                for q in d.queries})
            gt_txt = {f"{v:g}" for d in dss.values() for g in d.gt.values()
                      for v in (g.start_time, g.end_time)} | {
                          str(v) for d in dss.values() for g in d.gt.values()
                          for v in (g.start_time, g.end_time)}
            chk("no ground-truth timestamp string appears in any prompt",
                not any(t in p[0] or t in p[1] for p in prompts for t in gt_txt if len(t) > 1))
            res = {}
            for s in UAL_SUBSETS:
                for k in ("qwen2.5-vl-7b", "ui-tars-1.5-7b"):
                    rd = ual_paths(mk(), s, k)["run"]
                    res[(s, k)] = (read_json(os.path.join(rd, "predictions_official.json")),
                                   read_json(os.path.join(rd, "raw_outputs.json")))
            preds = {r["video_id"]: r for s in UAL_SUBSETS
                     for r in res[(s, "qwen2.5-vl-7b")][0]}
            chk("official prediction schema: exactly video_id/start_time/end_time/pred_start/pred_end",
                all(set(r) == set(UAL_OFFICIAL_KEYS) for v in res.values() for r in v[0]))
            chk("Mode A recovers every fixture interval at frame resolution",
                all(abs(preds[i]["pred_start"] - a_) < 1e-6 and abs(preds[i]["pred_end"] - b_) < 1e-6
                    for i, (a_, b_) in _UAL_FIX_EXPECTED.items()),
                str({i: (preds[i]["pred_start"], preds[i]["pred_end"]) for i in _UAL_FIX_EXPECTED}))
            o0 = preds["34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0"]
            raw0 = res[("uag_oops", "qwen2.5-vl-7b")][1]["items"][0]
            chk("UAG-OOPS duration-valued end_time kept verbatim (3.75) while the prediction "
                "stays inside the decoded video", o0["end_time"] == 3.75
                and o0["pred_end"] <= raw0["decoded_duration"] + 1e-9 < 3.75)
            chk("UAG-SSBD behaviours of one video stay distinct items with distinct predictions",
                preds["v_ArmFlapping_01_b_01"]["pred_start"] != preds["v_ArmFlapping_01_b_02"][
                    "pred_start"] and len(res[("uag_ssbd", "qwen2.5-vl-7b")][0]) == 2)
            chk("raw outputs are GT-free",
                not any(k in r for v in res.values() for r in v[1]["items"]
                        for k in ("start_time", "end_time")))
            met = read_json(os.path.join(ual_paths(mk(), "uag_ssbd", "qwen2.5-vl-7b")["run"],
                                         "metrics.json"))["official_metrics"]
            chk("perfect oracle on SSBD: R@1,IoU>=0.7 = 100 and mTD = 0",
                met["r1_iou_0.7"] == 100.0 and met["mtd"] == 0.0, str(met))
            man = read_json(os.path.join(ual_paths(mk(), "uag_oops", "qwen2.5-vl-7b")["run"],
                                         "manifest.json"))
            chk("manifest records model, prompt, rule and data fingerprints",
                man["model"] == "Qwen/Qwen2.5-VL-7B-Instruct" and man["selection_rule"][
                    "predeclared"] and man["annotation_sha256"] == oo.annotation_sha256
                and not man["partial"])
            with open(md, encoding="utf-8") as f:
                mdl = f.read().splitlines()
            chk("summary uses the exact required main-table header", UAL_MAIN_HEADER in mdl)
            chk("summary has one row per backbone per subset",
                sum(1 for l_ in mdl if l_.startswith("Qwen2.5-VL-7B-Instruct | ")) >= 3
                and sum(1 for l_ in mdl if l_.startswith("UI-TARS-1.5-7B | ")) >= 3)
            js = read_json(os.path.join(os.path.dirname(md), "results.json"))
            chk("both backbones were evaluated on the same item ids",
                all(v["same_item_set_for_all_backbones"] for v in js["subsets"].values()))
            if official_repo:
                for s in UAL_SUBSETS:
                    pf = os.path.join(ual_paths(mk(), s, "qwen2.5-vl-7b")["run"],
                                      "predictions_official.json")
                    mine = ual_format_like_official(ual_official_metrics(read_json(pf)))
                    chk(f"port == unmodified official evaluator on the {UAL_DISPLAY[s]} run",
                        mine == ual_run_official_evaluator(official_repo, pf))
            # partial runs are labelled and isolated
            md_p = cmd_run_ual(base, mk(subsets=("uag_ssbd",), limit=1), ["qwen2.5-vl-7b"],
                               oracle=oracle)
            rp = os.path.join(ual_paths(mk(limit=1), "uag_ssbd", "qwen2.5-vl-7b")["run"],
                              "raw_outputs.json")
            with open(md_p, encoding="utf-8") as f:
                txt_p = f.read()
            chk("--limit marks a partial run in paths, outputs and the summary",
                os.path.exists(rp) and read_json(rp)["partial"] and "partial_limit1" in rp
                and "PARTIAL RUN" in txt_p)
            # missing media: actionable failure by default, null predictions on request
            vf = fq.video_paths["H_A_101_1433_1631.mp4"]
            shutil.move(vf, vf + ".away")
            try:
                raises("a missing video aborts before inference with a remedy",
                       lambda: cmd_run_ual(base, mk(subsets=("uag_funqa",)), ["qwen2.5-vl-7b"],
                                           oracle=oracle), "--ual_allow_missing_videos")
                cmd_run_ual(base, mk(subsets=("uag_funqa",), allow_missing_videos=True,
                                     work_root=os.path.join(tmp, "w_missing")),
                            ["qwen2.5-vl-7b"], oracle=oracle)
                pm = read_json(os.path.join(ual_paths(mk(work_root=os.path.join(tmp, "w_missing")),
                                                      "uag_funqa", "qwen2.5-vl-7b")["run"],
                                            "metrics.json"))["official_metrics"]
                chk("allowed missing videos are scored as official null predictions",
                    pm["n_null_predictions"] == 1 and pm["miou"] == 0.0 and pm["mtd"] == 6.5, str(pm))
            finally:
                shutil.move(vf + ".away", vf)
        shutil.rmtree(tmp, ignore_errors=True)

    if official_repo:
        print("== UAL-Bench: parity with the official evaluator script ==")
        td = tempfile.mkdtemp()
        tf = os.path.join(td, "template.json")
        atomic_write_json(tf, _UAL_TEMPLATE)
        off = ual_run_official_evaluator(official_repo, tf)
        chk("official script on the template prints the README numbers", off == _UAL_TEMPLATE_README,
            str(off))
        rng2 = np.random.default_rng(3)
        rows = []
        for i in range(300):
            gs = float(rng2.uniform(0, 30))
            ge = gs + float(rng2.uniform(0.1, 10))
            if i % 17 == 0:
                rows.append({"video_id": f"r{i}", "start_time": gs, "end_time": ge,
                             "pred_start": None, "pred_end": None})
                continue
            ps = float(rng2.uniform(0, 35))
            rows.append({"video_id": f"r{i}", "start_time": gs, "end_time": ge, "pred_start": ps,
                         "pred_end": ps + float(rng2.uniform(0.05, 12))})
        rf = os.path.join(td, "random.json")
        atomic_write_json(rf, rows)
        mine = ual_format_like_official(ual_official_metrics(rows))
        off = ual_run_official_evaluator(official_repo, rf)
        chk("port == official script on 300 random items incl. null predictions", mine == off,
            str({k: (mine[k], off.get(k)) for k in mine if mine[k] != off.get(k)}))
        shutil.rmtree(td, ignore_errors=True)
    else:
        print("== UAL-Bench parity with the official evaluator: SKIPPED (pass --ual_repo) ==")

    _UAL_ECHO[0] = True
    print(f"\n{n[0] - len(fails)}/{n[0]} UAL-Bench checks passed")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        raise SystemExit(1)


# =============================================================================
# CLI
# =============================================================================

COMMANDS = ("selftest", "prepare", "diagnose", "probe", "calibrate", "run", "import_glide",
            "export", "evaluate", "ablate") + UAL_COMMANDS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HOVER v10 (see module docstring)")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--calib_splits", default="dev", help="comma list, never 'test'")
    ap.add_argument("--method", default="HOVER", choices=("HOVER", "vanilla"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--run_files", nargs="*", default=[])
    ap.add_argument("--run_file", default="")
    ap.add_argument("--batch_report", default="")
    ap.add_argument("--allow_degenerate", action="store_true")
    skip = {"fusion", "limit", "fail_on_degenerate"}
    for f in Config.__dataclass_fields__.values():
        if f.name in skip:
            continue
        default = f.default
        if isinstance(default, bool):
            ap.add_argument(f"--{f.name}", default=None, action=argparse.BooleanOptionalAction)
        elif isinstance(default, (int, float, str)):
            ap.add_argument(f"--{f.name}", default=None, type=type(default))
    ap.add_argument("--limit", type=int, default=None)
    # Dataset selector. None means VideoGlitchBench for the legacy commands and
    # UAL-Bench for the *_ual commands; any other combination is refused.
    ap.add_argument("--dataset", default=None, choices=("videoglitchbench", "ual_bench"))
    g = ap.add_argument_group("UAL-Bench (inspect_ual, run_ual, evaluate_ual, selftest_ual)")
    g.add_argument("--ual_dataset", default=None, choices=UAL_SUBSETS + ("all",))
    g.add_argument("--ual_root", default=None,
                   help="UAL_Bench repository clone or dataset folder (annotations + videos)")
    g.add_argument("--ual_annotations", default=None,
                   help="paper-version annotation file of the ONE selected subset")
    g.add_argument("--ual_video_root", default=None)
    for s_ in UAL_SUBSETS:
        g.add_argument(f"--ual_{_ual_short(s_)}_annotations", default=None)
        g.add_argument(f"--ual_{_ual_short(s_)}_videos", default=None)
    g.add_argument("--ual_work_root", default=None,
                   help="default: <parent of --ual_root>/HOVER_v10_ual_work")
    g.add_argument("--ual_mode", default="query_conditioned", choices=UAL_MODES)
    g.add_argument("--ual_seg_tau", type=float, default=UAL_PREDECLARED["seg_tau"])
    g.add_argument("--ual_seg_penalty", type=float, default=UAL_PREDECLARED["seg_penalty"])
    g.add_argument("--ual_refine_mode", default=UAL_PREDECLARED["refine_mode"],
                   choices=UAL_REFINE_MODES)
    g.add_argument("--ual_ssbd_time_format", default="mmss", choices=("mmss", "seconds"))
    g.add_argument("--ual_allow_missing_videos", action="store_true")
    g.add_argument("--ual_api_bases", default="",
                   help="backbone=url,... (one OpenAI-compatible server per backbone)")
    g.add_argument("--ual_predictions", default=None,
                   help="evaluate_ual: score any file in the official prediction schema")
    g.add_argument("--ual_repo", default=None,
                   help="official UAL_Bench clone: cross-check against its evaluator script")
    return ap


def cfg_from_args(a: argparse.Namespace) -> Config:
    base = Config()
    if a.backbone:
        base = base.with_backbone(a.backbone)
    over = {k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__
            and v is not None and k != "backbone"}
    cfg = Config(**{**asdict(base), **over})
    cfg.fail_on_degenerate = not a.allow_degenerate
    return cfg


def main() -> None:
    a = build_parser().parse_args()
    if a.command in UAL_COMMANDS:
        main_ual(a)     # own logging, paths and caches; never touches VideoGlitchBench
        return
    _legacy_dataset_guard(a)
    if a.command == "selftest":
        selftest(a.glide_repo or "")
        return
    cfg = cfg_from_args(a)
    setup_logging(cfg.paths()["work"], a.command)
    set_seed(cfg.seed)
    require(cfg.calib_label in ("judge", "iou"), "--calib_label must be 'judge' or 'iou'")
    require(cfg.refine_mode in REFINE_MODES, f"--refine_mode must be one of {REFINE_MODES}")
    LOG.info("HOVER v%s | %s | backbone=%s backend=%s model=%s", VERSION, a.command,
             cfg.backbone, cfg.backend, cfg.model)
    if a.command == "export":
        require(a.run_file, "--run_file is required")
        cmd_export(a.run_file)
        return
    recs = load_records(cfg)
    splits = load_splits(cfg, recs)
    if a.command == "prepare":
        prepare_split(take_limit(splits[a.split], cfg.limit, cfg.stratify_limit), cfg)
        return
    if a.command == "import_glide":
        require(a.batch_report, "--batch_report is required")
        cmd_import_glide(cfg, splits, a.split, a.batch_report)
        return
    if a.command == "evaluate":
        require(a.run_files, "--run_files is required (first file = reference method)")
        cmd_evaluate(cfg, make_judge(cfg), splits, a.split, a.run_files)
        return
    vlm = make_vlm(cfg)
    if a.command in ("diagnose", "probe"):
        cmd_diagnose(cfg, vlm, splits, a.split, hypotheses=a.command == "probe")
    elif a.command == "calibrate":
        cmd_calibrate(cfg, vlm, splits, a.calib_splits, a.tag,
                      make_judge(cfg) if cfg.calib_label == "judge" else None)
    elif a.command == "run":
        cmd_run(cfg, vlm, splits, a.method, a.split, a.tag)
    elif a.command == "ablate":
        cmd_ablate(cfg, vlm, make_judge(cfg), splits, a.split, a.calib_splits)
    LOG.info("budget: %s", asdict(vlm.budget))


if __name__ == "__main__":
    main()
