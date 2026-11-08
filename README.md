# Report Once: A Simple Training-Free Baseline and an Audit of Open-Ended Video Game Glitch Detection

Code, results and paper sources for the CVPR 2027 submission. HOVER turns frozen vision-language
models (Qwen2.5-VL-3B/7B, InternVL2.5-4B/8B, UI-TARS-1.5-7B) into glitch detectors for
VideoGlitchBench by forming hypotheses across windows, verifying them, and emitting a hypothesis only
when its predicted contribution to F1×IoU exceeds a closed-form threshold. Its core (stage H with the
best-supported hypothesis and a whole-clip span) is the paper's training-free baseline.

## Paper

| File | Contents |
|---|---|
| `HOVER_CVPR2027_revised.zip` | Submission sources: `main.tex`, `supp.tex`, sections, tables, figures, the data every number is read from (`data/`), and the scripts that regenerate tables and figures. Includes `main.pdf` and `supp.pdf` built with pdfLaTeX. Its own `README.md` gives the build order and the list of changes of all revisions (the third adds the human check, the clean GliDe re-runs and the full benchmark; the pre-submission round adds Prop. 1 under Hungarian matching, multi-glitch, per-game and budget checks; the final round adds same-session timing, the video-grounded human check and the CVPR 2027 LLM Usage Statement). |
| `author-kit-CVPR2027-v1-latex (1).zip` | The unmodified CVPR 2027 LaTeX author kit. |

## Method and analysis code

| File | Contents |
|---|---|
| `HOVER.py` | The full pipeline: preprocessing, HOVER, the vanilla baseline, a converter for GliDe outputs, a port of the official VideoGlitchBench evaluator (Llama-3.1-8B-Instruct judge), development-split tuning, statistics, the UAL-Bench extension, and unit tests (`python HOVER.py selftest`: exactness of the segmentation program, the F\*/2 property, validity of the tests on synthetic data). |
| `hover_revision.py` | Every revision analysis, paired with the main run (paper Tab. 1): controls on GliDe's outputs, vanilla and fixed-sentence baselines, five-backbone ablations, emission rule × span resolution, re-judging, window probe, segmentation grid and decomposition. Imports `HOVER.py` from the same folder. |
| `medoid_candidates.py` | Scores the 40 short-listed development descriptions as fixed whole-clip predictions and adds the TF-IDF medoid to the fixed-sentence controls (Supp. Tab. S11). |
| `runs.json` | Paths of the main-run output files per backbone, the refine mode chosen on the development split, and the vLLM endpoints. Edit the paths for your machine. |
| `run_revision_main.sh` | Driver: for each backbone, starts the detector, runs controls, post-hoc, segmentation grid, probe and ablations, then the decomposition. |
| `start_detector.sh` | Starts the vLLM server (docker `vllm/vllm-openai:v0.8.5.post1`) for one backbone with the main-run flags. Set the GPU device and Hugging Face cache path for your machine. |
| `warm_gather.py`, `warm_emission.py` | Fill the VLM cache in parallel for the ablation variants and for the GliDe emission-rule control, so the sequential analysis passes read cached calls. They do not change any result. |
| `make_profile_examples_main.py` | Rebuilds `data/profile_examples_qwen7b.json` of the paper sources (verification profiles, Supp. Fig. S9) from the main run. |
| `internvl_chat_template.jinja` | Chat template passed to vLLM for the InternVL2.5 backbones (`start_detector.sh` mounts it as `/tpl`). |
| `REPRODUCE.md` | Step-by-step reproduction in the order of Supp. Sec. S9: unit tests, main pipeline, controls/ablations/re-judging/probe, tables and figures. |
| `revision2/` | Analyses of the second and third revisions (scripts, CSVs, human check); see below. |
| `requirements.txt` | Python package versions used (Python 3.12.9); models are served with vLLM 0.8.5. |

## Results

| File | Contents |
|---|---|
| `main_run_outputs.zip` | The main run of paper Tab. 1: per-clip predictions of HOVER (all refine modes and the four ablation variants), the vanilla baseline and the GliDe re-run (`runs/<backbone>/`), development-split tuning outputs (`params/<backbone>/calibration_<mode>.json`, probe), and evaluation reports (`eval/`). These are the files `runs.json` points to. |
| `revision_results.zip` | CSVs of all revision analyses on the main run (controls, ablation, post-hoc, re-judging, probe, repairs, segmentation grid, decomposition, medoid candidates). Identical to `data/revision/` in the paper zip. |
| `CVPR-HOVER-publication_results.zip` | Per-backbone main results and paired statistical comparisons for every refine mode (whole, window, frame), and the all-modes summary. |
| `all_backbones_timing.csv` | Inference time per clip of HOVER, the GliDe re-run and the vanilla baseline on 100 test clips for all five backbones (paper Tab. 7). |
| `videoglitchbench.zip` | Per-clip predictions and tuned parameters from the second complete run used to measure run-to-run variation (Supp. Tab. S10), including the ablation variants. HOVER's files there carry the prefix `crux_`. |
| `HOVER_UAL_Bench.zip` | Outputs of the transfer study on UAL-Bench (UAG-OOPS, UAG-FunQA, UAG-SSBD) for three backbones. |

## Second and third revisions (`revision2/`, `revision2_outputs.zip`)

The scripts import `HOVER.py` and `hover_revision.py` from the repository root and read `revision2/runs_v2.json`
(same format as `runs.json`; edit its paths for your machine). They write to `revision2/results/`.

| File | Contents |
|---|---|
| `v2_analyses.py` | Every method under the released evaluator and under rating × IoU matching, clip-level detection on glitch-free clips, short-glitch subsets, second judge (Supp. second-revision tables). |
| `h_only.py`, `score_v2.py` | Stage H alone (the core: H + top-1, whole clip), glitch-definition variants and new backbones (Qwen3-VL-8B); scoring with the main judge. |
| `score_judgefix.py`, `check_empty.py` | InternVL2.5-8B re-tuned without failed judge requests; finds scored pairs whose judge request failed. |
| `judge_agreement.py` | Agreement of the two judges (Llama-3.1-8B-Instruct, Qwen2.5-72B-Instruct-AWQ) on every scored pair. |
| `timing_v2.py`, `warm.py` | Re-timing of the scored configurations with per-clip logs; parallel VLM-cache warm-up. |
| `count_tracker.py`, `score_glide_clean.py` | GliDe's SAM 3 tracker outcomes; clean GliDe re-runs (no tracker out-of-memory) on Qwen2.5-VL-3B, InternVL2.5-8B and UI-TARS-1.5-7B (Supp. Tabs. S27-S28). |
| `v3_controls.py` | Shuffled-description and abstention controls for the core (Supp. Sec. S9.7). |
| `score_full.py` | Core and vanilla on all 5,238 clips of 120 games, and on the 96 games never used (Supp. Sec. S9.8). |
| `v4_reviewer.py` | Clips with one vs. several glitches, per-game differences against the GliDe re-run, glitches per clip on the whole benchmark; cached outputs and ratings only (Supp. Sec. S9.9, Tab. S36). |
| `v4_split_stats.py` | The 12 development, 12 test and 96 unused games compared on duration, glitch-free and multi-glitch shares, glitch coverage and genre (Supp. Tab. S34). |
| `v4_timing_ci.py`, `timing_stats.py` | Intervals and paired contrasts of the first timing session for all five backbones from the per-clip logs (`timing_stats.py` holds the statistics code of the timing study verbatim; it reproduces `report_test_100.md`). Reads the paper zip's `data/timing/` after extracting it in the repository root. |
| `paired_timing.sh`, `v4_timing_paired_stats.py` | Same-session timing (Supp. Tab. S23): vanilla, GliDe, the core and HOVER timed back to back on one vLLM server per backbone, 20 test clips covering all 12 games; statistics with the code of the first session. `results/v4_timing_paired_per_clip.csv` holds every per-clip time (`--per_clip` reproduces `v4_timing_paired_stats.csv` exactly); `results/paired_timing_glide/` the GliDe and vanilla timing logs with their outputs. |
| `make_video_sheet.py`, `analyze_video_ratings.py` | Video-grounded human check (Supp. Sec. S9.13, Tabs. S41-S42): 60 test clips (45 with, 15 without a glitch), all 150 reports of the core and the GliDe re-run scored 0-5 blind by three raters who watch the clip (each clip by one fixed pair); per-clip core - GliDe tests, rater agreement and both judges on the same reports. `human_check_video/` holds the key and the rubric; `results/v4_video_ratings_per_report.csv` every rating with anonymised raters (`--per_report` reproduces the statistics exactly). |
| `make_human_sheet.py`, `human_agreement.py`, `human_methods.py` | Blinded human check: 140 matched test pairs (35 per method), agreement with both judges, per-method ratings and tests (Supp. Sec. S9.12). |
| `human_check/` | The two returned annotation workbooks (`human_sheet_annotator_{A,B}.xlsx`, sheet `Annotate`), the key (method, backbone, judge ratings) and the instructions given to the annotators. |
| `results/` | All CSVs of the second and third revisions and the pre-submission round (`v2_*`, `v3_*`, `v4_*`); identical to `data/v2/` in the paper zip. |
| `revision2_outputs.zip` | Raw outputs behind them (extract in the repository root): `h_only/` (stage H per window, all backbones and definitions), `h_full/` (core and vanilla on the full benchmark), `glide_clean/` (pass `--out_dir revision2/glide_clean/data_<backbone>_sam3_clean` to `score_glide_clean.py`; per-clip reports, logs and imported run files of the clean GliDe re-runs). |

## Splits and caches

| File | Contents |
|---|---|
| `splits/` | The game-disjoint VideoGlitchBench splits used throughout: `train.json` (96 games), `dev.json` (445 clips, 12 games), `test.json` (383 clips, 12 games) and `game_assignments.json` (game -> split, derived from the three files). Place them in `<VideoGlitchBench root>/splits/`. |
| `caches/judge_cache.tar.gz` | Every judge rating behind the reported numbers: Llama-3.1-8B-Instruct (main judge), the three re-judgings (`__repeat1..3`) and the second judge Qwen2.5-72B-Instruct-AWQ, including the second and third revisions. Keys depend only on the judge model, temperature, prompt and the two descriptions, so extracting it in `<root>/HOVER_v8_work/` makes every evaluation reproduce exactly without re-sampling the judge (temperature 0.3). |
| `caches/vlm_cache_<backbone>.tar.gz` | The raw response of every VLM call of the main run and the revision analyses, one JSON per call. Keys include the absolute path, size and modification time of each window image, so they act as a cache only on the original machine; elsewhere they are a complete record of model outputs. |

## Data not included

The VideoGlitchBench videos (`raw/`), `annotations.json` and the official `groundtruth.json` used by
the evaluator are third-party files and are not redistributed here; obtain them from the
VideoGlitchBench / GliDe release and put them in the VideoGlitchBench root that `runs.json` points to.

## Qualitative figures

| File | Contents |
|---|---|
| `main_internvl25-8b_window.zip` | Qualitative examples, InternVL2.5-8B with window refinement. |
| `supp_S1_internvl25-8b.zip` … `supp_S5_ui-tars-15-7b.zip` | Qualitative examples for the supplementary material, one backbone each. |
| `different_interval_*.zip` | The same clips under different refine modes (Qwen2.5-VL-3B/7B, UI-TARS-1.5-7B). |

Each of these holds the rendered figures (PDF and PNG) and a `_latex/` folder with the extracted frames
and the LaTeX source of the figure.

## Quick start

```bash
pip install -r requirements.txt
python HOVER.py selftest            # 59 unit checks, no GPU or servers needed
python hover_revision.py selftest   # 10 checks of the revision analyses
```

Then follow `REPRODUCE.md`. Scores computed from the released outputs with the released judge
cache reproduce exactly. Steps that call the VLM again (ablation re-runs, the GliDe emission-rule
control, the probe) need the backbone served as in `start_detector.sh` and can differ slightly
from ours (Supp. Tab. S10 bounds run-to-run variation).
