# HOVER: Metric-Aware Hypothesis Selection for Open-Ended Video Game Glitch Detection

Code, results and paper sources for the CVPR 2027 submission. HOVER turns frozen vision-language
models (Qwen2.5-VL-3B/7B, InternVL2.5-4B/8B, UI-TARS-1.5-7B) into glitch detectors for
VideoGlitchBench by forming hypotheses across windows, verifying them, and emitting a hypothesis only
when its predicted contribution to F1×IoU exceeds a closed-form threshold.

## Paper

| File | Contents |
|---|---|
| `HOVER_CVPR2027_revised.zip` | Submission sources: `main.tex`, `supp.tex`, sections, tables, figures, the data every number is read from (`data/`), and the scripts that regenerate tables and figures. Includes `main.pdf` and `supp.pdf` built with pdfLaTeX. Its own `README.md` gives the build order and the list of revision changes. |
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

## Splits and caches

| File | Contents |
|---|---|
| `splits/` | The game-disjoint VideoGlitchBench splits used throughout: `train.json` (96 games), `dev.json` (445 clips, 12 games), `test.json` (383 clips, 12 games) and `game_assignments.json`. Place them in `<VideoGlitchBench root>/splits/`. |
| `caches/judge_cache.tar.gz` | Every Llama-3.1-8B-Instruct judge rating behind the reported numbers, including the three re-judgings (`__repeat1..3`). Keys depend only on the judge model, temperature, prompt and the two descriptions, so extracting it in `<root>/HOVER_v8_work/` makes every evaluation reproduce exactly without re-sampling the judge (temperature 0.3). |
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
