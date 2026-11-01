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
| `HOVER.py` | The full pipeline: preprocessing, HOVER, the vanilla baseline, a converter for GliDe outputs, a port of the official VideoGlitchBench evaluator (Llama-3.1-8B-Instruct judge), development-split tuning, statistics and the UAL-Bench extension. |
| `hover_revision.py` | Every revision analysis, paired with the main run (paper Tab. 1): controls on GliDe's outputs, vanilla and fixed-sentence baselines, five-backbone ablations, emission rule × span resolution, re-judging, window probe, segmentation grid and decomposition. Imports `HOVER.py` from the same folder. |
| `medoid_candidates.py` | Scores the 40 short-listed development descriptions as fixed whole-clip predictions and adds the TF-IDF medoid to the fixed-sentence controls (Supp. Tab. S11). |
| `runs.json` | Paths of the main-run output files per backbone, the refine mode chosen on the development split, and the vLLM endpoints. Edit the paths for your machine. |
| `run_revision_main.sh` | Driver: for each backbone, starts the detector, runs controls, post-hoc, segmentation grid, probe and ablations, then the decomposition. |
| `start_detector.sh` | Starts the vLLM server (docker `vllm/vllm-openai:v0.8.5.post1`) for one backbone with the main-run flags. Set the GPU device and Hugging Face cache path for your machine. |
| `warm_gather.py`, `warm_emission.py` | Fill the VLM cache in parallel for the ablation variants and for the GliDe emission-rule control, so the sequential analysis passes read cached calls. They do not change any result. |

## Results

| File | Contents |
|---|---|
| `revision_results.zip` | CSVs of all revision analyses on the main run (controls, ablation, post-hoc, re-judging, probe, repairs, segmentation grid, decomposition, medoid candidates). Identical to `data/revision/` in the paper zip. |
| `CVPR-HOVER-publication_results.zip` | Per-backbone main results and paired statistical comparisons for every refine mode (whole, window, frame), and the all-modes summary. |
| `all_backbones_timing.csv` | Inference time per clip of HOVER, the GliDe re-run and the vanilla baseline on 100 test clips for all five backbones (paper Tab. 7). |
| `videoglitchbench.zip` | Per-clip predictions and tuned parameters from the second complete run used to measure run-to-run variation (Supp. Tab. S10), including the ablation variants. HOVER's files there carry the prefix `crux_`. |
| `HOVER_UAL_Bench.zip` | Outputs of the transfer study on UAL-Bench (UAG-OOPS, UAG-FunQA, UAG-SSBD) for three backbones. |

## Qualitative figures

| File | Contents |
|---|---|
| `main_internvl25-8b_window.zip` | Qualitative examples, InternVL2.5-8B with window refinement. |
| `supp_S1_internvl25-8b.zip` … `supp_S5_ui-tars-15-7b.zip` | Qualitative examples for the supplementary material, one backbone each. |
| `different_interval_*.zip` | The same clips under different refine modes (Qwen2.5-VL-3B/7B, UI-TARS-1.5-7B). |

Each of these holds the rendered figures (PDF and PNG) and a `_latex/` folder with the extracted frames
and the LaTeX source of the figure.

## Reproducing the revision analyses

1. Put `HOVER.py`, `hover_revision.py`, `runs.json` and the helper scripts in one folder, and point
   `runs.json` at your VideoGlitchBench root, ground truth and main-run output files.
   The main-run per-clip outputs are not in this repository.
2. Serve the judge (`meta-llama/Llama-3.1-8B-Instruct`, temperature 0.3) on port 8020 with vLLM.
3. Run `./run_revision_main.sh`. Results go to `revision_results/`. Then run
   `python medoid_candidates.py --runs runs.json --out revision_results`.
4. Copy `revision_results/*.csv` to `data/revision/` in the paper sources and follow the build order
   in the paper zip's `README.md`.

VLM and judge calls are cached by `HOVER.py`, so a re-run with the same caches reproduces every number.
