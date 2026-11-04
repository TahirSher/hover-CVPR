# Reproducing the results

This follows the order of operations in Supp. Sec. S9. Paths below are placeholders
(`/path/to/VideoGlitchBench`); set them for your machine in `runs.json` or with `--root`.

## 0. Unit tests (no GPU, no servers)

```bash
python HOVER.py selftest            # 59 checks: segmentation DP = brute-force optimum, F*/2 threshold,
                                    # permutation/bootstrap validity on null data, evaluator port, ...
python hover_revision.py selftest   # 10 checks of the revision analyses on the mock benchmark
```

Both run on a synthetic benchmark with a mock VLM and a mock judge, in a temporary directory.

## 1. Data

- VideoGlitchBench videos (`raw/`), `annotations.json` and the official `groundtruth.json` from the
  VideoGlitchBench / GliDe release (not redistributed here).
- `splits/*.json` from this repository go to `<root>/splits/`. The test split is used only in steps
  2, 5, 6 and 7 below.
- Optional: extract `caches/judge_cache.tar.gz` in `<root>/HOVER_v8_work/` to reuse every judge
  rating behind the paper (the judge samples at temperature 0.3, so fresh ratings vary slightly;
  see Supp. Tab. S16).

## 2. Main pipeline (paper Tab. 1), per backbone

```bash
# (1) serve the backbone (see start_detector.sh for the exact flags per backbone)
GPU_DEVICE=0 ./start_detector.sh qwen2.5-vl-7b
# (2) run GliDe unmodified on the test clips and import its batch report
python HOVER.py import_glide --split test --batch_report GliDe/data/results/batch_report.json
# (3) window probe on development clips
python HOVER.py probe --split dev --limit 72
# (4) serve the judge (meta-llama/Llama-3.1-8B-Instruct, temperature 0.3) and tune on dev
python HOVER.py calibrate --calib_splits dev
# (5) HOVER and the vanilla baseline on test
python HOVER.py run --method HOVER   --split test
python HOVER.py run --method vanilla --split test
# (6) evaluate on the intersection of clip sets with the official ground truth
python HOVER.py evaluate --split test --gt_json /path/to/groundtruth.json \
    --run_files HOVER_test.json glide_official_test.json vanilla_test.json
```

`python HOVER.py --help` lists all options (backbone, refine mode, API endpoints, work directory).
The per-clip outputs, tuning outputs and evaluation reports of our main run are in
`main_run_outputs.zip`; those of the second complete run (Supp. Tab. S10) are in
`videoglitchbench.zip`.

## 3. Controls, ablations, re-judging and probe (step 7; Supp. Sec. S6)

`hover_revision.py` scores every analysis against the main-run outputs, so all results are paired
with Tab. 1. Point `runs.json` at the files of `main_run_outputs.zip` (relative paths resolve against
the directory of `runs.json`) and at your judge and backbone servers.

```bash
python hover_revision.py controls  --runs runs.json --out revision_results   # GliDe-output controls, fixed sentences
python hover_revision.py decompose --runs runs.json --out revision_results
python hover_revision.py posthoc   --runs runs.json --out revision_results   # emission rule x span resolution
python hover_revision.py rejudge   --runs runs.json --out revision_results --repeats 3
python hover_revision.py seg_grid  --runs runs.json --out revision_results
python hover_revision.py probe     --runs runs.json --out revision_results --probe_limit 72
python hover_revision.py ablate    --runs runs.json --out revision_results   # re-tunes each variant on dev
python medoid_candidates.py        --runs runs.json --out revision_results   # Supp. Tab. S11
```

`run_revision_main.sh` runs all of these for the five backbones, starting each backbone's server
with `start_detector.sh` and filling the VLM cache in parallel with `warm_gather.py` /
`warm_emission.py` (these only speed things up). Our CSVs are in `revision_results.zip`.

Options: `--n_bootstrap` (default 2000), `--medoid_shortlist` (default 40), `--game_pool loo|dev`
(per-game fixed sentence from development and other test clips of the game, or development only).

## 3b. Second and third revisions (`revision2/`)

Edit the paths in `revision2/runs_v2.json`, serve the backbone and judge as above, and run from the
repository root (outputs go to `revision2/results/`; ours are already there):

```bash
python revision2/v2_analyses.py --runs runs.json        # matching protocols, detection, subsets
python revision2/judge_agreement.py                     # needs the Qwen2.5-72B-Instruct-AWQ judge
python revision2/v3_controls.py                         # shuffled-description and abstention controls
python revision2/score_glide_clean.py --backbone internvl2.5-8b   # also qwen2.5-vl-3b, ui-tars-1.5-7b
python revision2/score_full.py                          # full benchmark (all 5,238 clips)
python revision2/human_agreement.py && python revision2/human_methods.py   # human check
```

The raw outputs these read (stage H outputs, full-benchmark outputs, clean GliDe re-runs) are in
`revision2_outputs.zip` (extract it in the repository root; it fills `revision2/h_only`, `revision2/h_full` and
`revision2/glide_clean`); `caches/judge_cache.tar.gz` holds every rating of both judges, so scoring
reproduces our CSVs exactly. Copy `revision2/results/*.csv` to `paper/data/v2/` and run
`scripts/make_v2_tables.py` of the paper sources to rebuild the tables.

## 4. Tables and figures

`HOVER_CVPR2027_revised.zip` contains the paper sources with the data every number is read from
(`data/`) and the scripts that regenerate the tables and figures (`scripts/`); its `README.md` gives
the build order. To rebuild from your own outputs:

```bash
cp revision_results/revision_*.csv paper/data/revision/
cd paper && python scripts/make_revision_tables.py
latexmk -pdf main.tex && latexmk -pdf supp.tex
```

## Expected variation

Steps that call the VLM again (ablations, the GliDe emission-rule control, the probe) can differ
slightly from ours because of batched decoding on a different GPU; two complete runs of the pipeline
differ by up to 3.67 F1×IoU points (Supp. Tab. S10), and every paired conclusion against GliDe holds
in both. Scores computed from the released outputs with the released judge cache reproduce exactly.
