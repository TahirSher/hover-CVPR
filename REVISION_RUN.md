# Re-computing the revision analyses on the MAIN run (main Tab. 1)

`hover_revision.py` recomputes every revision analysis — controls E1–E3, decomposition E4,
emission × span table and re-tuned ablations E5, extended segmentation grid E6, re-judging,
window probe, bisection repairs — **paired with the run of main Tab. 1**. The paper then rests
on one HOVER run instead of a main run plus an analysis run.

The script imports your `HOVER.py` (v10) as a library. It uses that file's evaluator port, judge,
statistics, VLM/judge caches and pipeline functions, so the numbers come out exactly as in your
main evaluation. It writes CSVs in the same schema as `revision_results.zip`, and
`paper/scripts/make_revision_tables.py` reads them unchanged.

---

## 0. What you need on the server

| Item | Where it is in your setup |
|---|---|
| `HOVER.py` | Same directory as `hover_revision.py`, or pass `--hover_py /path/HOVER.py` |
| VideoGlitchBench root (`annotations.json`, `raw/`, `splits/`) | `Config.root`, default `/home/tahir/VLM-Project/VideoGlitchBench` |
| Official `groundtruth.json` | The file you passed as `--gt_json` for main Tab. 1 |
| Work directory with caches | `<root>/HOVER_v8_work/` (windows, VLM cache, judge cache) |
| **Main-run HOVER file per backbone** | The `*.json` (not `*_batch_report.json`) that produced the `HOVER_<mode>` rows of `master_metrics_all.csv` in the mode selected for Tab. 1. The rows to match are listed in the table below. |
| Its calibration | The `calibration.json` used for that run, under `<work>/v10/params/<backbone>/` |
| GliDe re-run import | `<work>/v10/runs/<backbone>/glide_official_test.json` |
| Vanilla of the main run | The `vanilla_test.json` whose numbers are in Tab. 1 (e.g. Qwen2.5-VL-7B: 21 reports, F1×IoU 1.63) |
| vLLM server for each backbone | Needed for frame-mode spans, the GliDe emission control, ablations, seg grid and probe. Cached calls cost nothing. |
| Judge server | Llama-3.1-8B-Instruct, temperature 0.3, exactly as for Tab. 1 |

**Check that you have the right main-run files.** The run file must reproduce these rows of
`data/master_metrics_all.csv`:

| backbone | mode | #reports | F1×IoU |
|---|---|---|---|
| qwen2.5-vl-3b | whole | 374 | 30.70 |
| qwen2.5-vl-7b | window | 345 | 22.60 |
| internvl2.5-4b | window | 376 (377 clips) | 27.28 |
| internvl2.5-8b | frame | 373 | 28.39 |
| ui-tars-1.5-7b | window | 370 | 24.94 |

The files in `videoglitchbench.zip` (`crux_*_test_batch_report.json`, 383/357/380/358/277 reports)
are **not** these runs. A quick check:

```bash
python - <<'EOF'
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "HOVER_window_test.json"))
print(d.get("method"), d.get("config", {}).get("refine_mode"),
      sum(len(o["bugs"]) for o in d["outputs"]), "reports;",
      "hypotheses cached:", all("hypotheses" in o for o in d["outputs"]))
EOF
```

The post-hoc table (E5) also needs `"hypotheses"` cached in every output. `HOVER.py run` writes
them; the batch reports do not contain them.

---

## 1. Write `runs.json`

```json
{
  "root": "/home/tahir/VLM-Project/VideoGlitchBench",
  "gt_json": "/home/tahir/VLM-Project/VideoGlitchBench/groundtruth.json",
  "judge_api_base": "http://localhost:8001/v1",
  "judge_model": "meta-llama/Llama-3.1-8B-Instruct",
  "backbones": {
    "qwen2.5-vl-3b": {
      "hover": "HOVER_v8_work/v10/runs/qwen2.5-vl-3b/<MAIN RUN FILE>.json",
      "glide": "HOVER_v8_work/v10/runs/qwen2.5-vl-3b/glide_official_test.json",
      "vanilla": "HOVER_v8_work/v10/runs/qwen2.5-vl-3b/<MAIN VANILLA FILE>.json",
      "calibration": "HOVER_v8_work/v10/params/qwen2.5-vl-3b/calibration.json",
      "mode": "whole",
      "api_base": "http://localhost:8000/v1"
    },
    "qwen2.5-vl-7b":  { "...": "same keys; mode window" },
    "internvl2.5-4b": { "...": "same keys; mode window" },
    "internvl2.5-8b": { "...": "same keys; mode frame" },
    "ui-tars-1.5-7b": { "...": "same keys; mode window" }
  }
}
```

- Relative paths are resolved against the directory of `runs.json`.
- `mode` overrides the refine mode stored in the run file. Set it to the mode used in Tab. 1.
- `api_base` is the vLLM server of that backbone.

---

## 2. Self-test (no servers needed, about 1 minute)

```bash
python hover_revision.py selftest
```

This runs the whole chain on `HOVER.py`'s synthetic benchmark with the mock VLM and the mock
judge. All ten checks must pass. Among them:

- whole-clip spans leave F1 unchanged;
- the controls reference equals the main-run file;
- the post-hoc selector reproduces the run file exactly;
- the decomposition identity holds.

---

## 3. Run

Run each step for all backbones, or one backbone at a time with `--backbones qwen2.5-vl-7b`.

```bash
# cheap: the judge cache covers most pairs; new pairs are only the fixed sentences and merged reports
python hover_revision.py controls  --runs runs.json --out revision_results
python hover_revision.py decompose --runs runs.json --out revision_results

# emission x span table; frame mode re-uses cached bisection calls, new calls only for non-emitted hypotheses
python hover_revision.py posthoc   --runs runs.json --out revision_results

# three fresh judgings (separate judge caches *_repeat1..3), about 3x the judge calls of one evaluation
python hover_revision.py rejudge   --runs runs.json --out revision_results --repeats 3

# development split: extended tau grid and the 72-clip window probe
python hover_revision.py seg_grid  --runs runs.json --out revision_results
python hover_revision.py probe     --runs runs.json --out revision_results --probe_limit 72

# most expensive: four variants per backbone, each re-tuned on dev and run on test
python hover_revision.py ablate    --runs runs.json --out revision_results

# or everything in order:
python hover_revision.py all --runs runs.json --out revision_results
```

If `posthoc` prints `WARNING: recomputed selector_<mode> ... differs from the run file`, the
calibration file in `runs.json` does not belong to the main run. Point it to the calibration that
produced the main run, otherwise the emission-rule rows are computed with the wrong selector.

Useful options:

| Option | Effect |
|---|---|
| `--n_bootstrap` | Default 2000, as in the paper |
| `--medoid_shortlist` | Default 40. Development descriptions short-listed by TF-IDF centrality before the judge picks the medoid. This takes about 40 × 371 judge calls. |
| `--game_pool loo` | Default. The per-game sentence comes from development descriptions **and the other test clips** of the same game, with the clip itself left out. |
| `--game_pool dev` | The per-game sentence uses development descriptions only, so no test labels are used at all. |

---

## 4. Rebuild the paper tables

```bash
cp revision_results/revision_*.csv paper/data/revision/
cd paper
python scripts/make_revision_tables.py
latexmk -pdf main.tex && latexmk -pdf supp.tex && latexmk -pdf -g main.tex && python scripts/make_xref.py
```

`make_revision_tables.py` picks up the ablation tag `E5_ablate_main` automatically.

---

## 5. Text to update once everything is on the main run

Every analysis then rests on the main run, so the "analysis run" framing must go. If you use Claude
on the server, give it this list together with the new CSVs.

1. **Main Sec. 5.1, paragraph "Main run and analysis run"**
   - Replace it with one sentence: all controls, ablations and re-judgings are paired with the run of Tab. 1.
   - Keep the run-to-run spread (main vs. earlier run) only if you also keep `s_runs` in the supplement.
2. **Captions of main Tabs. 3 and 6**
   - Remove "analysis run".
   - The reference row "HOVER, analysis run" in `tables/t_controls.tex` then equals Tab. 1. Drop it in `t_controls()` of `make_revision_tables.py`.
3. **Main Sec. 5.4** ("Is it the emission rule?") and **"What does the judge reward?"**
   - Re-read the numbers: whole-clip gain, longest-report gain, emission-rule + whole-clip comparison, medoid vs. HOVER.
   - Change "statistically indistinguishable on four backbones and better on UI-TARS" if the significance pattern changes.
4. **Main Sec. 5.6 ablation sentences**
   - Ranges of the event-resolution and utility-selection effects.
   - The InternVL2.5-4B "+1.8" remark.
   - "identical sets on three backbones" for selector vs. top-1.
5. **Fig. 3 / Supp. Fig. S9**
   - The profile examples come from the earlier run (frame mode for Qwen2.5-VL-7B).
   - Either regenerate them from the main-run file (`data/profile_examples_qwen7b.json`) or keep the caption that says which run they come from.
6. **Abstract and introduction**
   - "statistically indistinguishable from HOVER on four backbones".
   - "\constMedoid" (macro; it updates automatically).
7. **Supp. Sec. S6 "Why a second run"**
   - Rewrite it, or reduce it to the main-vs-earlier-run comparison.
8. **Supp. Tabs. S11–S16 captions and Limitations**
   - "analysis run" → "main run".
   - The Limitations sentence "Two complete runs differ by up to 3.67 points" stays true if you keep the comparison table.

Values that do **not** depend on the HOVER run, because GliDe's outputs, the ground truth and the
judge are shared:

- every GliDe control except the two emission-rule rows;
- the four fixed-sentence scores (26.75 for the medoid if the same sentence is chosen);
- the GliDe decomposition terms.

The constant medoid stays above the main-run HOVER on Qwen2.5-VL-7B (22.60) and UI-TARS-1.5-7B
(24.94) whatever the re-run gives. Report it as the paper now does.

---

## 6. Definitions implemented (they match Supp. Sec. S6)

| Variant | Definition |
|---|---|
| `glide_whole` | Every span of every GliDe report becomes [0, clip duration]. |
| `glide_nocopy` | Reports containing "red sports car" or "highway overpass" are dropped. |
| `glide_merge` | Reports of a clip whose spans overlap (connected components) become one report: union of spans, description of the earliest report. |
| `glide_first` / `glide_longest` | The report with the earliest start / longest total span. |
| `glide_emit_span` / `glide_emit_whole` | Each GliDe report becomes a hypothesis: members are the windows its span overlaps, support is \|members\|, and the entity is generic. Up to `max_hypotheses` reports, ranked by support, are verified with `verify_profile` (stage V), featurised with `hypothesis_features`, and emitted iff the main run's fusion model gives û ≥ t\* (stage U). Spans are GliDe's own or the whole clip. |
| `vanilla_top1_whole` | The earliest vanilla report with a whole-clip span. |
| `const_glide_example` | GliDe's prompt example, verbatim. |
| `const_generic` | "A character or object floats in mid-air without visible support." |
| `const_dev_medoid` | The development description with the highest mean judge rating as a prediction against all other development descriptions (short-list by TF-IDF centrality). |
| `const_game_loo` | The TF-IDF medoid of the same game's descriptions; see `--game_pool`. |
| Re-judging | Uses the cached judging (A0) and three fresh judgings (A1–A3, separate caches). Cohen's κ is quadratic-weighted, on the 0–5 ratings of identical report pairs. |
| Statistics | Game-cluster percentile bootstrap (B = 2000); exact game-level Fisher randomisation over 2^12 patterns; Holm within each tag and backbone (E1: eight comparisons, E2: two, E3: four, E5 ablation: four, E5 post-hoc: eight; re-judging: two). |
