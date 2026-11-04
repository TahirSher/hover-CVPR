# Human check of the judge ratings (two annotators)

Each annotator rates their own sheet independently and without discussing it:
- Annotator A: `human_sheet_annotator_A.csv`
- Annotator B: `human_sheet_annotator_B.csv`

Both sheets contain the same 140 pairs in different orders. Do not open `key.csv` (it holds the method,
backbone and judge ratings) before both sheets are finished.

For each row, compare the predicted description with the ground-truth description and write an integer
0-5 into `human_rating_0_to_5`. Use the benchmark's own rubric (the one the LLM judges receive):

- **0 No quality**: unrelated; a different event or glitch; irrelevant or nonsensical.
- **1 Very low**: mentions some elements, but they are incorrect or irrelevant; minimal overlap.
- **2 Low**: minor aspects related to the glitch, but misses the main behaviour.
- **3 Moderate**: identifies the general type of glitch but lacks significant details.
- **4 High**: captures the main glitch and most key details; minor differences only
  (e.g. general terms instead of specific names).
- **5 Nearly identical**: same event, all key details, similar specificity.

Rule from the rubric: first identify the main abnormal behaviour of the ground-truth glitch. As long as the
prediction names the same main behaviour, the rating should be high.

You only see text, not the video; rate the match between the two descriptions. About 140 rows take
roughly 45-60 minutes. When done, save each CSV in place and tell Claude; it computes human-human and
human-judge agreement (weighted kappa, Spearman, agreement on rating >= 3) per method.
