#!/usr/bin/env python3
"""Score a clean GliDe re-run (detector capped so that SAM 3 has memory): InternVL2.5-8B, Qwen2.5-VL-3B, UI-TARS-1.5-7B.

Combines the per-video reports of the three workers, imports them exactly like the first re-run
(bugs and time_nodes verbatim), counts tracker outcomes, and compares on the test split with
HOVER, the core (H + top-1, whole clip) and the first GliDe re-run, paired by clip.
  python score_glide_clean.py [--backbone B] [--judge_model M --judge_api_base URL --tag j2]
-> results/v3_glide_clean[_<backbone> unless InternVL2.5-8B][_tag].csv, v3_glide_clean_tests[...].csv (same suffix); runs file glide_clean_test.json
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="internvl2.5-8b")
ap.add_argument("--out_dir", default="")
ap.add_argument("--judge_model", default="")
ap.add_argument("--judge_api_base", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args()
a.out_dir = a.out_dir or f"/path/to/GliDe/data_{a.backbone}_sam3_clean"
sfx = ("" if a.backbone == "internvl2.5-8b" else f"_{a.backbone}") + (f"_{a.tag}" if a.tag else "")
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(os.path.join(HERE, "runs_v2.json"))
if a.judge_model:
    spec["judge_model"], spec["judge_api_base"] = a.judge_model, a.judge_api_base
ARGS = argparse.Namespace(n_bootstrap=2000, medoid_shortlist=40, game_pool="loo", ablate_variants="")
c = R.Ctx(spec, a.backbone, ARGS)

# per-video reports of all workers -> one batch report with bare video stems (as for the first re-run)
reps = []
for p in sorted(glob.glob(os.path.join(a.out_dir, "shard_*", "results", "*_report.json"))):
    if os.path.basename(p) == "batch_report.json":
        continue
    r = json.load(open(p))
    s = str(r.get("video_name") or r.get("video_id") or "")
    s = s.split("::", 1)[1] if "::" in s else s
    r["video_name"] = os.path.basename(s).replace(".mp4", "")
    reps.append(r)
batch = os.path.join(a.out_dir, "batch_report_fixed.json")
json.dump(reps, open(batch, "w"), indent=2)
outs, info = H.import_glide_report(batch, c.splits["test"])
run_path = os.path.join(a.out_dir, "glide_clean_test.json")
H.atomic_write_json(run_path, {"method": "glide_official", "split": "test", "outputs": outs,
                               "source": batch, "import_info": info, "calls_per_video": None})
print("imported", info, flush=True)

# tracker outcomes in the per-clip logs (same patterns as count_tracker.py)
KINDS = [("success", "[Executor/object_tracking] success"), ("no_match", "found no objects"),
         ("oom", "FAILED: CUDA out of memory"), ("frame_loading", "FAILED: Failure in frame loading"),
         ("not_initialised", "tracker could not be initialized")]
trk = {"tracking_calls": 0, **{k: 0 for k, _ in KINDS}}
for p in glob.glob(os.path.join(a.out_dir, "shard_*", "logs", "*.log")):
    for line in open(p, errors="ignore"):
        if "[Planner] → object_tracking" in line:
            trk["tracking_calls"] += 1
        if "[Executor/object_tracking]" in line:
            for k, pat in KINDS:
                if pat in line:
                    trk[k] += 1
print("tracker", trk, flush=True)

c.files["glide_clean"] = run_path
Hm, G1, G2 = c.run("hover"), c.run("glide"), c.run("glide_clean")
names = sorted(set(Hm) & set(G1) & set(G2) & set(c.gt) & set(c.by_name))
whole = {v: [[0.0, round(c.duration(v), 2)]] for v in names}
P = {"hover": {v: R.pred(Hm[v].get("bugs", []), R.spans_of(Hm[v])) for v in names},
     "top1_whole": {v: (R.pred([R.cached_hypotheses(Hm[v])[0].description], [whole[v]])
                        if Hm[v].get("hypotheses") else R.pred([], [])) for v in names},
     "glide": {v: R.pred(G1[v].get("bugs", []), G1[v].get("time_nodes", [])) for v in names},
     "glide_clean": {v: R.pred(G2[v].get("bugs", []), G2[v].get("time_nodes", [])) for v in names}}
per = {k: R.evaluate(c, p, names) for k, p in P.items()}
groups = [c.by_name[v].game for v in names]
rows = [{"method": k, "n_clips": len(names), **R.summarise(per[k]), **(trk if k == "glide_clean" else {})} for k in P]
tests = []
for ref, var in (("glide_clean", "glide"), ("hover", "glide_clean"), ("top1_whole", "glide_clean")):
    st = R.paired(c, per[ref], per[var], groups)
    tests.append({"ref": ref, "var": var, "f1_iou_ref": R.summarise(per[ref])["f1_iou"],
                  "f1_iou_var": R.summarise(per[var])["f1_iou"], **st})
for name, rr in ((f"v3_glide_clean{sfx}.csv", rows), (f"v3_glide_clean_tests{sfx}.csv", tests)):
    fields = list(dict.fromkeys(k for r in rr for k in r))
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rr:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print(r["method"], round(100 * r["f1_iou"], 2), r["num_pred_bugs"])
for t in tests:
    print(t)
