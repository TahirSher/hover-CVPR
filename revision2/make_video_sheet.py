#!/usr/bin/env python3
"""Video-grounded human check (reviewer point W4): annotators WATCH the clip and rate every report of the core
(H + top-1, whole clip) and of the GliDe re-run on it, matched or not, plus whether the clip contains a glitch.

60 test clips, 12 per backbone (9 with a ground-truth glitch, 3 without), every clip used once, spread round-robin
over the 12 test games; seed 0. Reports of a clip are shuffled and shown without the system's name; each
annotator gets the same rows in a different clip order. The ground truth is NOT shown.

  python make_video_sheet.py [--copy_videos]
-> human_check_video/annotator_{A,B}.xlsx, key.csv, INSTRUCTIONS.md, videos/ (with --copy_videos)
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # HOVER.py and hover_revision.py are in the repository root
import hover_revision as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "runs_v2.json"))
ap.add_argument("--out", default=os.path.join(HERE, "human_check_video"))
ap.add_argument("--per_backbone", type=int, default=12)
ap.add_argument("--clean_per_backbone", type=int, default=3)
ap.add_argument("--copy_videos", action="store_true")
a = ap.parse_args()
H = R.load_hover(os.path.join(os.path.dirname(HERE), "HOVER.py"))
spec = R.read_runs(a.runs)
ARGS = argparse.Namespace(n_bootstrap=0, medoid_shortlist=40, game_pool="loo", ablate_variants="")
BACKBONES = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "internvl2.5-8b", "ui-tars-1.5-7b"]
rng = random.Random(0)
used, rows = set(), []
for b in BACKBONES:
    R.H._PREP_MEMO.clear()
    c = R.Ctx(spec, b, ARGS)
    Hm, Gm = c.run("hover"), c.run("glide")
    names = sorted(set(Hm) & set(Gm) & set(c.gt) & set(c.by_name))
    has = {v: bool([d for d in c.gt[v].get("bugs", []) if str(d).strip()]) for v in names}
    pick = []
    for want, n in ((True, a.per_backbone - a.clean_per_backbone), (False, a.clean_per_backbone)):
        by_game = defaultdict(list)
        for v in names:
            if has[v] == want and v not in used:
                by_game[c.by_name[v].game].append(v)
        for g in by_game:
            rng.shuffle(by_game[g])
        games = sorted(by_game)
        rng.shuffle(games)
        k = 0
        while len([p for p in pick if has[p] == want]) < n and any(by_game.values()):
            g = games[k % len(games)]
            if by_game[g]:
                pick.append(by_game[g].pop())
            k += 1
    for v in pick:
        used.add(v)
        rec = c.by_name[v]
        top = R.cached_hypotheses(Hm[v])[0] if Hm[v].get("hypotheses") else None
        reps = ([("core", top.description, [[0.0, round(c.duration(v), 1)]])] if top else []) + \
               [("glide", d, sp) for d, sp in zip(Gm[v].get("bugs", []), Gm[v].get("time_nodes", []))]
        rng.shuffle(reps)
        fname = f"{rec.game}__{v}.mp4".replace("/", "_").replace(" ", "_")
        for j, (sysname, text, sp) in enumerate(reps):
            rows.append({"clip": v, "game": rec.game, "backbone": b, "video_file": f"videos/{fname}",
                         "video_src": rec.video_path, "duration": round(c.duration(v), 1), "report_no": j + 1,
                         "system": sysname, "report": str(text).strip(),
                         "span": "; ".join(f"{s:.1f}-{e:.1f}" for s, e in sp) if sp else "",
                         "gt_has_glitch": int(has[v]), "gt": " | ".join(c.gt[v].get("bugs", []))})
    print(b, len(pick), "clips", flush=True)

os.makedirs(a.out, exist_ok=True)
clips = sorted({r["clip"] for r in rows})
for i, r in enumerate(rows):
    r["row_id"] = i + 1
with open(os.path.join(a.out, "key.csv"), "w", newline="") as f:   # do not give to the annotators
    w = csv.DictWriter(f, fieldnames=["row_id", "clip", "game", "backbone", "system", "report_no", "report", "span",
                                      "gt_has_glitch", "gt", "video_file"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

for ann, seed in (("A", 1), ("B", 2)):
    order = clips[:]
    random.Random(seed).shuffle(order)
    wb = Workbook()
    ws = wb.active
    ws.title = "Reports"
    hdr = ["row_id", "clip_no", "video_file", "duration_s", "report_no", "report", "reported_span_s",
           "visible_0_1_2", "main_glitch_0_1_2", "comment"]
    ws.append(hdr)
    for n, v in enumerate(order, 1):
        for r in sorted((r for r in rows if r["clip"] == v), key=lambda r: r["report_no"]):
            ws.append([r["row_id"], n, r["video_file"], r["duration"], r["report_no"], r["report"], r["span"], None, None, None])
    ws2 = wb.create_sheet("Clips")
    ws2.append(["clip_no", "video_file", "duration_s", "contains_glitch_0_1", "comment"])
    for n, v in enumerate(order, 1):
        r = next(r for r in rows if r["clip"] == v)
        ws2.append([n, r["video_file"], r["duration"], None, None])
    for sh, cols in ((ws, ("H", "I")), (ws2, ("D",))):
        for col in cols:
            dv = DataValidation(type="whole", operator="between", formula1="0",
                                formula2="2" if sh is ws else "1", allow_blank=True)
            sh.add_data_validation(dv)
            dv.add(f"{col}2:{col}{sh.max_row}")
        for cell in sh[1]:
            cell.font = Font(bold=True)
    ws.column_dimensions["C"].width = 38
    ws.column_dimensions["F"].width = 80
    for row in ws.iter_rows(min_row=2, min_col=6, max_col=6):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws2.column_dimensions["B"].width = 38
    wb.properties.creator = ""
    wb.save(os.path.join(a.out, f"annotator_{ann}.xlsx"))

open(os.path.join(a.out, "INSTRUCTIONS.md"), "w").write(f"""# Video-grounded check of glitch reports

You get `annotator_A.xlsx` or `annotator_B.xlsx` and the folder `videos/`. Work alone; do not discuss
clips with the other annotator, and do not open `key.csv`.

Each clip ({len(clips)} in total) has one or more **reports** written by automatic systems. Some reports describe
a real glitch, some describe normal gameplay, some describe things that are not in the video at all. The
system names are hidden, and the order of reports is random.

For every clip:
1. Watch the whole clip (`video_file`; replay as often as you like, pausing is fine).
2. Sheet **Clips**: `contains_glitch_0_1` = 1 if the clip shows at least one glitch (something that should not
   happen in the game: objects floating or passing through each other, broken animation, physics errors,
   rendering errors, ...), else 0.
3. Sheet **Reports**, one row per report:
   - `visible_0_1_2`: is what the report describes **visible in the clip**?
     2 = yes, clearly; 1 = partly (right object or event, but details wrong or exaggerated);
     0 = no (not in the clip, or describes normal gameplay as a glitch).
   - `main_glitch_0_1_2`: does the report describe **the main glitch of this clip**?
     2 = yes; 1 = partly (related, but misses what is wrong); 0 = no, or the clip has no glitch.
   - `reported_span_s` is the time the system gives; judge the description, not the span.
   - Use `comment` for anything unclear.

Expected effort: about 60-90 minutes.
""")

if a.copy_videos:
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)
    for r in {r["video_file"]: r for r in rows}.values():
        shutil.copy2(r["video_src"], os.path.join(a.out, r["video_file"]))
print(len(clips), "clips,", len(rows), "reports,",
      {s: sum(r["system"] == s for r in rows) for s in ("core", "glide")}, "->", a.out)
