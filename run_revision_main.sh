#!/usr/bin/env bash
# Revision analyses paired with the MAIN run (Tab. 1). Per backbone: the cheap steps and sharded
# cache warm-ups of the four ablation variants run as parallel clients of one detector; then the
# ablation variants (in parallel) and the final 'ablate' pass, which read the caches.
cd "$(dirname "$0")"
PY=python
OUT=revision_results
NSH=${NSH:-3}
VARS=(no_ctx_calib no_hypothesis_scoring no_event_resolution no_utility_selection)
BKS=(${BKS:-qwen2.5-vl-3b qwen2.5-vl-7b internvl2.5-4b internvl2.5-8b ui-tars-1.5-7b})
say() { echo "=== [$(date +%F' '%T)] $*"; }
step() { say "start $1 $2"; $PY -u hover_revision.py "$1" --runs runs.json --out $OUT --backbones "$2" "${@:3}" || say "FAILED $1 $2"; say "end $1 $2"; }
mkdir -p logs/warm
for bk in "${BKS[@]}"; do
  ./start_detector.sh "$bk" || { say "detector failed $bk"; continue; }
  say "backbone $bk"
  ( step controls "$bk"; step posthoc "$bk"; step seg_grid "$bk"; step probe "$bk" --probe_limit 72 ) > logs/cheap_$bk.log 2>&1 &
  cheap=$!
  wp=()
  for v in "${VARS[@]}"; do for sp in dev test; do for k in $(seq 0 $((NSH-1))); do
    $PY -u warm_gather.py --backbone "$bk" --variant $v --split $sp --shard $k --nshards $NSH > logs/warm/${bk}_${v}_${sp}_$k.log 2>&1 &
    wp+=($!)
  done; done; done
  wait "${wp[@]}"; say "warm-up done $bk"; tail -qn 1 logs/warm/${bk}_*.log
  ap=()
  for v in "${VARS[@]}"; do
    step ablate "$bk" --ablate_variants $v --out revision_warm > logs/ablate_${bk}_$v.log 2>&1 &
    ap+=($!)
  done
  wait "${ap[@]}"; wait $cheap
  grep -h "FAILED\|WARNING" logs/cheap_$bk.log logs/ablate_${bk}_*.log
  step ablate "$bk" > logs/ablate_${bk}_final.log 2>&1; tail -n 2 logs/ablate_${bk}_final.log
done
say decompose; $PY -u hover_revision.py decompose --runs runs.json --out $OUT
say ALL DONE
