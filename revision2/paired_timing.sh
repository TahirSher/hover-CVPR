#!/usr/bin/env bash
# Same-session timing (Supp. Tab. S23): GliDe, vanilla, the core (H + top-1) and HOVER timed back to back
# on ONE vLLM server per backbone, 20 game-stratified test clips, sequential requests, no caches.
# Waits until our MIG slice has enough free memory; never touches other jobs.
# Outputs: VideoGlitchBench/paired_timing_v4/v10/timing/<bk>/timing_test_20.json (GliDe, vanilla)
#          revision2/results/v4_timing_paired.csv (core, HOVER)          log: revision2/logs/paired_timing.log
# GliDe and vanilla are timed by our GliDe harness (crux_v10.py timing, not part of this release); its per-clip
# times and GliDe outputs are in results/paired_timing_glide/.
set -uo pipefail
V2="$(cd "$(dirname "$0")" && pwd)"; VP=${VP:-/path/to/glide_runner}; ROOT=${ROOT:-/path/to/VideoGlitchBench}; WORK=paired_timing_v4
PORT=8013; NAME=hover_paired_timing; N=20; mkdir -p $V2/logs
SLICES=(${GPU_DEVICE:-0})   # one GPU (or MIG device)
declare -A SLOT=([${GPU_DEVICE:-0}]="${GPU_SLOT:-0 0}")   # GPU index and GPU-instance id in nvidia-smi
BACKBONES=(${ONLY:-qwen2.5-vl-3b qwen2.5-vl-7b internvl2.5-4b internvl2.5-8b ui-tars-1.5-7b})
declare -A M=([qwen2.5-vl-3b]=Qwen/Qwen2.5-VL-3B-Instruct [qwen2.5-vl-7b]=Qwen/Qwen2.5-VL-7B-Instruct [internvl2.5-4b]=OpenGVLab/InternVL2_5-4B [internvl2.5-8b]=OpenGVLab/InternVL2_5-8B [ui-tars-1.5-7b]=ByteDance-Seed/UI-TARS-1.5-7B)
declare -A F=([qwen2.5-vl-3b]=qwen [qwen2.5-vl-7b]=qwen [internvl2.5-4b]=internvl [internvl2.5-8b]=internvl [ui-tars-1.5-7b]=qwen)
declare -A L=([qwen2.5-vl-3b]=16384 [qwen2.5-vl-7b]=16384 [internvl2.5-4b]=24576 [internvl2.5-8b]=24576 [ui-tars-1.5-7b]=16384)
declare -A NEED=([qwen2.5-vl-3b]=18000 [qwen2.5-vl-7b]=26000 [internvl2.5-4b]=20000 [internvl2.5-8b]=28000 [ui-tars-1.5-7b]=26000)  # MiB for vLLM
SAM_MARGIN=16000   # MiB left for GliDe's SAM 3 tracker
TOTAL=71424
say() { printf "[%s] %s\n" "$(date '+%m-%d %H:%M:%S')" "$*"; }

used_mib() {   # used MiB of a slice, from the MIG table of nvidia-smi (GPU index, GI id)
  read -r g gi <<< "${SLOT[$1]}"
  nvidia-smi | awk -v g="$g" -v gi="$gi" '$2==g && $3==gi && $4==0 {for(i=1;i<=NF;i++) if($i ~ /MiB$/){sub("MiB","",$i); print $i; exit}}'
}

pick_slice() {  # first slice with room for detector + SAM 3; prints "uuid used"
  local bk=$1
  for s in "${SLICES[@]}"; do
    u=$(used_mib "$s"); [[ -z "$u" ]] && continue
    (( TOTAL - u >= NEED[$bk] + SAM_MARGIN )) && { echo "$s $u"; return 0; }
  done
  return 1
}

start_detector() {  # $1 backbone, $2 slice, $3 used MiB
  local bk=$1 s=$2 u=$3 extra=()
  local util; util=$(python3 -c "print(round(min(0.95, ($u + ${NEED[$bk]}) / $TOTAL), 3))")
  case "$bk" in internvl*) extra+=(--chat-template /tpl/internvl_chat_template.jinja);;
                *) extra+=(--mm-processor-kwargs '{"max_pixels": 802816}');; esac
  docker rm -f $NAME >/dev/null 2>&1; sleep 3
  docker run -d --name $NAME --gpus "\"device=$s\"" -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v "$(dirname "$V2")":/tpl:ro -p $PORT:$PORT --ipc=host vllm/vllm-openai:v0.8.5.post1 --model "${M[$bk]}" \
    --served-model-name "${M[$bk]}" --port $PORT --max-logprobs 20 --max-model-len "${L[$bk]}" \
    --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image": 8}' "${extra[@]}" \
    --generation-config vllm --trust-remote-code --max-num-batched-tokens 16384 >/dev/null
  for i in $(seq 180); do
    curl -s localhost:$PORT/v1/models | grep -q "${M[$bk]}" && { say "detector $bk up on $s (util $util)"; return 0; }
    docker ps -q -f name=$NAME | grep -q . || break; sleep 5
  done
  docker logs $NAME 2>&1 | tail -5; return 1
}

python3 - <<P   # runs file for the core/HOVER timer: same backbones, detector on port $PORT
import json; d = json.load(open("$V2/runs_v2.json"))
for b in d["backbones"].values(): b["api_base"] = "http://localhost:$PORT/v1"
json.dump(d, open("$V2/runs_v4_timing.json", "w"), indent=1)
P

i=0
for bk in "${BACKBONES[@]}"; do
  i=$((i+1))
  until pick=$(pick_slice "$bk"); do sleep 120; done
  read -r slice used <<< "$pick"
  say "=== $bk on $slice (used ${used} MiB)"
  start_detector "$bk" "$slice" "$used" || { say "$bk: detector failed, skipped"; continue; }
  glide() {
    CUDA_VISIBLE_DEVICES=$slice conda run -n sam3 python -u $VP/crux_v10.py timing --root $ROOT --work_subdir $WORK \
      --backbone "$bk" --model "${M[$bk]}" --family "${F[$bk]}" --backend openai --api_base http://localhost:$PORT/v1 \
      --api_key EMPTY --temperature 0.0 --top_p 1.0 --top_k -1 --top_logprobs 20 --fps 4.0 --window_size 8 \
      --jpeg_quality 95 --glide_repo ${GLIDE:-/path/to/GliDe} --timing_split test --timing_n $N --timing_methods glide,vanilla \
      --timing_backbones "$bk" --timing_warmup 2 --timing_repeats 1 --timing_reset_prefix_cache \
      --no-cache_read --no-cache_write > $V2/logs/paired_timing_${bk}_glide.log 2>&1
    say "$bk glide+vanilla exit $?"
  }
  ours() {
    python -u $V2/timing_v2.py --runs $V2/runs_v4_timing.json --backbone "$bk" --methods h_top1,hover \
      --subset $ROOT/$WORK/v10/timing/subset_test_$N.json --scratch paired_timing_v4_ours \
      --out $V2/results/v4_timing_paired.csv > $V2/logs/paired_timing_${bk}_ours.log 2>&1
    say "$bk core+hover exit $?"
  }
  if (( i % 2 )); then glide; ours; else glide_first=0; [[ -f $ROOT/$WORK/v10/timing/subset_test_$N.json ]] && { ours; glide; } || { glide; ours; }; fi
  docker rm -f $NAME >/dev/null 2>&1
done
say "PAIRED TIMING DONE"
