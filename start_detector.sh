#!/usr/bin/env bash
# usage: start_detector.sh <backbone>  -- vLLM detector on our MIG slice, port 8010, flags as in the main run
bk="$1"; UTIL="${UTIL:-0.92}"
declare -A M=([qwen2.5-vl-3b]=Qwen/Qwen2.5-VL-3B-Instruct [qwen2.5-vl-7b]=Qwen/Qwen2.5-VL-7B-Instruct [internvl2.5-4b]=OpenGVLab/InternVL2_5-4B [internvl2.5-8b]=OpenGVLab/InternVL2_5-8B [ui-tars-1.5-7b]=ByteDance-Seed/UI-TARS-1.5-7B)
declare -A L=([qwen2.5-vl-3b]=16384 [qwen2.5-vl-7b]=16384 [internvl2.5-4b]=24576 [internvl2.5-8b]=24576 [ui-tars-1.5-7b]=16384)
extra=()
case "$bk" in
  internvl*) extra+=(--chat-template /tpl/internvl_chat_template.jinja);;
  *) extra+=(--mm-processor-kwargs '{"max_pixels": 802816}');;
esac
curl -s localhost:8010/v1/models | grep -q "\"${M[$bk]}\"" && { echo "READY $bk (already up)"; exit 0; }
docker rm -f hover_rev_detector >/dev/null 2>&1; sleep 5
docker run -d --name hover_rev_detector --gpus '"device=MIG-150ebebd-23bf-5623-9d3f-034b5da709a4"' \
  -v /home/tahir/.cache/huggingface:/root/.cache/huggingface -v /home/tahir/VLM-Project:/tpl:ro \
  -p 8010:8010 --ipc=host vllm/vllm-openai:v0.8.5.post1 \
  --model "${M[$bk]}" --port 8010 --max-logprobs 20 --max-model-len "${L[$bk]}" \
  --gpu-memory-utilization "$UTIL" --limit-mm-per-prompt '{"image": 8}' "${extra[@]}" \
  --generation-config vllm --trust-remote-code --max-num-batched-tokens 16384 >/dev/null
for i in $(seq 120); do
  curl -s localhost:8010/v1/models >/dev/null && { echo "READY $bk"; exit 0; }
  docker ps -q -f name=hover_rev_detector | grep -q . || break; sleep 5
done
docker logs hover_rev_detector 2>&1 | grep -E "Error|error" | tail -5; exit 1
