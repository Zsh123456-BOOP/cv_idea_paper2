#!/usr/bin/env bash
# The diagnostic study of docs/struck_diagnostics_plan.md, unattended and resumable, on cards 1, 2 and 3 (card 3 on the user's word,
# 2026-10-11: ALLOW_GPU3=1):
#   1. build the versions and scopes (eval.struck_diagnostics build; the synthetic box remover needs a card) unless OUT_build is there;
#   2. per model a 20-request smoke (at least 95% of readings inside <transcription>, else the model is skipped and logged), then its
#      readings of both packs (boxed_read, boxed_marked); three one-card lanes at once; the secondary models read the S1 packs only;
#      MiniCPM-V-4 starts alone in its lane (it hung at start-up next to other engines) and, if it fails again, GLM-4.1V-9B-Thinking
#      takes its place as the fourth primary family;
#   3. the two-card models on cards 2 and 3 (gemma-3-12b in bf16 does not fit one 4090; Qwen3-VL-32B-FP8, reported apart);
#   4. score every model that read.
#     ALLOW_GPU3=1 run_struck_diagnostics.sh OUT_PREFIX
# A reading already complete is kept; one whose directory exists without complete.json is waited for (an earlier run may still be
# writing it).  Nothing is trained; no test writer is read (build refuses them).  Models load offline (HF_HUB_OFFLINE): remote-code
# models otherwise wait on the hub at start-up.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
M=/home/zsh/hme/models
require_clean_source
trap 'echo "pipeline failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT.FAILED"' ERR
stamp() { echo "$* $(date -u +%FT%TZ)"; }
gpu() { bash "$ROOT/scripts/launch_gpu_job.sh" "$@"; }
complete() { [ -f "$1/complete.json" ]; }
wait_for() {   # DIR: an earlier run's job still writing it
    until complete "$1" || [ -e "$1.FAILED" ]; do sleep 60; done
    complete "$1"
}

if [ ! -e "${OUT}_build" ]; then
    stamp "1. build"
    gpu $C1 0 "${OUT}_build" -- "$PY" -m revrender.eval.struck_diagnostics build --config "$ROOT/configs/struck_diagnostics.toml" --output-dir "${OUT}_build"
fi

parsed() {   # predictions.jsonl -> exit 0 when at least 95% of the raw outputs carry <transcription>
    "$PY" - "$1" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = sum('<transcription>' in r['raw'] for r in rows)
print(f'parsed {ok}/{len(rows)}')
sys.exit(0 if rows and ok >= .95*len(rows) else 1)
PY
}
read_model() {   # NAME MODEL_DIR CARD_UUIDS NUMA CUDA_INDICES SUFFIX EXTRA_ARGS...  -> 0 when both readings are complete
    local name="$1" dir="$2" card="$3" numa="$4" index="$5" suffix="$6"; shift 6
    local common=(--model "$dir" --max-model-len 6144 --gpu-memory .85 "$@")
    local run=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$index" "$PY" -m revrender.recognize_lines)
    if complete "${OUT}_read_${name}_marked"; then stamp "kept $name"; return 0; fi
    if [ -e "${OUT}_smoke_$name" ]; then
        complete "${OUT}_smoke_$name" || { stamp "skipped $name (an earlier smoke did not finish)"; return 1; }
    elif ! gpu "$card" "$numa" "${OUT}_smoke_$name" -- timeout 1800 "${run[@]}" --pack "${OUT}_build/pack_read$suffix" --conditions boxed_read \
              --limit 20 "${common[@]}" --output-dir "${OUT}_smoke_$name" > "${OUT}_smoke_$name.log" 2>&1 \
         || ! parsed "${OUT}_smoke_$name/predictions.jsonl" >> "${OUT}_smoke_$name.log" 2>&1; then
        stamp "skipped $name (smoke failed; see ${OUT}_smoke_$name.log)"; echo "$name" >> "${OUT}_skipped.txt"; return 1
    fi
    for kind in read marked; do
        local d="${OUT}_read_${name}_$kind"
        if [ -e "$d" ]; then wait_for "$d" || return 1; continue; fi
        gpu "$card" "$numa" "$d" -- "${run[@]}" --pack "${OUT}_build/pack_$kind$suffix" --conditions "boxed_$kind" "${common[@]}" \
            --output-dir "$d" > "$d.log" 2>&1
    done
    stamp "read $name"
}

stamp "2. one-card models"
( read_model qwen3vl8b $M/Qwen3-VL-8B-Instruct $C1 0 1 ''
  read_model qwen3vl4b $M/Qwen3-VL-4B-Instruct $C1 0 1 _s1 || true ) > "${OUT}_lane1.log" 2>&1 & L1=$!
( read_model qwen25vl7b $M/Qwen2.5-VL-7B-Instruct $C2 1 2 _s1 || true
  read_model qwen3vl2b $M/Qwen3-VL-2B-Instruct $C2 1 2 _s1 || true
  # MiniCPM alone on its card (its own smoke directory: the first one hung), else GLM-4.1V as the fourth primary family
  read_model minicpm $M/MiniCPM-V-4 $C2 1 2 '' --trust-remote-code \
      || read_model glm41v $M/GLM-4.1V-9B-Thinking $C2 1 2 '' --trust-remote-code --max-new-tokens 2048 || true ) > "${OUT}_lane2.log" 2>&1 & L2=$!
( read_model internvl35_8b $M/InternVL3_5-8B $C3 1 3 '' --trust-remote-code || true ) > "${OUT}_lane3.log" 2>&1 & L3=$!
wait $L1; wait $L2; wait $L3

stamp "3. two-card models"
read_model gemma3_12b $M/gemma-3-12b-it "$C2,$C3" 1 2,3 '' --tp 2 || true
read_model qwen3vl32b $M/Qwen3-VL-32B-Instruct-FP8 "$C2,$C3" 1 2,3 '' --tp 2 || true

stamp "4. score"
READINGS=()
for d in "${OUT}"_read_*_read "${OUT}"_read_*_marked; do
    complete "$d" || continue
    name=$(basename "$d"); name=${name#$(basename "$OUT")_read_}; name=${name%_read}; name=${name%_marked}
    READINGS+=(--readings "$name=$d/predictions.jsonl")
done
"$PY" -m revrender.eval.struck_diagnostics score --config "$ROOT/configs/struck_diagnostics.toml" --build "${OUT}_build" "${READINGS[@]}" \
    --output-dir "${OUT}_score" > "${OUT}_score.log" 2>&1
stamp "pipeline complete"
touch "$OUT.COMPLETE"
