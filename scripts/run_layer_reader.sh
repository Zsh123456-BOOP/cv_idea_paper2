#!/usr/bin/env bash
# The layer reader, first trial (user 2026-11-13): does telling a taught Qwen3-VL-8B which ink is the strike help it read the struck text?
#  0. eval.transcription over the readings of the pilot of 2026-11-08 (no reading anew): the same four ways of reading the 122 strikes,
#     now in the literature's scores (AR charges what a reading adds; the pilot's read-back is CR).
#  1. data.struck_pack folds: the confirmed regions outside the test page, five folds of writers; each region's scan, erased crop and
#     layer map come from the annotator's own pixels -- the best a separator could give, so this trial asks whether the condition can
#     help at all before a separator is trained fold by fold.
#  2. smoke: two steps of the mapped reader on fold 0, its projection no longer zero, eight regions read with it and not all alike.
#  3. per fold and arm -- raw (the scan), erased (the erased crop), pair (both, two images), mapped (the scan and the layer map into the
#     patch embedding) -- train_reader --fold K --keep-best, then recognize_lines --fold K on the writers it never saw.
#  4. the untaught 8B reads every region from the scan and from the erased crop.
#  5. eval.transcription, the reference the taught reader of the scan alone.
# The rule, written before any reading: the layer map is worth a separator of its own when the mapped reader beats the taught reader of
# the scan alone by more than 3 points of AR with its writer interval above 0, exact not lower.
# Unattended; a step whose marker is in $R is skipped.  CARDS (default "1") are the cards, by index, a job may go to, in that order,
# one job to a card.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
read -r -a WANT <<< "${CARDS:-1}"; PICK=(); for i in "${WANT[@]}"; do PICK+=("${ALL_CARDS[$i]}"); done; ALL_CARDS=("${PICK[@]}")
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_layer_reader20261113
DATA=$O/paper2_struck_dataset20261113; PACK=${N}_revision_folds_human; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
PILOT=$O/paper2_struck_reader_pilot20261108; ARMS=(raw erased pair mapped)
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

done_ rescore || { "$PY" -m revrender.eval.transcription --pack "${N}_struck_reader_eval_pack" --arm "erase_then_read=$PILOT/read_untaught@erased_neutral" \
    --arm "untaught_scan=$PILOT/read_untaught@raw_neutral" --arm "untaught_struck=$PILOT/read_untaught@raw_struck" --arm "taught_struck=$PILOT/read_taught" \
    --reference erase_then_read --output "$R/rescore_pilot.json" > "$R/rescore.log" 2>&1; mark rescore; }
if ! done_ pack; then
    [ -e "$PACK/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$PACK" > "$R/pack.log" 2>&1
    mark pack; fi
alone() {   # FUNCTION ARGS...: run FUNCTION CARD NUMA ARGS... in the background on a card of ours with room and no other job of ours (the 8B fills a card)
    local card c
    until card=$(for c in "${ALL_CARDS[@]}"; do [ "$(ours_on "$c")" -eq 0 ] && [ "$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 22000 ] && { echo "$c"; break; }; done; true) && [ -n "$card" ]; do sleep 30; done
    ( "$1" "$card" "$(numa_of "$card")" "${@:2}" ) & PLACED_PIDS+=($!); PLACED_CARDS+=("$card")
}
job() {   # CARD NUMA OUT -- COMMAND...: one GPU job
    local card="$1" numa="$2" out="$3"; shift 4
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$@" > "$out.log" 2>&1
}
teach() {   # OUT ARM FOLD [more arguments of train_reader]: a reader of the arm, fold left out
    local out="$1" arm="$2" k="$3"; shift 3
    echo "$PY" -m revrender.train_reader --pack "$PACK" --condition "${arm}_revision" --fold "$k" --model "$MODEL" --steps 100 --batch-size 2 --accumulate 8 \
        --warmup 10 --eval-every 10 --keep-best --checkpointing --workers 2 $([ "$arm" != mapped ] || echo --layer-map) "$@" --output-dir "$out"
}
read_with() {   # OUT ARM READER [more arguments of recognize_lines]
    local out="$1" arm="$2" reader="$3"; shift 3
    echo "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf --adapter "$reader/adapter" --batch-size 8 --conditions "${arm}_revision" \
        $([ "$arm" != mapped ] || echo --layer-map "$reader/layer_map.pt") "$@" --output-dir "$out"
}
if ! done_ smoke; then
    smoke_job() {   # CARD NUMA
        job "$1" "$2" "$R/smoke_reader" -- $(teach "$R/smoke_reader" mapped 0 --steps 2 --eval-every 1 --warmup 1)
        "$PY" -c "import sys, torch; w = torch.load(sys.argv[1])['weight']; assert float(w.abs().max()) > 0, 'the layer map was not taught'; print('layer map', tuple(w.shape), float(w.abs().max()))" "$R/smoke_reader/layer_map.pt"
        job "$1" "$2" "$R/smoke_read" -- $(read_with "$R/smoke_read" mapped "$R/smoke_reader" --fold 0 --limit 8)
        "$PY" -c "import json, sys; texts = {json.loads(l)['text'] for l in open(sys.argv[1])}; assert len(texts) > 2, f'eight regions read as {texts}: the reader no longer sees the image'" "$R/smoke_read/predictions.jsonl"
    }
    alone smoke_job; wait; test -e "$R/smoke_read/complete.json"; mark smoke; fi
arm_job() {   # CARD NUMA ARM FOLD: teach, then read the fold left out
    local reader="${N}_revision_reader_$3_f$4" reading="$R/read_$3_f$4"
    [ -e "$reader/complete.json" ] || job "$1" "$2" "$reader" -- $(teach "$reader" "$3" "$4")
    [ -e "$reading/complete.json" ] || job "$1" "$2" "$reading" -- $(read_with "$reading" "$3" "$reader" --fold "$4")
}
if ! done_ arms; then
    for k in 0 1 2 3 4; do for arm in "${ARMS[@]}"; do [ -e "$R/read_${arm}_f$k/complete.json" ] || alone arm_job "$arm" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for arm in "${ARMS[@]}"; do test -e "$R/read_${arm}_f$k/complete.json"; done; done
    mark arms; fi
if ! done_ untaught; then
    [ -e "$R/read_untaught/complete.json" ] || alone job "$R/read_untaught" -- "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions raw_revision erased_revision --output-dir "$R/read_untaught"
    wait; test -e "$R/read_untaught/complete.json"; mark untaught; fi
folds_of() { local a="$1" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$R/read_${a}_f$k"; done; echo "$s"; }
done_ score || { "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "taught_scan=$(folds_of raw)" --arm "taught_erased=$(folds_of erased)" \
    --arm "taught_pair=$(folds_of pair)" --arm "taught_mapped=$(folds_of mapped)" --arm "untaught_scan=$R/read_untaught@raw_revision" \
    --arm "untaught_erased=$R/read_untaught@erased_revision" --reference taught_scan --output "$R/scores.json" > "$R/score.log" 2>&1; mark score; }
touch "$R/COMPLETE"; stamp complete
