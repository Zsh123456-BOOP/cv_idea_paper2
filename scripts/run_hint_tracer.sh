#!/usr/bin/env bash
# The method stage, third round, the premise test (the planning session's plan, rules written before any reading:
# work/method_stage20261118/前提检验_带提示的描字器_方案与判定规则_2026-11-20.md): NAFNet given a hint of what is written -- a text
# typeset in print over the region's ink (data.hint_image) -- as a second input channel, its output only taking ink away (NAFNet-hint),
# once per fold of the second round (paper2_clean_nafnet_folds20261119: its folds, windows, shared readers, R0 and E-NAFNet readings),
# trained on the other folds with the answer or R0's reading as the hint half the time each, stopped by its own fold with R0's
# reading.  Read by each fold's shared reader: T-空 (white hint), T-真 (the answer), T-初读 (R0's reading), T-循环 (the loop,
# eval.hint_loop compose: up to three rounds, each round's hint the last reading) and T-错 (the answer with one symbol changed).
# eval.transcription against E-NAFNet, R0, T-初读 and H; eval.hint_loop the plan's rules, the loop, groups, pixels and the sheets.
# Card 1 only (the user's word, 2026-11-19: 「只用 1号卡」); no GPU work before the second round is COMPLETE.  PART=smoke: two passes,
# eight regions of fold 0 read; PART=main: the round.  Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_hint_tracer20261120; PART=${PART:-main}
case "$PART" in main) TAG=; BUDGET=method_round1; LIMIT=(); READ=(0 1 2 3 4) ;;
    smoke) TAG=_smoke; BUDGET=method_smoke; LIMIT=(--limit 8); READ=(0) ;; *) echo "PART is main or smoke"; exit 2 ;; esac
R=$B$TAG; mkdir -p "$R"
P2=${N}_nafnet_folds20261119; P2R=$P2; FOLDS_PACK=${N}_revision_folds_human_masked20261117b; V=${FOLDS_PACK}_views
DATASET=$O/paper2_clean_line_dataset20261117/regions.json; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
PAGES=(--pages "$O/paper2_struck_dataset20261113/pages" --pages "$O/paper2_full_review20261010/images"); HOLD=(--hold-out test-page)
UP=$O/paper2_restoration_sources20261008; SETS=${B}_sets
source "$ROOT/scripts/_restorers.sh"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2R/COMPLETE" || { echo 'the second round is not complete'; exit 2; }
CARD=${ALL_CARDS[1]}; NUMA=$(numa_of "$CARD")
place() { one_card_place "$CARD" "$@"; }   # MiB WHOLE FUNCTION ARGS...
job() {   # OUT -- COMMAND...: one GPU job on card 1
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
views() { [ -e "$2/pack.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$1" --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --output-dir "$2" > "$2.log" 2>&1; }
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' reading folders
all5() { local k s=; for k in 0 1 2 3 4; do s+="${s:+,}$1_f$k"; done; echo "$s"; }
R0=$(all5 "$P2R/read_R0")

# The hints, their typesetting, the hinted sets and each fold's training tiles.
if ! done_ data; then
    for s in answer wrong white; do [ -e "$B/hints_$s.json" ] || { mkdir -p "$B"; "$PY" -m revrender.data.hint_image hints --pack "$V" --source "$s" --output "$B/hints_$s.json"; }; done
    [ -e "$B/hints_r0.json" ] || "$PY" -m revrender.data.hint_image hints --pack "$V" --source readings --readings "$R0" --output "$B/hints_r0.json"
    [ -e "$B/typesetting.json" ] || "$PY" -m revrender.data.hint_image stats --pack "$V" --readings "$R0" --output "$B/typesetting.json" > "$B/typesetting.log" 2>&1
    # the plan's section two: more than a tenth falling back to the plain spelling stops the round until the planning session decides
    "$PY" -c 'import json,sys; t=json.load(open(sys.argv[1])); bad=[k for k,v in t.items() if v["plain_share"]>.1]; sys.exit("plain spelling above 10%: "+",".join(bad) if bad else 0)' "$B/typesetting.json"
    [ -e "$SETS/fold_4/training_native.npz" ] || "$PY" -m revrender.data.hint_image windows --sets "${P2}_sets" --answers "$B/hints_answer.json" \
        --readings "$B/hints_r0.json" --output-dir "$SETS" > "$SETS.log" 2>&1
    for k in 0 1 2 3 4; do [ -e "${B}_tiles_f$k/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$SETS/fold_$k" --no-replay \
        --output-dir "${B}_tiles_f$k" > "${B}_tiles_f$k.log" 2>&1; done
    mark data; fi

# The five hinted restorers.
tracer() {   # FOLD
    VAL=$SETS/fold_$1 RESTORER_MEMORY=6000 RESTORER_ARGS= finetune_restorer NAFNet-hint "$CARD" "$NUMA" - "${B}_tiles_f$1" "${B}${TAG}_NAFNet-hint_f$1" "$BUDGET"
    test -e "${B}${TAG}_NAFNet-hint_f$1/selection.json"
}
if ! done_ train; then
    for k in 0 1 2 3 4; do place 6000 0 tracer "$k"; done
    wait; for k in 0 1 2 3 4; do test -e "${B}${TAG}_NAFNet-hint_f$k/selection.json"; done
    mark train; fi

# A hinted pack: each region traced by its fold's restorer with the given hints, with its lines; then read by the fold's shared reader.
RUNS=($(for k in 0 1 2 3 4; do echo "--separator ${B}${TAG}_NAFNet-hint_f$k"; done))
trace() {   # OUT HINTS
    [ -e "$1/pack.json" ] || job "$1" -- "$PY" -m revrender.data.struck_pack separated --pack "$FOLDS_PACK" --sets "$SETS" "${RUNS[@]}" --upstream "$UP" \
        --hints "$2" --output-dir "$1"
}
read_with() {   # OUT PACK FOLD
    [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.recognize_lines --pack "$2" --model "$MODEL" --backend hf \
        --adapter "${P2}_reader_shared_f$3/adapter" --batch-size 8 --conditions context_revision --fold "$3" --loop-limit 4 "${LIMIT[@]}" --output-dir "$1"
}
arm() {   # NAME HINTS: trace, lines, read (every fold)
    local pack=${B}${TAG}_pack_$1
    if ! done_ "arm_$1"; then
        place 6000 0 trace "$pack" "$2"; wait; test -e "$pack/pack.json"; views "$pack" "${pack}_views"
        for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_$1_f$k" "${pack}_views" "$k"; done; wait
        for k in "${READ[@]}"; do test -e "$R/read_$1_f$k/complete.json"; done; mark "arm_$1"; fi
}
arm T-空 "$B/hints_white.json"; arm T-真 "$B/hints_answer.json"; arm T-错 "$B/hints_wrong.json"
arm T-初读 "$B/hints_r0.json"   # the loop's first round
for n in 2 3; do previous=$([ "$n" = 2 ] && echo T-初读 || echo T-loop2)
    # a region the last round did not read (the smoke reads eight) keeps R0's reading as its hint
    [ -e "$R/hints_loop$n.json" ] || "$PY" -m revrender.data.hint_image hints --pack "$V" --source readings --readings "$(dirs "$R/read_$previous")" \
        $([ "$PART" = main ] || echo --fallback "$B/hints_r0.json") --output "$R/hints_loop$n.json"
    arm "T-loop$n" "$R/hints_loop$n.json"; done
[ -e "$R/read_T-循环/complete.json" ] || "$PY" -m revrender.eval.hint_loop compose --initial "$R0" --round "$(dirs "$R/read_T-初读")" \
    --round "$(dirs "$R/read_T-loop2")" --round "$(dirs "$R/read_T-loop3")" --output-dir "$R/read_T-循环" > "$R/loop.log" 2>&1

# The scores, the rules and the sheets.
ARMS_=(--arm "T-空=$(dirs "$R/read_T-空")" --arm "T-真=$(dirs "$R/read_T-真")" --arm "T-初读=$(dirs "$R/read_T-初读")" --arm "T-循环=$R/read_T-循环"
       --arm "T-错=$(dirs "$R/read_T-错")" --arm "E-NAFNet=$(dirs "$P2R/read_E-NAFNet")" --arm "R0=$(dirs "$P2R/read_R0")" --arm "H=$(dirs "$P2R/read_H")")
if ! done_ score; then
    for ref in E-NAFNet R0 T-初读 H; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != E-NAFNet ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/hint/analysis.json" ] || "$PY" -m revrender.eval.hint_loop analyse --pack "$V" --sets "$SETS" --loop "$R/read_T-循环" "${ARMS_[@]}" \
        $(for ref in E-NAFNet R0 T-初读 H; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) --wrong "$B/hints_wrong.json" \
        --pixels "white=${B}${TAG}_pack_T-空" --pixels "R0=${B}${TAG}_pack_T-初读" --pixels "answer=${B}${TAG}_pack_T-真" --output-dir "$R/hint" > "$R/hint.log" 2>&1
    [ -e "$R/sheets/index.json" ] || "$PY" -m revrender.eval.hint_loop sheets --analysis "$R/hint" --human "$V" --blind "${P2}_erased" \
        --loop-packs "${B}${TAG}_pack_T-初读,${B}${TAG}_pack_T-loop2,${B}${TAG}_pack_T-loop3" --wrong-pack "${B}${TAG}_pack_T-错" --output-dir "$R/sheets" > "$R/sheets.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
