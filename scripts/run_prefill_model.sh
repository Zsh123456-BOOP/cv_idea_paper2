#!/usr/bin/env bash
# The prefill model, round 1 (plan work/model_plan20261025/预填模型_方案_2026-10-25.md): real data only, no synthetic pairs, one
# training from NAFNet's random start, distilled from four teachers.
#  1. data.prefill_set on every export so far (the 201 regions, the test page's of 10-22, training batch 1), its held-out writers by
#     hash; batch 1 also apart as the forward test; the 201 alone as the learning curve's first point (ablation 5).
#  2. forward test (eval.prefill score): the current prefill (the soft student) and the one batch 1 was prefilled with (NAFNet (b)
#     step_364) on batch 1's confirmed regions, with the annotator's edits and seconds.
#  3. three teachers (NAFNet, DE-GAN, COR (b), the versions chosen on the wider validation; the first round had a fourth teacher since retired) on the library's eligible
#     regions (pseudo_regions predict --prefill-set); their mean the soft target (combine --soft), the human regions a quarter of the
#     regions, a pseudo-labelled region's weight 0.5.
#  4. NAFNet trained from scratch ([prefill], versions kept on the held-out writers' blocks to fix: --select blocks; the forward test
#     showed the share of pixels to fix runs against the annotator's effort, the blocks with it), beside the
#     ablations: real only ([real_only]), pseudo weight 1, human share one half, the 201 regions alone.
#  5. every chosen version, the soft student and NAFNet (b) scored on the held-out writers and the fixed 16 (eval.prefill score); the
#     switch rule (eval.prefill switch) against the soft student on the held-out writers.  Replacing the library's prefill is done
#     after, by hand (headless Chrome first).
# Unattended; a step whose marker is in $R is skipped.  Every job goes to whichever card has room (_common.sh open_card).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_prefill_model20261025
TA=$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json; B1=$O/paper2_training_batch1_returned/export.json
TE=$O/paper2_test_export20261022/export.json; FIXED=$O/paper2_fixed_validation20261007; SAME=$O/paper2_splits/same_student_pairs.json
SET=${N}_prefill_set_r1; OLD=${N}_prefill_set_r1_201; LIB=${N}_full_review_prefill_soft
UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
CHOSEN=${N}_revsep_stroke_versions_final/chosen/selection.json; SOFT=${N}_pseudo_student_soft
VAL=$SET
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
spec_of() { "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1]))[sys.argv[2]]["spec"]; print(s["run"], s["checkpoint"])' "$CHOSEN" "$1"; }

if ! done_ sets; then
    [ -e "$SET/run_manifest.json" ] || "$PY" -m revrender.data.prefill_set --export "$TA" --export "$TE" --new-export "$B1" --pages "$O/paper2_full_review20261010/images" \
        --same-student "$SAME" --fixed-validation "$FIXED" --output-dir "$SET" > "$R/sets.log" 2>&1
    [ -e "$OLD/run_manifest.json" ] || "$PY" -m revrender.data.prefill_set --export "$TA" --pages "$O/paper2_full_review20261010/images" \
        --same-student "$SAME" --fixed-validation "$FIXED" --output-dir "$OLD" >> "$R/sets.log" 2>&1
    mark sets; fi
score_job() {   # CARD NUMA OUT MODEL... -- SET...: eval.prefill score
    local card="$1" numa="$2" out="$3"; shift 3
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$PY" -m revrender.eval.prefill score "$@" --upstream "$UP" --output-dir "$out" > "$out.log" 2>&1
}
if ! done_ forward; then
    place_job 8000 score_job "$R/forward" --model "soft-student=$SOFT:step_14999" --model "nafnet-b=${N}_finetune_long_NAFNet:step_364" --set "batch1=$SET/new_native.json"
    wait; test -e "$R/forward/scores.json"; mark forward; fi
teacher_job() {   # CARD NUMA FAMILY: one teacher's restoration of the eligible regions
    local card="$1" numa="$2" family="$3" runtime=$PY run step; read -r run step < <(spec_of "$family")
    (
        unset LD_LIBRARY_PATH
        if [ "$family" = DE-GAN ]; then
            runtime=/home/zsh/hme/envs/paper2_restoration_tf/bin/python; export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
            for d in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
        fi
        bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$R/teacher_$family" -- "$runtime" -m revrender.data.pseudo_regions predict --library "$LIB" \
            --run "$run" --checkpoint "$step" --upstream "$UP" --prefill-set "$SET" --output-dir "$R/teacher_$family" > "$R/teacher_$family.log" 2>&1
    )
}
TEACHERS=(NAFNet DE-GAN COR-Generator)
if ! done_ teachers; then
    for t in "${TEACHERS[@]}"; do [ -e "$R/teacher_$t/run_manifest.json" ] || place_job 12000 teacher_job "$t"; done
    wait; for t in "${TEACHERS[@]}"; do test -e "$R/teacher_$t/run_manifest.json"; done; mark teachers; fi
combine() {   # NAME SET SHARE WEIGHT TEACHER...: a pseudo-labelled set and its tiles
    local name="$1" set="$2" share="$3" weight="$4"; shift 4; local preds=(); for t in "$@"; do preds+=("$R/teacher_$t"); done
    [ -e "${N}_prefill_pseudo_$name/run_manifest.json" ] || "$PY" -m revrender.data.pseudo_regions combine --predictions "${preds[@]}" --review-set "$set" --soft \
        --human-share "$share" --pseudo-weight "$weight" --output-dir "${N}_prefill_pseudo_$name" > "$R/combine_$name.log" 2>&1
    [ -e "${N}_prefill_tiles_$name/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "${N}_prefill_pseudo_$name" --no-replay \
        ${TILE_ARGS:-} --output-dir "${N}_prefill_tiles_$name" >> "$R/combine_$name.log" 2>&1
}
if ! done_ data; then
    combine main "$SET" .25 .5 "${TEACHERS[@]}"; combine pw1 "$SET" .25 1 "${TEACHERS[@]}"; combine share50 "$SET" .5 .5 "${TEACHERS[@]}"
    combine old201 "$OLD" .25 .5 "${TEACHERS[@]}"
    [ -e "${N}_prefill_tiles_real/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --no-replay --output-dir "${N}_prefill_tiles_real" > "$R/tiles_real.log" 2>&1
    mark data; fi
arm_job() {   # CARD NUMA ARM: one arm trained from scratch, its versions kept on the held-out writers' blocks to fix
    local card="$1" numa="$2" arm="$3" model=NAFNet data="${N}_prefill_tiles_$3" budget=prefill
    case "$arm" in real) budget=real_only ;; esac
    RESTORER_MEMORY=10000 RESTORER_ARGS="--select blocks" finetune_restorer "$model" "$card" "$numa" - "$data" "${N}_prefill_model_$arm" "$budget" > "$R/train_$arm.log" 2>&1
}
ARMS=(main real pw1 share50 old201)
if ! done_ train; then
    for a in "${ARMS[@]}"; do [ -e "${N}_prefill_model_$a/selection.json" ] || place_job 10000 arm_job "$a"; done
    wait; for a in "${ARMS[@]}"; do test -e "${N}_prefill_model_$a/selection.json"; done; mark train; fi
if ! done_ score; then
    models=(--model "soft-student=$SOFT:step_14999" --model "nafnet-b=${N}_finetune_long_NAFNet:step_364")
    for a in "${ARMS[@]}"; do models+=(--model "$a=${N}_prefill_model_$a"); done
    place_job 10000 score_job "$R/scores" "${models[@]}" --set "held_out=$SET/validation_native.json" \
        --set "fixed16=$FIXED/validation_native.json"
    wait; test -e "$R/scores/scores.json"
    [ -e "$R/switch.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$R/scores/scores.json" --set held_out --current soft-student --candidate main --output "$R/switch.json"
    mark score; fi
touch "$R/COMPLETE"; stamp complete
