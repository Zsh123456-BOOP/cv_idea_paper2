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
# ROUND=2 (user 2026-11-16): every region confirmed so far (data.struck_dataset over the exports, then line_set merge with the line page's
# review and corrections) as one prefill set, a tenth of the writers held out by the same hash (a subset of round 1's held-out writers,
# so round 1's model never saw them); round 1's three teachers' labels, those on a page now labelled or of a held-out writer left out;
# NAFNet from scratch with them (main) and without (real); both and round 1's main scored on the held-out writers, each against round
# 1's main by the switch rule (main first, real only if main fails).  Card 2 only (the user's word); marker $O/paper2_prefill_model20261116.
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

score_job() {   # CARD NUMA OUT MODEL... -- SET...: eval.prefill score
    local card="$1" numa="$2" out="$3"; shift 3
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$PY" -m revrender.eval.prefill score "$@" --upstream "$UP" --output-dir "$out" > "$out.log" 2>&1
}
if [ "${ROUND:-1}" = 3 ]; then
    # The trials after round 2 (rules: work/prefill_round2_plan20261116/预填第2轮之后_三样试验与贴边框_判定规则_2026-11-16.md, user
    # 2026-11-16): the baseline is the NAFNet round 2's rule kept (round 1's main); each trial changes one thing of its recipe, trained
    # from scratch on its budget, seed 42, its versions chosen on the fixed validation alone, scored once on round 2's held-out writers.
    # G: --augment.  C: each region with half its box's height of the page round it (prefill_variants context), scored inside the region.
    # Both passing: C + G, against the better.  A: the real-only recipe on 25% and 50% of round 2's training writers (nested), beside
    # round 2's real-only (100%).  The tight polygon's second version (tight_region learn) on the baseline's strike.  Card 2 only, one job
    # at a time, after the library's prefill (paper2_prefill_text20261116/COMPLETE).  Marker $O/paper2_prefill_trials20261116.
    R=$O/paper2_prefill_trials20261116; BASE=${N}_prefill_model_main; SET2=${N}_prefill_set_r2; LIBIMG=$O/paper2_full_review20261010/images
    CTX_FIXED=${N}_fixed_validation_ctx; CTX_SET2=${N}_prefill_set_r2_ctx; CTX_PSEUDO=${N}_prefill_pseudo_main_ctx
    CARD=${ALL_CARDS[2]}; NUMA=$(numa_of "$CARD"); mkdir -p "$R"
    exec 9>"$R/lock"; flock -n 9 || { echo 'another run of the trials is still going'; exit 1; }
    trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
    until [ -e "$O/paper2_prefill_text20261116/COMPLETE" ]; do [ ! -e "$O/paper2_layer_reader20261113/PREFILL_FAILED" ] || exit 1; sleep 120; done
    if ! done_ data; then
        for pair in "$N"_prefill_pseudo_main:"$CTX_PSEUDO" "$FIXED":"$CTX_FIXED" "$SET2":"$CTX_SET2"; do
            [ -e "${pair#*:}/run_manifest.json" ] || "$PY" -m revrender.data.prefill_variants context --set "${pair%%:*}" --pages "$LIB/images" --pages "$LIBIMG" \
                --pages "$O/paper2_pages_clean_unlabeled3976" --output-dir "${pair#*:}" >> "$R/data.log" 2>&1
        done
        [ -e "${N}_prefill_tiles_main_ctx/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$CTX_PSEUDO" --no-replay \
            --output-dir "${N}_prefill_tiles_main_ctx" >> "$R/data.log" 2>&1
        for share in 25 50; do
            [ -e "${SET2}_s$share/run_manifest.json" ] || "$PY" -m revrender.data.prefill_variants subset --set "$SET2" --share "0.$share" --output-dir "${SET2}_s$share" >> "$R/data.log" 2>&1
            [ -e "${N}_prefill_tiles_r2_s$share/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "${SET2}_s$share" --no-replay \
                --output-dir "${N}_prefill_tiles_r2_s$share" >> "$R/data.log" 2>&1
        done
        mark data; fi
    trial() {   # NAME TILES BUDGET VALIDATION [more of the restorer]: one trial trained alone on card 2
        local name="$1" tiles="$2" budget="$3"; VAL="$4"; shift 4
        RESTORER_MEMORY=10000 RESTORER_ARGS="--select blocks $*" finetune_restorer NAFNet "$CARD" "$NUMA" - "$tiles" "${N}_prefill_model_t_$name" "$budget" > "$R/train_$name.log" 2>&1
        test -e "${N}_prefill_model_t_$name/selection.json"
    }
    if ! done_ train; then
        trial g "${N}_prefill_tiles_main" prefill "$FIXED" --augment
        trial c "${N}_prefill_tiles_main_ctx" prefill "$CTX_FIXED"
        trial a25 "${N}_prefill_tiles_r2_s25" real_only "$FIXED"; trial a50 "${N}_prefill_tiles_r2_s50" real_only "$FIXED"
        mark train; fi
    merged() {   # OUT SCORES...: the models of several score files in one (their held-out regions are the same ids)
        "$PY" -c 'import json,sys; m={}; [m.update(json.load(open(f))) for f in sys.argv[2:]]; json.dump(m, open(sys.argv[1], "x"))' "$@"
    }
    if ! done_ score; then
        [ -e "$R/plain/scores.json" ] || score_job "$CARD" "$NUMA" "$R/plain" --model "baseline=$BASE" --model "g=${N}_prefill_model_t_g" \
            --model "a25=${N}_prefill_model_t_a25" --model "a50=${N}_prefill_model_t_a50" --model "a100=${N}_prefill_model_r2_real" --set "held_out=$SET2/validation_native.json"
        [ -e "$R/context/scores.json" ] || score_job "$CARD" "$NUMA" "$R/context" --model "c=${N}_prefill_model_t_c" --set "held_out=$CTX_SET2/validation_native.json"
        [ -e "$R/scores.json" ] || merged "$R/scores.json" "$R/plain/scores.json" "$R/context/scores.json"
        for arm in g c; do [ -e "$R/switch_$arm.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$R/scores.json" --set held_out --current baseline \
            --candidate $arm --output "$R/switch_$arm.json"; done
        for pair in a25:a50 a50:a100; do [ -e "$R/curve_${pair/:/_}.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$R/scores.json" --set held_out \
            --current "${pair%%:*}" --candidate "${pair#*:}" --output "$R/curve_${pair/:/_}.json"; done
        mark score; fi
    passed() { "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["replace"] else 1)' "$1"; }
    if ! done_ both; then
        if passed "$R/switch_g.json" && passed "$R/switch_c.json"; then
            trial cg "${N}_prefill_tiles_main_ctx" prefill "$CTX_FIXED" --augment
            [ -e "$R/context_cg/scores.json" ] || score_job "$CARD" "$NUMA" "$R/context_cg" --model "cg=${N}_prefill_model_t_cg" --set "held_out=$CTX_SET2/validation_native.json"
            [ -e "$R/scores_cg.json" ] || merged "$R/scores_cg.json" "$R/scores.json" "$R/context_cg/scores.json"
            better=$("$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(min("gc", key=lambda m: s[m]["sets"]["held_out"]["summary"]["blocks"]))' "$R/scores.json")
            [ -e "$R/switch_cg.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$R/scores_cg.json" --set held_out --current "$better" --candidate cg --output "$R/switch_cg.json"
        fi
        [ -e "$R/tiers.json" ] || "$PY" -m revrender.eval.prefill tiers --scores "$( [ -e "$R/scores_cg.json" ] && echo "$R/scores_cg.json" || echo "$R/scores.json")" \
            --set held_out --output "$R/tiers.json" > "$R/tiers.log" 2>&1
        mark both; fi
    if ! done_ tight; then
        [ -e "$R/dataset_pages.txt" ] || "$PY" -c 'import json,sys; open(sys.argv[2], "x").write("\n".join(sorted({r["page"] for r in json.load(open(sys.argv[1]))}))+"\n")' \
            "${N}_prefill_data20261116/regions.json" "$R/dataset_pages.txt"
        if [ ! -e "$R/dataset_lib/COMPLETE" ]; then
            step=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$BASE/selection.json")
            bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$R/dataset_lib" -- "$PY" -m revrender.tools.annotate.pixel_library infer --restorer "$BASE" \
                --restorer-checkpoint "$step" --upstream "$UP" --pages "$O/paper2_pages_clean_unlabeled3976" --detector "$O/paper2_pages_clean_run_unlabeled3976_tight" \
                --mapping "$O/paper2_pixel_finetune_inputs20261007/index_manifest.json" --same-student "$O/paper2_pixel_finetune_inputs20261007/same_student_pairs.json" \
                --validation "$FIXED/validation_native.json" --ids "$R/dataset_pages.txt" --name paper2-dataset-pages --output-dir "$R/dataset_lib" > "$R/dataset_lib.log" 2>&1
        fi
        [ -e "$R/tight_v2/report.json" ] || "$PY" -m revrender.tools.annotate.tight_region learn --dataset "${N}_prefill_data20261116/regions.json" \
            --library "$R/dataset_lib" --writers "$SET2/summary.json" --output-dir "$R/tight_v2" > "$R/tight_v2.log" 2>&1
        mark tight; fi
    touch "$R/COMPLETE"; stamp 'trials complete'; exit 0
fi
if [ "${ROUND:-1}" = 2 ]; then
    R=$O/paper2_prefill_model20261116; DS=$O/paper2_clean_struck_dataset20261116; MERGED=${N}_prefill_data20261116; SET=${N}_prefill_set_r2; VAL=$SET
    R1=$O/paper2_prefill_model20261025; CARD=${ALL_CARDS[2]}; NUMA=$(numa_of "$CARD"); mkdir -p "$R"
    exec 9>"$R/lock"; flock -n 9 || { echo 'another run of round 2 is still going'; exit 1; }
    trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
    if ! done_ sets; then
        [ -e "$MERGED/run_manifest.json" ] || "$PY" -m revrender.data.line_set merge --dataset "$DS/regions.json" --folds "${N}_revision_folds_human_masked" \
            --review "$O/paper2_line_review_returned20261115/export_corrected.json" --fixes "$O/paper2_line_reading_fix_returned20261116b/export.json" \
            --output-dir "$MERGED" > "$R/sets.log" 2>&1
        [ -e "$SET/run_manifest.json" ] || "$PY" -m revrender.data.prefill_set --dataset "$MERGED/regions.json" --holdout .1 \
            --pages "$O/paper2_full_review20261010/images" --same-student "$SAME" --fixed-validation "$FIXED" --output-dir "$SET" >> "$R/sets.log" 2>&1
        mark sets; fi
    if ! done_ data; then
        teachers=(); for t in NAFNet DE-GAN COR-Generator; do teachers+=("$R1/teacher_$t"); done
        [ -e "${N}_prefill_pseudo_r2/run_manifest.json" ] || "$PY" -m revrender.data.pseudo_regions combine --predictions "${teachers[@]}" --review-set "$SET" \
            --soft --human-share .25 --pseudo-weight .5 --output-dir "${N}_prefill_pseudo_r2" > "$R/data.log" 2>&1
        for name in main real; do
            src=$SET; [ $name = real ] || src=${N}_prefill_pseudo_r2
            [ -e "${N}_prefill_tiles_r2_$name/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$src" --no-replay \
                --output-dir "${N}_prefill_tiles_r2_$name" >> "$R/data.log" 2>&1
        done
        mark data; fi
    if ! done_ train; then
        for arm in main real; do
            budget=prefill; [ $arm = main ] || budget=real_only
            RESTORER_MEMORY=10000 RESTORER_ARGS="--select blocks" finetune_restorer NAFNet "$CARD" "$NUMA" - "${N}_prefill_tiles_r2_$arm" \
                "${N}_prefill_model_r2_$arm" "$budget" > "$R/train_$arm.log" 2>&1 &
            sleep 90
        done
        wait; for arm in main real; do test -e "${N}_prefill_model_r2_$arm/selection.json"; done; mark train; fi
    if ! done_ score; then
        score_job "$CARD" "$NUMA" "$R/scores" --model "r1-main=${N}_prefill_model_main" --model "r2-main=${N}_prefill_model_r2_main" \
            --model "r2-real=${N}_prefill_model_r2_real" --set "held_out=$SET/validation_native.json" --set "fixed16=$FIXED/validation_native.json"
        test -e "$R/scores/scores.json"
        for arm in main real; do
            [ -e "$R/switch_$arm.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$R/scores/scores.json" --set held_out --current r1-main \
                --candidate "r2-$arm" --output "$R/switch_$arm.json"
        done
        mark score; fi
    touch "$R/COMPLETE"; stamp "round 2 complete"; exit 0
fi
if ! done_ sets; then
    [ -e "$SET/run_manifest.json" ] || "$PY" -m revrender.data.prefill_set --export "$TA" --export "$TE" --new-export "$B1" --pages "$O/paper2_full_review20261010/images" \
        --same-student "$SAME" --fixed-validation "$FIXED" --output-dir "$SET" > "$R/sets.log" 2>&1
    [ -e "$OLD/run_manifest.json" ] || "$PY" -m revrender.data.prefill_set --export "$TA" --pages "$O/paper2_full_review20261010/images" \
        --same-student "$SAME" --fixed-validation "$FIXED" --output-dir "$OLD" >> "$R/sets.log" 2>&1
    mark sets; fi
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
