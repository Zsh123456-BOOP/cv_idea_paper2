#!/usr/bin/env bash
# RevSep's stroke version (models.separation --stroke; the plan work/model_plan20261025/笔画对象模型_方案_2026-10-25.md): head A
# estimates the strike as an object (centreline, half-width, the mask S they draw), taught on the synthetic strikes' exact geometry
# (data.separation strokes) and the human regions' weak labels.  Declared before any result: the method is RevSep-both (a second pass
# ranged by the detector's masks and S side by side); RevSep-stroke (S in place of the detector's range) and RevSep-aux (head A only
# supervised, one pass) are ablations, and RevSep without head A is the existing run.  The recipe of (b): the synthetic pairs
# ([training.RevSep]), the version chosen on validation and the development half, then the 201 human regions alone ([finetune_long]);
# then one deciding batch with every kept version of the stroke arms, of RevSep and of the seven restorers' (b), chosen there, with
# the error zones and fragment shares (eval.restoration_errors zones) and RevSep-both paired with every other family.  The rule: more
# than 3 points of read-back above the best baseline, the writer-resampled interval above 0, the strike missed next to writing down.
# Unattended; a step whose marker is in $R is skipped.  Every job goes to whichever card has room (_common.sh open_card).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_revsep_stroke20261025
V4=${N}_restoration_pairs20261007_v4; STROKES=${N}_restoration_strokes_v4; TILES=${N}_revsep_tiles_human
VAL=$O/paper2_fixed_validation20261007; VAL_HINTS=${N}_revsep_validation_hints/hints.npz
UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
export DETECTOR=$O/paper2_pages_clean_run_unlabeled3976_tight
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
export DEV_EXPORT=$O/paper2_test_export20261022/export.json DEV_PAGES=$O/paper2_test_text20261018_page/site/images CHOOSE_SPLITS="val dev"
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
ARMS=(RevSep-both RevSep-stroke RevSep-aux)
BASELINES=(COR-Generator NAFNet DE-GAN Restormer DocDiff Strikethrough-CycleGAN StainRestorer)
mode() { case "$1" in RevSep-both) echo both ;; RevSep-stroke) echo stroke ;; RevSep-aux) echo detector ;; esac; }

done_ strokes || { [ -e "$STROKES/run_manifest.json" ] || "$PY" -m revrender.data.separation strokes --pairs "$V4" --gold-bank "${N}_strike_bank_strokes" \
    --human-validation "$VAL/validation_native.json" --same-student "$O/paper2_splits/same_student_pairs.json" --output-dir "$STROKES" > "$R/strokes.log" 2>&1; mark strokes; }
arm_job() {   # CARD NUMA ARM STAGE INIT DATA BUDGET EXTRA: one arm trained
    local card="$1" numa="$2" arm="$3" stage="$4" init="$5" data="$6" budget="$7" extra="$8"
    RESTORER_MEMORY=10000 RESTORER_FLAGS="--stroke --hint-from $(mode "$arm") $extra" finetune_restorer RevSep "$card" "$numa" "$init" "$data" \
        "${N}_revsep_${stage}_$arm" $budget > "$R/train_${stage}_$arm.log" 2>&1
}
train() {   # STAGE CHOSEN DATA BUDGET EXTRA: the three arms at once, each on a card with room
    local stage="$1" chosen="$2" data="$3" budget="$4" extra="$5"
    for arm in "${ARMS[@]}"; do
        local init=-; [ "$chosen" = - ] || init=$(chosen_of "$chosen" "$arm")
        place_job 10000 arm_job "$arm" "$stage" "$init" "$data" "$budget" "$extra"
    done
    wait
    for arm in "${ARMS[@]}"; do test -e "${N}_revsep_${stage}_$arm/selection.json"; done
}
versions() {   # STAGE LIST: the versions of LIST evaluated together, chosen, and their error zones
    local stage="$1" list="$2" evaluation=${N}_revsep_stroke_versions_$1
    evaluate_versions "$list" "$evaluation" "$R/versions_$stage.log"
    [ -e "$R/zones_$stage.json" ] || "$PY" -m revrender.eval.restoration_errors zones --evaluation "$evaluation" --splits val dev \
        --output "$R/zones_$stage.json" >> "$R/versions_$stage.log" 2>&1
}
candidates() {   # OUT RUN...: the kept versions of FAMILY=RUN pairs
    local out="$1"; shift; local runs=(); for r in "$@"; do runs+=(--run "$r"); done
    [ -e "$out" ] || "$PY" -m revrender.eval.unified_restoration candidates "${runs[@]}" --upstream "$UP" --stain-upstream "$STAIN_UP" --output "$out" >> "$R/candidates.log" 2>&1
}
done_ train_synthetic || { train synthetic - "$V4" "" "--strokes $STROKES"; mark train_synthetic; }
if ! done_ versions_synthetic; then
    list=(); for arm in "${ARMS[@]}"; do list+=("$arm=${N}_revsep_synthetic_$arm"); done
    candidates "$R/candidates_synthetic.json" "${list[@]}"; versions synthetic "$R/candidates_synthetic.json"; mark versions_synthetic; fi
done_ train_finetune_long || { train finetune_long "${N}_revsep_stroke_versions_synthetic/chosen" "$TILES" finetune_long ""; mark train_finetune_long; }
if ! done_ versions_final; then
    list=(); for arm in "${ARMS[@]}"; do list+=("$arm=${N}_revsep_finetune_long_$arm"); done
    list+=("RevSep=${N}_revsep_finetune_long_RevSep"); for m in "${BASELINES[@]}"; do list+=("$m=${N}_finetune_long_$m"); done
    candidates "$R/candidates_final.json" "${list[@]}"; versions final "$R/candidates_final.json"
    for other in "${BASELINES[@]}" RevSep RevSep-stroke RevSep-aux; do
        [ -e "$R/compare_final_$other" ] || "$PY" -m revrender.eval.unified_restoration compare --before "${N}_revsep_stroke_versions_final/chosen" \
            --after "${N}_revsep_stroke_versions_final/chosen" --pair RevSep-both="$other" --output-dir "$R/compare_final_$other" >> "$R/versions_final.log" 2>&1
    done
    mark versions_final; fi
touch "$R/COMPLETE"; stamp complete
