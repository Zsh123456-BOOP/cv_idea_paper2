#!/usr/bin/env bash
# Does the second paper's annotation help every restorer, not only ours?  (user 2026-10-22)  Unattended; a step whose marker is in
# $R is skipped.
#   run_annotation_value.sh
# 1. The box remover (RevRender) trained on the very synthetic pairs the seven restorers trained on (data.restoration_pairs boxes:
#    the pairs cut as its strips; configs/revrender_strokes.toml from ImageNet), so all eight have the same synthetic data.
# 2. Each model's synthetic version chosen on validation read-back (eval.unified_restoration choose): among the versions whose pixel
#    F1 on the 16 fixed regions improved and the last (what a restorer's training keeps; the box remover's every 2,000 steps).
# 3. The same 201 human regions with the same synthetic replay for every model (data.review_set with the pairs' strips as replay;
#    data.restoration_pairs human for the restorers' tiles), each fine-tuned from its chosen version (configs/restoration.toml
#    [finetune]; the box remover's revrender_strokes_human.toml), the version chosen again the same way.
# 4. Every model's paired read-back difference (eval.unified_restoration compare) and where the errors fall (eval.restoration_errors).
# Cards 1 and 2; card 3 only while nobody else is on it.  Validation only: the test set is read once, at the end.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_annotation_value20261022
V4=${N}_restoration_pairs20261007_v4; BOXES=${N}_v4_boxes; OURS=${N}_v4_control
SET=${N}_v4_finetune_set; TILES=${N}_v4_finetune_tiles; FT=${N}_v4_finetune
SEL1=${N}_v4_versions_synthetic; SEL2=${N}_v4_versions_finetuned
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
MAP=$O/paper2_splits/index_manifest.json; PAIRS=$O/paper2_splits/same_student_pairs.json
FT_IN=$O/paper2_pixel_finetune_inputs20261007; EXPORT=$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
SUITE=${N}_restoration_suite
declare -A RUN=([COR-Generator]=$SUITE/lane0/COR-Generator_gold_pixels [Restormer]=$SUITE/lane0/Restormer_gold_pixels
                [NAFNet]=$SUITE/lane2/NAFNet_gold_pixels [DE-GAN]=$SUITE/lane2/DE-GAN_gold_pixels
                [Strikethrough-CycleGAN]=$SUITE/lane3/Strikethrough-CycleGAN_gold_pixels [DocDiff]=$SUITE/lane3/DocDiff_gold_pixels
                [StainRestorer]=${N}_stainrestorer/StainRestorer_gold_pixels)
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
sha() { sha256sum "$1/run_manifest.json" | cut -d' ' -f1; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

source "$ROOT/scripts/_restorers.sh"
candidates() {   # STAGE OUT.json: the versions to choose from (stage synthetic | finetuned), every model
    local runs=() k root
    for k in "${!RUN[@]}"; do root=${RUN[$k]}; [ "$1" = synthetic ] || root=${FT}_$k; runs+=(--run "$k=$root"); done
    if [ "$1" = synthetic ]; then set -- "$1" "$2" --own "Ours=$OURS/formal" --every 2000; else set -- "$1" "$2" --own "Ours=${FT}_Ours/train/formal"; fi
    "$PY" -m revrender.eval.unified_restoration candidates "${@:3}" "${runs[@]}" --upstream "$UP" --stain-upstream "$STAIN_UP" --output "$2"
}
versions() {   # STAGE OUT
    [ -e "$R/candidates_$1.json" ] || candidates "$1" "$R/candidates_$1.json" >> "$R/$1.log" 2>&1
    evaluate_versions "$R/candidates_$1.json" "$2" "$R/$1.log"
}
finetune() {   # FAMILY CARD NUMA: a restorer fine-tuned from its chosen synthetic version
    finetune_restorer "$1" "$2" "$3" "$(chosen_of "$SEL1/chosen" "$1")" "$TILES" "${FT}_$1"
}

# ---- 1. the same synthetic data for the box remover; the shared fine-tuning set --------------------------------------------------
done_ boxes || { "$PY" -m revrender.data.restoration_pairs boxes --source "$V4" --output-dir "$BOXES" > "$R/boxes.log" 2>&1; mark boxes; }
if ! done_ set; then
    "$PY" -m revrender.data.review_set --review-export "$EXPORT" --pages "$O/paper2_full_review20261010/images" \
        --mapping "$FT_IN/index_manifest.json" --same-student "$FT_IN/same_student_pairs.json" --replay "$BOXES" --fixed-validation "$VAL" \
        --output-dir "$SET" > "$R/set.log" 2>&1
    "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --source "$V4" --output-dir "$TILES" >> "$R/set.log" 2>&1
    "$PY" -m revrender.eval.restoration_errors strokes --review-set "$SET" --pairs "$V4" --output "$R/strokes.json" >> "$R/set.log" 2>&1
    mark set; fi
if ! done_ ours; then
    SYNTHETIC=1 MAPPING=$MAP SAME_STUDENT=$PAIRS RECIPE=$ROOT/configs/revrender_strokes.toml bash "$ROOT/scripts/launch_train_arm.sh" \
        control "$C1" 0 "$BOXES" "$(sha "$BOXES")" - - "$OURS" > "$R/ours.log" 2>&1
    mark ours; fi

# ---- 2. synthetic versions, chosen on read-back ---------------------------------------------------------------------------------
done_ synthetic || { versions synthetic "$SEL1"; "$PY" -m revrender.eval.restoration_errors zones --evaluation "$SEL1" --output "$R/zones_synthetic.json" >> "$R/synthetic.log" 2>&1; mark synthetic; }

# ---- 3. the 201 human regions for every model -----------------------------------------------------------------------------------
if ! done_ finetune; then
    if [ ! -e "${FT}_Ours/COMPLETE" ]; then
        INIT=$(chosen_of "$SEL1/chosen" Ours)
        bash "$ROOT/scripts/run_review_round.sh" "$SET" "$INIT" "$C1" 0 "${FT}_Ours" > "$R/finetune_Ours.log" 2>&1
    fi
    ( finetune NAFNet "$C1" 0; finetune DE-GAN "$C1" 0; finetune Restormer "$C1" 0 ) > "$R/finetune_lane1.log" 2>&1 & A=$!
    ( finetune StainRestorer "$C2" 1; finetune COR-Generator "$C2" 1 ) > "$R/finetune_lane2.log" 2>&1 & B=$!
    ( finetune DocDiff "$C3" 1; finetune Strikethrough-CycleGAN "$C3" 1 ) > "$R/finetune_lane3.log" 2>&1 & C=$!
    wait $A; wait $B; wait $C
    mark finetune; fi
done_ finetuned || { versions finetuned "$SEL2"; "$PY" -m revrender.eval.restoration_errors zones --evaluation "$SEL2" --output "$R/zones_finetuned.json" >> "$R/finetuned.log" 2>&1; mark finetuned; }

# ---- 4. the paired difference ---------------------------------------------------------------------------------------------------
done_ compare || { "$PY" -m revrender.eval.unified_restoration compare --before "$SEL1/chosen" --after "$SEL2/chosen" --output-dir "$R/compare" > "$R/compare.log" 2>&1; mark compare; }
touch "$R/COMPLETE"; stamp "complete"
