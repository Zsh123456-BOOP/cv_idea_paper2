#!/usr/bin/env bash
# How the gain of the human regions grows with their number (user 2026-10-22): the box remover and the two restorers best after the
# 201-region fine-tuning (DE-GAN, Restormer) fine-tuned exactly as in run_annotation_value.sh on nested subsets of the regions --
# whole training writers in one seeded order until 50 and 100 regions (data.review_set --training-regions), with the same synthetic
# replay rule -- each from the synthetic version chosen there; versions chosen the same way.  The 0 and 201 points are that run's.
# Unattended; a step whose marker is in $R is skipped.  Cards 1 and 0 (card 2 for the OCR when it has room, else card 3 when idle).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_learning_curve20261022
V4=${N}_restoration_pairs20261007_v4; BOXES=${N}_v4_boxes; SEL1=${N}_v4_versions_synthetic
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
FT_IN=$O/paper2_pixel_finetune_inputs20261007; EXPORT=$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
test -e "$SEL1/chosen/selection.json" || { echo 'run_annotation_value.sh first'; exit 2; }

for n in 50 100; do
    SET=${N}_v4_finetune_set_n$n; TILES=${N}_v4_finetune_tiles_n$n; FT=${N}_v4_finetune_n$n; SEL=${N}_v4_versions_finetuned_n$n
    if ! done_ "set_$n"; then
        "$PY" -m revrender.data.review_set --review-export "$EXPORT" --pages "$O/paper2_full_review20261010/images" \
            --mapping "$FT_IN/index_manifest.json" --same-student "$FT_IN/same_student_pairs.json" --replay "$BOXES" --fixed-validation "$VAL" \
            --training-regions "$n" --output-dir "$SET" > "$R/set_$n.log" 2>&1
        "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --source "$V4" --output-dir "$TILES" >> "$R/set_$n.log" 2>&1
        mark "set_$n"; fi
    if ! done_ "finetune_$n"; then
        ( [ -e "${FT}_Ours/COMPLETE" ] || bash "$ROOT/scripts/run_review_round.sh" "$SET" "$(chosen_of "$SEL1/chosen" Ours)" "$C1" 0 "${FT}_Ours"
          finetune_restorer DE-GAN "$C1" 0 "$(chosen_of "$SEL1/chosen" DE-GAN)" "$TILES" "${FT}_DE-GAN" ) > "$R/finetune_${n}_lane1.log" 2>&1 & A=$!
        ( finetune_restorer Restormer "$C0" 0 "$(chosen_of "$SEL1/chosen" Restormer)" "$TILES" "${FT}_Restormer" ) > "$R/finetune_${n}_lane0.log" 2>&1 & B=$!
        wait $A; wait $B; mark "finetune_$n"; fi
    if ! done_ "versions_$n"; then
        [ -e "$R/candidates_$n.json" ] || "$PY" -m revrender.eval.unified_restoration candidates --own "Ours=${FT}_Ours/train/formal" \
            --run "DE-GAN=${FT}_DE-GAN" --run "Restormer=${FT}_Restormer" --upstream "$UP" --stain-upstream "$STAIN_UP" --output "$R/candidates_$n.json" >> "$R/versions_$n.log" 2>&1
        evaluate_versions "$R/candidates_$n.json" "$SEL" "$R/versions_$n.log"
        "$PY" -m revrender.eval.unified_restoration compare --before "$SEL1/chosen" --after "$SEL/chosen" --output-dir "$R/compare_$n" >> "$R/versions_$n.log" 2>&1
        mark "versions_$n"; fi
done
touch "$R/COMPLETE"; stamp complete
