#!/usr/bin/env bash
# Does synthetic data that looks like the real strikes train better?  (user 2026-10-22)  v2 pairs: the same hosts, copies and seeds as
# v1 (paper2_clean_restoration_pairs20261007_v4), the gold pieces at their real size against the writing, 1.6x as long, drawn back
# and forth up to 3 times, bent by up to 0.15 writing heights, 1.3x as thick, 20 grey levels below the writing's median -- the
# settings closest to the 201 training regions' strike statistics (eval.restoration_errors strokes; the validation regions were not
# used).  The box remover, NAFNet and DE-GAN trained on v2 with the v1 budgets; versions chosen as in run_annotation_value.sh and
# compared with their v1 versions (eval.unified_restoration compare) and by where the errors fall (eval.restoration_errors zones).
# Unattended; a step whose marker is in $R is skipped.  Card 2 for the box remover, card 0 for the restorers.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_synthetic_v2_20261022
V2=${N}_restoration_pairs_v2; BOXES=${N}_v2_boxes; OURS=${N}_v2_control; SUITE=${N}_v2_restoration_suite
SEL1=${N}_v4_versions_synthetic; SEL=${N}_v2_versions_synthetic
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
MAP=$O/paper2_splits/index_manifest.json; PAIRS=$O/paper2_splits/same_student_pairs.json
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
mkdir -p "$R" "$SUITE"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
sha() { sha256sum "$1/run_manifest.json" | cut -d' ' -f1; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
test -e "$SEL1/chosen/selection.json" || { echo 'run_annotation_value.sh first'; exit 2; }

if ! done_ pairs; then
    "$PY" -m revrender.data.restoration_pairs --corpus "${N}_reader_corpus/corpus.jsonl" --mapping "$MAP" --same-student "$PAIRS" \
        --human-validation "$VAL/validation_native.json" --gold-bank "${N}_strike_bank_strokes" --detections "${N}_pages_clean_run_unlabeled3976" \
        --templates "${N}_page_templates/templates.npz" --pages "$O/paper2_pages_clean_unlabeled3976" \
        --real-geometry --stretch 1.6 --passes 3 --wave .15 --thicken 1.3 --lighter -20 --output-dir "$V2" > "$R/pairs.log" 2>&1
    "$PY" -m revrender.data.restoration_pairs boxes --source "$V2" --output-dir "$BOXES" >> "$R/pairs.log" 2>&1
    "$PY" -m revrender.eval.restoration_errors strokes --review-set "${N}_v4_finetune_set" --pairs "$V2" --output "$R/strokes.json" >> "$R/pairs.log" 2>&1
    mark pairs; fi
if ! done_ train; then
    ( [ -e "$OURS/COMPLETE" ] || SYNTHETIC=1 MAPPING=$MAP SAME_STUDENT=$PAIRS RECIPE=$ROOT/configs/revrender_strokes.toml \
        bash "$ROOT/scripts/launch_train_arm.sh" control "$C2" 1 "$BOXES" "$(sha "$BOXES")" - - "$OURS" ) > "$R/ours.log" 2>&1 & A=$!
    ( [ -e "$SUITE/lane0/COMPLETED" ] || bash "$ROOT/scripts/run_restoration_suite.sh" 0 "$V2" "$UP" "$SUITE/lane0" formal 0 NAFNet DE-GAN ) > "$R/suite.log" 2>&1 & B=$!
    wait $A; wait $B; mark train; fi
if ! done_ versions; then
    [ -e "$R/candidates.json" ] || "$PY" -m revrender.eval.unified_restoration candidates --own "Ours=$OURS/formal" --every 2000 \
        --run "NAFNet=$SUITE/lane0/NAFNet_gold_pixels" --run "DE-GAN=$SUITE/lane0/DE-GAN_gold_pixels" --upstream "$UP" --stain-upstream "$STAIN_UP" \
        --output "$R/candidates.json" >> "$R/versions.log" 2>&1
    evaluate_versions "$R/candidates.json" "$SEL" "$R/versions.log"
    "$PY" -m revrender.eval.unified_restoration compare --before "$SEL1/chosen" --after "$SEL/chosen" --output-dir "$R/compare" >> "$R/versions.log" 2>&1
    "$PY" -m revrender.eval.restoration_errors zones --evaluation "$SEL" --output "$R/zones.json" >> "$R/versions.log" 2>&1
    mark versions; fi
touch "$R/COMPLETE"; stamp complete
