#!/usr/bin/env bash
# Does the synthetic data hold the restorers back?  (user 2026-10-23)  COR-Generator, NAFNet and DE-GAN on the 201 human regions
# alone, no synthetic replay (data.restoration_pairs human --no-replay): (a) from their own random start (configs/restoration.toml
# [real_only]); (b) from the synthetic version run_annotation_value.sh fine-tuned from, longer ([finetune_long]).  Versions chosen
# on validation and the development half (as run_wider_validation.sh), each arm compared with the current fine-tuning (synthetic
# training, then the regions with synthetic replay).  Unattended; a step whose marker is in $R is skipped.  Cards 1, 2 and 3 (3 only
# while nobody else is on it).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_real_only20261023
V4=${N}_restoration_pairs20261007_v4; SET=${N}_v4_finetune_set; TILES=${N}_v4_finetune_tiles_human; SEL1=${N}_v4_versions_synthetic
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
export DEV_EXPORT=$O/paper2_test_export20261022/export.json DEV_PAGES=$O/paper2_test_text20261018_page/site/images CHOOSE_SPLITS="val dev"
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
MODELS=(COR-Generator NAFNet DE-GAN)

done_ tiles || { "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --source "$V4" --no-replay --output-dir "$TILES" > "$R/tiles.log" 2>&1; mark tiles; }
if ! done_ train; then
    arms() { local card="$1" numa="$2" m="$3"
        finetune_restorer "$m" "$card" "$numa" - "$TILES" "${N}_real_only_$m" real_only
        finetune_restorer "$m" "$card" "$numa" "$(chosen_of "$SEL1/chosen" "$m")" "$TILES" "${N}_finetune_long_$m" finetune_long; }
    ( arms "$C1" 0 COR-Generator ) > "$R/train_COR.log" 2>&1 & A=$!
    ( arms "$C2" 1 NAFNet ) > "$R/train_NAFNet.log" 2>&1 & B=$!
    ( arms "$C3" 1 DE-GAN ) > "$R/train_DE-GAN.log" 2>&1 & C=$!
    wait $A; wait $B; wait $C; mark train; fi
for arm in real_only finetune_long; do
    if ! done_ "versions_$arm"; then
        runs=(); for m in "${MODELS[@]}"; do runs+=(--run "$m=${N}_${arm}_$m"); done
        [ -e "$R/candidates_$arm.json" ] || "$PY" -m revrender.eval.unified_restoration candidates "${runs[@]}" --upstream "$UP" \
            --stain-upstream "$STAIN_UP" --output "$R/candidates_$arm.json" >> "$R/versions_$arm.log" 2>&1
        evaluate_versions "$R/candidates_$arm.json" "${N}_wide_versions_$arm" "$R/versions_$arm.log"
        "$PY" -m revrender.eval.unified_restoration compare --before "${N}_wide_versions_r201/chosen" --after "${N}_wide_versions_$arm/chosen" \
            --output-dir "$R/compare_$arm" >> "$R/versions_$arm.log" 2>&1
        mark "versions_$arm"; fi
done
touch "$R/COMPLETE"; stamp complete
