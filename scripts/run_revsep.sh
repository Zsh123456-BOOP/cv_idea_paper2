#!/usr/bin/env bash
# RevSep (models.separation, the plan of 2026-10-25) without the recognition loss: the hint channel and the two heads, trained by the
# recipe of the restorers' best arm (b) (run_real_only.sh) -- on the synthetic pairs ([training.RevSep]: NAFNet's budget), the
# version chosen on validation and the development half, then fine-tuned on the 201 human regions alone ([finetune_long]) and chosen
# again -- beside two ablations trained the same way: without the hint (--no-hint) and without the heads (--no-heads: H1 alone, NAFNet
# with the hint channel).  Every crop of the evaluations gets the detector's masks (data.separation, DETECTOR).  The error zones of
# every version (eval.restoration_errors zones): the strike missed next to writing, the writing erased next to the strike.  The
# baselines' (b) versions were read in another batch: the deciding comparison is a later batch with all of them together.
# Unattended; a step whose marker is in $R is skipped.  Cards 0, 1 and 2 train, cards 1 and 2 (or 3 while nobody is on it) evaluate.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_revsep20261025
V4=${N}_restoration_pairs20261007_v4; SET=${N}_v4_finetune_set; TILES=${N}_revsep_tiles_human
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
ARMS=(RevSep RevSep-nohint RevSep-noheads)
flags() { case "$1" in RevSep-nohint) echo --no-hint ;; RevSep-noheads) echo --no-heads ;; esac; }

if ! done_ data; then
    [ -e "${VAL_HINTS%/*}/run_manifest.json" ] || "$PY" -m revrender.data.separation hints --native "$VAL/validation_native.json" \
        --detector "$DETECTOR" --output-dir "${VAL_HINTS%/*}" > "$R/data.log" 2>&1
    [ -e "$TILES/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --source "$V4" --no-replay \
        --detector "$DETECTOR" --output-dir "$TILES" >> "$R/data.log" 2>&1
    for f in gold_pixels.npy targets.npy records.json; do cmp "$TILES/$f" "${N}_v4_finetune_tiles_human/$f"; done   # the (b) tiles, plus hints
    mark data; fi
train() {   # STAGE INIT_SELECTION DATA BUDGET: the three arms on cards 0, 1 and 2
    local stage="$1" chosen="$2" data="$3" budget="$4" pids=() k=0 cards=("$C0" "$C1" "$C2") numas=(0 0 1)
    for arm in "${ARMS[@]}"; do
        local init=-; [ "$chosen" = - ] || init=$(chosen_of "$chosen" "$arm")
        ( RESTORER_FLAGS="$(flags "$arm")" finetune_restorer RevSep "${cards[$k]}" "${numas[$k]}" "$init" "$data" "${N}_revsep_${stage}_$arm" $budget ) \
            > "$R/train_${stage}_$arm.log" 2>&1 & pids+=($!); k=$((k+1))
    done
    for p in "${pids[@]}"; do wait "$p"; done
}
versions() {   # STAGE: every kept version of the three arms evaluated together, chosen, and its error zones
    local stage="$1" runs=() evaluation=${N}_revsep_versions_$1
    for arm in "${ARMS[@]}"; do runs+=(--run "$arm=${N}_revsep_${stage}_$arm"); done
    [ -e "$R/candidates_$stage.json" ] || "$PY" -m revrender.eval.unified_restoration candidates "${runs[@]}" --upstream "$UP" \
        --stain-upstream "$STAIN_UP" --output "$R/candidates_$stage.json" >> "$R/versions_$stage.log" 2>&1
    evaluate_versions "$R/candidates_$stage.json" "$evaluation" "$R/versions_$stage.log"
    [ -e "$R/zones_$stage.json" ] || "$PY" -m revrender.eval.restoration_errors zones --evaluation "$evaluation" --splits val dev \
        --output "$R/zones_$stage.json" >> "$R/versions_$stage.log" 2>&1
}
done_ train_synthetic || { train synthetic - "$V4" ""; mark train_synthetic; }
done_ versions_synthetic || { versions synthetic; mark versions_synthetic; }
done_ train_finetune_long || { train finetune_long "${N}_revsep_versions_synthetic/chosen" "$TILES" finetune_long; mark train_finetune_long; }
done_ versions_finetune_long || { versions finetune_long; mark versions_finetune_long; }
touch "$R/COMPLETE"; stamp complete
