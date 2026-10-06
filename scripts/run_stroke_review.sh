#!/usr/bin/env bash
# Matched material pilot. prepare writes banks/sets; train runs both arms and validation.
# Review the prepared sheets before starting training. No test question is used here.
#   bash scripts/run_stroke_review.sh /home/zsh/hme/outputs/paper2_stroke_review prepare|train
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; MODE="$2"
O=/home/zsh/hme/outputs
PAGES=(--pages /home/zsh/hme/data/unlabeled_v1/images --pages /home/zsh/hme/data/full1026_v1/images)
MAP="$O/paper2_splits/index_manifest.json"; PAIRS="$O/paper2_splits/same_student_pairs.json"
require_clean_source
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
trap 'echo "stroke review failed at line $LINENO"; touch "${OUT}.${MODE}.FAILED"' ERR
if [ "$MODE" = prepare ]; then
    "$PY" -m revrender.data.strike_bank --gold "$O/paper2_gold_pages" "${PAGES[@]}" \
        --mapping "$MAP" --same-student "$PAIRS" --render traced --output-dir "${OUT}_bank"
    for arm in pixels traced; do
        BANK="$O/paper2_strike_bank_strokes"
        if [ "$arm" = traced ]; then BANK="${OUT}_bank"; fi
        "$PY" -m revrender.data.synthetic_set --corpus "$O/paper2_reader_corpus_main/corpus.jsonl" \
            --bank "$BANK" "${PAGES[@]}" --copies 1 --over 1 --gold-share 1 --second .3 --no-blots --crops --union \
            --workers 12 --output-dir "${OUT}_set_$arm"
    done
    touch "${OUT}.prepare.COMPLETE"
    exit 0
fi
test "$MODE" = train
test -f "${OUT}.prepare.COMPLETE"
arm() {
    local name="$1" card="$2" numa="$3" sha
    sha=$(sha256sum "${OUT}_set_$name/run_manifest.json" | cut -d' ' -f1)
    SYNTHETIC=1 MAPPING="$MAP" SAME_STUDENT="$PAIRS" INIT="$O/paper2_strokes_boxes/formal/step_8000.pt" \
        RECIPE="$ROOT/configs/revrender_strokes_review.toml" bash "$ROOT/scripts/launch_train_arm.sh" \
        control "$card" "$numa" "${OUT}_set_$name" "$sha" - - "${OUT}_$name" > "${OUT}_$name.log" 2>&1
}
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
arm pixels "$C0" 0 & A=$!
arm traced "$C2" 1 & B=$!
wait "$A"; wait "$B"
VAL="$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json"
MODELS=(--model reference="$O/paper2_strokes_boxes_val_top1/step_8000")
for arm in pixels traced; do
    BOXES="$VAL" SELECTION="$O/paper2_annotation_labels/selection.json" TOP=1 POINT_SCORE=0 \
        bash "$ROOT/scripts/sweep_checkpoints.sh" "$C2" 1 "${OUT}_$arm/formal" val "${OUT}_val_$arm"
    for dir in "${OUT}_val_$arm"/step_*; do
        if [ -d "$dir" ]; then MODELS+=(--model "$arm$(basename "$dir")=$dir"); fi
    done
done
"$PY" -m revrender.eval.revision_boxes boxes --labels "$VAL" --selection "$O/paper2_annotation_labels/selection.json" \
    --split val "${MODELS[@]}" "${PAGES[@]}" --cuts 0.0 --output-dir "${OUT}_val_read"
bash "$ROOT/scripts/launch_gpu_job.sh" "$C2" 1 "${OUT}_val_qwen" -- "$PY" -m revrender.recognize_lines \
    --pack "${OUT}_val_read/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct --conditions erased_neutral \
    --output-dir "${OUT}_val_qwen"
"$PY" -m revrender.eval.revision_boxes read --pack "${OUT}_val_read/pack" \
    --readings "${OUT}_val_qwen/predictions.jsonl" --output "${OUT}_val_read/read_qwen8b.json"
touch "${OUT}.train.COMPLETE"
