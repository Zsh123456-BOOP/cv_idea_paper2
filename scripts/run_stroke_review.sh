#!/usr/bin/env bash
# Matched material pilot. prepare writes banks/sets; train runs both arms and validation.
# Review the prepared sheets before starting training. No test question is used here.
#   bash scripts/run_stroke_review.sh /home/zsh/hme/outputs/paper2_stroke_review prepare|train|test
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; MODE="$2"; DATA_PREFIX="${DATA_PREFIX:-$OUT}"
O=/home/zsh/hme/outputs
PAGES=(--pages /home/zsh/hme/data/unlabeled_v1/images --pages /home/zsh/hme/data/full1026_v1/images)
MAP="$O/paper2_splits/index_manifest.json"; PAIRS="$O/paper2_splits/same_student_pairs.json"
require_clean_source
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
trap 'echo "stroke review failed at line $LINENO"; touch "${OUT}.${MODE}.FAILED"' ERR
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
if [ "$MODE" = test ]; then
    test -f "${OUT}.train.COMPLETE"
    validation_losses() {
        for name in pixels traced; do
            bash "$ROOT/scripts/launch_gpu_job.sh" "$C0" 0 "${OUT}_loss_$name" -- "$PY" -m revrender.eval.validation_loss \
                --synthetic "${DATA_PREFIX}_set_$name" --checkpoints "${OUT}_$name/formal" --output-dir "${OUT}_loss_$name"
        done
    }
    validation_losses > "${OUT}_loss.log" 2>&1 & LOSS_JOB=$!
    # Freeze the validation choice before reading any test result. The legacy character
    # retention chooses the step; ties use box F1, then the earlier step. Exact is reported.
    "$PY" -m revrender.eval.revision_boxes read --pack "${OUT}_val_read/pack" \
        --readings "${OUT}_val_qwen/predictions.jsonl" --output "${OUT}_val_read/read_strict.json"
    "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = sys.argv[1]
read = json.load(open(f'{out}_val_read/read_strict.json'))['held']
boxes = json.load(open(f'{out}_val_read/summary.json'))['models']
val = {k.split('/')[0]: v for k, v in read.items() if k.endswith('/erased_neutral/line')}
selected = {}
for arm in ('pixels', 'traced'):
    versions = [m for m in val if m.startswith(arm+'step_')]
    chosen = max(versions, key=lambda m: (val[m]['held'], boxes[m]['at_pack_cut']['f1'], -int(m.split('_')[-1])))
    selected[arm] = {'chosen_on_validation': chosen, 'validation': val[chosen]}
with open(f'{out}_selected.json', 'x') as f:
    json.dump({'rule': 'validation held, box F1, earlier step', 'arms': selected}, f, indent=2)
PY
    LABELS="$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-test_2026-10-05-04-28.json"
    MODELS=(--model reference="$O/paper2_strokes_boxes_test_top1/step_8000")
    for name in pixels traced; do
        BOXES="$LABELS" SELECTION="$O/paper2_annotation_labels/selection.json" TOP=1 POINT_SCORE=0 \
            bash "$ROOT/scripts/sweep_checkpoints.sh" "$C2" 1 "${OUT}_$name/formal" test "${OUT}_test_$name"
        for dir in "${OUT}_test_$name"/step_*; do
            if [ -d "$dir" ]; then MODELS+=(--model "$name$(basename "$dir")=$dir"); fi
        done
    done
    "$PY" -m revrender.eval.revision_boxes boxes --labels "$LABELS" --selection "$O/paper2_annotation_labels/selection.json" \
        --split test "${MODELS[@]}" "${PAGES[@]}" --cuts 0.0 --cut-from "${OUT}_val_read/summary.json" --output-dir "${OUT}_test_read"
    bash "$ROOT/scripts/launch_gpu_job.sh" "$C2" 1 "${OUT}_test_qwen" -- "$PY" -m revrender.recognize_lines \
        --pack "${OUT}_test_read/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct --conditions erased_neutral \
        --output-dir "${OUT}_test_qwen"
    "$PY" -m revrender.eval.revision_boxes read --pack "${OUT}_test_read/pack" \
        --readings "${OUT}_test_qwen/predictions.jsonl" --output "${OUT}_test_read/read_strict.json"
    "$PY" -m revrender.eval.revision_boxes pixels --geometry "$O/paper2_geometry_labels_test7" "${MODELS[@]}" "${PAGES[@]}" \
        --cut-from "${OUT}_val_read/summary.json" --output "${OUT}_test_read/pixels_test7.json"
    "$PY" - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
summary = json.load(open(f'{out}_selected.json'))
read = lambda split: {k.split('/')[0]: v for k, v in json.load(open(f'{out}_{split}_read/read_strict.json'))['held'].items()
                      if k.endswith('/erased_neutral/line')}
val, test = read('val'), read('test')
pixels = json.load(open(f'{out}_test_read/pixels_test7.json'))['models']
summary['reference'] = {'validation': val['reference'], 'test': test['reference'], 'pixels': pixels['reference']}
summary['scan'] = {'validation': val['scan'], 'test': test['scan']}
for arm, result in summary['arms'].items():
    chosen = result['chosen_on_validation']
    best = max((m for m in test if m.startswith(arm+'step_')), key=lambda m: test[m]['held'])
    result.update(test=test[chosen], pixels=pixels[chosen], best_on_test=best, best_test=test[best], gap=test[best]['held']-test[chosen]['held'])
with open(f'{out}_summary.json', 'x') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
    wait "$LOSS_JOB"
    touch "${OUT}.test.COMPLETE"
    exit 0
fi
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
test -f "${DATA_PREFIX}.prepare.COMPLETE"
arm() {
    local name="$1" card="$2" numa="$3" sha
    sha=$(sha256sum "${DATA_PREFIX}_set_$name/run_manifest.json" | cut -d' ' -f1)
    SYNTHETIC=1 MAPPING="$MAP" SAME_STUDENT="$PAIRS" INIT="$O/paper2_strokes_boxes/formal/step_8000.pt" \
        RECIPE="$ROOT/configs/revrender_strokes_review.toml" bash "$ROOT/scripts/launch_train_arm.sh" \
        control "$card" "$numa" "${DATA_PREFIX}_set_$name" "$sha" - - "${OUT}_$name" > "${OUT}_$name.log" 2>&1
}
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
