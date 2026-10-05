#!/usr/bin/env bash
# The strike GAN end to end, unattended:
#   0. paper 1's RevRender (frozen) on the pages with a revised line of a usable writer that it has not read yet, cut to ink;
#   1. the GAN's real lines and host lines (data.gan_sets);
#   2. the GAN (train_gan), at most GAN_HOURS;
#   3. the box remover's training set it draws (data.gan_strikes);
#   4. two box removers on it at once: A from the first run's last weights (as the paste-trained box remover started), B from that
#      box remover's chosen version (step 8000);
#   5. every version of both read on the user's validation boxes (best mask per box), scored, read by Qwen3-VL-8B;
#   6. every version of both read on the test boxes (the cut chosen on validation) and on the seven stroke-labelled test questions:
#      recorded only, nothing is chosen there;
#   7. OUT_summary.json: per arm, the version validation chooses and its test reading, the best version on test, and the references.
#     run_strike_gan.sh OUT GAN_HOURS
# Card 1 (NUMA 0): the RevRender pass, the GAN, its drawing, remover A; card 2 (NUMA 1): remover B and the readings.  Every stage writes
# create-only under OUT_*; a failing stage stops the run and touches OUT.FAILED (its log names where).  No test row is trained on.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; GAN_HOURS="${2:-10}"
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
O=/home/zsh/hme/outputs; L=$O/paper2_annotation_labels; SPLITS=$O/paper2_splits
VAL="$L/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json"; TEST="$L/涂改标注_paper2-revision-boxes-test_2026-10-05-04-28.json"
BANK=$O/paper2_strike_bank_strokes; PAPER1=/home/zsh/hme/work/revision_tune880_v1/control
CATALOGUE=/home/zsh/hme/code_358_spatial_v1/gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl
PAGES=(--pages /home/zsh/hme/data/unlabeled_v1/images --pages /home/zsh/hme/data/full1026_v1/images)
IMAGES=(--images /home/zsh/hme/data/unlabeled_v1/images --images /home/zsh/hme/data/full1026_v1/images)
OCR=(--ocr-text /home/zsh/hme/annot_v1/ulocr_dedup.jsonl)
for k in 00 01 02 03; do OCR+=(--ocr-text "/home/zsh/hme/artifacts/line_edit_external_blind_20260811_v1/ulocr/shard_$k.jsonl"); done
test ! -e "${OUT}_sets" || { echo 'create-only output exists'; exit 2; }
require_clean_source
trap 'echo "pipeline failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT.FAILED"' ERR
stamp() { echo "$* $(date -u +%FT%TZ)"; }
gpu() { bash "$ROOT/scripts/launch_gpu_job.sh" "$@"; }

stamp "0. pages without detections"
"$PY" -m revrender.data.gan_sets --detections $O/paper2_pages_unlabeled3976_complete_tight_connected "${PAGES[@]}" --mapping $SPLITS/index_manifest.json \
    --same-student $SPLITS/same_student_pairs.json --catalogue $CATALOGUE --missing "${OUT}_missing.txt"
DETECTIONS=(--detections $O/paper2_pages_unlabeled3976_complete_tight_connected)
if [ -s "${OUT}_missing.txt" ]; then
    gpu $C1 0 "${OUT}_detections" -- "$PY" -m revrender.pages --checkpoint $PAPER1/formal/best.pt --freeze $PAPER1/freeze.json "${PAGES[@]}" "${OCR[@]}" \
        --ocr-model /home/zsh/hme/models/Unlimited-OCR --templates $O/paper2_page_templates/templates.npz --ids "${OUT}_missing.txt" --output-dir "${OUT}_detections"
    "$PY" -m revrender.tighten --pages-output "${OUT}_detections" "${IMAGES[@]}" --output-dir "${OUT}_detections_tight"
    DETECTIONS+=(--detections "${OUT}_detections_tight")
fi

stamp "1. real and host lines"
"$PY" -m revrender.data.gan_sets "${DETECTIONS[@]}" "${PAGES[@]}" --mapping $SPLITS/index_manifest.json --same-student $SPLITS/same_student_pairs.json \
    --catalogue $CATALOGUE --output-dir "${OUT}_sets"

stamp "2. the GAN"
gpu $C1 0 "${OUT}_gan" -- "$PY" -m revrender.train_gan --sets "${OUT}_sets" --bank $BANK --steps 300000 --max-hours "$GAN_HOURS" --output-dir "${OUT}_gan"

stamp "3. the box remover's training set"
gpu $C1 0 "${OUT}_boxes" -- "$PY" -m revrender.data.gan_strikes --generator "${OUT}_gan/generator_last.pt" --sets "${OUT}_sets" --bank $BANK "${PAGES[@]}" \
    --output-dir "${OUT}_boxes"
SHA=$(sha256sum "${OUT}_boxes/run_manifest.json" | cut -d' ' -f1)

stamp "4. box removers A and B"
arm() {   # NAME CARD NUMA INIT
    SYNTHETIC=1 MAPPING=$SPLITS/index_manifest.json SAME_STUDENT=$SPLITS/same_student_pairs.json INIT="$4" RECIPE=configs/revrender_strokes_boxes.toml \
        bash "$ROOT/scripts/launch_train_arm.sh" control "$2" "$3" "${OUT}_boxes" "$SHA" - - "${OUT}_remover_$1" > "${OUT}_remover_$1.launch.log" 2>&1
}
arm a $C1 0 $O/paper2_strokes_control/formal/step_28300.pt & A=$!
arm b $C2 1 $O/paper2_strokes_boxes/formal/step_8000.pt & B=$!
wait $A; wait $B
test -f "${OUT}_remover_a/formal/complete.json" && test -f "${OUT}_remover_b/formal/complete.json"

read_split() {   # SPLIT LABELS CUT_FROM
    local split="$1" labels="$2" models=(--model paper1=$O/paper2_pages_annotation_set_templates --model reference=$O/paper2_strokes_boxes_${1}_top1/step_8000)
    for a in a b; do
        BOXES="$labels" SELECTION=$L/selection.json TOP=1 POINT_SCORE=0 bash "$ROOT/scripts/sweep_checkpoints.sh" $C2 1 "${OUT}_remover_$a/formal" "$split" \
            "${OUT}_${split}_$a" > "${OUT}_${split}_$a.log" 2>&1
        for d in $(ls -d "${OUT}_${split}_$a"/step_* | grep -v '\.log$'); do models+=(--model "$a$(basename "$d")=$d"); done
    done
    "$PY" -m revrender.eval.revision_boxes boxes --labels "$labels" --selection $L/selection.json --split "$split" "${models[@]}" "${PAGES[@]}" \
        --cuts 0.0 0.1 0.39 ${3:+--cut-from "$3"} --output-dir "${OUT}_${split}_read" > "${OUT}_${split}_read.log" 2>&1
    gpu $C2 1 "${OUT}_${split}_qwen8b" -- env CUDA_VISIBLE_DEVICES=2 "$PY" -m revrender.recognize_lines --pack "${OUT}_${split}_read/pack" \
        --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct --conditions erased_neutral --output-dir "${OUT}_${split}_qwen8b"
    "$PY" -m revrender.eval.revision_boxes read --pack "${OUT}_${split}_read/pack" --readings "${OUT}_${split}_qwen8b/predictions.jsonl" \
        --output "${OUT}_${split}_read/read_qwen8b.json"
    if [ "$split" = test ]; then
        "$PY" -m revrender.eval.revision_boxes pixels --geometry $O/paper2_geometry_labels_test7 "${models[@]}" "${PAGES[@]}" --cut-from "$3" \
            --output "${OUT}_${split}_read/pixels_test7.json"
    fi
}
stamp "5. validation"
read_split val "$VAL" ""
stamp "6. test (recorded only)"
read_split test "$TEST" "${OUT}_val_read/summary.json"

stamp "7. summary"
"$PY" - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
held = lambda split: {k.split('/')[0]: v for k, v in json.load(open(f'{out}_{split}_read/read_qwen8b.json'))['held'].items() if k.endswith('/erased_neutral/line')}
val, test = held('val'), held('test')
pixels = json.load(open(f'{out}_test_read/pixels_test7.json'))['models']
summary = {'scan': {'val': val['scan'], 'test': test['scan']}, 'paper1': {'val': val['paper1'], 'test': test['paper1'], 'pixels': pixels['paper1']},
           'reference_step_8000': {'val': val['reference'], 'test': test['reference'], 'pixels': pixels['reference']}}
for arm in 'ab':
    versions = [m for m in val if m.startswith(arm+'step_')]
    chosen = max(versions, key=lambda m: (val[m]['held'], val[m]['whole']))
    best = max(versions, key=lambda m: test[m]['held'])
    summary[f'remover_{arm}'] = {'chosen_on_validation': chosen, 'val': val[chosen], 'test': test[chosen], 'pixels': pixels[chosen],
                                 'best_on_test': best, 'best_test': test[best], 'gap': test[best]['held']-test[chosen]['held']}
json.dump(summary, open(f'{out}_summary.json', 'x'), indent=1)
print(json.dumps(summary, indent=1))
PY
stamp "pipeline complete"
touch "$OUT.COMPLETE"
