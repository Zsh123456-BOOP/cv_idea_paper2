#!/usr/bin/env bash
# Whole-question inference (revrender.pages) of every checkpoint a run kept (step_<n>.pt) on a list of questions, at a low score cut:
# revision_boxes reads them on the user's boxes and cuts there.
#   sweep_checkpoints.sh GPU_UUID NUMA RUN_FORMAL_DIR IDS OUT [MIN_STEP]
# Holds the card's lock shared; OUT must not exist; OUT/step_<n> per checkpoint.  The OCR text and the printed templates are the
# ones the annotation questions were read with (README, 第二篇的人工标注 step 2).  BOXES=<labels.json> SELECTION=<selection.json> reads
# a box remover instead (revrender.box_remover inside the struck-line boxes of the labels); IDS is then the split (val / test).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
GPU_UUID="$1"; NUMA="$2"; RUN="$3"; IDS="$4"; OUT="$5"; MIN="${6:-0}"
refuse_card3 "$GPU_UUID"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
trap 'echo "sweep failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT/FAILED"' ERR
hold_card "$GPU_UUID"
export HME_NUMA_NODE="$NUMA"
O=/home/zsh/hme/outputs
OCR=(--ocr-text /home/zsh/hme/annot_v1/ulocr_dedup.jsonl)
for k in 00 01 02 03; do OCR+=(--ocr-text "/home/zsh/hme/artifacts/line_edit_external_blind_20260811_v1/ulocr/shard_$k.jsonl"); done
for ck in $(ls "$RUN" | grep -E '^step_[0-9]+\.pt$' | sort -t_ -k2 -n); do
    step=${ck#step_}; step=${step%.pt}
    [ "$step" -ge "$MIN" ] || continue
    if [ -n "${BOXES:-}" ]; then
        numactl --cpunodebind="$NUMA" --membind="$NUMA" "$PY" -m revrender.box_remover --checkpoint "$RUN/$ck" --point 0.4 0.5 0.1 \
            --labels "$BOXES" --selection "$SELECTION" --split "$IDS" \
            --pages /home/zsh/hme/data/unlabeled_v1/images --pages /home/zsh/hme/data/full1026_v1/images --output-dir "$OUT/step_$step" > "$OUT/step_$step.log" 2>&1
    else
        numactl --cpunodebind="$NUMA" --membind="$NUMA" "$PY" -m revrender.pages --checkpoint "$RUN/$ck" --point 0.4 0.5 0.1 \
            --pages /home/zsh/hme/data/unlabeled_v1/images --pages /home/zsh/hme/data/full1026_v1/images "${OCR[@]}" \
            --templates "$O/paper2_page_templates/templates.npz" --ids "$IDS" --output-dir "$OUT/step_$step" > "$OUT/step_$step.log" 2>&1
    fi
    echo "step_$step $(tail -1 "$OUT/step_$step.log")"
done
touch "$OUT/COMPLETE"
echo "sweep complete $(date -u +%FT%TZ)"
