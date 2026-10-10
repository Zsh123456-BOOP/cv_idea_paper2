#!/usr/bin/env bash
# Small incremental review round; evaluate against the starting model on fixed validation.
# Usage: run_review_round.sh DATASET INIT GPU_UUID NUMA OUT
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
DATASET="$1"; INIT_CHECKPOINT="$2"; GPU_UUID="$3"; NUMA="$4"; OUT="$5"
require_clean_source
test ! -e "$OUT"
mkdir "$OUT"
trap 'touch "$OUT/FAILED"' ERR
DATA_SHA="$(sha256sum "$DATASET/run_manifest.json" | cut -d' ' -f1)"
export RECIPE="$ROOT/configs/revrender_strokes_human.toml" SYNTHETIC=1 INIT="$INIT_CHECKPOINT"
export MAPPING=/home/zsh/hme/outputs/paper2_pixel_finetune_inputs20261007/index_manifest.json
export SAME_STUDENT=/home/zsh/hme/outputs/paper2_pixel_finetune_inputs20261007/same_student_pairs.json
bash "$ROOT/scripts/launch_train_arm.sh" control "$GPU_UUID" "$NUMA" "$DATASET" "$DATA_SHA" - - "$OUT/train"
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU_UUID" "$NUMA" "$OUT/eval" -- \
    "$PY" -m revrender.eval.review_removal --dataset "$DATASET" --baseline "$INIT_CHECKPOINT" \
    --checkpoints "$OUT/train/formal" --output-dir "$OUT/eval"
touch "$OUT/COMPLETE"
