#!/usr/bin/env bash
# Finish a prepared unified comparison without repeating model inference.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
require_clean_source
test -f "$OUT/pack/pack.json"
test ! -e "$OUT/qwen8b"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
GPU2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
# TP2 uses 55% of each card (~13.2 GiB), leaving a margin at 14,000 MiB free.
# All candidates and raw scans share this engine; no inference settings change.
wait_for_gpu_memory "$GPU0" 14000
wait_for_gpu_memory "$GPU2" 14000
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU0,$GPU2" 0 "$OUT/qwen8b" -- env CUDA_VISIBLE_DEVICES=0,2 \
  "$PY" -m revrender.recognize_lines --pack "$OUT/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --tp 2 --gpu-memory .55 --output-dir "$OUT/qwen8b" > "$OUT/qwen8b.log" 2>&1
"$PY" -m revrender.eval.revision_boxes read --pack "$OUT/pack" --readings "$OUT/qwen8b/predictions.jsonl" --output "$OUT/read_qwen8b.json"
"$PY" -m revrender.tools.unified_restoration_report --evaluation "$OUT" --output-dir "$OUT/report"
touch "$OUT/COMPLETED"
