#!/usr/bin/env bash
# Finish a prepared unified comparison without repeating model inference.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
require_clean_source
test -f "$OUT/pack/pack.json"
test ! -e "$OUT/qwen8b"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU_A=${GPU_A:-GPU-d0bcc564-5bd9-b943-f995-8536415088e3}; GPU_B=${GPU_B:-GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0}
index() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$1" '$2==u{print $1}'; }
# TP2 uses 55% of each card (~13.2 GiB), leaving a margin at 14,000 MiB free.
# All candidates and raw scans share this engine; no inference settings change.
wait_for_gpu_memory "$GPU_A" 14000
wait_for_gpu_memory "$GPU_B" 14000
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU_A,$GPU_B" 0 "$OUT/qwen8b" -- env CUDA_VISIBLE_DEVICES="$(index "$GPU_A"),$(index "$GPU_B")" \
  "$PY" -m revrender.recognize_lines --pack "$OUT/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --tp 2 --gpu-memory .55 --output-dir "$OUT/qwen8b" > "$OUT/qwen8b.log" 2>&1
"$PY" -m revrender.eval.revision_boxes read --pack "$OUT/pack" --readings "$OUT/qwen8b/predictions.jsonl" --output "$OUT/read_qwen8b.json"
"$PY" -m revrender.tools.unified_restoration_report --evaluation "$OUT" --output-dir "$OUT/report"
touch "$OUT/COMPLETED"
