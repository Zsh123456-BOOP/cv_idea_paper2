#!/usr/bin/env bash
# Add the pre-human-finetuning box remover to the completed shared evaluation.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
BASE="$1"; OUT="$2"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m revrender.eval.unified_restoration extend --base "$BASE" \
  --models "$ROOT/configs/synthetic_box_evaluation.json" --output-dir "$OUT"
GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
wait_for_gpu_memory "$GPU" 14000
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" 0 "$OUT/runs/Ours-synthetic" -- \
  "$PY" -m revrender.eval.unified_restoration infer --inputs "$OUT/inputs" --name Ours-synthetic \
  --output-dir "$OUT/runs/Ours-synthetic" > "$OUT/Ours-synthetic.log" 2>&1
"$PY" -m revrender.eval.unified_restoration merge --inputs "$OUT/inputs" --runs "$OUT/runs" --output-dir "$OUT/pack"
bash "$ROOT/scripts/finish_unified_restoration.sh" "$OUT"
