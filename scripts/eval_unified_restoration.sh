#!/usr/bin/env bash
# Frozen checkpoints, shared inputs, one OCR engine for every model: the validation split (the old test set is retired).
# Cards: GPU_A runs the restorations, GPU_A and GPU_B the OCR (TP2); defaults cards 1 and 2.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
require_clean_source
test ! -e "$OUT"
mkdir "$OUT"
trap 'touch "$OUT/FAILED"' ERR
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs
LAB=$O/paper2_annotation_labels
"$PY" -m revrender.eval.unified_restoration prepare \
  --models "$ROOT/configs/unified_restoration.json" \
  --selection "$O/paper2_annotation_labels_allowed20261021/selection.json" \
  --labels "$LAB/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json" \
  --human-validation "$O/paper2_fixed_validation20261007/validation_native.json" \
  --pages "$O/paper2_pages_clean_unlabeled3976" \
  --output-dir "$OUT/inputs"
mkdir "$OUT/runs"
export GPU_A=${GPU_A:-GPU-d0bcc564-5bd9-b943-f995-8536415088e3} GPU_B=${GPU_B:-GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0}
NUMA_A=0; case "$GPU_A" in *56971197*|*314fdf43*) NUMA_A=1 ;; esac
for name in $("$PY" -c 'import json,sys; print(" ".join(m["name"] for m in json.load(open(sys.argv[1]))))' "$ROOT/configs/unified_restoration.json"); do
  RUNTIME="$PY"
  export PYTHONPATH="$ROOT"
  unset LD_LIBRARY_PATH
  if [ "$name" = DE-GAN ]; then
    RUNTIME=/home/zsh/hme/envs/paper2_restoration_tf/bin/python
    export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
    for directory in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do
      export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
  elif [ "$name" = StainRestorer ]; then
    export PYTHONPATH="$ROOT:/home/zsh/hme/envs/paper2_stain_deps"
  fi
  wait_for_gpu_memory "$GPU_A" 14000
  bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU_A" "$NUMA_A" "$OUT/runs/$name" -- \
    "$RUNTIME" -m revrender.eval.unified_restoration infer --inputs "$OUT/inputs" --name "$name" \
    --output-dir "$OUT/runs/$name" > "$OUT/runs/$name.log" 2>&1
done
unset LD_LIBRARY_PATH
export PYTHONPATH="$ROOT"
"$PY" -m revrender.eval.unified_restoration merge --inputs "$OUT/inputs" --runs "$OUT/runs" --output-dir "$OUT/pack"
bash "$ROOT/scripts/finish_unified_restoration.sh" "$OUT"
