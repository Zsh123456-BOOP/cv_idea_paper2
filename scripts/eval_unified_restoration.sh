#!/usr/bin/env bash
# Frozen checkpoints, shared inputs, one OCR engine for every model and split.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
require_clean_source
test ! -e "$OUT"
mkdir "$OUT"
trap 'touch "$OUT/FAILED"' ERR
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs
"$PY" -m revrender.eval.unified_restoration prepare \
  --models "$ROOT/configs/unified_restoration.json" \
  --selection "$O/paper2_annotation_labels/selection.json" \
  --labels "$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json" \
  --labels "$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-test_2026-10-05-04-28.json" \
  --geometry "$O/paper2_geometry_labels_test7" \
  --human-validation "$O/paper2_pixel_finetune_set20261007/validation_native.json" \
  --pages "$O/paper2_pages_clean_unlabeled3976" --pages /home/zsh/hme/data/full1026_v1/images \
  --output-dir "$OUT/inputs"
mkdir "$OUT/runs"
GPU0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
GPU2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
for name in Ours-old Ours-round1 COR-Generator Restormer NAFNet DE-GAN Strikethrough-CycleGAN DocDiff StainRestorer; do
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
  wait_for_gpu_memory "$GPU0" 14000
  bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU0" 0 "$OUT/runs/$name" -- \
    "$RUNTIME" -m revrender.eval.unified_restoration infer --inputs "$OUT/inputs" --name "$name" \
    --output-dir "$OUT/runs/$name" > "$OUT/runs/$name.log" 2>&1
done
unset LD_LIBRARY_PATH
export PYTHONPATH="$ROOT"
"$PY" -m revrender.eval.unified_restoration merge --inputs "$OUT/inputs" --runs "$OUT/runs" --output-dir "$OUT/pack"
# Split one reader across the two authorized cards to fit alongside existing jobs.
# All candidates, including raw scans, use this same TP2 engine and request order.
wait_for_gpu_memory "$GPU0" 14500
wait_for_gpu_memory "$GPU2" 14500
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU0,$GPU2" 0 "$OUT/qwen8b" -- env CUDA_VISIBLE_DEVICES=0,2 \
  "$PY" -m revrender.recognize_lines --pack "$OUT/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --tp 2 --gpu-memory .55 --output-dir "$OUT/qwen8b" > "$OUT/qwen8b.log" 2>&1
"$PY" -m revrender.eval.revision_boxes read --pack "$OUT/pack" --readings "$OUT/qwen8b/predictions.jsonl" --output "$OUT/read_qwen8b.json"
"$PY" -m revrender.tools.unified_restoration_report --evaluation "$OUT" --output-dir "$OUT/report"
touch "$OUT/COMPLETED"
