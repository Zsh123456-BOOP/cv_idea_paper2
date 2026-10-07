#!/usr/bin/env bash
# Evaluate only frozen selections, after both training queues finish.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
LANE0="$1"; LANE3="$2"; UPSTREAM="$3"; OUT="$4"
test -f "$LANE0/COMPLETED" && test -f "$LANE3/COMPLETED"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir "$OUT"
GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0
O=/home/zsh/hme/outputs
LABELS=("$O"/paper2_annotation_labels/*revision-boxes-test_*.json)
test "${#LABELS[@]}" = 1
EXPORTS=()
for arm in gold_pixels predicted_pixels predicted_traced; do
  for model in NAFNet DE-GAN DocDiff; do
    run="$LANE0/${model}_${arm}"
    [ "$model" != DocDiff ] || run="$LANE3/${model}_${arm}"
    test -f "$run/selection.json"
    RUNTIME="$PY"
    if [ "$model" = DE-GAN ]; then
      RUNTIME=/home/zsh/hme/envs/paper2_restoration_tf/bin/python
      export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
      for directory in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do
        export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      done
    else
      unset LD_LIBRARY_PATH
    fi
    dest="$OUT/${model}_${arm}"
    require_free_memory "$GPU" 18000
    bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" 0 "$dest" -- \
      "$RUNTIME" -m revrender.baselines.restoration_export export --run "$run" --upstream "$UPSTREAM" \
      --labels "${LABELS[0]}" --selection "$O/paper2_annotation_labels/selection.json" \
      --geometry "$O/paper2_geometry_labels_test7" \
      --pages "$O/paper2_pages_clean_unlabeled3976" --pages /home/zsh/hme/data/full1026_v1/images \
      --output-dir "$dest" > "$dest.log" 2>&1
    EXPORTS+=("$dest")
  done
done
unset LD_LIBRARY_PATH
PYTHONPATH="$ROOT" "$PY" -m revrender.baselines.restoration_export merge --inputs "${EXPORTS[@]}" --output-dir "$OUT/pack"
require_free_memory "$GPU" 22500
# vLLM resolves logical CUDA ordinals; hold the UUID lock, then expose physical 0.
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" 0 "$OUT/qwen8b" -- env CUDA_VISIBLE_DEVICES=0 \
  "$PY" -m revrender.recognize_lines --pack "$OUT/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --gpu-memory .9 --output-dir "$OUT/qwen8b" > "$OUT/qwen8b.log" 2>&1
PYTHONPATH="$ROOT" "$PY" -m revrender.eval.revision_boxes read --pack "$OUT/pack" \
  --readings "$OUT/qwen8b/predictions.jsonl" --output "$OUT/read_qwen8b.json"
touch "$OUT/COMPLETED"
